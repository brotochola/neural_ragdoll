// DASH-ES: trains the corrective network (src/dash-net.js) on top of a tuned dash runner with evolution
// strategies: antithetic Gaussian perturbations of the weights, every member of a generation run on the
// same random courses (boxes and balls, src/terrain.js makeDashCourse), centred-rank fitness, Adam.
// Episodes are spread over worker threads. Every --every generations the current weights are scored on
// --held held-out courses; the best by that score is saved. The network starts with a zero output layer,
// so generation 0 is the runner alone.
//   node dash-es.js --runner trained/dash-e1b.json --gens 400 --out trained/dash-net.json
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads';

const root = path.dirname(fileURLToPath(import.meta.url));
const file = fileURLToPath(import.meta.url);

async function makeEnv(runnerFile, seconds, course) {
  const { config, setSkill } = await import('./config.js');
  const Box2D = await import(pathToFileURL(path.join(root, 'vendor/PhaserBox2D.min.js')).href);
  const { createSim } = await import('./src/sim.js');
  const { ElmanNetwork } = await import('./src/net.js');
  const { makeSeededRandom } = await import('./src/utils.js');
  const { evalMode } = await import('./src/training.js');
  const { installRunner } = await import('./src/run-selector.js');
  const { makeDashCourse } = await import('./src/terrain.js');
  const { makeDashNet } = await import('./src/dash-net.js');
  setSkill(6);
  config.walk.maxStandSpeed = 20;
  const sim = createSim(Box2D);
  const world = sim.build();
  const HZ = config.physics.hz;
  const weights = ElmanNetwork.copy(JSON.parse(fs.readFileSync(path.join(root, 'trained/stand-and-walk.json'), 'utf8')).theta);
  const runStates = JSON.parse(fs.readFileSync(path.join(root, 'trained/run-states.json'), 'utf8'));
  const p = JSON.parse(fs.readFileSync(path.join(root, runnerFile), 'utf8')).params;
  const correct = makeDashNet(world, config, Box2D);
  return function episode(theta, seed) {
    const r = makeSeededRandom(7000 + seed);
    const start = sim.fillTrainingStart({}, r, evalMode(6));
    const c = makeDashCourse(r, Object.assign({ length: 4 + 5.2 * seconds }, course));
    const rs = runStates[Math.floor(r() * runStates.length)];
    const v = p.vRun;
    Object.assign(start, { startPose: rs.pose.slice(), startVel: rs.vel.map((q, i) => (i % 3 === 0 ? q * v / rs.speed : q)), tilt: 0, drop: 0, settle: 0, frames: Math.round(seconds * HZ) });
    start.walk = null;
    start.terrain = { statics: c.statics, balls: c.balls };
    world.rebuild();
    const ep = new sim.Episode(world, weights, 0, 0, Object.assign({ phase: 0 }, start));
    ep.runner = { stance: rs.stance, t: -0.1, flight: false, jumping: false, jumpNext: false, strides: 0 };
    const obstacles = ep.terrain.statics.map((b) => ({ x0: b.x0, x1: b.x1, top: b.top }))
      .concat(ep.terrain.balls.map((b) => ({ live: () => { const q = Box2D.b2Body_GetPosition(b.id); return { x0: q.x - b.r, x1: q.x + b.r, top: q.y + b.r }; } })));
    installRunner(ep, world, p, { config, obstacles, net: theta ? { correct, theta } : null });
    const x0 = world.px[0];
    let frames = 0, fell = false, low = 0;
    while (!ep.advanceOneFrame()) {
      frames++;
      const m = ep.measure;
      low = m.headHeight < 0.95 || (m.torsoCos || 0) < 0.3 ? low + 1 : 0;
      if (m.headHeight < 0.7 || low > 15) { fell = true; break; }
    }
    const dist = world.px[0] - x0;
    const passed = obstacles.filter((o) => (o.live ? o.live() : o).x1 < world.px[0] - 0.3).length;
    const score = 2 * Math.max(0, dist) / seconds + 4 * Math.min(1, frames / (seconds * HZ)) + (fell ? 0 : 1);
    return { score, fell: fell ? 1 : 0, passed, dist, T: frames / HZ };
  };
}

if (!isMainThread) {
  const episode = await makeEnv(workerData.runner, workerData.seconds, workerData.course);
  parentPort.on('message', ({ id, theta, seeds }) => {
    const res = seeds.map((s) => episode(theta ? Float64Array.from(theta) : null, s));
    parentPort.postMessage({ id, res });
  });
} else {
  const args = process.argv.slice(2);
  const arg = (name, fallback) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : fallback; };
  const runner = arg('--runner', 'trained/dash-e1b.json');
  const gens = +arg('--gens', 400);
  const pairs = +arg('--pairs', 12);
  const k = +arg('--seeds', 4);
  const sigma = +arg('--sigma', 0.04);
  const lr = +arg('--lr', 0.02);
  const held = +arg('--held', 40);
  const every = +arg('--every', 10);
  const seconds = +arg('--seconds', 12);
  const course = JSON.parse(arg('--course', '{}'));
  const out = arg('--out', 'trained/dash-net.json');
  const init = arg('--init', null);
  const nW = +arg('--workers', Math.max(1, os.cpus().length));
  const { makeSeededRandom, randomNormalSample } = await import('./src/utils.js');
  const { NW, initTheta, DIN, DH, DOUT } = await import('./src/dash-net.js');
  const rand = makeSeededRandom(+arg('--rng', 5));
  const normal = () => randomNormalSample(rand);
  let theta = init ? Float64Array.from(JSON.parse(fs.readFileSync(init, 'utf8')).theta) : initTheta(rand);

  const workers = Array.from({ length: nW }, () => new Worker(file, { workerData: { runner, seconds, course } }));
  let nextId = 0;
  const pending = new Map();
  for (const w of workers) w.on('message', ({ id, res }) => { pending.get(id)(res); pending.delete(id); });
  // run jobs [{theta, seeds}] spread over the workers, one seed per message for balance
  async function runAll(jobs) {
    const tasks = [];
    jobs.forEach((j, ji) => j.seeds.forEach((s) => tasks.push({ ji, theta: j.theta, s })));
    const results = jobs.map(() => []);
    let qi = 0;
    await Promise.all(workers.map(async (w) => {
      while (qi < tasks.length) {
        const t = tasks[qi++];
        const id = nextId++;
        const res = await new Promise((resolve) => { pending.set(id, resolve); w.postMessage({ id, theta: t.theta ? Array.from(t.theta) : null, seeds: [t.s] }); });
        results[t.ji].push(res[0]);
      }
    }));
    return results;
  }
  const summarize = (rs) => {
    const n = rs.length, ob = rs.reduce((a, r) => a + r.passed, 0), falls = rs.reduce((a, r) => a + r.fell, 0);
    return { score: rs.reduce((a, r) => a + r.score, 0) / n, pass: n - falls, perObstacle: ob / Math.max(1, ob + falls), speed: rs.reduce((a, r) => a + r.dist / r.T, 0) / n, dist: rs.reduce((a, r) => a + r.dist, 0) / n };
  };
  const heldSeeds = Array.from({ length: held }, (_, i) => 500 + i);
  const base = summarize((await runAll([{ theta: null, seeds: heldSeeds }]))[0]);
  console.log('runner alone on held-out: score', base.score.toFixed(3), 'per obstacle', (100 * base.perObstacle).toFixed(1) + '%', 'pass', base.pass + '/' + held, 'speed', base.speed.toFixed(2));
  let best = null;
  const m = new Float64Array(NW), v = new Float64Array(NW);
  const t0 = Date.now();
  for (let g = 0; g < gens; g++) {
    const eps = [];
    const jobs = [];
    const seeds = Array.from({ length: k }, (_, i) => 10000 + g * k + i);
    for (let i = 0; i < pairs; i++) {
      const e = new Float64Array(NW);
      for (let d = 0; d < NW; d++) e[d] = normal();
      eps.push(e);
      const a = new Float64Array(NW), b = new Float64Array(NW);
      for (let d = 0; d < NW; d++) { a[d] = theta[d] + sigma * e[d]; b[d] = theta[d] - sigma * e[d]; }
      jobs.push({ theta: a, seeds }, { theta: b, seeds });
    }
    const res = await runAll(jobs);
    const f = res.map((rs) => rs.reduce((a, r) => a + r.score, 0) / rs.length);
    // centred ranks
    const order = f.map((x, i) => [x, i]).sort((p, q) => p[0] - q[0]);
    const u = new Float64Array(f.length);
    order.forEach(([, i], r) => (u[i] = r / (f.length - 1) - 0.5));
    const grad = new Float64Array(NW);
    for (let i = 0; i < pairs; i++) { const w = u[2 * i] - u[2 * i + 1]; for (let d = 0; d < NW; d++) grad[d] += w * eps[i][d]; }
    const t = g + 1;
    for (let d = 0; d < NW; d++) {
      const gd = grad[d] / (pairs * sigma) - 0.005 * theta[d];
      m[d] = 0.9 * m[d] + 0.1 * gd; v[d] = 0.999 * v[d] + 0.001 * gd * gd;
      theta[d] += lr * (m[d] / (1 - 0.9 ** t)) / (Math.sqrt(v[d] / (1 - 0.999 ** t)) + 1e-8);
    }
    const genMean = f.reduce((a, x) => a + x, 0) / f.length;
    if (g % every === every - 1 || g === gens - 1) {
      const h = summarize((await runAll([{ theta, seeds: heldSeeds }]))[0]);
      if (!best || h.score > best.score) { best = { ...h, theta: Float64Array.from(theta), gen: g }; fs.writeFileSync(out, JSON.stringify({ runner, DIN, DH, DOUT, gen: g, theta: Array.from(best.theta), held: { n: held, ...h }, base })); }
      console.log('gen', g, 'gen mean', genMean.toFixed(3), 'held-out score', h.score.toFixed(3), 'per obstacle', (100 * h.perObstacle).toFixed(1) + '%', 'pass', h.pass + '/' + held, 'speed', h.speed.toFixed(2), '| best gen', best.gen, (100 * best.perObstacle).toFixed(1) + '%', ((Date.now() - t0) / 60000).toFixed(1) + ' min');
    }
  }
  // fresh courses, never used: the runner alone and with the best network
  const freshSeeds = Array.from({ length: 60 }, (_, i) => 300 + i);
  const [a, b] = await runAll([{ theta: null, seeds: freshSeeds }, { theta: best.theta, seeds: freshSeeds }]);
  const fa = summarize(a), fb = summarize(b);
  console.log('fresh 60: runner alone', (100 * fa.perObstacle).toFixed(1) + '%', 'pass', fa.pass, 'dist', fa.dist.toFixed(1), '| with net', (100 * fb.perObstacle).toFixed(1) + '%', 'pass', fb.pass, 'dist', fb.dist.toFixed(1), 'speed', fb.speed.toFixed(2));
  const saved = JSON.parse(fs.readFileSync(out, 'utf8'));
  fs.writeFileSync(out, JSON.stringify(Object.assign(saved, { fresh: { alone: fa, net: fb } })));
  for (const w of workers) w.terminate();
}
