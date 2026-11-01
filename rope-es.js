// ROPE-ES: the trail corrective network trained on the rope swing's landing and run on (src/rope-swing.js,
// phases land and run2), with the rope numbers fixed (--runner = the rope params file). Same ES as trail-es.js.
//   node rope-es.js --runner trained/rope-r0.json --gens 300 --seconds 9 --out trained/rope-net.json
// (from trail-es.js:) TRAIL-ES: trains the trail corrective network (src/trail-net.js) on top of a running controller over the trail
// course (src/terrain.js makeTrailCourse: a raised floor whose level steps up and down, holes, boxes, balls) with
// evolution strategies, exactly as dash-es.js does for the dash course: antithetic perturbations, the same
// courses for every member of a generation, centred ranks, Adam, episodes over worker threads, the best on
// --held held-out courses saved, a fresh 60-course test at the end (or alone with --eval).
//   node trail-es.js --runner trained/dash-e1b.json --gens 400 --out trained/trail-net.json
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
  const pose = await import('./src/pose.js');
  const simbMod = await import('./src/simbicon.js');
  const { installRopeRunner } = await import('./src/rope-swing.js');
  const { makeRopeCourse } = await import('./src/terrain.js');
  const { makeTrailNet } = await import('./src/trail-net.js');
  setSkill(6);
  config.walk.maxStandSpeed = 20;
  const sim = createSim(Box2D);
  const world = sim.build();
  const HZ = config.physics.hz, P = config.partIndex, LIFT = 2.0;
  const load = (f) => JSON.parse(fs.readFileSync(path.join(root, f), 'utf8'));
  const weights = ElmanNetwork.copy(load('trained/stand-and-walk.json').theta);
  const runStates = load('trained/run-states.json');
  const wf = load('trained/walk-fast-controller.json');
  const walkP = Object.assign(simbMod.decode(wf.best), { track: !!wf.track });
  const runnerP = load('trained/runner-gap-12c.json').params;
  const ropeP = load(runnerFile).params || {};
  // the rope numbers as rope.js decodes them (SPEC defaults plus the saved offsets)
  const { SPEC: RSPEC } = await import('./src/rope-swing.js');
  const rv = load(runnerFile).best || [];
  const p = {}; RSPEC.forEach(([k, b], i) => (p[k] = b + (rv[i] || 0)));
  Object.assign(p, ropeP);
  for (const k of Object.keys(process.env)) if (k.startsWith('P_')) p[k.slice(2)] = +process.env[k]; // overrides to try
  const correct = makeTrailNet(world, config, Box2D);
  return function episode(theta, seed, log) {
    const r = makeSeededRandom(12000 + seed);
    const start = sim.fillTrainingStart({}, r, evalMode(6));
    const speed = 2.2 + r() * 0.6;
    const c = makeRopeCourse(r, { gapMin: 1.4, gapMax: 2.2, lift: LIFT, low: 1.25, ropeAt: 0.5, ropeDensity: 120 });
    const rs = runStates[Math.floor(r() * runStates.length)];
    const pz = rs.pose.slice();
    for (let k = 0; k < world.parts.length; k++) pz[k * 3 + 1] += LIFT;
    Object.assign(start, { startPose: pz, startVel: rs.vel.map((v, i) => (i % 3 === 0 ? v * speed / rs.speed : v)), tilt: 0, drop: 0, settle: 0, frames: Math.round(seconds * HZ) });
    start.walk = null;
    start.terrain = { statics: c.statics, ropes: c.ropes };
    world.rebuild();
    const ep = new sim.Episode(world, weights, 0, 0, Object.assign({ phase: 0 }, start));
    const x0 = (world.px[P.footLeft] + world.px[P.footRight]) / 2;
    ep.runner = { stance: rs.stance, t: -0.1, flight: false, jumping: false, jumpNext: false, strides: 0 };
    const hole = { x0: x0 + c.hole.x0, x1: x0 + c.hole.x1 };
    ep.floorAt = () => LIFT; ep.groundAt = (x) => (x > hole.x0 && x < hole.x1 ? 0 : LIFT);
    const RS = installRopeRunner(ep, world, p, { config, pose, Box2D, runnerP, speed: () => speed, hole, rope: ep.terrain.ropes[0], lift: LIFT, walkP, net: theta ? { correct, theta } : null });
    let fell = false, frames = 0, landedX = null, upAfter = 0, maxX = -1e9, caught = false;
    while (!ep.advanceOneFrame()) {
      frames++;
      const m = ep.measure, pelX = world.px[0];
      if (RS.phase === 'swing') caught = true;
      maxX = Math.max(maxX, pelX);
      const overFloor = pelX < hole.x0 || pelX > hole.x1;
      const feetLow = Math.min(world.py[P.footLeft], world.py[P.footRight]);
      if ((world.py[0] < LIFT - 0.1 || feetLow < LIFT - 0.25) && !(ep.grips[0].joint || ep.grips[1].joint)) { fell = true; break; }
      if (overFloor && m.headHeight - LIFT < 0.5) { fell = true; break; }
      if (['land', 'run2', 'drop', 'steady', 'walk', 'plant'].includes(RS.phase)) {
        if (landedX == null && (m.touch[P.footLeft] || m.touch[P.footRight]) && pelX > hole.x1) landedX = pelX;
        if (landedX != null && m.headHeight - LIFT > 1.3 && m.torsoCos > 0.7) upAfter++;
        if (landedX != null && pelX > hole.x1 + 4 && m.headHeight - LIFT > 1.3) break;
      }
    }
    const after = landedX != null ? Math.max(0, maxX - hole.x1) : 0;
    const pass = !fell && landedX != null && after > 2.5 ? 1 : 0;
    const score = (caught ? 2 : 0) + (landedX != null ? 2 : 0) + 3 * Math.min(1, after / 3) + (fell ? -1 : 1) + Math.min(3, upAfter / HZ) + (pass ? 3 : 0);
    const where = pass ? '' : (fell ? 'fell in ' : 'stopped in ') + RS.phase;
    return { score, fell: pass ? 0 : 1, passed: pass, dist: after, T: frames / HZ, where };
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
  const runner = arg('--runner', 'trained/rope-r0.json');
  const gens = +arg('--gens', 400);
  const pairs = +arg('--pairs', 12);
  const k = +arg('--seeds', 4);
  const sigma = +arg('--sigma', 0.04);
  const lr = +arg('--lr', 0.02);
  const held = +arg('--held', 40);
  const every = +arg('--every', 10);
  const seconds = +arg('--seconds', 12);
  const course = JSON.parse(arg('--course', '{}'));
  const out = arg('--out', 'trained/rope-net.json');
  const init = arg('--init', null);
  const nW = +arg('--workers', Math.max(1, os.cpus().length));
  const { makeSeededRandom, randomNormalSample } = await import('./src/utils.js');
  const { NW, initTheta, DIN, DH, DOUT } = await import('./src/trail-net.js');
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
    const where = {}; for (const r of rs) if (r.where) where[r.where] = (where[r.where] || 0) + 1;
    return { where, score: rs.reduce((a, r) => a + r.score, 0) / n, pass: n - falls, perObstacle: ob / Math.max(1, ob + falls), speed: rs.reduce((a, r) => a + r.dist / r.T, 0) / n, dist: rs.reduce((a, r) => a + r.dist, 0) / n };
  };
  if (args.includes('--eval')) {
    // only score a saved network on the 60 fresh courses (never used in training), against the runner alone
    const ev = arg('--eval', out);
    const th = ev === 'none' ? null : Float64Array.from(JSON.parse(fs.readFileSync(ev, 'utf8')).theta);
    const fs60 = Array.from({ length: +arg('--n', 60) }, (_, i) => +arg('--base', 300) + i);
    const [a, b] = await runAll([{ theta: null, seeds: fs60 }, { theta: th, seeds: fs60 }]);
    const fa = summarize(a), fb = summarize(b);
    console.log('fresh', fs60.length + ': runner alone', (100 * fa.perObstacle).toFixed(1) + '%', 'pass', fa.pass, 'dist', fa.dist.toFixed(1), 'speed', fa.speed.toFixed(2), '| with net', (100 * fb.perObstacle).toFixed(1) + '%', 'pass', fb.pass, 'dist', fb.dist.toFixed(1), 'speed', fb.speed.toFixed(2));
    console.log('falls, runner alone:', JSON.stringify(fa.where));
    console.log('falls, with net:   ', JSON.stringify(fb.where));
    for (const w of workers) w.terminate();
    process.exit(0);
  }
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
