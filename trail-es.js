// TRAIL-ES: trains the trail corrective network (src/trail-net.js) on top of a running controller over the trail
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

async function makeEnv(runnerFile, seconds, course, v2) {
  const { config, setSkill } = await import('./config.js');
  const Box2D = await import(pathToFileURL(path.join(root, 'vendor/PhaserBox2D.min.js')).href);
  const { createSim } = await import('./src/sim.js');
  const { ElmanNetwork } = await import('./src/net.js');
  const { makeSeededRandom } = await import('./src/utils.js');
  const { evalMode } = await import('./src/training.js');
  const { installRunner } = await import('./src/run-selector.js');
  const { makeTrailCourse } = await import('./src/terrain.js');
  const { installTrail } = await import('./src/trail.js');
  const { makeTrailNet } = await import('./src/trail-net.js');
  const { makeTrailNet2 } = await import('./src/trail-net2.js');
  setSkill(6);
  config.walk.maxStandSpeed = 20;
  const sim = createSim(Box2D);
  const world = sim.build();
  const HZ = config.physics.hz;
  const weights = ElmanNetwork.copy(JSON.parse(fs.readFileSync(path.join(root, 'trained/stand-and-walk.json'), 'utf8')).theta);
  const runStates = JSON.parse(fs.readFileSync(path.join(root, 'trained/run-states.json'), 'utf8'));
  const p = JSON.parse(fs.readFileSync(path.join(root, runnerFile), 'utf8')).params;
  for (const k of Object.keys(process.env)) if (k.startsWith('P_')) p[k.slice(2)] = +process.env[k]; // overrides to try
  const correct = v2 ? makeTrailNet2(world, config, Box2D) : makeTrailNet(world, config, Box2D);
  return function episode(theta, seed, log, dump) {
    const r = makeSeededRandom(9100 + seed);
    const start = sim.fillTrainingStart({}, r, evalMode(6));
    // course.mix: every other course is like the level's first stretch (on the ground, boxes, balls, low steps, no holes)
    const ground = course.mix && seed % 2 === 1;
    const c = makeTrailCourse(r, Object.assign({ length: 4 + 5.2 * seconds }, ground ? { start: 0, minLevel: 0, maxLevel: 0.45, pHole: 0, pBox: 0.3, pBall: 0.2 } : {}, course));
    const rs = runStates[Math.floor(r() * runStates.length)];
    const v = p.vRun || 2.6;
    const pz = rs.pose.slice();
    for (let q = 0; q < world.parts.length; q++) pz[q * 3 + 1] += c.lift;
    Object.assign(start, { startPose: pz, startVel: rs.vel.map((q, i) => (i % 3 === 0 ? q * v / rs.speed : q)), tilt: 0, drop: 0, settle: 0, frames: Math.round(seconds * HZ) });
    start.walk = null;
    start.terrain = { statics: c.statics, balls: c.balls };
    world.rebuild();
    const ep = new sim.Episode(world, weights, 0, 0, Object.assign({ phase: 0 }, start));
    ep.runner = { stance: rs.stance, t: -0.1, flight: false, jumping: false, jumpNext: false, strides: 0 };
    const tr = installTrail(ep, c, Box2D);
    installRunner(ep, world, p, { config, obstacles: tr.obstacles, floorAt: tr.floorAt, speed: () => v, net: theta ? { correct, theta } : null });
    const x0 = world.px[0];
    let frames = 0, fell = false, low = 0;
    while (!ep.advanceOneFrame()) {
      frames++;
      const m = ep.measure, fl = tr.floorAt(world.px[0]);
      const head = world.py[2] - fl;
      low = head < 1.15 || (m.torsoCos || 0) < 0.3 ? low + 1 : 0;
      // down in a hole (the hips under the floor), the head low, or crouched / tipped over for a quarter second
      if (world.py[0] < fl - 0.1 || head < 0.85 || low > 15) { fell = true; break; }
    }
    const dist = world.px[0] - x0;
    const feats = tr.features.filter((f) => f.type !== 'down');
    const passed = feats.filter((f) => f.x1 < world.px[0] - 0.3).length;
    // --dump-falls: let the body come to rest (motors relaxed 1.5 s), then keep it lying as it fell, with the
    // boxes and steps around it, heights from the floor it lies on (es-expert.js --falls trains the get-up on them)
    if (fell && dump) {
      for (let f = 0; f < 90; f++) { for (let j = 0; j < config.net.jointCount; j++) ep.action[j] = 0; ep.updateGrip = () => {}; if (ep.advanceOneFrame()) break; }
      const fl = tr.floorAt(world.px[0]);
      if (world.py[0] > fl - 0.1) {
        const pose = [], vel = [];
        for (let q = 0; q < world.parts.length; q++) { pose.push(+world.px[q].toFixed(4), +(world.py[q] - fl).toFixed(4), +Math.atan2(world.s[q], world.c[q]).toFixed(4)); vel.push(+world.vx[q].toFixed(3), +(world.vy[q]).toFixed(3), +world.spin[q].toFixed(3)); }
        const blocks = [];
        for (const b of ep.terrain.statics) if (b.x1 > world.px[0] - 3 && b.x0 < world.px[0] + 3 && b.top - fl > 0.03) blocks.push({ x0: +b.x0.toFixed(3), x1: +b.x1.toFixed(3), top: +(b.top - fl).toFixed(3) });
        dump.push({ pose, vel, blocks });
      }
    }
    let where = '';
    if (fell) {
      const px = world.px[0];
      const o = tr.features.reduce((a, b) => (Math.abs((b.x0 + b.x1) / 2 - px) < Math.abs((a.x0 + a.x1) / 2 - px) ? b : a), tr.features[0]);
      where = (px < o.x0 ? 'before ' : px <= o.x1 ? 'over ' : 'after ') + o.type;
    }
    if (log) console.log('seed', seed, fell ? 'FELL ' + (frames / HZ).toFixed(1) + 's ' + where : 'ok', 'dist', dist.toFixed(1), 'passed', passed);
    const score = 2 * Math.max(0, dist) / seconds + 4 * Math.min(1, frames / (seconds * HZ)) + (fell ? 0 : 1);
    return { score, fell: fell ? 1 : 0, passed, dist, T: frames / HZ, where };
  };
}

if (!isMainThread) {
  const episode = await makeEnv(workerData.runner, workerData.seconds, workerData.course, workerData.v2);
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
  const out = arg('--out', 'trained/trail-net.json');
  const init = arg('--init', null);
  const nW = +arg('--workers', Math.max(1, os.cpus().length));
  const { makeSeededRandom, randomNormalSample } = await import('./src/utils.js');
  // --v2: the v2 network (src/trail-net2.js: arms and step length too); a v1 --init is carried over (fromV1)
  const V2 = args.includes('--v2');
  const netMod = V2 ? await import('./src/trail-net2.js') : await import('./src/trail-net.js');
  const { NW, initTheta, DIN, DH, DOUT } = netMod;
  const rand = makeSeededRandom(+arg('--rng', 5));
  const normal = () => randomNormalSample(rand);
  let theta = init ? Float64Array.from(JSON.parse(fs.readFileSync(init, 'utf8')).theta) : initTheta(rand);
  if (V2 && theta.length !== NW) theta = netMod.fromV1(theta);

  const workers = Array.from({ length: nW }, () => new Worker(file, { workerData: { runner, seconds, course, v2: V2 } }));
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
  if (args.includes('--dump-falls')) {
    // in this thread: run courses with the network, keep the falls (lying, with the blocks around)
    for (const w of workers) w.terminate();
    const ep1 = await makeEnv(runner, seconds, course);
    const th = Float64Array.from(JSON.parse(fs.readFileSync(arg('--net', out), 'utf8')).theta);
    const dump = [];
    for (let i = 0; i < +arg('--n', 200) && dump.length < +arg('--max', 120); i++) ep1(th, +arg('--dump-base', 20000) + i, false, dump);
    fs.writeFileSync(arg('--dump-falls'), JSON.stringify(dump));
    console.log('falls', dump.length, '->', arg('--dump-falls'));
    process.exit(0);
  }
  if (args.includes('--eval')) {
    // only score a saved network on the 60 fresh courses (never used in training), against the runner alone
    const ev = arg('--eval', out);
    let th = ev === 'none' ? null : Float64Array.from(JSON.parse(fs.readFileSync(ev, 'utf8')).theta);
    if (th && V2 && th.length !== NW) th = netMod.fromV1(th);
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
      if (!best || h.score > best.score) { best = { ...h, theta: Float64Array.from(theta), gen: g }; fs.writeFileSync(out, JSON.stringify({ runner, v2: V2, DIN, DH, DOUT, gen: g, theta: Array.from(best.theta), held: { n: held, ...h }, base })); }
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
