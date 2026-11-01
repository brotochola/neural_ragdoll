// ES-RECOVER: fine-tunes the get-up expert by evolution strategies inside the level's own recovery
// (src/recover.js: settle, get up, stand, walk), on real falls recorded on the trail (trail-es.js
// --dump-falls) lying as they fell with the blocks around them, plus some ordinary lying starts so the
// clean get-up is not forgotten. Scored on what the level needs: up and walking, and soon.
//   node es-recover.js --from trained/getup-expert-trail.json --out trained/getup-expert-rec.json --gens 60
// The first 40 falls of the list are held out (selection); training draws from the rest.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads';

const root = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const arg = (n, f) => { const i = args.indexOf(n); return i >= 0 ? args[i + 1] : f; };
const fallsFiles = arg('--falls', 'trained/trail-falls.json,trained/trail-falls2.json');
const HELD = +arg('--held', 40);
const TRAVEL = +arg('--travel', 0);

async function makeEnv(files) {
  const { config, setSkill } = await import('./config.js');
  const Box2D = await import(pathToFileURL(path.join(root, 'vendor/PhaserBox2D.min.js')).href);
  const { createSim } = await import('./src/sim.js');
  const { ElmanNetwork } = await import('./src/net.js');
  const { makeSeededRandom } = await import('./src/utils.js');
  const { evalMode } = await import('./src/training.js');
  const { installRecovery } = await import('./src/recover.js');
  setSkill(6);
  config.walk.maxStandSpeed = 20;
  const sim = createSim(Box2D);
  const world = sim.build();
  const HZ = config.physics.hz, P = config.partIndex;
  const params = JSON.parse(fs.readFileSync(path.join(root, 'play/params.json'), 'utf8'));
  const weights = ElmanNetwork.copy(JSON.parse(fs.readFileSync(path.join(root, 'trained/stand-and-walk.json'), 'utf8')).theta);
  const falls = files.split(',').flatMap((f) => JSON.parse(fs.readFileSync(path.join(root, f), 'utf8')));
  // job: { k: fall index (>= 0) or -1 - n for ordinary lying start n }
  return function episode(theta, handover, k) {
    const r = makeSeededRandom(4000 + Math.abs(k));
    const start = sim.fillTrainingStart({}, r, evalMode(6));
    let statics = [];
    if (k >= 0) {
      const f = falls[k % falls.length];
      const mid = (f.pose[P.footLeft * 3] + f.pose[P.footRight * 3]) / 2;
      statics = f.blocks.map((b) => ({ dx: b.x0 - mid, w: b.x1 - b.x0, h: b.top }));
      Object.assign(start, { startPose: f.pose.slice(), startVel: f.vel, tilt: 0, drop: 0, settle: 0, shiftX: 0 });
    } else {
      setSkill(9);
      Object.assign(start, sim.fillTrainingStart({}, makeSeededRandom(config.eval.seed + 9 * 10007 + (-1 - k) * 17), evalMode(9)));
      setSkill(6);
    }
    start.frames = Math.round(16 * HZ);
    start.walk = null;
    start.getup = false;
    start.terrain = { statics };
    world.rebuild();
    const ep = new sim.Episode(world, weights, start.tilt || 0, start.drop || 0, Object.assign({ phase: 0 }, start));
    const R = installRecovery(ep, world, { config, ElmanNetwork, getup: { theta, handover }, walkP: params.walkFast, floorAt: () => 0 });
    R.start();
    let maxHead = 0, stood = false, walkT = 0, gx = null, travel = 0;
    while (!ep.advanceOneFrame()) {
      if (R.phase === 'getup') { maxHead = Math.max(maxHead, ep.measure.headHeight); if (gx == null) gx = world.px[0]; }
      if (R.phase === 'stand' && gx != null && !travel) travel = Math.abs(world.px[0] - gx) + 1e-6;
      if (R.phase === 'stand' || R.phase === 'walk') stood = true;
      if (R.phase === 'walk') walkT += 1 / HZ;
      if (R.phase === 'done' || R.phase === 'failed') break;
    }
    const done = R.phase === 'done';
    const t = ep.frame / HZ;
    // up and walking (sooner is better), else how far it got
    // TRAVEL (> 0): less for every metre the get-up carries the hips past 0.8 m (it went 3 m and hit what was ahead)
    let score = done ? 3 + (16 - t) / 16 : (stood ? 1 : 0) + Math.min(1, walkT / 3) + 0.5 * Math.min(1, maxHead / 1.6);
    if (TRAVEL && stood) score -= TRAVEL * Math.max(0, travel - 0.8);
    if (process.env.DEBUG_EP) console.log('ep', k, 'done', done, 't', t.toFixed(1), 'travel', travel.toFixed(2), 'score', score.toFixed(2), 'TRAVEL', TRAVEL);
    return { score, done: done ? 1 : 0, travel: stood ? travel : 0 };
  };
}

if (!isMainThread) {
  const episode = await makeEnv(workerData.files);
  parentPort.on('message', ({ id, theta, handover, ks }) => {
    const th = Float32Array.from(theta);
    parentPort.postMessage({ id, res: ks.map((k) => episode(th, handover, k)) });
  });
} else {
  const { makeSeededRandom, randomNormalSample } = await import('./src/utils.js');
  const from = arg('--from', 'trained/getup-expert-trail.json');
  const out = arg('--out', 'trained/getup-expert-rec.json');
  const gens = +arg('--gens', 60), pairs = +arg('--pairs', 8), per = +arg('--per', 6);
  const sigma = +arg('--sigma', 0.01), lr = +arg('--lr', 0.0005), every = +arg('--every', 5);
  const saved = JSON.parse(fs.readFileSync(path.join(root, from), 'utf8'));
  const handover = saved.handover;
  const theta = Float64Array.from(saved.theta);
  const NW = theta.length;
  const allFalls = fallsFiles.split(',').flatMap((f) => JSON.parse(fs.readFileSync(path.join(root, f), 'utf8')));
  const nFalls = allFalls.length;
  // --hard: training draws only falls with a block (over 10 cm) less than 3 m ahead (the get-up travels 2-3 m)
  const hardOf = (f) => f.blocks.some((b) => b.x1 > f.pose[0] && b.x0 - f.pose[0] < 3 && b.top > 0.1);
  const trainPool = Array.from({ length: nFalls - HELD }, (_, i) => HELD + i).filter((k) => !args.includes('--hard') || hardOf(allFalls[k]));
  console.log('training falls', trainPool.length, 'of', nFalls - HELD);
  const nW = +arg('--workers', Math.max(1, os.cpus().length));
  const workers = Array.from({ length: nW }, () => new Worker(new URL(import.meta.url), { argv: process.argv.slice(2), workerData: { files: fallsFiles } }));
  let nextId = 0;
  const pending = new Map();
  for (const w of workers) w.on('message', ({ id, res }) => { pending.get(id)(res); pending.delete(id); });
  // jobs: [{ theta, ks }] -> results per job, spread over the workers one episode at a time
  async function runAll(jobs) {
    const flat = [];
    jobs.forEach((j, ji) => j.ks.forEach((k, ki) => flat.push({ ji, ki, k, theta: j.theta })));
    const res = jobs.map((j) => new Array(j.ks.length));
    let cursor = 0;
    await Promise.all(workers.map(async (w) => {
      while (cursor < flat.length) {
        const job = flat[cursor++];
        const id = nextId++;
        const r = await new Promise((resolve) => { pending.set(id, resolve); w.postMessage({ id, theta: Array.from(job.theta), handover, ks: [job.k] }); });
        res[job.ji][job.ki] = r[0];
      }
    }));
    return res;
  }
  const heldKs = Array.from({ length: Math.min(HELD, 30) }, (_, i) => i).concat(Array.from({ length: args.includes('--ordinary') ? 20 : 8 }, (_, i) => -1 - i));
  const evalHeld = async (th) => { const [r] = await runAll([{ theta: th, ks: heldKs }]); const st = r.filter((x) => x.travel > 0); return { done: r.reduce((a, x) => a + x.done, 0), score: r.reduce((a, x) => a + x.score, 0) / r.length, travel: st.reduce((a, x) => a + x.travel, 0) / Math.max(1, st.length) }; };
  let best = await evalHeld(theta);
  console.log('start held-out: up and walking', best.done + '/' + heldKs.length, 'score', best.score.toFixed(3), 'travel', best.travel.toFixed(2));
  const rand = makeSeededRandom(+arg('--rng', 7));
  const normal = () => randomNormalSample(rand);
  const m = new Float64Array(NW), v = new Float64Array(NW);
  const t0 = Date.now();
  for (let g = 0; g < gens; g++) {
    // this generation's starts: falls outside the held-out ones, one ordinary lying start in six
    const ks = Array.from({ length: per }, (_, i) => (args.includes('--ordinary') ? -1000 - g * per - i : i === per - 1 && g % 2 === 0 ? -100 - g : trainPool[Math.floor(rand() * trainPool.length)]));
    const eps = [], jobs = [];
    for (let p = 0; p < pairs; p++) {
      const e = new Float64Array(NW);
      for (let d = 0; d < NW; d++) e[d] = normal();
      eps.push(e);
      const a = new Float64Array(NW), b = new Float64Array(NW);
      for (let d = 0; d < NW; d++) { a[d] = theta[d] + sigma * e[d]; b[d] = theta[d] - sigma * e[d]; }
      jobs.push({ theta: a, ks }, { theta: b, ks });
    }
    const res = await runAll(jobs);
    const f = res.map((rs) => rs.reduce((a, r) => a + r.score, 0) / rs.length);
    const order = f.map((x, i) => [x, i]).sort((p, q) => p[0] - q[0]);
    const u = new Float64Array(f.length);
    order.forEach(([, i], r) => (u[i] = r / (f.length - 1) - 0.5));
    const grad = new Float64Array(NW);
    for (let p = 0; p < pairs; p++) { const d = u[2 * p] - u[2 * p + 1]; if (d) for (let i = 0; i < NW; i++) grad[i] += d * eps[p][i]; }
    const c1 = 1 - Math.pow(0.9, g + 1), c2 = 1 - Math.pow(0.999, g + 1);
    for (let i = 0; i < NW; i++) {
      const gi = grad[i] / (pairs * sigma);
      m[i] = 0.9 * m[i] + 0.1 * gi;
      v[i] = 0.999 * v[i] + 0.001 * gi * gi;
      theta[i] += (lr * (m[i] / c1)) / (Math.sqrt(v[i] / c2) + 1e-8);
    }
    const passes = res.reduce((a, rs) => a + rs.reduce((b, r) => b + r.done, 0), 0);
    let msg = 'gen ' + g + ' mean ' + (f.reduce((a, b) => a + b, 0) / f.length).toFixed(3) + ' up ' + passes + '/' + pairs * 2 * per;
    if ((g + 1) % every === 0) {
      const e = await evalHeld(theta);
      msg += ' | held-out ' + e.done + '/' + heldKs.length + ' score ' + e.score.toFixed(3) + ' travel ' + e.travel.toFixed(2);
      if (e.score > best.score && e.done >= best.done - 2) {
        best = e;
        fs.writeFileSync(path.resolve(root, out), JSON.stringify(Object.assign({}, saved, { role: 'getup-expert-rec', gen: g, eval: { held: e.done + '/' + heldKs.length }, theta: Array.from(theta) })));
        msg += ' -> saved';
      }
    }
    console.log(msg, ((Date.now() - t0) / 60000).toFixed(1) + ' min');
  }
  for (const w of workers) w.terminate();
  process.exit(0);
}
