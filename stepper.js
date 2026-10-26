// A slow, careful stepper on the feet alone (for slopes, ledges and the walkable parts of the mountain):
// every step is SHIFT (both feet down, the centre of mass goes over the standing foot and settles there),
// SWING (the free foot travels up over the ground in between and down onto its foothold) and nothing
// else; the balance feedback of the pose executor (hips and ankles) keeps the centre of mass over the
// standing foot the whole time. Tested here on plain slopes and stairs before the mountain uses it.
//   node stepper.js --test slope:20 --seeds 2
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { config, setSkill } from './config.js';

const root = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const arg = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : fallback;
};
const test = arg('--test', 'slope:10');
const seeds = +arg('--seeds', 2);
const height = +arg('--height', 1.2);
const seconds = +arg('--seconds', 80);
const Box2D = await import(pathToFileURL(path.join(root, 'vendor/PhaserBox2D.min.js')).href);
const { createSim } = await import('./src/sim.js');
const { ElmanNetwork } = await import('./src/net.js');
const { makeSeededRandom } = await import('./src/utils.js');
const { evalMode } = await import('./src/training.js');
const pose = await import('./src/pose.js');
const { makeStepper } = await import('./src/stepper.js');
setSkill(6);
const sim = createSim(Box2D);
const world = sim.build();
const HZ = config.physics.hz;
const P = config.partIndex;
const weights = ElmanNetwork.copy(JSON.parse(fs.readFileSync(path.join(root, 'trained/stand-and-walk.json'), 'utf8')).theta);
const ENV = process.env;
const SPEC = [
  ['hipH', 0.88, 0.04], ['stepLen', 0.2, 0.06], ['stride', 0.5, 0.1], ['clear', 0.1, 0.04],
  ['tShift', 1.5, 0.4], ['tSwing', 1.0, 0.3], ['tol', 0.04, 0.02], ['vtol', 0.12, 0.05],
  ['kHip', 1.2, 0.4], ['kHipVel', 0.2, 0.15], ['kAnkle', 0.8, 0.4], ['kAnkleVel', 0.4, 0.2], ['kI', 1.5, 0.8],
  ['lean', -0.05, 0.08], ['leanK', 0.4, 0.3], ['torsoGain', 0.4, 0.3], ['ahead', -0.03, 0.04], ['footTilt', 0.3, 0.15], ['toeRoll', 0.2, 0.15],
];
const decode = (v) => {
  const p = {};
  SPEC.forEach(([k, b], i) => (p[k] = b + (v[i] || 0)));
  p.tShift = Math.max(0.3, p.tShift); p.tSwing = Math.max(0.4, p.tSwing); p.tol = Math.max(0.01, p.tol); p.vtol = Math.max(0.03, p.vtol);
  p.stepLen = Math.max(0.1, p.stepLen); p.toeRoll = Math.max(0, p.toeRoll); p.toeRoll = Math.max(0, p.toeRoll);
  for (const k of Object.keys(ENV)) if (k.startsWith('P_')) p[k.slice(2)] = +ENV[k];
  return p;
};
const init = arg('--init', null);
const p0 = decode(init ? JSON.parse(fs.readFileSync(init, 'utf8')).best : []);

function terrain(kind, v, r) {
  const pts = [[-4, 0], [0.75, 0]];
  let x = 0.75, y = 0;
  while (y < height) {
    if (kind === 'slope') { x += 0.3; y += 0.3 * Math.tan(+v * Math.PI / 180); }
    else if (kind === 'step') { x += 0.02; y += +v; pts.push([+x.toFixed(3), +y.toFixed(3)]); x += 0.4; }
    else if (kind === 'flat') { x += 1; if (x > 8) y = height; }
    else { const { makeMountain } = MT; return makeMountain(r, height, +v); }
    pts.push([+x.toFixed(3), +y.toFixed(3)]);
  }
  pts.push([x + 5, y]);
  const polys = [];
  for (let i = 1; i < pts.length - 1; i++) { const [x1, y1] = pts[i], [x2, y2] = pts[i + 1]; if (y1 <= 0.001 && y2 <= 0.001) continue; polys.push([[x1, -0.2], [x2, -0.2], [x2, y2], [x1, y1]]); }
  return { surface: pts, polys, summitX: x, H: y };
}
const MT = await import('./src/terrain.js');

function episode(seed, record, p = p0, testName = test) {
  const r = makeSeededRandom(8000 + seed);
  const start = sim.fillTrainingStart({}, r, evalMode(6));
  start.tilt = 0;
  start.walk = null;
  start.frames = Math.round(seconds * HZ);
  const [kind, v] = testName.split(':');
  const mtn = terrain(kind, v, r);
  start.terrain = { statics: [], polys: mtn.polys, surface: mtn.surface };
  world.rebuild();
  const ep = new sim.Episode(world, weights, 0, 0, Object.assign({ phase: 0 }, start));
  const S = ep.terrain.surface;
  const ground = (x) => {
    if (x <= S[0][0]) return 0;
    for (let i = 0; i < S.length - 1; i++) if (x <= S[i + 1][0]) { const [x1, y1] = S[i], [x2, y2] = S[i + 1]; return x2 > x1 ? y1 + (y2 - y1) * (x - x1) / (x2 - x1) : Math.max(y1, y2); }
    return S[S.length - 1][1];
  };
  const x0 = Math.min(...[0, 1].map((k) => pose.soleOf(world, k ? P.footRight : P.footLeft).x));
  const st = makeStepper(ep, world, p, { config, pose, ground, debug: +(ENV.DEBUG_ST || 0) });
  ep.updateGrip = ((orig) => function () { st.step(this); orig(); })(ep.updateGrip.bind(ep));
  const frames = record ? [] : null;
  let fell = false, best = 0;
  while (!ep.advanceOneFrame()) {
    const m = ep.measure;
    if (record && ep.frame % 2 === 0) {
      const pz = [];
      for (let k = 0; k < world.parts.length; k++) pz.push(+world.px[k].toFixed(3), +world.py[k].toFixed(3), +Math.atan2(world.s[k], world.c[k]).toFixed(3));
      frames.push({ p: pz, up: m.torsoCos > 0.7 ? 1 : 0 });
    }
    if (m.headHeight - ground(world.px[2]) < 0.9) { fell = true; break; }
    best = Math.max(best, Math.min(st.feet[0].y, st.feet[1].y) - 0.05);
    if (best >= mtn.H - 0.01 && st.steps > 3 && Math.min(st.feet[0].x, st.feet[1].x) > mtn.summitX + 0.3) break;
  }
  const dist = Math.min(st.feet[0].x, st.feet[1].x) - Math.min(...[0, 1].map((k) => pose.soleOf(world, k ? P.footRight : P.footLeft).x)) ;
  return { fell, best, H: mtn.H, t: ep.frame / HZ, steps: st.steps, frames, surface: S, x: Math.min(st.feet[0].x, st.feet[1].x) - x0 };
}

if (args.includes('--search')) {
  const { makeSeededRandom: mk, randomNormalSample } = await import('./src/utils.js');
  const tests = arg('--tests', 'flat:0,slope:10,slope:20,step:0.1,step:0.2,mountain:0.2').split(',');
  const iters = +arg('--iters', 40), pop = +arg('--pop', 24), elite = +arg('--elite', 6), out = arg('--out', 'trained/stepper.json');
  const scoreOf = (e, t) => { const g = t.startsWith('flat') ? Math.min(1, e.x / 5) : e.best / e.H; return 4 * g + (e.fell ? 0 : Math.min(1, 3 * g)); }; // standing still is not a success
  const evaluate = (v, base, n) => { const p = decode(v); let sc = 0; for (let i = 0; i < n; i++) for (const t of tests) sc += scoreOf(episode(base + i, false, p, t), t); return sc / (n * tests.length); };
  const rand = mk(+arg('--rng', 7));
  const mean = new Float64Array(SPEC.length);
  if (init) JSON.parse(fs.readFileSync(init, 'utf8')).best.forEach((x, d) => { if (d < SPEC.length) mean[d] = x; });
  const sigma = Float64Array.from(SPEC.map((q) => q[2]));
  let bestC = null;
  for (let it = 0; it < iters; it++) {
    const cands = [];
    for (let c = 0; c < pop; c++) { const v = Float64Array.from(mean, (m, d) => m + sigma[d] * randomNormalSample(rand)); cands.push({ v, score: evaluate(v, 1000 + it * seeds, seeds) }); }
    cands.sort((a, b) => b.score - a.score);
    if (it % 2 === 1 || it === 0) {
      for (const c of [cands[0], { v: Float64Array.from(mean) }]) { const val = evaluate(c.v, 500, 3); if (!bestC || val > bestC.score) bestC = { v: Float64Array.from(c.v), score: val }; }
      fs.writeFileSync(out, JSON.stringify({ best: Array.from(bestC.v), params: decode(bestC.v), score: bestC.score, tests }));
      console.log('iter', it, 'held-out best', bestC.score.toFixed(3), 'gen best', cands[0].score.toFixed(3));
    }
    const top = cands.slice(0, elite);
    for (let d = 0; d < SPEC.length; d++) {
      const m = top.reduce((a, c) => a + c.v[d], 0) / elite;
      const sd = Math.sqrt(top.reduce((a, c) => a + (c.v[d] - m) ** 2, 0) / elite);
      mean[d] = 0.7 * m + 0.3 * mean[d];
      sigma[d] = Math.max(SPEC[d][2] * 0.15, 0.7 * sd + 0.3 * sigma[d]);
    }
  }
  console.log('done ->', out);
  process.exit(0);
}
let tot = 0;
for (let i = 0; i < seeds; i++) {
  const e = episode(i);
  tot += e.best / e.H;
  console.log('seed', i, (e.fell ? 'FELL' : 'ok  '), 'up', e.best.toFixed(2) + '/' + e.H.toFixed(2), 'in', e.t.toFixed(1), 's', e.steps, 'steps');
}
console.log('mean share', (tot / seeds).toFixed(2));
const rec = arg('--record', null);
if (rec) {
  const shapes = world.parts.map((q) => ({ hx: +q.hx.toFixed(4), hy: +q.hy.toFixed(4), r: q.r ? +q.r.toFixed(4) : 0, ax: q.ax || 0, ay: q.ay || 0, capsule: !!q.capsule, circle: !!q.circle }));
  const e = episode(+arg('--rec-seed', 0), true);
  fs.writeFileSync(rec, JSON.stringify({ skill: 6, fps: HZ / 2, shapes, episodes: [{ seed: 0, surface: e.surface, frames: e.frames, status: (e.fell ? 'fell' : 'ok') + ' at ' + e.best.toFixed(2) }] }));
}
