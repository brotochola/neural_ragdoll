// MOUNTAIN: climb an irregular slope (a rigid polygon under a jagged line rising up and to the right):
// ledges, slopes, steps and steep walls. A planner in code looks at the terrain ahead and picks where each
// foot and hand goes; the pose executor (src/pose.js) moves the joints there. On gentle ground it steps
// with the feet only, keeping balance over the standing foot; where the ground ahead rises steeply it
// puts its hands on the rock (a hand grips wherever it touches the mountain) and scrambles on all fours.
// No external forces. The planner's numbers are tuned with the cross-entropy method on progress.
//   node mountain.js --iters 30 --out trained/mountain.json
//   node mountain.js --show --init trained/mountain.json --record out.json
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
const iters = +arg('--iters', 30);
const out = arg('--out', 'trained/mountain.json');
const init = arg('--init', null);
const pop = +arg('--pop', 24);
const elite = +arg('--elite', 6);
const seeds = +arg('--seeds', 3);
const seconds = +arg('--seconds', 60);
const height = +arg('--height', 3); // summit height
const difficulty = +arg('--difficulty', 0.5); // 0 gentle slopes and ledges .. 1 mostly steps and steep rock

const Box2D = await import(pathToFileURL(path.join(root, 'vendor/PhaserBox2D.min.js')).href);
const { createSim } = await import('./src/sim.js');
const { ElmanNetwork } = await import('./src/net.js');
const { makeSeededRandom, randomNormalSample } = await import('./src/utils.js');
const { evalMode } = await import('./src/training.js');
const pose = await import('./src/pose.js');
const { installMountainPlanner } = await import('./src/mountain-planner.js');
const { makeMountain: makeMountainBase } = await import('./src/terrain.js');
const makeMountain = (r) => {
  // TEST=slope:20 a plain slope of 20 degrees; TEST=step:0.2 stairs of 0.2 m (to try one thing at a time)
  const t = process.env.TEST;
  if (!t) return makeMountainBase(r, height, difficulty);
  const [kind, v] = t.split(':');
  const pts = [[-4, 0], [1.2, 0]];
  let x = 1.2, y = 0;
  while (y < height) {
    if (kind === 'slope') { x += 0.3; y += 0.3 * Math.tan(+v * Math.PI / 180); }
    else { x += 0.02; y += +v; pts.push([+x.toFixed(3), +y.toFixed(3)]); x += 0.4; }
    pts.push([+x.toFixed(3), +y.toFixed(3)]);
  }
  pts.push([x + 5, y]);
  const polys = [];
  for (let i = 1; i < pts.length - 1; i++) { const [x1, y1] = pts[i], [x2, y2] = pts[i + 1]; if (y1 <= 0.001 && y2 <= 0.001) continue; polys.push([[x1, -0.2], [x2, -0.2], [x2, y2], [x1, y1]]); }
  return { surface: pts, polys, summitX: x, H: y };
};

setSkill(6);
const sim = createSim(Box2D);
const world = sim.build();
const HZ = config.physics.hz;
const J = config.net.jointCount;
const OUT = J + config.net.gripOutputs;
const P = config.partIndex;
const weights = ElmanNetwork.copy(JSON.parse(fs.readFileSync(path.join(root, 'trained/stand-and-walk.json'), 'utf8')).theta);

const SPEC = [
  ['hipH', 0.9, 0.06], // hips above the ground under them, on two feet
  ['hipHands', 0.62, 0.1], // and when scrambling with the hands down
  ['stepMax', 0.42, 0.08], // longest step forward
  ['stepUp', 0.42, 0.08], // highest step up with the feet alone
  ['handRise', 0.35, 0.1], // hands go down when the ground rises this much within 0.6 m ahead
  ['handPull', 0.35, 0.15], // with hands down the hips sit this share of the way from the feet to the hands
  ['leanK', 0.25, 0.15], // trunk leans in by this share of the slope angle ahead
  ['handLean', 0.35, 0.2], // and this much more when scrambling
  ['clear', 0.12, 0.05], // a moving foot clears the ground in between by this much
  ['tFoot', 0.8, 0.2], // seconds per foot move (weight shift, then the step)
  ['tHand', 0.6, 0.2], // seconds per hand move
  ['shift', 0.35, 0.1], // share of a foot move spent shifting the weight before lifting the foot
  ['kHip', 0.3, 0.2], // balance by hips (as in the pose executor), on two feet
  ['kAnkle', 0.8, 0.3], // higher ankle and hip gains made a body standing on one foot oscillate and fall in 3 s // balance by ankles
  ['torsoGain', 0.4, 0.3], // supporting legs turn the trunk back to its lean
  ['handReach', 0.62, 0.08], // hands go this far at most from the shoulder
  ['footSlope', 0.6, 0.12], // steepest ground a foot stands on without hands (rad)
  ['handAhead', 0.45, 0.1], // a hand goes onto the rock this far ahead of the leading foot
  ['leanRate', 0.8, 0.3], // rad/s the trunk may turn toward its planned lean
];
const decode = (v) => {
  const p = {};
  SPEC.forEach(([k, b], i) => (p[k] = b + (v[i] || 0)));
  for (const k of ['tFoot', 'tHand']) p[k] = Math.max(0.3, p[k]);
  p.shift = Math.max(0.1, Math.min(0.6, p.shift));
  for (const k of Object.keys(process.env)) if (k.startsWith('P_')) p[k.slice(2)] = +process.env[k]; // overrides for trying things
  return p;
};
const smooth = (s) => s * s * (3 - 2 * s);

function episode(p, seed, record) {
  const r = makeSeededRandom(8000 + seed);
  const start = sim.fillTrainingStart({}, r, evalMode(6));
  start.tilt = 0;
  start.walk = null;
  start.frames = Math.round(seconds * HZ);
  const mtn = makeMountain(r);
  start.terrain = { statics: [], polys: mtn.polys, surface: mtn.surface };
  world.rebuild();
  const ep = new sim.Episode(world, weights, 0, 0, Object.assign({ phase: 0 }, start));
  const PL = installMountainPlanner(ep, world, p, { config, pose, Box2D, H: mtn.H });
  const { lim, ground, summitX, H, S } = PL;
  let best = 0;
  let fell = false;
  let topped = false;
  let upTop = 0;
  const recFrames = record ? [] : null;

  let frames = 0;
  const x0 = world.px[0];
  while (!ep.advanceOneFrame()) {
    frames++;
    const m = ep.measure;
    const px = world.px[0];
    if (record && frames % 2 === 0) {
      const pz = [];
      for (let k = 0; k < world.parts.length; k++) pz.push(+world.px[k].toFixed(3), +world.py[k].toFixed(3), +Math.atan2(world.s[k], world.c[k]).toFixed(3));
      recFrames.push({ p: pz, up: m.torsoCos > 0.7 ? 1 : 0, gl: ep.grips[0].joint ? 1 : 0, gr: ep.grips[1].joint ? 1 : 0 });
    }
    const headAbove = m.headHeight - ground(world.px[2]);
    if (headAbove < 0.22 || m.torsoCos < -0.3) { fell = true; break; } // scrambling bends the trunk over the rock, so only a head on the ground or a flipped trunk is a fall
    best = Math.max(best, Math.min(lim.foot[0].y, lim.foot[1].y));
    const onTop = lim.foot.every((f) => f.x > summitX + 0.1) && m.torsoCos > 0.8 && headAbove > 1.3;
    upTop = onTop ? upTop + 1 / HZ : 0;
    if (upTop > 1) { topped = true; break; }
    if (PL.stuck > 40) break;
  }
  const T = frames / HZ;
  const score = 6 * best / H + (fell ? -1 : 0) + (topped ? 4 + 2 * (1 - T / seconds) : 0);
  const why = (topped ? 'TOPPED ' : '') + 'feet up to ' + best.toFixed(2) + '/' + H.toFixed(2) + ' m in ' + T.toFixed(1) + ' s' + (fell ? ' FELL' : '') + (PL.stuck > 40 ? ' stuck' : '');
  const status = topped ? 'reached the top (' + H.toFixed(1) + ' m) in ' + T.toFixed(0) + ' s' : (fell ? 'fell' : 'stopped') + ' at ' + best.toFixed(1) + ' of ' + H.toFixed(1) + ' m';
  return { score, pass: topped ? 1 : 0, why, status, frames: recFrames, surface: S.map(([x, y]) => [+x.toFixed(3), +y.toFixed(3)]) };
}

function evaluate(v, base, n) {
  const p = decode(v);
  let s = 0, pass = 0;
  for (let i = 0; i < n; i++) { const r = episode(p, base + i); s += r.score; pass += r.pass; }
  return { score: s / n, pass };
}

if (args.includes('--show')) {
  const v = init ? JSON.parse(fs.readFileSync(init, 'utf8')).best : [];
  const p = decode(v);
  for (let i = 0; i < seeds; i++) { const r = episode(p, 100 + i); console.log('seed', i, r.why); }
  const rec = arg('--record', null);
  if (rec) {
    const shapes = world.parts.map((q) => ({ hx: +q.hx.toFixed(4), hy: +q.hy.toFixed(4), r: q.r ? +q.r.toFixed(4) : 0, ax: q.ax || 0, ay: q.ay || 0, capsule: !!q.capsule, circle: !!q.circle }));
    const eps = [];
    for (let i = 0; i < +arg('--n', 2); i++) {
      const r = episode(p, 100 + i, true);
      eps.push({ seed: i, surface: r.surface, box: null, walk: null, board: null, status: r.status, frames: r.frames });
    }
    fs.writeFileSync(rec, JSON.stringify({ skill: 6, fps: HZ / 2, shapes, episodes: eps }));
  }
  process.exit(0);
}

const rand = makeSeededRandom(+arg('--rng', 4321));
const mean = new Float64Array(SPEC.length);
if (init) { const sv = JSON.parse(fs.readFileSync(init, 'utf8')).best; for (let d = 0; d < SPEC.length && d < sv.length; d++) mean[d] = sv[d]; }
const sigma = Float64Array.from(SPEC.map((q) => q[2]));
let bestC = null;
for (let it = 0; it < iters; it++) {
  const cands = [];
  for (let c = 0; c < pop; c++) {
    const v = Float64Array.from(mean, (m, d) => m + sigma[d] * randomNormalSample(rand));
    cands.push({ v, ...evaluate(v, 1000 + it * seeds, seeds) });
  }
  cands.sort((a, b) => b.score - a.score);
  if (it % 3 === 2 || it === 0) {
    for (const c of [cands[0], { v: Float64Array.from(mean) }]) {
      const val = evaluate(c.v, 500, 6);
      if (!bestC || val.score > bestC.score) bestC = { v: Float64Array.from(c.v), ...val };
    }
    fs.writeFileSync(out, JSON.stringify({ height, difficulty, best: Array.from(bestC.v), params: decode(bestC.v), score: bestC.score, pass: bestC.pass }));
    console.log('iter', it, 'held-out best', bestC.score.toFixed(3), 'pass', bestC.pass + '/6', 'gen best', cands[0].score.toFixed(3));
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
