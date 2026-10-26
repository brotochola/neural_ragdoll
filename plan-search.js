// Tunes the climb planner's numbers (src/planner.js, ~30 offsets with a physical meaning) with the
// cross-entropy method, using the two-level climb in plan-climb.js. New practice starts every
// generation; the version kept is the one that does best on held-out starts.
//   node plan-search.js --height 0.75 --iters 40 --out trained/plan-climb-075.json [--init file]
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const arg = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : fallback;
};
const iters = +arg('--iters', 40);
const out = arg('--out', 'trained/plan-climb.json');
const init = arg('--init', null);
const pop = +arg('--pop', 16);
const elite = +arg('--elite', 4);
const seeds = +arg('--seeds', 3);
const { runClimb } = await import('./plan-climb.js');
const { makeSeededRandom, randomNormalSample } = await import('./src/utils.js');

// name, spread (the defaults live in climbPlan)
const SPEC = [
  ['handX', 0.05], ['handAbove', 0.05], ['lean', 0.15], ['standBack', 0.08], ['reachT', 0.2], ['downT', 0.1],
  ['crouchY', 0.06], ['crouchT', 0.15], ['supportArm', 0.06], ['supportLean', 0.2], ['pushT', 0.15],
  ['tuckX', 0.08], ['footX', 0.06], ['tuckT', 0.15],
  ['liftX', 0.08], ['liftY', 0.06], ['liftT', 0.2], ['placeT', 0.15],
  ['riseX', 0.08], ['riseY', 0.08], ['riseLean', 0.15], ['riseT', 0.25],
  ['trailX', 0.06], ['trailLift', 0.06], ['trailT', 0.2], ['standX', 0.06], ['standT', 0.2],
];
const DEFAULTS = {
  handX: 0.12, handAbove: 0.15, lean: -0.35, standBack: 0.30, reachT: 0.7, downT: 0.3,
  crouchY: 0.72, crouchT: 0.4, supportArm: 0.42, supportLean: -0.6, pushT: 0.45,
  tuckX: -0.10, footX: 0.12, tuckT: 0.5,
  liftX: -0.05, liftY: 0.15, liftT: 0.8, placeT: 0.6,
  riseX: 0.15, riseY: 0.78, riseLean: -0.45, riseT: 1.0,
  trailX: 0.33, trailLift: 0.2, trailT: 0.8, standX: 0.26, standT: 1.0,
};
const dims = SPEC.length;
const toParams = (v) => {
  const p = {};
  SPEC.forEach(([name], i) => (p[name] = DEFAULTS[name] + v[i]));
  for (const k of ['reachT', 'downT', 'crouchT', 'pushT', 'tuckT', 'liftT', 'placeT', 'riseT', 'trailT', 'standT']) p[k] = Math.max(0.1, p[k]);
  return p;
};

function score(r) {
  return 5 * (r.pass ? 1 : 0) + 2 * (r.onTop ? 1 : 0) + 2 * r.standShare + Math.max(-1, Math.min(1, r.headOver / 1.45)) + 0.3 * r.key - 3 * r.tipped;
}
function evaluate(v, base, n) {
  const p = toParams(v);
  let s = 0;
  let pass = 0;
  let top = 0;
  for (let i = 0; i < n; i++) {
    const r = runClimb(base + i, p, false);
    s += score(r);
    pass += r.pass ? 1 : 0;
    top += r.onTop ? 1 : 0;
  }
  return { score: s / n, pass, top };
}

const rand = makeSeededRandom(99);
const normal = () => randomNormalSample(rand);
const mean = new Float64Array(dims);
if (init) {
  const saved = JSON.parse(fs.readFileSync(path.resolve(root, init), 'utf8'));
  SPEC.forEach(([name], i) => (mean[i] = (saved.params[name] != null ? saved.params[name] : DEFAULTS[name]) - DEFAULTS[name]));
}
const sigma = Float64Array.from(SPEC.map((q) => q[1]));
let best = null;
for (let it = 0; it < iters; it++) {
  const cands = [];
  for (let c = 0; c < pop; c++) {
    const v = new Float64Array(dims);
    for (let d = 0; d < dims; d++) v[d] = mean[d] + sigma[d] * normal();
    cands.push({ v, ...evaluate(v, 1000 + it * seeds, seeds) });
  }
  cands.sort((a, b) => b.score - a.score);
  const top = cands.slice(0, elite);
  for (let d = 0; d < dims; d++) {
    let m = 0;
    for (const c of top) m += c.v[d];
    m /= top.length;
    let s = 0;
    for (const c of top) s += (c.v[d] - m) ** 2;
    s = Math.sqrt(s / top.length);
    mean[d] = 0.7 * m + 0.3 * mean[d];
    sigma[d] = Math.max(SPEC[d][1] * 0.2, 0.7 * s + 0.3 * sigma[d]);
  }
  if (it % 4 === 3 || it === 0) {
    for (const c of [cands[0], { v: Float64Array.from(mean) }]) {
      const val = evaluate(c.v, 500, 8);
      if (!best || val.score > best.score) {
        best = { v: Float64Array.from(c.v), ...val };
        fs.writeFileSync(path.resolve(root, out), JSON.stringify({ params: toParams(best.v), score: best.score, pass: best.pass, top: best.top }));
      }
    }
    console.log('iter', it, 'held-out best', best.score.toFixed(2), 'pass', best.pass + '/8', 'on top', best.top + '/8', 'gen best', cands[0].score.toFixed(2));
  }
}
console.log('done ->', out);
