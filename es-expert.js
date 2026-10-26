// Fine-tunes the get-up expert (from bc-expert.js) by evolution strategies on the real outcome:
// getting up from lying and then standing, with the stand-and-walk net taking over at the handover.
// Imitation can only be as good as the keyframe motion it copies (13/20); this goes past it.
//   node es-expert.js --from trained/getup-expert.json --out trained/getup-expert-es.json --gens 200
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
const from = arg('--from', 'trained/getup-expert.json');
const out = arg('--out', 'trained/getup-expert-es.json');
const standFile = arg('--stand', 'trained/stand-and-walk.json');
const gens = +arg('--gens', 200);
const pairs = +arg('--pairs', 12);
const seedsPer = +arg('--seeds', 4);
const sigma = +arg('--sigma', 0.02);
const lr = +arg('--lr', 0.003);
const evalN = +arg('--eval-n', 20);
const evalEvery = +arg('--eval-every', 5);
// --falls file: half the starts are real falls from the course (course.js --dump-falls), lying as they
// fell, with the blocks around them rebuilt; the other half the usual lying starts.
const fallsFile = arg('--falls', null);
const falls = fallsFile ? JSON.parse(fs.readFileSync(path.resolve(root, fallsFile), 'utf8')) : null;

const Box2D = await import(pathToFileURL(path.join(root, 'vendor/PhaserBox2D.min.js')).href);
const { createSim } = await import('./src/sim.js');
const { ElmanNetwork, outputCountOf } = await import('./src/net.js');
const { makeSeededRandom, randomNormalSample } = await import('./src/utils.js');
const { evalMode } = await import('./src/training.js');

const OUT = outputCountOf();
const HZ = config.physics.hz;
const sim = createSim(Box2D);
const world = sim.build();
const standW = Float32Array.from(JSON.parse(fs.readFileSync(path.join(root, standFile), 'utf8')).theta);
const saved = JSON.parse(fs.readFileSync(path.join(root, from), 'utf8'));
const handover = saved.handover;
let theta = Float64Array.from(saved.theta);

let useFalls = null;
function run(w, seed, phase) {
  setSkill(9);
  const rand = makeSeededRandom(seed);
  const start = sim.fillTrainingStart({}, rand, evalMode(9));
  let side = start.tilt > 0 ? 'up' : 'down';
  // useFalls: true = always a fall, false = never (official eval), null = half and half (training)
  if (falls && (useFalls === true || (useFalls === null && rand() < 0.5))) {
    const f = falls[Math.floor(rand() * falls.length)];
    const P = config.partIndex;
    const mid = (f.pose[P.footLeft * 3] + f.pose[P.footRight * 3]) / 2;
    start.startPose = f.pose;
    start.startVel = f.vel;
    start.tilt = 0;
    start.drop = 0;
    start.settle = 0;
    start.terrain = { statics: f.blocks.map((b) => ({ dx: b.x0 - mid, w: b.x1 - b.x0, h: b.top })) };
    side = f.pose[2] > 0 ? 'up' : 'down'; // torso turned back (face up) or forward (face down)
  }
  world.rebuild();
  const ep = new sim.Episode(world, w, start.tilt || 0, start.drop || 0, Object.assign({ phase }, start));
  const stand = new ElmanNetwork(Float32Array.from(standW));
  const t0 = handover[side];
  const origDecide = ep.decide.bind(ep);
  ep.decide = function () {
    origDecide();
    stand.forward(this.input);
    if (this.frame / HZ >= t0) for (let j = 0; j < OUT; j++) this.network.action[j] = stand.action[j];
  };
  let maxHead = 0;
  while (!ep.advanceOneFrame()) maxHead = Math.max(maxHead, ep.measure.headHeight);
  const s = ep.stats;
  const frames = s.windowFrames || 0;
  const share = frames > 0 ? s.windowStood / frames : 0;
  const pass = frames > 0 && s.stoodUp && share >= config.getup.standShare;
  const score = (pass ? 1 : 0) + share + (s.stoodUp ? 0.5 : 0) + 0.3 * Math.min(1, maxHead / config.body.standingHeadMeters);
  if (process.env.DEBUG_RUN) console.log('  run', seed, start.startPose ? 'fall' : 'normal', 'side', side, 'pass', pass, 'score', score.toFixed(2));
  return { pass, score };
}

function evaluate(w, n) {
  let ok = 0;
  useFalls = false;
  for (let i = 0; i < n; i++) if (run(w, config.eval.seed + 9 * 10007 + i * 17, i % 6).pass) ok++;
  if (!falls) { useFalls = null; return ok; }
  // With falls: the official lying starts plus as many real course falls (fixed seeds).
  useFalls = true;
  let okF = 0;
  for (let i = 0; i < n; i++) if (run(w, 777000 + i * 13, i % 6).pass) okF++;
  useFalls = null;
  lastFalls = okF;
  return ok + okF;
}
let lastFalls = 0;

const rng = makeSeededRandom(31337);
const normal = () => randomNormalSample(rng);
const N = theta.length;
const m = new Float64Array(N);
const v = new Float64Array(N);
let best = evaluate(Float32Array.from(theta), evalN);
console.log('start eval getup ' + best + '/' + evalN * (falls ? 2 : 1) + (falls ? ' (course falls ' + lastFalls + '/' + evalN + ')' : ''));
let seedBase = 500000;
for (let g = 0; g < gens; g++) {
  const seeds = Array.from({ length: seedsPer }, () => seedBase++);
  const eps = [];
  const fit = [];
  const w = new Float32Array(N);
  let passes = 0;
  for (let p = 0; p < pairs; p++) {
    const e = new Float32Array(N);
    for (let i = 0; i < N; i++) e[i] = normal();
    eps.push(e);
    for (const sign of [1, -1]) {
      for (let i = 0; i < N; i++) w[i] = theta[i] + sign * sigma * e[i];
      let f = 0;
      for (let k = 0; k < seeds.length; k++) {
        const r = run(w, seeds[k], k % 6);
        f += r.score;
        if (r.pass) passes++;
      }
      fit.push(f / seeds.length);
    }
  }
  // Rank-based fitness shaping, centred.
  const order = fit.map((f, i) => [f, i]).sort((a, b) => a[0] - b[0]);
  const util = new Float64Array(fit.length);
  order.forEach(([, i], r) => (util[i] = r / (fit.length - 1) - 0.5));
  const grad = new Float64Array(N);
  for (let p = 0; p < pairs; p++) {
    const d = util[2 * p] - util[2 * p + 1];
    if (!d) continue;
    const e = eps[p];
    for (let i = 0; i < N; i++) grad[i] += d * e[i];
  }
  const step = g + 1;
  const c1 = 1 - Math.pow(0.9, step);
  const c2 = 1 - Math.pow(0.999, step);
  for (let i = 0; i < N; i++) {
    const gi = -grad[i] / (pairs * sigma); // ascent
    m[i] = 0.9 * m[i] + 0.1 * gi;
    v[i] = 0.999 * v[i] + 0.001 * gi * gi;
    theta[i] -= (lr * (m[i] / c1)) / (Math.sqrt(v[i] / c2) + 1e-8);
  }
  const mean = fit.reduce((a, b) => a + b, 0) / fit.length;
  let msg = 'gen ' + g + ' mean fitness ' + mean.toFixed(3) + ' training passes ' + passes + '/' + pairs * 2 * seedsPer;
  if ((g + 1) % evalEvery === 0) {
    const ok = evaluate(Float32Array.from(theta), evalN);
    msg += ' | EVAL getup ' + ok + '/' + evalN * (falls ? 2 : 1) + (falls ? ' (course falls ' + lastFalls + '/' + evalN + ')' : '');
    if (ok >= best) {
      best = ok;
      fs.writeFileSync(path.resolve(root, out), JSON.stringify(Object.assign({}, saved, { role: 'getup-expert-es', gen: g, eval: { getup: ok + '/' + evalN }, theta: Array.from(theta) })));
      msg += ' -> saved';
    }
  }
  console.log(msg);
}
sim.destroyWorld(world);
