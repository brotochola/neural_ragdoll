// Searches an open-loop get-up: a few key poses (joint commands) with durations, blended in time,
// followed by the trained stand net for balance. The point is to find a reference motion that
// works with the REAL body (strength 1) from lying, which the get-up net can then imitate.
// Cross-entropy method over ~60 numbers instead of ES over 12,000 weights.
//   node getup-keyframes.js --side up --iters 150 --out keyframes-up.json
//   --side up = lying face up (supine), down = face down (prone).
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
const side = arg('--side', 'up');
const iters = +arg('--iters', 150);
const out = arg('--out', 'keyframes-' + side + '.json');
const strength = +arg('--strength', 1);
const keys = +arg('--keys', 4);
const pop = +arg('--pop', 24);
const elite = +arg('--elite', 6);
const seeds = +arg('--seeds', 3);
const netFile = arg('--net', 'trained/stand-and-walk.json');

const Box2D = await import(pathToFileURL(path.join(root, 'vendor/PhaserBox2D.min.js')).href);
const { createSim } = await import('./src/sim.js');
const { ElmanNetwork } = await import('./src/net.js');
const { makeSeededRandom, randomNormalSample } = await import('./src/utils.js');
const { isStanding } = await import('./src/reward.js');

setSkill(9);
const sim = createSim(Box2D);
const world = sim.build();
const weights = ElmanNetwork.copy(JSON.parse(fs.readFileSync(path.join(root, netFile), 'utf8')).theta);
const J = config.net.jointCount;
const P = config.partIndex;
const perKey = J + 1; // 14 joint commands + a duration
const dims = keys * perKey;
const rand = makeSeededRandom(side === 'up' ? 11 : 22);
const normal = () => randomNormalSample(rand);

function decode(v) {
  const frames = [];
  for (let k = 0; k < keys; k++) {
    const pose = [];
    for (let j = 0; j < J; j++) pose.push(Math.tanh(v[k * perKey + j]));
    const dur = 0.25 + 1.25 / (1 + Math.exp(-v[k * perKey + J])); // 0.25..1.5 s
    frames.push({ pose, dur });
  }
  return frames;
}

// The command at time t: hold each key pose for its duration, blending into it over the first 40%.
function commandAt(frames, t, outCmd) {
  let start = 0;
  let prev = null;
  for (let k = 0; k < frames.length; k++) {
    const f = frames[k];
    if (t < start + f.dur) {
      const blend = prev ? Math.min(1, (t - start) / (0.4 * f.dur)) : 1;
      for (let j = 0; j < J; j++) outCmd[j] = prev ? prev[j] + (f.pose[j] - prev[j]) * blend : f.pose[j];
      return true;
    }
    start += f.dur;
    prev = f.pose;
  }
  return false;
}

function episode(frames, seed, record) {
  const r = makeSeededRandom(1000 + seed);
  const sign = side === 'up' ? 1 : -1;
  const tilt = sign * (config.start.downTiltMinRadians + r() * config.start.downTiltSpanRadians);
  world.rebuild();
  const options = {
    frames: 0, period: 8, phase: 0, failOnHead: true, getup: true, settle: config.start.downSettleFrames,
    strength, commands: 'none',
    targets: config.targets.map(() => ({ on: true, x: 0, y: 0, angle: 0, dx: 0, dy: 0, turn: 0 })),
  };
  const total = frames.reduce((s, f) => s + f.dur, 0);
  options.frames = Math.round((total + 4) * config.physics.hz);
  const ep = new sim.Episode(world, weights, tilt, config.start.downDropMeters, options);
  const cmd = new Float32Array(J);
  const orig = ep.applyMotors.bind(ep);
  ep.applyMotors = function () {
    if (commandAt(frames, this.frame / config.physics.hz, cmd)) for (let j = 0; j < J; j++) this.action[j] = cmd[j];
    orig();
  };
  let score = 0;
  let n = 0;
  let maxHead = 0;
  const poses = record ? [] : null;
  let f = 0;
  const last = options.frames - 60; // the last second
  while (!ep.advanceOneFrame()) {
    f++;
    const m = ep.measure;
    maxHead = Math.max(maxHead, m.headHeight);
    if (record && f % 2 === 0) {
      const p = [];
      for (let i = 0; i < world.parts.length; i++) p.push(+world.px[i].toFixed(3), +world.py[i].toFixed(3), +Math.atan2(world.s[i], world.c[i]).toFixed(3));
      poses.push({ p, up: isStanding(m) ? 1 : 0 });
    }
    if (f < last) continue;
    const head = Math.min(1, m.headHeight / config.body.standingHeadMeters);
    const feet = (m.touch[P.footLeft] ? 0.25 : 0) + (m.touch[P.footRight] ? 0.25 : 0);
    score += 2 * head + Math.max(0, m.torsoCos) + feet + (isStanding(m) ? 1 : 0);
    n++;
  }
  // A small credit for the highest head reached, so a near miss beats lying still.
  return { score: score / Math.max(1, n) + 0.3 * Math.min(1, maxHead / 1.6), stood: ep.stats.stoodUp ? 1 : 0, poses };
}

function evaluate(v) {
  const frames = decode(v);
  let s = 0;
  let stood = 0;
  for (let i = 0; i < seeds; i++) {
    const r = episode(frames, i);
    s += r.score;
    stood += r.stood;
  }
  return { score: s / seeds + stood, stood };
}

let mean = new Float64Array(dims);
let sigma = new Float64Array(dims).fill(0.8);
let best = null;
for (let it = 0; it < iters; it++) {
  const cands = [];
  for (let c = 0; c < pop; c++) {
    const v = new Float64Array(dims);
    for (let d = 0; d < dims; d++) v[d] = mean[d] + sigma[d] * normal();
    cands.push({ v, ...evaluate(v) });
  }
  cands.sort((a, b) => b.score - a.score);
  if (!best || cands[0].score > best.score) best = cands[0];
  const top = cands.slice(0, elite);
  for (let d = 0; d < dims; d++) {
    let m = 0;
    for (const c of top) m += c.v[d];
    m /= top.length;
    let s = 0;
    for (const c of top) s += (c.v[d] - m) ** 2;
    s = Math.sqrt(s / top.length);
    mean[d] = 0.7 * m + 0.3 * mean[d];
    sigma[d] = Math.max(0.05, 0.7 * s + 0.3 * sigma[d]);
  }
  if (it % 5 === 0 || it === iters - 1) {
    console.log('iter', it, 'best', best.score.toFixed(3), 'stood', best.stood + '/' + seeds, 'gen best', cands[0].score.toFixed(3), 'sigma', (sigma.reduce((a, b) => a + b, 0) / dims).toFixed(3));
    fs.writeFileSync(out, JSON.stringify({ side, strength, keys, best: Array.from(best.v), score: best.score, stood: best.stood, frames: decode(best.v) }));
  }
}
// Re-test the best on fresh seeds and record one for viewing.
let fresh = 0;
for (let i = 0; i < 10; i++) fresh += episode(decode(best.v), 100 + i).stood;
const rec = episode(decode(best.v), 100, true);
fs.writeFileSync(out, JSON.stringify({ side, strength, keys, best: Array.from(best.v), score: best.score, stood: best.stood, freshStood: fresh, frames: decode(best.v), poses: rec.poses }));
console.log('done', side, 'best stood', best.stood + '/' + seeds, 'fresh seeds', fresh + '/10', '->', out);
sim.destroyWorld(world);
