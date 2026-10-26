// Searches an open-loop climb onto the tall box: a few key poses (14 joint commands + 2 grips) with
// durations, blended in time, then the stand-and-walk net balances on top. Same recipe that found
// the get-up reference (getup-keyframes.js): cross-entropy method over ~100 numbers. The result is
// a reference motion a climb expert network then learns (DAgger) and improves (evolution strategies).
//   node climb-keyframes.js --side 1 --height 0.5 --iters 120 --out keyframes-climb-r.json
//   --side 1 = box on the right (+x), -1 = box on the left.
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
const side = +arg('--side', 1);
const height = +arg('--height', 0.9);
const iters = +arg('--iters', 120);
const out = arg('--out', 'keyframes-climb-' + (side > 0 ? 'r' : 'l') + '.json');
const keys = +arg('--keys', 6);
const pop = +arg('--pop', 32);
const elite = +arg('--elite', 8);
const seeds = +arg('--seeds', 3);
const netFile = arg('--net', 'trained/stand-and-walk.json');
const init = arg('--init', null);
const sigmaStart = +arg('--sigma', 0.8);
const sigmaFloor = +arg('--sigma-floor', 0.15);
const settle = +arg('--settle', 3); // seconds of the stand net after the last key

const Box2D = await import(pathToFileURL(path.join(root, 'vendor/PhaserBox2D.min.js')).href);
const { createSim, feetOnBox } = await import('./src/sim.js');
const { ElmanNetwork, outputCountOf } = await import('./src/net.js');
const { makeSeededRandom, randomNormalSample } = await import('./src/utils.js');
const { isStanding } = await import('./src/reward.js');
const { evalMode } = await import('./src/training.js');

setSkill(11);
const sim = createSim(Box2D);
const world = sim.build();
const weights = ElmanNetwork.copy(JSON.parse(fs.readFileSync(path.join(root, netFile), 'utf8')).theta);
const J = config.net.jointCount;
const OUT = outputCountOf();
const G = OUT - J; // grip outputs
const P = config.partIndex;
const perKey = OUT + 1;
const dims = keys * perKey;
const rand = makeSeededRandom(side > 0 ? 33 : 44);
const normal = () => randomNormalSample(rand);
const standHead = config.body.standingHeadMeters;

function decode(v) {
  const frames = [];
  for (let k = 0; k < keys; k++) {
    const pose = [];
    for (let j = 0; j < OUT; j++) pose.push(Math.tanh(v[k * perKey + j]));
    const dur = 0.25 + 1.25 / (1 + Math.exp(-v[k * perKey + OUT]));
    frames.push({ pose, dur });
  }
  return frames;
}

// Joint commands blend into each key over the first 40% of it; grips switch at the key boundary.
export function commandAt(frames, t, outCmd) {
  let start = 0;
  let prev = null;
  for (const f of frames) {
    if (t < start + f.dur) {
      const blend = prev ? Math.min(1, (t - start) / (0.4 * f.dur)) : 1;
      for (let j = 0; j < J; j++) outCmd[j] = prev ? prev[j] + (f.pose[j] - prev[j]) * blend : f.pose[j];
      for (let j = J; j < f.pose.length; j++) outCmd[j] = f.pose[j];
      return true;
    }
    start += f.dur;
    prev = f.pose;
  }
  return false;
}

function makeStart(seed) {
  const r = makeSeededRandom(2000 + seed);
  const start = sim.fillTrainingStart({}, r, evalMode(11));
  start.climb.side = side;
  start.climb.height = height;
  return start;
}

function episode(frames, seed, record) {
  const start = makeStart(seed);
  const total = frames.reduce((s, f) => s + f.dur, 0);
  start.frames = Math.round((total + settle) * config.physics.hz);
  world.rebuild();
  const ep = new sim.Episode(world, weights, start.tilt || 0, start.drop || 0, Object.assign({ phase: 0 }, start));
  const cmd = new Float32Array(OUT);
  const setCommand = (e) => {
    if (commandAt(frames, e.frame / config.physics.hz, cmd)) for (let j = 0; j < OUT; j++) e.action[j] = cmd[j];
    else for (let j = J; j < OUT; j++) e.action[j] = -1; // let go once the stand net drives
  };
  // updateGrip runs before applyMotors each frame: set the command before both.
  const origGrip = ep.updateGrip.bind(ep);
  ep.updateGrip = function () {
    setCommand(this);
    origGrip();
  };
  const orig = ep.applyMotors.bind(ep);
  ep.applyMotors = function () {
    setCommand(this);
    orig();
  };
  const box = () => ep.climbBox;
  let score = 0;
  let n = 0;
  let bestFoot = 0; // highest the LOWER foot got, as a share of the box height (both feet up)
  let bestFeetOn = 0; // most feet on the box at once, with the head up
  let bestRise = 0; // highest head over the box top while both feet are on it
  let bestHip = 0;
  const poses = record ? [] : null;
  let f = 0;
  const last = start.frames - 60;
  while (!ep.advanceOneFrame()) {
    f++;
    const m = ep.measure;
    const b = box();
    const top = b ? b.top : height;
    const fl = world.py[P.footLeft];
    const fr = world.py[P.footRight];
    const low = Math.min(fl, fr);
    const high = Math.max(fl, fr);
    // Progress only counts with the head up: the first search lay on the floor with both feet
    // propped on the box (head 0.13 m), which paid the feet-on-top terms.
    const headUpNow = Math.max(0, Math.min(1, (m.headHeight - top) / standHead));
    if (m.headHeight > 0.9) {
      bestFoot = Math.max(bestFoot, Math.min(1, low / top) * 0.7 + Math.min(1, high / top) * 0.3);
      if (b) bestFeetOn = Math.max(bestFeetOn, feetOnBox(m, b));
    }
    // Rising on the box: the 0.3 m search got both feet up and then lay across the box (head 0.5 m).
    if (b && feetOnBox(m, b) === 2) bestRise = Math.max(bestRise, headUpNow);
    bestHip = Math.max(bestHip, Math.min(1, (world.py[P.torso] - top) / config.body.torsoHeightMeters) * (0.5 + 0.5 * headUpNow));
    if (record && f % 2 === 0) {
      const p = [];
      for (let i = 0; i < world.parts.length; i++) p.push(+world.px[i].toFixed(3), +world.py[i].toFixed(3), +Math.atan2(world.s[i], world.c[i]).toFixed(3));
      poses.push({ p, up: isStanding(m) ? 1 : 0 });
    }
    if (f < last) continue;
    const onTop = ep.footOnTop ? (ep.footOnTop() ? 1 : 0) : 0;
    const head = Math.max(0, Math.min(1, (m.headHeight - top) / standHead));
    // Only being on the box pays at the end: the second search stood still in front of it, which
    // the head-height term paid almost in full.
    const on = b ? feetOnBox(m, b) : 0;
    score += on * (0.5 + head) + onTop * (2 * head + Math.max(0, m.torsoCos)) + (onTop && isStanding(m) ? 3 : 0);
    n++;
  }
  const s = ep.stats;
  const frames2 = s.windowFrames || 0;
  const pass = !!s.climbOnTop && frames2 > 0 && s.windowStood / frames2 >= config.highbox.standShare && ep.measure.headHeight - (s.climbHeight || 0) >= standHead;
  const why = 'head ' + ep.measure.headHeight.toFixed(2) + ' top ' + (s.climbHeight || 0).toFixed(2) + ' stood ' + (frames2 ? (s.windowStood / frames2).toFixed(2) : 0) + ' torsoCos ' + ep.measure.torsoCos.toFixed(2);
  return { why, score: score / Math.max(1, n) + 1.5 * bestFoot + 1.0 * bestFeetOn + 0.5 * Math.max(0, bestHip) + 2 * bestRise, pass: pass ? 1 : 0, onTop: s.climbOnTop ? 1 : 0, poses };
}

function evaluate(v) {
  const frames = decode(v);
  let s = 0;
  let pass = 0;
  let top = 0;
  for (let i = 0; i < seeds; i++) {
    const r = episode(frames, i);
    s += r.score;
    pass += r.pass;
    top += r.onTop;
  }
  return { score: s / seeds + pass, pass, top };
}

// --show: replay the --init motion on --seeds starts, print why each passes or not, save a recording.
if (args.includes('--show')) {
  const v = JSON.parse(fs.readFileSync(init, 'utf8')).best;
  for (let i = 0; i < seeds; i++) {
    const r = episode(decode(v), i, i === 0);
    console.log('seed', i, 'score', r.score.toFixed(2), 'pass', r.pass, 'onTop', r.onTop, r.why);
    if (i === 0) fs.writeFileSync(out, JSON.stringify({ side, height, poses: r.poses }));
  }
  process.exit(0);
}
let mean = new Float64Array(dims);
if (init) {
  const saved = JSON.parse(fs.readFileSync(init, 'utf8')).best;
  for (let d = 0; d < dims && d < saved.length; d++) mean[d] = saved[d];
} else {
  for (let k = 0; k < keys; k++) for (let j = J; j < OUT; j++) mean[k * perKey + j] = -1; // grips start off
}
const sigma = new Float64Array(dims).fill(sigmaStart);
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
    sigma[d] = Math.max(sigmaFloor, 0.7 * s + 0.3 * sigma[d]);
  }
  if (it % 5 === 0 || it === iters - 1) {
    console.log('iter', it, 'height', height, 'best', best.score.toFixed(3), 'pass', best.pass + '/' + seeds, 'on top', best.top + '/' + seeds, 'gen best', cands[0].score.toFixed(3), 'sigma', (sigma.reduce((a, b) => a + b, 0) / dims).toFixed(3));
    fs.writeFileSync(out, JSON.stringify({ side, height, keys, best: Array.from(best.v), score: best.score, pass: best.pass, frames: decode(best.v) }));
  }
}
let fresh = 0;
for (let i = 0; i < 10; i++) fresh += episode(decode(best.v), 100 + i).pass;
const rec = episode(decode(best.v), 100, true);
fs.writeFileSync(out, JSON.stringify({ side, height, keys, best: Array.from(best.v), score: best.score, pass: best.pass, freshPass: fresh, frames: decode(best.v), poses: rec.poses }));
console.log('done side', side, 'height', height, 'best pass', best.pass + '/' + seeds, 'fresh seeds', fresh + '/10', '->', out);
sim.destroyWorld(world);
