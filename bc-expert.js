// A separate get-up EXPERT network, trained by behaviour cloning with DAgger (like bc-train.js), that
// hands control to the stand-and-walk network once the get-up is done. Two networks, one selector:
//   lying at the start -> the expert drives for the get-up time of that side (face up / face down),
//   then the stand-and-walk net drives. The stand-and-walk net is never changed, so standing and
//   walking cannot get worse while the expert learns.
//   node bc-expert.js --iters 60 --out trained/getup-expert.json
// No external help: every motor command comes from one of the two networks.
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
const iters = +arg('--iters', 60);
const out = arg('--out', 'trained/getup-expert.json');
const studentFile = arg('--from', 'trained/stand-and-walk.json');
const standFile = arg('--stand', 'trained/stand-and-walk.json');
const getupEps = +arg('--getups', 16);
const keep = +arg('--keep', 400);
const epochs = +arg('--epochs', 2);
const batch = +arg('--batch', 8);
const lr = +arg('--lr', 0.001);
const betaIters = +arg('--beta-iters', 10);
const evalN = +arg('--eval-n', 20);

const Box2D = await import(pathToFileURL(path.join(root, 'vendor/PhaserBox2D.min.js')).href);
const { createSim } = await import('./src/sim.js');
const { ElmanNetwork, outputCountOf } = await import('./src/net.js');
const { makeSeededRandom } = await import('./src/utils.js');
const { evalMode } = await import('./src/training.js');
const { forwardSequence, backwardSequence } = await import('./src/bptt.js');

const OUT = outputCountOf();
const J = config.net.jointCount;
const HZ = config.physics.hz;
const sim = createSim(Box2D);
const world = sim.build();
const standW = Float32Array.from(JSON.parse(fs.readFileSync(path.join(root, standFile), 'utf8')).theta);
let student = Float64Array.from(JSON.parse(fs.readFileSync(path.join(root, studentFile), 'utf8')).theta);
const reference = {
  up: JSON.parse(fs.readFileSync(path.join(root, config.getup.referenceFiles.up), 'utf8')).frames,
  down: JSON.parse(fs.readFileSync(path.join(root, config.getup.referenceFiles.down), 'utf8')).frames,
};
const refLength = { up: reference.up.reduce((a, f) => a + f.dur, 0), down: reference.down.reduce((a, f) => a + f.dur, 0) };
const costs = Array.from({ length: OUT }, (_, j) => (j < J ? 1 : 0.2));

function keyCommand(frames, t, outCmd) {
  let start = 0;
  let prev = null;
  for (const f of frames) {
    if (t < start + f.dur) {
      const b = prev ? Math.min(1, (t - start) / (0.4 * f.dur)) : 1;
      for (let j = 0; j < J; j++) outCmd[j] = prev ? prev[j] + (f.pose[j] - prev[j]) * b : f.pose[j];
      return true;
    }
    start += f.dur;
    prev = f.pose;
  }
  return false;
}

// One get-up episode. drive: 'teacher' (keyframes, then the stand net) or 'student' (the expert,
// then the stand net). The stand net runs all along so its memory is warm at the handover.
// Returns the expert's inputs and the keyframe targets for the expert's part of the episode.
function episode(seed, drive, expertW, record) {
  setSkill(9);
  const rand = makeSeededRandom(seed);
  const start = sim.fillTrainingStart({}, rand, evalMode(9));
  world.rebuild();
  const ep = new sim.Episode(world, Float32Array.from(expertW), start.tilt || 0, start.drop || 0, Object.assign({ phase: seed % 6 }, start));
  const stand = new ElmanNetwork(Float32Array.from(standW));
  const side = start.tilt > 0 ? 'up' : 'down';
  const handover = refLength[side];
  const inputs = [];
  const targets = [];
  const cmd = new Float64Array(J);
  const origDecide = ep.decide.bind(ep);
  ep.decide = function () {
    origDecide(); // the expert's forward pass
    stand.forward(this.input);
    const t = this.frame / HZ;
    if (t < handover) {
      keyCommand(reference[side], t, cmd);
      inputs.push(Float32Array.from(this.input));
      const target = new Array(OUT).fill(null);
      for (let j = 0; j < J; j++) target[j] = cmd[j];
      target[J] = -1; // no grips while getting up
      target[J + 1] = -1;
      targets.push(target);
    } else {
      for (let j = 0; j < OUT; j++) this.network.action[j] = stand.action[j];
    }
  };
  const origMotors = ep.applyMotors.bind(ep);
  ep.applyMotors = function () {
    const t = this.frame / HZ;
    if (drive === 'teacher' && t < handover) {
      keyCommand(reference[side], t, cmd);
      for (let j = 0; j < J; j++) this.action[j] = cmd[j];
    }
    origMotors();
  };
  const poses = record ? [] : null;
  while (!ep.advanceOneFrame()) {
    if (record && ep.frame % 2 === 0) {
      const p = [];
      for (let i = 0; i < world.parts.length; i++) p.push(+world.px[i].toFixed(3), +world.py[i].toFixed(3), +Math.atan2(world.s[i], world.c[i]).toFixed(3));
      poses.push(p);
    }
  }
  return { inputs, targets, stats: ep.stats, poses };
}

function passes(stats) {
  const frames = stats.windowFrames || 0;
  return frames > 0 && stats.stoodUp && stats.windowStood / frames >= config.getup.standShare;
}

// The official get-up eval seeds (same as train-headless / bc-train).
function evaluate(n, w) {
  let ok = 0;
  for (let i = 0; i < n; i++) {
    const seedFix = config.eval.seed + 9 * 10007 + i * 17;
    if (passes(evalEpisode(seedFix, i, w).stats)) ok++;
  }
  return ok;
}
function evalEpisode(seed, i, w) {
  // Same start as the eval in bc-train.js, then the expert + handover.
  setSkill(9);
  const rand = makeSeededRandom(seed);
  const start = sim.fillTrainingStart({}, rand, evalMode(9));
  world.rebuild();
  const ep = new sim.Episode(world, Float32Array.from(w), start.tilt || 0, start.drop || 0, Object.assign({ phase: i % 6 }, start));
  const stand = new ElmanNetwork(Float32Array.from(standW));
  const handover = refLength[start.tilt > 0 ? 'up' : 'down'];
  const origDecide = ep.decide.bind(ep);
  ep.decide = function () {
    origDecide();
    stand.forward(this.input);
    if (this.frame / HZ >= handover) for (let j = 0; j < OUT; j++) this.network.action[j] = stand.action[j];
  };
  while (!ep.advanceOneFrame());
  return { stats: ep.stats };
}

const m = new Float64Array(student.length);
const v = new Float64Array(student.length);
let step = 0;
function adam(grad, scale) {
  step++;
  const c1 = 1 - Math.pow(0.9, step);
  const c2 = 1 - Math.pow(0.999, step);
  for (let i = 0; i < student.length; i++) {
    const g = grad[i] * scale;
    m[i] = 0.9 * m[i] + 0.1 * g;
    v[i] = 0.999 * v[i] + 0.001 * g * g;
    student[i] -= (lr * (m[i] / c1)) / (Math.sqrt(v[i] / c2) + 1e-8);
  }
}

if (args.includes('--eval-only')) {
  console.log('expert eval getup', evaluate(evalN, student) + '/' + evalN);
  process.exit(0);
}

const data = [];
let seedCounter = 1;
let best = -1;
const rng = makeSeededRandom(777);
for (let it = 0; it < iters; it++) {
  const beta = Math.max(0, 1 - it / Math.max(1, betaIters));
  let made = 0;
  let tried = 0;
  for (let i = 0; i < getupEps; i++) {
    const drive = rng() < beta ? 'teacher' : 'student';
    const e = episode(200000 + seedCounter++ * 7, drive, student);
    if (drive === 'student') {
      tried++;
      if (passes(e.stats)) made++;
    }
    data.push(e);
  }
  while (data.length > keep) data.shift();
  let lossSum = 0;
  let stepSum = 0;
  for (let epoch = 0; epoch < epochs; epoch++) {
    const order = data.map((_, i) => i);
    for (let i = order.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]];
    }
    for (let b = 0; b < order.length; b += batch) {
      const grad = new Float64Array(student.length);
      let steps = 0;
      for (let k = b; k < Math.min(order.length, b + batch); k++) {
        const e = data[order[k]];
        const cache = forwardSequence(student, e.inputs);
        lossSum += backwardSequence(student, cache, e.targets, costs, grad);
        steps += e.inputs.length;
      }
      stepSum += steps;
      adam(grad, 1 / Math.max(1, steps));
    }
  }
  const ok = evaluate(evalN, student);
  let msg = 'iter ' + it + ' beta ' + beta.toFixed(2) + ' loss/step ' + (lossSum / Math.max(1, stepSum)).toFixed(4)
    + ' student get-ups in training ' + made + '/' + tried + ' | EVAL getup ' + ok + '/' + evalN;
  if (ok > best || (ok === best && ok > 0)) {
    best = ok;
    fs.writeFileSync(path.resolve(root, out), JSON.stringify({ role: 'getup-expert', iter: it, inputCount: config.net.inputCount, handover: refLength, standNet: standFile, eval: { getup: ok + '/' + evalN }, theta: Array.from(student) }));
    msg += ' -> saved';
  }
  console.log(msg);
}
sim.destroyWorld(world);
