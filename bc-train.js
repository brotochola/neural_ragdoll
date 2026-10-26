// Behaviour cloning with DAgger: the network learns to get up from a teacher, by gradient descent
// (backpropagation through time, src/bptt.js), and keeps standing and walking from the same teacher.
//   node bc-train.js --iters 60 --out trained/bc-student.json
// Teacher: during a get-up, the searched keyframe motion (trained/keyframes-up/down.json) for its
// duration, then the stand-and-walk net; in stand and walk episodes, the stand-and-walk net.
// DAgger: early episodes are driven by the teacher; later ones by the student, and the teacher
// labels what it would have done in the states the student actually reaches.
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
const out = arg('--out', 'trained/bc-student.json');
const studentFile = arg('--from', 'trained/stand-and-walk.json');
const teacherFile = arg('--teacher', 'trained/stand-and-walk.json');
const getupEps = +arg('--getups', 16);
const standEps = +arg('--stands', 4);
const walkEps = +arg('--walks', 4);
const keep = +arg('--keep', 400); // episodes kept in the aggregated dataset
const epochs = +arg('--epochs', 2);
const batch = +arg('--batch', 8);
const lr = +arg('--lr', 0.001);
const betaIters = +arg('--beta-iters', 10); // teacher-driven share goes 1 -> 0 over this many iterations
const evalN = +arg('--eval-n', 20);

const Box2D = await import(pathToFileURL(path.join(root, 'vendor/PhaserBox2D.min.js')).href);
const { createSim } = await import('./src/sim.js');
const { ElmanNetwork, outputCountOf } = await import('./src/net.js');
const { makeSeededRandom } = await import('./src/utils.js');
const { evalMode } = await import('./src/training.js');
const { forwardSequence, backwardSequence } = await import('./src/bptt.js');

const OUT = outputCountOf();
const J = config.net.jointCount;
const sim = createSim(Box2D);
const world = sim.build();
const teacherW = ElmanNetwork.copy(JSON.parse(fs.readFileSync(path.join(root, teacherFile), 'utf8')).theta);
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

// One episode. drive: 'teacher' or 'student'. Returns { inputs, targets, stats }.
function episode(level, seed, drive) {
  setSkill(level);
  const rand = makeSeededRandom(seed);
  const start = sim.fillTrainingStart({}, rand, evalMode(level));
  // Training starts get a little extra variety over the eval: a random shift and tilt jitter.
  world.rebuild();
  const studentW = Float32Array.from(student);
  const ep = new sim.Episode(world, studentW, start.tilt || 0, start.drop || 0, Object.assign({ phase: seed % 6 }, start));
  const teacher = new ElmanNetwork(Float32Array.from(teacherW));
  const side = start.getup ? (start.tilt > 0 ? 'up' : 'down') : null;
  const inputs = [];
  const targets = [];
  const target = new Float64Array(OUT);
  const cmd = new Float64Array(J);
  const origDecide = ep.decide.bind(ep);
  ep.decide = function () {
    origDecide();
    teacher.forward(this.input);
    for (let j = 0; j < OUT; j++) target[j] = teacher.action[j];
    if (side && keyCommand(reference[side], this.frame / config.physics.hz, cmd)) for (let j = 0; j < J; j++) target[j] = cmd[j];
    inputs.push(Float32Array.from(this.input));
    targets.push(Array.from(target));
  };
  const origMotors = ep.applyMotors.bind(ep);
  ep.applyMotors = function () {
    if (drive === 'teacher') {
      // The teacher acts every frame: the keyframes blend continuously, the net holds its action.
      for (let j = 0; j < OUT; j++) this.action[j] = target[j];
      if (side && keyCommand(reference[side], this.frame / config.physics.hz, cmd)) for (let j = 0; j < J; j++) this.action[j] = cmd[j];
    }
    origMotors();
  };
  while (!ep.advanceOneFrame());
  return { inputs, targets, stats: ep.stats, level };
}

function passes(level, stats) {
  const frames = stats.windowFrames || 0;
  if (!(frames > 0)) return false;
  if (level === 9) return stats.stoodUp && stats.windowStood / frames >= config.getup.standShare;
  if (level === 1) return stats.stoodUp && stats.windowStood / frames >= config.eval.standWindowShare;
  if (level === 6) return !stats.headHit && stats.windowStood / frames >= config.walk.standShare && stats.walkCmd > 0 && stats.walkDist >= config.walk.passShare * stats.walkCmd;
  return false;
}

function evaluate(level, n) {
  let ok = 0;
  for (let i = 0; i < n; i++) {
    setSkill(level);
    const rand = makeSeededRandom(config.eval.seed + level * 10007 + i * 17);
    const start = sim.fillTrainingStart({}, rand, evalMode(level));
    world.rebuild();
    const ep = new sim.Episode(world, Float32Array.from(student), start.tilt || 0, start.drop || 0, Object.assign({ phase: i % 6 }, start));
    while (!ep.advanceOneFrame());
    if (passes(level, ep.stats)) ok++;
  }
  return ok;
}

// Adam
const m = new Float64Array(student.length);
const v = new Float64Array(student.length);
let step = 0;
function adam(grad, scale) {
  step++;
  const b1 = 0.9;
  const b2 = 0.999;
  const c1 = 1 - Math.pow(b1, step);
  const c2 = 1 - Math.pow(b2, step);
  for (let i = 0; i < student.length; i++) {
    const g = grad[i] * scale;
    m[i] = b1 * m[i] + (1 - b1) * g;
    v[i] = b2 * v[i] + (1 - b2) * g * g;
    student[i] -= (lr * (m[i] / c1)) / (Math.sqrt(v[i] / c2) + 1e-8);
  }
}

const data = [];
let seedCounter = 1;
let best = -1;
const rng = makeSeededRandom(4242);
for (let it = 0; it < iters; it++) {
  const beta = Math.max(0, 1 - it / Math.max(1, betaIters));
  const plan = [];
  for (let i = 0; i < getupEps; i++) plan.push(9);
  for (let i = 0; i < standEps; i++) plan.push(1);
  for (let i = 0; i < walkEps; i++) plan.push(6);
  let made = 0;
  let tried = 0;
  for (const level of plan) {
    const drive = rng() < beta ? 'teacher' : 'student';
    const e = episode(level, 100000 + seedCounter++ * 7, drive);
    if (level === 9 && drive === 'student') {
      tried++;
      if (passes(9, e.stats)) made++;
    }
    data.push(e);
  }
  while (data.length > keep) data.shift();
  // Train on the aggregated dataset.
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
  const getupOk = evaluate(9, evalN);
  const msg = 'iter ' + it + ' beta ' + beta.toFixed(2) + ' loss/step ' + (lossSum / Math.max(1, stepSum)).toFixed(4)
    + ' student get-ups in training ' + made + '/' + tried + ' | EVAL getup ' + getupOk + '/' + evalN;
  if (getupOk >= best) {
    best = getupOk;
    const standOk = evaluate(1, 10);
    const walkOk = evaluate(6, 10);
    fs.writeFileSync(path.resolve(root, out), JSON.stringify({ role: 'bc-student', iter: it, inputCount: config.net.inputCount, eval: { getup: getupOk + '/' + evalN, stand: standOk + '/10', walk: walkOk + '/10' }, theta: Array.from(student) }));
    console.log(msg + ' stand ' + standOk + '/10 walk ' + walkOk + '/10 -> saved');
  } else {
    console.log(msg);
  }
}
sim.destroyWorld(world);
