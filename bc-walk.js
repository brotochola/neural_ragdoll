// A WALK EXPERT network that copies the SIMBICON-style controller (src/simbicon.js) by behaviour
// cloning with DAgger, like bc-expert.js does for the get-up. The controller alternates the legs by
// construction; the network learns to do the same from what it senses, then can be improved further.
// The selector uses it when there is a walk command, and the stand-and-walk net otherwise.
//   node bc-walk.js --iters 40 --out trained/walk-expert.json
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
const iters = +arg('--iters', 40);
const out = arg('--out', 'trained/walk-expert.json');
const studentFile = arg('--from', 'trained/stand-and-walk.json');
const standFile = arg('--stand', 'trained/stand-and-walk.json');
const teacherFile = arg('--teacher', 'trained/walk-simbicon.json');
const eps = +arg('--eps', 16);
const keep = +arg('--keep', 300);
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
const { decode, makeController } = await import('./src/simbicon.js');

const OUT = outputCountOf();
const J = config.net.jointCount;
const P = config.partIndex;
const sim = createSim(Box2D);
const world = sim.build();
const simbicon = makeController(world);
const teacherSaved = JSON.parse(fs.readFileSync(path.resolve(root, teacherFile), 'utf8'));
const params = Object.assign(decode(teacherSaved.best), { track: !!teacherSaved.track });
const standW = Float32Array.from(JSON.parse(fs.readFileSync(path.resolve(root, standFile), 'utf8')).theta);
let student = Float64Array.from(JSON.parse(fs.readFileSync(path.resolve(root, studentFile), 'utf8')).theta);
const costs = Array.from({ length: OUT }, (_, j) => (j < J ? 1 : 0.2));

// One walk episode. drive: 'teacher' or 'student'. The teacher (stand net during the start delay,
// then the controller) runs every frame on whatever state the episode is in, and labels the
// student's decision frames.
function episode(seed, drive, w, level = 6) {
  setSkill(level);
  const rand = makeSeededRandom(seed);
  const start = sim.fillTrainingStart({}, rand, evalMode(level));
  world.rebuild();
  const ep = new sim.Episode(world, Float32Array.from(w), start.tilt || 0, start.drop || 0, Object.assign({ phase: seed % 6 }, start));
  const stand = new ElmanNetwork(Float32Array.from(standW));
  const delay = Math.round(((start.walk && start.walk.delay) || 0) * config.physics.hz);
  const teacher = new Float64Array(OUT);
  const inputs = [];
  const targets = [];
  let decided = false;
  const origDecide = ep.decide.bind(ep);
  ep.decide = function () {
    origDecide();
    stand.forward(this.input);
    decided = true;
  };
  const studentAction = new Float64Array(OUT);
  const origMotors = ep.applyMotors.bind(ep);
  ep.applyMotors = function () {
    for (let j = 0; j < OUT; j++) studentAction[j] = this.action[j];
    if (this.frame < delay || !start.walk) for (let j = 0; j < OUT; j++) teacher[j] = stand.action[j];
    else simbicon(params, this, teacher);
    if (decided) {
      inputs.push(Float32Array.from(this.input));
      targets.push(Array.from(teacher));
      decided = false;
    }
    if (drive === 'teacher') for (let j = 0; j < OUT; j++) this.action[j] = teacher[j];
    origMotors();
    if (drive === 'teacher') for (let j = 0; j < OUT; j++) this.action[j] = studentAction[j];
  };
  // Steps per foot: lifts 3 cm or more off the floor, lands 4 cm or more ahead. By height, not
  // contact: a dragged foot's contact flickers while it slides, which counted as steps.
  const up = [false, false];
  const leftAt = [0, 0];
  const steps = [0, 0];
  const feet = [P.footLeft, P.footRight];
  const ground = config.skill.footGroundMeters;
  while (!ep.advanceOneFrame()) {
    for (let k = 0; k < 2; k++) {
      const h = world.py[feet[k]] - ground;
      const down = up[k] ? h < 0.015 : h < 0.03;
      const x = world.px[feet[k]];
      if (!down && !up[k]) {
        up[k] = true;
        leftAt[k] = x;
      } else if (down && up[k]) {
        up[k] = false;
        if (x - leftAt[k] > 0.04) steps[k]++;
      }
    }
  }
  return { inputs, targets, stats: ep.stats, steps };
}

function passes(stats) {
  const wf = stats.windowFrames || 0;
  return !stats.headHit && wf > 0 && stats.windowStood / wf >= config.walk.standShare && stats.walkCmd > 0 && stats.walkDist >= config.walk.passShare * stats.walkCmd;
}

// Official walk eval starts, the student alone. Pass, and alternation: the smaller step count over the larger.
function evaluate(n, w, drive = 'student') {
  let ok = 0;
  let alt = 0;
  for (let i = 0; i < n; i++) {
    const e = episode(config.eval.seed + 6 * 10007 + i * 17, drive, w);
    if (passes(e.stats)) ok++;
    const lo = Math.min(e.steps[0], e.steps[1]);
    const hi = Math.max(e.steps[0], e.steps[1]);
    alt += hi > 0 ? (lo / hi) * Math.min(1, lo / 3) : 0;
  }
  return { ok, alt: alt / n };
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

if (args.includes('--eval-teacher')) {
  const r = evaluate(evalN, student, 'teacher');
  console.log('teacher walk', r.ok + '/' + evalN, 'alternation', r.alt.toFixed(2));
  process.exit(0);
}
if (args.includes('--eval-only')) {
  const r = evaluate(evalN, student);
  console.log('walk', r.ok + '/' + evalN, 'alternation', r.alt.toFixed(2));
  process.exit(0);
}

{
  const t = evaluate(evalN, student);
  console.log('start: student walk ' + t.ok + '/' + evalN + ' alternation ' + t.alt.toFixed(2));
}
const data = [];
let seedCounter = 1;
let bestScore = -1;
const rng = makeSeededRandom(888);
for (let it = 0; it < iters; it++) {
  const beta = Math.max(0, 1 - it / Math.max(1, betaIters));
  let made = 0;
  let tried = 0;
  for (let i = 0; i < eps; i++) {
    const drive = rng() < beta ? 'teacher' : 'student';
    const e = episode(300000 + seedCounter++ * 7, drive, student);
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
  const r = evaluate(evalN, student);
  const score = r.ok + 10 * r.alt;
  let msg = 'iter ' + it + ' beta ' + beta.toFixed(2) + ' loss/step ' + (lossSum / Math.max(1, stepSum)).toFixed(4)
    + ' student walks in training ' + made + '/' + tried + ' | EVAL walk ' + r.ok + '/' + evalN + ' alternation ' + r.alt.toFixed(2);
  if (score > bestScore) {
    bestScore = score;
    fs.writeFileSync(path.resolve(root, out), JSON.stringify({ role: 'walk-expert', iter: it, inputCount: config.net.inputCount, teacher: teacherFile, eval: { walk: r.ok + '/' + evalN, alternation: +r.alt.toFixed(2) }, theta: Array.from(student) }));
    msg += ' -> saved';
  }
  console.log(msg);
}
sim.destroyWorld(world);
