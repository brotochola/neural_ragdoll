// Searches the parameters of the step-up controller (src/stepup.js) for climbing onto a box, with
// the cross-entropy method, then the stand-and-walk net balances on top. The open-loop key-pose
// search (climb-keyframes.js) kept finding junk (sitting down, legs over the box); a structured
// controller with feedback only has ~26 meaningful numbers to tune.
//   node climb-stepup.js --side 1 --height 0.4 --iters 60 --out trained/stepup-r.json
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
const out = arg('--out', 'trained/stepup-' + (side > 0 ? 'r' : 'l') + '.json');
const keys = +arg('--keys', 6);
const pop = +arg('--pop', 24);
const elite = +arg('--elite', 6);
const seeds = +arg('--seeds', 3);
const netFile = arg('--net', 'trained/stand-and-walk.json');
const init = arg('--init', null);
const sigmaStart = +arg('--sigma', 0.8);
const sigmaFloor = +arg('--sigma-floor', 0.15); // share of each parameter's starting spread
const settle = +arg('--settle', 3); // seconds of the stand net after the controller is done
const maxSeconds = +arg('--seconds', 6); // controller time limit

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
const { SPEC, decode: decodeBase, makeController } = await import('./src/stepup.js');
const stairsN = +arg('--stairs', 0); // a flight of N steps instead of one box
const rise = +arg('--rise', 0.18);
const run = +arg('--run', 0.3);
const stairPause = +arg('--pause', 1.0); // seconds of the stand net between steps
const hands = args.includes('--hands');
// --robust: varied starting stance (feet 0.12-0.8 m apart) and distance to the edge (0.2-0.55 m),
// as after walking up to a box on the course.
const robust = args.includes('--robust');
// --states file: start from recorded arrivals (course.js --dump-states): body positions relative to the
// box edge, with velocities. Half the starts, the other half --robust random ones if given.
const statesFile = arg('--states', null);
const arrivals = statesFile ? JSON.parse(fs.readFileSync(statesFile, 'utf8')) : null; // reach, grab the top edge, pull and mantle (tall boxes)
const decode = (v) => Object.assign(decodeBase(v), { hands });
const controller = makeController(world);
const dims = SPEC.length;
const rand = makeSeededRandom(side > 0 ? 77 : 88);
const normal = () => randomNormalSample(rand);
const standHead = config.body.standingHeadMeters;

function makeStart(seed) {
  const r = makeSeededRandom(2000 + seed);
  const start = sim.fillTrainingStart({}, r, evalMode(11));
  start.climb.side = side;
  start.climb.height = height;
  if (robust) {
    start.stance = 0.06 + r() * 0.34;
    start.climb.near = 0.2 + r() * 0.35;
  }
  if (arrivals && (!robust || r() < 0.5)) {
    const a = arrivals[Math.floor(r() * arrivals.length)];
    const P = config.partIndex;
    const mid = (a.pose[P.footLeft * 3] + a.pose[P.footRight * 3]) / 2;
    start.startPose = a.pose;
    start.startVel = a.vel;
    start.climb.near = -mid; // the box edge sits where it was when the course handed over
    start.tilt = 0;
    start.drop = 0;
    start.settle = 0;
  }
  if (stairsN > 0) {
    start.climb = null;
    start.stairs = { n: stairsN, rise, run, near: 0.25 + r() * 0.2, landing: 1 };
  }
  return start;
}

function episode(p, seed, record) {
  const start = makeStart(seed);
  start.frames = Math.round((maxSeconds + settle) * config.physics.hz);
  world.rebuild();
  const ep = new sim.Episode(world, weights, start.tilt || 0, start.drop || 0, Object.assign({ phase: 0 }, start));
  let done = false;
  let step = 0; // stairs: which step the controller is climbing
  let pause = 0; // stairs: frames of the stand net between steps, to settle on the tread
  const setCommand = (e) => {
    if (pause > 0) {
      pause--;
      for (let j = J; j < OUT; j++) e.action[j] = -1;
      if (pause === 0) {
        step++;
        e.climbBox = e.stairs[step];
        e.measure.climbBox = e.climbBox;
        e.stepup = null;
      }
      return;
    }
    if (!done && e.frame / config.physics.hz < maxSeconds) {
      done = !controller(p, e, e.action);
      // Stairs: one step done; the stand net steadies it for stairPause s, then the next step-up.
      if (done && e.stairs && step < e.stairs.length - 1) {
        done = false;
        pause = Math.round(stairPause * config.physics.hz);
        for (let j = J; j < OUT; j++) e.action[j] = -1;
        return;
      }
    } else done = true;
    e.controllerState = done ? 5 : (e.stepup ? e.stepup.state : 0);
    if (done) for (let j = J; j < OUT; j++) e.action[j] = -1; // let go once the stand net drives
  };
  // updateGrip runs before applyMotors each frame: set the command before both.
  const origGrip = ep.updateGrip.bind(ep);
  ep.updateGrip = function () {
    setCommand(this);
    origGrip();
  };
  const box = () => ep.climbBox;
  let score = 0;
  let n = 0;
  let bestFoot = 0; // highest the LOWER foot got, as a share of the box height (both feet up)
  let bestFeetOn = 0; // most feet on the box at once, with the head up
  let bestRise = 0; // highest head over the box top while both feet are on it
  let tipped = 0; // frames with the torso tipped past ~80 degrees: the hands search flipped head-first onto the box
  let allFrames = 0;
  let inverted = 0; // the torso went past horizontal at some point (a dive or flip)
  let stepsUp = 0; // stairs: highest step with both feet on it and the head up
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
    allFrames++;
    if (ep.stairs && m.headHeight > 1.0) for (let k = stepsUp; k < ep.stairs.length; k++) if (feetOnBox(m, ep.stairs[k]) === 2) stepsUp = k + 1;
    if (m.torsoCos < 0.2) tipped++;
    if (m.torsoCos < 0) inverted = 1;
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
    // Feet on the box count only with the head well above it (it dived and hung head-down).
    const on = b && m.headHeight > top + 0.5 ? feetOnBox(m, b) : 0;
    score += on * (0.5 + head) + onTop * (2 * head + Math.max(0, m.torsoCos)) + (onTop && isStanding(m) ? 3 : 0);
    n++;
  }
  const s = ep.stats;
  const frames2 = s.windowFrames || 0;
  const pass = (!ep.stairs || stepsUp === ep.stairs.length) && !!s.climbOnTop && frames2 > 0 && s.windowStood / frames2 >= config.highbox.standShare && ep.measure.headHeight - (s.climbHeight || 0) >= standHead;
  const why = (ep.stairs ? 'steps ' + stepsUp + '/' + ep.stairs.length + ' ' : '') + 'head ' + ep.measure.headHeight.toFixed(2) + ' top ' + (s.climbHeight || 0).toFixed(2) + ' stood ' + (frames2 ? (s.windowStood / frames2).toFixed(2) : 0) + ' torsoCos ' + ep.measure.torsoCos.toFixed(2);
  const boxInfo = ep.climbBox ? { x0: ep.climbBox.x0, x1: ep.climbBox.x1, top: ep.climbBox.top } : null;
  return { boxInfo, why, score: score / Math.max(1, n) + 1.5 * bestFoot + 1.0 * bestFeetOn + 0.5 * Math.max(0, bestHip) + 2 * bestRise - 4 * tipped / Math.max(1, allFrames) + 2 * stepsUp - 6 * inverted, pass: pass ? 1 : 0, onTop: s.climbOnTop ? 1 : 0, poses };
}

// seedBase: each generation draws new practice starts. With 3 fixed ones the 0.3 m search passed
// all 3 and only 4 of 10 new starts.
function evaluate(v, seedBase = 0, n = seeds) {
  const p = decode(v);
  let s = 0;
  let pass = 0;
  let top = 0;
  for (let i = 0; i < n; i++) {
    const r = episode(p, seedBase + i);
    s += r.score;
    pass += r.pass;
    top += r.onTop;
  }
  return { score: s / n + pass / n * 3, pass, top, n };
}

// --show: replay the --init motion on --seeds starts, print why each passes or not, save a recording.
if (args.includes('--show')) {
  const v = JSON.parse(fs.readFileSync(init, 'utf8')).best;
  for (let i = 0; i < seeds; i++) {
    const r = episode(decode(v), i, i === 0);
    console.log('seed', i, 'score', r.score.toFixed(2), 'pass', r.pass, 'onTop', r.onTop, r.why);
    if (i === 0) fs.writeFileSync(out, JSON.stringify({ side, height, box: r.boxInfo, poses: r.poses }));
  }
  process.exit(0);
}
let mean = new Float64Array(dims);
if (init) {
  const saved = JSON.parse(fs.readFileSync(init, 'utf8')).best;
  for (let d = 0; d < dims && d < saved.length; d++) mean[d] = saved[d];
}
const sigma = init ? new Float64Array(dims).fill(sigmaStart) : Float64Array.from(SPEC.map((q) => q[2]));
let best = null;
for (let it = 0; it < iters; it++) {
  const cands = [];
  for (let c = 0; c < pop; c++) {
    const v = new Float64Array(dims);
    for (let d = 0; d < dims; d++) v[d] = mean[d] + sigma[d] * normal();
    cands.push({ v, ...evaluate(v, 1000 + it * seeds) });
  }
  cands.sort((a, b) => b.score - a.score);
  // Keep the version that does best on 10 held-out starts (500+), checked every 5 generations.
  if (it % 5 === 4 || it === 0) {
    for (const c of [cands[0], { v: Float64Array.from(mean) }]) {
      const val = evaluate(c.v, 500, 10);
      if (!best || val.score > best.score) best = { v: Float64Array.from(c.v), ...val };
    }
  }
  const top = cands.slice(0, elite);
  for (let d = 0; d < dims; d++) {
    let m = 0;
    for (const c of top) m += c.v[d];
    m /= top.length;
    let s = 0;
    for (const c of top) s += (c.v[d] - m) ** 2;
    s = Math.sqrt(s / top.length);
    mean[d] = 0.7 * m + 0.3 * mean[d];
    sigma[d] = Math.max(sigmaFloor * SPEC[d][2], 0.7 * s + 0.3 * sigma[d]);
  }
  if (it % 5 === 0 || it === iters - 1) {
    console.log('iter', it, 'height', height, 'best', best.score.toFixed(3), 'held-out pass', best.pass + '/10', 'on top', best.top + '/10', 'gen best', cands[0].score.toFixed(3), 'sigma', (sigma.reduce((a, b) => a + b, 0) / dims).toFixed(3));
    fs.writeFileSync(out, JSON.stringify({ side, height, controller: 'stepup', hands, best: Array.from(best.v), score: best.score, pass: best.pass, frames: decode(best.v) }));
  }
}
let fresh = 0;
for (let i = 0; i < 10; i++) fresh += episode(decode(best.v), 100 + i).pass;
const rec = episode(decode(best.v), 100, true);
fs.writeFileSync(out, JSON.stringify({ side, height, controller: 'stepup', hands, best: Array.from(best.v), score: best.score, pass: best.pass, freshPass: fresh, frames: decode(best.v), poses: rec.poses }));
console.log('done side', side, 'height', height, 'best pass', best.pass + '/' + seeds, 'fresh seeds', fresh + '/10', '->', out);
sim.destroyWorld(world);
