// PARKOUR: tunes the running controller (src/runner.js) with the cross-entropy method.
//   --mode run   flat ground, commanded speed --speed-min..--speed-max, scored on speed, staying up, flight
//   --mode gap   a raised floor (0.6 m) with --gaps holes --gap-min..--gap-max wide; falling in fails
//   --mode box   boxes --box-h tall, --box-w wide on the floor, to jump over without slowing down
// Each episode starts already running (every part moving at 0.8 x the commanded speed), standing on
// the ground (or on the raised floor). Practice starts are new every generation; the kept version is
// the best on 12 held-out starts.
//   node parkour.js --mode run --speed-min 2 --speed-max 3 --iters 40 --out trained/run-runner.json
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
const mode = arg('--mode', 'run');
const iters = +arg('--iters', 40);
const out = arg('--out', 'trained/runner-' + mode + '.json');
const pop = +arg('--pop', 24);
const elite = +arg('--elite', 6);
const seeds = +arg('--seeds', 4);
const init = arg('--init', null);
const sigmaScale = +arg('--sigma', 1);
const speedMin = +arg('--speed-min', 2);
const speedMax = +arg('--speed-max', 3);
const seconds = +arg('--seconds', 7);
const gapN = +arg('--gaps', 3);
const gapMin = +arg('--gap-min', 0.5);
const gapMax = +arg('--gap-max', 1.0);
const floorH = +arg('--floor', 0.6);
const boxH = +arg('--box-h', 0.4);
const boxW = +arg('--box-w', 0.4);
const boxN = +arg('--boxes', 2);
const boxMax = +arg('--box-max', boxH);
// --run-states file: start from recorded running states (mid-run touchdowns, --dump-run-states) instead
// of the standing pose thrown forward (a third of the failures were in the first 2 s, before any
// obstacle). --dump-run-states out: record them from the --init runner on flat ground.
const runStatesFile = arg('--run-states', null);
const runStates = runStatesFile ? JSON.parse(fs.readFileSync(runStatesFile, 'utf8')) : null;
const dumpRunStates = arg('--dump-run-states', null);
const dumped = []; // box heights random in --box-h..--box-max

const Box2D = await import(pathToFileURL(path.join(root, 'vendor/PhaserBox2D.min.js')).href);
const { createSim } = await import('./src/sim.js');
const { ElmanNetwork } = await import('./src/net.js');
const { makeSeededRandom, randomNormalSample } = await import('./src/utils.js');
const { evalMode } = await import('./src/training.js');
const runnerMod = await import('./src/runner.js');

setSkill(6);
config.walk.maxStandSpeed = Math.max(config.walk.maxStandSpeed, speedMax * 2);
const sim = createSim(Box2D);
const world = sim.build();
const HZ = config.physics.hz;
const J = config.net.jointCount;
const OUT = J + config.net.gripOutputs;
const P = config.partIndex;
const weights = ElmanNetwork.copy(JSON.parse(fs.readFileSync(path.join(root, 'trained/stand-and-walk.json'), 'utf8')).theta);
const { SPEC, decode } = runnerMod;
const runner = runnerMod.makeController(world);
const dims = SPEC.length;

// The standing pose, read once: every episode starts from it lifted onto the floor and moving.
let basePose = null;
{
  const r = makeSeededRandom(1);
  const s = sim.fillTrainingStart({}, r, evalMode(6));
  s.tilt = 0;
  world.rebuild();
  const ep = new sim.Episode(world, weights, 0, 0, Object.assign({ phase: 0 }, s));
  for (let f = 0; f < 30; f++) ep.advanceOneFrame();
  basePose = [];
  const fx = (world.px[P.footLeft] + world.px[P.footRight]) / 2;
  for (let k = 0; k < world.parts.length; k++) basePose.push(world.px[k] - fx, world.py[k], Math.atan2(world.s[k], world.c[k]));
}

function makeCourse(r) {
  // statics relative to the feet; the body starts at x = 0
  const statics = [];
  const holes = [];
  const boxes = [];
  let lift = 0;
  if (mode === 'gap') {
    lift = floorH;
    let x = -3;
    let next = 4 + r() * 1.5; // first edge 4-5.5 m ahead: a few strides to settle
    for (let i = 0; i < gapN; i++) {
      statics.push({ dx: x, w: next - x, h: floorH });
      const w = gapMin + r() * (gapMax - gapMin);
      holes.push({ x0: next, x1: next + w });
      x = next + w;
      next = x + 3 + r() * 2;
    }
    statics.push({ dx: x, w: 30, h: floorH });
  } else if (mode === 'box') {
    let x = 4 + r() * 1.5;
    for (let i = 0; i < boxN; i++) {
      const h = boxH + r() * (boxMax - boxH);
      statics.push({ dx: x, w: boxW, h });
      boxes.push({ x0: x, x1: x + boxW, top: h });
      x += boxW + 4 + r() * 2;
    }
  }
  return { statics, holes, boxes, lift };
}

function episode(p, seed, record) {
  const r = makeSeededRandom(9000 + seed);
  const start = sim.fillTrainingStart({}, r, evalMode(6));
  const speed = speedMin + r() * (speedMax - speedMin);
  const course = makeCourse(r);
  let pose = basePose.slice();
  let vel = [];
  const v0 = 0.8 * speed;
  for (let k = 0; k < world.parts.length; k++) vel.push(v0, 0, 0);
  let rs = null;
  if (runStates) {
    rs = runStates[Math.floor(r() * runStates.length)];
    pose = rs.pose.slice();
    vel = rs.vel.map((v, i) => (i % 3 === 0 ? v * speed / rs.speed : v));
  }
  for (let k = 0; k < world.parts.length; k++) pose[k * 3 + 1] += course.lift;
  Object.assign(start, { startPose: pose, startVel: vel, tilt: 0, drop: 0, settle: 0, frames: Math.round(seconds * HZ) });
  start.walk = null;
  start.terrain = course.statics.length ? { statics: course.statics } : null;
  world.rebuild();
  const ep = new sim.Episode(world, weights, 0, 0, Object.assign({ phase: 0 }, start));
  const x0 = (world.px[P.footLeft] + world.px[P.footRight]) / 2;
  // t starts at -0.1 s: the body is placed 1 cm up and the stance foot's contact registers a few frames
  // later; read as a lift-off, the stance leg tucked and the body folded
  if (rs) ep.runner = { stance: rs.stance, t: -0.1, flight: false, jumping: false, jumpNext: false, strides: 0 };
  if (process.env.DEBUG_RS) console.log('after start: torso', world.px[0].toFixed(3), world.py[0].toFixed(3), Math.atan2(world.s[0], world.c[0]).toFixed(3), 'state', rs && rs.pose.slice(0, 3));
  // absolute obstacle positions
  const holes = course.holes.map((h) => ({ x0: x0 + h.x0, x1: x0 + h.x1 }));
  const boxes = course.boxes.map((b) => ({ x0: x0 + b.x0, x1: x0 + b.x1, top: b.top }));
  const obstacles = mode === 'gap' ? holes : boxes;
  const tmp = new Float64Array(OUT);
  ep.applyMotors = ((orig) => function () {
    const m = this.measure;
    m.walkSpeed = speed;
    this.speedCap = +(process.env.RUN_SPEED_CAP || config.body.maxJointSpeed);
    runner(p, this, tmp);
    const st = this.runner;
    // At each touchdown: jump on the stride that lands within jumpWin before the next obstacle's front
    // edge; before that (APPROACH, as long jumpers do) stretch or shorten the steps so a foot lands
    // `takeoff` before the edge: n steps left of the average length, each made (D - takeoff) / n long.
    if (st && st.t === 1 / HZ && obstacles.length) {
      const fx = world.px[(st.stance ? P.footRight : P.footLeft)];
      if (st.lastX != null) { const L = fx - st.lastX; if (L > 0.2 && L < 2.5) st.avgL = st.avgL ? 0.7 * st.avgL + 0.3 * L : L; }
      st.lastX = fx;
      const nextO = obstacles.find((o) => o.x1 > fx);
      st.placeAdj = 0;
      // jump now if an ordinary step (average length) would land within jumpMargin of the edge or beyond
      // it (a foot landing on the edge slid into the hole); before the average is known, jumpWin
      const win = st.avgL ? st.avgL + p.jumpMargin : p.jumpWin;
      if (nextO && nextO.x0 - fx > 0.03 && nextO.x0 - fx < win) st.jumping = true;
      else if (nextO && st.avgL && p.kStride) {
        const D = nextO.x0 - fx - p.takeoff;
        if (D > 0 && D < 5) {
          const n = Math.max(1, Math.round(D / st.avgL));
          st.placeAdj = Math.max(-0.35, Math.min(0.35, p.kStride * (D / n - st.avgL)));
        }
      }
    }
    for (let j = 0; j < OUT; j++) this.action[j] = tmp[j];
    orig();
  })(ep.applyMotors.bind(ep));
  let frames = 0, up = 0, flight = 0, fell = false, fellAt = null, cleared = 0;
  const top = course.lift;
  const recFrames = record ? [] : null;
  let xs = [];
  while (!ep.advanceOneFrame()) {
    frames++;
    const m = ep.measure;
    const head = m.headHeight - top;
    if (head > 1.3 && m.torsoCos > 0.7) up++;
    if (!m.touch[P.footLeft] && !m.touch[P.footRight]) flight++;
    if (frames % 6 === 0) xs.push(world.px[0]);
    if (record && frames % 2 === 0) {
      const pz = [];
      for (let k = 0; k < world.parts.length; k++) pz.push(+world.px[k].toFixed(3), +world.py[k].toFixed(3), +Math.atan2(world.s[k], world.c[k]).toFixed(3));
      recFrames.push({ p: pz, up: head > 1.3 && m.torsoCos > 0.7 ? 1 : 0 });
    }
    // fell: head low, or (gap) a foot down in a hole
    const inHole = mode === 'gap' && Math.min(world.py[P.footLeft], world.py[P.footRight]) < top - 0.25;
    if (head < 0.8 || inHole) { fell = true; fellAt = frames; break; }
    // dumping running states: every touchdown between 2 and 5 s
    if (dumpRunStates && ep.runner && ep.runner.t === 0 && frames > 2 * HZ && frames < 5 * HZ && ep.runner.strides > 2) {
      const fx = (world.px[P.footLeft] + world.px[P.footRight]) / 2;
      const ps = [], vs = [];
      for (let k = 0; k < world.parts.length; k++) {
        ps.push(+(world.px[k] - fx).toFixed(4), +(world.py[k] - top).toFixed(4), +Math.atan2(world.s[k], world.c[k]).toFixed(4));
        vs.push(+world.vx[k].toFixed(3), +world.vy[k].toFixed(3), +world.spin[k].toFixed(3));
      }
      dumped.push({ pose: ps, vel: vs, stance: ep.runner.stance, speed });
    }
  }
  const T = frames / HZ;
  const dist = world.px[0] - (x0 + basePose[0]);
  // obstacles passed: torso beyond the far edge while up
  for (const o of obstacles) if (world.px[0] > o.x1 + 0.3) cleared++;
  // speed over the last part of the run
  const tail = xs.slice(Math.floor(xs.length / 3));
  const vAvg = tail.length > 1 ? (tail[tail.length - 1] - tail[0]) / ((tail.length - 1) * 6 / HZ) : 0;
  const alive = frames / Math.round(seconds * HZ);
  const upShare = up / Math.max(1, frames);
  const speedErr = Math.abs(Math.min(2, vAvg / speed) - 1);
  const flightShare = flight / Math.max(1, frames);
  const passRun = !fell && upShare > 0.9 && speedErr < (obstacles.length ? 0.3 : 0.15); // over obstacles a jump costs some speed
  const pass = passRun && (obstacles.length === 0 || cleared === obstacles.length);
  let score = 4 * alive + 2 * upShare * alive + 2 * Math.min(1, Math.max(0, dist) / (speed * seconds)) - 2 * speedErr * alive + 1.5 * Math.min(1, flightShare / 0.25) * alive + (pass ? 3 : 0);
  if (obstacles.length) score += 3 * cleared / obstacles.length + (cleared === obstacles.length && !fell ? 2 : 0);
  const why = 'speed ' + speed.toFixed(2) + ' avg ' + vAvg.toFixed(2) + ' up ' + upShare.toFixed(2) + ' flight ' + flightShare.toFixed(2) + ' fell ' + (fell ? (fellAt / HZ).toFixed(1) + 's' : '-') + (obstacles.length ? ' cleared ' + cleared + '/' + obstacles.length + ' sizes ' + obstacles.map((o) => (o.top ? o.top : o.x1 - o.x0).toFixed(2)).join(',') : '') + ' strides ' + (ep.runner ? ep.runner.strides : 0);
  const blocks = ep.terrain ? ep.terrain.statics : null;
  const status = (fell ? 'fell at ' + (fellAt / HZ).toFixed(1) + ' s' : 'kept running') + (obstacles.length ? ' · cleared ' + cleared + '/' + obstacles.length : '') + ' · ' + vAvg.toFixed(1) + ' m/s';
  return { score, pass: pass ? 1 : 0, why, frames: recFrames, blocks, status };
}

function evaluate(v, seedBase, n) {
  const p = decode(v);
  let s = 0, pass = 0;
  for (let i = 0; i < n; i++) { const r = episode(p, seedBase + i); s += r.score; pass += r.pass; }
  return { score: s / n, pass };
}

// jumpWin is a selector parameter, searched with the rest
SPEC.push(['jumpWin', 0.9, 0.3]);
SPEC.push(['takeoff', 0.3, 0.15], ['kStride', 1.0, 0.5]);
SPEC.push(['jumpMargin', 0.2, 0.15]);
SPEC.push(['jumpTrail', 0, 0.4]);
SPEC.push(['landLean', 0, 0.15], ['landKnee', 0, 0.3], ['landPlace', 0, 0.25]);

if (dumpRunStates) {
  const v = JSON.parse(fs.readFileSync(init, 'utf8')).best;
  for (let i = 0; i < +arg('--n', 20); i++) episode(decode(v), 300 + i);
  fs.writeFileSync(dumpRunStates, JSON.stringify(dumped));
  console.log('running states', dumped.length, '->', dumpRunStates);
  process.exit(0);
}

if (args.includes('--show')) {
  const v = JSON.parse(fs.readFileSync(init, 'utf8')).best;
  const p = decode(v);
  for (let i = 0; i < seeds; i++) { const r = episode(p, 100 + i); console.log('seed', i, 'pass', r.pass, r.why); }
  const rec = arg('--record', null);
  if (rec) {
    const shapes = world.parts.map((q) => ({ hx: +q.hx.toFixed(4), hy: +q.hy.toFixed(4), r: q.r ? +q.r.toFixed(4) : 0, ax: q.ax || 0, ay: q.ay || 0, capsule: !!q.capsule, circle: !!q.circle }));
    const eps = [];
    for (let i = 0; i < +arg('--n', 3); i++) {
      const r = episode(p, 100 + i, true);
      eps.push({ seed: i, boxes: r.blocks ? r.blocks.map((b) => ({ x0: +b.x0.toFixed(3), x1: +b.x1.toFixed(3), top: +b.top.toFixed(3) })) : null, box: null, walk: null, board: null, finished: !!r.pass, status: r.status, frames: r.frames });
    }
    fs.writeFileSync(rec, JSON.stringify({ skill: 6, fps: HZ / 2, shapes, episodes: eps }));
  }
  process.exit(0);
}

const rand = makeSeededRandom(4242);
const normal = () => randomNormalSample(rand);
const mean = new Float64Array(SPEC.length);
if (init) { const saved = JSON.parse(fs.readFileSync(init, 'utf8')).best; for (let d = 0; d < SPEC.length && d < saved.length; d++) mean[d] = saved[d]; }
// --only prefixes: search only the parameters whose names start with one of them (e.g. jump,land);
// the others stay at the --init values (a full search drifted the running gait it depends on).
const only = arg('--only', null);
const searched = (name) => !only || only.split(',').some((q) => name.startsWith(q));
const sigma = Float64Array.from(SPEC.map((s) => (searched(s[0]) ? s[2] * sigmaScale : 0)));
const floor = SPEC.map((s) => s[2] * 0.15);
let best = null;
const save = (extra = {}) => fs.writeFileSync(out, JSON.stringify(Object.assign({ mode, speedMin, speedMax, gapMin, gapMax, floorH, boxH, boxW, best: Array.from(best.v), params: decode(best.v), score: best.score, pass: best.pass }, extra)));
for (let it = 0; it < iters; it++) {
  const cands = [];
  for (let c = 0; c < pop; c++) {
    const v = new Float64Array(SPEC.length);
    for (let d = 0; d < SPEC.length; d++) v[d] = mean[d] + sigma[d] * normal();
    cands.push({ v, ...evaluate(v, 1000 + it * seeds, seeds) });
  }
  cands.sort((a, b) => b.score - a.score);
  if (it % 5 === 4 || it === 0) {
    for (const c of [cands[0], { v: Float64Array.from(mean) }]) {
      const val = evaluate(c.v, 500, 12);
      if (!best || val.score > best.score) best = { v: Float64Array.from(c.v), ...val };
    }
    save();
  }
  const top = cands.slice(0, elite);
  for (let d = 0; d < SPEC.length; d++) {
    let m = 0;
    for (const c of top) m += c.v[d];
    m /= top.length;
    let s = 0;
    for (const c of top) s += (c.v[d] - m) ** 2;
    s = Math.sqrt(s / top.length);
    mean[d] = 0.7 * m + 0.3 * mean[d];
    sigma[d] = searched(SPEC[d][0]) ? Math.max(floor[d], 0.7 * s + 0.3 * sigma[d]) : 0;
  }
  if (it % 5 === 0 || it === iters - 1) console.log('iter', it, 'held-out best', best.score.toFixed(3), 'pass', best.pass + '/12', 'gen best', cands[0].score.toFixed(3));
}
let fresh = 0;
for (let i = 0; i < 12; i++) fresh += episode(decode(best.v), 100 + i).pass;
save({ freshPass: fresh });
console.log('done fresh pass', fresh + '/12', '->', out);
