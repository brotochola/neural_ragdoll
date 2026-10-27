// DASH: run right as far as possible in a set time over random boxes and balls, without falling.
// Tunes the running controller (src/runner.js) and its jump selector (src/run-selector.js, shared with the
// playground) with the cross-entropy method: every gait and jump number, the commanded speed vRun, and how
// the jump scales with the obstacle's height (jumpGainH, winH, takeoffH).
// Every episode is a new random course (src/terrain.js makeDashCourse) and starts mid-run from a recorded
// running state. Score: average speed until the end or a fall, plus a bonus for not falling. The kept
// version is the best on 16 held-out courses.
//   node dash.js --init trained/runner-box-02.json --iters 60 --out trained/dash.json
//   node dash.js --show --init trained/dash.json --seeds 40
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { config, setSkill } from './config.js';

const root = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const arg = (name, fallback) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : fallback; };
const iters = +arg('--iters', 60);
const out = arg('--out', 'trained/dash.json');
const init = arg('--init', null);
const pop = +arg('--pop', 20);
const elite = +arg('--elite', 5);
const seeds = +arg('--seeds', 4);
const held = +arg('--held', 16);
const seconds = +arg('--seconds', 12);
const sigmaScale = +arg('--sigma', 1);
const courseOpts = JSON.parse(arg('--course', '{}'));

const Box2D = await import(pathToFileURL(path.join(root, 'vendor/PhaserBox2D.min.js')).href);
const { createSim } = await import('./src/sim.js');
const { ElmanNetwork } = await import('./src/net.js');
const { makeSeededRandom, randomNormalSample } = await import('./src/utils.js');
const { evalMode } = await import('./src/training.js');
const { SPEC: RSPEC } = await import('./src/runner.js');
const { installRunner } = await import('./src/run-selector.js');
const { makeDashCourse } = await import('./src/terrain.js');

setSkill(6);
config.walk.maxStandSpeed = 20;
const sim = createSim(Box2D);
const world = sim.build();
const HZ = config.physics.hz;
const P = config.partIndex;
const weights = ElmanNetwork.copy(JSON.parse(fs.readFileSync(path.join(root, 'trained/stand-and-walk.json'), 'utf8')).theta);
const runStates = JSON.parse(fs.readFileSync(path.join(root, 'trained/run-states.json'), 'utf8'));

// the runner's SPEC, the selector numbers parkour.js appends, and the dash ones
export const SPEC = RSPEC.concat([
  ['jumpWin', 0.9, 0.3], ['takeoff', 0.3, 0.15], ['kStride', 1.0, 0.5], ['jumpMargin', 0.2, 0.15], ['jumpTrail', 0, 0.4],
  ['landLean', 0, 0.15], ['landKnee', 0, 0.3], ['landPlace', 0, 0.25],
  ['vRun', 2.7, 0.3], ['jumpGainH', 0.5, 0.4], ['winH', 0, 0.5], ['takeoffH', 0, 0.3], ['landGainH', 0, 0.5], ['dropPlace', 0, 0.8], ['dropLean', 0, 0.8],
]);
const decode = (v) => {
  const p = {};
  SPEC.forEach(([name, base], i) => (p[name] = base + (v[i] || 0)));
  p.tMin = Math.max(0.05, Math.min(0.4, p.tMin));
  p.vRun = Math.max(1.5, Math.min(5, p.vRun));
  return p;
};
// start from a saved parameter set by NAME (runner files differ in length and order of the extras)
function encode(params) { return Float64Array.from(SPEC.map(([name, base]) => (params[name] != null ? params[name] - base : 0))); }

function episode(p, seed, log) {
  const r = makeSeededRandom(7000 + seed);
  const start = sim.fillTrainingStart({}, r, evalMode(6));
  const course = makeDashCourse(r, Object.assign({ length: 4 + 5.2 * seconds }, courseOpts));
  const rs = runStates[Math.floor(r() * runStates.length)];
  const v = p.vRun;
  Object.assign(start, { startPose: rs.pose.slice(), startVel: rs.vel.map((q, i) => (i % 3 === 0 ? q * v / rs.speed : q)), tilt: 0, drop: 0, settle: 0, frames: Math.round(seconds * HZ) });
  start.walk = null;
  start.terrain = { statics: course.statics, balls: course.balls };
  world.rebuild();
  const ep = new sim.Episode(world, weights, 0, 0, Object.assign({ phase: 0 }, start));
  ep.runner = { stance: rs.stance, t: -0.1, flight: false, jumping: false, jumpNext: false, strides: 0 };
  const obstacles = ep.terrain.statics.map((b) => ({ x0: b.x0, x1: b.x1, top: b.top }))
    .concat(ep.terrain.balls.map((b) => ({ live: () => { const q = Box2D.b2Body_GetPosition(b.id); return { x0: q.x - b.r, x1: q.x + b.r, top: q.y + b.r }; } })));
  installRunner(ep, world, p, { config, obstacles });
  const x0 = world.px[0];
  let frames = 0, fell = false, low = 0;
  while (!ep.advanceOneFrame()) {
    frames++;
    const m = ep.measure;
    low = m.headHeight < 0.95 || (m.torsoCos || 0) < 0.3 ? low + 1 : 0;
    if (m.headHeight < 0.7 || low > 15) { fell = true; break; }
  }
  const T = frames / HZ, dist = world.px[0] - x0;
  const passed = obstacles.filter((o) => (o.live ? o.live() : o).x1 < world.px[0] - 0.3).length;
  // dense: average speed over the whole time, plus the share of the time it stayed up (a binary 'never fell'
  // bonus was rare on full courses and drowned the search in noise), plus a little for finishing
  const score = 2 * Math.max(0, dist) / seconds + 4 * Math.min(1, frames / (seconds * HZ)) + (fell ? 0 : 1);
  let where = '';
  if (fell) {
    // where it fell, against the obstacle nearest the hips: before its front edge (tripped), over it, or after it (landing)
    const now = obstacles.map((o) => (o.live ? Object.assign(o.live(), { ball: 1 }) : o));
    const px = world.px[0];
    let o = now.reduce((a, b) => (Math.abs((b.x0 + b.x1) / 2 - px) < Math.abs((a.x0 + a.x1) / 2 - px) ? b : a), now[0]);
    const rel = px < o.x0 ? 'before' : px <= o.x1 ? 'over' : 'after';
    where = ' ' + rel + ' ' + (o.ball ? 'ball' : 'box') + ' h ' + o.top.toFixed(2) + ' w ' + (o.x1 - o.x0).toFixed(2) + ' (' + (px - o.x0).toFixed(2) + ' m)';
    if (log) failLog.push(rel + ' ' + (o.ball ? 'ball' : 'box'));
  }
  if (log) console.log('seed', seed, (fell ? 'FELL ' + T.toFixed(1) + 's' + where : 'ok'), 'dist', dist.toFixed(1), 'm', 'avg', (dist / T).toFixed(2), 'm/s', 'passed', passed, 'of', course.statics.length + course.balls.length);
  return { score, pass: fell ? 0 : 1, dist, T, passed, fell };
}

function evaluate(v, base, n, log) {
  const p = decode(v);
  let s = 0, pass = 0, d = 0, ob = 0, falls = 0, dist = 0;
  for (let i = 0; i < n; i++) { const e = episode(p, base + i, log); s += e.score; pass += e.pass; d += e.dist / e.T; ob += e.passed; falls += e.fell ? 1 : 0; dist += e.dist; }
  // per obstacle: the share of obstacles reached that were got over (a fall is one not got over)
  return { score: s / n, pass, speed: d / n, perObstacle: ob / Math.max(1, ob + falls), dist: dist / n };
}

const failLog = [];
const initP = init ? JSON.parse(fs.readFileSync(init, 'utf8')).params : {};
if (args.includes('--show')) {
  const e = evaluate(encode(initP), +arg('--base', 100), seeds, true);
  const tally = {}; for (const f of failLog) tally[f] = (tally[f] || 0) + 1; console.log('falls by place', JSON.stringify(tally));
  console.log('pass', e.pass + '/' + seeds, 'mean speed', e.speed.toFixed(2), 'mean dist', e.dist.toFixed(1), 'per obstacle', (100 * e.perObstacle).toFixed(1) + '%', 'score', e.score.toFixed(3));
  process.exit(0);
}

const rand = makeSeededRandom(+arg('--rng', 4243));
const normal = () => randomNormalSample(rand);
const mean = encode(initP);
const sigma = Float64Array.from(SPEC.map((s) => s[2] * sigmaScale));
const floor = SPEC.map((s) => s[2] * 0.15);
let best = null;
const save = () => fs.writeFileSync(out, JSON.stringify({ mode: 'dash', seconds, course: courseOpts, best: Array.from(best.v), params: decode(best.v), score: best.score, pass: best.pass, held, speed: best.speed }));
const t0 = Date.now();
for (let it = 0; it < iters; it++) {
  const cands = [];
  for (let c = 0; c < pop; c++) {
    const v = new Float64Array(SPEC.length);
    for (let d = 0; d < SPEC.length; d++) v[d] = mean[d] + sigma[d] * normal();
    cands.push({ v, ...evaluate(v, 1000 + it * seeds, seeds) });
  }
  cands.sort((a, b) => b.score - a.score);
  if (it % 3 === 2 || it === 0) {
    for (const c of [cands[0], { v: Float64Array.from(mean) }]) {
      const val = evaluate(c.v, 500, held);
      if (!best || val.score > best.score) best = { v: Float64Array.from(c.v), ...val };
    }
    save();
    console.log('iter', it, 'held-out best', best.score.toFixed(3), 'pass', best.pass + '/' + held, 'speed', best.speed.toFixed(2), 'gen best', cands[0].score.toFixed(3), ((Date.now() - t0) / 60000).toFixed(1) + ' min');
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
    sigma[d] = Math.max(floor[d], 0.7 * s + 0.3 * sigma[d]);
  }
}
const fresh = evaluate(best.v, 100, 40);
fs.writeFileSync(out, JSON.stringify({ mode: 'dash', seconds, course: courseOpts, best: Array.from(best.v), params: decode(best.v), score: best.score, pass: best.pass, held, speed: best.speed, freshPass: fresh.pass, freshN: 40, freshSpeed: fresh.speed }));
console.log('done fresh pass', fresh.pass + '/40', 'speed', fresh.speed.toFixed(2), '->', out);
