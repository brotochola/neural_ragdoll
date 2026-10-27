// ROPE SWING over a hole: running, the body jumps at the edge, catches a rope hanging over the hole
// in the air, swings across on it, lets go past the far edge, lands and runs on. The running and the
// jump are the parkour runner (src/runner.js with the hole-jump numbers); the rest is src/rope-swing.js,
// whose numbers (how the arms reach, the tuck, when to let go, the landing) are tuned here with the
// cross-entropy method. No external forces: motors and grips only.
//   node rope.js --iters 30 --out trained/rope.json
//   node rope.js --show --init trained/rope.json [--record out.json]
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
const iters = +arg('--iters', 30);
const out = arg('--out', 'trained/rope.json');
const init = arg('--init', null);
const pop = +arg('--pop', 20);
const elite = +arg('--elite', 5);
const seeds = +arg('--seeds', 4);
const seconds = +arg('--seconds', 9);
const gapMin = +arg('--gap-min', 1.4), gapMax = +arg('--gap-max', 2.2);
const speedMin = +arg('--speed-min', 2.2), speedMax = +arg('--speed-max', 2.8);
const LIFT = 0.6; // the floor is a raised block, so the hole is a real drop

const Box2D = await import(pathToFileURL(path.join(root, 'vendor/PhaserBox2D.min.js')).href);
const { createSim } = await import('./src/sim.js');
const { ElmanNetwork } = await import('./src/net.js');
const { makeSeededRandom, randomNormalSample } = await import('./src/utils.js');
const { evalMode } = await import('./src/training.js');
const pose = await import('./src/pose.js');
const { installRopeRunner, SPEC } = await import('./src/rope-swing.js');
const { makeRopeCourse } = await import('./src/terrain.js');

setSkill(6);
const sim = createSim(Box2D);
const world = sim.build();
const HZ = config.physics.hz;
const P = config.partIndex;
const weights = ElmanNetwork.copy(JSON.parse(fs.readFileSync(path.join(root, 'trained/stand-and-walk.json'), 'utf8')).theta);
const runStates = JSON.parse(fs.readFileSync(path.join(root, 'trained/run-states.json'), 'utf8'));
const runnerP = JSON.parse(fs.readFileSync(path.join(root, arg('--runner', 'trained/runner-gap-12c.json')), 'utf8')).params;

const decode = (v) => {
  const p = {};
  SPEC.forEach(([k, b], i) => (p[k] = b + (v[i] || 0)));
  for (const k of Object.keys(process.env)) if (k.startsWith('P_')) p[k.slice(2)] = +process.env[k];
  return p;
};

function episode(p, seed, record) {
  const r = makeSeededRandom(12000 + seed);
  const start = sim.fillTrainingStart({}, r, evalMode(6));
  const speed = speedMin + r() * (speedMax - speedMin);
  const course = makeRopeCourse(r, { gapMin, gapMax });
  const rs = runStates[Math.floor(r() * runStates.length)];
  const pz = rs.pose.slice();
  for (let k = 0; k < world.parts.length; k++) pz[k * 3 + 1] += LIFT;
  Object.assign(start, { startPose: pz, startVel: rs.vel.map((v, i) => (i % 3 === 0 ? v * speed / rs.speed : v)), tilt: 0, drop: 0, settle: 0, frames: Math.round(seconds * HZ) });
  start.walk = null;
  start.terrain = { statics: course.statics, ropes: course.ropes };
  world.rebuild();
  const ep = new sim.Episode(world, weights, 0, 0, Object.assign({ phase: 0 }, start));
  const x0 = (world.px[P.footLeft] + world.px[P.footRight]) / 2;
  ep.runner = { stance: rs.stance, t: -0.1, flight: false, jumping: false, jumpNext: false, strides: 0 };
  const hole = { x0: x0 + course.hole.x0, x1: x0 + course.hole.x1 };
  const RS = installRopeRunner(ep, world, p, { config, pose, Box2D, runnerP, speed: () => speed, hole, rope: ep.terrain.ropes[0], lift: LIFT, debug: !!process.env.DEBUG_ROPE });
  let fell = false, frames = 0, caught = false, landedX = null, upAfter = 0, maxX = -1e9;
  const recFrames = record ? [] : null;
  while (!ep.advanceOneFrame()) {
    frames++;
    const m = ep.measure;
    if (RS.phase === 'swing') caught = true;
    if (record && frames % 2 === 0) {
      const q = [];
      for (let k = 0; k < world.parts.length; k++) q.push(+world.px[k].toFixed(3), +world.py[k].toFixed(3), +Math.atan2(world.s[k], world.c[k]).toFixed(3));
      const rope = ep.terrain.ropes[0].ids.map((id) => { const v = Box2D.b2Body_GetPosition(id); return [+v.x.toFixed(3), +v.y.toFixed(3)]; });
      recFrames.push({ p: q, up: m.headHeight - LIFT > 1.3 && m.torsoCos > 0.7 ? 1 : 0, rope, gl: ep.grips[0].joint ? 1 : 0, gr: ep.grips[1].joint ? 1 : 0 });
    }
    const pelX = world.px[0];
    maxX = Math.max(maxX, pelX);
    // down in the hole, or the head on the floor
    const overFloor = pelX < hole.x0 || pelX > hole.x1;
    const feetLow = Math.min(world.py[P.footLeft], world.py[P.footRight]);
    if ((world.py[0] < LIFT - 0.1 || feetLow < LIFT - 0.25) && !(ep.grips[0].joint || ep.grips[1].joint)) { fell = true; break; }
    if (overFloor && m.headHeight - LIFT < 0.5) { fell = true; break; }
    if (RS.phase === 'land' || RS.phase === 'run2') {
      if (landedX == null && (m.touch[P.footLeft] || m.touch[P.footRight]) && pelX > hole.x1) landedX = pelX;
      if (landedX != null && m.headHeight - LIFT > 1.3 && m.torsoCos > 0.7) upAfter++;
    }
  }
  const T = frames / HZ;
  const after = landedX != null ? Math.max(0, maxX - hole.x1) : 0;
  const ran = Math.min(1, after / 3);
  const pass = !fell && landedX != null && after > 2.5 ? 1 : 0;
  const score = (caught ? 2 : Math.max(0, Math.min(1.5, (maxX - hole.x0) / 1))) + (landedX != null ? 2 : 0) + 3 * ran + (fell ? -1 : 1) + upAfter / HZ * 0.3 + (pass ? 3 : 0);
  const why = 'gap ' + (hole.x1 - hole.x0).toFixed(2) + ' speed ' + speed.toFixed(1) + (caught ? ' CAUGHT' : ' missed') + (landedX != null ? ' landed, ran ' + after.toFixed(1) + ' m' : '') + (fell ? ' FELL ' + T.toFixed(1) + 's' : '') + ' phase ' + RS.phase + (pass ? ' PASS' : '');
  const status = (pass ? 'swung across a ' : fell ? 'fell at a ' : 'stopped at a ') + (hole.x1 - hole.x0).toFixed(1) + ' m hole' + (caught ? ' (caught the rope)' : ' (missed the rope)');
  return { score, pass, why, status, frames: recFrames, blocks: course.statics.map((b) => ({ x0: +(x0 + b.dx).toFixed(3), x1: +(x0 + b.dx + b.w).toFixed(3), top: b.h })), rope: { x: ep.terrain.ropes[0].x, top: ep.terrain.ropes[0].top } };
}

function evaluate(v, base, n) {
  const p = decode(v);
  let s = 0, pass = 0;
  for (let i = 0; i < n; i++) { const e = episode(p, base + i); s += e.score; pass += e.pass; }
  return { score: s / n, pass };
}

if (args.includes('--show')) {
  const v = init ? JSON.parse(fs.readFileSync(init, 'utf8')).best : [];
  const p = decode(v);
  let pass = 0;
  for (let i = 0; i < seeds; i++) { const e = episode(p, 100 + i); pass += e.pass; console.log('seed', i, e.why); }
  console.log('pass', pass + '/' + seeds);
  const rec = arg('--record', null);
  if (rec) {
    const shapes = world.parts.map((q) => ({ hx: +q.hx.toFixed(4), hy: +q.hy.toFixed(4), r: q.r ? +q.r.toFixed(4) : 0, ax: q.ax || 0, ay: q.ay || 0, capsule: !!q.capsule, circle: !!q.circle }));
    const eps = [];
    for (let i = 0; i < +arg('--n', 3); i++) { const e = episode(p, 100 + i, true); eps.push({ seed: i, boxes: e.blocks, rope: e.rope, status: e.status, frames: e.frames }); }
    fs.writeFileSync(rec, JSON.stringify({ skill: 6, fps: HZ / 2, shapes, episodes: eps }));
  }
  process.exit(0);
}

const rand = makeSeededRandom(+arg('--rng', 99));
const mean = new Float64Array(SPEC.length);
if (init) { const sv = JSON.parse(fs.readFileSync(init, 'utf8')).best; for (let d = 0; d < SPEC.length && d < sv.length; d++) mean[d] = sv[d]; }
const sigma = Float64Array.from(SPEC.map((q) => q[2]));
let bestC = null;
for (let it = 0; it < iters; it++) {
  const cands = [];
  for (let c = 0; c < pop; c++) {
    const v = Float64Array.from(mean, (m, d) => m + sigma[d] * randomNormalSample(rand));
    cands.push({ v, ...evaluate(v, 1000 + it * seeds, seeds) });
  }
  cands.sort((a, b) => b.score - a.score);
  if (it % 2 === 1 || it === 0) {
    for (const c of [cands[0], { v: Float64Array.from(mean) }]) {
      const val = evaluate(c.v, 500, 8);
      if (!bestC || val.score > bestC.score) bestC = { v: Float64Array.from(c.v), ...val };
    }
    fs.writeFileSync(out, JSON.stringify({ gapMin, gapMax, speedMin, speedMax, best: Array.from(bestC.v), params: decode(bestC.v), score: bestC.score, pass: bestC.pass }));
    console.log('iter', it, 'held-out best', bestC.score.toFixed(3), 'pass', bestC.pass + '/8', 'gen best', cands[0].score.toFixed(3));
  }
  const top = cands.slice(0, elite);
  for (let d = 0; d < SPEC.length; d++) {
    const m = top.reduce((a, c) => a + c.v[d], 0) / elite;
    const sd = Math.sqrt(top.reduce((a, c) => a + (c.v[d] - m) ** 2, 0) / elite);
    mean[d] = 0.7 * m + 0.3 * mean[d];
    sigma[d] = Math.max(SPEC[d][2] * 0.15, 0.7 * sd + 0.3 * sigma[d]);
  }
}
console.log('done ->', out);
