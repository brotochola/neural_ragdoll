// POP: runs the Prince of Persia level (src/terrain.js makePopLevel, src/director.js) headless with the
// playground's own numbers (play/params.json, built by play/make-params.mjs) and counts how far each attempt
// gets: the first stretch, the leap and grab at the rock, over its top, the rope's hole, the end.
//   node pop.js --seeds 10 [--base 100] [--seconds 150]
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { config, setSkill } from './config.js';

const root = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const arg = (n, f) => { const i = args.indexOf(n); return i >= 0 ? args[i + 1] : f; };
const seeds = +arg('--seeds', 10), base = +arg('--base', 100), seconds = +arg('--seconds', 300);
const Box2D = await import(pathToFileURL(path.join(root, 'vendor/PhaserBox2D.min.js')).href);
const { createSim } = await import('./src/sim.js');
const { ElmanNetwork } = await import('./src/net.js');
const { makeSeededRandom } = await import('./src/utils.js');
const { evalMode } = await import('./src/training.js');
const pose = await import('./src/pose.js');
const { makePopLevel } = await import('./src/terrain.js');
const { installPopDirector } = await import('./src/director.js');
const { makeTrailNet } = await import('./src/trail-net.js');
setSkill(6);
config.walk.maxStandSpeed = 20;
const sim = createSim(Box2D);
const world = sim.build();
const HZ = config.physics.hz, P = config.partIndex;
const params = JSON.parse(fs.readFileSync(path.join(root, 'play/params.json'), 'utf8'));
if (arg('--trail-net', null)) { const j = JSON.parse(fs.readFileSync(arg('--trail-net'), 'utf8')); params.trailNet = j.theta; params.trail = JSON.parse(fs.readFileSync(path.join(root, j.runner), 'utf8')).params; }
const weights = ElmanNetwork.copy(JSON.parse(fs.readFileSync(path.join(root, 'trained/stand-and-walk.json'), 'utf8')).theta);
const STAGES = ['start', 'reached the rock', 'grabbed the rock', 'over the rock', 'on to the rope', 'across the rope', 'end'];

// --at-rope: only the level's second half: the body starts running on the rock's top 3 m before trail B
// (the rock and the first stretch gone), for measuring trail B, the rope and trail C quickly
const atRope = args.includes('--at-rope');
function ropeOnly(L) {
  const Z = L.zones, d = -(Z.wall.plateauEnd - 3), H = Z.wall.H;
  const keep = (x) => x + d > -6;
  const statics = L.statics.filter((b) => keep(b.dx + (b.w || 0))).map((b) => Object.assign({}, b, { dx: +(b.dx + d).toFixed(3) }));
  statics.unshift({ dx: -6, w: 9, h: H });
  return Object.assign({}, L, {
    statics, balls: L.balls.map((b) => Object.assign({}, b, { dx: b.dx + d })), polys: L.polys.map((q) => q.map(([x, y]) => [x - 300, y])), surface: L.surface.map(([x, y]) => [x - 300, y]), crag: L.crag,
    features: L.features.filter((f) => keep(f.x1)).map((f) => Object.assign({}, f, { x0: f.x0 + d, x1: f.x1 + d })),
    levels: [[-6, 3, H]].concat(L.levels.filter((l) => l[0] + d > 2.9).map((l) => [l[0] + d, l[1] + d, l[2]])),
    ropes: L.ropes.map((q) => Object.assign({}, q, { dx: q.dx + d })), lift: H,
    zones: { wall: Object.assign({}, Z.wall, { x0: -100, summit: -99, plateauEnd: 3 }), rope: Object.assign({}, Z.rope, { x0: Z.rope.x0 + d, x1: Z.rope.x1 + d }), down: -100 },
  });
}
function episode(seed) {
  const r = makeSeededRandom(seed);
  const start = sim.fillTrainingStart({}, r, evalMode(6));
  let L = makePopLevel(r);
  if (atRope) L = ropeOnly(L);
  const rs = params.runStates[Math.floor(r() * params.runStates.length)];
  const v0 = params.trail.vRun || 2.6;
  const pz = rs.pose.slice();
  for (let k = 0; k < world.parts.length; k++) pz[k * 3 + 1] += L.lift;
  Object.assign(start, { startPose: pz, startVel: rs.vel.map((v, i) => (i % 3 === 0 ? v * v0 / rs.speed : v)), tilt: 0, drop: 0, settle: 0, frames: Math.round(seconds * HZ), shiftX: 0 });
  start.walk = null;
  start.terrain = { statics: L.statics, balls: L.balls, polys: L.polys, surface: L.surface, ropes: L.ropes };
  world.rebuild();
  const ep = new sim.Episode(world, weights, 0, 0, Object.assign({ phase: 0 }, start));
  ep.runner = { stance: rs.stance, t: -0.1, flight: false, jumping: false, jumpNext: false, strides: 0 };
  const correct = params.trailNet ? makeTrailNet(world, config, Box2D) : null;
  const D = installPopDirector(ep, world, L, { config, pose, Box2D, ElmanNetwork, params, trailNet: correct ? { correct, theta: params.trailNet } : null, skipWall: atRope });
  const endX = D.hole.x1 + 25;
  let lastRec = null, stage = 0, down = 0, stallX = -1e9, stallT = 0, why = 'time';
  while (!ep.advanceOneFrame()) {
    const px = world.px[0], z = D.zone();
    if (process.env.DEBUG_POP && ep.frame % 300 === 0) { const pl = D.chain.planner; console.log('   t', (ep.frame / HZ).toFixed(0), 'zone', z, 'chain', D.chain.mode, pl ? 'move ' + (pl.move ? pl.move.kind + (pl.move.k != null ? pl.move.k : '') + '->' + (pl.move.to != null ? pl.move.to : '') : '-') + ' hands ' + pl.lim.hand.map((h) => h.rung + (h.grip ? 'g' : '')).join(',') + ' feet ' + pl.lim.foot.map((f) => f.rung).join(',') + ' phase ' + pl.phase : '', 'pelvis', px.toFixed(2), world.py[0].toFixed(2), 'leap', JSON.stringify({ aim: D.chain.leap.aimI, hold: D.chain.leap.holdI, missed: D.chain.leap.missed }), 'rec', D.rec ? D.rec.phase + '/' + D.rec.count : '-', 'log', D.chain.log.slice(-3).join(',')); }
    if (process.env.DEBUG_REC && D.rec && D.rec.phase !== lastRec) { lastRec = D.rec.phase; const nx = D.features.filter((f) => f.x1 > px - 0.2).sort((a, b) => a.x0 - b.x0)[0]; console.log('   rec', (ep.frame / HZ).toFixed(1) + 's', lastRec, '#' + D.rec.count, 'x', px.toFixed(2), 'floor', D.floorAt(px).toFixed(2), 'zone', z, 'next', nx ? nx.type + ' at ' + (nx.x0 - px).toFixed(2) + (nx.top != null ? ' top ' + nx.top.toFixed(2) : nx.to != null ? ' to ' + nx.to.toFixed(2) : '') : '-'); }
    if (stage < 1 && z === 'wall') stage = 1;
    if (stage < 2 && D.chain.leap.caughtY != null) stage = 2;
    if (stage < 3 && D.wallDone) stage = 3;
    if (stage < 4 && D.rope.phase === 'swing') stage = 4;
    if (stage < 5 && D.ropeDone) stage = 5;
    if (px > endX) { stage = 6; why = 'end'; break; }
    // falls, as the playground counts them
    const busy = (z === 'wall' && D.chain.mode !== 'run' && D.chain.mode !== 'run2') || ep.grips[0].joint || ep.grips[1].joint || (D.rec && D.rec.busy());
    const fl = D.floorAt(world.px[2]);
    const isDown = !busy && (world.py[2] - fl < 0.95 + 0.0 && ep.measure.headHeight - fl < 0.95 || (ep.measure.torsoCos || 0) < 0.3 || world.py[0] < D.floorAt(px) - 0.1);
    down = isDown ? down + 1 : 0;
    if (down > (D.rec ? 3 : 1.2) * HZ) {
      const f = D.features.reduce((a, b) => (Math.abs((b.x0 + b.x1) / 2 - px) < Math.abs((a.x0 + a.x1) / 2 - px) ? b : a), D.features[0]);
      why = 'fell in ' + z + (z === 'wall' ? '/' + D.chain.mode : z === 'rope' ? '/' + D.rope.phase : '') + ' at x ' + px.toFixed(1) + ' near ' + f.type + ' (' + (px - f.x0).toFixed(1) + ' m) floor ' + D.floorAt(px).toFixed(2) + ' rock at ' + D.wall.x0.toFixed(1);
      break;
    }
    if (px > stallX + 0.5 || busy) { stallX = px; stallT = 0; } else if (++stallT > 25 * HZ) { why = 'stuck in ' + z; break; }
  }
  return { stage, why, t: ep.frame / HZ, x: world.px[0], getups: D.rec ? D.rec.count : 0 };
}

const count = new Array(STAGES.length).fill(0);
for (let i = 0; i < seeds; i++) {
  const e = episode(base + i);
  for (let s = 0; s <= e.stage; s++) count[s]++;
  console.log('seed', base + i, STAGES[e.stage].padEnd(18), '|', e.why, (e.t).toFixed(0) + 's', e.getups ? '(got up ' + e.getups + 'x)' : '');
}
console.log(STAGES.map((s, i) => s + ' ' + count[i] + '/' + seeds).join(' · '));
