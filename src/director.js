// DIRECTOR for the Prince of Persia level (src/terrain.js makePopLevel): which skill drives, by where the body
// is and how far the current skill got. Each skill is installed once with an active() switch:
//   trail  the runner + jump selector (+ the trail network) on every raised-floor stretch (A, B, C)
//   wall   the obstacle runner in leap mode: run at the rock, jump, grab a knob, hang, climb (crag planner),
//          over the top, stand, walk, run (falls back to walk-stop-climb if the grab misses)
//   rope   the rope runner over the wide hole: jump, catch the rope, swing, let go, land, run on
// Zones: wall from 4 m before the rock until it runs on 2.5 m past the summit edge; rope from 4.5 m before the
// hole until it runs on 3 m past it; trail everywhere else. Code, not a network: the rules are the zones.
import { installRunner } from './run-selector.js';
import { installObstacleRunner } from './obstacle-runner.js';
import { installRopeRunner } from './rope-swing.js';

export function installPopDirector(ep, world, level, ctx) {
  const { config, pose, Box2D, ElmanNetwork, params } = ctx;
  const mid = ep.terrain.statics[0].x0 - level.statics[0].dx;
  const X = (x) => x + mid;
  const levels = level.levels.map(([a, b, l]) => [X(a), X(b), l]);
  const features = level.features.map((f) => Object.assign({}, f, { x0: X(f.x0), x1: X(f.x1) }));
  const balls = ep.terrain.balls || [];
  const Z = level.zones;
  const wall = { x0: X(Z.wall.x0), summit: X(Z.wall.summit), end: X(Z.wall.plateauEnd), H: Z.wall.H };
  const hole = { x0: X(Z.rope.x0), x1: X(Z.rope.x1) };
  // the floor (holes bridged): the level of the last piece that starts at or before x
  function floorAt(x) {
    let lv = levels.length ? levels[0][2] : 0;
    for (const [a, b, l] of levels) { if (x >= a) lv = l; else break; }
    if (x >= wall.x0 && x < wall.summit) lv = Math.max(lv, 0);
    return lv;
  }
  function groundAt(x) {
    let g = 0;
    for (const [a, b, l] of levels) if (x >= a && x <= b) g = Math.max(g, l);
    for (const f of features) if (f.type === 'box' && x >= f.x0 && x <= f.x1) g = Math.max(g, f.top);
    for (const b of balls) { const q = Box2D.b2Body_GetPosition(b.id); if (Math.abs(x - q.x) < b.r) g = Math.max(g, q.y + Math.sqrt(b.r * b.r - (x - q.x) ** 2)); }
    return g;
  }
  ep.floorAt = floorAt; ep.groundAt = groundAt;
  // trail obstacles: every feature but the steps down, the rock and the rope's hole
  const obstacles = [];
  for (const f of features) {
    if (f.type === 'hole') obstacles.push({ x0: f.x0, x1: f.x1, kind: 'agujero' });
    else if (f.type === 'up') obstacles.push({ x0: f.x0, x1: f.x0 + 0.02, top: f.to, kind: 'escalón' });
    else if (f.type === 'box') obstacles.push({ x0: f.x0, x1: f.x1, top: f.top, kind: 'caja' });
    else if (f.type === 'ball') { const b = balls[f.ball]; if (b) obstacles.push({ live: () => { const q = Box2D.b2Body_GetPosition(b.id); return { x0: q.x - b.r, x1: q.x + b.r, top: q.y + b.r, kind: 'bola' }; } }); }
  }
  let wallDone = false, ropeDone = false, zoneF = -1, zoneNow = 'trail';
  const zone = () => {
    if (zoneF === ep.frame) return zoneNow;
    zoneF = ep.frame;
    const px = world.px[0];
    if (!wallDone && chain && chain.mode === 'run2' && px > wall.summit + 2.5) wallDone = true;
    if (!ropeDone && rope && rope.phase === 'run2' && px > hole.x1 + 3) ropeDone = true;
    zoneNow = !wallDone && px > wall.x0 - 4 ? 'wall' : wallDone && !ropeDone && px > hole.x0 - 4.5 ? 'rope' : 'trail';
    return zoneNow;
  };
  const vt = params.trail.vRun || 2.6;
  const net = ctx.trailNet ? { correct: ctx.trailNet.correct, theta: ctx.trailNet.theta } : null;
  installRunner(ep, world, params.trail, { config, obstacles, floorAt, speed: () => vt, net, active: () => zone() === 'trail' });
  const rope = installRopeRunner(ep, world, params.rope, { config, pose, Box2D, runnerP: params.runner, speed: () => 2.4, hole, rope: ep.terrain.ropes[0], lift: Z.rope.lift, walkP: params.walkFast, active: () => zone() === 'rope' });
  const chain = installObstacleRunner(ep, world, { config, pose, Box2D, ElmanNetwork, crag: level.crag, H: wall.H, speed: () => vt, stopAt: 0.6, vswitch: 1.3, leapP: params.leap, active: () => zone() === 'wall', params: { runP: params.runner, walkP: params.walkFast, slowP: params.slow, stopP: params.stop, stopK: params.stopK, cragP: params.cragObstacle, getup: params.getup } });
  return { zone, chain, rope, floorAt, groundAt, features, obstacles, wall, hole, get wallDone() { return wallDone; }, get ropeDone() { return ropeDone; } };
}
