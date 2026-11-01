// DIRECTOR: which skill drives, by where the body is and how far the current skill got, on any course made of
// trail pieces, rocks to leap at and climb, and rope holes (the Prince of Persia level, src/terrain.js
// makePopLevel, or a course built in the playground's sandbox, buildSandbox). Each skill is installed once
// with an active() switch:
//   trail  the runner + jump selector (+ the trail network) between the big pieces
//   wall   one obstacle runner per rock, in leap mode: run at it, jump, grab a knob, hang, climb (crag
//          planner), over the top, stand, walk, run (falls back to walk-stop-climb if the grab misses)
//   rope   one rope runner per rope hole: jump, catch the rope, swing, let go, land, run on
//   pit    a hole with ledges on its far wall: a body dropping in reaches for the top ledge, hangs and climbs
//          out (an obstacle runner for that wall, installed when needed)
//   recover after a fall: get up (the get-up network), walk, run on
// Zones: a rock from up to 7 m before it (1 m past the piece before it) until it runs on 5 m past the summit
// edge; a rope hole from 4 m before it until it runs on 3 m past it; trail everywhere else. Code, not a
// network: the rules are the zones.
import { installRunner } from './run-selector.js';
import { installObstacleRunner } from './obstacle-runner.js';
import { installRopeRunner } from './rope-swing.js';
import { installRecovery } from './recover.js';

// level: { statics, balls, features, levels, walls: [{x0, summit, plateauEnd, H, crag}], ropeHoles: [{x0, x1,
// lift, rope}] } in coordinates relative to the start (as the terrain was given to the episode)
export function installCourseDirector(ep, world, level, ctx) {
  const { config, pose, Box2D, ElmanNetwork, params } = ctx;
  const P_ = config.partIndex;
  // where the course's 0 is in the world: the first floor piece, else the feet at the start
  const mid = level.statics.length && ep.terrain.statics.length ? ep.terrain.statics[0].x0 - level.statics[0].dx : (world.px[P_.footLeft] + world.px[P_.footRight]) / 2;
  const X = (x) => x + mid;
  const levels = level.levels.map(([a, b, l]) => [X(a), X(b), l]).sort((a, b) => a[0] - b[0]);
  const features = level.features.map((f) => Object.assign({}, f, { x0: X(f.x0), x1: X(f.x1) })).sort((a, b) => a.x0 - b.x0);
  const balls = ep.terrain.balls || [];
  const HZ = config.physics.hz;
  // the end of the last piece before x (where a run-up can start)
  const prevEnd = (x) => { let e = -1e9; for (const f of features) if (f.x1 <= x - 0.05) e = Math.max(e, f.x1); return e; };
  const walls = (level.walls || []).map((w, i) => {
    const x0 = X(w.x0), surface = w.crag.surface.map(([x, y]) => [X(x), y]);
    return { kind: 'wall', i, x0, summit: X(w.summit), end: X(w.plateauEnd), H: w.H, crag: w.crag, surface, runFrom: Math.max(prevEnd(x0) + 1, x0 - 7), done: !!ctx.skipWall, on: false, getups: 0 };
  });
  const ropes = (level.ropeHoles || []).map((h, i) => {
    const x0 = X(h.x0);
    return { kind: 'rope', i, x0, x1: X(h.x1), lift: h.lift, rope: ep.terrain.ropes[h.rope], runFrom: Math.max(prevEnd(x0) + 0.3, x0 - 4), done: false, on: false };
  });
  const items = walls.concat(ropes).sort((a, b) => a.x0 - b.x0);
  // the floor (holes bridged): the level of the last piece that starts at or before x
  function floorAt(x) {
    let lv = levels.length ? levels[0][2] : 0;
    for (const [a, b, l] of levels) { if (x >= a) lv = l; else break; }
    for (const w of walls) if (x >= w.x0 && x < w.summit) lv = Math.max(lv, 0);
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
  // trail obstacles: every feature but the steps down, the rocks and the rope holes
  const obstacles = [];
  for (const f of features) {
    if (f.type === 'hole') obstacles.push({ x0: f.x0, x1: f.x1, kind: 'agujero' });
    else if (f.type === 'up') obstacles.push({ x0: f.x0, x1: f.x0 + 0.02, top: f.to, kind: 'escalón' });
    else if (f.type === 'box') obstacles.push({ x0: f.x0, x1: f.x1, top: f.top, kind: 'caja' });
    else if (f.type === 'ball') { const b = balls[f.ball]; if (b) obstacles.push({ live: () => { const q = Box2D.b2Body_GetPosition(b.id); return { x0: q.x - b.r, x1: q.x + b.r, top: q.y + b.r, kind: 'bola' }; } }); }
  }
  let zoneF = -1, zoneNow = 'trail', cur = null;
  const curItem = () => items.find((it) => !it.done) || null;
  // a fall on the trail (head low or tipped over for half a second, not in a pit, no hand holding): get up and go on
  let downF = 0, resumeAt = -1e9;
  const fallCheck = () => {
    // on the trail, or at a rock before any hand is on it (or on its top, walking or running off)
    const atRock = zoneNow === 'wall' && ['run', 'leap', 'walk', 'stop', 'hold', 'drop', 'walk2', 'run2'].includes(cur.OR.mode);
    // at a rope: before the jump, or once past the hole
    const atRope = zoneNow === 'rope' && (cur.RR.phase === 'run' || world.px[0] > cur.x1 + 0.3);
    if (!rec || rec.busy() || !(zoneNow === 'trail' || atRock || atRope)) { downF = 0; return; }
    const m = ep.measure, fl = floorAt(world.px[0]);
    const down = (m.headHeight - fl < 0.95 || (m.torsoCos || 0) < 0.3) && world.py[0] > fl - 0.1 && !(ep.grips[0].joint || ep.grips[1].joint);
    downF = down ? downF + 1 : 0;
    if (downF > 0.5 * HZ) {
      downF = 0;
      // at the foot of a rock a get-up rarely works (the rock is in the way): three tries there at most
      if (atRock) { cur.getups++; if (cur.getups > 3) return; }
      if (rec.count >= (ctx.maxGetups || 6)) return; // six get-ups in one level: enough
      rec.noWalk = false;
      rec.start();
    }
  };
  // PITS (holes with ledges on the far wall, terrain makeTrailCourse knobs)
  const pits = features.filter((f) => (f.type === 'hole' || f.type === 'ropehole') && f.knobs && f.knobs.length);
  let pitDownF = 0;
  let pit = null;
  const pitLog = [];
  const pitCheck = () => {
    if (pit || (rec && rec.busy())) return;
    // on the trail; at a rope when not holding it (fell short, or off the far edge)
    const ropeFall = zoneNow === 'rope' && !(ep.grips[0].joint || ep.grips[1].joint);
    if (zoneNow !== 'trail' && !ropeFall) return;
    const px = world.px[0], vy = ep.measure.centerVelocityY || 0;
    for (const f of pits) {
      if ((f.type === 'ropehole') !== ropeFall) continue;
      if (px > f.x0 + 0.02 && px < f.x1 + 0.1 && world.py[0] < f.from + 0.15 && vy < -0.4) {
        const x1 = f.x1, H = f.to;
        const knobs = f.knobs.map((k) => ({ lip: X(k.lip), back: X(k.back), y: k.y }));
        const surface = [[x1 - 4, 0], [x1, 0], [x1, H], [x1 + 8, H]];
        const crag = { knobs, surface, H, edgeX: x1, summitX: x1 };
        const P = params;
        const OR = installObstacleRunner(ep, world, { config, pose, Box2D, ElmanNetwork, crag, H, surface, aimTop: true, speed: () => vt, stopAt: 0.3, vswitch: 1.3, leapP: P.leap, active: () => pit && pit.OR === OR && !pit.done, params: { runP: P.runner, walkP: P.walkFast, slowP: P.slow, stopP: P.stop, stopK: P.stopK, cragP: P.cragLeap || P.cragObstacle, getup: P.getup } });
        pit = { f, OR, done: false, t0: ep.frame };
        // (in the rope's wide hole the reach caught the rope as often as the ledge: there it lands, gets up and
        // climbs the far wall from standing)
        if (f.type === 'ropehole') OR.reset(true); else OR.leapNow();
        pitLog.push({ x: +px.toFixed(2), t: +(ep.frame / HZ).toFixed(1) });
        return;
      }
    }
  };
  const zone = () => {
    if (zoneF === ep.frame) return zoneNow;
    zoneF = ep.frame;
    const px = world.px[0];
    if (rec && rec.busy()) { zoneNow = 'recover'; return zoneNow; }
    if (pit && !pit.done) {
      // got up in the pit: walk to the far wall and climb it from standing
      if (rec && rec.phase === 'done') { rec.phase = null; pit.OR.reset(true); }
      // out: running again 4 m past the pit (or given up after 120 s)
      if ((pit.OR.mode === 'run2' && px > pit.f.x1 + 4) || ep.frame - pit.t0 > 120 * HZ) {
        pit.done = true; pitLog[pitLog.length - 1].out = pit.OR.mode === 'run2';
        if (pit.f.type === 'ropehole' && pit.OR.mode === 'run2') for (const r of ropes) if (Math.abs(r.x0 - pit.f.x0) < 0.05) r.done = true;
        pit = null; resumeAt = ep.frame - 10 * HZ;
      } else {
        // down at the foot of the pit (missed the ledge, or let go): get up there first
        const m = ep.measure, OM = pit.OR.mode, low = world.py[0] < pit.f.from - 0.5;
        const down = low && (m.headHeight < 0.95 || (m.torsoCos || 0) < 0.3) && !(ep.grips[0].joint || ep.grips[1].joint) && ['walk', 'stop', 'hold', 'drop', 'leap'].includes(OM);
        pitDownF = down ? pitDownF + 1 : 0;
        if (rec && pitDownF > 0.5 * HZ && rec.count < 8) { pitDownF = 0; rec.noWalk = true; rec.start(); zoneNow = 'recover'; return zoneNow; }
        zoneNow = 'pit'; return zoneNow;
      }
    }
    cur = curItem();
    if (rec && rec.phase === 'done') {
      rec.phase = null; resumeAt = ep.frame;
      // at a rock: walk up to it and climb from standing (a get-up leaves no room for a run-up)
      if (cur && cur.on && cur.kind === 'wall' && !['walk2', 'run2'].includes(cur.OR.mode)) cur.OR.reset(true);
      // past a rope's hole: that part is done
      if (cur && cur.on && cur.kind === 'rope' && px > cur.x1) cur.done = true;
    }
    // through: running again past the piece (up to speed first: run2 ramps it up)
    for (const it of items) if (!it.done && (it.kind === 'wall' ? it.OR.mode === 'run2' && px > it.summit + 5 : it.RR.phase === 'run2' && px > it.x1 + 3)) it.done = true;
    cur = curItem();
    // once a zone has started it stays on until its skill is through (a missed leap can step back out of it)
    if (cur && !cur.on && px > cur.runFrom) {
      cur.on = true;
      // too little room to run at the rock: walk up to it and climb from standing
      if (cur.kind === 'wall' && cur.x0 - Math.max(cur.runFrom, px) < 3) cur.OR.reset(true);
    }
    zoneNow = cur && cur.on ? cur.kind : 'trail';
    pitCheck();
    if (pit) { zoneNow = 'pit'; return zoneNow; }
    fallCheck();
    if (rec && rec.busy()) zoneNow = 'recover';
    return zoneNow;
  };
  const vt = params.trail.vRun || 2.6;
  const net = ctx.trailNet ? { correct: ctx.trailNet.correct, theta: ctx.trailNet.theta } : null;
  // back to running after a get-up: the commanded speed comes up from a walk over two seconds
  const RAMP = ctx.ramp || +((globalThis.process && process.env && process.env.RAMP) || 0.6);
  const speedT = () => Math.min(vt, 1.2 + Math.max(0, (ep.frame - resumeAt) / HZ) * RAMP);
  installRunner(ep, world, params.trail, { config, obstacles, floorAt, speed: speedT, net, active: () => zone() === 'trail' });
  const ropeNet = ctx.trailNet && params.ropeNet ? { correct: ctx.trailNet.correct, theta: params.ropeNet } : null;
  for (const r of ropes) r.RR = installRopeRunner(ep, world, params.rope, { config, pose, Box2D, runnerP: params.runner, speed: () => 2.4, hole: r, rope: r.rope, lift: r.lift, walkP: params.walkFast, net: ropeNet, active: () => zone() === 'rope' && cur === r });
  for (const w of walls) w.OR = installObstacleRunner(ep, world, { config, pose, Box2D, ElmanNetwork, crag: w.crag, surface: w.surface, H: w.H, speed: () => vt, stopAt: 0.6, vswitch: 1.3, leapP: params.leap, active: () => zone() === 'wall' && cur === w, params: { runP: params.runner, walkP: params.walkFast, slowP: params.slow, stopP: params.stop, stopK: params.stopK, cragP: params.cragLeap || params.cragObstacle, getup: params.getup } });
  // recovery last: its wrapper runs first, and the skills are off while it is busy
  const rec = params.getup && params.walkFast ? installRecovery(ep, world, { config, ElmanNetwork, getup: params.getupTrail || params.getup, walkP: params.walkFast, floorAt: (x) => (pit && !pit.done && x > pit.f.x0 - 0.15 && x < pit.f.x1 + 0.05 ? 0 : floorAt(x)) }) : null;
  // the current (or first) rock and rope, for whoever reads one of each (pop.js, the inspector)
  const wallOf = () => (cur && cur.kind === 'wall' ? cur : walls.find((w) => !w.done) || walls[0]);
  const ropeOf = () => (cur && cur.kind === 'rope' ? cur : ropes.find((r) => !r.done) || ropes[0]);
  return {
    pitLog, rec, zone, walls, ropes, floorAt, groundAt, features, obstacles,
    get pit() { return pit; }, get cur() { return cur; },
    get chain() { const w = wallOf(); return w && w.OR; }, get rope() { const r = ropeOf(); return r && r.RR; },
    get wall() { return wallOf(); }, get hole() { return ropeOf(); },
    get wallDone() { return walls.every((w) => w.done); }, get ropeDone() { return ropes.every((r) => r.done); },
  };
}

// The Prince of Persia level (makePopLevel): one rock, one rope hole
export function installPopDirector(ep, world, level, ctx) {
  const Z = level.zones;
  const L = Object.assign({}, level, {
    walls: [{ x0: Z.wall.x0, summit: Z.wall.summit, plateauEnd: Z.wall.plateauEnd, H: Z.wall.H, crag: level.crag }],
    ropeHoles: [{ x0: Z.rope.x0, x1: Z.rope.x1, lift: Z.rope.lift, rope: 0 }],
  });
  return installCourseDirector(ep, world, L, ctx);
}
