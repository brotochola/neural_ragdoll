// CLIMBING a wall with rungs, hands and feet, by a planner in code plus the pose executor (src/pose.js):
// three points stay on the wall while the fourth limb moves one rung up (left foot, right foot, push the
// pelvis up, left hand, right hand, ...). Hands close their grip when they touch a rung (the sim pins
// the hand to it); feet stand on the rungs. No external forces: joint motors and grips only.
// The planner's numbers (how far the pelvis hangs from the wall, how high above the feet, timings,
// clearances) are tuned with the cross-entropy method on the height reached without falling.
//   node climb-wall.js --iters 30 --out trained/climb-wall.json
//   node climb-wall.js --show --init trained/climb-wall.json --record out.json
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
const out = arg('--out', 'trained/climb-wall.json');
const init = arg('--init', null);
const pop = +arg('--pop', 24);
const elite = +arg('--elite', 6);
const seeds = +arg('--seeds', 3);
const seconds = +arg('--seconds', 25);
const wallH = +arg('--wall', 3.3); // wall height; rungs up to 0.3 below the top
const spacing = +arg('--spacing', 0.3); // rung spacing

const Box2D = await import(pathToFileURL(path.join(root, 'vendor/PhaserBox2D.min.js')).href);
const { createSim } = await import('./src/sim.js');
const { ElmanNetwork } = await import('./src/net.js');
const { makeSeededRandom, randomNormalSample } = await import('./src/utils.js');
const { evalMode } = await import('./src/training.js');
const pose = await import('./src/pose.js');

setSkill(6);
const sim = createSim(Box2D);
const world = sim.build();
const HZ = config.physics.hz;
const J = config.net.jointCount;
const OUT = J + config.net.gripOutputs;
const P = config.partIndex;
const weights = ElmanNetwork.copy(JSON.parse(fs.readFileSync(path.join(root, 'trained/stand-and-walk.json'), 'utf8')).theta);

// [name, start, spread]
const SPEC = [
  // Geometry (2D: the knees cannot go sideways round the wall, and leaning in costs overhead reach since
  // the shoulder bends 2.5 rad at most): hips about 0.4 m out, trunk upright, legs fairly straight.
  ['pelvisOff', 0.56, 0.08], // pelvis this far from the wall face
  ['pelvisAbove', 0.84, 0.08], // pelvis this high above the higher foot
  ['armDown', 0.35, 0.1], // and at least this far below the lower hand
  ['lean', 0.0, 0.12], // torso world angle (negative leans in toward the wall)
  ['spanMax', 1.7, 0.1], // a hand moves up while hands-to-feet stays within this
  ['spanMin', 1.3, 0.1], // a foot moves up while hands-to-feet stays above this
  ['tHand', 0.6, 0.2], // seconds for a hand move
  ['tFoot', 0.7, 0.2], // seconds for a foot move
  ['tRise', 0.6, 0.2], // seconds to push the pelvis up
  ['clear', 0.1, 0.05], // a moving limb arcs this far out from the wall
  ['torsoGain', 0.5, 0.4], // supporting legs turn the trunk back to its lean
  ['footIn', 0.0, 0.04], // foot target nudged into the wall (+) or out (-)
  ['handUp', 0.03, 0.03], // hand target above the rung centre
  ['topHand', 0.03, 0.04], // hands on the top surface this far past the edge
  ['mantleLean', -0.8, 0.25], // trunk lean as the hips come over the edge
  ['mantleUp', 0.55, 0.12], // hips above the top with one foot on it
  ['spanTop', 1.2, 0.12], // hands go onto the top once the feet are within this of it
  ['spanLow', 0.8, 0.1], // otherwise a foot may still come up to this far below the hands
  ['pelvisOffTop', 0.36, 0.08], // hips this far from the wall once a hand is at the top
];
const decode = (v) => {
  const p = {};
  SPEC.forEach(([k, b], i) => (p[k] = b + (v[i] || 0)));
  for (const k of ['tHand', 'tFoot', 'tRise']) p[k] = Math.max(0.2, p[k]);
  return p;
};

const smooth = (s) => s * s * (3 - 2 * s);

function episode(p, seed, record) {
  const r = makeSeededRandom(5000 + seed);
  const start = sim.fillTrainingStart({}, r, evalMode(6));
  start.tilt = 0;
  start.walk = null;
  start.frames = Math.round(seconds * HZ);
  const wallDx = +(process.env.WALL_DX || 0.6) + r() * 0.08;
  const holds = [];
  for (let y = spacing; y < wallH - 0.25; y += spacing) holds.push({ dx: wallDx - 0.1, y, w: 0.1, h: 0.04 });
  start.terrain = { statics: [{ dx: wallDx, w: 3, h: wallH }], holds };
  world.rebuild();
  const ep = new sim.Episode(world, weights, 0, 0, Object.assign({ phase: 0 }, start));
  const rungs = ep.terrain.holds; // absolute
  const wallX = ep.terrain.statics[0].x0;
  const rungAt = (i) => (i <= 0 ? null : rungs[i - 1]);
  // index N+1 is the TOP of the wall: hands on its top surface past the edge, then the feet
  const N = rungs.length;
  const TOP = N + 1;
  const ry = (i) => (i <= 0 ? 0 : i === TOP ? wallH : rungs[i - 1].top);
  // limb state: rung index (0 = floor) and current world target
  const feet0 = [P.footLeft, P.footRight].map((q) => pose.soleOf(world, q));
  const lim = {
    hand: [{ rung: 0, pos: null, grip: false }, { rung: 0, pos: null, grip: false }],
    foot: [{ rung: 0, pos: { x: feet0[0].x, y: feet0[0].y } }, { rung: 0, pos: { x: feet0[1].x, y: feet0[1].y } }],
  };
  const handTarget = (i) => (i === TOP ? { x: wallX + p.topHand, y: wallH + 0.03 } : { x: wallX - 0.05, y: ry(i) + p.handUp });
  const footTarget = (i) => (i <= 0 ? null : i === TOP ? { x: wallX + 0.22, y: wallH + 0.005 } : { x: wallX - 0.07 + p.footIn, y: ry(i) + 0.005 });
  const handsTop = () => lim.hand[0].rung === TOP && lim.hand[1].rung === TOP;
  const feetTop = () => lim.foot.filter((f) => f.rung === TOP).length;
  // first rung the hands can reach standing
  let reach = 0;
  for (let i = 1; i <= rungs.length; i++) if (ry(i) < +(process.env.REACH_Y || 1.55)) reach = i; // about shoulder height: an arm raised overhead next to the wall hit the rungs
  // the move list is generated as the climb goes
  let move = null;
  const pv0 = pose.pelvisOf(world);
  let pelvisGoal = { x: pv0.x, y: pv0.y };
  let pelvisFrom = { ...pelvisGoal };
  const next = () => {
    // highest rung for hands, feet
    const hMin = Math.min(lim.hand[0].rung, lim.hand[1].rung);
    const fMax = Math.max(lim.foot[0].rung, lim.foot[1].rung);
    const fMin = Math.min(lim.foot[0].rung, lim.foot[1].rung);
    if (lim.hand[0].rung === 0) return { kind: 'hand', k: 0, to: reach };
    if (lim.hand[1].rung === 0) return { kind: 'hand', k: 1, to: reach };
    // feet climb while they stay a body length below the hands; then the pelvis rises, then hands
    const handsY = Math.min(ry(lim.hand[0].rung), ry(lim.hand[1].rung));
    const feetY = Math.min(ry(lim.foot[0].rung), ry(lim.foot[1].rung));
    // hands first while they can still reach from the feet, the lower hand first; then the lower foot
    const hk = lim.hand[0].rung <= lim.hand[1].rung ? 0 : 1;
    const hTo = lim.hand[hk].rung + 1;
    if (hTo < TOP && ry(hTo) - feetY <= p.spanMax) return { kind: 'hand', k: hk, to: hTo };
    // onto the top only once the feet are high enough for the hips to bring the shoulders up to the edge
    if (hTo === TOP && wallH - feetY <= p.spanTop) return { kind: 'hand', k: hk, to: hTo };
    const fk = lim.foot[0].rung <= lim.foot[1].rung ? 0 : 1;
    const fTo = lim.foot[fk].rung + 1;
    // MANTLE: both hands on top; the feet walk up the last rungs, then step onto the top
    if (handsTop()) {
      if (feetTop() === 1) { const k2 = lim.foot[0].rung === TOP ? 1 : 0; return { kind: 'foot', k: k2, to: TOP }; }
      if (fTo <= N) return { kind: 'foot', k: fk, to: fTo };
      if (fTo === TOP || Math.max(lim.foot[0].rung, lim.foot[1].rung) >= N) { const k1 = lim.foot[0].rung >= lim.foot[1].rung ? 0 : 1; if (lim.foot[k1].rung !== TOP) return { kind: 'foot', k: k1, to: TOP }; }
      return null;
    }
    if (fTo <= rungs.length && handsY - ry(fTo) >= p.spanMin) return { kind: 'foot', k: fk, to: fTo };
    // stuck (a hand caught a lower rung than planned): a foot goes up anyway while there is room
    if (fTo <= rungs.length && handsY - ry(fTo) >= p.spanLow) return { kind: 'foot', k: fk, to: fTo };
    return null;
  };
  let leanNow = p.lean;
  const pelvisFor = () => {
    const fMax = Math.max(lim.foot[0].pos.y, lim.foot[1].pos.y);
    leanNow = p.lean;
    const ft = feetTop();
    if (ft === 2) return { x: wallX + 0.28, y: wallH + 0.98 };
    if (ft === 1) { leanNow = -0.5; return { x: wallX + 0.12, y: wallH + p.mantleUp }; }
    if (handsTop()) {
      // as the feet near the top the hips come in over the edge and the chest leans over the top
      const frac = Math.max(0, Math.min(1, (fMax - (wallH - 1.3)) / 1.0));
      leanNow = p.lean + (p.mantleLean - p.lean) * frac;
      return { x: wallX - p.pelvisOff + (p.pelvisOff - 0.12) * frac, y: Math.min(fMax + p.pelvisAbove, wallH + 0.3) };
    }
    const hands = lim.hand.filter((h) => h.pos).map((h) => h.pos.y);
    const onRung = lim.foot.some((f) => f.rung > 0);
    let y = onRung ? fMax + p.pelvisAbove : 0.98; // sole heights read 5 cm high, so 0.98 is standing
    if (hands.length) y = Math.min(y, Math.min(...hands) - p.armDown);
    const onWall = lim.foot.some((f) => f.rung > 0) || lim.hand.some((h) => h.rung > 0);
    // near the top (a hand on the last rung or over the edge): hips in closer so the arms reach over it
    const nearTop = lim.hand.some((h) => h.rung >= N);
    return { x: onWall ? wallX - (nearTop ? p.pelvisOffTop : p.pelvisOff) : pv0.x, y: Math.max(0.6, Math.min(y, 0.95 + Math.max(0, fMax))) };
  };
  const startMove = () => {
    const m = next();
    if (!m) { move = { kind: 'hold', t: 0, dur: 99 }; return; }
    const limb = m.kind === 'hand' ? lim.hand[m.k] : lim.foot[m.k];
    const from = limb.pos || pose.handOf(world, m.k === 0 ? P.handLeft : P.handRight);
    const to = m.kind === 'hand' ? handTarget(m.to) : footTarget(m.to);
    move = Object.assign(m, { t: 0, dur: m.kind === 'hand' ? p.tHand : p.tFoot, from: { x: from.x, y: from.y }, target: to, wait: 0 });
    if (m.kind === 'hand') limb.grip = false;
    pelvisFrom = { ...pelvisGoal };
  };
  let best = 0;
  let maxFoot = 0;
  let fell = false;
  const recFrames = record ? [] : null;
  const tmp = new Float64Array(OUT);
  ep.updateGrip = ((orig) => function () {
    const m = this.measure;
    if (!move) startMove();
    move.t += 1 / HZ;
    const s = Math.min(1, move.t / move.dur);
    const e = smooth(s);
    // limb targets
    const hands = lim.hand.map((h) => h.pos);
    const feet = lim.foot.map((f) => ({ x: f.pos.x, y: f.pos.y, angle: 0 }));
    const gripping = lim.hand.map((h) => h.grip);
    const planted = lim.foot.map(() => true);
    if (move.kind === 'hand' || move.kind === 'foot') {
      // path: to a point in front of the hold (out from the wall, a little above it) by 70% of the move,
      // then in onto it; straight lines dragged the hand up the rungs and pushed the body off backward
      const wp = { x: move.target.x - p.clear - 0.08, y: move.target.y + (move.kind === 'foot' ? 0.08 : 0.03) };
      // a hand first comes up close in front of the shoulder, elbow bent: swung up straight, the arm's
      // arc hit the rungs and pushed the body over backward
      const pts = [move.from];
      if (move.kind === 'hand') { const pv = pose.pelvisOf(world); pts.push({ x: Math.min(wp.x, pv.x + 0.15), y: Math.max(move.from.y, pv.y + 0.75) }); }
      // a foot first rises where it is (knee up, away from the wall), then goes in above the rung and down:
      // going forward first slid it under the rung below
      else { pts.push({ x: Math.min(move.from.x, wp.x) - 0.04, y: move.from.y + 0.04 }); pts.push({ x: Math.min(move.from.x, wp.x) - 0.04, y: wp.y }); }
      pts.push(wp, move.target);
      const segs = pts.length - 1;
      const cuts = segs === 4 ? [0, 0.2, 0.55, 0.8, 1] : segs === 3 ? [0, 0.4, 0.75, 1] : [0, 0.7, 1];
      let si = 0;
      while (si < segs - 1 && s > cuts[si + 1]) si++;
      const q = smooth(Math.min(1, (s - cuts[si]) / (cuts[si + 1] - cuts[si])));
      const pt = { x: pts[si].x + (pts[si + 1].x - pts[si].x) * q, y: pts[si].y + (pts[si + 1].y - pts[si].y) * q };
      if (move.kind === 'hand') { hands[move.k] = pt; gripping[move.k] = false; }
      else { feet[move.k] = { x: pt.x, y: pt.y, angle: 0 }; planted[move.k] = false; }
    }
    // pelvis: rise after a foot is placed (towards the pose the new holds allow)
    const goalNow = pelvisFor();
    const pr = Math.min(1, move.t / p.tRise);
    pelvisGoal = { x: pelvisFrom.x + (goalNow.x - pelvisFrom.x) * smooth(pr), y: pelvisFrom.y + (goalNow.y - pelvisFrom.y) * smooth(pr) };
    // both feet on top: let go and stand up
    if (feetTop() === 2) { hands[0] = hands[1] = null; gripping[0] = gripping[1] = false; lim.hand[0].grip = lim.hand[1].grip = false; }
    const goal = { pelvis: pelvisGoal, torso: leanNow, feet, hands, gripping, planted, torsoGain: p.torsoGain };
    // on the floor with no hand on the wall yet: keep the balance over the feet (ankles and hips)
    const floorOnly = lim.foot.every((f) => f.rung === 0) && !lim.hand.some((h) => h.grip);
    const jt = floorOnly && process.env.FLOOR_BAL ? pose.balancedTargets(goal, m, { planted }, pose.actualOf(world)) : pose.jointTargets(goal, pose.actualOf(world));
    pose.commandsFor(world, jt, tmp);
    if (process.env.DEBUG_LEG && move.kind === 'foot' && Math.round(move.t * HZ) % 10 === 0) {
      const L = move.k === 0 ? [5, 6, 7] : [11, 12, 13];
      const so = pose.soleOf(world, move.k === 0 ? P.footLeft : P.footRight);
      const pv = pose.pelvisOf(world);
      const S2 = move.k === 0 ? [11, 12, 13] : [5, 6, 7];
      const ang = (j) => Box2D.b2RevoluteJoint_GetAngle(world.joints[j].id);
      console.log('  stance foot pos', JSON.stringify(lim.foot[1 - move.k].pos), 'pelvis goal', JSON.stringify(pelvisGoal));
      console.log('  stance hip/knee/ankle target', jt[S2[0]].toFixed(2), jt[S2[1]].toFixed(2), jt[S2[2]].toFixed(2), 'actual', ang(S2[0]).toFixed(2), ang(S2[1]).toFixed(2), ang(S2[2]).toFixed(2), 'arms tgt', jt[2].toFixed(2), jt[3].toFixed(2), 'act', ang(2).toFixed(2), ang(3).toFixed(2));
      console.log('  foot', move.k, 't', move.t.toFixed(2), 'tgt', feet[move.k].x.toFixed(2), feet[move.k].y.toFixed(2), 'sole', so.x.toFixed(2), so.y.toFixed(2), 'pelvis', pv.x.toFixed(2), pv.y.toFixed(2), 'goal pel', pelvisGoal.x.toFixed(2), pelvisGoal.y.toFixed(2), 'hip/knee target', jt[L[0]].toFixed(2), jt[L[1]].toFixed(2), 'actual', world.joints[L[0]].angle != null ? '' : '', 'cmd', tmp[L[0]].toFixed(2), tmp[L[1]].toFixed(2));
    }
    for (let j = 0; j < J; j++) this.action[j] = tmp[j];
    // grips: held hands stay closed; a moving hand closes once it is at its rung
    for (let k = 0; k < 2; k++) {
      const moving = move.kind === 'hand' && move.k === k;
      // a moving hand closes only once it is at its rung (closing on the way caught lower rungs)
      let near = false;
      if (moving && s > 0.6) { const hp = pose.handOf(world, k === 0 ? P.handLeft : P.handRight); near = Math.hypot(hp.x - move.target.x, hp.y - move.target.y) < 0.07; }
      this.action[J + k] = lim.hand[k].grip || near ? 1 : -1;
    }
    // end of move: hand gripped (or give up after a wait), foot down
    if (s >= 1) {
      if (move.kind === 'hand') {
        const g = this.grips[move.k];
        if (g && g.joint) {
          // the rung actually held: the one nearest the hand (it may have caught a lower one on the way)
          const hp = pose.handOf(world, move.k === 0 ? P.handLeft : P.handRight);
          let bi = move.to, bd = 9;
          for (let i = 1; i <= rungs.length; i++) { const d = Math.hypot(hp.x - (rungs[i - 1].x0 + 0.05), hp.y - rungs[i - 1].y); if (d < bd) { bd = d; bi = i; } }
          lim.hand[move.k] = { rung: bi, pos: { x: hp.x, y: hp.y }, grip: true };
          move = null;
        }
        else if ((move.wait += 1 / HZ) > 0.6) { if (process.env.DEBUG_CW) { const hp = pose.handOf(world, move.k === 0 ? P.handLeft : P.handRight); console.log('  grip miss hand', move.k, 'at', hp.x.toFixed(2), hp.y.toFixed(2), 'target', move.target.x.toFixed(2), move.target.y.toFixed(2)); } move = null; } // retry the same move
      } else if (move.kind === 'foot') {
        // standing on the rung? (sole within 6 cm of its top and over it); else try again
        const so = pose.soleOf(world, move.k === 0 ? P.footLeft : P.footRight);
        const tgt = move.target;
        if (Math.abs(so.y - tgt.y) < 0.07 && Math.abs(so.x - tgt.x) < 0.12) { lim.foot[move.k] = { rung: move.to, pos: tgt }; move = null; }
        else if ((move.wait += 1 / HZ) > 0.5) {
          if (process.env.DEBUG_CW) console.log('  foot miss', move.k, 'at', so.x.toFixed(2), so.y.toFixed(2), 'target', tgt.x.toFixed(2), tgt.y.toFixed(2));
          move = null;
        }
      }
    }
    orig();
  })(ep.updateGrip.bind(ep));
  let frames = 0;
  let upTop = 0;
  let topped = false;
  while (!ep.advanceOneFrame()) {
    frames++;
    const m = ep.measure;
    const pel = world.py[0];
    const footLow = Math.min(world.py[P.footLeft], world.py[P.footRight]);
    if (record && frames % 2 === 0) {
      const pz = [];
      for (let k = 0; k < world.parts.length; k++) pz.push(+world.px[k].toFixed(3), +world.py[k].toFixed(3), +Math.atan2(world.s[k], world.c[k]).toFixed(3));
      recFrames.push({ p: pz, up: m.torsoCos > 0.7 ? 1 : 0, gl: ep.grips[0].joint ? 1 : 0, gr: ep.grips[1].joint ? 1 : 0 });
    }
    // fell: head below the hips, or a body length below its highest point with no grip
    const gripped = ep.grips[0].joint || ep.grips[1].joint;
    best = Math.max(best, footLow);
    if (m.headHeight < world.py[0] || (!gripped && footLow < best - 0.5 && best > 0.3)) { fell = true; break; }
    maxFoot = Math.max(maxFoot, Math.min(lim.foot[0].rung, lim.foot[1].rung));
    // on top: both feet up there and standing for a second
    const onTopNow = feetTop() === 2 && footLow > wallH - 0.05 && m.headHeight > wallH + 1.3 && m.torsoCos > 0.8;
    upTop = onTopNow ? upTop + 1 / HZ : 0;
    if (upTop > 1.0) { topped = true; break; }
  }
  const top = rungs.length;
  const reached = Math.min(lim.foot[0].rung, lim.foot[1].rung);
  const T = frames / HZ;
  const score = 3 * Math.min(best, wallH) / wallH * 3 + (fell ? -1 : 1) + (handsTop() ? 1 : 0) + feetTop() * 1.5 + (topped ? 5 + 2 * (1 - T / seconds) : 0);
  const why = (topped ? 'TOPPED ' + T.toFixed(1) + 's ' : '') + 'feet rung ' + reached + '/' + top + ' best foot height ' + best.toFixed(2) + ' fell ' + (fell ? T.toFixed(1) + 's' : '-') + ' hands ' + lim.hand[0].rung + '/' + lim.hand[1].rung + ' feet on top ' + feetTop();
  const status = topped ? 'climbed onto the top (' + wallH.toFixed(1) + ' m) in ' + T.toFixed(0) + ' s' : (fell ? 'fell' : 'held on') + ' · feet up to ' + best.toFixed(2) + ' m';
  return { score, pass: topped ? 1 : 0, why, status, frames: recFrames, rungs: rungs.map((h) => ({ x0: +h.x0.toFixed(3), x1: +h.x1.toFixed(3), y: +h.y.toFixed(3) })), wall: { x0: wallX, top: wallH } };
}

function evaluate(v, base, n) {
  const p = decode(v);
  let s = 0, pass = 0;
  for (let i = 0; i < n; i++) { const r = episode(p, base + i); s += r.score; pass += r.pass; }
  return { score: s / n, pass };
}

if (args.includes('--show')) {
  const v = init ? JSON.parse(fs.readFileSync(init, 'utf8')).best : [];
  const p = decode(v);
  for (let i = 0; i < seeds; i++) { const r = episode(p, 100 + i); console.log('seed', i, r.why); }
  const rec = arg('--record', null);
  if (rec) {
    const shapes = world.parts.map((q) => ({ hx: +q.hx.toFixed(4), hy: +q.hy.toFixed(4), r: q.r ? +q.r.toFixed(4) : 0, ax: q.ax || 0, ay: q.ay || 0, capsule: !!q.capsule, circle: !!q.circle }));
    const eps = [];
    for (let i = 0; i < +arg('--n', 2); i++) {
      const r = episode(p, 100 + i, true);
      eps.push({ seed: i, boxes: [{ x0: r.wall.x0, x1: r.wall.x0 + 3, top: r.wall.top }], holds: r.rungs, box: null, walk: null, board: null, status: r.status, frames: r.frames });
    }
    fs.writeFileSync(rec, JSON.stringify({ skill: 6, fps: HZ / 2, shapes, episodes: eps }));
  }
  process.exit(0);
}

const rand = makeSeededRandom(+arg('--rng', 777));
const mean = new Float64Array(SPEC.length);
if (init) { const sv = JSON.parse(fs.readFileSync(init, 'utf8')).best; for (let d = 0; d < SPEC.length && d < sv.length; d++) mean[d] = sv[d]; }
const sigma = Float64Array.from(SPEC.map((q) => q[2]));
let best = null;
for (let it = 0; it < iters; it++) {
  const cands = [];
  for (let c = 0; c < pop; c++) {
    const v = Float64Array.from(mean, (m, d) => m + sigma[d] * randomNormalSample(rand));
    cands.push({ v, ...evaluate(v, 1000 + it * seeds, seeds) });
  }
  cands.sort((a, b) => b.score - a.score);
  if (it % 3 === 2 || it === 0) {
    for (const c of [cands[0], { v: Float64Array.from(mean) }]) {
      const val = evaluate(c.v, 500, 6);
      if (!best || val.score > best.score) best = { v: Float64Array.from(c.v), ...val };
    }
    fs.writeFileSync(out, JSON.stringify({ wallH, spacing, best: Array.from(best.v), params: decode(best.v), score: best.score, pass: best.pass }));
    console.log('iter', it, 'held-out best', best.score.toFixed(3), 'pass', best.pass + '/6', 'gen best', cands[0].score.toFixed(3));
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
