// The wall-climbing planner (climb-wall.js): three points on the wall, the fourth moves one rung up, then over
// the top. Installs itself on an episode (wraps ep.updateGrip). Shared by the trainer and the playground.
const ENV = globalThis.process?.env || {};
const smooth = (s) => s * s * (3 - 2 * s);

export function installWallPlanner(ep, world, p, ctx) {
  const { config, pose, wallH } = ctx;
  const HZ = config.physics.hz, J = config.net.jointCount, OUT = J + config.net.gripOutputs, P = config.partIndex;
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
  for (let i = 1; i <= rungs.length; i++) if (ry(i) < +(ENV.REACH_Y || 1.55)) reach = i; // about shoulder height: an arm raised overhead next to the wall hit the rungs
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
    const jt = floorOnly && ENV.FLOOR_BAL ? pose.balancedTargets(goal, m, { planted }, pose.actualOf(world)) : pose.jointTargets(goal, pose.actualOf(world));
    pose.commandsFor(world, jt, tmp);
    if (ENV.DEBUG_LEG && move.kind === 'foot' && Math.round(move.t * HZ) % 10 === 0) {
      const L = move.k === 0 ? [5, 6, 7] : [11, 12, 13];
      const so = pose.soleOf(world, move.k === 0 ? P.footLeft : P.footRight);
      const pv = pose.pelvisOf(world);
      const S2 = move.k === 0 ? [11, 12, 13] : [5, 6, 7];
      const ang = (j) => ctx.Box2D.b2RevoluteJoint_GetAngle(world.joints[j].id);
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
        else if ((move.wait += 1 / HZ) > 0.6) { if (ENV.DEBUG_CW) { const hp = pose.handOf(world, move.k === 0 ? P.handLeft : P.handRight); console.log('  grip miss hand', move.k, 'at', hp.x.toFixed(2), hp.y.toFixed(2), 'target', move.target.x.toFixed(2), move.target.y.toFixed(2)); } move = null; } // retry the same move
      } else if (move.kind === 'foot') {
        // standing on the rung? (sole within 6 cm of its top and over it); else try again
        const so = pose.soleOf(world, move.k === 0 ? P.footLeft : P.footRight);
        const tgt = move.target;
        if (Math.abs(so.y - tgt.y) < 0.07 && Math.abs(so.x - tgt.x) < 0.12) { lim.foot[move.k] = { rung: move.to, pos: tgt }; move = null; }
        else if ((move.wait += 1 / HZ) > 0.5) {
          if (ENV.DEBUG_CW) console.log('  foot miss', move.k, 'at', so.x.toFixed(2), so.y.toFixed(2), 'target', tgt.x.toFixed(2), tgt.y.toFixed(2));
          move = null;
        }
      }
    }
    orig();
  })(ep.updateGrip.bind(ep));
  return { lim, feetTop, handsTop, rungs, wallX, N, TOP };
}
