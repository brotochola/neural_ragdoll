// OBSTACLE RUNNER (obstacle.js, the playground): running into a tall rock and over it, by chaining
// earlier skills: the parkour runner slows down, the SIMBICON walker walks up and the stop controller
// stops a set distance from the rock, the pose executor holds still, the crag planner climbs (knobs,
// belly mantle, the get-up expert on top), then the walker walks off and the runner runs on.
//   modes: run -> walk -> stop -> hold -> climb -> walk2 -> run2
// LEAP (ctx.leapP set): it runs at the rock without slowing, jumps on the stride nearest leapP.takeoff from
// the face, reaches in the air for the highest knob under leapP.reachH and grabs it (a cat leap), hangs, and
// the crag planner climbs on from there:  run -> leap -> hang -> climb -> walk2 -> run2. Missing the knob
// (feet back on the ground with no hand on), it falls back to walk -> stop -> hold -> climb.
import * as simb from './simbicon.js';
import * as runMod from './runner.js';
import { installCragPlanner } from './crag-planner.js';

export function installObstacleRunner(ep, world, ctx) {
  const { config, pose, Box2D, ElmanNetwork, crag, H } = ctx;
  const { runP, walkP, slowP, stopP, cragP, getup } = ctx.params;
  const st = ctx.params.stopK;
  const HZ = config.physics.hz, J = config.net.jointCount, P = config.partIndex;
  // ctx.surface: this wall's own top line (absolute), when it is not the terrain's (a pit's far wall)
  const S = ctx.surface || ep.terrain.surface;
  const face = S[1][0];
  const stopAt = ctx.stopAt != null ? ctx.stopAt : 0.6, DECEL = ctx.decel || 8, VWALK = ctx.vwalk || 1.1, VSWITCH = ctx.vswitch || 1.3;
  const speedOf = () => ctx.speed();
  const walker = simb.makeController(world), runner = runMod.makeController(world);
  const tmp = new Float64Array(J + 2);
  let mode = 'run', modeT = 0, PL = null, holdFeet = null, climbBest = -1, climbT = 0;
  const LP = ctx.leapP || null;
  // the knobs, absolute (shifted like the terrain), and the one the leap aims at: the highest under reachH
  const kdx = S[1][0] - crag.surface[1][0];
  const knobs = crag.knobs.map((k) => ({ lip: k.lip + kdx, back: k.back + kdx, y: k.y }));
  let aimI = 0;
  if (LP) for (let i = 0; i < knobs.length; i++) if (knobs[i].y <= LP.reachH) aimI = i + 1;
  if (LP && ctx.aimTop) aimI = knobs.length; // falling into a pit: the highest ledge, just under the edge
  const aimAt = () => { const k = knobs[aimI - 1]; return { x: k.lip + 0.03 + LP.aimX, y: k.y + 0.03 + LP.aimY }; };
  const leap = { aimI, caughtY: null, caughtAt: null, missed: false, tookOff: null };
  const angleOf = (i) => Math.atan2(world.s[i], world.c[i]);
  const command = (j, target) => { const jt = world.joints[j]; const t = Math.max(jt.lo, Math.min(jt.hi, target)); return t >= 0 ? (jt.hi > 0 ? t / jt.hi : 0) : jt.lo < 0 ? t / -jt.lo : 0; };
  const HAND = [P.handLeft, P.handRight];
  // both arms reach for the aim point (two-link IK from each shoulder), a hand within catchR closes on it
  const reachArms = (e, a, out, only) => {
    const ca = angleOf(1);
    const shW = { x: world.px[1] - Math.sin(ca) * 0.10, y: world.py[1] + Math.cos(ca) * 0.10 };
    const ik = pose.twoLink(shW, a, 0.28, 0.29, 1);
    for (const [k, sh, el] of [[0, 2, 3], [1, 8, 9]]) {
      if (only != null && k !== only) continue;
      out[sh] = command(sh, ik.upper - ca); out[el] = command(el, ik.middle);
      const d = Math.hypot(world.px[HAND[k]] - a.x, world.py[HAND[k]] - a.y);
      out[J + k] = d < LP.catchR ? 1 : -1; // closes on whatever rock it touches within catchR of the aim
    }
  };
  const log = [];
  const setMode = (m2) => { if (m2 !== mode) { log.push((ep.frame / HZ).toFixed(1) + 's ' + m2 + (ctx.debug ? '@' + (face - Math.max(world.px[P.footLeft], world.px[P.footRight])).toFixed(2) : '')); mode = m2; modeT = 0; } };
  ep.updateGrip = ((orig) => function () {
    // ctx.active(): a director (src/director.js) may switch this skill off; off, it leaves the motors alone
    if (ctx.active && !ctx.active()) { orig(); return; }
    const m = this.measure;
    modeT += 1 / HZ;
    const front = Math.max(world.px[P.footLeft], world.px[P.footRight]);
    // only the runner uses the fast joints (20 rad/s); everything else was tuned at 10 (the get-up expert
    // at 20 rolled right over itself)
    this.speedCap = mode === 'run' || mode === 'run2' || mode === 'leap' ? config.body.maxJointSpeed : 0;
    if (mode === 'run' || mode === 'run2') {
      // slow down ahead of the rock, then walk
      const dist = face - front;
      m.walkSpeed = mode === 'run2' ? Math.min(speedOf(), 1.2 + modeT * 0.6) : LP && aimI ? LP.vRun : dist < DECEL ? Math.max(VWALK, speedOf() * dist / DECEL) : speedOf();
      this.speedCap = config.body.maxJointSpeed;
      runner(runP, this, tmp);
      const rs = this.runner;
      if (mode === 'run' && LP && aimI && rs && rs.t === 1 / HZ) {
        // jump on the touchdown whose distance to the face is nearest takeoff (within half a stride of it)
        const fx = world.px[rs.stance ? P.footRight : P.footLeft];
        if (rs.lastX != null) { const L = fx - rs.lastX; if (L > 0.2 && L < 2.5) rs.avgL = rs.avgL ? 0.7 * rs.avgL + 0.3 * L : L; }
        rs.lastX = fx;
        const L = rs.avgL || 1.2, d = face - fx;
        rs.placeAdj = 0;
        // (or if the next ordinary step would already reach the rock: a long stride skipped the take-off zone)
        if (d > 0.15 && (d < LP.takeoff + L / 2 || d - L < 0.25)) { rs.jumping = true; rs.jumpScale = LP.jumpScale; leap.tookOff = ep.frame; }
        else if (rs.avgL && LP.kStride) {
          // stretch or shorten the steps so a foot comes down at the take-off spot (as the jump selector does)
          const D = d - LP.takeoff;
          if (D > 0 && D < 6) { const n = Math.max(1, Math.round(D / L)); rs.placeAdj = Math.max(-0.35, Math.min(0.35, LP.kStride * (D / n - L))); }
        }
      }
      // the arms go up toward the knob already in the last strides
      // and on the push of the jump stride itself (raising them earlier threw the running gait off)
      if (mode === 'run' && LP && aimI && (face - front < LP.preReach || (rs && rs.jumping))) { reachArms(this, aimAt(), tmp); for (let k = 0; k < 2; k++) tmp[J + k] = -1; }
      if (mode === 'run' && LP && aimI && rs && rs.jumping && rs.flight && leap.tookOff != null) setMode('leap');
      if (mode === 'run' && LP && aimI) { /* no slowing down: the leap replaces walk and stop */ }
      else if (mode === 'run' && dist < 5 && (m.centerVelocityX || 0) < VSWITCH && this.runner.t < 1.5 / HZ && !this.runner.flight) { setMode('walk'); this.simbicon = { stance: this.runner.stance, t: 0 }; }
    } else if (mode === 'leap') {
      // in the air: the runner's flight for the trunk, the legs tucked up, both arms at the knob
      m.walkSpeed = LP.vRun;
      runner(runP, this, tmp);
      const onFeet = m.touch[P.footLeft] || m.touch[P.footRight];
      if (onFeet && modeT > 0.15) leap.landed = true;
      // legs tucked in the air; once down again they stand tall on the toes, the arms still reaching up
      if (!leap.landed) for (const [h, kn] of [[5, 6], [11, 12]]) { tmp[h] = command(h, LP.legHip); tmp[kn] = command(kn, LP.legKnee); }
      else for (const [h, kn, an] of [[5, 6, 7], [11, 12, 13]]) { tmp[h] = command(h, 0); tmp[kn] = command(kn, 0); tmp[an] = command(an, LP.toe || 0.4); }
      tmp[0] = command(0, LP.leapWaist);
      reachArms(this, aimAt(), tmp);
      { const a = aimAt(); for (const q of HAND) leap.minD = Math.min(leap.minD == null ? 9 : leap.minD, Math.hypot(world.px[q] - a.x, world.py[q] - a.y)); }
      if (ctx.debug && ep.frame % 3 === 0) { const a = aimAt(); console.log('  leap t', modeT.toFixed(2), 'grip out', tmp[J].toFixed(0), tmp[J + 1].toFixed(0), 'shR cmd', tmp[8].toFixed(2), 'ang', Box2D.b2RevoluteJoint_GetAngle(world.joints[8].id).toFixed(2), 'gripsJ', this.grips[0].joint ? 1 : 0, this.grips[1].joint ? 1 : 0, 'touch hands', m.touch[P.handLeft] ? 1 : 0, m.touch[P.handRight] ? 1 : 0, 'feet', m.touch[P.footLeft] ? 1 : 0, m.touch[P.footRight] ? 1 : 0, 'aim', a.x.toFixed(2), a.y.toFixed(2), 'handR', world.px[HAND[1]].toFixed(2), world.py[HAND[1]].toFixed(2), 'pelvis', world.px[0].toFixed(2), world.py[0].toFixed(2), 'face', face.toFixed(2), 'vx', (m.centerVelocityX || 0).toFixed(2)); }
      const held = [0, 1].map((k) => !!(this.grips[k] && this.grips[k].joint));
      if (held[0] || held[1]) {
        const k = held[0] ? 0 : 1; leap.caughtY = world.py[HAND[k]]; leap.caughtAt = ep.frame; leap.first = k;
        // the knob the planner will count the hands on: the highest one at or under the hand that caught
        leap.holdI = 0; for (let i = 0; i < knobs.length; i++) if (knobs[i].y <= leap.caughtY + 0.05) leap.holdI = i + 1;
        if (!leap.holdI) leap.holdI = 1;
        if (ctx.aimTop && leap.caughtY > H - 0.04) leap.holdI = knobs.length + 1; // caught the top edge itself
        setMode('hang');
      }
      else if (modeT > (LP.reachT || 0.6) && (m.touch[P.footLeft] || m.touch[P.footRight])) {
        // missed: back on the ground with no hand on: the old way (walk up, stop, climb from standing)
        leap.missed = true; setMode('walk'); this.simbicon = { stance: this.runner.stance, t: 0 };
      }
    } else if (mode === 'hang') {
      // hanging: the holding hand keeps its arm, the other one reaches the same knob and closes; knees and hips
      // bend to bring the feet to the rock; then the crag planner climbs on from here
      const held = [0, 1].map((k) => !!(this.grips[k] && this.grips[k].joint));
      for (let j = 0; j < tmp.length; j++) tmp[j] = j < J ? this.action[j] : 1;
      const h0 = HAND[leap.first];
      const a = { x: world.px[h0] + 0.02, y: world.py[h0] - 0.02 };
      for (let k = 0; k < 2; k++) if (!held[k]) reachArms(this, a, tmp, k); else tmp[J + k] = 1;
      for (const [h, kn, an] of [[5, 6, 7], [11, 12, 13]]) { tmp[h] = command(h, LP.hangHip); tmp[kn] = command(kn, LP.hangKnee); tmp[an] = command(an, 0); }
      tmp[0] = command(0, 0);
      if (!held[0] && !held[1]) { leap.missed = true; setMode('walk'); this.simbicon = { stance: 0, t: 0 }; }
      else if (modeT > LP.hangT) {
        setMode('climb');
        const hp = HAND.map((q) => ({ x: world.px[q], y: world.py[q] }));
        PL = installCragPlanner(ep, world, cragP, { config, pose, Box2D, crag, surface: ctx.surface, start: { hands: [leap.holdI, leap.holdI], handPos: hp }, nets: { ElmanNetwork, getupW: getup.theta, handover: getup.handover } });
      }
    } else if (mode === 'walk') {
      const dist = face - front;
      m.walkSpeed = Math.max(0.35, Math.min(1.0, dist / 2.5)); m.walkDir = 1;
      walker(dist > 1.5 ? walkP : slowP, this, tmp);
      if (dist < stopAt + 0.25) setMode('stop');
    } else if (mode === 'stop') {
      m.walkSpeed = 0; m.walkDir = 1;
      walker(stopP, this, tmp);
      if (modeT > 2) setMode('hold');
    } else if (mode === 'hold') {
      if (!holdFeet) holdFeet = [P.footLeft, P.footRight].map((q) => pose.soleOf(world, q));
      const mid = (holdFeet[0].x + holdFeet[1].x) / 2;
      const goal = { pelvis: { x: mid, y: 0.92 }, torso: 0, feet: holdFeet.map((q) => ({ x: q.x, y: q.y, angle: 0 })), planted: [true, true] };
      pose.commandsFor(world, pose.balancedTargets(goal, m, { planted: [true, true], kHip: st.holdKHip, kAnkle: st.holdKAnkle, kAnkleVel: 0.15, kHipVel: 0.1 }), tmp);
      for (let k = J; k < tmp.length; k++) tmp[k] = -1;
      if (modeT > 1.5) {
        setMode('climb');
        // the crag planner takes over (it wraps updateGrip from outside; this wrapper then only passes on)
        PL = installCragPlanner(ep, world, cragP, { config, pose, Box2D, crag, surface: ctx.surface, nets: { ElmanNetwork, getupW: getup.theta, handover: getup.handover } });
      }
    } else if (mode === 'drop') {
      // gave up a climb from a leap: let go, legs under the body; once down, stand and climb from standing
      for (let j = 0; j < J; j++) tmp[j] = 0;
      // knees a little bent and the hips a little forward to take the landing, arms down at the sides
      for (const [h, kn, an] of [[5, 6, 7], [11, 12, 13]]) { tmp[h] = command(h, 0.35); tmp[kn] = command(kn, -0.6); tmp[an] = command(an, 0.25); }
      for (const sh of [2, 8]) tmp[sh] = command(sh, -angleOf(1));
      for (let k = J; k < tmp.length; k++) tmp[k] = -1;
      if (modeT > 0.8 && (m.touch[P.footLeft] || m.touch[P.footRight])) { holdFeet = null; setMode('hold'); }
    } else if (mode === 'climb') {
      this.speedCap = 0;
      // no limb has gone higher for 25 s (and not over the edge yet): from a leap, drop and climb from standing
      if (PL && !PL.phase) {
        const prog = PL.lim.hand[0].rung + PL.lim.hand[1].rung + PL.lim.foot[0].rung + PL.lim.foot[1].rung;
        if (prog > climbBest) { climbBest = prog; climbT = 0; } else climbT += 1 / HZ;
        if (climbT > 25 && LP && !leap.dropped) { leap.dropped = true; PL.handOff(); PL = null; climbBest = -1; climbT = 0; setMode('drop'); }
      }
      if (mode === 'drop') { orig(); return; }
      if (PL.stood && PL.phase === 'stand' && modeT > 1) {
        // standing on top for a moment: walk off, then run
        if (!this.standT) this.standT = 0;
        this.standT += 1 / HZ;
        if (this.standT > 1.5) { PL.handOff(); setMode('walk2'); this.simbicon = { stance: 0, t: 0 }; }
      }
      orig();
      return;
    } else if (mode === 'walk2') {
      m.walkSpeed = Math.min(1.2, 0.3 + modeT * 0.5); m.walkDir = 1;
      walker(walkP, this, tmp);
      const s = this.simbicon;
      if (modeT > 3 && s && s.t < 1.5 / HZ) { setMode('run2'); this.runner = { stance: s.stance, t: 0, flight: false, jumping: false, jumpNext: false, strides: 0 }; }
    }
    // the hands stay open except in the leap and while hanging (the reach closes them on the rock)
    const gripping = mode === 'leap' || mode === 'hang' || (mode === 'run' && false);
    for (let j = 0; j < tmp.length; j++) this.action[j] = j < J || gripping ? tmp[j] : -1;
    orig();
  })(ep.updateGrip.bind(ep));
  // back to the start of the approach (after a get-up, src/recover.js): run at the rock and leap again
  // walkUp: got up near the rock, too close for a run-up: walk up to it and climb from standing (the old way)
  const reset = (walkUp) => { mode = walkUp ? 'walk' : 'run'; modeT = 0; holdFeet = null; Object.assign(leap, { caughtY: null, caughtAt: null, missed: !!walkUp, tookOff: null, landed: false, minD: null }); };
  // falling into a pit (the director's call): straight into the leap's reach for the far wall's top ledge
  const leapNow = () => { mode = 'leap'; modeT = 0; log.push((ep.frame / HZ).toFixed(1) + 's fall-catch'); Object.assign(leap, { tookOff: ep.frame, landed: false, missed: false, caughtY: null }); };
  return { get mode() { return mode; }, get planner() { return PL; }, log, leap, reset, leapNow };
}
