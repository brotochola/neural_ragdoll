// A slow, careful stepper on the feet alone (stepper.js tests it; the mountain planner uses it on the
// walkable parts). Each step:
//   SHIFT  both feet down; the hips carry the centre of mass over the front (standing) foot and it has to
//          settle there (within tol, nearly still) before anything lifts. An integral term removes the
//          steady offset the plain hip feedback leaves (the body stood 7 cm short of its foot).
//   SWING  the rear foot rises over the ground in between, travels past the standing foot and comes
//          down on its foothold; the hips stay over the standing foot.
// Balance during both: the pose executor's hip and ankle feedback around the standing foot.
const SOLE = 0.05;
const smooth = (s) => s * s * (3 - 2 * s);
const DEF = {
  hipH: 0.88, // hips above the standing sole
  stepLen: 0.32, // a foothold at most this far ahead of the standing foot
  stride: 0.6, // and the moving foot travels at most this far
  maxRise: 0.3, // nor this much higher than the standing foot
  maxSlope: 0.55, // nor on ground steeper than this (rad)
  footTilt: 0.3, // the sole follows the slope up to this much (the ankle turns only 0.6)
  clear: 0.1, // swing foot clears the ground in between by this much
  tShift: 0.8, // seconds to carry the hips over (then it waits for the body to settle)
  tSwing: 1.2, // seconds per swing
  tol: 0.04, // centre of mass within this of the standing foot, and
  vtol: 0.12, // slower than this, to lift the other foot
  kHip: 1.2, kHipVel: 0.2, kAnkle: 0.8, kAnkleVel: 0.4, kI: 1.5,
  lean: -0.05, leanK: 0.4, // trunk lean: forward a little, more on rising ground
  torsoGain: 0.4,
  ahead: 0.0, // hips this far ahead of the standing foot
  // HANDS (p.hands > 0): before a high step, both hands go onto the rock ahead and grip; they hold the
  // body while the foot goes up and the weight moves onto it, then let go
  hands: 0, handRise: 0.16, handExtra: 0.2, handAhead: 0.12, handReach: 0.56, tReach: 0.9, leash: 0.08,
};

export function makeStepper(ep, world, pIn, ctx) {
  const { config, pose, ground } = ctx;
  const p = Object.assign({}, DEF, pIn);
  const HZ = config.physics.hz, J = config.net.jointCount, P = config.partIndex;
  const FOOT = [P.footLeft, P.footRight];
  const slopeAt = (x) => Math.atan2(ground(x + 0.06) - ground(x - 0.06), 0.12);
  const groundMax = (a, b) => { let m = -1e9; for (let x = a; x <= b; x += 0.02) m = Math.max(m, ground(x)); return Math.max(m, ground(b)); };
  const plant = (k) => { const so = pose.soleOf(world, FOOT[k]); return { x: so.x, y: ground(so.x) + SOLE, angle: Math.max(-p.footTilt, Math.min(p.footTilt, slopeAt(so.x))) }; };
  const feet = [plant(0), plant(1)];
  const pv0 = pose.pelvisOf(world);
  const tmp = new Float64Array(J + config.net.gripOutputs);
  const st = { hands: [null, null], reach: null, feet, steps: 0, phase: 'shift', t: 0, stance: feet[0].x >= feet[1].x ? 0 : 1, ix: 0, settled: 0, from: null, target: null, pelvisFrom: { x: pv0.x, y: pv0.y }, stuck: 0 };

  function foothold(stance, moving, extra = 0) {
    let best = null;
    for (let x = stance.x + 0.06; x <= stance.x + p.stepLen + 1e-9; x += 0.02) {
      if (x - moving.x > p.stride) break;
      const y = ground(x) + SOLE;
      const a = slopeAt(x);
      const flat = Math.max(ground(x - 0.06), ground(x), ground(x + 0.06)) - Math.min(ground(x - 0.06), ground(x), ground(x + 0.06));
      if (y - stance.y > p.maxRise + extra || Math.abs(a) > p.maxSlope + extra || flat > 0.07) continue;
      best = { x, y, angle: Math.max(-p.footTilt, Math.min(p.footTilt, a)) };
    }
    return best;
  }
  function pelvisGoal() {
    const s = feet[st.stance], o = feet[1 - st.stance];
    const x = s.x + p.ahead;
    let y = s.y + p.hipH;
    // the rear foot must still reach while both are down
    if (st.phase === 'shift') { const dx = x - o.x; y = Math.min(y, o.y + Math.sqrt(Math.max(0, 0.9 * 0.9 - dx * dx))); }
    return { x, y };
  }
  st.step = function (e) {
    const m = e.measure;
    st.t += 1 / HZ;
    const S = st.stance, K = 1 - S;
    const fs = feet.map((f) => ({ x: f.x, y: f.y, angle: f.angle || 0 }));
    const planted = [true, true];
    const goalP = pelvisGoal();
    const err = (m.centerX || 0) - feet[S].x;
    const vel = m.centerVelocityX || 0;
    let pr = 1;
    if (st.phase === 'shift') {
      pr = smooth(Math.min(1, st.t / p.tShift));
      // TOE-OFF: the rear foot rolls onto its toes as the hips go forward (flat, its ankle ran out of
      // range and the hips stuck behind it, then lurched over)
      const roll = (p.toeRoll || 0) * pr;
      if (roll > 0) {
        const f = feet[K], a0 = f.angle || 0, half = 0.08;
        const toe = { x: f.x + Math.cos(a0) * half, y: f.y + Math.sin(a0) * half };
        const a = a0 - roll;
        fs[K] = { x: toe.x - Math.cos(a) * half, y: toe.y - Math.sin(a) * half, angle: a };
      }
      if (st.t > p.tShift * 0.5) st.ix = Math.max(-0.15, Math.min(0.15, st.ix + err / HZ));
      const held = st.hands[0] || st.hands[1];
      st.settled = (Math.abs(err) < p.tol || held) && Math.abs(vel) < p.vtol ? st.settled + 1 / HZ : 0;
      if (st.t > p.tShift && st.settled > 0.15) {
        let target = foothold(feet[S], feet[K]);
        let assist = false;
        if (p.hands > 0) {
          const hi = foothold(feet[S], feet[K], p.handExtra);
          if (hi && (!target || hi.y - feet[S].y > p.handRise)) { target = hi; assist = hi.y - feet[S].y > p.handRise || hi.x > target.x + 0.05; }
        }
        if (held && !assist) {
          // walkable again: let go first
          st.hands = [null, null];
          st.t = p.tShift * 0.8; st.settled = 0;
          target = null;
        } else if (assist && !held) {
          // hands onto the rock just past the foothold, as far as the shoulders reach
          st.phase = 'reach'; st.t = 0; st.target = target;
          const pv = pose.pelvisOf(world);
          const sh = { x: pv.x - Math.sin(pv.angle) * 0.52, y: pv.y + Math.cos(pv.angle) * 0.52 };
          st.reach = [0, 1].map((k) => {
            let best = null;
            for (let x = target.x - 0.1; x <= target.x + p.handAhead + 0.3; x += 0.02) {
              const y = ground(x) + 0.03;
              const d = Math.hypot(x - sh.x, y - sh.y);
              if (d > p.handReach) continue;
              const want = Math.abs(x - (target.x + p.handAhead + 0.04 * k));
              if (!best || want < best.want) best = { x, y, want };
            }
            const hp = pose.handOf(world, k ? P.handRight : P.handLeft);
            return best ? { from: { x: hp.x, y: hp.y }, to: best } : null;
          });
          if (!st.reach[0] && !st.reach[1]) { st.phase = 'shift'; st.stuck += 0.5; if (!foothold(feet[S], feet[K])) target = null; else { target = foothold(feet[S], feet[K]); assist = false; } }
          else { if (ctx.debug) console.log('  t', (e.frame / HZ).toFixed(1), 'reach', st.reach.map((r) => r ? r.to.x.toFixed(2) + ',' + r.to.y.toFixed(2) : '-').join(' '), 'for', target.x.toFixed(2), target.y.toFixed(2)); target = null; }
        }
        if (target && st.phase === 'shift') { st.phase = 'swing'; st.t = 0; st.from = { ...feet[K] }; st.target = target; if (ctx.debug) console.log('  t', (e.frame / HZ).toFixed(1), 'swing', K, '->', target.x.toFixed(2), target.y.toFixed(2), 'stance', feet[S].x.toFixed(2), feet[S].y.toFixed(2)); }
        else st.stuck += 1 / HZ;
      }
    } else if (st.phase === 'reach') {
      const s2 = Math.min(1, st.t / p.tReach);
      for (let k = 0; k < 2; k++) {
        const r = st.reach[k];
        if (!r || st.hands[k]) continue;
        const a = r.from, b = r.to;
        const mid = { x: (a.x + b.x) / 2, y: Math.max(a.y, b.y) + 0.15 };
        const q = smooth(s2);
        const u = q < 0.5 ? q * 2 : (q - 0.5) * 2;
        const P0 = q < 0.5 ? a : mid, P1 = q < 0.5 ? mid : { x: b.x, y: b.y };
        r.now = { x: P0.x + (P1.x - P0.x) * u, y: P0.y + (P1.y - P0.y) * u };
        const g = e.grips && e.grips[k];
        if (g && g.joint) { const hp = pose.handOf(world, k ? P.handRight : P.handLeft); st.hands[k] = { x: hp.x, y: hp.y }; }
      }
      const want = st.reach.filter((r) => r).length;
      const got = st.hands.filter((h) => h).length;
      if (got >= want || (st.t > p.tReach + 0.6 && got > 0)) { st.phase = 'swing'; st.t = 0; st.from = { ...feet[K] }; if (ctx.debug) console.log('  t', (e.frame / HZ).toFixed(1), 'gripped', st.hands.map((h) => h ? h.x.toFixed(2) + ',' + h.y.toFixed(2) : '-').join(' ')); }
      else if (st.t > p.tReach + 1.2) { st.phase = 'shift'; st.t = p.tShift; st.settled = 0; st.stuck += 1; st.hands = [null, null]; }
    } else if (st.phase === 'swing') {
      st.ix = Math.max(-0.15, Math.min(0.15, st.ix + err / HZ));
      const s = Math.min(1, st.t / p.tSwing);
      const a = st.from, b = st.target;
      const top = Math.max(groundMax(Math.min(a.x, b.x), Math.max(a.x, b.x)) + SOLE, a.y, b.y) + p.clear;
      const pts = [a, { x: a.x - 0.02, y: top }, { x: b.x, y: top }, b];
      const cuts = [0, 0.3, 0.75, 1];
      let si = 0;
      while (si < 2 && s > cuts[si + 1]) si++;
      const q = smooth(Math.min(1, (s - cuts[si]) / (cuts[si + 1] - cuts[si])));
      fs[K] = { x: pts[si].x + (pts[si + 1].x - pts[si].x) * q, y: pts[si].y + (pts[si + 1].y - pts[si].y) * q, angle: si === 2 ? b.angle * q : (si === 0 ? a.angle * (1 - q) : 0) };
      planted[K] = false;
      if (s >= 1) {
        const so = pose.soleOf(world, FOOT[K]);
        const down = e.measure.touch[FOOT[K]];
        if ((down && Math.abs(so.x - b.x) < 0.1) || st.t > p.tSwing + 0.5) {
          feet[K] = plant(K);
          st.steps++;
          st.phase = 'shift'; st.t = 0; st.stance = K; st.ix = 0; st.settled = 0;
          st.pelvisFrom = { ...pose.pelvisOf(world) };
          if (ctx.debug) console.log('  t', (e.frame / HZ).toFixed(1), 'landed', K, feet[K].x.toFixed(2), feet[K].y.toFixed(2), 'wanted', b.x.toFixed(2), b.y.toFixed(2));
        }
      }
    }
    let pel;
    if (p.pelRate > 0) {
      // the commanded hips move toward their goal no faster than pelRate (m/s): when the rear foot left
      // a step the goal jumped 20 cm up at once and the kick threw the body back
      if (!st.pc) st.pc = { ...st.pelvisFrom };
      const dx = goalP.x - st.pc.x, dy = goalP.y - st.pc.y, d = Math.hypot(dx, dy), mx = p.pelRate / HZ;
      if (d > mx) { st.pc.x += dx * mx / d; st.pc.y += dy * mx / d; } else { st.pc.x = goalP.x; st.pc.y = goalP.y; }
      pel = { ...st.pc };
    } else pel = st.phase === 'shift' ? { x: st.pelvisFrom.x + (goalP.x - st.pelvisFrom.x) * pr, y: st.pelvisFrom.y + (goalP.y - st.pelvisFrom.y) * pr } : goalP;
    pel.x -= p.kI * st.ix;
    const slope = Math.atan2(ground(feet[S].x + 0.4) - ground(feet[S].x - 0.2), 0.6);
    const lean = p.lean - p.leanK * Math.max(0, slope);
    // ARMS as a counterweight (p.armK): with nothing to hold they reach forward when the body sags back
    // (and back when it pitches forward), at armFwd rad from hanging plus armK per metre of error
    let armHand = null;
    if (p.armK || p.armFwd) {
      const pv = pose.pelvisOf(world);
      const sh = { x: pv.x - Math.sin(pv.angle) * 0.52, y: pv.y + Math.cos(pv.angle) * 0.52 };
      const ang = Math.max(-0.8, Math.min(2.4, (p.armFwd || 0) - (p.armK || 0) * err - (p.armKv || 0) * vel));
      armHand = { x: sh.x + Math.sin(ang) * 0.5, y: sh.y - Math.cos(ang) * 0.5 };
    }
    const hands = st.hands.map((h, k) => h || (st.phase === 'reach' && st.reach[k] && st.reach[k].now) || armHand);
    const gripping = st.hands.map((h) => !!h);
    if (gripping[0] || gripping[1]) {
      // with hands on the rock the loop feet-hands is closed: lead the pelvis on a leash so the limbs agree
      const pv = pose.pelvisOf(world);
      const dx = pel.x - pv.x, dy = pel.y - pv.y, d = Math.hypot(dx, dy);
      if (d > p.leash) { pel.x = pv.x + dx * p.leash / d; pel.y = pv.y + dy * p.leash / d; }
    }
    const goal = { pelvis: pel, torso: lean - (st.phase === 'reach' || gripping[0] || gripping[1] ? p.handLean || 0.3 : 0), feet: fs, hands, gripping, planted, torsoGain: p.torsoGain };
    const actual = pose.actualOf(world);
    const jt = pose.balancedTargets(goal, m, { planted, support: feet[S].x, kHip: p.kHip, kAnkle: p.kAnkle, kAnkleVel: p.kAnkleVel, kHipVel: p.kHipVel }, actual);
    pose.commandsFor(world, jt, tmp);
    for (let j = 0; j < J; j++) e.action[j] = tmp[j];
    for (let j = J; j < tmp.length; j++) e.action[j] = -1;
    for (let k = 0; k < 2; k++) {
      let on = gripping[k];
      if (!on && st.phase === 'reach' && st.reach[k] && st.t > p.tReach * 0.4) { const hp = pose.handOf(world, k ? P.handRight : P.handLeft); on = Math.hypot(hp.x - st.reach[k].to.x, hp.y - st.reach[k].to.y) < 0.15; }
      e.action[J + k] = on ? 1 : -1;
    }
    if (ctx.debug && ctx.debug > 1 && e.frame % 6 === 0) { const pv = pose.pelvisOf(world); console.log('   f', (e.frame / HZ).toFixed(2), st.phase, 'S', S, 'err', err.toFixed(2), 'vel', vel.toFixed(2), 'ix', st.ix.toFixed(2), 'pel', pel.x.toFixed(2), pel.y.toFixed(2), 'act', pv.x.toFixed(2), pv.y.toFixed(2), 'torso', pv.angle.toFixed(2)); }
  };
  return st;
}
