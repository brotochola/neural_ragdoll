// RUNNING controller with a flight phase, for the 2D ragdoll (parkour: run, jump gaps, jump boxes).
// In the spirit of Raibert's hoppers and SIMBICON (src/simbicon.js), driven by contacts, not a timer:
//   STANCE  the stance foot is down. Its hip holds the torso lean plus a push (thigh driven back), its
//           knee yields to kneeLoad, then once the centre of mass passes the foot the knee and ankle
//           extend (pushKnee, pushAnkle): the push-off that makes the flight. The other leg swings
//           through fast, knee tucked.
//   FLIGHT  the stance foot has left the ground. That leg trails (knee tucked); the swing leg reaches
//           to a WORLD-frame landing angle place + cd*d + cv*(v - vTarget) (Raibert foot placement).
//   The swing foot touching down (after tMin) makes it the stance leg.
// A JUMP (p.jump set by the caller for one stride) uses the jump* parameters for that push and flight:
// a harder push, both knees tucked in the air (to clear a box), the landing leg reaching further.
import { config } from "../config.js";

export const SPEC = [
  ['lean', -0.25, 0.1], // torso world angle (negative = forward)
  ['hipPush', 0.3, 0.2], // extra stance-hip extension during the push
  ['kneeLoad', -0.5, 0.2], // stance knee while loading
  ['pushKnee', -0.05, 0.15], // stance knee during the push
  ['pushAnkle', 0.5, 0.25], // stance foot angle during the push (plantarflex)
  ['pushD', 0.0, 0.08], // push starts when the centre of mass is this far past the stance foot
  ['swingHip', 1.0, 0.3], // swing thigh world angle while the other foot is down
  ['tuck', -2.0, 0.3], // swing knee while swinging through
  ['place', 0.35, 0.15], // landing thigh world angle in flight
  ['cd', 0.3, 0.3],
  ['cv', 0.15, 0.1],
  ['reachKnee', -0.4, 0.2], // landing knee
  ['trailHip', -0.3, 0.3], // trailing thigh world angle in flight
  ['trailKnee', -1.6, 0.4],
  ['tMin', 0.12, 0.05], // shortest stance-to-touchdown time before a swap
  ['arm', 0.6, 0.3],
  ['elbow', 1.2, 0.4],
  // jump stride
  ['jumpHipPush', 0.6, 0.3],
  ['jumpKnee', 0.0, 0.1],
  ['jumpAnkle', 0.6, 0.25],
  ['jumpLean', -0.2, 0.15],
  ['jumpTuck', -2.2, 0.3], // both knees in the air
  ['jumpHip', 1.4, 0.3], // leading thigh in the air
  ['jumpPlace', 0.5, 0.2], // landing thigh as it comes down
  ['jumpArm', 2.0, 0.6], // arms thrown up and forward at take-off
];

export function decode(v) {
  const p = {};
  SPEC.forEach(([name, base], i) => (p[name] = base + (v[i] || 0)));
  p.tMin = Math.max(0.05, Math.min(0.4, p.tMin));
  return p;
}

export function makeController(world) {
  const J = config.net.jointCount;
  const P = config.partIndex;
  const LEG = [
    { thigh: 6, shin: 7, foot: P.footLeft, hip: 5, knee: 6, ankle: 7, shoulder: 2, elbow: 3 },
    { thigh: 12, shin: 13, foot: P.footRight, hip: 11, knee: 12, ankle: 13, shoulder: 8, elbow: 9 },
  ];
  const angleOf = (i) => Math.atan2(world.s[i], world.c[i]);
  function command(jointIndex, target) {
    const joint = world.joints[jointIndex];
    const t = Math.max(joint.lo, Math.min(joint.hi, target));
    return t >= 0 ? (joint.hi > 0 ? t / joint.hi : 0) : joint.lo < 0 ? t / -joint.lo : 0;
  }
  // ep.runner = { stance, t, flight, jumping }; the caller may set ep.runner.jumpNext = true to make the
  // next push a jump (it applies from the next touchdown's push through the following landing).
  return function runner(p, ep, action) {
    const st = ep.runner || (ep.runner = { stance: 1, t: 0, flight: false, jumping: false, jumpNext: false, strides: 0 });
    const dt = 1 / config.physics.hz;
    st.t += dt;
    const m = ep.measure;
    let stance = LEG[st.stance];
    let swing = LEG[1 - st.stance];
    const down = (leg) => !!m.touch[leg.foot];
    if (!st.flight && !down(stance) && st.t > 0.05) st.flight = true;
    if (st.t > p.tMin && down(swing)) {
      // touchdown: swap
      st.stance = 1 - st.stance;
      st.t = 0;
      st.flight = false;
      st.strides++;
      st.jumping = !!st.jumpNext;
      st.jumpNext = false;
      [stance, swing] = [swing, stance];
    }
    const J2 = st.jumping;
    const torso = angleOf(0);
    const d = (m.centerX || 0) - world.px[stance.foot];
    const v = (m.centerVelocityX || 0) - (m.walkSpeed || 0);
    const lean = J2 ? p.jumpLean : p.lean;
    if (!st.flight) {
      const push = d > p.pushD;
      const stanceHip = angleOf(stance.thigh) - torso;
      action[stance.hip] = command(stance.hip, stanceHip + (torso - lean) - (push ? (J2 ? p.jumpHipPush : p.hipPush) : 0));
      action[stance.knee] = command(stance.knee, push ? (J2 ? p.jumpKnee : p.pushKnee) : p.kneeLoad);
      action[stance.ankle] = command(stance.ankle, (push ? (J2 ? p.jumpAnkle : p.pushAnkle) : 0) - angleOf(stance.shin));
      action[swing.hip] = command(swing.hip, (J2 ? p.jumpHip : p.swingHip) - torso);
      action[swing.knee] = command(swing.knee, J2 ? p.jumpTuck : p.tuck);
      action[swing.ankle] = command(swing.ankle, 0.1 - angleOf(swing.shin));
    } else {
      // flight: the leg that pushed trails, the other reaches for the landing
      const placeW = (J2 ? p.jumpPlace : p.place) + p.cd * d + p.cv * v;
      action[swing.hip] = command(swing.hip, placeW - torso);
      // in a jump the landing knee stays tucked while rising, extends as the body comes down
      const falling = (m.centerVelocityY || 0) < 0;
      action[swing.knee] = command(swing.knee, J2 && !falling ? p.jumpTuck : p.reachKnee);
      action[swing.ankle] = command(swing.ankle, 0.0 - angleOf(swing.shin));
      action[stance.hip] = command(stance.hip, (J2 ? p.jumpHip * 0.5 : p.trailHip) - torso);
      action[stance.knee] = command(stance.knee, J2 ? p.jumpTuck : p.trailKnee);
      action[stance.ankle] = command(stance.ankle, 0.1 - angleOf(stance.shin));
    }
    action[0] = 0;
    action[1] = 0;
    // arms against the legs; thrown up at a jump
    if (J2 && (st.flight || d > p.pushD)) {
      for (const leg of LEG) action[leg.shoulder] = command(leg.shoulder, p.jumpArm - angleOf(1));
    } else {
      action[swing.shoulder] = command(swing.shoulder, -p.arm - torso);
      action[stance.shoulder] = command(stance.shoulder, p.arm - torso);
    }
    action[swing.elbow] = command(swing.elbow, p.elbow);
    action[stance.elbow] = command(stance.elbow, p.elbow);
    action[4] = 0;
    action[10] = 0;
    for (let j = J; j < action.length; j++) action[j] = -1;
  };
}
