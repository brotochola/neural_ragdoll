// SIMBICON-style walking controller (Yin, Loken & van de Panne, 2007) for the 2D ragdoll. Shared by
// walk-simbicon.js (parameter search) and bc-walk.js (the walk expert copies it).
// Stance on one leg for T seconds (until the swing foot lands), then swap. Swing hip: a WORLD-frame
// target theta + cd*d + cv*(v - vTarget), d = centre of mass ahead of the stance ankle, v its speed.
// Stance hip: holds the torso at a set lean. Knees and ankles: fixed targets per state.
import { config } from "../config.js";

export const SPEC = [
  ['T', 0.35, 0.1], // seconds per step
  ['swing0', 0.5, 0.25], // swing hip world angle, early part of the step
  ['swing1', 0.2, 0.25], // late part
  ['cd', 0.5, 0.3],
  ['cv', 0.2, 0.15],
  ['kneeSwing0', -1.1, 0.3],
  ['kneeSwing1', -0.1, 0.2],
  ['kneeStance', -0.05, 0.1],
  ['ankleSwing', 0.1, 0.15],
  ['ankleStance', 0.0, 0.15],
  ['lean', -0.05, 0.08], // torso world angle: negative leans forward (CCW is positive, facing +x)
  ['split', 0.5, 0.15], // share of the step in the early state
  ['arm', 0.3, 0.3], // arm swing amplitude
];

export function decode(v) {
  const p = {};
  SPEC.forEach(([name, base], i) => (p[name] = base + (v[i] || 0)));
  p.T = Math.max(0.15, Math.min(0.8, p.T));
  p.split = Math.max(0.1, Math.min(0.9, p.split));
  return p;
}

// Returns controller(p, ep, action): writes the 14 joint commands (and grips off) into action.
export function makeController(world) {
  const J = config.net.jointCount;
  const P = config.partIndex;
  // Parts: 0 torso, 6/12 thighs, 7/13 shins, feet. Joints: 0 waist, 1 neck, 2/8 shoulder,
  // 3/9 elbow, 4/10 wrist, 5/11 hip, 6/12 knee, 7/13 ankle (left, right).
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
  return function simbicon(p, ep, action) {
  const st = ep.simbicon || (ep.simbicon = { stance: 0, t: 0 });
  const dt = 1 / config.physics.hz;
  st.t += dt;
  const stance = LEG[st.stance];
  const swing = LEG[1 - st.stance];
  const m = ep.measure;
  if ((st.t >= p.T && m.touch[swing.foot]) || st.t >= 2 * p.T) {
    st.stance = 1 - st.stance;
    st.t = 0;
    return simbicon(p, ep, action);
  }
  const early = st.t < p.split * p.T;
  const torso = angleOf(0);
  const d = (m.centerX || 0) - world.px[stance.foot];
  const v = (m.centerVelocityX || 0) - (p.track ? m.walkSpeed || 0 : 0);
  // Swing hip: world-frame thigh target with balance feedback; joint angle = thigh - torso.
  const swingWorld = (early ? p.swing0 : p.swing1) + p.cd * d + p.cv * v;
  action[swing.hip] = command(swing.hip, swingWorld - torso);
  // Stance hip: hold the torso at the lean. Moving the joint by (torso - lean) turns the torso back.
  const stanceHip = angleOf(stance.thigh) - torso;
  action[stance.hip] = command(stance.hip, stanceHip + (torso - p.lean));
  action[swing.knee] = command(swing.knee, early ? p.kneeSwing0 : p.kneeSwing1);
  action[stance.knee] = command(stance.knee, p.kneeStance);
  // Ankles: foot angle relative to the floor.
  action[swing.ankle] = command(swing.ankle, p.ankleSwing - angleOf(swing.shin));
  action[stance.ankle] = command(stance.ankle, p.ankleStance - angleOf(stance.shin));
  action[0] = 0; // waist
  action[1] = 0; // neck
  // Arms swing against the legs.
  action[swing.shoulder] = command(swing.shoulder, -p.arm);
  action[stance.shoulder] = command(stance.shoulder, p.arm);
  action[swing.elbow] = command(swing.elbow, 0.3);
  action[stance.elbow] = command(stance.elbow, 0.3);
  action[4] = 0;
  action[10] = 0;
  for (let j = J; j < action.length; j++) action[j] = -1;
  };
}
