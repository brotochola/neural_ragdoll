// A step-up controller for climbing onto a box, in the spirit of SIMBICON (src/simbicon.js): a
// finite-state machine with joint targets per state, hips in WORLD frame, the stance hip holding the
// torso lean, and transitions on what the feet touch. States:
//   0 LIFT   the lead leg rises (hip flexed, knee bent) until its foot is over the box top
//   1 PLACE  the lead leg comes down onto the box
//   2 PUSH   the lead leg straightens and the torso leans over it; the trail leg pushes off
//   3 TUCK   the trail leg rises over the edge
//   4 PLANT  the trail leg comes down onto the box
//   5 DONE   the stand net takes over (the caller decides)
// Arms: a pose per state, and the grips close in the states listed by gripStates when the hands touch.
// Works for a box on either side: "forward" is toward the box (side = +1 or -1).
import { config } from "../config.js";

// [name, starting value, search spread]
export const SPEC = [
  ['liftHip', 1.5, 0.3], ['liftKnee', -2.0, 0.3], ['liftT', 0.6, 0.2],
  ['placeHip', 0.9, 0.3], ['placeKnee', -1.2, 0.3], ['placeT', 0.5, 0.2],
  ['pushLean', -0.5, 0.25], ['pushKnee', -0.2, 0.3], ['pushTrailHip', -0.3, 0.3], ['pushAnkle', 0.4, 0.3], ['pushT', 0.7, 0.25],
  ['tuckHip', 1.5, 0.3], ['tuckKnee', -2.0, 0.3], ['tuckT', 0.5, 0.2],
  ['plantHip', 0.3, 0.3], ['plantKnee', -0.6, 0.3], ['plantT', 0.5, 0.2],
  ['standLean', -0.1, 0.15], ['stanceKnee', -0.15, 0.15],
  ['cd', 0.3, 0.3], ['cv', 0.1, 0.15],
  ['armLift', 1.2, 0.5], ['armPush', -0.5, 0.5], ['elbow', 0.5, 0.4],
  ['lead', 0, 1], // > 0: right leg leads, else left
  ['grip', -1, 1], // > 0: grip during PLACE and PUSH
];

export function decode(v) {
  const p = {};
  SPEC.forEach(([name, base], i) => (p[name] = base + (v[i] || 0)));
  for (const k of ['liftT', 'placeT', 'pushT', 'tuckT', 'plantT']) p[k] = Math.max(0.15, Math.min(2, p[k]));
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
  // World-frame angles are mirrored for a box on the left: the body faces +x, so stepping toward -x
  // means swinging the thigh backward (negative). Joint limits still apply.
  return function stepup(p, ep, action) {
    const box = ep.climbBox;
    const st = ep.stepup || (ep.stepup = { state: 0, t: 0 });
    if (!box) return false;
    const dt = 1 / config.physics.hz;
    st.t += dt;
    const side = box.side > 0 ? 1 : -1;
    const lead = LEG[p.lead > 0 ? 1 : 0];
    const trail = LEG[p.lead > 0 ? 0 : 1];
    const m = ep.measure;
    const top = box.top;
    const near = side > 0 ? box.x0 : box.x1;
    const over = (foot) => side * (world.px[foot] - near) > 0.06;
    const onBox = (foot) => over(foot) && world.py[foot] > top - 0.02 && m.touch[foot];
    const next = () => {
      st.state++;
      st.t = 0;
    };
    if (st.state === 0 && ((over(lead.foot) && world.py[lead.foot] > top + 0.03) || st.t > p.liftT)) next();
    else if (st.state === 1 && (onBox(lead.foot) || st.t > p.placeT)) next();
    else if (st.state === 2 && st.t > p.pushT) next();
    else if (st.state === 3 && ((over(trail.foot) && world.py[trail.foot] > top + 0.03) || st.t > p.tuckT)) next();
    else if (st.state === 4 && (onBox(trail.foot) || st.t > p.plantT)) next();
    if (st.state >= 5) return false;
    const torso = angleOf(0);
    const s = st.state;
    // Which leg holds the body: the trail leg until the lead foot is on the box, then the lead leg.
    const stance = s <= 1 ? trail : lead;
    const swing = s <= 1 ? lead : trail;
    const lean = side * (s === 2 ? p.pushLean : p.standLean);
    const d = (m.centerX || 0) - world.px[stance.foot];
    const v = m.centerVelocityX || 0;
    const fb = p.cd * d + p.cv * v;
    // Stance hip holds the torso lean (as in SIMBICON).
    const stanceHip = angleOf(stance.thigh) - torso;
    action[stance.hip] = command(stance.hip, stanceHip + (torso - lean));
    action[stance.knee] = command(stance.knee, s === 2 ? p.pushKnee : p.stanceKnee);
    action[stance.ankle] = command(stance.ankle, 0 - angleOf(stance.shin));
    // Swing leg.
    let hip;
    let knee;
    if (s === 0) [hip, knee] = [p.liftHip, p.liftKnee];
    else if (s === 1) [hip, knee] = [p.placeHip + fb, p.placeKnee];
    else if (s === 2) [hip, knee] = [p.pushTrailHip, p.stanceKnee];
    else if (s === 3) [hip, knee] = [p.tuckHip, p.tuckKnee];
    else [hip, knee] = [p.plantHip + fb, p.plantKnee];
    action[swing.hip] = command(swing.hip, side * hip - torso);
    action[swing.knee] = command(swing.knee, knee);
    action[swing.ankle] = command(swing.ankle, s === 2 ? p.pushAnkle : 0 - angleOf(swing.shin));
    action[0] = 0;
    action[1] = 0;
    const arm = side * (s <= 1 ? p.armLift : p.armPush);
    for (const leg of LEG) {
      action[leg.shoulder] = command(leg.shoulder, arm - torso);
      action[leg.elbow] = command(leg.elbow, p.elbow);
    }
    action[4] = 0;
    action[10] = 0;
    const grip = p.grip > 0 && (s === 1 || s === 2) ? 1 : -1;
    for (let j = J; j < action.length; j++) action[j] = grip;
    return true;
  };
}
