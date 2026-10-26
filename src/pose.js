// POSE EXECUTOR: turns a body-space goal ("pelvis here, torso at this lean, left foot there, right
// hand on that edge") into joint commands, with 2D two-link inverse kinematics for legs and arms.
// This is the low level of a two-level controller: a planner (code) decides where each foot, hand and
// the pelvis should be; this executor (code now, a network later) moves the joints so they get there.
// No external forces: only the joint motors, as for the networks.
//
// Geometry (standing, from src/ragdoll.js): hip joint 0.96 m, knee 0.52, ankle 0.10, so thigh 0.44 and
// shin 0.42. Waist joint 1.26, shoulder 1.48, elbow 1.20, wrist 0.94: upper arm 0.28, forearm 0.26.
// The foot's sole centre is 0.06 forward of and 0.10 below the ankle. World angles: 0 = as standing,
// positive = counter-clockwise (the body faces +x, so + swings a limb forward).
import { config } from "../config.js";

export const BODY = {
  thigh: 0.44, shin: 0.42, upperArm: 0.28, forearm: 0.26,
  hipToWaist: 0.30, waistToShoulder: 0.22,
  soleFromAnkle: { x: 0.06, y: -0.10 },
  hipHeight: 0.96,
};

const rot = (a, x, y) => ({ x: Math.cos(a) * x - Math.sin(a) * y, y: Math.sin(a) * x + Math.cos(a) * y });

// Two-link IK from a root to a target. Returns world angles of the two segments (0 = pointing down).
// bend = -1 puts the middle joint forward (knee), +1 puts it behind (elbow).
export function twoLink(root, target, l1, l2, bend) {
  let dx = target.x - root.x;
  let dy = target.y - root.y;
  let r = Math.hypot(dx, dy);
  const max = l1 + l2 - 1e-4;
  const min = Math.abs(l1 - l2) + 1e-4;
  if (r > max) { dx *= max / r; dy *= max / r; r = max; }
  if (r < min) { const s = min / Math.max(r, 1e-6); dx *= s; dy *= s; r = min; }
  const phi = Math.atan2(dx, -dy); // direction of root->target, measured from straight down
  const alpha = Math.acos(Math.max(-1, Math.min(1, (l1 * l1 + r * r - l2 * l2) / (2 * l1 * r))));
  const gamma = Math.acos(Math.max(-1, Math.min(1, (l1 * l1 + l2 * l2 - r * r) / (2 * l1 * l2))));
  const upper = phi - bend * alpha;
  const middle = bend * (Math.PI - gamma); // joint angle: knee negative, elbow positive
  return { upper, lower: upper + middle, middle, reach: r };
}

// Goal -> joint target angles (radians). goal: { pelvis: {x,y}, torso, chest, feet: [{x,y,angle}x2],
// hands: [{x,y}x2] } with feet[0]/hands[0] the LEFT limbs. Missing hands: arms hang.
// actual (optional): { pelvis:{x,y}, torso, chest } of the body now. Limbs that are NOT supporting
// (feet not planted, hands not gripping) are aimed from the actual hip/shoulder and torso, so they
// reach their world targets even when the trunk is off its plan; supporting limbs use the goal, so
// moving them moves the body toward the goal.
export function jointTargets(goal, actual) {
  const t = new Float64Array(14);
  const torso = goal.torso || 0;
  const chest = goal.chest != null ? goal.chest : torso;
  const hip = goal.pelvis;
  const planted = goal.planted || [true, true];
  const gripping = goal.gripping || [false, false];
  const aTorso = actual ? actual.torso : torso;
  const aChest = actual ? (actual.chest != null ? actual.chest : actual.torso) : chest;
  const aHip = actual ? actual.pelvis : hip;
  t[0] = chest - torso; // waist
  t[1] = 0; // neck
  const legs = [[5, 6, 7], [11, 12, 13]];
  for (let k = 0; k < 2; k++) {
    const f = goal.feet[k];
    const fa = f.angle || 0;
    const off = rot(fa, BODY.soleFromAnkle.x, BODY.soleFromAnkle.y);
    const ankle = { x: f.x - off.x, y: f.y - off.y };
    const free = !planted[k] && actual;
    const ik = twoLink(free ? aHip : hip, ankle, BODY.thigh, BODY.shin, -1);
    t[legs[k][0]] = ik.upper - (free ? aTorso : torso); // hip joint = thigh - torso
    // Supporting leg: the hip also holds the TORSO upright in world frame (as SIMBICON's stance hip).
    // If the trunk pitches past its goal, the hip joint moves so the trunk turns back.
    if (!free && actual) t[legs[k][0]] += (goal.torsoGain != null ? goal.torsoGain : +(globalThis.process?.env?.TORSO_GAIN || 0)) * (aTorso - torso);
    t[legs[k][1]] = ik.middle; // knee
    t[legs[k][2]] = fa - ik.lower; // ankle = foot - shin
  }
  const shoulderOf = (h, tr, ch) => {
    const w = { x: h.x + rot(tr, 0, BODY.hipToWaist).x, y: h.y + rot(tr, 0, BODY.hipToWaist).y };
    return { x: w.x + rot(ch, 0, BODY.waistToShoulder).x, y: w.y + rot(ch, 0, BODY.waistToShoulder).y };
  };
  const shoulderGoal = shoulderOf(hip, torso, chest);
  const shoulderNow = shoulderOf(aHip, aTorso, aChest);
  const arms = [[2, 3, 4], [8, 9, 10]];
  for (let k = 0; k < 2; k++) {
    const h = goal.hands && goal.hands[k];
    if (!h) {
      t[arms[k][0]] = 0 - chest;
      t[arms[k][1]] = 0.3;
      t[arms[k][2]] = 0;
      continue;
    }
    const free = !gripping[k] && actual;
    const ik = twoLink(free ? shoulderNow : shoulderGoal, h, BODY.upperArm, BODY.forearm, 1);
    t[arms[k][0]] = ik.upper - (free ? aChest : chest); // shoulder = upper arm - chest
    t[arms[k][1]] = ik.middle; // elbow
    t[arms[k][2]] = 0;
  }
  return t;
}

// BALANCE feedback on a goal, with the feet as support: the centre of mass should stay over the
// support point (by default the middle of the feet that are down). Two strategies, as people do:
//   hips: the pelvis target moves against the error (kHip),
//   ankles: the planted feet tilt the shins back toward balance (kAnkle, kAnkleVel).
// Returns a corrected copy of the goal and the ankle offset to add to planted ankles.
export function balanced(goal, measure, opts = {}) {
  const kHip = opts.kHip != null ? opts.kHip : 0.8;
  const kHipVel = opts.kHipVel != null ? opts.kHipVel : 0.2;
  const kAnkle = opts.kAnkle != null ? opts.kAnkle : 1.5;
  const kAnkleVel = opts.kAnkleVel != null ? opts.kAnkleVel : 0.3;
  const planted = opts.planted || [true, true];
  let sx = 0;
  let n = 0;
  for (let k = 0; k < 2; k++) if (planted[k]) { sx += goal.feet[k].x; n++; }
  const support = opts.support != null ? opts.support : n ? sx / n : goal.pelvis.x;
  const err = (measure.centerX || 0) - support;
  const vel = measure.centerVelocityX || 0;
  const out = Object.assign({}, goal, { pelvis: { x: goal.pelvis.x - kHip * err - kHipVel * vel, y: goal.pelvis.y } });
  return { goal: out, ankle: -(kAnkle * err + kAnkleVel * vel), planted };
}

// jointTargets plus the balance ankle offset on planted feet.
export function balancedTargets(goal, measure, opts, actual) {
  const b = balanced(goal, measure, opts);
  const t = jointTargets(b.goal, actual);
  if (b.planted[0]) t[7] += b.ankle;
  if (b.planted[1]) t[13] += b.ankle;
  return t;
}

// Joint target angles -> commands in [-1, 1] (the network's action space), clamped to the joint limits.
export function commandsFor(world, targets, action) {
  for (let j = 0; j < targets.length; j++) {
    const joint = world.joints[j];
    const a = Math.max(joint.lo, Math.min(joint.hi, targets[j]));
    action[j] = a >= 0 ? (joint.hi > 0 ? a / joint.hi : 0) : joint.lo < 0 ? a / -joint.lo : 0;
  }
}

// Where a body point is now (for latching a planted foot or a gripped hand).
export function soleOf(world, footPart) {
  const a = Math.atan2(world.s[footPart], world.c[footPart]);
  // foot body centre is 0.04 above the sole centre (capsule radius)
  const d = rot(a, 0, -0.04);
  return { x: world.px[footPart] + d.x, y: world.py[footPart] + d.y, angle: a };
}
export function handOf(world, handPart) {
  return { x: world.px[handPart], y: world.py[handPart] };
}
export function actualOf(world) {
  const pv = pelvisOf(world);
  return { pelvis: { x: pv.x, y: pv.y }, torso: pv.angle, chest: Math.atan2(world.s[1], world.c[1]) };
}
export function pelvisOf(world) {
  const a = Math.atan2(world.s[0], world.c[0]);
  // torso body centre is 1.14 standing; the hip joint 0.18 below it
  const d = rot(a, 0, -0.18);
  return { x: world.px[0] + d.x, y: world.py[0] + d.y, angle: a };
}

export const PARTS = () => ({
  feet: [config.partIndex.footLeft, config.partIndex.footRight],
  hands: [config.partIndex.handLeft, config.partIndex.handRight],
});
