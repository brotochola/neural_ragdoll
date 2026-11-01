// TRAIL NET v2: as src/trail-net.js, plus both feet's contact as inputs (DIN 40) and three more outputs (DOUT
// 10): both shoulders (arms for balance; the runner swings them on a fixed pattern) and the step length in the
// air (added to the selector's placeAdj, +-0.2 rad). fromV1(theta) carries a v1 network over with the new
// weights at zero, so it starts out doing exactly what the v1 net does.
// (v1:) The corrective network for the trail course (src/terrain.js makeTrailCourse): like src/dash-net.js, added to
// the running controller's commands every frame, but it sees the ground itself:
//   body (10): trunk angle and spin, centre-of-mass velocity against the commanded speed, hips over the floor,
//     flight, time in the stride, jumping, landing, stepping down
//   feet (4): both feet relative to the hips (stance leg first)
//   next two obstacles (8): distance, height over the floor it stands on (a hole reads -1), width, is a hole
//   ground profile (9): the ground from 0.4 m behind to 2.8 m ahead of the hips, every 0.4 m, against the floor
//     under the stance foot (a hole reads deep)
//   leg joints (6), bias (1)
// out (7): stance hip, knee, ankle, swing hip, knee, ankle, waist; tanh x OUT_SCALE.
// ep.groundAt(x) (the real ground, 0 in a hole) and ep.floorAt(x) (holes bridged) come from the episode builder.
export const DIN = 40, DH = 24, DOUT = 10, OUT_SCALE = 0.5;
const V1 = { DIN: 38, DOUT: 7 };
export const NW = DH * (DIN + 1) + DOUT * (DH + 1);
const PROFILE = [-0.4, 0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8];

const LEGS = [
  { hip: 5, knee: 6, ankle: 7, foot: 8 },
  { hip: 11, knee: 12, ankle: 13, foot: 14 },
];
const clip = (v, a) => (v > a ? a : v < -a ? -a : v);

export function makeTrailNet2(world, config, Box2D) {
  const x = new Float64Array(DIN), h = new Float64Array(DH), y = new Float64Array(DOUT);
  const ang = (i) => Math.atan2(world.s[i], world.c[i]);
  const jointAngle = (j) => Box2D.b2RevoluteJoint_GetAngle(world.joints[j].id);
  function features(ep, st, now, vRun) {
    const m = ep.measure, px = world.px[0], py = world.py[0];
    const S = LEGS[st.stance], W = LEGS[1 - st.stance];
    const floorAt = ep.floorAt || (() => 0), groundAt = ep.groundAt || floorAt;
    const g0 = floorAt(world.px[S.foot]);
    let k = 0;
    x[k++] = clip(ang(0), 2);
    x[k++] = clip(world.spin[0] / 5, 2);
    x[k++] = clip(((m.centerVelocityX || 0) - vRun) / 2, 2);
    x[k++] = clip((m.centerVelocityY || 0) / 2, 2);
    x[k++] = clip(py - floorAt(px) - 0.95, 1);
    x[k++] = st.flight ? 1 : -1;
    x[k++] = clip(st.t / 0.3 - 1, 2);
    x[k++] = st.jumping ? 1 : -1;
    x[k++] = st.landing ? 1 : -1;
    x[k++] = clip((st.dropH || 0) / 0.3, 2);
    for (const L of [S, W]) { x[k++] = clip(world.px[L.foot] - px, 1.5); x[k++] = clip(world.py[L.foot] - py + 0.9, 1.5); }
    const ahead = now.filter((o) => o.x1 > px - 0.2);
    for (let i = 0; i < 2; i++) {
      const o = ahead[i];
      if (o) {
        const hole = o.top == null;
        x[k++] = clip((o.x0 - px) / 3, 1.5);
        x[k++] = hole ? -1 : clip((o.top - g0) / 0.3, 2);
        x[k++] = clip((o.x1 - o.x0) / 0.8, 2);
        x[k++] = hole ? 1 : -1;
      } else { x[k++] = 1.5; x[k++] = 0; x[k++] = 0; x[k++] = -1; }
    }
    for (const d of PROFILE) x[k++] = clip((groundAt(px + d) - g0) / 0.4, 1.5);
    for (const L of [S, W]) { x[k++] = jointAngle(L.hip) / 1.5; x[k++] = jointAngle(L.knee) / 1.5; x[k++] = jointAngle(L.ankle) / 0.6; }
    x[k++] = 1;
    x[k++] = m.touch[S.foot] ? 1 : -1; x[k++] = m.touch[W.foot] ? 1 : -1;
    return k;
  }
  const correct = function (theta, ep, st, now, vRun, action) {
    features(ep, st, now, vRun);
    let w = 0;
    for (let i = 0; i < DH; i++) { let s = theta[w++]; for (let j = 0; j < DIN; j++) s += theta[w++] * x[j]; h[i] = Math.tanh(s); }
    for (let i = 0; i < DOUT; i++) { let s = theta[w++]; for (let j = 0; j < DH; j++) s += theta[w++] * h[j]; y[i] = Math.tanh(s) * OUT_SCALE; }
    const S = LEGS[st.stance], W = LEGS[1 - st.stance];
    const add = (j, d) => { action[j] = Math.max(-1, Math.min(1, action[j] + d)); };
    add(S.hip, y[0]); add(S.knee, y[1]); add(S.ankle, y[2]);
    add(W.hip, y[3]); add(W.knee, y[4]); add(W.ankle, y[5]);
    add(0, y[6]);
    // arms: the stance side's shoulder, the swing side's; the step length in the air (next frame's placement)
    add(S.foot === 8 ? 2 : 8, y[7]); add(W.foot === 8 ? 2 : 8, y[8]);
    st.placeAdjNet = 0.4 * y[9];
  };
  correct.last = { x, h, y };
  return correct;
}

export function initTheta(rand) {
  const t = new Float64Array(NW);
  for (let i = 0; i < DH * (DIN + 1); i++) t[i] = (rand() * 2 - 1) * 0.3;
  return t;
}

// a v1 network (src/trail-net.js layout) in this layout: same weights, the new ones zero
export function fromV1(v1) {
  const t = new Float64Array(NW);
  let r = 0, w = 0;
  for (let i = 0; i < DH; i++) { t[w++] = v1[r++]; for (let j = 0; j < V1.DIN; j++) t[w++] = v1[r++]; w += DIN - V1.DIN; }
  for (let i = 0; i < DOUT; i++) { if (i < V1.DOUT) { t[w++] = v1[r++]; for (let j = 0; j < DH; j++) t[w++] = v1[r++]; } else w += DH + 1; }
  return t;
}
