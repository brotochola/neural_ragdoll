// A small corrective network on top of the running controller (src/runner.js) for the dash course.
// Each physics frame it reads the body and the next two obstacles and adds a correction to the runner's
// commands for the stance leg, the swing leg and the waist. Written per leg role (stance / swing), so the
// same weights serve both sides. All-zero output = the runner exactly.
//   inputs (DIN): trunk angle and spin, centre-of-mass velocity, hips over the ground, phase (flight, time
//   in the stride, jumping, landing, stepping down), both feet relative to the hips, the two next obstacles
//   (distance, height, width), the six leg joint angles
//   outputs (DOUT): stance hip, knee, ankle, swing hip, knee, ankle, waist; tanh, times OUT_SCALE, added to
//   the runner's normalized commands
export const DIN = 28, DH = 16, DOUT = 7, OUT_SCALE = 0.5;
export const NW = DH * (DIN + 1) + DOUT * (DH + 1);

const LEGS = [
  { hip: 5, knee: 6, ankle: 7, foot: 8 },
  { hip: 11, knee: 12, ankle: 13, foot: 14 },
];
const clip = (v, a) => (v > a ? a : v < -a ? -a : v);

export function makeDashNet(world, config, Box2D) {
  const x = new Float64Array(DIN), h = new Float64Array(DH), y = new Float64Array(DOUT);
  const ang = (i) => Math.atan2(world.s[i], world.c[i]);
  const jointAngle = (j) => Box2D.b2RevoluteJoint_GetAngle(world.joints[j].id);
  // ground height under x: the top of an obstacle there, else the floor
  const groundAt = (now, px) => { let g = 0; for (const o of now) if (o.top != null && px >= o.x0 && px <= o.x1) g = Math.max(g, o.top); return g; };
  function features(ep, st, now, vRun) {
    const m = ep.measure, px = world.px[0], py = world.py[0];
    const S = LEGS[st.stance], W = LEGS[1 - st.stance];
    let k = 0;
    x[k++] = clip(ang(0), 2);
    x[k++] = clip(world.spin[0] / 5, 2);
    x[k++] = clip(((m.centerVelocityX || 0) - vRun) / 2, 2);
    x[k++] = clip((m.centerVelocityY || 0) / 2, 2);
    x[k++] = clip(py - groundAt(now, px) - 0.95, 1);
    x[k++] = st.flight ? 1 : -1;
    x[k++] = clip(st.t / 0.3 - 1, 2);
    x[k++] = st.jumping ? 1 : -1;
    x[k++] = st.landing ? 1 : -1;
    x[k++] = clip((st.dropH || 0) / 0.3, 2);
    for (const L of [S, W]) { x[k++] = clip(world.px[L.foot] - px, 1.5); x[k++] = clip(world.py[L.foot] - py + 0.9, 1.5); }
    const ahead = now.filter((o) => o.x1 > px - 0.2);
    for (let i = 0; i < 2; i++) {
      const o = ahead[i];
      if (o) { x[k++] = clip((o.x0 - px) / 3, 1.5); x[k++] = clip((o.top || 0) / 0.3, 2); x[k++] = clip((o.x1 - o.x0) / 0.8, 2); }
      else { x[k++] = 1.5; x[k++] = 0; x[k++] = 0; }
    }
    for (const L of [S, W]) { x[k++] = jointAngle(L.hip) / 1.5; x[k++] = jointAngle(L.knee) / 1.5; x[k++] = jointAngle(L.ankle) / 0.6; }
    x[k++] = 1; // bias input
    return k;
  }
  // adds the correction into action (normalized joint commands) in place
  return function correct(theta, ep, st, now, vRun, action) {
    features(ep, st, now, vRun);
    let w = 0;
    for (let i = 0; i < DH; i++) { let s = theta[w++]; for (let j = 0; j < DIN; j++) s += theta[w++] * x[j]; h[i] = Math.tanh(s); }
    for (let i = 0; i < DOUT; i++) { let s = theta[w++]; for (let j = 0; j < DH; j++) s += theta[w++] * h[j]; y[i] = Math.tanh(s) * OUT_SCALE; }
    const S = LEGS[st.stance], W = LEGS[1 - st.stance];
    const add = (j, d) => { action[j] = Math.max(-1, Math.min(1, action[j] + d)); };
    add(S.hip, y[0]); add(S.knee, y[1]); add(S.ankle, y[2]);
    add(W.hip, y[3]); add(W.knee, y[4]); add(W.ankle, y[5]);
    add(0, y[6]);
  };
}

// small random first layer, zero output layer: starts as the runner exactly
export function initTheta(rand) {
  const t = new Float64Array(NW);
  for (let i = 0; i < DH * (DIN + 1); i++) t[i] = (rand() * 2 - 1) * 0.3;
  return t;
}
