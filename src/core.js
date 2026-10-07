// Shared ragdoll sim + Elman policy. Works in Node and the browser.
// Network: x_t = [obs, a_{t-1}], h_t = leak * h_{t-1} + (1-leak) * tanh(Wx x + Wh h + b), a_t = tanh(Wo h)
export const DT = 1 / 60;
export const H = 20;
export const O = 13;
export const NI0 = 58;
export const NI = 79;
export function nwOf(h) { return h * (NI + h + 1) + O * (h + 1); }
export function hiddenOf(len) {
  const b = NI + 1 + O;
  const disc = b * b - 4 * (O - len);
  if (!(disc >= 0)) return 0;
  const h = Math.round((-b + Math.sqrt(disc)) / 2);
  return h >= 1 && nwOf(h) === len ? h : 0;
}
export const NW = nwOf(H);
export const LEAK = 0.25;
export const PW = 2;
export const HH = 0.1;
export const MAXTILT = 0.1745;
export const SIG0 = 0.1;
export const SIG_GETUP = 0.12;
export const getupTask = t => t === 'getup';
// Gate only. Comfort (~0.29 m) stays in the stand reward (STANCE_PEAK), not here.
const UP_ANG = 15 * Math.PI / 180, UP_HEAD = 1.45, STANCE_LO = 0.05, STANCE_HI = 0.9;
export function isUp(m) {
  if (!m || !m.touch || !m.touch[7] || !m.touch[13]) return false;
  for (let i = 0; i < m.touch.length; i++) if (i !== 7 && i !== 13 && m.touch[i]) return false;
  return Math.abs(m.a) < UP_ANG && m.hy > UP_HEAD && m.stance >= STANCE_LO && m.stance <= STANCE_HI
    && Math.abs(m.w) < 1.5 && Math.abs(m.cvx) < 0.5;
}
// ponytail: viz copy of the trainer score. Height pays as the torso turns up, unless the hands are holding it.
export function prizeBreakdown(m) {
  const s = liveScore(m);
  return { total: s.total, max: s.max, up: s.onFeet > 0.5, terms: s.terms };
}
export function liveScore(m) {
  const face = Math.cos(m.a || 0);
  const hip01 = cl((m.ty || 0) / 1.26, 0, 1);
  const head01 = cl((m.hy || 0) / 1.68, 0, 1);
  // Height pays as the torso turns up, and only while the hands are not the support.
  const rise = cl((face + 0.2) / 1.2, 0, 1);
  const hands = (1 - cl((m.clear?.[4] ?? 1) / 0.15, 0, 1)) + (1 - cl((m.clear?.[10] ?? 1) / 0.15, 0, 1));
  const handFree = 1 - cl(hands, 0, 1);
  const headTerm = -1 + rise * handFree * 2 * head01;
  const hipTerm = -1.2 + rise * handFree * 2.4 * hip01;
  const posture = (3 * face + Math.cos(m.ha || 0) + 2 * (m.legCos || 0)) / 6 - 1;
  const postureTerm = 1.2 * posture;
  const thigh = i => 1 - cl((m.clear?.[i] ?? 1) / 0.35, 0, 1);
  const feet = (m.touch?.[7] ? 1 : 0) + (m.touch?.[13] ? 1 : 0);
  const near = i => 1 - cl((m.clear?.[i] ?? 1) / 0.4, 0, 1);
  let other = 0;
  if (m.touch) for (let i = 0; i < m.touch.length; i++) if (i !== 7 && i !== 13 && m.touch[i]) other = 1;
  const upright = cl((face - 0.7) / 0.3, 0, 1);
  const onFeet = (feet === 2 ? 1 : 0) * (1 - other) * upright * cl((m.hy || 0) / 1.45, 0, 1);
  const stance = Math.exp(-14 * ((m.stance || 0) - 0.29) * ((m.stance || 0) - 0.29));
  const impact = (m.clear && m.clear[1] < 0.03 && m.hvy < -1) ? 0.8 * cl(-m.hvy / 3, 0, 2) : 0;
  const stay = onFeet * (2.5 + 0.4 * stance);
  const park = cl((face - 0.55) / 0.35, 0, 1);
  const handTerm = -(0.1 + 1.5 * park) * hands;
  const legTerm = -0.35 * (thigh(5) + thigh(11));
  const footTerm = -0.5 * (2 - feet) + 0.35 * (near(7) + near(13));
  const inv = -1.2 * Math.max(0, -face);
  const total = headTerm + hipTerm + postureTerm + inv + legTerm + handTerm + footTerm - impact + stay;
  const terms = [
    { n: 'Cabeza', tip: 'La altura suma si el torso se endereza y las manos no sostienen el cuerpo. Si no, resta 1. De pie, a 1,68 m, suma 1.', v: headTerm },
    { n: 'Cadera', tip: 'Igual que la cabeza: la altura no paga apoyado en las manos. En el piso resta 1,2. De pie suma 1,2.', v: hipTerm },
    { n: 'Postura', tip: 'Torso, cabeza, muslos y tibias. El torso pesa más: enderezarse suma aunque sigas en el piso. Vale 0 solo si todo está vertical.', v: postureTerm },
    { n: 'Invertido', tip: 'Resta extra si el torso está dado vuelta.', v: inv },
    { n: 'Piernas', tip: 'Resta si un muslo está cerca del piso. La tibia no entra: de pie queda baja. Hasta −0,35 por muslo.', v: legTerm },
    { n: 'Manos', tip: 'Empujar casi no resta. Si el torso ya está bastante erguido y la mano sigue en el piso, resta hasta cerca de 1,6 por mano.', v: handTerm },
    { n: 'Pies', tip: 'Resta 0,5 por cada pie en el aire. Suma hasta 0,35 por pie cuanto más cerca del piso está.', v: footTerm },
    { n: 'Golpe', tip: 'Resta solo si la cabeza baja rápido a menos de 3 cm del piso. Quieta no cuenta.', v: -impact },
    { n: 'De pie', tip: 'Extra si ya está sobre los dos pies, erguido, con la cabeza alta y sin otra cosa en el piso.', v: stay }
  ];
  return { total, max: 1 + 1.2 + 0.7 + 2.9, onFeet, terms };
}
export function padWeights(src) {
  if (!src || !src.length) return null;
  const hNow = hiddenOf(src.length);
  if (hNow) return Float32Array.from(src);
  const oldNI = (src.length - O * (H + 1)) / H - H - 1;
  if (oldNI !== (oldNI | 0) || oldNI < 1 || oldNI >= NI) return null;
  const extra = NI - oldNI, t = new Float32Array(NW);
  let oi = 0, ni = 0;
  for (let j = 0; j < H; j++) {
    t[ni++] = src[oi++];
    for (let i = 0; i < oldNI; i++) t[ni++] = src[oi++];
    ni += extra;
    for (let i = 0; i < H; i++) t[ni++] = src[oi++];
  }
  for (let i = 0; i < O * (H + 1); i++) t[ni++] = src[oi++];
  return t;
}
export function netsFromCkpt(c) {
  const seed = padWeights(c.bestTheta) || padWeights(c.thetaStand) || padWeights(c.theta);
  if (!seed) throw new Error('checkpoint NW mismatch');
  const take = a => padWeights(a) || seed;
  const old = a => a && a.length && a.length !== NW;
  const getup = take(c.thetaGetup);
  const supine = (padWeights(c.thetaSupine) || getup).slice();
  const prone = (padWeights(c.thetaProne) || getup).slice();
  return {
    thetaStand: take(c.thetaStand),
    thetaGetup: supine,
    thetaSupine: supine,
    thetaProne: prone,
    bestStand: take(c.bestStand || c.bestTheta),
    bestGetup: (padWeights(c.bestSupine) || padWeights(c.bestGetup) || supine).slice(),
    bestSupine: (padWeights(c.bestSupine) || padWeights(c.bestGetup) || supine).slice(),
    bestProne: (padWeights(c.bestProne) || padWeights(c.bestGetup) || prone).slice(),
    migrated: !(c.thetaStand && c.thetaStand.length),
    padded: old(c.theta) || old(c.thetaStand) || old(c.thetaGetup) || old(c.thetaSupine) || old(c.bestTheta)
  };
}
export const KINDS = [
  { r: 0.12, d: 100, c: 0xff6b6b, v: 14 },
  { box: 1, hx: 0.15, d: 100, c: 0xc08a4b, v: 9 },
  { r: 0.16, d: 310, c: 0x9aa3b8, v: 4 }
];
export const cl = (x, a, b) => Math.max(a, Math.min(b, x));
export function rngMk(s) {
  return () => {
    s |= 0;
    s = s + 0x6D2B79F5 | 0;
    let t = Math.imul(s ^ s >>> 15, 1 | s);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

export function createSim(B, opts = {}) {
  const {
    CreateWorld, CreateBoxPolygon, CreateRevoluteJoint, b2Vec2, b2BodyType, b2World_Step,
    b2Body_GetPosition, b2Body_GetRotation, b2Rot_GetAngle, b2Body_SetTransform, b2MakeRot,
    b2Body_SetLinearVelocity, b2Body_SetAngularVelocity, b2Body_SetAwake, b2Body_GetLinearVelocity,
    b2Body_GetAngularVelocity, b2RevoluteJoint_GetAngle, b2RevoluteJoint_SetMotorSpeed,
    CreateCircle, b2DestroyBody, b2World_CastRay, b2DefaultQueryFilter, b2Shape_GetFilter
  } = B;
  const DYN = b2BodyType.b2_dynamicBody, STA = b2BodyType.b2_staticBody, KIN = b2BodyType.b2_kinematicBody;
  const V = (x, y) => new b2Vec2(x, y);
  let rng = rngMk(opts.seed ?? 7);
  const randn = () => Math.sqrt(-2 * Math.log(rng() + 1e-12)) * Math.cos(6.2831853 * rng());

  function spawnObj(R, k, x, y, vx, vy) {
    const pos = V(x, y);
    const r = k.box
      ? CreateBoxPolygon({ worldId: R.wid, type: DYN, position: pos, size: V(k.hx, k.hx), density: k.d, friction: 0.6, restitution: 0.2 })
      : CreateCircle({ worldId: R.wid, type: DYN, position: pos, radius: k.r, density: k.d, friction: 0.6, restitution: 0.3 });
    b2Body_SetLinearVelocity(r.bodyId, V(vx, vy));
    return r.bodyId;
  }

  function build() {
    const { worldId } = CreateWorld({});
    const gr = CreateBoxPolygon({ worldId, type: STA, position: V(0, -1), size: V(800, 0.5), friction: 0.9 });
    const pl = CreateBoxPolygon({ worldId, type: KIN, position: V(0, -HH), size: V(PW, HH), friction: 0.9 });
    const P = [], J = [];
    const box = (cx, cy, hx, hy, d = 70) => {
      const r = CreateBoxPolygon({ worldId, type: DYN, position: V(cx, cy), size: V(hx, hy), density: d, friction: 0.9, groupIndex: -1 });
      P.push({ id: r.bodyId, cx, cy, hx, hy, m: d * 4 * hx * hy, th: Math.max(hx, hy) + 0.04 });
      return P.length - 1;
    };
    const jt = (a, b, wx, wy, lo, hi, tq) => {
      const A = P[a], B = P[b];
      const r = CreateRevoluteJoint({
        worldId, bodyIdA: A.id, bodyIdB: B.id,
        anchorA: V(wx - A.cx, wy - A.cy), anchorB: V(wx - B.cx, wy - B.cy),
        lowerAngle: lo, upperAngle: hi, enableLimit: true, enableMotor: true, motorSpeed: 0, maxMotorTorque: tq
      });
      J.push({ id: r.jointId, lo, hi, n: Math.max(-lo, hi), a, ax: wx - A.cx, ay: wy - A.cy, mark: '' });
    };
    box(0, 1.26, 0.13, 0.30); box(0, 1.68, 0.11, 0.12, 40);
    for (let s = 0; s < 2; s++) {
      box(0, 1.34, 0.05, 0.14, 50); box(0, 1.07, 0.045, 0.13, 50); box(0, 0.88, 0.05, 0.06, 50);
      box(0, 0.74, 0.07, 0.22); box(0, 0.31, 0.06, 0.21); box(0.06, 0.04, 0.12, 0.04, 90);
    }
    jt(0, 1, 0, 1.56, -0.5, 0.5, 20);
    for (let s = 0; s < 2; s++) {
      const b = 2 + 6 * s;
      jt(0, b, 0, 1.48, -2, 2.5, 70); jt(b, b + 1, 0, 1.2, 0, 2.4, 50); jt(b + 1, b + 2, 0, 0.94, -1.0, 1.0, 25);
      jt(0, b + 3, 0, 0.96, -0.7, 2, 140); jt(b + 3, b + 4, 0, 0.52, -2.4, 0, 140); jt(b + 4, b + 5, 0, 0.1, -0.6, 0.6, 60);
      J[J.length - 5].mark = 'elbow'; J[J.length - 3].mark = 'hip'; J[J.length - 2].mark = 'knee';
    }
    const rHip = 0.13 * 1.1;
    const hip = CreateCircle({ worldId, type: DYN, position: V(0, 0.96), radius: rHip, density: 40, friction: 0.9, groupIndex: -1 });
    P.push({ id: hip.bodyId, cx: 0, cy: 0.96, hx: rHip, hy: rHip, r: rHip, circle: true, m: 40 * Math.PI * rHip * rHip, th: rHip + 0.04 });
    CreateRevoluteJoint({
      worldId, bodyIdA: P[0].id, bodyIdB: hip.bodyId,
      anchorA: V(0 - P[0].cx, 0.96 - P[0].cy), anchorB: V(0, 0),
      lowerAngle: 0, upperAngle: 0, enableLimit: true, enableMotor: false, motorSpeed: 0, maxMotorTorque: 0
    });
    return { wid: worldId, P, J, ground: gr.bodyId, plat: pl.bodyId };
  }

  const platAng = (o, t) => o.amp * Math.sin(6.2831853 / o.T * t + o.ph);

  function reset(R, tilt, drop, o) {
    const c = Math.cos(tilt), s = Math.sin(tilt), ac = Math.abs(c), as = Math.abs(s);
    b2Body_SetTransform(R.plat, V(0, -HH), b2MakeRot(platAng(o, 0)));
    b2Body_SetLinearVelocity(R.plat, V(0, 0));
    b2Body_SetAngularVelocity(R.plat, 0);
    const poses = [];
    let minY = Infinity;
    for (const p of R.P) {
      const dx = p.cx, dy = p.cy - 0.04;
      const x = c * dx - s * dy, y = 0.04 + drop + s * dx + c * dy;
      poses.push({ p, x, y });
      minY = Math.min(minY, y - as * p.hx - ac * p.hy);
    }
    const lift = Math.max(0, 0.05 - minY);
    for (const { p, x, y } of poses) {
      b2Body_SetTransform(p.id, V(x, y + lift), b2MakeRot(tilt));
      b2Body_SetLinearVelocity(p.id, V(0, 0));
      b2Body_SetAngularVelocity(p.id, 0);
      b2Body_SetAwake(p.id, true);
    }
    for (const j of R.J) b2RevoluteJoint_SetMotorSpeed(j.id, 0);
  }

  // ponytail: 5mm, box2d slop. Resting parts overlap (<=0). 3cm was air.
  const TOUCH = 0.005, STANCE_PEAK = 0.29, FEET = [7, 13], HANDS = [4, 10];
  const QF = b2DefaultQueryFilter(), RAY = 2, LOOK = 0.2;
  function rayCast(wid, x, y, dx, dy) {
    if (dx * dx + dy * dy < 1e-8) return 1;
    let frac = 1;
    b2World_CastRay(wid, V(x, y), V(dx, dy), QF, (shapeId, _p, _n, f) => {
      if (b2Shape_GetFilter(shapeId).groupIndex === -1) return -1;
      frac = f;
      return f;
    }, 0);
    return frac;
  }
  function rayDown(wid, x, y) { return rayCast(wid, x, y, 0, -RAY); }
  // Gap from the drawn box to the platform slab or the ground top. Not limb-on-limb, not the infinite tilt line.
  function partClear(p, pq, pc, ps) {
    const q = b2Body_GetPosition(p.id), a = b2Rot_GetAngle(b2Body_GetRotation(p.id)), c = Math.cos(a), s = Math.sin(a);
    let gap = 1e9;
    for (const [sx, sy] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
      const wx = q.x + c * sx * p.hx - s * sy * p.hy, wy = q.y + s * sx * p.hx + c * sy * p.hy;
      const dx = wx - pq.x, dy = wy - pq.y, lx = pc * dx + ps * dy, ly = -ps * dx + pc * dy;
      const ax = Math.abs(lx) - PW, ay = Math.abs(ly) - HH;
      gap = Math.min(gap, ax > 0 && ay > 0 ? Math.hypot(ax, ay) : Math.max(ax, ay), wy + 0.5);
    }
    return gap;
  }
  function sense(R) {
    const P = R.P, ys = new Float32Array(P.length), pa = b2Rot_GetAngle(b2Body_GetRotation(R.plat)), tp = Math.tan(pa);
    const pq = b2Body_GetPosition(R.plat), pc = Math.cos(pa), ps = Math.sin(pa);
    let X = 0, Y = 0, VX = 0, VY = 0, M = 0, qx = [0, 0, 0], qy = [0, 0, 0];
    const m = { ys, pa, pw: b2Body_GetAngularVelocity(R.plat), touch: new Array(P.length), clear: new Array(P.length) };
    for (let i = 0; i < P.length; i++) {
      const p = P[i], q = b2Body_GetPosition(p.id), v = b2Body_GetLinearVelocity(p.id);
      ys[i] = q.y - q.x * tp; M += p.m; X += p.m * q.x; Y += p.m * ys[i]; VX += p.m * v.x; VY += p.m * v.y;
      m.clear[i] = partClear(p, pq, pc, ps);
      m.touch[i] = m.clear[i] < TOUCH;
      if (i === 0) { m.tx = q.x; m.tvx = v.x; m.tvy = v.y; qx[0] = q.x; qy[0] = q.y; }
      if (i === 1) { m.hvx = v.x; m.hvy = v.y; m.hx = q.x; m.hyw = q.y; }
      if (i === 7) { m.fx0 = q.x; qx[1] = q.x; qy[1] = q.y; }
      if (i === 13) { m.fx1 = q.x; qx[2] = q.x; qy[2] = q.y; }
    }
    m.ty = ys[0]; m.hy = ys[1]; m.fy0 = ys[7]; m.fy1 = ys[13];
    m.cx = X / M; m.cy = Y / M; m.cvx = VX / M; m.cvy = VY / M;
    m.a = b2Rot_GetAngle(b2Body_GetRotation(P[0].id)); m.w = b2Body_GetAngularVelocity(P[0].id);
    m.ha = b2Rot_GetAngle(b2Body_GetRotation(P[1].id));
    m.legCos = [5, 6, 11, 12].reduce((s, i) => s + Math.cos(b2Rot_GetAngle(b2Body_GetRotation(P[i].id))), 0) / 4;
    m.cl = [];
    for (const i of FEET) {
      const q = b2Body_GetPosition(P[i].id), a = b2Rot_GetAngle(b2Body_GetRotation(P[i].id)), c = Math.cos(a), s = Math.sin(a), hx = P[i].hx, hy = P[i].hy;
      m.cl.push(
        Math.max(0, q.y - s * hx - c * hy - (q.x - c * hx + s * hy) * tp),
        Math.max(0, q.y + s * hx - c * hy - (q.x + c * hx + s * hy) * tp)
      );
    }
    m.footOn = false; for (const i of FEET) if (m.touch[i]) m.footOn = true;
    m.handOn = false; for (const i of HANDS) if (m.touch[i]) m.handOn = true;
    m.shinOn = !!(m.touch[6] || m.touch[12]);
    m.bodyOn = false;
    for (let i = 0; i < P.length; i++) if (!FEET.includes(i) && m.touch[i]) m.bodyOn = true;
    m.stance = Math.abs(m.fx0 - m.fx1);
    m.rayT = rayDown(R.wid, qx[0], qy[0]);
    m.rayF0 = rayDown(R.wid, qx[1], qy[1]);
    m.rayF1 = rayDown(R.wid, qx[2], qy[2]);
    m.rayH = rayDown(R.wid, m.hx, m.hyw);
    m.rayHv = rayCast(R.wid, m.hx, m.hyw, m.hvx * LOOK, m.hvy * LOOK);
    return m;
  }

  const ttf = (y, vy) => cl((vy + Math.sqrt(vy * vy + 19.62 * Math.max(y - 0.15, 0))) / 9.81, 0, 1.5) / 1.5;
  // Cost falls to 0 at standing height / upright. Upside-down (cos<0) costs more than lying, and a hip raise does not pay until the torso turns upright.
  function getupLadder(m) {
    const face = Math.max(0, Math.cos(m.a));
    const leg = (a, b) => 0.25 * (1 - cl(Math.min(m.clear[a], m.clear[b]) / 0.35, 0, 1));
    const hand = i => 0.10 * (1 - cl(m.clear[i] / 0.15, 0, 1));
    const slam = m.clear[1] < 0.15 ? 1.5 * cl(-(m.hvy || 0) / 4, 0, 1) : 0;
    const pred = m.hvy < -0.4 && m.rayHv < 1 ? 1.2 * (1 - m.rayHv) * cl(-m.hvy / 3, 0, 1) : 0;
    return 0.60 * (1 - cl(m.hy / 1.65, 0, 1))
      + 0.40 * (1 - Math.cos(m.ha))
      + 0.35 * (1 - face * cl(m.ty / 1.26, 0, 1))
      + 0.25 * (1 - Math.cos(m.a))
      + 1.5 * Math.max(0, -Math.cos(m.a))
      + slam + pred
      + leg(5, 6) + leg(11, 12)
      + hand(4) + hand(10)
      + (m.touch[7] ? 0 : 0.40) + (m.touch[13] ? 0 : 0.40);
  }

  function forward(w, x, h) {
    const hdn = h.length;
    let k = 0;
    const nh = new Float32Array(hdn), y = new Float32Array(O);
    for (let j = 0; j < hdn; j++) {
      let s = w[k++];
      for (let i = 0; i < NI; i++) s += w[k++] * x[i];
      for (let i = 0; i < hdn; i++) s += w[k++] * h[i];
      nh[j] = LEAK * h[j] + (1 - LEAK) * Math.tanh(s);
    }
    for (let j = 0; j < O; j++) {
      let s = w[k++];
      for (let i = 0; i < hdn; i++) s += w[k++] * nh[i];
      y[j] = Math.tanh(s);
    }
    return [y, nh];
  }

  class Ep {
    constructor(R, w, tilt, drop, o = {}) {
      this.o = Object.assign({ push: 0, amp: 0, T: 8, ph: 0, proj: [], steps: 480, task: 'stand' }, o);
      this.free = false; this.fallen = false; this.objs = [];
      this.R = R; this.w = w; this.cand = w; this.t = 0; this.ret = 0; this.dead = false;
      this.h = new Float32Array(hiddenOf(w.length) || H); this.a = new Float32Array(O);
      this.upStreak = 0; this.recovered = false; this.stood = 0; this.nDec = 0; this.sumHy = 0;
      this.wasDown = false; this.quiet = false;
      this.prevA = new Float32Array(O);
      reset(R, tilt, drop, this.o);
      const settle = this.o.settle != null ? this.o.settle : ((this.o.task === 'getup' || this.o.chain) ? 50 : 0);
      if (settle > 0) {
        for (const j of R.J) b2RevoluteJoint_SetMotorSpeed(j.id, 0);
        for (let i = 0; i < settle; i++) b2World_Step(R.wid, DT, 4);
      }
      this.pa = R.J.map(j => b2RevoluteJoint_GetAngle(j.id));
      this.m = sense(R);
      this.t = 0; this.ret = 0; this.upStreak = 0; this.stood = 0; this.nDec = 0; this.sumHy = 0;
      this.sumAbsA = 0; this.sumHand = 0; this.prevPhi = 0; this.holdPay = 0; this.lastTick = 0; this.headImpact = 0;
      this.wasDown = this.o.task === 'getup' || !!this.o.chain; this.recovered = false; this.fallen = this.wasDown;
      this.prevHy = undefined;
      this.policy = o.policy || null;
      this.downNet = this.o.pose === 'prone' ? 'prone' : 'supine';
      this.activeNet = this.wasDown ? this.downNet : 'stand';
      if (this.policy) {
        if (this.policy.mode === 'stand' && !this.wasDown) this.activeNet = 'stand';
        else if (this.policy.mode === 'getup' || this.wasDown) this.activeNet = this.downNet;
        const nw = netW(this.policy, this.activeNet);
        if (nw) this.w = nw;
      }
    }
    cleanup() { for (const id of this.objs) b2DestroyBody(id); this.objs = []; }
    decide() {
      const R = this.R, m = this.m = sense(R), fmx = (m.fx0 + m.fx1) / 2;
      const gg = Math.sqrt(Math.max(m.cy, 0.3) / 9.81), cp = m.cx + m.cvx * gg, up = Math.max(0, Math.cos(m.a));
      let en = 0, mot = 0, jerk = 0;
      const ang = new Array(O);
      for (let i = 0; i < O; i++) {
        const j = R.J[i], a = this.a[i], an = b2RevoluteJoint_GetAngle(j.id);
        ang[i] = an;
        en += a * a;
        mot += Math.abs(a * (a >= 0 ? j.hi : -j.lo) - an) / j.n;
        jerk += Math.abs(a - this.prevA[i]);
      }
      const upNow = isUp(m);
      this.fallen = m.bodyOn || up < 0.25;
      if (!this.quiet) {
        if (m.clear[1] < 0.03) this.headImpact = Math.max(this.headImpact || 0, -(m.hvy || 0));
        if (upNow) { this.upStreak++; this.stood++; } else this.upStreak = 0;
        this.nDec++; this.sumHy += m.hy; this.sumAbsA += Math.abs(m.a); this.sumHand += m.handOn ? 1 : 0;
        let pose = 0;
        for (let i = 0; i < O; i++) {
          const excess = Math.max(0, Math.abs(ang[i]) / R.J[i].n - 0.15);
          pose += excess * excess;
        }
        const flat = i => Math.max(0, Math.cos(b2Rot_GetAngle(b2Body_GetRotation(R.P[i].id)) - m.pa));
        const sc = liveScore(m);
        this.lastTick = sc.total + sc.onFeet * (0.15 * (flat(7) + flat(13)) / 2 - 0.4 * pose / O);
        this.ret += this.lastTick;
        this.prevHy = m.hy;
      }
      if (this.policy) {
        let use = this.activeNet;
        const mode = this.policy.mode || 'auto';
        if (mode === 'stand') use = 'stand';
        else if (mode === 'getup') use = this.downNet;
        else if (!upNow) use = this.downNet;
        else if (this.upStreak >= 15) use = 'stand';
        if (use !== this.activeNet) this.h.fill(0);
        this.activeNet = use;
        const nw = netW(this.policy, use);
        if (nw) this.w = nw;
        else if (use === 'stand') this.w = this.cand;
      }
      const x = new Float32Array(NI); let k = 0;
      for (let i = 0; i < O; i++) {
        const j = R.J[i], an = b2RevoluteJoint_GetAngle(j.id);
        x[k + i] = an / j.n;
        x[k + O + i] = cl((an - this.pa[i]) / (2 * DT) / 6, -1, 1);
        this.pa[i] = an;
      }
      k += 2 * O;
      x[k++] = Math.sin(m.a); x[k++] = Math.cos(m.a); x[k++] = m.w / 5; x[k++] = m.tvx / 3; x[k++] = m.tvy / 3;
      x[k++] = cl((m.cx - fmx) / 0.3, -2, 2); x[k++] = m.cy - 0.85; x[k++] = m.cvx / 2; x[k++] = m.cvy / 2;
      x[k++] = cl((cp - fmx) / 0.3, -2, 2); x[k++] = ttf(m.hy, m.hvy); x[k++] = ttf(m.ty, m.tvy);
      x[k++] = m.hy / 1.7;
      for (let i = 0; i < 4; i++) x[k++] = cl(1 - m.cl[i] / 0.06, 0, 1);
      x[k++] = m.pa / 0.2; x[k++] = cl(m.pw / 0.15, -2, 2);
      for (let i = 0; i < O; i++) x[k++] = this.a[i];
      x[k++] = m.touch[0] ? 1 : 0; x[k++] = m.touch[1] ? 1 : 0;
      x[k++] = m.touch[4] ? 1 : 0; x[k++] = m.touch[10] ? 1 : 0;
      x[k++] = m.touch[7] ? 1 : 0; x[k++] = m.touch[13] ? 1 : 0;
      x[k++] = m.rayT; x[k++] = m.rayF0; x[k++] = m.rayF1;
      const nz = d => cl((d || 0) / 0.6, 0, 1);
      x[k++] = nz(m.clear[4]); x[k++] = nz(m.clear[10]);
      x[k++] = nz(m.clear[6]); x[k++] = nz(m.clear[12]);
      x[k++] = nz(m.clear[1]);
      x[k++] = m.rayH;
      x[k++] = Math.sin(m.ha); x[k++] = Math.cos(m.ha);
      x[k++] = nz(m.clear[5]); x[k++] = nz(m.clear[11]);
      x[k++] = nz(m.clear[3]); x[k++] = nz(m.clear[9]);
      this.x = x;
      const [y, nh] = forward(this.w, x, this.h);
      this.prevA.set(this.a);
      this.a = y; this.h = nh;
      return false;
    }
    step() {
      if (this.t % 2 === 0) this.decide();
      const R = this.R, o = this.o;
      if (this.t === 90 && o.push) {
        for (const p of R.P) {
          const v = b2Body_GetLinearVelocity(p.id);
          b2Body_SetLinearVelocity(p.id, V(v.x + o.push, v.y + (o.task === 'recover' ? 1.5 : 0)));
        }
        if (o.task === 'recover') b2Body_SetAngularVelocity(R.P[0].id, o.push * 2.2);
      }
      for (const pj of o.proj) if (pj.t === this.t) this.objs.push(spawnObj(R, KINDS[pj.k], this.m.cx + pj.sd * 3.5, pj.y0, pj.vx, pj.vy));
      for (let i = 0; i < O; i++) {
        const j = R.J[i], a = this.a[i], tg = a * (a >= 0 ? j.hi : -j.lo);
        b2RevoluteJoint_SetMotorSpeed(j.id, cl(12 * (tg - b2RevoluteJoint_GetAngle(j.id)), -10, 10));
      }
      const tt = this.t / 60, wv = 6.2831853 / o.T, thd = o.amp * wv * Math.cos(wv * tt + o.ph);
      b2Body_SetAngularVelocity(R.plat, thd + 6 * (platAng(o, tt) - b2Rot_GetAngle(b2Body_GetRotation(R.plat))));
      b2World_Step(R.wid, DT, 4);
      this.t++;
      const lim = o.steps || 480;
      if (!o.full && getupTask(o.task) && this.upStreak >= 30 && !this.free) {
        this.ret += Math.max(0, (lim - this.t) / 2) * this.holdPay;
        this.recovered = true; this.endedUp = true; this.dead = false;
        return true;
      }
      // ponytail: stuck get-up cut only under 0.6 m. The 0.6–isUp band is the stand slope; cutting it flattens the reward.
      if (o.cut && !o.full && !this.free && this.t > 240) {
        const hy = this.m.hy;
        if (this.hyMax == null) { this.hyMax = hy; this.hyBase = hy; this.stall = 0; }
        else if (hy > this.hyMax) this.hyMax = hy;
        if (this.hyMax > this.hyBase + 0.02) { this.hyBase = this.hyMax; this.stall = 0; }
        else if (++this.stall >= 180 && hy < 0.6 && this.upStreak === 0) {
          this.ret += Math.max(0, (lim - this.t) / 2) * this.lastTick;
          this.dead = true; this.endedUp = false;
          return true;
        }
      }
      if (this.t >= lim && !this.free) {
        this.dead = !this.recovered && this.fallen;
        this.endedUp = this.upStreak >= 15;
        return true;
      }
      return false;
    }
    get fit() { return this.ret / Math.max(1, (this.o.steps || 480) / 2); }
    get stats() {
      return {
        f: this.fit, d: this.dead, rec: this.recovered, up: !!this.endedUp,
        hy: this.nDec ? this.sumHy / this.nDec : 0, stood: this.stood / Math.max(1, this.nDec),
        ang: this.nDec ? this.sumAbsA / this.nDec * (180 / Math.PI) : 0,
        hand: this.nDec ? this.sumHand / this.nDec : 0,
        headImpact: this.headImpact || 0
      };
    }
  }

  function initTheta() {
    const t = new Float32Array(NW);
    const hid = H * (NI + H + 1);
    for (let i = 0; i < hid; i++) t[i] = randn() * 0.12;
    for (let i = hid; i < NW; i++) t[i] = randn() * 0.04;
    return t;
  }

  function supinePose(r) {
    return { tilt: 1.48 + r() * 0.22, drop: 0.03 + r() * 0.04 };
  }
  function pronePose(r) {
    return { tilt: -(1.48 + r() * 0.22), drop: 0.03 + r() * 0.04 };
  }

  function fallenPose(r, hard = 0.7) {
    const side = r() < 0.5 ? 1 : -1;
    const u = r();
    if (u < 0.42) return { tilt: side * (0.5 + r() * 0.45), drop: 0.05 + r() * 0.08 };
    if (u < 0.78) return { tilt: side * (0.95 + r() * 0.4), drop: 0.06 + r() * 0.09 };
    return { tilt: side * (1.35 + r() * (0.4 + 0.8 * hard)), drop: 0.07 + r() * 0.1 };
  }

  function addProjs(r, o, Q) {
    if (Q <= 0.02) return;
    for (let i = 0; i < 2; i++) {
      const ki = Math.floor(r() * 3), sd = r() < 0.5 ? -1 : 1;
      const v = Math.max(1.5, (0.3 + 0.7 * r()) * Q * KINDS[ki].v);
      const y0 = 0.4 + 1.1 * r(), ty = 0.3 + 1.4 * r(), T2 = 3.5 / v;
      o.proj.push({
        t: i ? 200 + Math.floor(r() * 90) : 40 + Math.floor(r() * 100),
        k: ki, sd, y0, vx: -sd * v, vy: (ty - y0) / T2 + 4.905 * T2
      });
    }
  }

  function genOpts(r) {
    const base = { push: 0, amp: 0, T: 8, ph: 0, proj: [], steps: 960, full: true, cut: false, task: 'live' };
    const u = r();
    if (u < 0.25) {
      const p = supinePose(r);
      return { ...base, tilt: p.tilt, drop: p.drop, settle: 50, pose: 'supine' };
    }
    if (u < 0.5) {
      const p = fallenPose(r);
      return { ...base, tilt: p.tilt, drop: p.drop, settle: 50, pose: 'fallen' };
    }
    if (u < 0.75) return { ...base, tilt: (r() * 2 - 1) * 0.08, drop: 0.12, settle: 0, pose: 'up' };
    const p = pronePose(r);
    return { ...base, tilt: p.tilt, drop: p.drop, settle: 50, pose: 'prone' };
  }

  function esAsk(st, np) {
    const o = genOpts(rng), eps = [], ws = [];
    const parent = st.theta, n = parent.length;
    const sig = Math.max(0.18, 0.22 * Math.pow(0.5, (st.gen || 0) / 6000));
    for (let i = 0; i < np; i++) {
      const e = new Float32Array(n);
      for (let k = 0; k < n; k++) e[k] = randn();
      eps.push(e);
      for (const sg of [1, -1]) {
        const w = new Float32Array(n);
        for (let k = 0; k < n; k++) w[k] = parent[k] + sg * sig * e[k];
        ws.push(w);
      }
    }
    return { o, eps, ws, np, sig };
  }

  function esTell(st, pop, fits) {
    const np = pop.np, all = fits.map(f => f.f), n = st.theta.length;
    st.m = st.m && st.m.length === n ? st.m : new Float32Array(n);
    st.v = st.v && st.v.length === n ? st.v : new Float32Array(n);
    st.k = st.k || 0;
    const ord = all.map((_, i) => i).sort((a, b) => all[a] - all[b]), rk = new Array(all.length);
    ord.forEach((i, rnk) => { rk[i] = rnk / (all.length - 1) - 0.5; });
    const g = new Float32Array(n), sig = pop.sig || 0.12;
    for (let i = 0; i < np; i++) {
      const d = rk[2 * i] - rk[2 * i + 1];
      for (let k = 0; k < n; k++) g[k] += d * pop.eps[i][k];
    }
    const kk = ++st.k;
    const b1 = 0.9, b2 = 0.999, c1 = 1 - Math.pow(b1, kk), c2 = 1 - Math.pow(b2, kk);
    const lr = Math.max(0.02, 0.05 * Math.pow(0.5, (st.gen || 0) / 8000));
    for (let k = 0; k < n; k++) {
      const gk = g[k] / (np * sig);
      st.m[k] = b1 * st.m[k] + (1 - b1) * gk;
      st.v[k] = b2 * st.v[k] + (1 - b2) * gk * gk;
      st.theta[k] += lr * (st.m[k] / c1) / (Math.sqrt(st.v[k] / c2) + 1e-8);
    }
    st.mut = '';
    const bi = ord[ord.length - 1];
    if (all[bi] > 3 && all[bi] > (st.eliteF ?? -1e9) && pop.ws[bi] && pop.ws[bi].length === n) {
      st.theta.set(pop.ws[bi]);
      st.eliteF = all[bi];
      st.mut = `elite ${all[bi].toFixed(2)}`;
      st.m = new Float32Array(n);
      st.v = new Float32Array(n);
      st.k = 0;
    }
    st.H = hiddenOf(st.theta.length);
    st.gen++;
    const nF = fits.length;
    const up = fits.filter(f => f.up && (f.headImpact || 0) < 1.5).length / fits.length;
    st.hist.push({
      m: all.reduce((a, b) => a + b, 0) / all.length,
      b: Math.max(...all), s: up, task: pop.o.pose,
      hy: fits.reduce((a, b) => a + (b.hy || 0), 0) / nF,
      ang: fits.reduce((a, b) => a + (b.ang || 0), 0) / nF,
      hand: fits.reduce((a, b) => a + (b.hand || 0), 0) / nF,
      head: fits.reduce((a, b) => a + (b.headImpact || 0), 0) / nF
    });
    if (st.hist.length > 400) st.hist = st.hist.slice(-400);
  }

  function freshState() {
    const theta = initTheta();
    return {
      theta, H, gen: 0, hist: [], sig: 0.12, lr: 0.01, k: 0,
      m: new Float32Array(theta.length), v: new Float32Array(theta.length)
    };
  }
  return {
    build, Ep, spawnObj, sense, reset, platAng, forward, initTheta, genOpts, esAsk, esTell,
    freshState, fallenPose, supinePose, pronePose, addProjs, randn, rngMk,
    setRng(s) { rng = rngMk(s); },
    getRng: () => rng
  };
}
