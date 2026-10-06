// Shared ragdoll sim + Elman policy. Works in Node and the browser.
// Network: x_t = [obs, a_{t-1}], h_t = leak * h_{t-1} + (1-leak) * tanh(Wx x + Wh h + b), a_t = tanh(Wo h)
export const DT = 1 / 60;
export const H = 20;
export const O = 13;
export const NI = 58;
export const NW = H * (NI + H + 1) + O * (H + 1);
export const LEAK = 0.25;
export const PW = 2;
export const HH = 0.1;
export const MAXTILT = 0.1745;
export const SIG0 = 0.1;
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
    CreateCircle, b2DestroyBody
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
      J.push({ id: r.jointId, lo, hi, n: Math.max(-lo, hi) });
    };
    box(0, 1.26, 0.13, 0.30); box(0, 1.68, 0.11, 0.12, 40);
    for (let s = 0; s < 2; s++) {
      box(0, 1.34, 0.05, 0.14, 50); box(0, 1.07, 0.045, 0.13, 50); box(0, 0.88, 0.05, 0.06, 50);
      box(0, 0.74, 0.07, 0.22); box(0, 0.31, 0.06, 0.21); box(0.06, 0.04, 0.12, 0.04, 90);
    }
    jt(0, 1, 0, 1.56, -0.5, 0.5, 20);
    for (let s = 0; s < 2; s++) {
      const b = 2 + 6 * s;
      jt(0, b, 0, 1.48, -2, 2.5, 40); jt(b, b + 1, 0, 1.2, 0, 2.4, 25); jt(b + 1, b + 2, 0, 0.94, -0.7, 0.7, 10);
      jt(0, b + 3, 0, 0.96, -0.7, 2, 140); jt(b + 3, b + 4, 0, 0.52, -2.4, 0, 140); jt(b + 4, b + 5, 0, 0.1, -0.6, 0.6, 60);
    }
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

  const TOUCH = 0.03, STANCE_OK = 0.72, STANCE_COMFY = 0.38, FEET = [7, 13], HANDS = [4, 10];
  function partClear(p, tp) {
    const q = b2Body_GetPosition(p.id), a = b2Rot_GetAngle(b2Body_GetRotation(p.id)), c = Math.cos(a), s = Math.sin(a);
    let plat = 1e9, gnd = 1e9;
    for (const [sx, sy] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
      const lx = sx * p.hx, ly = sy * p.hy, wx = q.x + c * lx - s * ly, wy = q.y + s * lx + c * ly;
      plat = Math.min(plat, wy - wx * tp);
      gnd = Math.min(gnd, wy + 0.5);
    }
    return Math.min(plat, gnd);
  }
  function sense(R) {
    const P = R.P, ys = new Float32Array(P.length), pa = b2Rot_GetAngle(b2Body_GetRotation(R.plat)), tp = Math.tan(pa);
    let X = 0, Y = 0, VX = 0, VY = 0, M = 0, m = { ys, pa, pw: b2Body_GetAngularVelocity(R.plat), touch: new Array(P.length) };
    for (let i = 0; i < P.length; i++) {
      const p = P[i], q = b2Body_GetPosition(p.id), v = b2Body_GetLinearVelocity(p.id);
      ys[i] = q.y - q.x * tp; M += p.m; X += p.m * q.x; Y += p.m * ys[i]; VX += p.m * v.x; VY += p.m * v.y;
      m.touch[i] = partClear(p, tp) < TOUCH;
      if (i === 0) { m.tx = q.x; m.tvx = v.x; m.tvy = v.y; }
      if (i === 1) m.hvy = v.y;
      if (i === 7) m.fx0 = q.x;
      if (i === 13) m.fx1 = q.x;
    }
    m.ty = ys[0]; m.hy = ys[1]; m.fy0 = ys[7]; m.fy1 = ys[13];
    m.cx = X / M; m.cy = Y / M; m.cvx = VX / M; m.cvy = VY / M;
    m.a = b2Rot_GetAngle(b2Body_GetRotation(P[0].id)); m.w = b2Body_GetAngularVelocity(P[0].id);
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
    m.bodyOn = false;
    for (let i = 0; i < P.length; i++) if (!FEET.includes(i) && m.touch[i]) m.bodyOn = true;
    m.stance = Math.abs(m.fx0 - m.fx1);
    return m;
  }

  const ttf = (y, vy) => cl((vy + Math.sqrt(vy * vy + 19.62 * Math.max(y - 0.15, 0))) / 9.81, 0, 1.5) / 1.5;

  function forward(w, x, h) {
    let k = 0;
    const nh = new Float32Array(H), y = new Float32Array(O);
    for (let j = 0; j < H; j++) {
      let s = w[k++];
      for (let i = 0; i < NI; i++) s += w[k++] * x[i];
      for (let i = 0; i < H; i++) s += w[k++] * h[i];
      nh[j] = LEAK * h[j] + (1 - LEAK) * Math.tanh(s);
    }
    for (let j = 0; j < O; j++) {
      let s = w[k++];
      for (let i = 0; i < H; i++) s += w[k++] * nh[i];
      y[j] = Math.tanh(s);
    }
    return [y, nh];
  }

  class Ep {
    constructor(R, w, tilt, drop, o = {}) {
      this.o = Object.assign({ push: 0, amp: 0, T: 8, ph: 0, proj: [], steps: 480, task: 'stand' }, o);
      this.free = false; this.fallen = false; this.objs = [];
      this.R = R; this.w = w; this.t = 0; this.ret = 0; this.dead = false;
      this.h = new Float32Array(H); this.a = new Float32Array(O);
      this.upStreak = 0; this.recovered = false; this.stood = 0; this.nDec = 0; this.sumHy = 0;
      this.wasDown = false; this.quiet = false;
      this.prevA = new Float32Array(O);
      if (this.o.task === 'getup') {
        reset(R, 0.03, 0.14, { amp: 0, T: 8, ph: 0, proj: [] });
        this.pa = R.J.map(j => b2RevoluteJoint_GetAngle(j.id));
        this.m = sense(R);
        this.quiet = true;
        for (let i = 0; i < 10; i++) this.decide();
        this.quiet = false;
      }
      reset(R, tilt, drop, this.o);
      this.pa = R.J.map(j => b2RevoluteJoint_GetAngle(j.id));
      this.m = sense(R);
      this.t = 0; this.ret = 0; this.upStreak = 0; this.stood = 0; this.nDec = 0; this.sumHy = 0;
      this.wasDown = false; this.recovered = false; this.fallen = false; this.prevHy = undefined;
    }
    cleanup() { for (const id of this.objs) b2DestroyBody(id); this.objs = []; }
    decide() {
      const R = this.R, m = this.m = sense(R), fmx = (m.fx0 + m.fx1) / 2;
      const gg = Math.sqrt(Math.max(m.cy, 0.3) / 9.81), cp = m.cx + m.cvx * gg, up = Math.max(0, Math.cos(m.a));
      let en = 0, mot = 0, jerk = 0;
      for (let i = 0; i < O; i++) {
        const j = R.J[i], a = this.a[i], an = b2RevoluteJoint_GetAngle(j.id);
        en += a * a;
        mot += Math.abs(a * (a >= 0 ? j.hi : -j.lo) - an) / j.n;
        jerk += Math.abs(a - this.prevA[i]);
      }
      const hy = cl(m.hy / 1.65, 0, 1);
      const standing = m.footOn && !m.bodyOn && up > 0.5 && m.hy > 0.9 && m.stance < STANCE_OK;
      this.fallen = m.bodyOn || up < 0.25;
      if (!this.quiet) {
        if (this.fallen) this.wasDown = true;
        if (standing) {
          this.upStreak++; this.stood++;
          if (this.wasDown && this.upStreak >= 30 && this.t > 90) {
            if (!this.recovered) this.ret += 28;
            this.recovered = true;
          }
        } else this.upStreak = 0;
        this.nDec++; this.sumHy += m.hy;
        if (standing) {
          let fp = 0; for (let i = 0; i < 4; i++) fp += cl(m.cl[i] / 0.06, 0, 1);
          this.ret += 1.1 + 0.55 * up + 0.7 * hy + 0.5 * Math.exp(-4 * Math.abs(cp - fmx))
            + 0.7 * Math.exp(-5 * Math.max(0, m.stance - STANCE_COMFY))
            - 0.55 * fp - 0.04 * en / O - 0.22 * mot / O - 0.14 * jerk / O
            - 0.12 * cl(Math.abs(m.w) / 6, 0, 1) - 0.08 * cl(Math.abs(m.cvx) / 2, 0, 1);
        } else {
          const dhy = m.hy - (this.prevHy ?? m.hy);
          const ty = cl(m.ty / 1.26, 0, 1);
          this.ret += 0.02 + 0.5 * (m.handOn ? 1 : 0) + 0.25 * (m.footOn ? 1 : 0)
            + 0.6 * up + 0.7 * ty + 0.15 * hy
            + 1.0 * cl(dhy, -0.04, 0.06) - 0.02 * en / O;
        }
        this.prevHy = m.hy;
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
        hy: this.nDec ? this.sumHy / this.nDec : 0, stood: this.stood / Math.max(1, this.nDec)
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

  function fallenPose(r, hard = 0.7) {
    const side = r() < 0.5 ? 1 : -1;
    const u = r();
    if (u < 0.4 - 0.15 * hard) return { tilt: side * (0.38 + r() * 0.32), drop: 0.05 + r() * 0.08 };
    if (u < 0.72) return { tilt: side * (0.78 + r() * 0.38), drop: 0.07 + r() * 0.1 };
    return { tilt: side * (1.25 + r() * (0.45 + 0.9 * hard)), drop: 0.07 + r() * 0.1 };
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

  function genOpts(r, st) {
    const u = r();
    if (u < 0.34) {
      const p = fallenPose(r, Math.min(1, 0.2 + 0.8 * (st.G || 0)));
      return {
        tilt: p.tilt, drop: p.drop, push: 0,
        amp: st.Pa * MAXTILT * 0.2 * r(), T: 5 + 3 * r(), ph: 6.2832 * r(),
        proj: [], task: 'getup', steps: 780
      };
    }
    if (u < 0.52) {
      const sd = r() < 0.5 ? -1 : 1;
      return {
        tilt: (r() * 2 - 1) * 0.08, drop: 0.12 + r() * 0.08,
        push: sd * (4.5 + r() * 2.2),
        amp: st.Pa * MAXTILT * 0.25 * r(), T: 5 + 3 * r(), ph: 6.2832 * r(),
        proj: [], task: 'recover', steps: 780
      };
    }
    const o = {
      tilt: (r() * 2 - 1) * st.A, drop: 0.12 + r() * 0.25,
      push: (r() * 2 - 1) * st.Pm,
      amp: st.Pa * MAXTILT * (0.5 + 0.5 * r()), T: 5 + 3 * r(), ph: 6.2832 * r(),
      proj: [], task: 'stand', steps: 540
    };
    addProjs(r, o, st.Q);
    return o;
  }

  function esAsk(st, np) {
    const o = genOpts(rng, st), eps = [], ws = [];
    const sig = st.sig || SIG0;
    for (let i = 0; i < np; i++) {
      const e = new Float32Array(NW);
      for (let k = 0; k < NW; k++) e[k] = randn();
      eps.push(e);
      for (const sg of [1, -1]) {
        const w = new Float32Array(NW);
        for (let k = 0; k < NW; k++) w[k] = st.theta[k] + sg * sig * e[k];
        ws.push(w);
      }
    }
    return { o, eps, ws, np };
  }

  function esTell(st, pop, fits) {
    const np = pop.np, all = fits.map(f => f.f);
    st.m = st.m || new Float32Array(NW);
    st.v = st.v || new Float32Array(NW);
    st.k = st.k || 0;
    const ord = all.map((_, i) => i).sort((a, b) => all[a] - all[b]), rk = new Array(all.length);
    ord.forEach((i, rnk) => { rk[i] = rnk / (all.length - 1) - 0.5; });
    const g = new Float32Array(NW), sig = st.sig || SIG0;
    for (let i = 0; i < np; i++) {
      const d = rk[2 * i] - rk[2 * i + 1];
      for (let k = 0; k < NW; k++) g[k] += d * pop.eps[i][k];
    }
    st.k++;
    const b1 = 0.9, b2 = 0.999, c1 = 1 - Math.pow(b1, st.k), c2 = 1 - Math.pow(b2, st.k);
    const lr = st.lr || 0.01;
    for (let k = 0; k < NW; k++) {
      const gk = g[k] / (np * sig);
      st.m[k] = b1 * st.m[k] + (1 - b1) * gk;
      st.v[k] = b2 * st.v[k] + (1 - b2) * gk * gk;
      st.theta[k] += lr * (st.m[k] / c1) / (Math.sqrt(st.v[k] / c2) + 1e-8);
    }
    const rec = fits.filter(f => f.rec).length / fits.length;
    const up = fits.filter(f => f.up).length / fits.length;
    const stoodAvg = fits.reduce((a, b) => a + (b.stood || 0), 0) / fits.length;
    const standTask = pop.o.task === 'stand';
    const surv = standTask ? up : rec;
    if (standTask) {
      if (stoodAvg >= 0.4 || up >= 0.3) {
        if (st.A < 0.4) st.A = Math.min(0.4, st.A + 0.02);
        else {
          st.Pa = Math.min(1, Math.max(st.Pa || 0, 0.75) + 0.01);
          st.Q = Math.min(1, (st.Q || 0) + 0.012);
          st.Pm = Math.min(1.2, (st.Pm || 0) + 0.015);
        }
      } else if (stoodAvg < 0.15) {
        if (st.Q > 0.3) { st.Q = Math.max(0.3, st.Q - 0.02); st.Pm = Math.max(0.35, (st.Pm || 0) - 0.02); }
      }
    } else if (pop.o.task === 'getup' || pop.o.task === 'recover') {
      if (rec >= 0.08) st.G = Math.min(0.55, (st.G || 0) + 0.03);
      else st.G = Math.min(0.5, (st.G || 0) + 0.005);
    }
    st.gen++;
    st.hist.push({
      m: all.reduce((a, b) => a + b, 0) / all.length,
      b: Math.max(...all), s: surv, Q: st.Q, Pa: st.Pa, G: st.G || 0, task: pop.o.task
    });
    if (st.hist.length > 400) st.hist = st.hist.slice(-400);
  }

  function freshState() {
    return { theta: initTheta(), gen: 0, hist: [], A: 0.12, Pm: 0, Q: 0, Pa: 0, G: 0, sig: SIG0, lr: 0.01 };
  }

  return {
    build, Ep, spawnObj, sense, reset, platAng, forward, initTheta, genOpts, esAsk, esTell,
    freshState, fallenPose, addProjs, randn, rngMk,
    setRng(s) { rng = rngMk(s); },
    getRng: () => rng
  };
}
