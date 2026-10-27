// ROPE SWING (rope.js tunes it; the playground uses it): the parkour runner runs and jumps at the edge of
// a hole; in the air the arms reach up and forward and the hands close on the rope when they come near
// it; hanging on, the legs tuck to clear the drop, then reach forward for the landing; past the far
// edge the hands open, the runner takes the landing as the end of a jump and runs on.
//   phases: run -> reach (jumped) -> swing (a hand holds) -> land (let go) -> run2 (back on the ground)
import { makeController } from './runner.js';

// [name, start, search spread]
export const SPEC = [
  ['armUp', 2.5, 0.3], // upper-arm world angle reaching for the rope (0 hanging, + forward and up)
  ['armElbow', 0.15, 0.15], // elbows while reaching
  ['catchR', 0.18, 0.06], // a hand closes when it is this close to the rope
  ['tuckHip', 1.3, 0.3], // hips while swinging (thighs up, + = flexed)
  ['tuckKnee', -1.6, 0.3], // knees while swinging
  ['extDx', 0.2, 0.2], // past the rope by this much the legs reach forward for the landing
  ['preHip', 0.9, 0.3], // hips then
  ['preKnee', -0.3, 0.2], // knees then
  ['relDx', 0.1, 0.2], // the hands open with the hips this far past the far edge
  ['holdElbow', 0.3, 0.3], // elbows while hanging (bent pulls the body up toward the rope)
  ['takeoff', 0.0, 0.15], // nudge to the jump stride selection margin (m)
  ['landSpeed', 0.0, 0.5], // the runner's target speed after landing, relative to before
  ['lean', 0.0, 0.3], // trunk (waist) while swinging
  ['relV', 0.5, 0.3], // or near the forward end of the swing, slower than this (m/s)
];

export function installRopeRunner(ep, world, p, ctx) {
  const { config } = ctx;
  const HZ = config.physics.hz, J = config.net.jointCount, P = config.partIndex;
  const runner = makeController(world);
  const rp = Object.assign({}, ctx.runnerP);
  const tmp = new Float64Array(J + config.net.gripOutputs);
  const hole = ctx.hole, rope = ctx.rope;
  const getPos = ctx.Box2D.b2Body_GetPosition;
  const angleOf = (i) => Math.atan2(world.s[i], world.c[i]);
  const command = (j, target) => { const jt = world.joints[j]; const t = Math.max(jt.lo, Math.min(jt.hi, target)); return t >= 0 ? (jt.hi > 0 ? t / jt.hi : 0) : jt.lo < 0 ? t / -jt.lo : 0; };
  const HAND = [P.handLeft, P.handRight];
  const ropeDist = (hx, hy) => {
    let best = 9;
    for (const id of rope.ids) { const v = getPos(id); best = Math.min(best, Math.hypot(hx - v.x, hy - v.y) - rope.seg / 2 * 0); }
    // along the segments: distance to the polyline through their centres
    let prev = { x: rope.x, y: rope.top };
    for (const id of rope.ids) {
      const v = getPos(id);
      const dx = v.x - prev.x, dy = v.y - prev.y, L2 = dx * dx + dy * dy || 1e-9;
      const t = Math.max(0, Math.min(1, ((hx - prev.x) * dx + (hy - prev.y) * dy) / L2));
      best = Math.min(best, Math.hypot(hx - prev.x - t * dx, hy - prev.y - t * dy));
      prev = { x: v.x, y: v.y };
    }
    return best;
  };
  const S = { phase: 'run', t: 0 };
  ep.updateGrip = ((orig) => function () {
    const m = this.measure;
    S.t += 1 / HZ;
    m.walkSpeed = ctx.speed() * (S.phase === 'land' || S.phase === 'run2' ? 1 + p.landSpeed * 0.3 : 1);
    this.speedCap = config.body.maxJointSpeed;
    const held = [0, 1].map((k) => !!(this.grips[k] && this.grips[k].joint));
    if ((S.phase === 'run' || S.phase === 'reach') && (held[0] || held[1])) { S.phase = 'swing'; S.t = 0; }
    if (S.phase === 'run' || S.phase === 'reach' || S.phase === 'land' || S.phase === 'run2') {
      runner(rp, this, tmp);
      const st = this.runner;
      // the jump: on the stride whose ordinary landing would fall near or past the edge
      if (S.phase === 'run' && st && st.t === 1 / HZ) {
        const fx = world.px[st.stance ? P.footRight : P.footLeft];
        if (st.lastX != null) { const L = fx - st.lastX; if (L > 0.2 && L < 2.5) st.avgL = st.avgL ? 0.7 * st.avgL + 0.3 * L : L; }
        st.lastX = fx;
        const win = (st.avgL ? st.avgL + rp.jumpMargin : rp.jumpWin) + p.takeoff;
        st.placeAdj = 0;
        if (hole.x0 - fx > 0.03 && hole.x0 - fx < win) st.jumping = true;
        else if (st.avgL && rp.kStride) {
          const D = hole.x0 - fx - rp.takeoff;
          if (D > 0 && D < 5) { const n = Math.max(1, Math.round(D / st.avgL)); st.placeAdj = Math.max(-0.35, Math.min(0.35, rp.kStride * (D / n - st.avgL))); }
        }
      }
      if (S.phase === 'run' && st && st.jumping && st.flight) { S.phase = 'reach'; S.t = 0; }
      for (let j = 0; j < tmp.length; j++) this.action[j] = j < J ? tmp[j] : -1;
      if (S.phase === 'land' && S.t > 0.3 && (m.touch[P.footLeft] || m.touch[P.footRight]) && S.t > 1.2) { S.phase = 'run2'; S.t = 0; }
      if (S.phase === 'reach') {
        // arms up and forward; a hand near the rope closes
        const torso = angleOf(0);
        for (const [sh, el] of [[2, 3], [8, 9]]) { this.action[sh] = command(sh, p.armUp - torso); this.action[el] = command(el, p.armElbow); }
        for (let k = 0; k < 2; k++) { const hx = world.px[HAND[k]], hy = world.py[HAND[k]]; this.action[J + k] = ropeDist(hx, hy) < p.catchR ? 1 : -1; }
      }
    } else if (S.phase === 'swing') {
      // hanging on: the arms keep the angle of the holding arm, the other hand closes when it gets to the
      // rope; the legs tuck, then reach forward; past the far edge the hands open
      const k0 = held[0] ? 0 : 1;
      const shA = world.joints[k0 === 0 ? 2 : 8], elA = world.joints[k0 === 0 ? 3 : 9];
      const a = ctx.Box2D.b2RevoluteJoint_GetAngle(shA.id);
      for (const [sh, el] of [[2, 3], [8, 9]]) { this.action[sh] = command(sh, a); this.action[el] = command(el, p.holdElbow); }
      const px = world.px[0];
      const ext = px > rope.x + p.extDx;
      for (const [h, kn, an] of [[5, 6, 7], [11, 12, 13]]) { this.action[h] = command(h, ext ? p.preHip : p.tuckHip); this.action[kn] = command(kn, ext ? p.preKnee : p.tuckKnee); this.action[an] = command(an, 0.2); }
      this.action[0] = command(0, p.lean); this.action[1] = 0; this.action[4] = 0; this.action[10] = 0;
      for (let k = 0; k < 2; k++) {
        if (held[k]) { this.action[J + k] = 1; continue; }
        const hx = world.px[HAND[k]], hy = world.py[HAND[k]];
        this.action[J + k] = ropeDist(hx, hy) < p.catchR ? 1 : -1;
      }
      const vx = m.centerVelocityX || 0;
      // past the far edge (relDx), or near the forward end of the swing (slower than relV) and past the rope
      if ((px > hole.x1 + p.relDx && vx > 0) || (px > rope.x + 0.2 && vx > 0 && vx < p.relV && S.t > 0.4)) {
        // let go: the runner takes over in the air, the front foot reaching for the landing
        S.phase = 'land'; S.t = 0;
        this.action[J] = this.action[J + 1] = -1;
        const rear = world.px[P.footLeft] < world.px[P.footRight] ? 0 : 1;
        this.runner = { stance: rear, t: 0.2, flight: true, jumping: false, landing: true, jumpNext: false, strides: 0 };
      }
    }
    orig();
  })(ep.updateGrip.bind(ep));
  return S;
}
