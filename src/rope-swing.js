// ROPE SWING (rope.js tunes it; the playground uses it): the parkour runner runs and jumps at the edge of
// a hole; in the air the arms reach up and forward and the hands close on the rope when they come near
// it; hanging on, the legs tuck to clear the drop, then reach forward for the landing; past the far
// edge the hands open, the runner takes the landing as the end of a jump and runs on.
//   phases: run -> reach (jumped) -> swing (a hand holds) -> land (let go) -> run2 (back on the ground)
import { makeController } from './runner.js';
import { makeController as makeWalker } from './simbicon.js';

// [name, start, search spread]
export const SPEC = [
  ['catchFwd', 0.25, 0.12], // the hands aim at the rope point nearest to this far ahead of the shoulder
  ['catchUp', 0.45, 0.12], // and this far above it
  ['catchR', 0.18, 0.06], // a hand closes when it is this close to the rope
  ['tuckHip', 1.8, 0.3], // hips while swinging (thighs up, + = flexed)
  ['tuckKnee', -2.0, 0.3], // knees while swinging
  ['extDx', 0.0, 0.2], // the legs reach forward for the landing with the hips this far from 0.3 m before the far edge
  ['preHip', 0.9, 0.3], // hips then
  ['preKnee', -0.3, 0.2], // knees then
  ['relDx', 0.1, 0.2], // the hands open with the hips this far past the far edge
  ['holdElbow', 0.3, 0.3], // elbows while hanging (bent pulls the body up toward the rope)
  ['takeoff', 0.0, 0.15], // nudge to the jump stride selection margin (m)
  ['landSpeed', 0.0, 0.5], // the runner's target speed after landing, relative to before
  ['lean', 0.0, 0.3], // trunk (waist) while swinging
  ['relV', 0.5, 0.3], // let go over the far floor moving faster than this (or nearly still at the end of the swing)
  ['landV', 1.3, 0.3], // let go faster than this: land running; slower: drop, steady, walk off, run
  ['dropHip', 0.3, 0.2], ['dropKnee', -0.4, 0.2], // legs while dropping onto the far floor (thigh world angle, knee)
  ['steadyT', 1.2, 0.4], // seconds the stand network steadies the body after the drop
  ['relCos', 0.8, 0.1], // let go only with the trunk this upright (cosine of its tilt)
  ['poseHold', 1, 0], // > 0: steady with the pose executor instead of the stand network
  ['holdH', 0.75, 0.08], ['holdLean', -0.35, 0.15], // its crouch height and trunk lean at the start
  ['relLand', 0.5, 0.2], // let go when a free flight from here lands this far past the far edge
  ['ropeLandLean', -0.2, 0.15], ['ropeLandKnee', 0, 0.2], ['ropeLandPlace', 0, 0.2], // the landing stride after the rope
  ['landArm', 1.5, 0.5], // arms (world angle) in the air after letting go
  ['plant', 1, 0], // > 0: feet down on the far floor still holding: stand up over the feet first, then let go
  ['plantT', 1.2, 0.4], ['plantDx', 0.03, 0.05],
  ['holdShoulder', 2.4, 0.3], // holding arm angle to the trunk while swinging (0: keep the angle it has)
];

export function installRopeRunner(ep, world, p, ctx) {
  const { config } = ctx;
  const HZ = config.physics.hz, J = config.net.jointCount, P = config.partIndex;
  const runner = makeController(world);
  const walker = makeWalker(world);
  const rp = Object.assign({}, ctx.runnerP);
  // the landing stride after the rope: its own changes to the runner's landing (a rope landing comes in
  // leaning back, not as a jump does)
  const rpLand = Object.assign({}, rp, { landLean: (rp.landLean || 0) + (p.ropeLandLean || 0), landKnee: (rp.landKnee || 0) + (p.ropeLandKnee || 0), landPlace: (rp.landPlace || 0) + (p.ropeLandPlace || 0) });
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
    m.walkSpeed = S.phase === 'run2' ? Math.min(ctx.speed(), 1.2 + S.t * 0.6) : ctx.speed() * (S.phase === 'land' ? 1 + p.landSpeed * 0.3 : 1);
    this.speedCap = config.body.maxJointSpeed;
    const held = [0, 1].map((k) => !!(this.grips[k] && this.grips[k].joint));
    if ((S.phase === 'run' || S.phase === 'reach') && (held[0] || held[1])) { S.phase = 'swing'; S.t = 0; }
    if (S.phase === 'run' || S.phase === 'reach' || S.phase === 'land' || S.phase === 'run2') {
      runner(S.phase === 'land' || (S.phase === 'run2' && S.t < 0.6) ? rpLand : rp, this, tmp);
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
      if (S.phase === 'land' && !(m.touch[P.footLeft] || m.touch[P.footRight])) {
        // in the air after letting go: arms forward and the trunk folded forward at the waist, to land over the feet
        const torso = angleOf(0);
        for (const [sh, el] of [[2, 3], [8, 9]]) { tmp[sh] = command(sh, p.landArm - torso); tmp[el] = command(el, 0.4); }
        tmp[0] = command(0, -0.2);
      }
      if (S.phase === 'reach') {
        // arms up and forward; a hand near the rope closes
        // each hand aims at the rope: the point of it nearest to up-and-forward of the shoulder (IK)
        const ca = angleOf(1);
        const shW = { x: world.px[1] - Math.sin(ca) * 0.10, y: world.py[1] + Math.cos(ca) * 0.10 };
        const want = { x: shW.x + p.catchFwd, y: shW.y + p.catchUp };
        let best = null, prev = { x: rope.x, y: rope.top };
        for (const id of rope.ids) {
          const v = getPos(id); const dx = v.x - prev.x, dy = v.y - prev.y, L2 = dx * dx + dy * dy || 1e-9;
          const t = Math.max(0, Math.min(1, ((want.x - prev.x) * dx + (want.y - prev.y) * dy) / L2));
          const q = { x: prev.x + t * dx, y: prev.y + t * dy }, d = Math.hypot(q.x - want.x, q.y - want.y);
          if (!best || d < best.d) best = { x: q.x, y: q.y, d };
          prev = { x: v.x, y: v.y };
        }
        const ik = ctx.pose.twoLink(shW, best || want, 0.28, 0.29, 1);
        for (const [sh, el] of [[2, 3], [8, 9]]) { this.action[sh] = command(sh, ik.upper - ca); this.action[el] = command(el, ik.middle); }
        for (let k = 0; k < 2; k++) { const hx = world.px[HAND[k]], hy = world.py[HAND[k]]; const d = ropeDist(hx, hy); this.action[J + k] = d < p.catchR ? 1 : -1; if (ctx.debug && k === 0 && this.frame % 3 === 0) { const b = getPos(rope.ids[rope.ids.length - 1]); console.log('  reach t', S.t.toFixed(2), 'hand', hx.toFixed(2), hy.toFixed(2), 'rope d', d.toFixed(2), 'want', best ? best.x.toFixed(2) + ',' + best.y.toFixed(2) : '-', 'rope end', b.x.toFixed(2), b.y.toFixed(2), 'pelvis', world.px[0].toFixed(2), world.py[0].toFixed(2)); } }
      }
    } else if (S.phase === 'drop' || S.phase === 'steady' || S.phase === 'walk') {
      // slow landing: legs under the body until the feet are down, the stand network steadies it, then the
      // walker walks off and the runner takes over at a touchdown
      this.speedCap = 0;
      if (S.phase === 'drop') {
        const torso = angleOf(0);
        for (const [h, kn, an] of [[5, 6, 7], [11, 12, 13]]) { this.action[h] = command(h, p.dropHip - torso); this.action[kn] = command(kn, p.dropKnee); this.action[an] = command(an, 0); }
        for (const [sh, el] of [[2, 3], [8, 9]]) { this.action[sh] = command(sh, 0.5 - torso); this.action[el] = command(el, 0.5); }
        this.action[0] = 0; this.action[1] = 0; this.action[J] = this.action[J + 1] = -1;
        if (m.touch[P.footLeft] && m.touch[P.footRight] || S.t > 1.5) { S.phase = 'steady'; S.t = 0; }
      } else if (S.phase === 'steady') {
        this.groundOffset = ctx.lift || 0;
        // the stand network's marks on this floor (at y = 0 they were 2 m below it and it reached for them)
        if (!this.options.targets || this.options.targets.length !== config.targets.length) this.options.targets = config.targets.map((t) => Object.assign({}, t));
        for (let i = 0; i < config.targets.length; i++) {
          const spec = config.targets[i], mk = this.options.targets[i];
          if (spec.ride) { mk.x = world.px[0] + (spec.x || 0); mk.y = (ctx.lift || 0) + (spec.y || 0); }
          else { mk.x = world.px[spec.id === 'footR' ? P.footRight : P.footLeft]; mk.y = (ctx.lift || 0) + config.skill.footGroundMeters; }
        }
        if (p.poseHold > 0) {
          // or the pose executor: crouch over the feet, trunk a little forward, balance on hips and ankles
          if (!S.feet) S.feet = [P.footLeft, P.footRight].map((q) => ctx.pose.soleOf(world, q));
          const mid = (S.feet[0].x + S.feet[1].x) / 2, q = Math.min(1, S.t / p.steadyT);
          const goal = { pelvis: { x: mid, y: (ctx.lift || 0) + p.holdH + (0.92 - p.holdH) * q }, torso: p.holdLean * (1 - q), feet: S.feet.map((f) => ({ x: f.x, y: f.y, angle: 0 })), planted: [true, true] };
          ctx.pose.commandsFor(world, ctx.pose.balancedTargets(goal, m, { planted: [true, true], kHip: 1.5, kAnkle: 3.1, kAnkleVel: 0.15, kHipVel: 0.1 }, ctx.pose.actualOf(world)), this.action);
        } else for (let j = 0; j < J; j++) this.action[j] = this.network.action[j];
        this.action[J] = this.action[J + 1] = -1;
        if (S.t > p.steadyT + (p.poseHold > 0 ? 0.5 : 0)) { S.phase = 'walk'; S.t = 0; this.simbicon = { stance: 0, t: 0 }; }
      } else {
        m.walkSpeed = Math.min(1.2, 0.3 + S.t * 0.6); m.walkDir = 1;
        walker(ctx.walkP, this, tmp);
        for (let j = 0; j < tmp.length; j++) this.action[j] = j < J ? tmp[j] : -1;
        const sb = this.simbicon;
        if (S.t > 2.5 && sb && sb.t < 1.5 / HZ) { S.phase = 'run2'; S.t = 0; this.runner = { stance: sb.stance, t: 0, flight: false, jumping: false, jumpNext: false, strides: 0 }; }
      }
    } else if (S.phase === 'plant') {
      // three or four points down: feet on the floor, hands on the rope; the hips come forward over the
      // feet and the trunk straightens (pose executor), then the hands open and it steadies
      this.speedCap = 0;
      const mid = (S.feet[0].x + S.feet[1].x) / 2, q = Math.min(1, S.t / p.plantT);
      const pv = ctx.pose.pelvisOf(world);
      if (!S.pel0) S.pel0 = { x: pv.x, y: pv.y, a: pv.angle };
      const goal = {
        pelvis: { x: S.pel0.x + (mid + p.plantDx - S.pel0.x) * q, y: S.pel0.y + ((ctx.lift || 0) + 0.9 - S.pel0.y) * q },
        torso: S.pel0.a * (1 - q), feet: S.feet.map((f) => ({ x: f.x, y: f.y, angle: 0 })), planted: [true, true],
        hands: [0, 1].map((k) => (held[k] ? { x: world.px[HAND[k]], y: world.py[HAND[k]] } : null)), gripping: held.slice(),
      };
      ctx.pose.commandsFor(world, ctx.pose.jointTargets(goal, ctx.pose.actualOf(world)), this.action);
      for (let k = 0; k < 2; k++) this.action[J + k] = held[k] ? 1 : -1;
      const err = Math.abs((m.centerX || 0) - mid);
      if ((q >= 1 && err < 0.08 && (m.torsoCos || 0) > 0.9) || S.t > p.plantT + 1.5) { this.action[J] = this.action[J + 1] = -1; S.phase = 'steady'; S.t = 0; S.feet = null; }
    } else if (S.phase === 'swing') {
      // hanging on: the arms keep the angle of the holding arm, the other hand closes when it gets to the
      // rope; the legs tuck, then reach forward; past the far edge the hands open
      const k0 = held[0] ? 0 : 1;
      const shA = world.joints[k0 === 0 ? 2 : 8];
      const a = ctx.Box2D.b2RevoluteJoint_GetAngle(shA.id);
      // the holding arm keeps its angle and bends its elbow (pulling the body up); the free hand reaches
      // for the rope just under the holding one (IK) and closes there
      const ca = angleOf(1);
      const shW = { x: world.px[1] - Math.sin(ca) * 0.10, y: world.py[1] + Math.cos(ca) * 0.10 };
      for (let k = 0; k < 2; k++) {
        const sh = k === 0 ? 2 : 8, el = k === 0 ? 3 : 9;
        // the holding arm: overhead at a set angle to the trunk (keeping whatever angle it had let the body
        // spin round the hands and flip)
        if (held[k]) { this.action[sh] = command(sh, p.holdShoulder > 0 ? p.holdShoulder : a); this.action[el] = command(el, p.holdElbow); continue; }
        const hk = HAND[k0];
        const ik = ctx.pose.twoLink(shW, { x: world.px[hk], y: world.py[hk] - 0.06 }, 0.28, 0.29, 1);
        this.action[sh] = command(sh, ik.upper - ca); this.action[el] = command(el, ik.middle);
      }
      const px = world.px[0];
      // the legs stay tucked up until the hips are over the far edge (reaching down earlier, the feet hit
      // the far wall), then reach forward for the landing
      const ext = px > hole.x1 + p.extDx - 0.3;
      for (const [h, kn, an] of [[5, 6, 7], [11, 12, 13]]) { this.action[h] = command(h, ext ? p.preHip : p.tuckHip); this.action[kn] = command(kn, ext ? p.preKnee : p.tuckKnee); this.action[an] = command(an, 0.2); }
      this.action[0] = command(0, p.lean); this.action[1] = 0; this.action[4] = 0; this.action[10] = 0;
      for (let k = 0; k < 2; k++) {
        if (held[k]) { this.action[J + k] = 1; continue; }
        const hx = world.px[HAND[k]], hy = world.py[HAND[k]];
        this.action[J + k] = ropeDist(hx, hy) < p.catchR ? 1 : -1;
      }
      const vx = m.centerVelocityX || 0;
      // past the far edge (relDx), or near the forward end of the swing (slower than relV) and past the rope
      // let go over the far floor: moving on (or at the end of the swing, nearly still)
      // let go as the feet come over the far floor, still moving on (later, the feet were on the floor with
      // the rope pulling the body back and it fell over backward); or if the feet already touch it
      const feetX = Math.min(world.px[P.footLeft], world.px[P.footRight]);
      const feetOver = feetX > hole.x1 + p.relDx;
      const touching = (m.touch[P.footLeft] || m.touch[P.footRight]) && px > hole.x1;
      // BALLISTIC: let go as soon as the centre of mass, flying free from here, would come down (to standing
      // height over the far floor) at least relLand past the edge, the trunk upright enough
      const cy = m.centerY || world.py[0], vy = m.centerVelocityY || 0, yl = (ctx.lift || 0) + 0.85;
      const disc = vy * vy + 2 * 9.81 * Math.max(0, cy - yl);
      const tl = (vy + Math.sqrt(disc)) / 9.81;
      const xLand = (m.centerX || px) + vx * tl;
      const ballistic = p.relLand != null && vx > 0.5 && xLand > hole.x1 + p.relLand && cy > yl - 0.1;
      if (touching && p.plant > 0) {
        // feet on the far floor still holding the rope: keep holding, stand up over the feet, then let go
        S.phase = 'plant'; S.t = 0; S.feet = [P.footLeft, P.footRight].map((q) => ctx.pose.soleOf(world, q));
        S.hands = [0, 1].map((k) => (held[k] ? { x: world.px[HAND[k]], y: world.py[HAND[k]] } : null));
      } else if ((ballistic && (m.torsoCos || 0) > p.relCos) || (p.relLand == null && feetOver && vx > p.relV && (m.torsoCos || 0) > p.relCos) || touching) {
        this.action[J] = this.action[J + 1] = -1;
        if (ctx.debug) console.log('  release: torso', angleOf(0).toFixed(2), 'cos', (m.torsoCos || 0).toFixed(2), 'vx', vx.toFixed(2), 'vy', (m.centerVelocityY || 0).toFixed(2), 'pelvis above floor', (world.py[0] - (ctx.lift || 0)).toFixed(2), 'past edge', (px - hole.x1).toFixed(2));
        if (vx > p.landV || !ctx.walkP) {
          // fast: the runner takes over in the air, the front foot reaching for the landing
          S.phase = 'land'; S.t = 0;
          const rear = world.px[P.footLeft] < world.px[P.footRight] ? 0 : 1;
          this.runner = { stance: rear, t: 0.2, flight: true, jumping: false, landing: true, jumpNext: false, strides: 0 };
        } else { S.phase = 'drop'; S.t = 0; }
      }
    }
    orig();
  })(ep.updateGrip.bind(ep));
  return S;
}
