// The running controller (src/runner.js) with the jump selector of parkour.js: at each touchdown, jump on the
// stride whose ordinary landing would fall within jumpMargin of the next obstacle's front edge, and before
// that stretch or shorten the steps toward a take-off spot. Installs itself on an episode (wraps
// ep.updateGrip). Used by the playground and by dash.js (random boxes and balls).
//
// ctx.obstacles: [{x0, x1, top?}] in world metres (a hole has no top), or {live: () => ({x0, x1, top})}
// for something that moves (a ball read from its body each time). With a top, the jump is scaled by its
// height: jumpScale = 1 + jumpGainH * (top - 0.2) / 0.2 (all of the runner's jump changes), and the
// take-off window grows by winH * (top - 0.2). Both 0 (or missing) = the old selector exactly.
// ctx.speed(): the commanded speed; without it, p.vRun.
// ctx.net: an optional corrective network (src/dash-net.js) run every frame after the runner.
import { makeController } from './runner.js';

export const HREF = 0.2; // the obstacle height the jump parameters are tuned at

export function obstaclesNow(obstacles) {
  const out = obstacles.map((o) => (o.live ? o.live() : o));
  out.sort((a, b) => a.x0 - b.x0);
  return out;
}

export function installRunner(ep, world, p, ctx) {
  const { config } = ctx;
  const HZ = config.physics.hz;
  const J = config.net.jointCount;
  const P = config.partIndex;
  const runner = makeController(world);
  const tmp = new Float64Array(J + config.net.gripOutputs);
  const obstacles = ctx.obstacles || [];
  const speed = ctx.speed || (() => p.vRun || 2.7);
  // ep.sel: what the selector sees and decides, for the playground's inspector (read only, changes nothing):
  // the next obstacle, the take-off window, the last jump (frame, scale), the step adjustment, stepping down,
  // and the runner's commands before (base) and the network's change to them (corr)
  const sel = ep.sel = { next: null, fx: 0, win: 0, jumpAt: -1e9, jumpScale: 1, placeAdj: 0, dropH: 0, base: new Float64Array(J), corr: new Float64Array(J), net: !!ctx.net };
  ep.updateGrip = ((orig) => function () {
    const m = this.measure;
    m.walkSpeed = speed();
    this.speedCap = config.body.maxJointSpeed;
    runner(p, this, tmp);
    const st = this.runner;
    if (st && st.t === 1 / HZ && obstacles.length) {
      const fx = world.px[st.stance ? P.footRight : P.footLeft];
      if (st.lastX != null) { const L = fx - st.lastX; if (L > 0.2 && L < 2.5) st.avgL = st.avgL ? 0.7 * st.avgL + 0.3 * L : L; }
      st.lastX = fx;
      const now = obstaclesNow(obstacles);
      const next = now.find((o) => o.x1 > fx);
      st.placeAdj = 0;
      // standing on top of something with the next step coming down off it: dropH = how far down (the runner
      // reaches further and leans less by dropPlace, dropLean per metre)
      const fy = world.py[st.stance ? P.footRight : P.footLeft];
      const on = now.find((o) => o.top != null && o.top > 0.05 && fx >= o.x0 - 0.03 && fx <= o.x1 + 0.03 && fy > o.top - 0.08);
      st.dropH = on && st.avgL && on.x1 - fx < st.avgL ? on.top : 0;
      const dh = next && next.top != null ? next.top - HREF : 0;
      const win = (st.avgL ? st.avgL + p.jumpMargin : p.jumpWin) + (p.winH || 0) * dh;
      sel.next = next || null; sel.fx = fx; sel.win = win; sel.dropH = st.dropH;
      if (next && next.x0 - fx > 0.03 && next.x0 - fx < win) {
        st.jumping = true;
        st.jumpScale = Math.max(0.3, Math.min(3, 1 + (p.jumpGainH || 0) * dh / HREF));
        sel.jumpAt = this.frame; sel.jumpScale = st.jumpScale;
      } else if (next && st.avgL && p.kStride) {
        const D = next.x0 - fx - p.takeoff - (p.takeoffH || 0) * dh;
        if (D > 0 && D < 5) {
          const n = Math.max(1, Math.round(D / st.avgL));
          st.placeAdj = Math.max(-0.35, Math.min(0.35, p.kStride * (D / n - st.avgL)));
        }
      }
    }
    // ctx.net = { correct, theta }: the dash corrective network (src/dash-net.js) adds to the commands each frame
    if (st) sel.placeAdj = st.placeAdj || 0;
    for (let j = 0; j < J; j++) sel.base[j] = tmp[j];
    if (ctx.net && st) ctx.net.correct(ctx.net.theta, this, st, obstaclesNow(obstacles), speed(), tmp);
    for (let j = 0; j < J; j++) sel.corr[j] = tmp[j] - sel.base[j];
    for (let j = 0; j < tmp.length; j++) this.action[j] = j < J ? tmp[j] : -1;
    orig();
  })(ep.updateGrip.bind(ep));
}
