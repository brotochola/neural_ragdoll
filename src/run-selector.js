// The running controller (src/runner.js) with the jump selector of parkour.js: at each touchdown, jump on the
// stride whose ordinary landing would fall within jumpMargin of the next obstacle's front edge, and before
// that stretch or shorten the steps toward a take-off spot. Installs itself on an episode (wraps
// ep.updateGrip). Used by the playground; parkour.js keeps its own copy for training.
import { makeController } from './runner.js';

export function installRunner(ep, world, p, ctx) {
  const { config } = ctx;
  const HZ = config.physics.hz;
  const J = config.net.jointCount;
  const P = config.partIndex;
  const runner = makeController(world);
  const tmp = new Float64Array(J + config.net.gripOutputs);
  const obstacles = ctx.obstacles || [];
  ep.updateGrip = ((orig) => function () {
    const m = this.measure;
    m.walkSpeed = ctx.speed();
    this.speedCap = config.body.maxJointSpeed;
    runner(p, this, tmp);
    const st = this.runner;
    if (st && st.t === 1 / HZ && obstacles.length) {
      const fx = world.px[st.stance ? P.footRight : P.footLeft];
      if (st.lastX != null) { const L = fx - st.lastX; if (L > 0.2 && L < 2.5) st.avgL = st.avgL ? 0.7 * st.avgL + 0.3 * L : L; }
      st.lastX = fx;
      const next = obstacles.find((o) => o.x1 > fx);
      st.placeAdj = 0;
      const win = st.avgL ? st.avgL + p.jumpMargin : p.jumpWin;
      if (next && next.x0 - fx > 0.03 && next.x0 - fx < win) st.jumping = true;
      else if (next && st.avgL && p.kStride) {
        const D = next.x0 - fx - p.takeoff;
        if (D > 0 && D < 5) {
          const n = Math.max(1, Math.round(D / st.avgL));
          st.placeAdj = Math.max(-0.35, Math.min(0.35, p.kStride * (D / n - st.avgL)));
        }
      }
    }
    for (let j = 0; j < tmp.length; j++) this.action[j] = j < J ? tmp[j] : -1;
    orig();
  })(ep.updateGrip.bind(ep));
}
