// RECOVER: get up after a fall and go on. Installed once (wraps ep.updateGrip and ep.decide); start() runs it:
//   settle  hands open, joints relaxed, until the body is still (or 2 s)
//   getup   the get-up expert network drives (as it was trained: the episode's network with its get-up clock,
//           the stand network's output after the handover), told the floor is where the body lies
//   stand   the stand network for a moment
//   walk    the SIMBICON walker, speeding up from 0.3 to 1.2 m/s over 3 s
//   done    ep.runner is set from the walker's stance: the caller's runner can take over
// A get-up that has not stood the body up 6 s after its handover settles and tries once more, then 'failed'.
import { makeController as makeWalker } from './simbicon.js';

export function installRecovery(ep, world, ctx) {
  const { config, ElmanNetwork } = ctx;
  const HZ = config.physics.hz, J = config.net.jointCount, OUT = J + config.net.gripOutputs, P = config.partIndex;
  const walker = makeWalker(world);
  const tmp = new Float64Array(J + 2);
  const R = { phase: null, t: 0, tries: 0, count: 0, floor: 0, stand: null, side: 'up' };
  const floorAt = ctx.floorAt || (() => 0);
  const toSettle = () => { R.phase = 'settle'; R.t = 0; };
  R.start = () => { if (R.phase && R.phase !== 'done' && R.phase !== 'failed') return; R.tries = 0; R.count++; toSettle(); };
  R.busy = () => !!R.phase && R.phase !== 'done' && R.phase !== 'failed';
  const restore = () => { if (R.stand) { ep.network = R.stand; R.stand = null; } if (R.marks !== undefined) { ep.options.targets = R.marks; R.marks = undefined; } ep.options.getup = false; ep.measure.getup = false; ep.options.startTime = 0; };
  // the networks read their marks on the floor the body lies on
  const origDecide = ep.decide.bind(ep);
  ep.decide = function () {
    // (the get-up's own marks stay where they were planted when it started, as in training: planted once at
    // reset and frozen; marks that followed the hips told it the opposite of what it learned)
    if (R.phase === 'stand' && this.options.targets) {
      const t = this.options.targets, px = world.px[0];
      for (let i = 0; i < config.targets.length; i++) {
        const spec = config.targets[i], mk = t[i];
        if (!mk) continue;
        if (spec.ride) { mk.x = px + (spec.x || 0); mk.y = R.floor + (spec.y || 0); mk.angle = 0; if (mk.walkBase != null) mk.walkBase = mk.x - (this.walkOffset ? this.walkOffset() : 0); }
        else { mk.x = world.px[spec.id === 'footR' ? P.footRight : P.footLeft]; mk.y = R.floor + config.skill.footGroundMeters; }
      }
    }
    origDecide();
    if (R.phase === 'getup' && R.stand) R.stand.forward(this.input);
  };
  ep.updateGrip = ((orig) => function () {
    if (!R.busy()) { orig(); return; }
    const m = this.measure;
    R.t += 1 / HZ;
    this.speedCap = 0;
    if (R.phase === 'settle') {
      for (let j = 0; j < J; j++) this.action[j] = 0;
      for (let k = J; k < OUT; k++) this.action[k] = -1;
      let still = true;
      for (let q = 0; q < world.parts.length; q++) if (Math.abs(world.vx[q]) > 0.15 || Math.abs(world.vy[q]) > 0.15) still = false;
      if ((R.t > 0.6 && still) || R.t > 2) {
        R.phase = 'getup'; R.t = 0; R.tries++;
        R.floor = floorAt(world.px[0]);
        R.side = Math.atan2(world.s[0], world.c[0]) > 0 ? 'up' : 'down';
        R.stand = this.network;
        this.network = new ElmanNetwork(Float32Array.from(ctx.getup.theta));
        this.groundOffset = R.floor;
        // the marks as the get-up was trained with them (a skill-9 start): head and hands on, flat (angle 0),
        // span 0.25, gain 1.3; the feet off. The episode's own marks (another skill's) come back after.
        R.marks = this.options.targets;
        this.options.targets = config.targets.map((t) => (t.ride ? { on: true, x: world.px[0] + (t.x || 0), y: R.floor + (t.y || 0), angle: 0, span: 0.25, gain: 1.3, frozen: true } : { on: false, x: 0, y: 0, angle: 0, gain: 0 }));
        this.options.getup = true;
        this.measure.getup = true; // the reward the networks read (an input) is the get-up's, as in training
        this.options.startTime = -this.frame / HZ;
      }
    } else if (R.phase === 'getup') {
      const hand = (ctx.getup.handover && ctx.getup.handover[R.side]) || 3;
      if (R.t >= hand && R.stand) for (let j = 0; j < OUT; j++) this.network.action[j] = R.stand.action[j];
      for (let k = J; k < OUT; k++) this.action[k] = -1;
      const up = m.headHeight - R.floor > 1.35 && (m.torsoCos || 0) > 0.8;
      if (R.t > hand + 1 && up) { restore(); R.phase = 'stand'; R.t = 0; }
      else if (R.t > hand + 6) { restore(); if (R.tries < 2) toSettle(); else R.phase = 'failed'; }
    } else if (R.phase === 'stand') {
      for (let k = J; k < OUT; k++) this.action[k] = -1;
      if (R.t > 1.2) { R.phase = 'walk'; R.t = 0; this.simbicon = { stance: 0, t: 0 }; }
    } else if (R.phase === 'walk') {
      m.walkSpeed = Math.min(1.2, 0.3 + R.t * 0.3); m.walkDir = 1;
      walker(ctx.walkP, this, tmp);
      for (let j = 0; j < tmp.length; j++) this.action[j] = j < J ? tmp[j] : -1;
      const s = this.simbicon;
      if (R.t > 3 && s && s.t < 1.5 / HZ) { R.phase = 'done'; this.runner = { stance: s.stance, t: 0, flight: false, jumping: false, jumpNext: false, strides: 0 }; }
      if ((m.headHeight - floorAt(world.px[0])) < 0.9) { if (R.tries < 2) toSettle(); else R.phase = 'failed'; }
    }
    orig();
  })(ep.updateGrip.bind(ep));
  return R;
}
