// OBSTACLE RUNNER (obstacle.js, the playground): running into a tall rock and over it, by chaining
// earlier skills: the parkour runner slows down, the SIMBICON walker walks up and the stop controller
// stops a set distance from the rock, the pose executor holds still, the crag planner climbs (knobs,
// belly mantle, the get-up expert on top), then the walker walks off and the runner runs on.
//   modes: run -> walk -> stop -> hold -> climb -> walk2 -> run2
import * as simb from './simbicon.js';
import * as runMod from './runner.js';
import { installCragPlanner } from './crag-planner.js';

export function installObstacleRunner(ep, world, ctx) {
  const { config, pose, Box2D, ElmanNetwork, crag, H } = ctx;
  const { runP, walkP, slowP, stopP, cragP, getup } = ctx.params;
  const st = ctx.params.stopK;
  const HZ = config.physics.hz, J = config.net.jointCount, P = config.partIndex;
  const S = ep.terrain.surface;
  const face = S[1][0];
  const stopAt = ctx.stopAt != null ? ctx.stopAt : 0.6, DECEL = ctx.decel || 8, VWALK = ctx.vwalk || 1.1, VSWITCH = ctx.vswitch || 1.3;
  const speedOf = () => ctx.speed();
  const walker = simb.makeController(world), runner = runMod.makeController(world);
  const tmp = new Float64Array(J + 2);
  let mode = 'run', modeT = 0, PL = null, holdFeet = null;
  const log = [];
  const setMode = (m2) => { if (m2 !== mode) { log.push((ep.frame / HZ).toFixed(1) + 's ' + m2); mode = m2; modeT = 0; } };
  ep.updateGrip = ((orig) => function () {
    const m = this.measure;
    modeT += 1 / HZ;
    const front = Math.max(world.px[P.footLeft], world.px[P.footRight]);
    // only the runner uses the fast joints (20 rad/s); everything else was tuned at 10 (the get-up expert
    // at 20 rolled right over itself)
    this.speedCap = mode === 'run' || mode === 'run2' ? config.body.maxJointSpeed : 0;
    if (mode === 'run' || mode === 'run2') {
      // slow down ahead of the rock, then walk
      const dist = face - front;
      m.walkSpeed = mode === 'run2' ? Math.min(speedOf(), 1.2 + modeT * 0.6) : dist < DECEL ? Math.max(VWALK, speedOf() * dist / DECEL) : speedOf();
      this.speedCap = config.body.maxJointSpeed;
      runner(runP, this, tmp);
      if (mode === 'run' && dist < 5 && (m.centerVelocityX || 0) < VSWITCH && this.runner.t < 1.5 / HZ && !this.runner.flight) { setMode('walk'); this.simbicon = { stance: this.runner.stance, t: 0 }; }
    } else if (mode === 'walk') {
      const dist = face - front;
      m.walkSpeed = Math.max(0.35, Math.min(1.0, dist / 2.5)); m.walkDir = 1;
      walker(dist > 1.5 ? walkP : slowP, this, tmp);
      if (dist < stopAt + 0.25) setMode('stop');
    } else if (mode === 'stop') {
      m.walkSpeed = 0; m.walkDir = 1;
      walker(stopP, this, tmp);
      if (modeT > 2) setMode('hold');
    } else if (mode === 'hold') {
      if (!holdFeet) holdFeet = [P.footLeft, P.footRight].map((q) => pose.soleOf(world, q));
      const mid = (holdFeet[0].x + holdFeet[1].x) / 2;
      const goal = { pelvis: { x: mid, y: 0.92 }, torso: 0, feet: holdFeet.map((q) => ({ x: q.x, y: q.y, angle: 0 })), planted: [true, true] };
      pose.commandsFor(world, pose.balancedTargets(goal, m, { planted: [true, true], kHip: st.holdKHip, kAnkle: st.holdKAnkle, kAnkleVel: 0.15, kHipVel: 0.1 }), tmp);
      for (let k = J; k < tmp.length; k++) tmp[k] = -1;
      if (modeT > 1.5) {
        setMode('climb');
        // the crag planner takes over (it wraps updateGrip from outside; this wrapper then only passes on)
        PL = installCragPlanner(ep, world, cragP, { config, pose, Box2D, crag, nets: { ElmanNetwork, getupW: getup.theta, handover: getup.handover } });
      }
    } else if (mode === 'climb') {
      this.speedCap = 0;
      if (PL.stood && PL.phase === 'stand' && modeT > 1) {
        // standing on top for a moment: walk off, then run
        if (!this.standT) this.standT = 0;
        this.standT += 1 / HZ;
        if (this.standT > 1.5) { PL.handOff(); setMode('walk2'); this.simbicon = { stance: 0, t: 0 }; }
      }
      orig();
      return;
    } else if (mode === 'walk2') {
      m.walkSpeed = Math.min(1.2, 0.3 + modeT * 0.5); m.walkDir = 1;
      walker(walkP, this, tmp);
      const s = this.simbicon;
      if (modeT > 3 && s && s.t < 1.5 / HZ) { setMode('run2'); this.runner = { stance: s.stance, t: 0, flight: false, jumping: false, jumpNext: false, strides: 0 }; }
    }
    for (let j = 0; j < tmp.length; j++) this.action[j] = j < J ? tmp[j] : -1;
    orig();
  })(ep.updateGrip.bind(ep));
  return { get mode() { return mode; }, get planner() { return PL; }, log };
}
