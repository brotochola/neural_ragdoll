// OBSTACLE: running into a 2 m irregular rock obstacle and going on over it. Running (the parkour
// runner), it slows to a walk (the SIMBICON walker), stops a set distance from the rock and stands
// still (the stop controller, then the pose executor), climbs it (the crag planner: knobs, three points,
// belly mantle, the get-up expert on top), walks off and runs on along the top. Each piece is an
// earlier skill; this script chains them and counts how far the chain gets.
//   node obstacle.js --show --seeds 6 [--crag trained/crag-g0.json] [--record out.json]
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { config, setSkill } from './config.js';

const root = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const arg = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : fallback;
};
const seeds = +arg('--seeds', 6);
const seconds = +arg('--seconds', 150);
const H = +arg('--height', 2.0);

const Box2D = await import(pathToFileURL(path.join(root, 'vendor/PhaserBox2D.min.js')).href);
const { createSim } = await import('./src/sim.js');
const { ElmanNetwork } = await import('./src/net.js');
const { makeSeededRandom } = await import('./src/utils.js');
const { evalMode } = await import('./src/training.js');
const pose = await import('./src/pose.js');
const simb = await import('./src/simbicon.js');
const runMod = await import('./src/runner.js');
const { installCragPlanner } = await import('./src/crag-planner.js');
const { makeCrag } = await import('./src/terrain.js');

setSkill(6);
config.walk.maxStandSpeed = Math.max(config.walk.maxStandSpeed, 3);
const sim = createSim(Box2D);
const world = sim.build();
const HZ = config.physics.hz, J = config.net.jointCount, P = config.partIndex;
const load = (f) => JSON.parse(fs.readFileSync(path.join(root, f), 'utf8'));
const weights = ElmanNetwork.copy(load('trained/stand-and-walk.json').theta);
const runStates = load('trained/run-states.json');
const runP = load('trained/runner-gap-12c.json').params;
const wf = load('trained/walk-fast-controller.json');
const walkP = Object.assign(simb.decode(wf.best), { track: !!wf.track });
const ws = load('trained/walk-controller.json');
const slowP = Object.assign(simb.decode(ws.best), { track: !!ws.track });
const st = load('trained/stop-controller.json').params;
const stopP = Object.assign({}, slowP, st, { track: true });
const getup = load('trained/getup-expert-19.json');
// the crag planner's numbers, decoded as crag.js does
const cragSrc = fs.readFileSync(path.join(root, 'crag.js'), 'utf8');
const CSPEC = eval(cragSrc.slice(cragSrc.indexOf('const SPEC = [') + 13, cragSrc.indexOf('];', cragSrc.indexOf('const SPEC = [')) + 1));
const cragV = load(arg('--crag', 'trained/crag-g0.json')).best;
const cragP = {};
CSPEC.forEach(([k, b], i) => (cragP[k] = b + (cragV[i] || 0)));
for (const k of ['tHand', 'tFoot', 'tRise', 'tPress', 'tOver', 'tPull']) cragP[k] = Math.max(0.2, cragP[k]);
cragP.belly = 1;
cragP.doneX = +(process.env.DONE_X || 3.0); // crawl well away from the edge: getting up it can go 3 m either way
const stopAt = +(process.env.STOP_AT || 0.6); // front foot this far from the rock when it stops

function episode(seed, record) {
  const r = makeSeededRandom(15000 + seed);
  const start = sim.fillTrainingStart({}, r, evalMode(6));
  const speed = 2.2 + r() * 0.6;
  const crag = makeCrag(r, H, 9 + r() * 3, { plateau: 14 });
  const rs = runStates[Math.floor(r() * runStates.length)];
  Object.assign(start, { startPose: rs.pose.slice(), startVel: rs.vel.map((v, i) => (i % 3 === 0 ? v * speed / rs.speed : v)), tilt: 0, drop: 0, settle: 0, frames: Math.round(seconds * HZ) });
  start.walk = null;
  start.terrain = { statics: [], polys: crag.polys, surface: crag.surface };
  world.rebuild();
  const ep = new sim.Episode(world, weights, 0, 0, Object.assign({ phase: 0 }, start));
  ep.runner = { stance: rs.stance, t: -0.1, flight: false, jumping: false, jumpNext: false, strides: 0 };
  const S = ep.terrain.surface;
  const face = S[1][0]; // the rock's foot
  const edgeX = S[S.length - 2][0];
  const walker = simb.makeController(world), runner = runMod.makeController(world);
  const tmp = new Float64Array(J + 2);
  let mode = 'run', modeT = 0, PL = null, holdFeet = null;
  const log = [];
  const setMode = (m2) => { if (m2 !== mode) { log.push((ep.frame / HZ).toFixed(1) + 's ' + m2); mode = m2; modeT = 0; } };
  ep.updateGrip = ((orig) => function () {
    const m = this.measure;
    modeT += 1 / HZ;
    const front = Math.max(world.px[P.footLeft], world.px[P.footRight]);
    const onTop = world.py[0] > H + 0.4;
    // only the runner uses the fast joints (20 rad/s); everything else was tuned at 10 (the get-up expert
    // at 20 rolled right over itself)
    this.speedCap = mode === 'run' || mode === 'run2' ? config.body.maxJointSpeed : 0;
    if (mode === 'run' || mode === 'run2') {
      // slow down ahead of the rock, then walk
      const dist = face - front;
      m.walkSpeed = mode === 'run2' ? Math.min(speed, 1.2 + modeT * 0.6) : dist < +(process.env.DECEL || 8) ? Math.max(+(process.env.VWALK || 1.1), speed * dist / +(process.env.DECEL || 8)) : speed;
      this.speedCap = config.body.maxJointSpeed;
      runner(runP, this, tmp);
      if (mode === 'run' && dist < 5 && (m.centerVelocityX || 0) < +(process.env.VSWITCH || 1.5) && this.runner.t < 1.5 / HZ && !this.runner.flight) { setMode('walk'); this.simbicon = { stance: this.runner.stance, t: 0 }; }
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
  const frames = record ? [] : null;
  let fell = false, done = false;
  while (!ep.advanceOneFrame()) {
    const m = ep.measure;
    if (record && ep.frame % 2 === 0) {
      const q = [];
      for (let k = 0; k < world.parts.length; k++) q.push(+world.px[k].toFixed(3), +world.py[k].toFixed(3), +Math.atan2(world.s[k], world.c[k]).toFixed(3));
      frames.push({ p: q, up: m.torsoCos > 0.7 ? 1 : 0 });
    }
    const floorY = world.px[0] > edgeX - 0.1 && world.py[0] > H - 0.2 ? H : 0;
    if (mode !== 'climb' && m.headHeight - floorY < 0.6) { fell = true; break; }
    if (mode === 'climb' && PL && !PL.phase && m.headHeight < world.py[0]) { fell = true; break; }
    if (mode === 'run2' && world.px[0] > edgeX + 5) { done = true; break; }
  }
  return { done, fell, log, t: ep.frame / HZ, mode, frames, surface: S.map(([x, y]) => [+x.toFixed(3), +y.toFixed(3)]), phase: PL ? PL.phase : null };
}

let ok = 0;
for (let i = 0; i < seeds; i++) {
  const e = episode(100 + i);
  ok += e.done ? 1 : 0;
  console.log('seed', i, e.done ? 'OVER AND RUNNING' : e.fell ? 'FELL in ' + e.mode + (e.phase ? '/' + e.phase : '') : 'stopped in ' + e.mode + (e.phase ? '/' + e.phase : ''), e.t.toFixed(1) + 's |', e.log.join(', '));
}
console.log('over', ok + '/' + seeds);
const rec = arg('--record', null);
if (rec) {
  const shapes = world.parts.map((q) => ({ hx: +q.hx.toFixed(4), hy: +q.hy.toFixed(4), r: q.r ? +q.r.toFixed(4) : 0, ax: q.ax || 0, ay: q.ay || 0, capsule: !!q.capsule, circle: !!q.circle }));
  const eps = [];
  for (let i = 0; i < +arg('--n', 2); i++) { const e = episode(100 + +arg('--first', 0) + i, true); eps.push({ seed: i, surface: e.surface, status: e.done ? 'over the 2 m rock and running' : 'stopped in ' + e.mode, frames: e.frames }); }
  fs.writeFileSync(rec, JSON.stringify({ skill: 6, fps: HZ / 2, shapes, episodes: eps }));
}
