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
const { installObstacleRunner } = await import('./src/obstacle-runner.js');
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
  const OR = installObstacleRunner(ep, world, { debug: !!process.env.DEBUG_OB, config, pose, Box2D, ElmanNetwork, crag, H, speed: () => speed, stopAt, vswitch: +(process.env.VSWITCH || 1.3), decel: +(process.env.DECEL || 8), params: { runP, walkP, slowP, stopP, stopK: st, cragP, getup } });
  const log = OR.log;
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
    if (OR.mode !== 'climb' && m.headHeight - floorY < 0.6) { fell = true; break; }
    if (OR.mode === 'climb' && OR.planner && !OR.planner.phase && m.headHeight < world.py[0]) { fell = true; break; }
    if (OR.mode === 'run2' && world.px[0] > edgeX + 5) { done = true; break; }
  }
  return { done, fell, log, t: ep.frame / HZ, mode: OR.mode, frames, surface: S.map(([x, y]) => [+x.toFixed(3), +y.toFixed(3)]), phase: OR.planner ? OR.planner.phase : null };
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
