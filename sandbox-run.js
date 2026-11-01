// SANDBOX-RUN: runs a course built from pieces (src/terrain.js buildSandbox, as the playground's "Armá tu
// pista" builds it) headless with the playground's numbers, a few attempts, and tells how far each got.
//   node sandbox-run.js --pieces '[{"type":"box","x":5},{"type":"rock","x":12}]' [--tries 3] [--seconds 200]
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { config, setSkill } from './config.js';
const root = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const arg = (n, f) => { const i = args.indexOf(n); return i >= 0 ? args[i + 1] : f; };
const Box2D = await import(pathToFileURL(path.join(root, 'vendor/PhaserBox2D.min.js')).href);
const { createSim } = await import('./src/sim.js');
const { ElmanNetwork } = await import('./src/net.js');
const { makeSeededRandom } = await import('./src/utils.js');
const { evalMode } = await import('./src/training.js');
const pose = await import('./src/pose.js');
const { buildSandbox } = await import('./src/terrain.js');
const { installCourseDirector } = await import('./src/director.js');
const { makeTrailNet } = await import('./src/trail-net.js');
setSkill(6);
config.walk.maxStandSpeed = 20;
const sim = createSim(Box2D);
const world = sim.build();
const HZ = config.physics.hz;
const params = JSON.parse(fs.readFileSync(path.join(root, 'play/params.json'), 'utf8'));
const weights = ElmanNetwork.copy(JSON.parse(fs.readFileSync(path.join(root, 'trained/stand-and-walk.json'), 'utf8')).theta);
const pieces = JSON.parse(arg('--pieces', '[]'));
const L = buildSandbox(pieces);
const seconds = +arg('--seconds', 200);
for (let t = 0; t < +arg('--tries', 3); t++) {
  const r = makeSeededRandom(100 + t);
  const start = sim.fillTrainingStart({}, r, evalMode(6));
  const rs = params.runStates[Math.floor(r() * params.runStates.length)];
  const v0 = params.trail.vRun || 2.6;
  const pz = rs.pose.slice();
  for (let k = 0; k < world.parts.length; k++) pz[k * 3 + 1] += L.lift;
  Object.assign(start, { startPose: pz, startVel: rs.vel.map((v, i) => (i % 3 === 0 ? v * v0 / rs.speed : v)), tilt: 0, drop: 0, settle: 0, frames: Math.round(seconds * HZ), shiftX: 0 });
  start.walk = null;
  start.terrain = { statics: L.statics, balls: L.balls, polys: L.polys, surface: L.walls[0] ? L.walls[0].crag.surface : null, ropes: L.ropes };
  world.rebuild();
  const ep = new sim.Episode(world, weights, 0, 0, Object.assign({ phase: 0 }, start));
  ep.runner = { stance: rs.stance, t: -0.1, flight: false, jumping: false, jumpNext: false, strides: 0 };
  const correct = makeTrailNet(world, config, Box2D);
  const D = installCourseDirector(ep, world, L, { config, pose, Box2D, ElmanNetwork, params, trailNet: { correct, theta: params.trailNet } });
  const mid = ep.terrain.statics[0].x0 - L.statics[0].dx;
  const finish = L.finishX + mid;
  let why = 'time', zones = [];
  while (!ep.advanceOneFrame()) {
    const z = D.zone();
    if (zones[zones.length - 1] !== z) zones.push(z);
    if (world.px[0] > finish) { why = 'FINISH'; break; }
  }
  console.log('try', t, why, (ep.frame / HZ).toFixed(0) + 's', 'x', (world.px[0] - mid).toFixed(1) + '/' + L.finishX.toFixed(1), 'got up', D.rec.count, 'pits', JSON.stringify(D.pitLog), '|', zones.join('>'));
}
