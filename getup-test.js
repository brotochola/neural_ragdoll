// GETUP-TEST: the level's recovery (src/recover.js) on real falls (trail-es.js --dump-falls), lying as they
// fell with the blocks around them, in the level's settings. Counts how many get up and walk.
//   node getup-test.js --falls trained/trail-falls.json [--n 30] [--getup trained/getup-expert-trail.json] [--lift 0]
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
const { installRecovery } = await import('./src/recover.js');
setSkill(+arg('--skill', 6));
config.walk.maxStandSpeed = 20;
const sim = createSim(Box2D);
const world = sim.build();
const HZ = config.physics.hz, P = config.partIndex;
const params = JSON.parse(fs.readFileSync(path.join(root, 'play/params.json'), 'utf8'));
const getup = arg('--getup', null) ? JSON.parse(fs.readFileSync(arg('--getup'), 'utf8')) : (params.getupTrail || params.getup);
const weights = ElmanNetwork.copy(JSON.parse(fs.readFileSync(path.join(root, 'trained/stand-and-walk.json'), 'utf8')).theta);
const falls = arg('--falls', 'trained/trail-falls.json').split(',').flatMap((f) => JSON.parse(fs.readFileSync(f, 'utf8')));
const n = Math.min(falls.length, +arg('--n', 30)), lift = +arg('--lift', 0), first = +arg('--first', 0);
let ok = 0;
for (let i = first; i < first + n; i++) {
  const f = falls[i];
  const r = makeSeededRandom(4000 + i);
  const start = sim.fillTrainingStart({}, r, evalMode(6));
  const mid = (f.pose[P.footLeft * 3] + f.pose[P.footRight * 3]) / 2;
  const pz = f.pose.slice();
  for (let q = 0; q < world.parts.length; q++) pz[q * 3 + 1] += lift;
  const statics = f.blocks.map((b) => ({ dx: b.x0 - mid, w: b.x1 - b.x0, h: b.top + lift }));
  // the floor it lies on, lift high, from 6 m behind to 6 m ahead
  if (lift > 0) statics.unshift({ dx: -6, w: 40, h: lift });
  if (args.includes('--official')) { setSkill(9); Object.assign(start, sim.fillTrainingStart({}, makeSeededRandom(config.eval.seed + 9 * 10007 + i * 17), evalMode(9))); setSkill(+arg('--skill', 6)); start.frames = Math.round(20 * HZ); start.walk = null; }
  else Object.assign(start, { startPose: pz, startVel: f.vel, tilt: 0, drop: 0, settle: 0, frames: Math.round(20 * HZ), shiftX: 0 });
  start.walk = null;
  if (!args.includes('--official')) start.terrain = { statics };
  world.rebuild();
  const ep = new sim.Episode(world, weights, 0, 0, Object.assign({ phase: 0 }, start));
  if (process.env.FLOOR_ALL) world.floorShapes = world.floorShapes.concat(world.thrownShapes);
  const R = installRecovery(ep, world, { config, ElmanNetwork, getup, walkP: params.walkFast, floorAt: () => lift });
  R.start();
  if (process.env.SHOW_T) console.log('statics', JSON.stringify(ep.terrain.statics.map((b) => [b.x0, b.x1, b.top].map((v) => +v.toFixed(2)))), 'pelvis', world.px[0].toFixed(2), world.py[0].toFixed(2), 'head', world.py[2].toFixed(2));
  let walked = false;
  if (process.env.DUMP_IN) { const od = ep.decide; let k = 0; ep.decide = function () { od.call(this); if (R.phase === 'getup' && k++ === 3) console.log('IN', JSON.stringify(Array.from(this.input).map((v) => +v.toFixed(3)))); }; }
  const phases = []; const sides = [];
  while (!ep.advanceOneFrame()) {
    if (phases[phases.length - 1] !== R.phase) { phases.push(R.phase); if (R.phase === 'getup') sides.push(R.side + ' tc' + (ep.measure.torsoCos || 0).toFixed(2)); }
    if (process.env.SHOW_T && ep.frame % 60 === 0) console.log('  t', (ep.frame / HZ).toFixed(1), R.phase, 'pelvis', world.px[0].toFixed(2), world.py[0].toFixed(2), 'head', world.py[2].toFixed(2), 'go', ep.groundOffset);
    if (R.phase === 'done') { walked = true; break; }
    if (R.phase === 'failed') break;
  }
  if (walked) ok++;
  if (args.includes('--log')) console.log('fall', i, walked ? 'OK' : 'no', phases.join('>'), sides.join(' | '), f.blocks.length + ' blocks', (ep.frame / HZ).toFixed(1) + 's');
}
console.log('got up and walked', ok + '/' + n, 'lift', lift);
