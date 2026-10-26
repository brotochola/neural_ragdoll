// Probe: with both feet planted (one forward), how far forward can the pose executor carry the pelvis?
//   node probe-shift.js [lean]
import path from 'node:path';
import fs from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { config, setSkill } from './config.js';
const root = path.dirname(fileURLToPath(import.meta.url));
const Box2D = await import(pathToFileURL(path.join(root, 'vendor/PhaserBox2D.min.js')).href);
const { createSim } = await import('./src/sim.js');
const { ElmanNetwork } = await import('./src/net.js');
const { makeSeededRandom } = await import('./src/utils.js');
const { evalMode } = await import('./src/training.js');
const pose = await import('./src/pose.js');
setSkill(6);
const sim = createSim(Box2D);
const world = sim.build();
const HZ = config.physics.hz, J = config.net.jointCount, P = config.partIndex;
const weights = ElmanNetwork.copy(JSON.parse(fs.readFileSync(path.join(root, 'trained/stand-and-walk.json'), 'utf8')).theta);
const lean = +(process.argv[2] || 0);
const stepLen = +(process.argv[3] || 0.35);
const balance = process.argv[4] === 'bal';
const r = makeSeededRandom(1);
const start = sim.fillTrainingStart({}, r, evalMode(6));
start.tilt = 0; start.walk = null; start.frames = 12 * HZ;
world.rebuild();
const ep = new sim.Episode(world, weights, 0, 0, Object.assign({ phase: 0 }, start));
const f0 = [P.footLeft, P.footRight].map((q) => pose.soleOf(world, q));
const pv0 = pose.pelvisOf(world);
const tmp = new Float64Array(J + 2);
let phase = 0;
ep.updateGrip = ((orig) => function () {
  const t = ep.frame / HZ;
  // 0-2 s: step the right foot forward (quasi-static, slow); 2-10 s: pelvis goal goes from the middle to past the front foot
  const feet = [{ x: f0[0].x, y: f0[0].y }, { x: f0[1].x, y: f0[1].y }];
  const planted = [true, true];
  const mid = (f0[0].x + f0[1].x) / 2;
  let pg = { x: mid, y: pv0.y - 0.04 };
  if (t < 3) {
    // shift over left, lift right, put it stepLen ahead of the left
    if (t < 1) pg.x = mid + (f0[0].x - mid) * (t / 1);
    else { pg.x = f0[0].x; const s = Math.min(1, (t - 1) / 1.5); planted[1] = s >= 1; feet[1] = { x: f0[1].x + (f0[0].x + stepLen - f0[1].x) * s, y: f0[1].y + 0.12 * Math.sin(Math.PI * s) }; }
  } else {
    feet[1] = { x: f0[0].x + stepLen, y: f0[1].y };
    const s = Math.min(1, (t - 3) / 6);
    pg.x = f0[0].x + s * (stepLen + 0.05);
  }
  const goal = { pelvis: pg, torso: -lean, feet, hands: [null, null], gripping: [false, false], planted, torsoGain: 0.4 };
  const actual = pose.actualOf(world);
  const jt = balance ? pose.balancedTargets(goal, ep.measure, { planted, support: t < 3 ? f0[0].x : undefined, kHip: +(process.env.KHIP || 0.3), kAnkle: +(process.env.KANK || 0.8), kAnkleVel: +(process.env.KAV || 0.4), kHipVel: +(process.env.KHV || 0.2) }, actual) : pose.jointTargets(goal, actual);
  pose.commandsFor(world, jt, tmp);
  for (let j = 0; j < J; j++) this.action[j] = tmp[j];
  this.action[J] = this.action[J + 1] = -1;
  if (ep.frame % (HZ / 4) === 0) {
    const pv = pose.pelvisOf(world);
    const so = [P.footLeft, P.footRight].map((q) => pose.soleOf(world, q));
    console.log('t', t.toFixed(1), 'goal-rear', (pg.x - f0[0].x).toFixed(2), 'pelvis-rear', (pv.x - f0[0].x).toFixed(2), 'com-rear', (ep.measure.centerX - f0[0].x).toFixed(2), 'torso', pv.angle.toFixed(2), 'feet', so.map((f) => (f.x - f0[0].x).toFixed(2) + ',' + f.y.toFixed(2)).join(' '), 'head', ep.measure.headHeight.toFixed(2));
  }
  orig();
})(ep.updateGrip.bind(ep));
while (!ep.advanceOneFrame()) {}
