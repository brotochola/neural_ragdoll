// Records whole-body states (pose, velocities, time) along the keyframe get-ups that succeed, for
// reference-state initialisation (RSI) in the get-up imitation training.
//   node make-getup-states.js --n 12 --out trained/getup-states.json
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
const n = +arg('--n', 12);
const out = arg('--out', config.getup.statesFile);
const every = +arg('--every', 12); // frames between saved states (0.2 s)

const Box2D = await import(pathToFileURL(path.join(root, 'vendor/PhaserBox2D.min.js')).href);
const { createSim } = await import('./src/sim.js');
const { ElmanNetwork } = await import('./src/net.js');
const { makeSeededRandom } = await import('./src/utils.js');

setSkill(9);
const sim = createSim(Box2D);
const world = sim.build();
const weights = ElmanNetwork.copy(JSON.parse(fs.readFileSync(path.join(root, 'trained/stand-and-walk.json'), 'utf8')).theta);
const J = config.net.jointCount;
const torso = config.partIndex.torso;
const result = { up: [], down: [] };
for (const side of ['up', 'down']) {
  const frames = JSON.parse(fs.readFileSync(path.join(root, config.getup.referenceFiles[side]), 'utf8')).frames;
  const total = frames.reduce((a, f) => a + f.dur, 0);
  let kept = 0;
  for (let s = 0; s < n; s++) {
    const r = makeSeededRandom(7000 + s);
    const sign = side === 'up' ? 1 : -1;
    const tilt = sign * (config.start.downTiltMinRadians + r() * config.start.downTiltSpanRadians);
    world.rebuild();
    const options = {
      frames: Math.round((total + 3) * config.physics.hz), period: 8, phase: 0, failOnHead: true, getup: true,
      settle: config.start.downSettleFrames, commands: 'none',
      targets: config.targets.map(() => ({ on: true, x: 0, y: 0, angle: 0, dx: 0, dy: 0, turn: 0 })),
    };
    const ep = new sim.Episode(world, weights, tilt, config.start.downDropMeters, options);
    const orig = ep.applyMotors.bind(ep);
    ep.applyMotors = function () {
      const t = this.frame / config.physics.hz;
      let st = 0;
      let prev = null;
      for (const f of frames) {
        if (t < st + f.dur) {
          const b = prev ? Math.min(1, (t - st) / (0.4 * f.dur)) : 1;
          for (let j = 0; j < J; j++) this.action[j] = prev ? prev[j] + (f.pose[j] - prev[j]) * b : f.pose[j];
          break;
        }
        st += f.dur;
        prev = f.pose;
      }
      orig();
    };
    const states = [];
    let f = 0;
    while (!ep.advanceOneFrame()) {
      f++;
      if (f % every !== 0 || f / config.physics.hz > total) continue;
      const px = world.px[torso];
      const pose = [];
      const vel = [];
      for (let i = 0; i < world.parts.length; i++) {
        pose.push(+(world.px[i] - px).toFixed(4), +world.py[i].toFixed(4), +Math.atan2(world.s[i], world.c[i]).toFixed(4));
        vel.push(+world.vx[i].toFixed(3), +world.vy[i].toFixed(3), +world.spin[i].toFixed(3));
      }
      states.push({ t: +(f / config.physics.hz).toFixed(3), pose, vel });
    }
    if (!ep.stats.stoodUp) continue; // only get-ups that worked
    result[side].push(...states);
    kept++;
  }
  console.log(side, 'kept', kept, 'of', n, 'get-ups,', result[side].length, 'states');
}
sim.destroyWorld(world);
fs.writeFileSync(out, JSON.stringify(result));
console.log('wrote', out);
