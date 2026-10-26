// Records passive falls (motors and springs off) from a standing start, for the get-up reverse
// curriculum. Each fall is a list of whole-body poses, from standing to lying still. A get-up
// episode can start from any pose along a fall: early ones are a crouch, late ones are on the floor.
//   node make-falls.js --n 24 --out getup-falls.json
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { config } from './config.js';

const root = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const arg = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : fallback;
};
const n = +arg('--n', 24);
const out = arg('--out', 'getup-falls.json');

const Box2D = await import(pathToFileURL(path.join(root, 'vendor/PhaserBox2D.min.js')).href);
const { createSim } = await import('./src/sim.js');
const { ElmanNetwork } = await import('./src/net.js');
const { makeSeededRandom } = await import('./src/utils.js');

const sim = createSim(Box2D);
const world = sim.build();
const weights = new Float32Array(ElmanNetwork.weightCount);
const torso = config.partIndex.torso;
const falls = [];
for (let k = 0; k < n; k++) {
  const rand = makeSeededRandom(9000 + k);
  const sign = k % 2 === 0 ? 1 : -1;
  const tilt = sign * (0.12 + rand() * 0.3);
  world.rebuild();
  const ep = new sim.Episode(world, weights, tilt, 0.05, { frames: 600, period: 8, phase: 0, failOnHead: false });
  ep.passive = true;
  const poses = [];
  let still = 0;
  for (let f = 0; f < 600; f++) {
    ep.advanceOneFrame();
    if (f % 6 !== 0) continue;
    const px = world.px[torso];
    const pose = [];
    for (let i = 0; i < world.parts.length; i++)
      pose.push(+(world.px[i] - px).toFixed(4), +world.py[i].toFixed(4), +Math.atan2(world.s[i], world.c[i]).toFixed(4));
    poses.push(pose);
    // Stop once the body has been still for half a second.
    let speed = 0;
    for (let i = 0; i < world.parts.length; i++) speed = Math.max(speed, Math.hypot(world.vx[i], world.vy[i]));
    still = speed < 0.05 ? still + 1 : 0;
    if (still >= 5) break;
  }
  falls.push(poses);
  const last = poses[poses.length - 1];
  console.log('fall', k, 'poses', poses.length, 'end pelvis', last[torso * 3 + 1].toFixed(2), 'end head', last[config.partIndex.head * 3 + 1].toFixed(2));
}
sim.destroyWorld(world);
fs.writeFileSync(out, JSON.stringify({ parts: world.parts.length, falls }));
console.log('wrote', out);
