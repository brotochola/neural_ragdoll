import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { config, setSkill } from './config.js';
import { makeSeededRandom } from './src/utils.js';

const skill = 2;
setSkill(skill);
const root = path.dirname(fileURLToPath(import.meta.url));
const Box2D = await import(pathToFileURL(path.join(root, 'vendor/PhaserBox2D.min.js')).href);
const { createSim } = await import('./src/sim.js');
const sim = createSim(Box2D);
const theta = Float32Array.from(JSON.parse(fs.readFileSync(path.join(root, 'backups/g200_s7p368_phase_1791767817432.json'), 'utf8')).theta);
const worlds = [];
for (let w = 0; w < 12; w++) worlds.push(new sim.Episode(sim.build(), theta, 0, 0, {}));
let stood = 0;
let feet = 0;
const why = {};
for (let i = 0; i < 20; i++) {
  const episode = worlds[i % 12];
  const start = sim.fillTrainingStart({}, makeSeededRandom(config.eval.seed + skill * 10007 + i * 17), { full: false, level: 2, shiftMeters: 0.16 });
  if (start.place && start.place.extra < 0) start.place.extra = -start.place.extra;
  episode.reset(theta, start.tilt || 0, start.drop || 0, start);
  while (!episode.step());
  const targets = episode.options.targets;
  const step2 = 0.12 + makeSeededRandom(config.eval.seed + 5000 + i * 17)() * 0.04;
  targets[3].x += step2;
  targets[4].x += step2;
  const mid = (targets[3].x + targets[4].x) / 2;
  targets[0].x = mid;
  targets[1].x = mid;
  targets[2].x = mid;
  episode.options.replayRight = 1;
  episode.phaseFrame = episode.frame;
  for (let n = 0; n < 360; n++) episode.step();
  const s = episode.stats;
  const up = !!s.stoodUp;
  const both = up && s.footL <= 0.08 && s.footR <= 0.08;
  if (up) stood++;
  if (s.footL <= 0.08 && s.footR <= 0.08) feet++;
  const w = episode.replayWhy || 'none';
  why[w] = (why[w] || 0) + 1;
  console.log(
    i, both ? 'pass' : 'miss', 'up', up ? 1 : 0,
    'eL', s.footL.toFixed(3), 'eR', s.footR.toFixed(3),
    'why', w,
    'dR', episode.place3FootR == null ? 'na' : episode.place3FootR.toFixed(3),
    'swingR', episode.swingR == null ? 'na' : episode.swingR.toFixed(3),
    'dx', episode.place3Dx == null ? 'na' : episode.place3Dx.toFixed(3),
  );
  episode.cleanup();
}
console.log('stood', stood + '/20', 'feet', feet + '/20', JSON.stringify(why));
