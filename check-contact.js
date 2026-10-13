// Contact penalties are real, and hands and feet do not pay them.
// A settled upright pose stays near zero. A body lying on the floor pays.
//
//   node check-contact.js
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { config } from './config.js';
import { ElmanNetwork } from './src/net.js';
import { scoreDecision } from './src/reward.js';
import { makeSeededRandom } from './src/utils.js';

const root = path.dirname(fileURLToPath(import.meta.url));
const Box2D = await import(pathToFileURL(path.join(root, 'vendor/PhaserBox2D.min.js')).href);
const { createSim } = await import('./src/sim.js');
const sim = createSim(Box2D);
const weights = new Float32Array(ElmanNetwork.weightCount);

const exempt = ['handLeft', 'handRight', 'footLeft', 'footRight'];
let failures = 0;
function check(name, ok, detail) {
  if (!ok) failures++;
  console.log((ok ? 'ok   ' : 'FAIL') + '  ' + name + (detail ? '   ' + detail : ''));
}

for (const group of config.contact.groups) {
  for (const name of group.parts) {
    const banned = exempt.some((part) => name === part);
    check(group.id + ' does not include ' + name, !banned);
  }
}
check('four contact groups', config.contact.groups.length === 4, String(config.contact.groups.length));

function settle(name, tilt, settle) {
  const start = sim.fillTrainingStart({}, makeSeededRandom(21), { full: true, level: 1 });
  start.tilt = tilt;
  start.drop = tilt ? 0.05 : config.start.uprightDropMeters;
  start.settle = settle;
  const world = sim.build();
  const episode = new sim.Episode(world, weights, start.tilt, start.drop, start);
  const contact = episode.measure.contact;
  const amounts = [];
  for (let i = 0; i < contact.length; i++) amounts.push(contact[i]);
  const scored = scoreDecision(episode.measure);
  const penalty = scored.terms.slice(config.targets.length * 2).reduce((sum, term) => sum + term.v, 0);
  episode.cleanup();
  sim.destroyWorld(world);
  return { name, amounts, penalty };
}

const standing = settle('standing', 0, 20);
const lying = settle('lying', Math.PI / 2, 90);
const format = (row) => row.amounts.map((value) => value.toFixed(2)).join(' ');

console.log('standing  ' + format(standing) + '  penalty ' + standing.penalty.toFixed(3));
console.log('lying     ' + format(lying) + '  penalty ' + lying.penalty.toFixed(3));

const standSum = standing.amounts.reduce((sum, value) => sum + value, 0);
const lieMax = Math.max(...lying.amounts);
check('standing contact stays small', standSum < 0.5, standSum.toFixed(3));
check('lying pose presses a penalised part', lieMax > 0.5, lieMax.toFixed(3));
check('lying pose pays', lying.penalty < -0.5, lying.penalty.toFixed(3));
check('standing pays less than lying', standing.penalty > lying.penalty);

console.log(failures ? failures + ' checks failed' : 'ok');
process.exitCode = failures ? 1 : 0;
