// Comprueba que toda orden sorteada sea fisicamente alcanzable. Sin fisica: solo geometria.
//   node check-reach.js [muestras]
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { config } from './config.js';

const root = path.dirname(fileURLToPath(import.meta.url));
const Box2D = await import(pathToFileURL(path.join(root, 'vendor/PhaserBox2D.min.js')).href);
const { createSim } = await import('./src/sim.js');
const sim = createSim(Box2D);
const { makeSeededRandom } = await import('./src/utils.js');

const samples = Number(process.argv[2]) || 4000;
const P = config.body.torsoHeightMeters;
const shoulder = { x: 0, y: config.body.shoulderMeters - P };
const hip = { x: 0, y: config.body.hipMeters - P };
const armReach = config.body.armReachMeters;
const legReach = config.body.legReachMeters;
const headArcR = config.body.headRadius + config.body.headAxis;
const neckY = config.body.neckOffsetMeters;

let failures = 0;

for (const group of ['head', 'hand', 'foot', 'all']) {
  const random = makeSeededRandom(1234);
  let worstArm = 0;
  let worstLeg = 0;
  let worstHead = 0;
  let worstStep = 0;
  for (let n = 0; n < samples; n++) {
    const start = sim.fillTrainingStart({}, random, { full: true, level: 1, commands: group });
    for (let t = 0; t < config.targets.length; t++) {
      const spec = config.targets[t];
      const target = start.targets[t];
      if (spec.ride) {
        if (target.lx == null) continue;
        if (spec.cmdArc) {
          // La cabeza solo puede estar sobre su arco.
          const d = Math.hypot(target.lx, target.ly - neckY);
          worstHead = Math.max(worstHead, Math.abs(d - headArcR));
        } else {
          const d = Math.hypot(target.lx - shoulder.x, target.ly - shoulder.y);
          worstArm = Math.max(worstArm, d - armReach);
        }
      } else {
        worstStep = Math.max(worstStep, Math.abs(target.dx));
      }
    }
    void hip;
    void legReach;
  }
  const armOk = worstArm <= 1e-9;
  const headOk = worstHead <= 1e-9;
  console.log(
    '  ' + group.padEnd(5) +
      '  brazo fuera de alcance ' + (worstArm * 1000).toFixed(3) + ' mm ' + (armOk ? 'ok' : 'FUERA') +
      '   cabeza fuera del arco ' + (worstHead * 1000).toFixed(3) + ' mm ' + (headOk ? 'ok' : 'FUERA') +
      '   paso maximo ' + worstStep.toFixed(3) + ' m',
  );
  if (!armOk || !headOk) failures++;
}

// El pie tiene su propio problema: la pierna tambien arranca estirada, asi que un paso de
// 20 cm con el pie pegado al piso es imposible. Se informa, no se prohibe: el pie tiene que
// despegar para obedecer.
const random = makeSeededRandom(99);
let worst = 0;
for (let n = 0; n < samples; n++) {
  const start = sim.fillTrainingStart({}, random, { full: true, level: 1, commands: 'foot' });
  for (let t = 0; t < config.targets.length; t++) {
    const spec = config.targets[t];
    if (spec.ride) continue;
    const foot = { x: (spec.x || 0) + start.targets[t].dx, y: spec.y - P };
    worst = Math.max(worst, Math.hypot(foot.x - hip.x, foot.y - hip.y) - legReach);
  }
}
console.log('');
console.log('  pie: el paso maximo deja al pie ' + (worst * 100).toFixed(1) + ' cm mas lejos de la cadera de lo que la pierna estirada alcanza.');
console.log('  eso es esperado: para obedecer, el pie tiene que DESPEGAR del piso. No es un error del workspace.');

console.log('');
console.log(failures ? failures + ' grupos con ordenes inalcanzables' : 'todas las ordenes del cuerpo son alcanzables');
process.exitCode = failures ? 1 : 0;