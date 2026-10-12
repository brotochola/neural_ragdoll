// ¿El premio de las diez filas ya castiga un contacto, o haria falta un termino extra?
// Recorre la altura de la cabeza con todo lo demas fijo y mira cuanto vale cada centimetro.
//   node check-contact-cost.js
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { config } from './config.js';
import { ElmanNetwork } from './src/net.js';
import { scoreByTarget } from './src/reward.js';
import { makeSeededRandom } from './src/utils.js';

const root = path.dirname(fileURLToPath(import.meta.url));
const Box2D = await import(pathToFileURL(path.join(root, 'vendor/PhaserBox2D.min.js')).href);
const { createSim } = await import('./src/sim.js');
const sim = createSim(Box2D);
const weights = new Float32Array(ElmanNetwork.weightCount);

const start = sim.fillTrainingStart({}, makeSeededRandom(21), { full: true, level: 1 });
const episode = new sim.Episode(sim.build(), weights, 0, 0, start);
episode.reset(weights, 0, 0, start);

// Congelamos el mundo en la pose plantada y movemos SOLO la cabeza a distintas alturas.
// Asi lo unico que cambia es donde esta la cabeza: la fila de cabeza, y nada mas.
const measure = episode.measure;
const headIndex = config.partIndex.head;
const markY = start.targets[0].y;
const total = (row) => row.position + row.rotation;

console.log('Cuanto vale cada centimetro que baja la cabeza, con el resto de la pose igual.');
console.log('El premio es SOLO las diez filas. Una cabeza en el piso esta a ' +
  (markY - 0.09).toFixed(2) + ' m de su marca.');
console.log('');
console.log('altura cabeza   fila cabeza   cambio   premio total');
let prev = null;
for (const y of [1.70, 1.55, 1.40, 1.20, 1.00, 0.80, 0.60, 0.40, 0.20, 0.09]) {
  measure.partY[headIndex] = y;
  measure.partX[headIndex] = start.targets[0].x;
  const rows = scoreByTarget(measure);
  const sum = rows.reduce((a, r) => a + r.position + r.rotation, 0);
  const head = total(rows[0]);
  const delta = prev == null ? '' : (head - prev).toFixed(3).padStart(7);
  console.log(
    y.toFixed(2).padStart(12) + head.toFixed(3).padStart(13) + delta.padStart(9) +
    sum.toFixed(2).padStart(15),
  );
  prev = head;
}
console.log('');
console.log('Lectura: la fila de cabeza vale ' + config.targets[0].weight + ' en la marca y -' +
  config.targets[0].weight + ' en el infinito. Bajar la cabeza al piso NO cuesta un');
console.log('termino aparte: ya se paga entero en la fila que existe.');
console.log('');
console.log('Lo que NO se mide: muslo, pantorrilla, brazo, mano que roce algo. Ninguno de esos');
console.log('es un target. config.targets tiene head, handR, handL, footR, footL. Un muslo en el');
console.log('suelo no cambia ninguna de las diez filas por si mismo: solo cuenta si arrastra');
console.log('la cadera, y con eso mueve la cabeza y los pies.');
console.log('Si queres que "el muslo no toca el piso" cuente, ese dato NO esta en el premio hoy.');
episode.cleanup();
