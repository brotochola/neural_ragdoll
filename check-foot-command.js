// De donde sale el error de un pie. El probe decia "corrige 0%" para los pies, y eso NO era
// fisica: el piloto de referencia solo movia el cuello (probe-targets.js, oracleNeck). Un 0% de
// correccion tiene que significar "imposible", no "nadie midio".
//
// Y aparecio una cosa que no cuadraba: la orden del pie es de ~0.09 m pero el error era de
// ~0.30 m. Si el pie simplemente ignorara la orden, el error SERIA la orden. Tres veces mas
// grande significa que hay un termino que no es la orden. Este script los separa:
//   y0      : el pie nace a ~0.09 del piso y la marca se planta a footGroundMeters
//   orden   : el dx que se pidio
//   deriva  : cuanto se movio el pie en el mundo durante el episodio
//   err_x   : lo que la fila mide de verdad en horizontal
//   err_y   : idem en vertical
//
//   node check-foot-command.js
import path from 'node:path';
import fs from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { config } from './config.js';
import { ElmanNetwork } from './src/net.js';
import { markError } from './src/reward.js';
import { makeSeededRandom } from './src/utils.js';

const root = path.dirname(fileURLToPath(import.meta.url));
const Box2D = await import(pathToFileURL(path.join(root, 'vendor/PhaserBox2D.min.js')).href);
const { createSim } = await import('./src/sim.js');
const sim = createSim(Box2D);

const argv = process.argv.slice(2);
let LEVEL = 4;
let N = 8;
let netFile = null;
for (let i = 0; i < argv.length; i++) {
  if (argv[i] === '--skill') LEVEL = +argv[++i];
  else if (argv[i] === '--n') N = +argv[++i];
  else if (argv[i] === '--net') netFile = argv[++i];
}

// Con --net se mide una red que por lo menos trata de pararse. Con red en cero se mide un cuerpo
// que se cae al piso, y su deriva no dice nada sobre si la fila premia obediencia.
function loadWeights() {
  if (!netFile) return new Float32Array(ElmanNetwork.weightCount);
  const raw = JSON.parse(fs.readFileSync(path.resolve(root, netFile), 'utf8'));
  const theta = raw.theta || (raw.search && raw.search.theta) || raw.weights;
  if (!theta || theta.length !== ElmanNetwork.weightCount) {
    console.log('FALLA: ' + netFile + ' tiene ' + (theta ? theta.length : 'nada') +
      ' pesos y la red necesita ' + ElmanNetwork.weightCount + '. No se puede medir obedience con el.');
    process.exitCode = 1;
    return null;
  }
  return Float32Array.from(theta);
}

const weights = loadWeights();
if (!weights) process.exit(1);
const footSlot = config.targets.findIndex((s) => s.id === 'footR');
const footPart = config.partIndex.footRight;

console.log('nivel ' + LEVEL + ' (comanda pies), ' + N + ' semillas, red: ' + (netFile || 'en CEROS (se cae)'));
console.log('Descomposicion del error del pie DERECHO.');
console.log('');
console.log('semilla   orden     y0      deriva   |   err_x    err_y    err');
console.log('');

const sum = { orden: 0, y0: 0, drift: 0, ex: 0, ey: 0, err: 0, stood: 0 };
for (let i = 1; i <= N; i++) {
  const start = sim.fillTrainingStart({}, makeSeededRandom(config.eval.seed + LEVEL * 10007 + i * 17), {
    full: true,
    level: LEVEL,
  });
  // A fresh world per seed, like every scored episode. This one needs the foot position both
  // at the start and at the end, so it steps the episode itself instead of using runOneEpisode.
  const world = sim.build();
  const episode = new sim.Episode(world, weights, start.tilt || 0, start.drop || 0, start);
  const footAtStart = episode.measure.partX[footPart];
  const y0 = episode.measure.partY[footPart] - start.targets[footSlot].y;
  const orden = Math.abs(start.targets[footSlot].dx || 0);
  while (!episode.advanceOneFrame());
  const drift = episode.measure.partX[footPart] - footAtStart;
  const errX = (start.targets[footSlot].x || 0) - episode.measure.partX[footPart];
  const errY = (start.targets[footSlot].y || 0) - episode.measure.partY[footPart];
  const err = markError(episode.measure, footSlot).dist;
  sum.orden += orden;
  sum.y0 += Math.abs(y0);
  sum.drift += Math.abs(drift);
  sum.ex += Math.abs(errX);
  sum.ey += Math.abs(errY);
  sum.err += err;
  if (episode.stats.stoodUp) sum.stood++;
  console.log(
    String(i).padStart(7) + orden.toFixed(3).padStart(9) + y0.toFixed(3).padStart(9) +
    drift.toFixed(3).padStart(9) + '   |' + errX.toFixed(3).padStart(9) +
    errY.toFixed(3).padStart(9) + err.toFixed(3).padStart(9),
  );
  episode.cleanup();
  sim.destroyWorld(world);
}

const mean = (key) => sum[key] / N;
console.log('');
console.log('promedio            |   err_x ' + mean('ex').toFixed(3) + '    err_y ' + mean('ey').toFixed(3) +
  '    err ' + mean('err').toFixed(3));
console.log('orden ' + mean('orden').toFixed(3) + '   desvio del piso al plantar ' + mean('y0').toFixed(3) +
  '   deriva del cuerpo ' + mean('drift').toFixed(3) + '   episodes parado ' + sum.stood + '/' + N);
console.log('');

// El diagnostico: el error tiene que ser explicable por la orden y por la deriva del cuerpo.
// Si hay mucho mas, hay un termino que se nos esta escapando y la fila no mide lo que creemos.
//
// OJO con la deriva: con la marca del pie congelada en el mundo, la fila cobra DOS cosas a la vez,
// (a) obedecer y (b) que el cuerpo no se vaya. La segunda NO es obediencia. Y si la red todavia no
// sabe pararse, (b) es enorme y (a) es invisible: aca el numero no dice nada sobre la fila del pie,
// dice que skill 1 todavia no esta. Por eso esto solo FALLA si la red se para; si no, avisa.
const unexplained = mean('err') - Math.hypot(mean('orden'), mean('ey'));
const stands = sum.stood / N;
const canStand = stands >= 0.6;
console.log('orden + piso explican ' + Math.hypot(mean('orden'), mean('ey')).toFixed(3) +
  ' m de los ' + mean('err').toFixed(3) + ' m que ve la fila.');
console.log('');
if (!canStand) {
  console.log('AVISO: la red se para en ' + Math.round(stands * 100) + '% de los episodios, asi que este');
  console.log('       numero mide skill 1, no la fila del pie. Con el cuerpo en el piso la deriva');
  console.log('       (' + mean('drift').toFixed(3) + ' m) le gana a la orden (' + mean('orden').toFixed(3) +
    ' m) por ' + (mean('drift') / Math.max(1e-6, mean('orden'))).toFixed(1) + ' a 1.');
  console.log('       La fila del pie no se puede juzgar hasta que la red se pare. Correr con --net.');
  console.log('       No se falla el chequeo por esto: seria un fallo de skill 1, no del premio.');
} else if (mean('ey') > mean('orden') * 1.5) {
  console.log('FALLA: el error vertical domina. La marca se planta a ' +
    config.skill.footGroundMeters + ' m pero el pie nace mas arriba. La fila le cobra al pie una');
  console.log('       altura que nadie le pidio, y ese termino dwarfs la orden.');
  process.exitCode = 1;
} else if (mean('ex') > mean('orden') * 2) {
  console.log('FALLA: la red se para y aun asi el error horizontal es mas del doble de la orden.');
  console.log('       Deriva ' + mean('drift').toFixed(3) + ' m contra una orden de ' +
    mean('orden').toFixed(3) + ' m. Se esta cobrando que el cuerpo no se vaya, no que el pie obedezca.');
  process.exitCode = 1;
} else if (unexplained > mean('orden') * 1.5) {
  console.log('FALLA: sobra ' + unexplained.toFixed(3) + ' m sin explicar.');
  process.exitCode = 1;
} else {
  console.log('ok: el error de la fila es la orden mas el piso. La fila mide lo que dice medir.');
}
console.log('');
console.log('Leyer: con la marca del pie CONGELADA en el mundo, la fila cobra dos cosas a la vez:');
console.log('      (a) obedecer, y (b) que el cuerpo no se vaya. La segunda no es obediencia.');
console.log('      Se probo anclar la marca a la cadera para borrar (b), y ARRUINA skill 1: la cadera');
console.log('      baja con los pies cuando el torso cae, asi que el pie se cobra el punto mientras el');
console.log('      cuerpo esta en el piso. Ver README. La deriva no se arregla con el ancla: es skill 1.');
