// ¿Puede el premio SOLO de targets premiar estar tirado? Monte Carlo sobre la fisica real,
// sin red: se prueban poses y se mide el premio de las diez filas.
//
//   node check-collapse.js
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { config } from './config.js';
import { ElmanNetwork } from './src/net.js';

const root = path.dirname(fileURLToPath(import.meta.url));
const Box2D = await import(pathToFileURL(path.join(root, 'vendor/PhaserBox2D.min.js')).href);
const { createSim } = await import('./src/sim.js');
const { scoreTotal, scoreByTarget, isStanding } = await import('./src/reward.js');
const { makeSeededRandom } = await import('./src/utils.js');

const sim = createSim(Box2D);
const frames = Math.round(config.episode.seconds * config.physics.hz);
// Pesos del largo EXACTO. Un vector corto hace que la red lea fuera y devuelva NaN, y un NaN
// pasa toFixed como "0.00": asi un chequeo roto se hace pasar por bueno.
const weights = new Float32Array(ElmanNetwork.weightCount);
let nan = false;

// tilt 0 es parado. Los otros lo tiran al piso y lo dejan'asentar'.
// La pregunta es: con el premio SOLO de las diez filas, ¿parado vale mas que tirado?
// Esto se mide sobre una red REAL (la de backups/train-state.json), porque una red de pesos
// cero se cae sola y no distingue nada. Ademas se apagan los joints para congelar la pose:
// lo que se compara es el PREMIO de cada pose, no quien la sostiene.
const poses = [
  { name: 'parado', tilt: 0, drop: config.start.uprightDropMeters },
  { name: 'tumbado de espaldas', tilt: Math.PI / 2, drop: 0.05 },
  { name: 'boca abajo', tilt: -Math.PI / 2, drop: 0.05 },
];

function loadTheta() {
  const file = path.join(root, 'backups', 'train-state.json');
  if (!fs.existsSync(file)) return null;
  const saved = JSON.parse(fs.readFileSync(file, 'utf8'));
  if (!saved.theta || saved.theta.length !== ElmanNetwork.weightCount) return null;
  return Float32Array.from(saved.theta);
}

let theta = loadTheta();
if (!theta) {
  theta = weights;
  console.log('sin train-state compatible: uso pesos en cero (la comparacion sera pesima)');
}

const results = [];
for (const pose of poses) {
  const start = sim.fillTrainingStart({}, makeSeededRandom(11), { full: true, level: 1 });
  start.tilt = pose.tilt;
  start.drop = pose.drop;
  start.settle = pose.tilt ? 60 : 0;
  // A fresh world per pose, like every scored episode. The constructor already resets.
  const world = sim.build();
  const episode = new sim.Episode(world, theta, 0, 0, start);
  // Se avanza hasta que la pose se acomode, y se congela el reloj: el frame no avanza mas,
  // asi que el episodio no termina y el doll queda congelado en esa pose.
  for (let i = 0; i < (pose.tilt ? 90 : 30); i++) episode.advanceOneFrame();
  episode.frame = frames - 2;
  let sum = 0;
  let n = 0;
  let stoodFrames = 0;
  while (!episode.advanceOneFrame()) {
    const total = scoreTotal(episode.measure);
    if (!Number.isFinite(total)) nan = true;
    sum += total;
    n++;
    if (isStanding(episode.measure)) stoodFrames++;
    if (n >= 40) break;
  }
  results.push({
    name: pose.name,
    reward: n ? sum / n : 0,
    standing: (stoodFrames / n).toFixed(2),
    height: episode.measure.headHeight,
    torsoCos: episode.measure.torsoCos,
    rows: scoreByTarget(episode.measure).map((row) => ({ id: row.id, position: row.position, rotation: row.rotation })),
  });
  episode.cleanup();
  sim.destroyWorld(world);
}

console.log('las diez filas en la POSE, sin red y sin terminos de supervivencia:');
console.log('');
for (const row of results) {
  console.log(
    '  ' + row.name.padEnd(22) +
      ' premio ' + row.reward.toFixed(2).padStart(7) +
      '   altura ' + row.height.toFixed(2) +
      '   torsoCos ' + row.torsoCos.toFixed(2),
  );
}
console.log('');
// Por fila: si tirado sigue sacando puntos en una fila, esa fila no esta premiando pararse.
console.log('desglose por fila (posicion + giro), en la pose final de cada una:');
console.log('  ' + 'pose'.padEnd(22) + config.targets.map((spec) => spec.id.padStart(15)).join(''));
for (const row of results) {
  const cells = row.rows.map((cell) => (cell.position + cell.rotation).toFixed(2).padStart(15)).join('');
  console.log('  ' + row.name.padEnd(22) + cells);
}
console.log('');
console.log('  (tirado de espaldas deberia perder casi todo: cabeza y manos a 0.12 m contra una');
console.log('   marca a ~1.6 m dan ~0.01, y el pie solo se salva si sigue en el piso.)');
console.log('');
const best = results.reduce((a, b) => (b.reward > a.reward ? b : a));
const stand = results[0];
console.log('  mejor pose: ' + best.name + ' con ' + best.reward.toFixed(2));
console.log('  parado da:  ' + stand.name + ' con ' + stand.reward.toFixed(2));
console.log('');
if (nan) {
  console.log('FALLA: la fisica dio NaN. Ninguna de estas cifras significa nada.');
  process.exitCode = 1;
} else if (Math.abs(stand.height - results[1].height) < 0.05) {
  console.log('FALLA: las poses son iguales, el chequeo no prueba nada.');
  process.exitCode = 1;
} else if (best.reward > stand.reward + 0.05) {
  console.log('PELIGROSO: la mejor pose NO es estar parado. Con el premio solo de targets, la red aprende tirarse.');
  process.exitCode = 1;
} else {
  console.log('ok: estar parado es la pose que mas premio da. Las diez filas ya premian pararse.');
}