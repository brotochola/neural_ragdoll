// Una barra que un net que IGNORA las ordenes puede cumplir no sirve de nada.
// Esto sortea ordenes, supone que la parte se queda en el neutral, y mide que fraction
// de episodios pasaria igual. Tiene que quedar MUY por debajo de config.eval.passRate.
//   node check-obey.js [muestras]
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
const neutral = config.targets.map((spec) => ({ x: spec.x || 0, y: (spec.y || 0) - P, angle: 0 }));

function groupOfId(id) {
  return id === 'head' ? 'head' : id === 'footR' || id === 'footL' ? 'foot' : 'hand';
}

console.log('Un net que ignora las ordenes y se queda en el neutral:');
console.log('');

let worst = 0;
for (const group of ['head', 'hand', 'foot']) {
  const random = makeSeededRandom(2024);
  let pass = 0;
  const per = { head: [0, 0], hand: [0, 0], foot: [0, 0] };
  for (let n = 0; n < samples; n++) {
    const start = sim.fillTrainingStart({}, random, { full: true, level: 1, commands: group });
    let allPass = true;
    for (let t = 0; t < config.targets.length; t++) {
      const spec = config.targets[t];
      const g = groupOfId(spec.id);
      if (g !== group) continue;
      const target = start.targets[t];
      let dist;
      let angle;
      if (spec.ride) {
        // La parte se queda en el neutral; el error es la orden misma.
        dist = Math.hypot(target.lx - neutral[t].x, target.ly - neutral[t].y);
        angle = Math.abs(target.ca || 0);
      } else {
        dist = Math.abs(target.dx || 0);
        angle = Math.abs(target.turn || 0);
      }
      const tol = spec.ride ? config.skill.targetReachMeters : config.skill.footReachMeters;
      const tolAngle = spec.ride ? config.skill.targetReachRadians : config.skill.footReachRadians;
      const good = dist <= tol && angle <= tolAngle;
      per[g][1]++;
      if (good) per[g][0]++;
      else allPass = false;
    }
    if (allPass) pass++;
  }
  worst = Math.max(worst, pass / samples);
  const rate = pass / samples;
  const bar = config.eval.passRate[group];
  // El criterio: ignorar tiene que dar menos de la mitad de la barra. Si da mas, la barra
  // no mide obediencia y el nivel se puede pasar sin aprender a obedecer.
  const good = rate <= bar * 0.5;
  console.log(
    '  ' + group.padEnd(5) +
      '  ignora -> pasa ' + (rate * 100).toFixed(0) + '%' +
      '   barra ' + (bar * 100).toFixed(0) + '%' +
      '   ' + (good ? 'ok' : 'PELIGROSO, la barra no mide obediencia'),
  );
}

console.log('');
const allGood = worst === 0 || worst <= 0.5 * Math.max(...['head', 'hand', 'foot'].map((g) => config.eval.passRate[g]));
console.log(allGood
  ? 'ok: ignorar las ordenes no pasa ninguna barra. La barra mide obediencia.'
  : 'PELIGROSO: ignorar las ordenes pasa una barra. Hay que angostar el comando o apretar la tolerancia.');
process.exitCode = allGood ? 0 : 1;