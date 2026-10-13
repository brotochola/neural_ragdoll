import path from 'node:path';
import { pathToFileURL } from 'node:url';
const root = 'D:/xampp/htdocs/neural_ragdoll';
const Box2D = await import(pathToFileURL(path.join(root, 'vendor/PhaserBox2D.min.js')).href);
const { createSim } = await import('./src/sim.js');
const { config, setSkill } = await import('./config.js');
const { ElmanNetwork } = await import('./src/net.js');
const { makeSeededRandom } = await import('./src/utils.js');
setSkill(1);
const sim = createSim(Box2D);

const w = new Float32Array(ElmanNetwork.weightCount);
for (let i = 0; i < ElmanNetwork.weightCount; i++) w[i] = (Math.random() - 0.5) * 0.4;
function opts(i) {
  const rand = makeSeededRandom(config.eval.seed + i * 17);
  const start = sim.fillTrainingStart({}, rand, { full: true, level: 1 });
  return Object.assign({ phase: i % config.cadence.actionEveryNFrames }, start);
}

// ONE world, rebuild() before every episode, 60 episodes total: three times more than the
// 32-world ceiling, which proves the slot is being reused.
const world = sim.build();
const ep = new sim.Episode(world, w, 0, 0, {});
const list = [];
for (let i = 0; i < 12; i++) list.push(opts(i));

const passes = [];
for (let pass = 0; pass < 3; pass++) {
  const row = [];
  for (const o of list) {
    world.rebuild();
    ep.reset(w, o.tilt || 0, o.drop || 0, o);
    while (!ep.advanceOneFrame());
    row.push(ep.stats.fitness);
  }
  passes.push(row);
}
console.log('Un mundo, rebuild() por episodio, 36 episodios en el MISMO mundo:');
let worst = 0;
for (let i = 0; i < list.length; i++) {
  for (let p = 1; p < passes.length; p++)
    worst = Math.max(worst, Math.abs(passes[p][i] - passes[0][i]));
}
console.log('  mayor diferencia de fitness entre pasadas: ' + worst.toExponential(2));
console.log('');
console.log(worst < 1e-9
  ? 'ok: rebuild() da el mismo resultado sin importar que episode corrio antes.'
  : 'AUN NO: rebuild() no alcanza.');
ep.cleanup();
