// The official evaluation must not depend on how many workers ran it.
//
// Box2D warm-starts every contact from the impulse it applied last step, and that cache survives
// a body transform reset. Workers used to build one world and reset it for every episode, so a
// worker that had already run six episodes answered differently from one running its first. The
// gate said 1/20 with 12 workers and 8/20 with 1, on the same network and the same seeds.
//
//   node check-determinism.js <checkpoint.json>
//   node check-determinism.js --all <a.json> <b.json>
import path from 'node:path';
import fs from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { config, setSkill } from './config.js';
import { ElmanNetwork } from './src/net.js';
import { makeSeededRandom } from './src/utils.js';

const root = path.dirname(fileURLToPath(import.meta.url));
const Box2D = await import(pathToFileURL(path.join(root, 'vendor/PhaserBox2D.min.js')).href);
const { createSim } = await import('./src/sim.js');
const { isStanding } = await import('./src/reward.js');
const sim = createSim(Box2D);

const argv = process.argv.slice(2);
const files = argv.filter((a) => a.endsWith('.json'));
const level = argv.includes('--skill') ? +argv[argv.indexOf('--skill') + 1] : 1;
setSkill(level);

// The same job list the official eval builds: fixed seeds, full length, no ramp.
function evalJobs(n) {
  const jobs = [];
  for (let i = 0; i < n; i++) {
    const rand = makeSeededRandom(config.eval.seed + level * 10007 + i * 17);
    const start = sim.fillTrainingStart({}, rand, { full: true, level });
    jobs.push(Object.assign({ phase: i % config.cadence.actionEveryNFrames }, start));
  }
  return jobs;
}

// One world, rebuilt before every episode. This is what the workers do now.
function scoreRebuilt(jobs, weights) {
  const world = sim.build();
  const out = [];
  for (const opts of jobs) {
    const episode = sim.runOneEpisode(weights, opts, world);
    const s = episode.stats;
    out.push({
      fitness: s.fitness,
      stood: s.stoodUp ? 1 : 0,
      window: s.windowFrames ? s.windowStood / s.windowFrames : 0,
    });
  }
  episodeCleanup(world);
  return out;
}

function episodeCleanup(world) {
  sim.destroyWorld(world);
}

// The old behaviour: one world, reset() for every episode.
function scoreReused(jobs, weights) {
  const world = sim.build();
  const episode = new sim.Episode(world, weights, 0, 0, {});
  const out = [];
  for (const opts of jobs) {
    episode.reset(weights, opts.tilt || 0, opts.drop || 0, opts);
    while (!episode.advanceOneFrame());
    const s = episode.stats;
    out.push({
      fitness: s.fitness,
      stood: s.stoodUp ? 1 : 0,
      window: s.windowFrames ? s.windowStood / s.windowFrames : 0,
      headMark: s.headMark == null ? 0 : s.headMark,
    });
  }
  episode.cleanup();
  sim.destroyWorld(world);
  return out;
}
const N = 20;
const jobs = evalJobs(N);
let failures = 0;
function check(name, ok, detail) {
  if (!ok) failures++;
  console.log((ok ? 'ok   ' : 'FALLA') + '  ' + name + (detail ? '   ' + detail : ''));
}

console.log('El eval oficial NO puede depender de cuantos workers lo corro.');
console.log('');

if (files.length >= 2) {
  // Two networks, each scored twice on fresh worlds. Identical results are the requirement.
  for (const file of files) {
    const raw = JSON.parse(fs.readFileSync(path.resolve(root, file), 'utf8'));
    if (!raw.theta || raw.theta.length !== ElmanNetwork.weightCount) {
      console.log('FALLA: ' + file + ' no carga en la red de ' + ElmanNetwork.weightCount + ' pesos');
      failures++;
      continue;
    }
    const w = Float32Array.from(raw.theta);
    const a = scoreRebuilt(jobs, w);
    const b = scoreRebuilt(jobs, w);
    const same = a.every((row, i) =>
      Math.abs(row.fitness - b[i].fitness) < 1e-12 &&
      row.stood === b[i].stood &&
      Math.abs(row.window - b[i].window) < 1e-12);
    const ok = a.filter((r) => r.stood).length;
    check(file.split(/[\\/]/).pop() + ' identical en dos corridas', same,
      'barra ' + ok + '/20  fitness ' + (a.reduce((s, r) => s + r.fitness, 0) / N).toFixed(4));
  }
} else {
  // No checkpoint needed: the same net is scored many times, and the episodes are counted too,
  // because b2CreateWorldArray caps a process at 32 worlds and a worker must not run out.
  const w = new Float32Array(ElmanNetwork.weightCount);
  for (let i = 0; i < ElmanNetwork.weightCount; i++) w[i] = (Math.sin(i * 12.9898) * 43758.5453 % 1) * 0.4;
  const a = scoreRebuilt(jobs, w);
  const b = scoreRebuilt(jobs, w);
  const same = a.every((row, i) =>
    Math.abs(row.fitness - b[i].fitness) < 1e-12 &&
    row.stood === b[i].stood &&
    Math.abs(row.window - b[i].window) < 1e-12);
  check('el mismo net da lo mismo en dos corridas', same,
    'barra ' + a.filter((r) => r.stood).length + '/20');

  // More episodes than the 32-world ceiling, all in one world.
  const many = [];
  for (let i = 0; i < 40; i++) many.push(evalJobs(1)[0]);
  const world = sim.build();
  let survived = 0;
  for (const o of many) {
    const episode = sim.runOneEpisode(w, o, world);
    if (Number.isFinite(episode.stats.fitness)) survived++;
  }
  sim.destroyWorld(world);
  check('40 episodios en un solo mundo (el techo de Box2D es 32)', survived === many.length,
    survived + '/' + many.length);

  const reused = scoreReused(jobs, w);
  const drift = Math.max(...reused.map((r, i) => Math.abs(r.fitness - a[i].fitness)));
  console.log('');
  console.log('  un mundo reusado SIN rebuild: el fitness se mueve hasta ' + drift.toExponential(2));
  console.log('  (asi era antes. Esta vez se usa solo para mostrar por que existia el bug.)');
}

console.log('');
if (failures) {
  console.log(failures + ' chequeos fallaron');
  process.exitCode = 1;
} else {
  console.log('ok: el eval es reproducible');
}
void isStanding;
