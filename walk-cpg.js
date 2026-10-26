// Searches a reference WALK with both legs: the stand-and-walk net keeps the balance, and a rhythm
// added on top of its hip, knee and ankle commands (and the arms) makes the legs alternate. The left
// leg gets A*sin(clock + phase) + bias, the right leg the same half a cycle later, on the same gait
// clock the net sees (config.walk.periodSeconds). Cross-entropy method over ~16 numbers.
// The trained walker scooted: the right foot did every step, the left one dragged. This scores
// alternation (both feet lifting and landing ahead, about as often) together with walking and standing.
//   node walk-cpg.js --iters 60 --out trained/walk-cpg.json
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { config, setSkill } from './config.js';

const root = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const arg = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : fallback;
};
const iters = +arg('--iters', 60);
const out = arg('--out', 'trained/walk-cpg.json');
const pop = +arg('--pop', 24);
const elite = +arg('--elite', 6);
const seeds = +arg('--seeds', 4);
const netFile = arg('--net', 'trained/stand-and-walk.json');
const init = arg('--init', null);

const Box2D = await import(pathToFileURL(path.join(root, 'vendor/PhaserBox2D.min.js')).href);
const { createSim } = await import('./src/sim.js');
const { ElmanNetwork } = await import('./src/net.js');
const { makeSeededRandom, randomNormalSample } = await import('./src/utils.js');
const { evalMode } = await import('./src/training.js');

setSkill(6);
const sim = createSim(Box2D);
const world = sim.build();
const weights = ElmanNetwork.copy(JSON.parse(fs.readFileSync(path.join(root, netFile), 'utf8')).theta);
const P = config.partIndex;
// Joint order: Waist, Neck, Shoulder L, Elbow L, Wrist L, Hip L, Knee L, Ankle L, Shoulder R, ...
const LEFT = { hip: 5, knee: 6, ankle: 7, shoulder: 2 };
const RIGHT = { hip: 11, knee: 12, ankle: 13, shoulder: 8 };
const parts = ['hip', 'knee', 'ankle', 'shoulder'];
const dims = parts.length * 3 + 1; // amplitude, phase, bias per joint pair + one gain on the net's own leg commands
const rand = makeSeededRandom(55);
const normal = () => randomNormalSample(rand);

function decode(v) {
  const p = {};
  parts.forEach((name, k) => {
    p[name] = { amp: 0.8 * Math.tanh(v[3 * k]), phase: v[3 * k + 1], bias: 0.5 * Math.tanh(v[3 * k + 2]) };
  });
  p.netGain = 1 / (1 + Math.exp(-(v[dims - 1] + 2))); // starts near 0.88: mostly the net's own legs
  return p;
}

// The rhythm on top of the net's command, in command units, clamped to [-1, 1].
export function addRhythm(p, phase, action) {
  for (const name of parts) {
    const q = p[name];
    const l = LEFT[name];
    const r = RIGHT[name];
    const g = name === 'shoulder' ? 1 : p.netGain;
    action[l] = Math.max(-1, Math.min(1, g * action[l] + q.amp * Math.sin(phase + q.phase) + q.bias));
    action[r] = Math.max(-1, Math.min(1, g * action[r] + q.amp * Math.sin(phase + q.phase + Math.PI) + q.bias));
  }
}

function episode(p, seed) {
  const r = makeSeededRandom(3000 + seed);
  const start = sim.fillTrainingStart({}, r, evalMode(6));
  world.rebuild();
  const ep = new sim.Episode(world, weights, start.tilt || 0, start.drop || 0, Object.assign({ phase: seed % 6 }, start));
  const period = config.physics.hz * (config.walk.periodSeconds || 1);
  const orig = ep.applyMotors.bind(ep);
  ep.applyMotors = function () {
    addRhythm(p, (2 * Math.PI * this.frame) / period, this.action);
    orig();
  };
  // Steps: a foot that leaves the floor and comes down at least 4 cm ahead of where it left.
  const feet = [P.footLeft, P.footRight];
  const up = [false, false];
  const leftAt = [0, 0];
  const steps = [0, 0];
  const air = [0, 0];
  let flight = 0; // frames with neither foot down
  let frames = 0;
  const delay = (start.walk.delay || 0) * config.physics.hz;
  while (!ep.advanceOneFrame()) {
    if (ep.frame < delay) continue;
    frames++;
    if (!ep.measure.touch[feet[0]] && !ep.measure.touch[feet[1]]) flight++;
    for (let k = 0; k < 2; k++) {
      const down = !!ep.measure.touch[feet[k]];
      const x = world.px[feet[k]];
      if (!down) air[k]++;
      if (!down && !up[k]) {
        up[k] = true;
        leftAt[k] = x;
      } else if (down && up[k]) {
        up[k] = false;
        if (start.walk.dir * (x - leftAt[k]) > 0.04) steps[k]++;
      }
    }
  }
  const s = ep.stats;
  const wf = s.windowFrames || 0;
  const share = wf > 0 ? s.windowStood / wf : 0;
  const walked = s.walkCmd > 0 ? Math.max(0, Math.min(1, s.walkDist / s.walkCmd)) : 0;
  const pass = !s.headHit && share >= config.walk.standShare && s.walkCmd > 0 && s.walkDist >= config.walk.passShare * s.walkCmd;
  const lo = Math.min(steps[0], steps[1]);
  const hi = Math.max(steps[0], steps[1]);
  const airL = air[0] / Math.max(1, frames);
  const airR = air[1] / Math.max(1, frames);
  // Alternation: both feet step, about equally; capped at the clock's rate (one step per foot per cycle).
  const cycles = frames / period;
  const both = Math.min(1, lo / Math.max(1, 0.6 * cycles));
  const even = hi > 0 ? lo / hi : 0;
  const airEven = Math.max(airL, airR) > 0 ? Math.min(airL, airR) / Math.max(airL, airR) : 0;
  // The first search found a bounding gait: each foot in the air 80-90% of the time, standing 0%.
  // Walking keeps a foot down: flight (no foot down) and a foot up more than 55% cost.
  const flightShare = flight / Math.max(1, frames);
  const overAir = Math.max(0, Math.max(airL, airR) - 0.55) / 0.45;
  const alt = both * (0.6 * even + 0.4 * airEven) * Math.max(0, 1 - 2 * flightShare) * (1 - Math.min(1, overAir));
  const score = 3 * share + 2 * walked * share + 4 * alt * (0.3 + 0.7 * share) + (pass ? 2 : 0) + (pass && alt > 0.6 ? 2 : 0);
  const why = 'flight ' + flightShare.toFixed(2) + ' stood ' + share.toFixed(2) + ' walked ' + (s.walkDist || 0).toFixed(2) + '/' + (s.walkCmd || 0).toFixed(2) + (s.headHit ? ' headHit' : '');
  return { score, pass: pass ? 1 : 0, alt, steps, airL, airR, why };
}

function evaluate(v) {
  const p = decode(v);
  let s = 0;
  let pass = 0;
  let alt = 0;
  for (let i = 0; i < seeds; i++) {
    const r = episode(p, i);
    s += r.score;
    pass += r.pass;
    alt += r.alt;
  }
  return { score: s / seeds, pass, alt: alt / seeds };
}

// --show: run the --init rhythm on --seeds walk starts and print what happened.
if (args.includes('--show')) {
  const p = decode(JSON.parse(fs.readFileSync(init, 'utf8')).best);
  for (let i = 0; i < seeds; i++) {
    const r = episode(p, 100 + i);
    console.log('seed', i, 'pass', r.pass, 'alt', r.alt.toFixed(2), 'steps', r.steps.join('/'), 'air', r.airL.toFixed(2), r.airR.toFixed(2), r.why);
  }
  process.exit(0);
}
let mean = new Float64Array(dims);
if (init) {
  const saved = JSON.parse(fs.readFileSync(init, 'utf8')).best;
  for (let d = 0; d < dims; d++) mean[d] = saved[d];
} else {
  parts.forEach((_, k) => (mean[3 * k] = 0.3)); // a small rhythm to start
}
const sigma = new Float64Array(dims).fill(0.6);
let best = null;
for (let it = 0; it < iters; it++) {
  const cands = [];
  for (let c = 0; c < pop; c++) {
    const v = new Float64Array(dims);
    for (let d = 0; d < dims; d++) v[d] = mean[d] + sigma[d] * normal();
    cands.push({ v, ...evaluate(v) });
  }
  cands.sort((a, b) => b.score - a.score);
  if (!best || cands[0].score > best.score) best = cands[0];
  const top = cands.slice(0, elite);
  for (let d = 0; d < dims; d++) {
    let m = 0;
    for (const c of top) m += c.v[d];
    m /= top.length;
    let s = 0;
    for (const c of top) s += (c.v[d] - m) ** 2;
    s = Math.sqrt(s / top.length);
    mean[d] = 0.7 * m + 0.3 * mean[d];
    sigma[d] = Math.max(0.08, 0.7 * s + 0.3 * sigma[d]);
  }
  if (it % 5 === 0 || it === iters - 1) {
    console.log('iter', it, 'best', best.score.toFixed(3), 'pass', best.pass + '/' + seeds, 'alt', best.alt.toFixed(2), 'gen best', cands[0].score.toFixed(3), 'sigma', (sigma.reduce((a, b) => a + b, 0) / dims).toFixed(3));
    fs.writeFileSync(out, JSON.stringify({ best: Array.from(best.v), params: decode(best.v), score: best.score, pass: best.pass, alt: best.alt }));
  }
}
let fresh = 0;
let freshAlt = 0;
const detail = [];
for (let i = 0; i < 12; i++) {
  const r = episode(decode(best.v), 100 + i);
  fresh += r.pass;
  freshAlt += r.alt;
  detail.push(r.steps.join('/'));
}
fs.writeFileSync(out, JSON.stringify({ best: Array.from(best.v), params: decode(best.v), score: best.score, pass: best.pass, alt: best.alt, freshPass: fresh, freshAlt: freshAlt / 12 }));
console.log('done fresh pass', fresh + '/12', 'alt', (freshAlt / 12).toFixed(2), 'steps L/R', detail.join(' '), '->', out);
sim.destroyWorld(world);
