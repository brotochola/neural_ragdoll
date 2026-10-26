// A walking controller in the style of SIMBICON (Yin, Loken & van de Panne, 2007), searched with the
// cross-entropy method, as a teacher for a walk expert network. Two legs, a finite-state machine:
//   stance on one leg for T seconds (until the swing foot lands), then swap.
//   Swing hip: a target in WORLD frame, theta0 + cd*d + cv*v, where d is the centre of mass ahead of
//   the stance ankle and v its speed. This balance feedback is what keeps SIMBICON walkers up.
//   Stance hip: whatever holds the torso at a set lean. Knees and ankles: fixed targets per state.
// The trained walker scooted (one leg did every step); this alternates by construction.
//   node walk-simbicon.js --iters 60 --out trained/walk-simbicon.json
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
const out = arg('--out', 'trained/walk-simbicon.json');
const pop = +arg('--pop', 24);
const elite = +arg('--elite', 6);
const seeds = +arg('--seeds', 4);
const init = arg('--init', null);

const Box2D = await import(pathToFileURL(path.join(root, 'vendor/PhaserBox2D.min.js')).href);
const { createSim, feetOnBox } = await import('./src/sim.js');
const { ElmanNetwork, outputCountOf } = await import('./src/net.js');
const { makeSeededRandom, randomNormalSample } = await import('./src/utils.js');
const { evalMode } = await import('./src/training.js');

setSkill(6);
const sim = createSim(Box2D);
const world = sim.build();
const weights = ElmanNetwork.copy(JSON.parse(fs.readFileSync(path.join(root, 'trained/stand-and-walk.json'), 'utf8')).theta);
const J = config.net.jointCount;
const OUT = outputCountOf();
const P = config.partIndex;
const { SPEC, decode: decodeBase, makeController } = await import('./src/simbicon.js');
const simbicon = makeController(world);
const dims = SPEC.length;
const track = args.includes('--track'); // swing feedback on speed relative to the commanded speed
// --clock: the controller follows the network's gait clock (copyable). --hold N: it acts only every
// N frames, like the network (config.cadence.actionEveryNFrames = 6).
const clock = args.includes('--clock');
const hold = +arg('--hold', 0);
// --speed-min/--speed-max: commanded speed range (the walk level asks 0.2-0.45 m/s, a very slow
// stroll with 10-20 cm steps; people walk 1-1.4 m/s). --period: the gait clock cycle (s).
// --run: running is scored as staying upright with the head high; flight between steps is allowed.
const speedMin = +arg('--speed-min', 0);
const speedMax = +arg('--speed-max', 0);
const run = args.includes('--run');
// --stairs N --rise R --tread T: walk up a flight of stairs that starts 0.6-0.9 m ahead.
const stairsN = +arg('--stairs', 0);
const stairRise = +arg('--rise', 0.1);
const stairTread = +arg('--tread', 0.3);
if (arg('--period', null)) config.walk.periodSeconds = +arg('--period', 0.9);
if (speedMax > 0) config.walk.maxStandSpeed = Math.max(config.walk.maxStandSpeed, speedMax * 1.6);
const decode = (v) => Object.assign(decodeBase(v), { track, clock });

function episode(p, seed) {
  const r = makeSeededRandom(3000 + seed);
  const start = sim.fillTrainingStart({}, r, evalMode(6));
  if (speedMax > 0) start.walk.speed = speedMin + r() * (speedMax - speedMin);
  if (stairsN > 0) {
    start.stairs = { n: stairsN, rise: stairRise, run: stairTread, near: 0.6 + r() * 0.3, landing: 2 };
    start.frames = Math.round((6 + stairsN * 1.6) * config.physics.hz);
    start.walk.speed = Math.max(start.walk.speed, 0.35);
  }
  world.rebuild();
  const ep = new sim.Episode(world, weights, start.tilt || 0, start.drop || 0, Object.assign({ phase: seed % 6 }, start));
  const delay = Math.round((start.walk.delay || 0) * config.physics.hz);
  const orig = ep.applyMotors.bind(ep);
  const now = new Float64Array(OUT);
  const held = new Float64Array(OUT);
  ep.applyMotors = function () {
    if (this.frame >= delay) {
      simbicon(p, this, now);
      if (!hold || this.frame % hold === 0) held.set(now);
      for (let j = 0; j < OUT; j++) this.action[j] = held[j];
    }
    orig();
  };
  const feet = [P.footLeft, P.footRight];
  const up = [false, false];
  const leftAt = [0, 0];
  const steps = [0, 0];
  let flight = 0;
  let frames = 0;
  let upright = 0;
  let stride = 0;
  let stepsUp = 0;
  while (!ep.advanceOneFrame()) {
    if (ep.frame < delay) continue;
    frames++;
    if (!ep.measure.touch[feet[0]] && !ep.measure.touch[feet[1]]) flight++;
    if (ep.measure.headHeight > 1.35 && ep.measure.torsoCos > 0.8) upright++;
    // A step counts when a foot stands on it (on stairs the feet are on different steps).
    if (ep.stairs && ep.measure.torsoCos > 0.7) for (let k = stepsUp; k < ep.stairs.length; k++) if (feetOnBox(ep.measure, ep.stairs[k]) >= 1) stepsUp = k + 1;
    for (let k = 0; k < 2; k++) {
      const down = !!ep.measure.touch[feet[k]];
      const x = world.px[feet[k]];
      if (!down && !up[k]) {
        up[k] = true;
        leftAt[k] = x;
      } else if (down && up[k]) {
        up[k] = false;
        if (x - leftAt[k] > 0.04) {
          steps[k]++;
          stride += x - leftAt[k];
        }
      }
    }
  }
  const s = ep.stats;
  const wf = s.windowFrames || 0;
  const share = run ? upright / Math.max(1, frames) : wf > 0 ? s.windowStood / wf : 0;
  const walked = s.walkCmd > 0 ? Math.max(0, Math.min(1.3, s.walkDist / s.walkCmd)) : 0;
  const pass = !s.headHit && share >= (run ? 0.9 : config.walk.standShare) && s.walkCmd > 0 && s.walkDist >= config.walk.passShare * s.walkCmd;
  const flightShare = flight / Math.max(1, frames);
  const lo = Math.min(steps[0], steps[1]);
  const hi = Math.max(steps[0], steps[1]);
  const alt = hi > 0 ? (lo / hi) * Math.min(1, lo / 3) : 0;
  const alive = Math.min(1, (ep.measure.headHeight || 0) / config.body.standingHeadMeters);
  // With --track: walking at the commanded speed matters (the first controller walked 3.2 m for 2.0).
  const speedErr = track && s.walkCmd > 0 ? Math.abs(Math.min(2, (s.walkDist || 0) / s.walkCmd) - 1) : 0;
  const score = -1.5 * speedErr + 3 * share + 2 * Math.min(1, walked) * (0.3 + 0.7 * share) + 2 * alt * alive + 2 * alive - (run ? 0 : 2) * flightShare + (pass ? 3 : 0);
  if (stairsN > 0) {
    const topNow = ep.stairs ? ep.stairs[ep.stairs.length - 1].top : 0;
    const up = (ep.measure.headHeight - topNow) > 1.35 && ep.measure.torsoCos > 0.8;
    const passS = stepsUp === stairsN && up && feetOnBox(ep.measure, ep.stairs[ep.stairs.length - 1]) === 2;
    const scoreS = 3 * stepsUp / stairsN + (up ? 1 : 0) + 2 * alive + (passS ? 3 : 0) + alt;
    return { score: scoreS, pass: passS ? 1 : 0, alt, why: 'steps ' + stepsUp + '/' + stairsN + ' up ' + (up ? 1 : 0) + ' head ' + ep.measure.headHeight.toFixed(2) };
  }
  const why = 'stood ' + share.toFixed(2) + ' walked ' + (s.walkDist || 0).toFixed(2) + '/' + (s.walkCmd || 0).toFixed(2) + ' steps ' + steps.join('/') + ' flight ' + flightShare.toFixed(2) + ' step ' + (stride / Math.max(1, steps[0] + steps[1])).toFixed(2) + 'm speed ' + (start.walk.speed).toFixed(2) + ' head ' + (ep.measure.headHeight || 0).toFixed(2);
  return { score, pass: pass ? 1 : 0, alt, why };
}

// seedBase: each generation draws new practice walks (the first search used 4 fixed ones and
// overfit them: 10/12 on new walks, 12/20 on the official starts).
function evaluate(v, seedBase = 0, n = seeds) {
  const p = decode(v);
  let s = 0;
  let pass = 0;
  let alt = 0;
  for (let i = 0; i < n; i++) {
    const r = episode(p, seedBase + i);
    s += r.score;
    pass += r.pass;
    alt += r.alt;
  }
  return { score: s / n, pass, alt: alt / n };
}

if (args.includes('--show')) {
  const p = decode(JSON.parse(fs.readFileSync(init, 'utf8')).best);
  for (let i = 0; i < seeds; i++) {
    const r = episode(p, 100 + i);
    console.log('seed', i, 'pass', r.pass, 'alt', r.alt.toFixed(2), r.why);
  }
  process.exit(0);
}

const rand = makeSeededRandom(66);
const normal = () => randomNormalSample(rand);
const mean = new Float64Array(dims);
if (init) {
  const saved = JSON.parse(fs.readFileSync(init, 'utf8')).best;
  for (let d = 0; d < dims; d++) mean[d] = saved[d];
}
const sigma = Float64Array.from(SPEC.map((s) => s[2] * (init ? 0.5 : 1)));
const floor = SPEC.map((s) => s[2] * 0.15);
let best = null;
for (let it = 0; it < iters; it++) {
  const cands = [];
  for (let c = 0; c < pop; c++) {
    const v = new Float64Array(dims);
    for (let d = 0; d < dims; d++) v[d] = mean[d] + sigma[d] * normal();
    cands.push({ v, ...evaluate(v, 1000 + it * seeds) });
  }
  cands.sort((a, b) => b.score - a.score);
  // Keep the version that does best on 12 held-out walks (starts 500+), checked every 5 generations.
  if (it % 5 === 4 || it === 0) {
    for (const c of [cands[0], { v: Float64Array.from(mean) }]) {
      const val = evaluate(c.v, 500, 12);
      if (!best || val.score > best.score) best = { v: Float64Array.from(c.v), ...val };
    }
  }
  const top = cands.slice(0, elite);
  for (let d = 0; d < dims; d++) {
    let m = 0;
    for (const c of top) m += c.v[d];
    m /= top.length;
    let s = 0;
    for (const c of top) s += (c.v[d] - m) ** 2;
    s = Math.sqrt(s / top.length);
    mean[d] = 0.7 * m + 0.3 * mean[d];
    sigma[d] = Math.max(floor[d], 0.7 * s + 0.3 * sigma[d]);
  }
  if (it % 5 === 0 || it === iters - 1) {
    console.log('iter', it, 'held-out best', best.score.toFixed(3), 'pass', best.pass + '/12', 'alt', best.alt.toFixed(2), 'gen best', cands[0].score.toFixed(3));
    fs.writeFileSync(out, JSON.stringify({ track, clock, hold, run, period: config.walk.periodSeconds, speedMin, speedMax, stairs: stairsN ? { n: stairsN, rise: stairRise, tread: stairTread } : null, best: Array.from(best.v), params: decode(best.v), score: best.score, pass: best.pass, alt: best.alt }));
  }
}
let fresh = 0;
for (let i = 0; i < 12; i++) fresh += episode(decode(best.v), 100 + i).pass;
fs.writeFileSync(out, JSON.stringify({ track, clock, hold, run, period: config.walk.periodSeconds, speedMin, speedMax, stairs: stairsN ? { n: stairsN, rise: stairRise, tread: stairTread } : null, best: Array.from(best.v), params: decode(best.v), score: best.score, pass: best.pass, alt: best.alt, freshPass: fresh }));
console.log('done fresh pass', fresh + '/12', '->', out);
sim.destroyWorld(world);
