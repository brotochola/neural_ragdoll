// LEAP: the cat leap onto a rock. Tunes (CEM) how the obstacle runner (src/obstacle-runner.js, leap mode)
// runs at the rock, jumps, grabs the highest knob it can reach in the air and hangs, before the crag planner
// climbs on. --stage catch: episodes end once the planner has a foot on the rock (fast); --stage full: on
// over the top and running. Held-out rocks pick the kept version.
//   node leap.js --iters 40 --out trained/leap.json [--stage catch|full] [--init trained/leap.json]
//   node leap.js --show --init trained/leap.json --seeds 10 [--stage full]
// (from obstacle.js:) OBSTACLE: running into a 2 m irregular rock obstacle and going on over it. Running (the parkour
// runner), it slows to a walk (the SIMBICON walker), stops a set distance from the rock and stands
// still (the stop controller, then the pose executor), climbs it (the crag planner: knobs, three points,
// belly mantle, the get-up expert on top), walks off and runs on along the top. Each piece is an
// earlier skill; this script chains them and counts how far the chain gets.
//   node obstacle.js --show --seeds 6 [--crag trained/crag-g0.json] [--record out.json]
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
const seeds = +arg('--seeds', 6);
const seconds = +arg('--seconds', 150);
const H = +arg('--height', 2.0);

const Box2D = await import(pathToFileURL(path.join(root, 'vendor/PhaserBox2D.min.js')).href);
const { createSim } = await import('./src/sim.js');
const { ElmanNetwork } = await import('./src/net.js');
const { makeSeededRandom } = await import('./src/utils.js');
const { evalMode } = await import('./src/training.js');
const pose = await import('./src/pose.js');
const simb = await import('./src/simbicon.js');
const runMod = await import('./src/runner.js');
const { installObstacleRunner } = await import('./src/obstacle-runner.js');
const { makeCrag } = await import('./src/terrain.js');

setSkill(6);
config.walk.maxStandSpeed = Math.max(config.walk.maxStandSpeed, 3);
const sim = createSim(Box2D);
const world = sim.build();
const HZ = config.physics.hz, J = config.net.jointCount, P = config.partIndex;
const load = (f) => JSON.parse(fs.readFileSync(path.join(root, f), 'utf8'));
const weights = ElmanNetwork.copy(load('trained/stand-and-walk.json').theta);
const runStates = load('trained/run-states.json');
const runP = load('trained/runner-gap-12c.json').params;
const wf = load('trained/walk-fast-controller.json');
const walkP = Object.assign(simb.decode(wf.best), { track: !!wf.track });
const ws = load('trained/walk-controller.json');
const slowP = Object.assign(simb.decode(ws.best), { track: !!ws.track });
const st = load('trained/stop-controller.json').params;
const stopP = Object.assign({}, slowP, st, { track: true });
const getup = load('trained/getup-expert-19.json');
// the crag planner's numbers, decoded as crag.js does
const cragSrc = fs.readFileSync(path.join(root, 'crag.js'), 'utf8');
const CSPEC = eval(cragSrc.slice(cragSrc.indexOf('const SPEC = [') + 13, cragSrc.indexOf('];', cragSrc.indexOf('const SPEC = [')) + 1));
const cragV = load(arg('--crag', 'trained/crag-g0.json')).best;
const cragP = {};
CSPEC.forEach(([k, b], i) => (cragP[k] = b + (cragV[i] || 0)));
for (const k of ['tHand', 'tFoot', 'tRise', 'tPress', 'tOver', 'tPull']) cragP[k] = Math.max(0.2, cragP[k]);
cragP.belly = 1;
cragP.doneX = +(process.env.DONE_X || 3.0); // crawl well away from the edge: getting up it can go 3 m either way
const stopAt = +(process.env.STOP_AT || 0.6); // front foot this far from the rock when it stops

const stage = arg('--stage', 'catch');
const iters = +arg('--iters', 40), pop = +arg('--pop', 16), elite = +arg('--elite', 4), nTrain = +arg('--n', 6), held = +arg('--held', 16);
const out = arg('--out', 'trained/leap.json');
export const SPEC = [
  ['vRun', 1.9, 0.4], // run-up speed
  ['takeoff', 1.0, 0.3], // jump on the touchdown nearest this far from the rock's foot
  ['jumpScale', 1.3, 0.4], // the runner's jump changes, times this
  ['reachH', 2.0, 0.15], // aim at the highest knob up to this high
  ['aimX', 0.06, 0.06], ['aimY', 0.35, 0.15], // aim point off the knob's lip
  ['catchR', 0.6, 0.15], // a hand this near the aim point closes
  ['legHip', 0.1, 0.5], ['legKnee', -1.4, 0.4], ['leapWaist', -0.2, 0.25], // in the air
  ['hangHip', 0.6, 0.4], ['hangKnee', -1.0, 0.4], ['hangT', 0.6, 0.3], // hanging, before the planner
  ['kStride', 0.8, 0.3], ['preReach', 0.0, 0.3], // step adjustment to the take-off spot; arms up from this far
  ['reachT', 0.8, 0.3], ['toe', 0.4, 0.3], // keeps reaching this long (standing tall on the toes once down)
];
const decode = (v) => { const q = {}; SPEC.forEach(([k, b], i) => (q[k] = process.env['P_' + k] != null ? +process.env['P_' + k] : b + (v[i] || 0))); q.catchR = Math.max(0.05, q.catchR); q.hangT = Math.max(0.2, q.hangT); return q; };

function episode(seed, leapP, log) {
  const r = makeSeededRandom(17000 + seed);
  const start = sim.fillTrainingStart({}, r, evalMode(6));
  const crag = makeCrag(r, H, 7 + r() * 3, { plateau: 14 });
  const rs = runStates[Math.floor(r() * runStates.length)];
  const speed = leapP.vRun;
  const T = stage === 'catch' ? 14 : seconds;
  Object.assign(start, { startPose: rs.pose.slice(), startVel: rs.vel.map((v, i) => (i % 3 === 0 ? v * speed / rs.speed : v)), tilt: 0, drop: 0, settle: 0, frames: Math.round(T * HZ) });
  start.walk = null;
  start.terrain = { statics: [], polys: crag.polys, surface: crag.surface };
  world.rebuild();
  const ep = new sim.Episode(world, weights, 0, 0, Object.assign({ phase: 0 }, start));
  ep.runner = { stance: rs.stance, t: -0.1, flight: false, jumping: false, jumpNext: false, strides: 0 };
  const S = ep.terrain.surface, edgeX = S[S.length - 2][0];
  const OR = installObstacleRunner(ep, world, { debug: !!process.env.DEBUG_LEAP, config, pose, Box2D, ElmanNetwork, crag, H, speed: () => speed, stopAt, leapP, params: { runP, walkP, slowP, stopP, stopK: st, cragP, getup } });
  let fell = false, done = false, footOn = false, maxHand = 0;
  while (!ep.advanceOneFrame()) {
    const m = ep.measure;
    const floorY = world.px[0] > edgeX - 0.1 && world.py[0] > H - 0.2 ? H : 0;
    if (OR.mode !== 'climb' && OR.mode !== 'hang' && OR.mode !== 'leap' && m.headHeight - floorY < 0.6) { fell = true; break; }
    if ((OR.mode === 'hang' || OR.mode === 'leap') && world.py[0] < 0.45) { fell = true; break; }
    if (OR.mode === 'climb' && OR.planner && !OR.planner.phase && m.headHeight < world.py[0]) { fell = true; break; }
    if (process.env.DEBUG_PL && OR.planner && ep.frame % 120 === 0) { const L = OR.planner.lim, mv = OR.planner.move; console.log('   t', (ep.frame / HZ).toFixed(0), 'move', mv ? mv.kind + (mv.k != null ? mv.k : '') + '->' + (mv.to != null ? mv.to : '') : '-', 'hands', L.hand.map((h) => h.rung + (h.grip ? 'g' : '')).join(','), 'feet', L.foot.map((f) => f.rung + '@' + f.pos.y.toFixed(2)).join(','), 'grips', ep.grips.map((g) => (g.joint ? 1 : 0)).join(''), 'pelvis', world.px[0].toFixed(2), world.py[0].toFixed(2), 'phase', OR.planner.phase); }
    if (OR.planner) { const L = OR.planner.lim; maxHand = Math.max(maxHand, L.hand[0].rung, L.hand[1].rung); if (L.foot[0].rung > 0 || L.foot[1].rung > 0) footOn = true; }
    if (stage === 'catch' && footOn && OR.mode === 'climb') { done = true; break; }
    if (stage === 'catch' && OR.leap.missed) break;
    if (OR.mode === 'run2' && world.px[0] > edgeX + 5) { done = true; break; }
  }
  const L = OR.leap, N = crag.knobs.length;
  const caught = L.caughtY != null;
  let score = 0;
  if (L.tookOff != null) score += 1;
  if (L.minD != null) score += 1.5 * (1 - Math.min(1, L.minD / 0.5)); // a hand brought near the knob
  if (caught) score += 3 + 2 * Math.max(0, Math.min(1, (L.caughtY - 1.2) / 0.8));
  if (OR.mode === 'climb' || OR.mode === 'walk2' || OR.mode === 'run2') score += 2;
  if (footOn) score += 2;
  if (stage === 'full') {
    score += 2 * maxHand / (N + 1);
    if (OR.planner && OR.planner.mantled) score += 3;
    if (OR.planner && OR.planner.stood) score += 3;
    if (done) score += 4;
  }
  if (fell) score -= 1;
  const pass = stage === 'catch' ? (done ? 1 : 0) : done ? 1 : 0;
  if (log) console.log('seed', seed, pass ? 'OK' : fell ? 'FELL in ' + OR.mode : 'stopped in ' + OR.mode + (OR.planner && OR.planner.phase ? '/' + OR.planner.phase : ''), '| aim knob', L.aimI + '/' + N, caught ? 'caught at ' + L.caughtY.toFixed(2) + ' m' : L.missed ? 'MISSED' : 'no catch', '| ' + OR.log.join(', '), (ep.frame / HZ).toFixed(1) + 's');
  return { score, pass, caught: caught ? 1 : 0 };
}
const evaluate = (v, base, n, log) => { const q = decode(v); let s = 0, pass = 0, c = 0; for (let i = 0; i < n; i++) { const e = episode(base + i, q, log); s += e.score; pass += e.pass; c += e.caught; } return { score: s / n, pass, caught: c }; };

const init = arg('--init', null);
const mean = new Float64Array(SPEC.length);
if (init) { const sv = load(init).best; for (let d = 0; d < SPEC.length && d < sv.length; d++) mean[d] = sv[d]; }
if (args.includes('--show')) {
  const e = evaluate(mean, +arg('--base', 100), seeds, true);
  console.log('pass', e.pass + '/' + seeds, 'caught', e.caught + '/' + seeds, 'score', e.score.toFixed(3));
  process.exit(0);
}
const { randomNormalSample } = await import('./src/utils.js');
const rand = makeSeededRandom(+arg('--rng', 31));
const sigma = Float64Array.from(SPEC.map((q) => q[2]));
const floor = SPEC.map((q) => q[2] * 0.15);
let best = null;
const t0 = Date.now();
for (let it = 0; it < iters; it++) {
  const cands = [];
  for (let c = 0; c < pop; c++) {
    const v = new Float64Array(SPEC.length);
    for (let d = 0; d < SPEC.length; d++) v[d] = mean[d] + sigma[d] * randomNormalSample(rand);
    cands.push({ v, ...evaluate(v, 1000 + it * nTrain, nTrain) });
  }
  cands.sort((a, b) => b.score - a.score);
  if (it % 3 === 2 || it === 0) {
    for (const c of [cands[0], { v: Float64Array.from(mean) }]) { const h = evaluate(c.v, 500, held); if (!best || h.score > best.score) best = { v: Float64Array.from(c.v), ...h }; }
    fs.writeFileSync(out, JSON.stringify({ stage, best: Array.from(best.v), params: decode(best.v), score: best.score, pass: best.pass, caught: best.caught, held }));
    console.log('iter', it, 'held-out best', best.score.toFixed(3), 'pass', best.pass + '/' + held, 'caught', best.caught + '/' + held, 'gen best', cands[0].score.toFixed(3), ((Date.now() - t0) / 60000).toFixed(1) + ' min');
  }
  const top = cands.slice(0, elite);
  for (let d = 0; d < SPEC.length; d++) {
    let mu = 0; for (const c of top) mu += c.v[d]; mu /= top.length;
    let sd = 0; for (const c of top) sd += (c.v[d] - mu) ** 2; sd = Math.sqrt(sd / top.length);
    mean[d] = 0.7 * mu + 0.3 * mean[d]; sigma[d] = Math.max(floor[d], 0.7 * sd + 0.3 * sigma[d]);
  }
}
const fresh = evaluate(best.v, 100, 20);
fs.writeFileSync(out, JSON.stringify({ stage, best: Array.from(best.v), params: decode(best.v), score: best.score, pass: best.pass, caught: best.caught, held, freshPass: fresh.pass, freshCaught: fresh.caught, freshN: 20 }));
console.log('done fresh pass', fresh.pass + '/20', 'caught', fresh.caught + '/20', '->', out);
