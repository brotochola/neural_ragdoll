// THE CRAG: climbing an irregular rock face (src/terrain.js makeCrag: near-vertical tiers of different
// heights, each with a narrow ledge of its own depth) with hands and feet, by the crag planner
// (src/crag-planner.js, the wall planner generalised to ledges at their own heights and distances) and
// the pose executor. It picks the next ledge for each limb, three points stay on the rock, hands grip
// the lips, feet stand on the ledges, then over the summit edge. No external forces.
//   node crag.js --iters 30 --out trained/crag.json
//   node crag.js --show --init trained/crag.json --record out.json
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
const iters = +arg('--iters', 30);
const out = arg('--out', 'trained/crag.json');
const init = arg('--init', null);
const pop = +arg('--pop', 24);
const elite = +arg('--elite', 6);
const seeds = +arg('--seeds', 3);
const seconds = +arg('--seconds', 25);
const cragH = +arg('--height', 2.6); // summit height
const tierMin = +arg('--tier-min', 0.22), tierMax = +arg('--tier-max', 0.36);
const spacing = 0;

const Box2D = await import(pathToFileURL(path.join(root, 'vendor/PhaserBox2D.min.js')).href);
const { createSim } = await import('./src/sim.js');
const { ElmanNetwork } = await import('./src/net.js');
const { makeSeededRandom, randomNormalSample } = await import('./src/utils.js');
const { evalMode } = await import('./src/training.js');
const pose = await import('./src/pose.js');
const { installCragPlanner } = await import('./src/crag-planner.js');
const { makeCrag } = await import('./src/terrain.js');

setSkill(6);
const sim = createSim(Box2D);
const world = sim.build();
const HZ = config.physics.hz;
const J = config.net.jointCount;
const OUT = J + config.net.gripOutputs;
const P = config.partIndex;
const weights = ElmanNetwork.copy(JSON.parse(fs.readFileSync(path.join(root, 'trained/stand-and-walk.json'), 'utf8')).theta);

// [name, start, spread]
const SPEC = [
  // Geometry (2D: the knees cannot go sideways round the wall, and leaning in costs overhead reach since
  // the shoulder bends 2.5 rad at most): hips about 0.4 m out, trunk upright, legs fairly straight.
  ['pelvisOff', 0.56, 0.08], // pelvis this far from the wall face
  ['pelvisAbove', 0.84, 0.08], // pelvis this high above the higher foot
  ['armDown', 0.35, 0.1], // and at least this far below the lower hand
  ['lean', 0.0, 0.12], // torso world angle (negative leans in toward the wall)
  ['spanMax', 1.7, 0.1], // a hand moves up while hands-to-feet stays within this
  ['spanMin', 1.3, 0.1], // a foot moves up while hands-to-feet stays above this
  ['tHand', 0.6, 0.2], // seconds for a hand move
  ['tFoot', 0.7, 0.2], // seconds for a foot move
  ['tRise', 0.6, 0.2], // seconds to push the pelvis up
  ['clear', 0.1, 0.05], // a moving limb arcs this far out from the wall
  ['torsoGain', 0.5, 0.4], // supporting legs turn the trunk back to its lean
  ['footIn', 0.0, 0.04], // foot target nudged into the wall (+) or out (-)
  ['handUp', 0.03, 0.03], // hand target above the rung centre
  ['topHand', 0.03, 0.04], // hands on the top surface this far past the edge
  ['mantleLean', -0.8, 0.25], // trunk lean as the hips come over the edge
  ['mantleUp', 0.55, 0.12], // hips above the top with one foot on it
  ['spanTop', 1.2, 0.12], // hands go onto the top once the feet are within this of it
  ['spanLow', 0.8, 0.1], // otherwise a foot may still come up to this far below the hands
  ['pelvisOffTop', 0.36, 0.08], // hips this far from the wall once a hand is at the top
  ['handIn', 0.0, 0.03], // hands nudged in toward the rock (+) or out (-) on a knob
  ['chestUp', 0.35, 0.15], // the hips keep their distance from the rock at this height above them
  ['footBelow', 0.75, 0.12], // while a foot goes up, the hips hang at most this far above its knob
  ['retryPelX', -0.04, 0.04], // each retry of a foot move: hips this much closer (+) or farther (-)
  ['retryPelY', 0.06, 0.04], // and this much lower
];
const decode = (v) => {
  const p = {};
  SPEC.forEach(([k, b], i) => (p[k] = b + (v[i] || 0)));
  for (const k of ['tHand', 'tFoot', 'tRise']) p[k] = Math.max(0.2, p[k]);
  for (const k of Object.keys(process.env)) if (k.startsWith('P_')) p[k.slice(2)] = +process.env[k];
  return p;
};

const smooth = (s) => s * s * (3 - 2 * s);

function episode(p, seed, record) {
  const r = makeSeededRandom(9000 + seed);
  const start = sim.fillTrainingStart({}, r, evalMode(6));
  start.tilt = 0;
  start.walk = null;
  start.frames = Math.round(seconds * HZ);
  const crag = makeCrag(r, cragH, +(process.env.WALL_DX || 0.55) + r() * 0.08, { tierMin, tierMax });
  start.terrain = { statics: [], polys: crag.polys, surface: crag.surface };
  world.rebuild();
  const ep = new sim.Episode(world, weights, 0, 0, Object.assign({ phase: 0 }, start));
  const PL = installCragPlanner(ep, world, p, { config, pose, Box2D, crag });
  const { lim, feetTop, handsTop, rungs, wallX, wallH } = PL;
  let best = 0;
  let maxFoot = 0;
  let fell = false;
  const recFrames = record ? [] : null;

  let frames = 0;
  let upTop = 0;
  let topped = false;
  while (!ep.advanceOneFrame()) {
    frames++;
    const m = ep.measure;
    const pel = world.py[0];
    const footLow = Math.min(world.py[P.footLeft], world.py[P.footRight]);
    if (record && frames % 2 === 0) {
      const pz = [];
      for (let k = 0; k < world.parts.length; k++) pz.push(+world.px[k].toFixed(3), +world.py[k].toFixed(3), +Math.atan2(world.s[k], world.c[k]).toFixed(3));
      recFrames.push({ p: pz, up: m.torsoCos > 0.7 ? 1 : 0, gl: ep.grips[0].joint ? 1 : 0, gr: ep.grips[1].joint ? 1 : 0 });
    }
    // fell: head below the hips, or a body length below its highest point with no grip
    const gripped = ep.grips[0].joint || ep.grips[1].joint;
    best = Math.max(best, footLow);
    if (m.headHeight < world.py[0] || (!gripped && footLow < best - 0.5 && best > 0.3)) { fell = true; break; }
    maxFoot = Math.max(maxFoot, Math.min(lim.foot[0].rung, lim.foot[1].rung));
    // on top: both feet up there and standing for a second
    const onTopNow = feetTop() === 2 && footLow > wallH - 0.05 && m.headHeight > wallH + 1.3 && m.torsoCos > 0.8;
    upTop = onTopNow ? upTop + 1 / HZ : 0;
    if (upTop > 1.0) { topped = true; break; }
  }
  const top = rungs.length;
  const reached = Math.min(lim.foot[0].rung, lim.foot[1].rung);
  const T = frames / HZ;
  const score = 3 * Math.min(best, wallH) / wallH * 3 + (fell ? -1 : 1) + (handsTop() ? 1 : 0) + feetTop() * 1.5 + (topped ? 5 + 2 * (1 - T / seconds) : 0);
  const why = (topped ? 'TOPPED ' + T.toFixed(1) + 's ' : '') + 'feet rung ' + reached + '/' + top + ' best foot height ' + best.toFixed(2) + ' fell ' + (fell ? T.toFixed(1) + 's' : '-') + ' hands ' + lim.hand[0].rung + '/' + lim.hand[1].rung + ' feet on top ' + feetTop();
  const status = topped ? 'climbed onto the top (' + wallH.toFixed(1) + ' m) in ' + T.toFixed(0) + ' s' : (fell ? 'fell' : 'held on') + ' · feet up to ' + best.toFixed(2) + ' m';
  return { score, pass: topped ? 1 : 0, why, status, frames: recFrames, surface: ep.terrain.surface.map(([x, y]) => [+x.toFixed(3), +y.toFixed(3)]), rungs: rungs.map((h) => ({ x0: +h.lip.toFixed(3), x1: +h.back.toFixed(3), y: +(h.y - 0.02).toFixed(3) })) };
}

function evaluate(v, base, n) {
  const p = decode(v);
  let s = 0, pass = 0;
  for (let i = 0; i < n; i++) { const r = episode(p, base + i); s += r.score; pass += r.pass; }
  return { score: s / n, pass };
}

if (args.includes('--show')) {
  const v = init ? JSON.parse(fs.readFileSync(init, 'utf8')).best : [];
  const p = decode(v);
  for (let i = 0; i < seeds; i++) { const r = episode(p, 100 + i); console.log('seed', i, r.why); }
  const rec = arg('--record', null);
  if (rec) {
    const shapes = world.parts.map((q) => ({ hx: +q.hx.toFixed(4), hy: +q.hy.toFixed(4), r: q.r ? +q.r.toFixed(4) : 0, ax: q.ax || 0, ay: q.ay || 0, capsule: !!q.capsule, circle: !!q.circle }));
    const eps = [];
    for (let i = 0; i < +arg('--n', 2); i++) {
      const r = episode(p, 100 + i, true);
      eps.push({ seed: i, surface: r.surface, holds: r.rungs, box: null, walk: null, board: null, status: r.status, frames: r.frames });
    }
    fs.writeFileSync(rec, JSON.stringify({ skill: 6, fps: HZ / 2, shapes, episodes: eps }));
  }
  process.exit(0);
}

const rand = makeSeededRandom(+arg('--rng', 777));
const mean = new Float64Array(SPEC.length);
if (init) { const sv = JSON.parse(fs.readFileSync(init, 'utf8')).best; for (let d = 0; d < SPEC.length && d < sv.length; d++) mean[d] = sv[d]; }
const sigma = Float64Array.from(SPEC.map((q) => q[2]));
let best = null;
for (let it = 0; it < iters; it++) {
  const cands = [];
  for (let c = 0; c < pop; c++) {
    const v = Float64Array.from(mean, (m, d) => m + sigma[d] * randomNormalSample(rand));
    cands.push({ v, ...evaluate(v, 1000 + it * seeds, seeds) });
  }
  cands.sort((a, b) => b.score - a.score);
  if (it % 3 === 2 || it === 0) {
    for (const c of [cands[0], { v: Float64Array.from(mean) }]) {
      const val = evaluate(c.v, 500, 6);
      if (!best || val.score > best.score) best = { v: Float64Array.from(c.v), ...val };
    }
    fs.writeFileSync(out, JSON.stringify({ cragH, tierMin, tierMax, best: Array.from(best.v), params: decode(best.v), score: best.score, pass: best.pass }));
    console.log('iter', it, 'held-out best', best.score.toFixed(3), 'pass', best.pass + '/6', 'gen best', cands[0].score.toFixed(3));
  }
  const top = cands.slice(0, elite);
  for (let d = 0; d < SPEC.length; d++) {
    const m = top.reduce((a, c) => a + c.v[d], 0) / elite;
    const sd = Math.sqrt(top.reduce((a, c) => a + (c.v[d] - m) ** 2, 0) / elite);
    mean[d] = 0.7 * m + 0.3 * mean[d];
    sigma[d] = Math.max(SPEC[d][2] * 0.15, 0.7 * sd + 0.3 * sigma[d]);
  }
}
console.log('done ->', out);
