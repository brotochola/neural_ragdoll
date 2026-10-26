// Tunes a STOP for the walking controller (src/simbicon.js) with the cross-entropy method: walk at
// 0.3 m/s for 3 s with the slow walker, switch to the stop parameters for 2 s, then hold still with the
// pose executor for 2 s. Scored by ending upright, still, on both feet, with a stance the climb accepts.
//   node stop-search.js --iters 40 --out trained/stop-controller.json
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
const iters = +arg('--iters', 40);
const out = arg('--out', 'trained/stop-controller.json');
const seeds = +arg('--seeds', 5);
const Box2D = await import(pathToFileURL(path.join(root, 'vendor/PhaserBox2D.min.js')).href);
const { createSim } = await import('./src/sim.js');
const { ElmanNetwork } = await import('./src/net.js');
const { makeSeededRandom, randomNormalSample } = await import('./src/utils.js');
const { evalMode } = await import('./src/training.js');
const simb = await import('./src/simbicon.js');
const pose = await import('./src/pose.js');

setSkill(6);
const sim = createSim(Box2D);
const world = sim.build();
const HZ = config.physics.hz;
const OUT = config.net.jointCount + config.net.gripOutputs;
const standW = ElmanNetwork.copy(JSON.parse(fs.readFileSync(path.join(root, 'trained/stand-and-walk.json'), 'utf8')).theta);
const slowSaved = JSON.parse(fs.readFileSync(path.join(root, 'trained/walk-controller.json'), 'utf8'));
const slowP = Object.assign(simb.decode(slowSaved.best), { track: true });
const ctl = simb.makeController(world);
// Stop parameters (offsets on the slow walker): swing targets, feedback gains, knee, step time.
const SPEC = [['swing0', 0.05, 0.15], ['swing1', 0, 0.15], ['cd', 0.8, 0.4], ['cv', 0.4, 0.3], ['kneeSwing0', -0.6, 0.3], ['T', 0.3, 0.1], ['holdKHip', 1.5, 0.6], ['holdKAnkle', 3, 1]];
const toP = (v) => {
  const p = Object.assign({}, slowP);
  SPEC.forEach(([k, base], i) => (p[k] = base + v[i]));
  p.T = Math.max(0.15, p.T);
  return p;
};

function episode(v, seed) {
  const p = toP(v);
  const r = makeSeededRandom(6000 + seed);
  const start = sim.fillTrainingStart({}, r, evalMode(6));
  start.tilt = 0;
  start.walk.speed = 0.3;
  start.frames = 7.5 * HZ;
  world.rebuild();
  const ep = new sim.Episode(world, standW, 0, start.drop || 0, Object.assign({ phase: 0 }, start));
  const tmp = new Float64Array(OUT);
  let hold = null;
  const switchAt = (2.6 + r() * 0.6) * HZ; // stop at a random moment of the stride
  const orig = ep.applyMotors.bind(ep);
  ep.applyMotors = function () {
    const m = this.measure;
    const f = this.frame;
    if (f < 0.5 * HZ) return orig();
    if (f < switchAt) { m.walkSpeed = 0.3; m.walkDir = 1; ctl(slowP, this, tmp); }
    else if (f < switchAt + 2 * HZ) { m.walkSpeed = 0; m.walkDir = 1; ctl(p, this, tmp); }
    else {
      if (!hold) hold = [config.partIndex.footLeft, config.partIndex.footRight].map((q) => pose.soleOf(world, q));
      const mid = (hold[0].x + hold[1].x) / 2;
      const goal = { pelvis: { x: mid, y: 0.92 }, torso: 0, feet: hold.map((q) => ({ x: q.x, y: q.y, angle: 0 })), planted: [true, true] };
      pose.commandsFor(world, pose.balancedTargets(goal, m, { planted: [true, true], kHip: p.holdKHip, kAnkle: p.holdKAnkle, kAnkleVel: 0.15, kHipVel: 0.1 }), tmp);
    }
    for (let j = 0; j < OUT; j++) this.action[j] = j < config.net.jointCount ? tmp[j] : -1;
    orig();
  };
  while (!ep.advanceOneFrame());
  const m = ep.measure;
  const up = m.headHeight > config.body.standingHeadMeters && m.torsoCos > 0.9;
  const still = Math.abs(m.centerVelocityX || 0) < 0.1;
  const both = m.touch[config.partIndex.footLeft] && m.touch[config.partIndex.footRight];
  const stance = Math.abs(world.px[config.partIndex.footRight] - world.px[config.partIndex.footLeft]);
  const pass = up && still && both && stance < 0.6;
  const score = (up ? 2 : 0) + (still ? 1 : 0) + (both ? 1 : 0) + Math.min(1, m.headHeight / 1.6) + (stance < 0.6 ? 1 : 0.6 / stance) + (pass ? 3 : 0);
  return { score, pass, stance };
}

function evaluate(v, base, n) {
  let s = 0;
  let pass = 0;
  for (let i = 0; i < n; i++) { const r = episode(v, base + i); s += r.score; pass += r.pass ? 1 : 0; }
  return { score: s / n, pass };
}

const rnd = makeSeededRandom(17);
const mean = new Float64Array(SPEC.length);
const sigma = Float64Array.from(SPEC.map((q) => q[2]));
let best = null;
for (let it = 0; it < iters; it++) {
  const cands = [];
  for (let c = 0; c < 16; c++) {
    const v = Array.from(mean, (m, d) => m + sigma[d] * randomNormalSample(rnd));
    cands.push({ v, ...evaluate(v, 1000 + it * seeds, seeds) });
  }
  cands.sort((a, b) => b.score - a.score);
  const top = cands.slice(0, 4);
  for (let d = 0; d < SPEC.length; d++) {
    const m = top.reduce((a, c) => a + c.v[d], 0) / 4;
    const sd = Math.sqrt(top.reduce((a, c) => a + (c.v[d] - m) ** 2, 0) / 4);
    mean[d] = 0.7 * m + 0.3 * mean[d];
    sigma[d] = Math.max(SPEC[d][2] * 0.15, 0.7 * sd + 0.3 * sigma[d]);
  }
  if (it % 4 === 3 || it === 0) {
    for (const c of [cands[0], { v: Array.from(mean) }]) {
      const val = evaluate(c.v, 500, 10);
      if (!best || val.score > best.score) {
        best = { v: c.v, ...val };
        fs.writeFileSync(path.resolve(root, out), JSON.stringify({ params: Object.fromEntries(SPEC.map(([k, b], i) => [k, b + best.v[i]])), pass: best.pass }));
      }
    }
    console.log('iter', it, 'held-out stops', best.pass + '/10', 'gen best', cands[0].score.toFixed(2));
  }
}
console.log('done ->', out);
