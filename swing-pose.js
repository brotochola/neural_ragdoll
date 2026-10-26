// Balancing on the hanging swing with the pose executor (src/pose.js): feet stay where they are on the
// plank, the pelvis stays above the plank's centre, and the balance feedback (hips and ankles) keeps
// the centre of mass over that moving point. No network, no external forces.
//   node swing-pose.js --n 10 [--push 0.6] [--record out.json]
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
const n = +arg('--n', 10);
const push = arg('--push', null);
const recordFile = arg('--record', null);
const kHip = +arg('--khip', 0.8);
const kAnkle = +arg('--kankle', 1.5);
const kVel = +arg('--kvel', 0.3);
let gains = { kHip, kAnkle, kVel, crouch: +arg('--crouch', 0), lean: +arg('--lean', 0), kAcc: +arg('--kacc', 0), torsoGain: +arg('--torso', 0) };
const gainsFile = arg('--gains', null);
if (gainsFile) gains = Object.assign(gains, JSON.parse(fs.readFileSync(gainsFile, 'utf8')).gains);

const Box2D = await import(pathToFileURL(path.join(root, 'vendor/PhaserBox2D.min.js')).href);
const { createSim } = await import('./src/sim.js');
const { ElmanNetwork } = await import('./src/net.js');
const { makeSeededRandom } = await import('./src/utils.js');
const { evalMode } = await import('./src/training.js');
const { isStanding } = await import('./src/reward.js');
const pose = await import('./src/pose.js');

setSkill(12);
const sim = createSim(Box2D);
const world = sim.build();
const weights = ElmanNetwork.copy(JSON.parse(fs.readFileSync(path.join(root, 'trained/stand-and-walk.json'), 'utf8')).theta);
const P = pose.PARTS();
const shapes = world.parts.map((q) => ({ hx: +q.hx.toFixed(4), hy: +q.hy.toFixed(4), r: q.r ? +q.r.toFixed(4) : 0, ax: q.ax || 0, ay: q.ay || 0, capsule: !!q.capsule, circle: !!q.circle }));
function runAll(g, count, seedBase, record, quiet) {
const episodes = [];
let ok = 0;
let score = 0;
for (let i = 0; i < count; i++) {
  const rand = makeSeededRandom(seedBase + i * 17);
  const start = sim.fillTrainingStart({}, rand, evalMode(12));
  if (push != null) start.swing.push = +push * Math.sign(start.swing.push || 1);
  world.rebuild();
  const ep = new sim.Episode(world, weights, start.tilt || 0, start.drop || 0, Object.assign({ phase: i % 6 }, start));
  let feet = null;
  let lastV = 0;
  const orig = ep.applyMotors.bind(ep);
  ep.applyMotors = function () {
    const plank = Box2D.b2Body_GetPosition(this.swing.plank);
    const vel = Box2D.b2Body_GetLinearVelocity(this.swing.plank);
    const acc = (vel.x - lastV) * config.physics.hz;
    lastV = vel.x;
    if (!feet) feet = P.feet.map((f) => { const s = pose.soleOf(world, f); return { dx: s.x - plank.x, y: s.y - plank.y }; });
    // Feet ride the plank; the pelvis sits above the plank centre.
    const goal = {
      // Lean against the plank's acceleration (feed-forward), knees bent by g.crouch.
      pelvis: { x: plank.x - g.kAcc * acc, y: plank.y + 0.03 + 0.92 - g.crouch }, torso: g.lean - 0.3 * g.kAcc * acc, torsoGain: g.torsoGain,
      feet: feet.map((f) => ({ x: plank.x + f.dx, y: plank.y + f.y, angle: 0 })),
      planted: [true, true],
    };
    // Balance relative to the plank: its speed is not the body's error.
    const m = { centerX: this.measure.centerX, centerVelocityX: (this.measure.centerVelocityX || 0) - vel.x };
    pose.commandsFor(world, pose.balancedTargets(goal, m, { planted: [true, true], support: plank.x - g.kAcc * acc, kHip: g.kHip, kAnkle: g.kAnkle, kAnkleVel: g.kVel, kHipVel: g.kVel * 0.6 }, pose.actualOf(world)), this.action);
    orig();
  };
  const frames = [];
  let f = 0;
  for (;;) {
    if (record && f % 2 === 0) {
      const pz = [];
      for (let k = 0; k < world.parts.length; k++) pz.push(+world.px[k].toFixed(3), +world.py[k].toFixed(3), +Math.atan2(world.s[k], world.c[k]).toFixed(3));
      const t = Box2D.b2Body_GetTransform(ep.swing.plank);
      frames.push({ p: pz, up: isStanding(ep.measure) ? 1 : 0, sw: [+t.p.x.toFixed(3), +t.p.y.toFixed(3), +Math.atan2(t.q.s, t.q.c).toFixed(3)] });
    }
    if (args.includes('--debug') && f % 60 === 30 && i === 0) {
      const m = ep.measure; const t = []; for (let k = 0; k < m.touch.length; k++) if (m.touch[k]) t.push(k);
      console.log('  t', (f / 60).toFixed(1), 'touch', t.join(','), 'cos', m.torsoCos.toFixed(2), 'head', m.headHeight.toFixed(2), 'stance', m.stanceWidth.toFixed(2), 'spin', m.torsoSpin.toFixed(2), 'v', m.centerVelocityX.toFixed(2), 'lim', m.standSpeedLimit, 'standing', isStanding(m));
    }
    if (ep.advanceOneFrame()) break;
    f++;
  }
  const s = ep.stats;
  const wf = s.windowFrames || 0;
  const share = wf ? s.windowStood / wf : 0;
  const pass = s.stoodUp && !s.headHit && s.onSwing && share >= config.swing.standShare;
  if (pass) ok++;
  score += (pass ? 3 : 0) + share + Math.min(1, ep.measure.headHeight / 1.8) + (s.onSwing ? 0.5 : 0);
  if (!quiet) console.log('episode', i, 'push', start.swing.push.toFixed(2), 'pass', pass ? 1 : 0, 'stood', share.toFixed(2), 'onSwing', s.onSwing, 'head', ep.measure.headHeight.toFixed(2));
  episodes.push({ seed: i, swing: { x: +ep.swing.x.toFixed(3), half: config.swing.halfLength, thick: config.swing.halfThick, inset: config.swing.ropeInset, barY: config.swing.top + config.swing.ropeLength }, box: null, board: null, walk: null, fitness: 0, stoodUp: pass, standShare: +share.toFixed(2), walkDist: 0, walkCmd: 0, frames });
}
return { ok, score: score / count, episodes };
}

if (args.includes('--search')) {
  // Cross-entropy search over the balance numbers; held-out check on the official starts.
  const { randomNormalSample } = await import('./src/utils.js');
  const rnd = makeSeededRandom(5);
  const names = ['kHip', 'kAnkle', 'kVel', 'crouch', 'lean', 'kAcc', 'torsoGain'];
  const spread = [0.6, 1.0, 0.15, 0.08, 0.1, 0.05, 0.4];
  let mean = names.map((k) => gains[k]);
  let sig = spread.slice();
  let best = null;
  for (let it = 0; it < +arg('--iters', 25); it++) {
    const cands = [];
    for (let c = 0; c < 14; c++) {
      const v = mean.map((m, d) => m + sig[d] * randomNormalSample(rnd));
      const g = Object.fromEntries(names.map((k, d) => [k, v[d]]));
      g.kVel = Math.max(0, g.kVel); g.crouch = Math.max(0, g.crouch);
      cands.push({ v, g, ...runAll(g, 4, 900000 + it * 101, false, true) });
    }
    cands.sort((a, b) => b.score - a.score);
    const top = cands.slice(0, 4);
    mean = mean.map((m, d) => 0.7 * top.reduce((a, c) => a + c.v[d], 0) / 4 + 0.3 * m);
    sig = sig.map((sv, d) => Math.max(spread[d] * 0.15, 0.7 * Math.sqrt(top.reduce((a, c) => a + (c.v[d] - top.reduce((x, y) => x + y.v[d], 0) / 4) ** 2, 0) / 4) + 0.3 * sv));
    const val = runAll(cands[0].g, 10, config.eval.seed + 12 * 10007, false, true);
    if (!best || val.score > best.score) { best = { g: cands[0].g, ...val }; fs.writeFileSync(path.resolve(root, arg('--out', 'trained/swing-gains.json')), JSON.stringify({ gains: best.g, ok: best.ok })); }
    console.log('iter', it, 'held-out', val.ok + '/10', 'best', best.ok + '/10', JSON.stringify(Object.fromEntries(Object.entries(best.g).map(([k, x]) => [k, +x.toFixed(3)]))));
  }
  process.exit(0);
}
const res = runAll(gains, n, config.eval.seed + 12 * 10007, !!recordFile, false);
console.log('swing passed', res.ok + '/' + n);
if (recordFile) fs.writeFileSync(recordFile, JSON.stringify({ skill: 12, fps: config.physics.hz / 2, shapes, episodes: res.episodes }));
sim.destroyWorld(world);
