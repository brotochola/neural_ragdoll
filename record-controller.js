// Records episodes driven by one of the hand-structured controllers (the teachers the expert
// networks copy), in the replay format of record-episodes.js.
//   node record-controller.js --walk trained/walk-controller.json --n 4 --out rec-walk.json
//   node record-controller.js --stepup trained/stepup-r-03-final.json --height 0.3 --n 4 --out rec-climb.json
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
const walkFile = arg('--walk', null);
const stepFile = arg('--stepup', null);
const n = +arg('--n', 4);
const out = arg('--out', 'rec-controller.json');
const height = +arg('--height', 0.3);
const side = +arg('--side', 1);
const first = +arg('--first', 100);
const speedMin = +arg('--speed-min', 0);
const speedMax = +arg('--speed-max', 0);
const bumps = +arg('--bumps', 0); // uneven ground from 1 m to 7 m ahead, blocks up to this tall

const Box2D = await import(pathToFileURL(path.join(root, 'vendor/PhaserBox2D.min.js')).href);
const { createSim } = await import('./src/sim.js');
const { ElmanNetwork, outputCountOf } = await import('./src/net.js');
const { makeSeededRandom } = await import('./src/utils.js');
const { evalMode } = await import('./src/training.js');
const { isStanding } = await import('./src/reward.js');

const skill = walkFile ? 6 : 11;
setSkill(skill);
const sim = createSim(Box2D);
const world = sim.build();
const weights = ElmanNetwork.copy(JSON.parse(fs.readFileSync(path.join(root, 'trained/stand-and-walk.json'), 'utf8')).theta);
const J = config.net.jointCount;
const OUT = outputCountOf();
let drive;
if (speedMax > 0) config.walk.maxStandSpeed = Math.max(config.walk.maxStandSpeed, speedMax * 1.6);
if (walkFile) {
  const { decode, makeController } = await import('./src/simbicon.js');
  const saved = JSON.parse(fs.readFileSync(path.resolve(root, walkFile), 'utf8'));
  const p = Object.assign(decode(saved.best), { track: !!saved.track });
  const c = makeController(world);
  drive = (ep, start) => {
    const delay = Math.round((start.walk.delay || 0) * config.physics.hz);
    const orig = ep.applyMotors.bind(ep);
    ep.applyMotors = function () {
      if (this.frame >= delay) c(p, this, this.action);
      orig();
    };
  };
} else {
  const { decode, makeController } = await import('./src/stepup.js');
  const savedStep = JSON.parse(fs.readFileSync(path.resolve(root, stepFile), 'utf8'));
  const p = Object.assign(decode(savedStep.best), { hands: !!savedStep.hands });
  const c = makeController(world);
  drive = (ep) => {
    let done = false;
    const origGrip = ep.updateGrip.bind(ep);
    ep.updateGrip = function () {
      if (!done && this.frame / config.physics.hz < 9) done = !c(p, this, this.action);
      else done = true;
      if (done) for (let j = J; j < OUT; j++) this.action[j] = -1;
      origGrip();
    };
  };
}
const shapes = world.parts.map((q) => ({
  hx: +q.hx.toFixed(4), hy: +q.hy.toFixed(4), r: q.r ? +q.r.toFixed(4) : 0,
  ax: q.ax || 0, ay: q.ay || 0, capsule: !!q.capsule, circle: !!q.circle,
}));
const episodes = [];
for (let e = 0; e < n; e++) {
  const seed = first + e;
  const r = makeSeededRandom((walkFile ? 3000 : 2000) + seed);
  const start = sim.fillTrainingStart({}, r, evalMode(skill));
  if (start.walk && speedMax > 0) start.walk.speed = speedMin + r() * (speedMax - speedMin);
  if (bumps > 0) {
    const statics = [];
    let x = 1.0;
    while (x < 7) {
      const w = 0.2 + r() * 0.4;
      statics.push({ dx: x, w, h: 0.01 + r() * (bumps - 0.01) });
      x += w + 0.1 + r() * 0.5;
    }
    start.terrain = { statics };
  }
  if (start.climb) {
    start.climb.side = side;
    start.climb.height = height;
    start.frames = 12 * config.physics.hz;
  }
  world.rebuild();
  const ep = new sim.Episode(world, weights, start.tilt || 0, start.drop || 0, Object.assign({ phase: seed % 6 }, start));
  drive(ep, start);
  const frames = [];
  let f = 0;
  for (;;) {
    if (f % 2 === 0) {
      const pose = [];
      for (let k = 0; k < world.parts.length; k++) pose.push(+world.px[k].toFixed(3), +world.py[k].toFixed(3), +Math.atan2(world.s[k], world.c[k]).toFixed(3));
      const head = ep.options.targets ? ep.options.targets[0] : null;
      frames.push({ p: pose, up: isStanding(ep.measure) ? 1 : 0, mx: head && head.on ? +(+head.x).toFixed(3) : null, my: head && head.on ? +(+head.y).toFixed(3) : null });
    }
    if (ep.advanceOneFrame()) break;
    f++;
  }
  const s = ep.stats;
  episodes.push({
    seed, walk: start.walk || null,
    boxes: ep.terrain ? ep.terrain.statics.map((b) => ({ x0: +b.x0.toFixed(3), x1: +b.x1.toFixed(3), top: +b.top.toFixed(3) })) : undefined,
    box: ep.climbBox ? { x0: +ep.climbBox.x0.toFixed(3), x1: +ep.climbBox.x1.toFixed(3), top: +ep.climbBox.top.toFixed(3) } : null,
    onTop: !!s.climbOnTop, board: null, fitness: +s.fitness.toFixed(3), stoodUp: !!s.stoodUp,
    standShare: s.windowFrames ? +(s.windowStood / s.windowFrames).toFixed(2) : 0,
    walkDist: +(s.walkDist || 0).toFixed(2), walkCmd: +(s.walkCmd || 0).toFixed(2), frames,
  });
  console.log('episode', seed, 'stood', episodes[e].standShare, start.walk ? 'walked ' + episodes[e].walkDist + '/' + episodes[e].walkCmd : 'on top ' + episodes[e].onTop);
}
sim.destroyWorld(world);
fs.writeFileSync(out, JSON.stringify({ skill, fps: config.physics.hz / 2, shapes, episodes }));
console.log('wrote', out);
