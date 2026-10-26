// Records held-out episodes of a trained net as frame-by-frame part poses, for the replay page.
//   node record-episodes.js --weights backups/best-skill1.json --skill 1 --n 4 --out replay-stand.json
// The starts are the same seeds the headless eval uses, so what you watch is what was scored.
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
const weightsFile = arg('--weights', 'backups/best-skill1.json');
const skill = +arg('--skill', 1);
const n = +arg('--n', 4);
const out = arg('--out', 'replay.json');
const every = +arg('--every', 2); // Keep one frame in this many. 2 = 30 fps.
const seconds = +arg('--seconds', 0); // 0 = the eval length.
const firstSeed = +arg('--first', 0);
const boxHeight = +arg('--height', 0); // Climb: force this box height instead of the eval draw.
const boardAmp = arg('--amp', null); // Board: force this tilt amplitude (radians).
const assist = +arg('--assist', 0); // Get-up: harness strength (0 = none, like the eval).

const Box2D = await import(pathToFileURL(path.join(root, 'vendor/PhaserBox2D.min.js')).href);
const { createSim } = await import('./src/sim.js');
const { ElmanNetwork } = await import('./src/net.js');
const { makeSeededRandom } = await import('./src/utils.js');
const { isStanding } = await import('./src/reward.js');
const { evalMode } = await import('./src/training.js');

setSkill(skill);
const saved = JSON.parse(fs.readFileSync(weightsFile, 'utf8'));
const weights = ElmanNetwork.copy(saved.theta);
if (!weights) throw new Error(weightsFile + ': weight count does not match this net');
const sim = createSim(Box2D);
const world = sim.build();
const shapes = world.parts.map((p) => ({
  hx: +p.hx.toFixed(4), hy: +p.hy.toFixed(4), r: p.r ? +p.r.toFixed(4) : 0,
  ax: p.ax || 0, ay: p.ay || 0, capsule: !!p.capsule, circle: !!p.circle,
}));
const episodes = [];
for (let e = 0; e < n; e++) {
  const i = firstSeed + e;
  const rand = makeSeededRandom(config.eval.seed + skill * 10007 + i * 17);
  const mode = evalMode(skill);
  if (seconds > 0) mode.seconds = seconds;
  const start = sim.fillTrainingStart({}, rand, mode);
  if (boxHeight > 0 && start.climb) start.climb.height = boxHeight;
  if (boardAmp != null && start.board) start.board.amp = +boardAmp;
  if (assist > 0 && start.getup) start.assist = assist;
  world.rebuild();
  const ep = new sim.Episode(world, weights, start.tilt || 0, start.drop || 0, Object.assign({ phase: i % config.cadence.actionEveryNFrames }, start));
  const frames = [];
  let f = 0;
  let done = false;
  while (!done) {
    if (f % every === 0) {
      const pose = [];
      for (let p = 0; p < world.parts.length; p++) {
        pose.push(+world.px[p].toFixed(3), +world.py[p].toFixed(3), +Math.atan2(world.s[p], world.c[p]).toFixed(3));
      }
      const head = ep.options.targets ? ep.options.targets[0] : null;
      frames.push({ p: pose, up: isStanding(ep.measure) ? 1 : 0, mx: head && head.on ? +(+head.x).toFixed(3) : null, my: head && head.on ? +(+head.y).toFixed(3) : null, ba: ep.board ? +(ep.boardAngle || 0).toFixed(3) : undefined });
    }
    done = ep.advanceOneFrame();
    f++;
  }
  const s = ep.stats;
  episodes.push({
    seed: i,
    walk: start.walk || null,
    box: ep.climbBox ? { x0: +ep.climbBox.x0.toFixed(3), x1: +ep.climbBox.x1.toFixed(3), top: +ep.climbBox.top.toFixed(3) } : null,
    onTop: s.climbOnTop ? true : false,
    board: ep.board ? { x: +ep.boardX.toFixed(3), pivot: config.board.pivotHeight, half: config.board.halfLength, thick: config.board.halfThick, amp: start.board.amp, period: start.board.period } : null,
    fitness: +s.fitness.toFixed(3),
    stoodUp: !!s.stoodUp,
    standShare: s.windowFrames ? +(s.windowStood / s.windowFrames).toFixed(2) : 0,
    walkDist: +(s.walkDist || 0).toFixed(2),
    walkCmd: +(s.walkCmd || 0).toFixed(2),
    frames,
  });
  console.log('episode', i, 'fitness', s.fitness.toFixed(2), 'stoodUp', s.stoodUp, 'standShare', episodes[e].standShare, start.walk ? 'walked ' + episodes[e].walkDist + '/' + episodes[e].walkCmd + ' m' : '', ep.climbBox ? 'box ' + ep.climbBox.top.toFixed(2) + ' m, on top ' + !!s.climbOnTop : '');
}
sim.destroyWorld(world);
fs.writeFileSync(out, JSON.stringify({ skill, fps: config.physics.hz / every, shapes, episodes }));
console.log('wrote', out);
