// Runs the two-level climb (src/planner.js -> src/pose.js) on box starts and reports what happened.
//   node plan-climb.js --height 0.9 --n 5 [--record out.json] [--params p.json]
// After the plan ends, the stand-and-walk network balances on top.
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
const height = +arg('--height', 0.9);
const n = +arg('--n', 5);
const first = +arg('--first', 0);
const recordFile = arg('--record', null);
const paramsFile = arg('--params', null);
const debug = args.includes('--debug');

const Box2D = await import(pathToFileURL(path.join(root, 'vendor/PhaserBox2D.min.js')).href);
const { createSim } = await import('./src/sim.js');
const { ElmanNetwork, outputCountOf } = await import('./src/net.js');
const { makeSeededRandom } = await import('./src/utils.js');
const { evalMode } = await import('./src/training.js');
const { isStanding } = await import('./src/reward.js');
const pose = await import('./src/pose.js');
const { climbPlan, makePlanRunner } = await import('./src/planner.js');

setSkill(11);
const sim = createSim(Box2D);
const world = sim.build();
const weights = ElmanNetwork.copy(JSON.parse(fs.readFileSync(path.join(root, 'trained/stand-and-walk.json'), 'utf8')).theta);
const J = config.net.jointCount;
const OUT = outputCountOf();
const params = paramsFile ? JSON.parse(fs.readFileSync(paramsFile, 'utf8')).params || {} : {};

export function runClimb(seed, p, record) {
  const r = makeSeededRandom(2000 + seed);
  const start = sim.fillTrainingStart({}, r, evalMode(11));
  start.climb.side = 1;
  start.climb.height = height;
  start.frames = 14 * config.physics.hz;
  world.rebuild();
  const ep = new sim.Episode(world, weights, start.tilt || 0, start.drop || 0, Object.assign({ phase: 0 }, start));
  let runner = null;
  let goal = null;
  let lastGoal = null;
  let planDone = false;
  const origGrip = ep.updateGrip.bind(ep);
  ep.updateGrip = function () {
    if (!runner && this.climbBox) runner = makePlanRunner(climbPlan(height, this.climbBox.x0, p), world, config.physics.hz);
    const before = runner ? runner.key : -1;
    goal = !planDone && runner ? runner.step() : null;
    if (debug && runner && runner.key !== before && before >= 0 && lastGoal) {
      const pv = pose.pelvisOf(world);
      const P = pose.PARTS();
      const f = P.feet.map((q) => pose.soleOf(world, q));
      const hd = P.hands.map((q) => pose.handOf(world, q));
      const fmt = (a) => a.x.toFixed(2) + ',' + a.y.toFixed(2);
      console.log('  end key', before, 'pelvis', fmt(pv), 'goal', fmt(lastGoal.pelvis), 'torso', pv.angle.toFixed(2), 'goal', lastGoal.torso.toFixed(2),
        '| feet', fmt(f[0]), fmt(f[1]), 'goal', fmt(lastGoal.feet[0]), fmt(lastGoal.feet[1]),
        '| hands', fmt(hd[0]), fmt(hd[1]), 'goal', lastGoal.hands[0] ? fmt(lastGoal.hands[0]) : '-', lastGoal.hands[1] ? fmt(lastGoal.hands[1]) : '-',
        '| grips', this.grips.map((g) => (g.joint ? 1 : 0)).join(''), 'box edge', this.climbBox.x0.toFixed(2));
    }
    if (goal) lastGoal = goal;
    if (!goal) planDone = true;
    if (goal) {
      const anyDown = goal.planted[0] || goal.planted[1];
      const t = pose.balancedTargets(goal, this.measure, anyDown ? { planted: goal.planted } : { planted: goal.planted, kHip: 0, kHipVel: 0 });
      pose.commandsFor(world, t, this.action);
      for (let j = J; j < OUT; j++) this.action[j] = goal.grip ? 1 : -1;
    } else for (let j = J; j < OUT; j++) this.action[j] = -1;
    origGrip();
  };
  const frames = [];
  let f = 0;
  let tipped = 0;
  for (;;) {
    if (record && f % 2 === 0) {
      const pz = [];
      for (let i = 0; i < world.parts.length; i++) pz.push(+world.px[i].toFixed(3), +world.py[i].toFixed(3), +Math.atan2(world.s[i], world.c[i]).toFixed(3));
      frames.push({ p: pz, up: isStanding(ep.measure) ? 1 : 0 });
    }
    if (ep.measure.torsoCos < 0) tipped = 1;
    if (ep.advanceOneFrame()) break;
    f++;
  }
  const s = ep.stats;
  const wf = s.windowFrames || 0;
  const standShare = wf ? s.windowStood / wf : 0;
  const onTop = !!s.climbOnTop;
  const headOver = ep.measure.headHeight - height;
  const pass = onTop && standShare >= config.highbox.standShare && headOver >= config.body.standingHeadMeters;
  const box = ep.climbBox ? { x0: +ep.climbBox.x0.toFixed(3), x1: +ep.climbBox.x1.toFixed(3), top: +ep.climbBox.top.toFixed(3) } : null;
  return { pass, onTop, standShare, headOver, tipped, key: runner ? runner.key : -1, box, frames, stats: s };
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  const shapes = world.parts.map((q) => ({ hx: +q.hx.toFixed(4), hy: +q.hy.toFixed(4), r: q.r ? +q.r.toFixed(4) : 0, ax: q.ax || 0, ay: q.ay || 0, capsule: !!q.capsule, circle: !!q.circle }));
  const episodes = [];
  let ok = 0;
  for (let i = 0; i < n; i++) {
    const res = runClimb(first + i, params, !!recordFile);
    if (res.pass) ok++;
    console.log('seed', first + i, 'pass', res.pass ? 1 : 0, 'onTop', res.onTop ? 1 : 0, 'stood', res.standShare.toFixed(2), 'head over box', res.headOver.toFixed(2), 'tipped', res.tipped, 'last key', res.key);
    episodes.push({ seed: first + i, box: res.box, onTop: res.onTop, walk: null, board: null, fitness: 0, stoodUp: res.pass, standShare: +res.standShare.toFixed(2), walkDist: 0, walkCmd: 0, frames: res.frames });
  }
  console.log('height', height, 'passed', ok + '/' + n);
  if (recordFile) fs.writeFileSync(recordFile, JSON.stringify({ skill: 11, fps: config.physics.hz / 2, shapes, episodes }));
  sim.destroyWorld(world);
}
