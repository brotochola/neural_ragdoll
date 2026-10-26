// THE COURSE: one long run over mixed terrain, driven by a SELECTOR (code) that looks at the ground
// ahead and picks a skill: walk (walking controller), climb (step-up controller), steady (the
// stand-and-walk network), get up (the get-up expert network, then the stand network).
//   node course.js --n 3 [--record out.json] [--seed 0]
// Layout, all forward (no drops): flat, uneven ground, a 0.3 m platform to climb, loose boxes on it,
// a short flight of stairs, and the finish line on the top landing.
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
const n = +arg('--n', 3);
const firstSeed = +arg('--seed', 0);
const recordFile = arg('--record', null);
const walkFile = arg('--walk', 'trained/walk-fast-controller.json');
const stairsFile = arg('--stairs-walk', null);
const stepFile = arg('--stepup', 'trained/stepup-r-03-final.json');
const seconds = +arg('--seconds', 70);
const verbose = args.includes('--verbose');

const Box2D = await import(pathToFileURL(path.join(root, 'vendor/PhaserBox2D.min.js')).href);
const { createSim } = await import('./src/sim.js');
const { ElmanNetwork, outputCountOf } = await import('./src/net.js');
const { makeSeededRandom } = await import('./src/utils.js');
const { evalMode } = await import('./src/training.js');
const { isStanding } = await import('./src/reward.js');
const simb = await import('./src/simbicon.js');
const step = await import('./src/stepup.js');

setSkill(6);
config.walk.maxStandSpeed = Math.max(config.walk.maxStandSpeed, 2.2);
const sim = createSim(Box2D);
const world = sim.build();
const J = config.net.jointCount;
const OUT = outputCountOf();
const HZ = config.physics.hz;
const load = (f) => JSON.parse(fs.readFileSync(path.resolve(root, f), 'utf8'));
const standW = ElmanNetwork.copy(load('trained/stand-and-walk.json').theta);
const getupSaved = load('trained/getup-expert-19.json');
const getupW = Float32Array.from(getupSaved.theta);
const walkSaved = load(walkFile);
const walkP = Object.assign(simb.decode(walkSaved.best), { track: !!walkSaved.track });
// The slow walker (0.2-0.45 m/s) for approaching a step and braking: the fast one barely slows down
// when asked (it reached the platform at 1.25 m/s with 0.3 requested).
const slowSaved = load(arg('--slow-walk', 'trained/walk-controller.json'));
const slowP = Object.assign(simb.decode(slowSaved.best), { track: !!slowSaved.track });
const stairsSaved = stairsFile ? load(stairsFile) : null;
const stairsP = stairsSaved ? Object.assign(simb.decode(stairsSaved.best), { track: !!stairsSaved.track }) : walkP;
const stepP = step.decode(load(stepFile).best);
const walkCtl = simb.makeController(world);
const stepCtl = step.makeController(world);

// The course, relative to the feet at the start (x forward). Heights in metres.
function layout(r) {
  const statics = [];
  // uneven ground 2-6 m
  let x = 2;
  while (x < 6) {
    const w = 0.25 + r() * 0.35;
    statics.push({ dx: x, w, h: 0.01 + r() * 0.03 });
    x += w + 0.15 + r() * 0.4;
  }
  // platform 0.3 m from 8 m to the end
  const platformX = 8;
  const platformTop = 0.3;
  const stairsX = 13;
  const rise = 0.1;
  const tread = 0.35;
  const nSteps = 3;
  const end = stairsX + nSteps * tread + 4;
  statics.push({ dx: platformX, w: end - platformX, h: platformTop });
  for (let i = 0; i < nSteps; i++) statics.push({ dx: stairsX + i * tread, w: end - (stairsX + i * tread), h: platformTop + (i + 1) * rise });
  // loose light boxes on the platform, 9.5-12 m
  const dynamics = [];
  for (let i = 0; i < 3; i++) dynamics.push({ dx: 9.5 + i * 0.8 + r() * 0.3, w: 0.15, h: 0.12, density: 8, y: platformTop });
  return { statics, dynamics, platformX, platformTop, stairsX, finish: stairsX + nSteps * tread + 1.5, end };
}

function run(seed, record) {
  const r = makeSeededRandom(4000 + seed);
  const start = sim.fillTrainingStart({}, r, evalMode(6));
  const L = layout(r);
  start.tilt = 0;
  start.walk.speed = 0.9;
  start.walk.delay = 0.5;
  start.terrain = { statics: L.statics, dynamics: L.dynamics };
  start.frames = Math.round(seconds * HZ);
  world.rebuild();
  const ep = new sim.Episode(world, Float32Array.from(standW), 0, start.drop || 0, Object.assign({ phase: 0 }, start));
  const stand = new ElmanNetwork(Float32Array.from(standW));
  let getup = null;
  const startX = (world.px[config.partIndex.footLeft] + world.px[config.partIndex.footRight]) / 2;
  const blocks = ep.terrain.statics;
  let mode = 'steady';
  let modeT = 0;
  let downT = 0;
  const log = [];
  const setMode = (m) => {
    if (m !== mode) {
      log.push(((ep.frame / HZ).toFixed(1)) + 's ' + m);
      if (verbose) console.log('  ', (ep.frame / HZ).toFixed(1), 's ->', m, 'x', ((world.px[0] - startX)).toFixed(2));
    }
    mode = m;
    modeT = 0;
    if (m === 'climb') ep.stepup = null;
    if (m === 'walk') ep.simbicon = null;
  };
  // The next block whose front edge is ahead of the feet and taller than the ground under them.
  const groundUnder = (x) => {
    let h = 0;
    for (const b of blocks) if (x > b.x0 && x < b.x1) h = Math.max(h, b.top);
    return h;
  };
  const nextRise = () => {
    // distances from the middle of the feet, as the step-up controller was tuned (0.25-0.45 m)
    const fx = (world.px[config.partIndex.footLeft] + world.px[config.partIndex.footRight]) / 2;
    const here = groundUnder(fx - 0.05);
    let best = null;
    for (const b of blocks) {
      if (b.x0 > fx - 0.02 && b.top - here > 0.06 && (!best || b.x0 < best.x0)) best = b;
    }
    return best ? { box: best, dist: best.x0 - fx, rise: best.top - here, here } : null;
  };
  const tmp = new Float64Array(OUT);
  const walkSpec = ep.options.walk;
  const origDecide = ep.decide.bind(ep);
  ep.decide = function () {
    origDecide();
    stand.forward(this.input);
    if (getup) getup.forward(this.input);
  };
  // While the stand net drives (steady, get-up handover), its marks sit on the body where it is: the
  // walk marks slide forward at the walk speed, and the stand net walked a metre chasing them.
  const pinMarks = (e) => {
    const t = e.options.targets;
    if (!t) return;
    const pelvisX = world.px[0];
    const fl = world.px[config.partIndex.footLeft];
    const fr = world.px[config.partIndex.footRight];
    const ground = groundUnder(pelvisX);
    for (let i = 0; i < config.targets.length; i++) {
      const spec = config.targets[i];
      const mk = t[i];
      if (!mk) continue;
      if (spec.ride) {
        mk.walkBase = pelvisX + (spec.x || 0) - e.walkOffset();
        mk.x = pelvisX + (spec.x || 0);
        mk.y = ground + (spec.y || 0);
      } else {
        mk.x = spec.id === 'footR' ? fr : fl;
        mk.y = ground + config.skill.footGroundMeters;
      }
    }
  };
  const origGrip = ep.updateGrip.bind(ep);
  ep.updateGrip = function () {
    modeT += 1 / HZ;
    // The stand-and-walk net reads the walk command as an input: zero it unless walking, or it walks.
    // The gait clock is an input too: outside walking the stand net gets no walk command at all,
    // exactly as in its standing training.
    if (mode === 'walk' || mode === 'brake') { if (!this.options.walk) this.options.walk = walkSpec; this.options.walk.speed = start.walk.speed; }
    else this.options.walk = null;
    if (mode === 'steady' || mode === 'getup' || mode === 'backoff') pinMarks(this);
    const m = this.measure;
    // Fallen? (head low for half a second)
    downT = m.headHeight - groundUnder(world.px[0]) < 0.8 ? downT + 1 / HZ : 0;
    if (downT > 0.5 && mode !== 'getup') {
      setMode('getup');
      getup = new ElmanNetwork(Float32Array.from(getupW));
      this.getupStart = this.frame;
    }
    const ahead = nextRise();
    if (mode === 'getup') {
      const t = (this.frame - this.getupStart) / HZ;
      const side = 'up';
      const hand = getupSaved.handover[side] || 3;
      const src = t < hand ? getup.action : stand.action;
      for (let j = 0; j < OUT; j++) this.action[j] = src[j];
      if (t > hand + 2 && isStanding(m)) setMode('steady');
    } else if (mode === 'steady') {
      for (let j = 0; j < OUT; j++) this.action[j] = stand.action[j];
      if (modeT > 1.0) {
        if (ahead && ahead.dist < 0.6 && ahead.dist > 0.15 && ahead.rise <= 0.45 && ahead.rise > 0.15) setMode('climb');
        else if (ahead && ahead.dist <= 0.15 && ahead.rise > 0.15) setMode('backoff');
        else setMode('walk');
      }
    } else if (mode === 'walk') {
      // A tall enough rise ahead: slow down from 1.5 m, stop 0.4 m before it, steady, then climb.
      // Low steps: the stairs walk.
      const tall = ahead && ahead.rise > 0.15;
      if (tall && ahead.dist < 0.8) setMode('brake');
      else {
        const onStairs = ahead && ahead.rise <= 0.15 && ahead.dist < 1.0 && ahead.box.top > L.platformTop + 0.01;
        const near = tall && ahead.dist < 3.0;
        m.walkSpeed = near ? 0.3 : start.walk.speed;
        m.walkDir = 1;
        walkCtl(onStairs ? stairsP : near ? slowP : walkP, this, tmp);
        for (let j = 0; j < OUT; j++) this.action[j] = tmp[j];
      }
    } else if (mode === 'brake') {
      // The walking controller asked for zero speed steps in place and comes to a stop; the stand net
      // took over from a 0.3 m/s walk badly (it fell forward onto the platform).
      m.walkSpeed = 0;
      m.walkDir = 1;
      walkCtl(slowP, this, tmp);
      for (let j = 0; j < OUT; j++) this.action[j] = tmp[j];
      if (modeT > 1.2 && Math.abs(m.centerVelocityX || 0) < 0.15) setMode('steady');
      else if (modeT > 2.5) setMode('steady');
    } else if (mode === 'backoff') {
      // Too close to climb: the stand net holds while we accept it (rare); then try the climb anyway.
      for (let j = 0; j < OUT; j++) this.action[j] = stand.action[j];
      if (modeT > 0.5) setMode('climb');
    } else if (mode === 'climb') {
      this.climbBox = ahead && ahead.dist < 0.6 ? ahead.box : this.climbBox;
      if (this.climbBox) {
        const going = stepCtl(stepP, this, tmp);
        for (let j = 0; j < OUT; j++) this.action[j] = tmp[j];
        if (!going || modeT > 6) setMode('steady');
      } else setMode('steady');
    }
    if (mode !== 'climb') for (let j = J; j < OUT; j++) this.action[j] = -1;
    origGrip();
  };
  const frames = [];
  let f = 0;
  let finished = false;
  for (;;) {
    if (record && f % 3 === 0) {
      const pz = [];
      for (let k = 0; k < world.parts.length; k++) pz.push(+world.px[k].toFixed(3), +world.py[k].toFixed(3), +Math.atan2(world.s[k], world.c[k]).toFixed(3));
      const dyn = ep.terrain.dynamics.map((id) => { const t = Box2D.b2Body_GetTransform(id); return [+t.p.x.toFixed(3), +t.p.y.toFixed(3), +Math.atan2(t.q.s, t.q.c).toFixed(3)]; });
      frames.push({ p: pz, up: isStanding(ep.measure) ? 1 : 0, dyn, mode });
    }
    const x = world.px[0] - startX;
    if (x > L.finish && isStanding(ep.measure)) { finished = true; break; }
    if (ep.advanceOneFrame()) break;
    f++;
  }
  const progress = (world.px[0] - startX) / L.finish;
  return { finished, progress, time: ep.frame / HZ, log, frames, blocks: blocks.map((b) => ({ x0: +b.x0.toFixed(3), x1: +b.x1.toFixed(3), top: +b.top.toFixed(3) })), dynSize: L.dynamics.map((d) => [d.w / 2, d.h / 2]) };
}

const shapes = world.parts.map((q) => ({ hx: +q.hx.toFixed(4), hy: +q.hy.toFixed(4), r: q.r ? +q.r.toFixed(4) : 0, ax: q.ax || 0, ay: q.ay || 0, capsule: !!q.capsule, circle: !!q.circle }));
const episodes = [];
let ok = 0;
for (let i = 0; i < n; i++) {
  const res = run(firstSeed + i, !!recordFile);
  if (res.finished) ok++;
  console.log('run', firstSeed + i, 'finished', res.finished ? 1 : 0, 'progress', (100 * res.progress).toFixed(0) + '%', 'time', res.time.toFixed(1) + 's', '|', res.log.join(', '));
  episodes.push({ seed: firstSeed + i, boxes: res.blocks, dynSize: res.dynSize, box: null, walk: null, board: null, frames: res.frames, finished: res.finished });
}
console.log('course finished', ok + '/' + n);
if (recordFile) fs.writeFileSync(recordFile, JSON.stringify({ skill: 6, fps: HZ / 3, shapes, episodes }));
sim.destroyWorld(world);
