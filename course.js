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
// --dump-states file: save the whole-body state (positions relative to the step's edge, velocities)
// each time the selector hands over to the climb, to tune the climb from real arrivals.
const dumpFile = arg('--dump-states', null);
const dumped = [];
// --dump-falls file: 1.2 s after each fall, the lying body (x relative to the pelvis) and the blocks
// within 2.5 m, to tune the get-up from real falls on the course.
const fallsFile = arg('--dump-falls', null);
// --dump-starts file: the body as each walk starts on top of a climbed block (steady -> walk), for
// tuning the walk start from them (walk-simbicon.js --states).
const startsFile = arg('--dump-starts', null);
// --dump-slow file: the body as the fast walker hands over to the slow one before a tall rise.
const slowFile = arg('--dump-slow', null);
const slows = [];
const starts = [];
const falls = [];

const Box2D = await import(pathToFileURL(path.join(root, 'vendor/PhaserBox2D.min.js')).href);
const { createSim } = await import('./src/sim.js');
const { ElmanNetwork, outputCountOf } = await import('./src/net.js');
const { makeSeededRandom } = await import('./src/utils.js');
const { evalMode } = await import('./src/training.js');
const { isStanding } = await import('./src/reward.js');
const simb = await import('./src/simbicon.js');
const step = await import('./src/stepup.js');
const pose = await import('./src/pose.js');
const { makePlanRunner } = await import('./src/planner.js');

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
// STOP: the walking controller with no forward swing target and strong balance feedback places each
// swing foot under where the body is heading (the capture point), so it halts in a step or two.
const stopFile = arg('--stop', 'trained/stop-controller.json');
const stopTuned = fs.existsSync(path.resolve(root, stopFile)) ? load(stopFile).params : null;
const stopP = stopTuned
  ? Object.assign({}, slowP, stopTuned, { track: true })
  : Object.assign({}, slowP, { swing0: +(process.env.STOP_SWING || 0.05), swing1: 0, cd: +(process.env.STOP_CD || 1.2), cv: +(process.env.STOP_CV || 0.5), track: true });
const holdK = stopTuned ? { kHip: stopTuned.holdKHip, kAnkle: stopTuned.holdKAnkle } : { kHip: 1.5, kAnkle: 3 };
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
  // platform 0.3 m from 10 m to the end (4 m of flat ground after the bumps: the slow approach walker
  // starts 3 m before it and was never tuned on bumps)
  const platformX = 10;
  const platformTop = 0.3;
  // then two big steps of 0.3 m, each with 2.5 m of flat top (the step climb, repeated); normal stairs
  // (10-18 cm) are not learned yet.
  const stairsX = 16;
  const rise = 0.3;
  const tread = 3.0;
  const nSteps = 2;
  const end = stairsX + nSteps * tread + 4;
  statics.push({ dx: platformX, w: end - platformX, h: platformTop });
  for (let i = 0; i < nSteps; i++) statics.push({ dx: stairsX + i * tread, w: end - (stairsX + i * tread), h: platformTop + (i + 1) * rise });
  // loose light boxes on the platform, 11.5-14 m
  const dynamics = [];
  for (let i = 0; i < (process.env.NO_DYN ? 0 : 3); i++) dynamics.push({ dx: 11.5 + i * 0.8 + r() * 0.3, w: +(process.env.BOX_W || 0.1), h: +(process.env.BOX_H || 0.06), density: +(process.env.BOX_D || 3), y: platformTop });
  return { statics, dynamics, platformX, platformTop, stairsX, finish: stairsX + (nSteps - 1) * tread + 1.5, end };
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
      const fl = world.px[config.partIndex.footLeft], fr = world.px[config.partIndex.footRight];
      log.push(((ep.frame / HZ).toFixed(1)) + 's ' + m + (m === 'climb' ? ' [L ' + fl.toFixed(2) + ' R ' + fr.toFixed(2) + (ep.climbBox || nextRiseLog() ? ' edge ' + nextRiseLog() : '') + ']' : '') + (m === 'getup' ? ' @' + (world.px[0] - startX).toFixed(1) + 'm from ' + mode : ''));
      if (verbose) console.log('  ', (ep.frame / HZ).toFixed(1), 's ->', m, 'x', ((world.px[0] - startX)).toFixed(2));
    }
    if (m === 'walk' && mode === 'steady' && startsFile && groundUnder(world.px[0]) > 0.1) {
      const px0 = world.px[0];
      const pose = [];
      const vel = [];
      for (let k = 0; k < world.parts.length; k++) {
        pose.push(+(world.px[k] - px0).toFixed(4), +world.py[k].toFixed(4), +Math.atan2(world.s[k], world.c[k]).toFixed(4));
        vel.push(+world.vx[k].toFixed(3), +world.vy[k].toFixed(3), +world.spin[k].toFixed(3));
      }
      const near = blocks.filter((b) => b.x1 > px0 - 2.5 && b.x0 < px0 + 6).map((b) => ({ x0: +(b.x0 - px0).toFixed(3), x1: +(Math.min(b.x1, px0 + 6) - px0).toFixed(3), top: b.top }));
      starts.push({ pose, vel, blocks: near });
    }
    if (m === 'climb' && dumpFile) {
      const a = nextRise();
      if (a) {
        const pose = [];
        const vel = [];
        for (let k = 0; k < world.parts.length; k++) {
          pose.push(+(world.px[k] - a.box.x0).toFixed(4), +(world.py[k] - a.here).toFixed(4), +Math.atan2(world.s[k], world.c[k]).toFixed(4));
          vel.push(+world.vx[k].toFixed(3), +world.vy[k].toFixed(3), +world.spin[k].toFixed(3));
        }
        dumped.push({ pose, vel, rise: +a.rise.toFixed(3) });
      }
    }
    if (m !== 'getup' && ep.options.getup) { ep.options.getup = false; ep.options.startTime = 0; }
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
    // The floor we are on: the higher of the two feet's (a foot on a block means we are on it; by the
    // middle of the feet, a body standing with its back foot at the edge saw the block it had just
    // climbed as a rise ahead, braked and stepped back off it).
    const here = process.env.HERE_MID ? groundUnder(fx - 0.05) : Math.max(groundUnder(world.px[config.partIndex.footLeft] - 0.03), groundUnder(world.px[config.partIndex.footRight] - 0.03));
    let best = null;
    for (const b of blocks) {
      if (b.x0 > fx - 0.02 && b.top - here > 0.06 && (!best || b.x0 < best.x0)) best = b;
    }
    return best ? { box: best, dist: best.x0 - fx, rise: best.top - here, here } : null;
  };
  const nextRiseLog = () => { const a = nextRise(); return a ? a.box.x0.toFixed(2) : '-'; };
  const tmp = new Float64Array(OUT);
  const walkSpec = ep.options.walk;
  const origDecide = ep.decide.bind(ep);
  ep.decide = function () {
    // the stand net sees heights from the floor under its feet (see sim.js groundOffset)
    if (process.env.GROUND_REL !== '0') {
      const fx = (world.px[config.partIndex.footLeft] + world.px[config.partIndex.footRight]) / 2;
      this.groundOffset = groundUnder(fx);
    }
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
    if (mode === 'walk' || mode === 'brake' || mode === 'netwalk' || mode === 'stop') { if (!this.options.walk) this.options.walk = walkSpec; this.options.walk.speed = mode === 'netwalk' ? 0.3 : start.walk.speed; }
    else this.options.walk = null;
    if (mode === 'steady' || mode === 'getup' || mode === 'backoff') pinMarks(this);
    const m = this.measure;
    // Fallen? (head low for half a second)
    downT = m.headHeight - groundUnder(world.px[0]) < 0.8 ? downT + 1 / HZ : 0;
    // Capture once the body has come to rest (every part slower than 0.15 m/s) or after 3 s: captured
    // 0.7 s after the fall, the bodies were still tumbling, legs in the air.
    const settled = () => { for (let k = 0; k < world.parts.length; k++) if (Math.abs(world.vx[k]) > 0.15 || Math.abs(world.vy[k]) > 0.15) return false; return true; };
    const sinceFall = (this.frame - (this.getupStart || 0)) / HZ;
    if (fallsFile && mode === 'getup' && !this.fallCaptured && sinceFall > 0.8 && (settled() || sinceFall > 3)) {
      this.fallCaptured = true;
      const px0 = world.px[0];
      const pose = [];
      const vel = [];
      for (let k = 0; k < world.parts.length; k++) {
        pose.push(+(world.px[k] - px0).toFixed(4), +world.py[k].toFixed(4), +Math.atan2(world.s[k], world.c[k]).toFixed(4));
        vel.push(+world.vx[k].toFixed(3), +world.vy[k].toFixed(3), +world.spin[k].toFixed(3));
      }
      const near = blocks.filter((b) => b.x1 > px0 - 2.5 && b.x0 < px0 + 2.5).map((b) => ({ x0: +(b.x0 - px0).toFixed(3), x1: +(Math.min(b.x1, px0 + 3) - px0).toFixed(3), top: b.top }));
      falls.push({ pose, vel, blocks: near });
    }
    if (downT > 0.5 && mode !== 'getup') {
      setMode('getup');
      getup = new ElmanNetwork(Float32Array.from(getupW));
      this.getupStart = this.frame;
      this.fallCaptured = false;
    }
    const ahead = nextRise();
    if (mode === 'getup' && fallsFile) {
      // Collecting falls: the body comes to rest with the arms and legs relaxed toward standing (the
      // stand net), no get-up yet; the run ends once the state is captured.
      for (let j = 0; j < OUT; j++) this.action[j] = stand.action[j];
      if (this.fallCaptured) this.stopNow = true;
    } else if (mode === 'getup' && !process.env.GETUP_OLD) {
      // Get up as the expert was trained: from lying at rest (the stand net relaxes the body until every
      // part is still, or 3 s), with the handover time for the side it lies on (face up: torso turned
      // back), a fresh network each attempt; not up 5 s after the handover: rest and try again.
      if (!this.gu || modeT < 1 / HZ + 1e-9) this.gu = { phase: 'rest', t0: this.frame, tries: 0 };
      const g = this.gu;
      const t = (this.frame - g.t0) / HZ;
      if (g.phase === 'rest') {
        for (let j = 0; j < OUT; j++) this.action[j] = stand.action[j];
        // A retry from the same resting pose failed the same way every time (every unfinished course
        // ended retrying): first curl up for a second (hips flexed, knees bent) so it lies differently.
        if (g.tries > 0 && t < 1.0 && !process.env.NO_CURL) {
          const curl = g.tries % 2 ? 1 : -1;
          for (const j of [5, 11]) this.action[j] = 0.8 * curl;
          for (const j of [6, 12]) this.action[j] = -1;
          for (const j of [2, 8]) this.action[j] = 0.7 * curl;
        }
        if ((t > (g.tries > 0 ? 1.6 : 0.6) && settled()) || t > 4) {
          g.phase = 'go'; g.t0 = this.frame; g.tries++;
          g.side = Math.atan2(world.s[0], world.c[0]) > 0 ? 'up' : 'down';
          getup = new ElmanNetwork(Float32Array.from(getupW));
          // The expert reads a get-up clock (time since the start of the get-up) that only runs with
          // options.getup; without it the clock inputs were zero and it flailed (1 of 31 course falls).
          this.options.getup = true;
          this.options.startTime = -this.frame / HZ;
        }
      } else {
        const hand = getupSaved.handover[g.side] || 3;
        const src = t < hand ? getup.action : stand.action;
        for (let j = 0; j < OUT; j++) this.action[j] = src[j];
        if (t > hand + 2 && isStanding(m)) { this.gu = null; setMode('steady'); }
        else if (t > hand + 5) { g.phase = 'rest'; g.t0 = this.frame; }
      }
    } else if (mode === 'getup') {
      const t = (this.frame - this.getupStart) / HZ;
      const side = 'up';
      const hand = getupSaved.handover[side] || 3;
      const src = t < hand ? getup.action : stand.action;
      for (let j = 0; j < OUT; j++) this.action[j] = src[j];
      if (t > hand + 2 && isStanding(m)) setMode('steady');
    } else if (mode === 'steady') {
      for (let j = 0; j < OUT; j++) this.action[j] = stand.action[j];
      if (modeT > +(process.env.STEADY_T || 1.0)) {
        const front = Math.max(world.px[config.partIndex.footLeft], world.px[config.partIndex.footRight]);
        const gap = ahead ? ahead.box.x0 - front : 9;
        if (ahead && gap < 0.55 && gap > 0.08 && ahead.rise <= 0.45 && ahead.rise > 0.15) setMode(process.env.SQUARE ? 'square' : 'climb');
        else if (ahead && gap <= 0.08 && ahead.rise > 0.15) setMode('backoff');
        else setMode('walk');
      }
    } else if (mode === 'walk') {
      // A tall enough rise ahead: slow down from 1.5 m, stop 0.4 m before it, steady, then climb.
      // Low steps: the stairs walk.
      const tall = ahead && ahead.rise > 0.15;
      // DIRECT: from the slow walk, as soon as the front foot is down 0.25-0.5 m from the edge with the
      // back foot down too, start the climb without stopping (stopping left the feet up to 1.1 m apart).
      if (tall && process.env.DIRECT) {
        const fl = world.px[config.partIndex.footLeft], fr = world.px[config.partIndex.footRight];
        const gapF = ahead.box.x0 - Math.max(fl, fr);
        const both = m.touch[config.partIndex.footLeft] && m.touch[config.partIndex.footRight];
        if (gapF < +(process.env.DIRECT_GAP || 0.5) && gapF > 0.15 && both && Math.abs(fr - fl) < 0.6) { setMode('climb'); }
      }
      // Near a tall step: the stand-and-walk NETWORK walks the last metres slowly and stops on its own
      // when its walk command goes to zero (it was trained to do both); the walking controllers do not
      // stop when asked.
      if (tall && ahead.dist < +(process.env.NETWALK_AT || 1.8) && process.env.NETWALK) { setMode('netwalk'); pinMarks(this); }
      if (mode !== 'walk') { /* switched above */ }
      else if (tall && ahead.dist < +(process.env.STOP_AT || 0.8)) setMode(process.env.STOP_MODE || (process.env.NO_BRAKE ? 'steady' : 'brake'));
      else {
        const onStairs = ahead && ahead.rise <= 0.15 && ahead.dist < 1.0 && ahead.box.top > L.platformTop + 0.01;
        // STRIKE: hand over to the slow walker only as a foot lands (switching mid-swing tripped it).
        let near = tall && ahead.dist < 3.0;
        if (process.env.STRIKE) {
          const tl = !!m.touch[config.partIndex.footLeft], tr = !!m.touch[config.partIndex.footRight];
          const strike = (tl && !this.prevTouchL) || (tr && !this.prevTouchR);
          this.prevTouchL = tl; this.prevTouchR = tr;
          if (!tall || modeT < 0.05) this.slowLatch = false;
          if (near && strike) this.slowLatch = true;
          near = near && this.slowLatch;
        }
        // START_SLOW s: the first seconds of each walk with the slow walker (starting at 0.9 m/s from
        // the stance a climb leaves fell within half a second).
        // SLOW_UP: on top of a climbed block walk with the slow walker all the way (it starts from the
        // stance a climb leaves 8/8; the fast one fell in a third of them).
        const starting = modeT < +(process.env.START_SLOW || 0) || (process.env.SLOW_UP && groundUnder(world.px[0]) > 0.1);
        if (near && !this.slowDumped && slowFile && groundUnder(world.px[0]) < 0.1) {
          this.slowDumped = true;
          const px0 = world.px[0];
          const pose = [];
          const vel = [];
          for (let k = 0; k < world.parts.length; k++) {
            pose.push(+(world.px[k] - px0).toFixed(4), +world.py[k].toFixed(4), +Math.atan2(world.s[k], world.c[k]).toFixed(4));
            vel.push(+world.vx[k].toFixed(3), +world.vy[k].toFixed(3), +world.spin[k].toFixed(3));
          }
          const nearB = blocks.filter((b) => b.x1 > px0 - 2.5 && b.x0 < px0 + 6).map((b) => ({ x0: +(b.x0 - px0).toFixed(3), x1: +(b.x1 - px0).toFixed(3), top: b.top }));
          slows.push({ pose, vel, blocks: nearB });
          this.stopNow = true; // collected: end this run
        }
        m.walkSpeed = near || starting ? 0.3 : start.walk.speed;
        m.walkDir = 1;
        walkCtl(onStairs ? stairsP : near || starting ? slowP : walkP, this, tmp);
        for (let j = 0; j < OUT; j++) this.action[j] = tmp[j];
      }
    } else if (mode === 'square') {
      // Set the feet as the climb was tuned: weight onto the front (right) foot, bring the back foot in
      // to 0.25 m behind it, settle. Done by the pose executor (src/pose.js) with a short plan.
      if (modeT < 0.02 || !this.squareRunner) {
        const g = groundUnder(world.px[0]);
        const R = pose.soleOf(world, config.partIndex.footRight);
        const back = { x: R.x - 0.25, y: g };
        this.squareRunner = makePlanRunner([
          { dur: 0.7, pelvis: { x: { front: -0.02 }, y: g + 0.9 }, torso: 0, feet: ['hold', 'hold'], hands: [null, null], planted: [true, true], grip: false, support: 'front' },
          { dur: 0.4, pelvis: { x: { front: 0 }, y: g + 0.9 }, torso: 0, feet: [{ x: R.x - 0.4, y: g + 0.08 }, 'hold'], hands: [null, null], planted: [false, true], grip: false, support: 'front' },
          { dur: 0.4, pelvis: { x: { front: -0.05 }, y: g + 0.9 }, torso: 0, feet: [back, 'hold'], hands: [null, null], planted: [false, true], grip: false, support: 'front' },
          { dur: 0.8, pelvis: { x: { front: -0.12 }, y: g + 0.92 }, torso: 0, feet: ['hold', 'hold'], hands: [null, null], planted: [true, true], grip: false },
        ], world, HZ);
      }
      const goal = this.squareRunner.step();
      if (!goal) { this.squareRunner = null; setMode('climb'); }
      else {
        pose.commandsFor(world, pose.balancedTargets(goal, m, { planted: goal.planted, support: goal.support != null ? goal.support : undefined }), this.action);
      }
    } else if (mode === 'netwalk') {
      for (let j = 0; j < OUT; j++) this.action[j] = stand.action[j];
      // Stop by the FRONT foot: the climb was tuned with it 0.15-0.35 m from the edge; stopping by the
      // middle of a wide stance left it 14 cm away and the knee hit the edge.
      const front = Math.max(world.px[config.partIndex.footLeft], world.px[config.partIndex.footRight]);
      if (!ahead || ahead.box.x0 - front < +(process.env.STOP_FRONT || 0.3)) setMode('steady');
    } else if (mode === 'stop') {
      m.walkSpeed = 0;
      m.walkDir = 1;
      walkCtl(stopP, this, tmp);
      for (let j = 0; j < OUT; j++) this.action[j] = tmp[j];
      if (modeT > 2) setMode(process.env.HOLD_NET ? 'steady' : 'hold'); // as tuned in stop-search.js
    } else if (mode === 'hold') {
      // Stand still on both feet with the pose executor, then decide.
      if (!this.holdFeet) this.holdFeet = [config.partIndex.footLeft, config.partIndex.footRight].map((f) => pose.soleOf(world, f));
      const g = groundUnder(world.px[0]);
      const mid = (this.holdFeet[0].x + this.holdFeet[1].x) / 2;
      const goal = { pelvis: { x: mid, y: g + 0.92 }, torso: 0, feet: this.holdFeet.map((q) => ({ x: q.x, y: q.y, angle: 0 })), planted: [true, true] };
      pose.commandsFor(world, pose.balancedTargets(goal, m, { planted: [true, true], kHip: holdK.kHip, kAnkle: holdK.kAnkle, kAnkleVel: 0.15, kHipVel: 0.1 }), this.action);
      if (modeT > 2.0) {
        this.holdFeet = null;
        const front = Math.max(world.px[config.partIndex.footLeft], world.px[config.partIndex.footRight]);
        const gap = ahead ? ahead.box.x0 - front : 9;
        if (ahead && ahead.rise > 0.15 && gap < +(process.env.CLIMB_GAP || 0.6)) setMode('climb');
        else setMode('walk');
      }
    } else if (mode === 'upright') {
      // After a climb the body is left leaning 25-30 degrees forward and the stand net holds that lean
      // (then the walk falls forward). The pose executor straightens it gradually: torso lean and pelvis
      // ramp from where they are to upright over the feet, then decide as 'hold' does.
      if (!this.up0) {
        this.up0 = { feet: [config.partIndex.footLeft, config.partIndex.footRight].map((q) => pose.soleOf(world, q)), torso: Math.atan2(world.s[0], world.c[0]), px: world.px[0], py: world.py[0] };
      }
      const u = this.up0;
      const T = +(process.env.UP_T || 1.5);
      const a = Math.min(1, modeT / T);
      const e = a * a * (3 - 2 * a);
      const g = groundUnder(world.px[0]);
      const mid = (u.feet[0].x + u.feet[1].x) / 2 + +(process.env.UP_DX || 0);
      const goal = { pelvis: { x: u.px + (mid - u.px) * e, y: g + 0.92 }, torso: u.torso * (1 - e), feet: u.feet.map((q) => ({ x: q.x, y: q.y, angle: 0 })), planted: [true, true] };
      pose.commandsFor(world, pose.balancedTargets(goal, m, { planted: [true, true], kHip: holdK.kHip, kAnkle: holdK.kAnkle, kAnkleVel: 0.15, kHipVel: 0.1 }), this.action);
      if (modeT > T + 0.6) {
        this.up0 = null;
        const front = Math.max(world.px[config.partIndex.footLeft], world.px[config.partIndex.footRight]);
        const gap = ahead ? ahead.box.x0 - front : 9;
        if (ahead && ahead.rise > 0.15 && gap < +(process.env.CLIMB_GAP || 0.6)) setMode('climb');
        else setMode('walk');
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
      if (modeT < 0.05) {
        const fl = world.px[config.partIndex.footLeft];
        const fr = world.px[config.partIndex.footRight];
        const front = Math.max(fl, fr);
        this.climbBox = ahead && ahead.box.x0 - front < 0.7 ? ahead.box : null;
        // Lead with whichever foot is in front (the climb was tuned with the right one in front and
        // the legs are identical in 2D); with the left in front it tried to lift the back leg forever.
        this.climbP = process.env.LEAD_PARAM ? stepP : Object.assign({}, stepP, { lead: fr >= fl ? 1 : -1 });
      }
      if (this.climbBox) {
        const going = stepCtl(this.climbP || stepP, this, tmp);
        if (process.env.DEBUG_CLIMB && Math.round(modeT * 60) % 20 === 1) console.log('    climb t', modeT.toFixed(2), 'state', this.stepup && this.stepup.state, 'going', going, 'lead', this.climbP && this.climbP.lead, 'hipL', tmp[5].toFixed(2), 'hipR', tmp[11].toFixed(2), 'box', this.climbBox.x0.toFixed(2), this.climbBox.top, 'thighL', Math.atan2(world.s[6], world.c[6]).toFixed(2), 'thighR', Math.atan2(world.s[12], world.c[12]).toFixed(2), 'act5', this.action[5].toFixed(2), 'locked', !!this.legLocked, 'passive', !!this.passive);
        for (let j = 0; j < OUT; j++) this.action[j] = tmp[j];
        if (!going || modeT > 6) setMode(process.env.CLIMB_THEN || 'steady');
      } else setMode(process.env.CLIMB_THEN || 'steady');
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
    if (ep.stopNow) break;
    if (ep.advanceOneFrame()) break;
    f++;
  }
  const progress = (world.px[0] - startX) / L.finish;
  return { finished, progress, time: ep.frame / HZ, log, frames, finish: +(startX + L.finish).toFixed(3), finishY: blocks.length ? Math.max(...blocks.map((b) => b.top)) : 0, blocks: blocks.map((b) => ({ x0: +b.x0.toFixed(3), x1: +b.x1.toFixed(3), top: +b.top.toFixed(3) })), dynSize: L.dynamics.map((d) => [d.w / 2, d.h / 2]) };
}

const shapes = world.parts.map((q) => ({ hx: +q.hx.toFixed(4), hy: +q.hy.toFixed(4), r: q.r ? +q.r.toFixed(4) : 0, ax: q.ax || 0, ay: q.ay || 0, capsule: !!q.capsule, circle: !!q.circle }));
const episodes = [];
let ok = 0;
for (let i = 0; i < n; i++) {
  const res = run(firstSeed + i, !!recordFile);
  if (res.finished) ok++;
  console.log('run', firstSeed + i, 'finished', res.finished ? 1 : 0, 'progress', (100 * res.progress).toFixed(0) + '%', 'time', res.time.toFixed(1) + 's', '|', res.log.join(', '));
  episodes.push({ seed: firstSeed + i, boxes: res.blocks, dynSize: res.dynSize, finish: res.finish, finishY: res.finishY, box: null, walk: null, board: null, frames: res.frames, finished: res.finished });
}
console.log('course finished', ok + '/' + n);
if (recordFile) fs.writeFileSync(recordFile, JSON.stringify({ skill: 6, fps: HZ / 3, shapes, episodes }));
if (dumpFile) { fs.writeFileSync(dumpFile, JSON.stringify(dumped)); console.log('states', dumped.length, '->', dumpFile); }
if (slowFile) { fs.writeFileSync(slowFile, JSON.stringify(slows)); console.log('slow handovers', slows.length, '->', slowFile); }
if (startsFile) { fs.writeFileSync(startsFile, JSON.stringify(starts)); console.log('starts', starts.length, '->', startsFile); }
if (fallsFile) { fs.writeFileSync(fallsFile, JSON.stringify(falls)); console.log('falls', falls.length, '->', fallsFile); }
sim.destroyWorld(world);
