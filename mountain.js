// MOUNTAIN: climb an irregular slope (a rigid polygon under a jagged line rising up and to the right):
// ledges, slopes, steps and steep walls. A planner in code looks at the terrain ahead and picks where each
// foot and hand goes; the pose executor (src/pose.js) moves the joints there. On gentle ground it steps
// with the feet only, keeping balance over the standing foot; where the ground ahead rises steeply it
// puts its hands on the rock (a hand grips wherever it touches the mountain) and scrambles on all fours.
// No external forces. The planner's numbers are tuned with the cross-entropy method on progress.
//   node mountain.js --iters 30 --out trained/mountain.json
//   node mountain.js --show --init trained/mountain.json --record out.json
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
const out = arg('--out', 'trained/mountain.json');
const init = arg('--init', null);
const pop = +arg('--pop', 24);
const elite = +arg('--elite', 6);
const seeds = +arg('--seeds', 3);
const seconds = +arg('--seconds', 60);
const height = +arg('--height', 3); // summit height
const difficulty = +arg('--difficulty', 0.5); // 0 gentle slopes and ledges .. 1 mostly steps and steep rock

const Box2D = await import(pathToFileURL(path.join(root, 'vendor/PhaserBox2D.min.js')).href);
const { createSim } = await import('./src/sim.js');
const { ElmanNetwork } = await import('./src/net.js');
const { makeSeededRandom, randomNormalSample } = await import('./src/utils.js');
const { evalMode } = await import('./src/training.js');
const pose = await import('./src/pose.js');

setSkill(6);
const sim = createSim(Box2D);
const world = sim.build();
const HZ = config.physics.hz;
const J = config.net.jointCount;
const OUT = J + config.net.gripOutputs;
const P = config.partIndex;
const weights = ElmanNetwork.copy(JSON.parse(fs.readFileSync(path.join(root, 'trained/stand-and-walk.json'), 'utf8')).theta);

const SPEC = [
  ['hipH', 0.9, 0.06], // hips above the ground under them, on two feet
  ['hipHands', 0.62, 0.1], // and when scrambling with the hands down
  ['stepMax', 0.42, 0.08], // longest step forward
  ['stepUp', 0.42, 0.08], // highest step up with the feet alone
  ['handRise', 0.35, 0.1], // hands go down when the ground rises this much within 0.6 m ahead
  ['handPull', 0.35, 0.15], // with hands down the hips sit this share of the way from the feet to the hands
  ['leanK', 0.5, 0.2], // trunk leans in by this share of the slope angle ahead
  ['handLean', 0.35, 0.2], // and this much more when scrambling
  ['clear', 0.12, 0.05], // a moving foot clears the ground in between by this much
  ['tFoot', 0.8, 0.2], // seconds per foot move (weight shift, then the step)
  ['tHand', 0.6, 0.2], // seconds per hand move
  ['shift', 0.35, 0.1], // share of a foot move spent shifting the weight before lifting the foot
  ['kHip', 1.0, 0.4], // balance by hips (as in the pose executor), on two feet
  ['kAnkle', 2.0, 0.6], // balance by ankles
  ['torsoGain', 0.4, 0.3], // supporting legs turn the trunk back to its lean
  ['handReach', 0.5, 0.06], // hands go this far at most from the shoulder
];
const decode = (v) => {
  const p = {};
  SPEC.forEach(([k, b], i) => (p[k] = b + (v[i] || 0)));
  for (const k of ['tFoot', 'tHand']) p[k] = Math.max(0.3, p[k]);
  p.shift = Math.max(0.1, Math.min(0.6, p.shift));
  return p;
};
const smooth = (s) => s * s * (3 - 2 * s);

// A jagged line from flat ground up to a summit plateau, x strictly increasing (each piece a convex
// trapezoid down to below the floor), relative to the start of the feet.
export function makeMountain(r, H = height, d = difficulty) {
  const pts = [[-4, 0], [1.2, 0]];
  let x = 1.2, y = 0;
  while (y < H) {
    const u = r();
    const pSteep = 0.1 + 0.2 * d, pStep = 0.2 + 0.2 * d, pSlope = 0.35 - 0.1 * d;
    let dx, dy;
    if (u < pSteep) { const a = (55 + r() * 20) * Math.PI / 180; const L = 0.25 + r() * 0.3; dx = L * Math.cos(a); dy = L * Math.sin(a); }
    else if (u < pSteep + pStep) { dx = 0.04 + r() * 0.04; dy = 0.12 + r() * (0.15 + 0.15 * d); }
    else if (u < pSteep + pStep + pSlope) { const a = (18 + r() * 22) * Math.PI / 180; const L = 0.25 + r() * 0.35; dx = L * Math.cos(a); dy = L * Math.sin(a); }
    else { dx = 0.25 + r() * 0.35; dy = dx * Math.tan((-4 + r() * 12) * Math.PI / 180); }
    if (y + dy > H) dy = H - y;
    x += dx; y = Math.max(0, y + dy);
    pts.push([+x.toFixed(3), +y.toFixed(3)]);
  }
  pts.push([x + 5, y]);
  const polys = [];
  for (let i = 1; i < pts.length - 1; i++) {
    const [x1, y1] = pts[i], [x2, y2] = pts[i + 1];
    if (y1 <= 0.001 && y2 <= 0.001) continue;
    polys.push([[x1, -0.2], [x2, -0.2], [x2, y2], [x1, y1]]);
  }
  return { surface: pts, polys, summitX: x, H: y };
}

function episode(p, seed, record) {
  const r = makeSeededRandom(8000 + seed);
  const start = sim.fillTrainingStart({}, r, evalMode(6));
  start.tilt = 0;
  start.walk = null;
  start.frames = Math.round(seconds * HZ);
  const mtn = makeMountain(r);
  start.terrain = { statics: [], polys: mtn.polys, surface: mtn.surface };
  world.rebuild();
  const ep = new sim.Episode(world, weights, 0, 0, Object.assign({ phase: 0 }, start));
  const S = ep.terrain.surface; // absolute
  const summitX = S[S.length - 2][0];
  const H = mtn.H;
  const ground = (x) => {
    if (x <= S[0][0]) return 0;
    for (let i = 0; i < S.length - 1; i++) if (x <= S[i + 1][0]) { const [x1, y1] = S[i], [x2, y2] = S[i + 1]; return x2 > x1 ? y1 + (y2 - y1) * (x - x1) / (x2 - x1) : Math.max(y1, y2); }
    return S[S.length - 1][1];
  };
  const groundMax = (a, b) => { let m = Math.max(ground(a), ground(b)); for (const [x, y] of S) if (x > a && x < b) m = Math.max(m, y); return m; };
  // a foothold: the ground within 7 cm either side is level to 5 cm and the foot is not against a rise
  const flatAt = (x) => { const a = ground(x - 0.07), b = ground(x), c = ground(x + 0.07); return Math.max(a, b, c) - Math.min(a, b, c) < 0.05; };
  const footholdAfter = (fromX, fromY, maxRise) => {
    let best = null;
    for (let x = fromX + 0.08; x <= fromX + p.stepMax; x += 0.02) {
      const y = ground(x);
      if (y - fromY > maxRise || !flatAt(x)) continue;
      best = { x, y };
    }
    return best;
  };
  const feet0 = [P.footLeft, P.footRight].map((q) => pose.soleOf(world, q));
  const lim = {
    foot: feet0.map((f) => ({ x: f.x, y: f.y })),
    hand: [null, null], // {x, y} while on the rock
    grip: [false, false],
  };
  const pv0 = pose.pelvisOf(world);
  let pelvisGoal = { x: pv0.x, y: pv0.y };
  let pelvisFrom = { ...pelvisGoal };
  let leanNow = 0;
  let move = null;
  let stuck = 0;
  const scrambling = () => lim.grip[0] || lim.grip[1];
  const lead = () => (lim.foot[0].x >= lim.foot[1].x ? 0 : 1);
  const riseAhead = (x, y) => groundMax(x, x + 0.6) - y;
  const slopeAhead = (x) => Math.atan2(ground(x + 0.5) - ground(x - 0.1), 0.6);
  const shoulderAt = (pg, lean) => ({ x: pg.x + Math.sin(-lean) * 0.52, y: pg.y + Math.cos(lean) * 0.52 });
  const handholdFrom = (sh) => {
    let best = null;
    for (let x = sh.x - 0.1; x <= sh.x + p.handReach; x += 0.02) {
      const y = ground(x) + 0.03;
      if (Math.hypot(x - sh.x, y - sh.y) > p.handReach) continue;
      if (!best || x > best.x) best = { x, y };
    }
    return best;
  };
  const next = () => {
    const L = lead(), T = 1 - L;
    const lf = lim.foot[L], tf = lim.foot[T];
    const steep = riseAhead(lf.x, lf.y) > p.handRise;
    // the planned hips and shoulders for this stance, to see what the hands can reach
    const pg = pelvisFor(null);
    const sh = shoulderAt(pg, leanNow);
    if (steep) {
      // hands: the one behind (or not down) goes forward first, while it can move ahead
      for (const k of [0, 1].sort((a, b) => (lim.hand[a] ? lim.hand[a].x : -9) - (lim.hand[b] ? lim.hand[b].x : -9))) {
        const h = lim.hand[k];
        const hold = handholdFrom(sh);
        if (hold && (!h || hold.x > h.x + 0.1)) return { kind: 'hand', k, target: hold };
      }
    } else if (scrambling() && riseAhead(lf.x, lf.y) < p.handRise * 0.5 && lim.hand.every((h) => !h || h.x < pg.x + 0.1)) {
      return { kind: 'release' };
    }
    // feet: the trailing foot steps past the leading one onto the next foothold
    const maxRise = scrambling() ? p.stepUp + 0.15 : p.stepUp;
    let hold = footholdAfter(lf.x, lf.y, maxRise);
    if (hold) return { kind: 'foot', k: T, target: hold };
    // no foothold past the lead foot: bring the trailing foot up next to it
    hold = footholdAfter(tf.x, tf.y, maxRise);
    if (hold && hold.x < lf.x + 0.02) return { kind: 'foot', k: T, target: hold };
    return null;
  };
  // hips: over the standing foot when a foot is in the air, between the feet otherwise; pulled toward the
  // hands when scrambling; at a set height above the ground, no higher than the legs reach.
  function pelvisFor(m) {
    const planted = [0, 1].filter((k) => !(m && m.kind === 'foot' && m.k === k && m.lifted));
    let x;
    if (m && m.kind === 'foot') x = lim.foot[1 - m.k].x + 0.02;
    else x = (lim.foot[0].x + lim.foot[1].x) / 2;
    const hands = lim.hand.filter((h, k) => h && lim.grip[k]);
    if (hands.length) {
      const hx = hands.reduce((a, h) => a + h.x, 0) / hands.length;
      x = x + p.handPull * (hx - x);
    }
    leanNow = -p.leanK * slopeAhead(x) - (hands.length ? p.handLean : 0);
    let y = ground(x) + (hands.length ? p.hipHands : p.hipH);
    for (const k of planted) y = Math.min(y, lim.foot[k].y + 0.93);
    return { x, y };
  }
  const startMove = () => {
    const m = next();
    if (!m) { stuck += 1; move = { kind: 'hold', t: 0, dur: 0.5 }; return; }
    move = Object.assign(m, { t: 0, wait: 0 });
    if (m.kind === 'foot') { move.dur = p.tFoot; move.from = { ...lim.foot[m.k] }; }
    else if (m.kind === 'hand') { move.dur = p.tHand; const hp = lim.hand[m.k] || pose.handOf(world, m.k === 0 ? P.handLeft : P.handRight); move.from = { x: hp.x, y: hp.y }; lim.grip[m.k] = false; }
    else move.dur = 0.5;
    pelvisFrom = { ...pelvisGoal };
  };
  let best = 0;
  let fell = false;
  let topped = false;
  let upTop = 0;
  const recFrames = record ? [] : null;
  const tmp = new Float64Array(OUT);
  ep.updateGrip = ((orig) => function () {
    const m = this.measure;
    if (!move) startMove();
    move.t += 1 / HZ;
    const s = Math.min(1, move.t / move.dur);
    const feet = lim.foot.map((f) => ({ x: f.x, y: f.y, angle: 0 }));
    const hands = lim.hand.map((h) => (h ? { x: h.x, y: h.y } : null));
    const gripping = lim.grip.slice();
    const planted = [true, true];
    if (move.kind === 'foot') {
      // weight shift first, then lift: up over the highest ground in between, across, and down
      move.lifted = s > p.shift;
      if (move.lifted) {
        const q = (s - p.shift) / (1 - p.shift);
        const top = Math.max(groundMax(Math.min(move.from.x, move.target.x), Math.max(move.from.x, move.target.x)), move.from.y, move.target.y) + p.clear;
        const pts = [move.from, { x: move.from.x - 0.02, y: top }, { x: move.target.x, y: top }, move.target];
        const cuts = [0, 0.3, 0.75, 1];
        let si = 0;
        while (si < 2 && q > cuts[si + 1]) si++;
        const e = smooth(Math.min(1, (q - cuts[si]) / (cuts[si + 1] - cuts[si])));
        feet[move.k] = { x: pts[si].x + (pts[si + 1].x - pts[si].x) * e, y: pts[si].y + (pts[si + 1].y - pts[si].y) * e, angle: 0 };
        planted[move.k] = false;
      }
    } else if (move.kind === 'hand') {
      const top = Math.max(move.from.y, move.target.y) + 0.12;
      const pts = [move.from, { x: move.from.x, y: top }, { x: move.target.x, y: move.target.y + 0.08 }, move.target];
      const cuts = [0, 0.3, 0.75, 1];
      let si = 0;
      while (si < 2 && s > cuts[si + 1]) si++;
      const e = smooth(Math.min(1, (s - cuts[si]) / (cuts[si + 1] - cuts[si])));
      hands[move.k] = { x: pts[si].x + (pts[si + 1].x - pts[si].x) * e, y: pts[si].y + (pts[si + 1].y - pts[si].y) * e };
      gripping[move.k] = false;
    } else if (move.kind === 'release') {
      for (let k = 0; k < 2; k++) { lim.hand[k] = null; lim.grip[k] = false; hands[k] = null; gripping[k] = false; }
    }
    const goalNow = pelvisFor(move);
    const pr = smooth(Math.min(1, move.t / Math.max(0.2, move.kind === 'foot' ? move.dur * p.shift : move.dur * 0.6)));
    pelvisGoal = { x: pelvisFrom.x + (goalNow.x - pelvisFrom.x) * pr, y: pelvisFrom.y + (goalNow.y - pelvisFrom.y) * pr };
    const goal = { pelvis: pelvisGoal, torso: leanNow, feet, hands, gripping, planted, torsoGain: p.torsoGain };
    const actual = pose.actualOf(world);
    const support = move.kind === 'foot' ? lim.foot[1 - move.k].x : undefined;
    const jt = scrambling() ? pose.jointTargets(goal, actual) : pose.balancedTargets(goal, m, { planted, support, kHip: p.kHip, kAnkle: p.kAnkle, kAnkleVel: 0.15, kHipVel: 0.1 }, actual);
    pose.commandsFor(world, jt, tmp);
    for (let j = 0; j < J; j++) this.action[j] = tmp[j];
    for (let k = 0; k < 2; k++) {
      let near = false;
      if (move.kind === 'hand' && move.k === k && s > 0.6) { const hp = pose.handOf(world, k === 0 ? P.handLeft : P.handRight); near = Math.hypot(hp.x - move.target.x, hp.y - move.target.y) < 0.08; }
      this.action[J + k] = lim.grip[k] || near ? 1 : -1;
    }
    if (s >= 1) {
      if (move.kind === 'foot') {
        const so = pose.soleOf(world, move.k === 0 ? P.footLeft : P.footRight);
        if (Math.abs(so.y - move.target.y) < 0.08 && Math.abs(so.x - move.target.x) < 0.12) { lim.foot[move.k] = { x: so.x, y: ground(so.x) }; move = null; stuck = 0; }
        else if ((move.wait += 1 / HZ) > 0.4) { lim.foot[move.k] = { x: so.x, y: Math.max(so.y, ground(so.x)) }; move = null; stuck++; }
      } else if (move.kind === 'hand') {
        const g = this.grips[move.k];
        if (g && g.joint) { const hp = pose.handOf(world, move.k === 0 ? P.handLeft : P.handRight); lim.hand[move.k] = { x: hp.x, y: hp.y }; lim.grip[move.k] = true; move = null; stuck = 0; }
        else if ((move.wait += 1 / HZ) > 0.5) { move = null; stuck++; }
      } else move = null;
    }
    orig();
  })(ep.updateGrip.bind(ep));
  let frames = 0;
  const x0 = world.px[0];
  while (!ep.advanceOneFrame()) {
    frames++;
    const m = ep.measure;
    const px = world.px[0];
    if (record && frames % 2 === 0) {
      const pz = [];
      for (let k = 0; k < world.parts.length; k++) pz.push(+world.px[k].toFixed(3), +world.py[k].toFixed(3), +Math.atan2(world.s[k], world.c[k]).toFixed(3));
      recFrames.push({ p: pz, up: m.torsoCos > 0.7 ? 1 : 0, gl: ep.grips[0].joint ? 1 : 0, gr: ep.grips[1].joint ? 1 : 0 });
    }
    const headAbove = m.headHeight - ground(world.px[2]);
    if (headAbove < 0.5 || m.torsoCos < 0) { fell = true; break; }
    best = Math.max(best, Math.min(lim.foot[0].y, lim.foot[1].y));
    const onTop = lim.foot.every((f) => f.x > summitX + 0.1) && m.torsoCos > 0.8 && headAbove > 1.3;
    upTop = onTop ? upTop + 1 / HZ : 0;
    if (upTop > 1) { topped = true; break; }
    if (stuck > 40) break;
  }
  const T = frames / HZ;
  const score = 6 * best / H + (fell ? -1 : 0) + (topped ? 4 + 2 * (1 - T / seconds) : 0);
  const why = (topped ? 'TOPPED ' : '') + 'feet up to ' + best.toFixed(2) + '/' + H.toFixed(2) + ' m in ' + T.toFixed(1) + ' s' + (fell ? ' FELL' : '') + (stuck > 40 ? ' stuck' : '');
  const status = topped ? 'reached the top (' + H.toFixed(1) + ' m) in ' + T.toFixed(0) + ' s' : (fell ? 'fell' : 'stopped') + ' at ' + best.toFixed(1) + ' of ' + H.toFixed(1) + ' m';
  return { score, pass: topped ? 1 : 0, why, status, frames: recFrames, surface: S.map(([x, y]) => [+x.toFixed(3), +y.toFixed(3)]) };
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
      eps.push({ seed: i, surface: r.surface, box: null, walk: null, board: null, status: r.status, frames: r.frames });
    }
    fs.writeFileSync(rec, JSON.stringify({ skill: 6, fps: HZ / 2, shapes, episodes: eps }));
  }
  process.exit(0);
}

const rand = makeSeededRandom(+arg('--rng', 4321));
const mean = new Float64Array(SPEC.length);
if (init) { const sv = JSON.parse(fs.readFileSync(init, 'utf8')).best; for (let d = 0; d < SPEC.length && d < sv.length; d++) mean[d] = sv[d]; }
const sigma = Float64Array.from(SPEC.map((q) => q[2]));
let bestC = null;
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
      if (!bestC || val.score > bestC.score) bestC = { v: Float64Array.from(c.v), ...val };
    }
    fs.writeFileSync(out, JSON.stringify({ height, difficulty, best: Array.from(bestC.v), params: decode(bestC.v), score: bestC.score, pass: bestC.pass }));
    console.log('iter', it, 'held-out best', bestC.score.toFixed(3), 'pass', bestC.pass + '/6', 'gen best', cands[0].score.toFixed(3));
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
