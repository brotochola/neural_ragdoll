// A walking controller in the style of SIMBICON (Yin, Loken & van de Panne, 2007), searched with the
// cross-entropy method, as a teacher for a walk expert network. Two legs, a finite-state machine:
//   stance on one leg for T seconds (until the swing foot lands), then swap.
//   Swing hip: a target in WORLD frame, theta0 + cd*d + cv*v, where d is the centre of mass ahead of
//   the stance ankle and v its speed. This balance feedback is what keeps SIMBICON walkers up.
//   Stance hip: whatever holds the torso at a set lean. Knees and ankles: fixed targets per state.
// The trained walker scooted (one leg did every step); this alternates by construction.
//   node walk-simbicon.js --iters 60 --out trained/walk-simbicon.json
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
const iters = +arg('--iters', 60);
const out = arg('--out', 'trained/walk-simbicon.json');
const pop = +arg('--pop', 24);
const elite = +arg('--elite', 6);
const seeds = +arg('--seeds', 4);
const init = arg('--init', null);

const Box2D = await import(pathToFileURL(path.join(root, 'vendor/PhaserBox2D.min.js')).href);
const { createSim } = await import('./src/sim.js');
const { ElmanNetwork, outputCountOf } = await import('./src/net.js');
const { makeSeededRandom, randomNormalSample } = await import('./src/utils.js');
const { evalMode } = await import('./src/training.js');

setSkill(6);
const sim = createSim(Box2D);
const world = sim.build();
const weights = ElmanNetwork.copy(JSON.parse(fs.readFileSync(path.join(root, 'trained/stand-and-walk.json'), 'utf8')).theta);
const J = config.net.jointCount;
const OUT = outputCountOf();
const P = config.partIndex;
// Parts: 0 torso, 6/12 thighs, 7/13 shins, 8/14 feet. Joints: 0 waist, 1 neck, 2/8 shoulder,
// 3/9 elbow, 4/10 wrist, 5/11 hip, 6/12 knee, 7/13 ankle (left, right).
const LEG = [
  { thigh: 6, shin: 7, foot: P.footLeft, hip: 5, knee: 6, ankle: 7, shoulder: 2, elbow: 3 },
  { thigh: 12, shin: 13, foot: P.footRight, hip: 11, knee: 12, ankle: 13, shoulder: 8, elbow: 9 },
];

// Parameter vector -> controller. Names and starting values from the SIMBICON walk, in radians.
const SPEC = [
  ['T', 0.35, 0.1], // seconds per step
  ['swing0', 0.5, 0.25], // swing hip world angle, early part of the step
  ['swing1', 0.2, 0.25], // late part
  ['cd', 0.5, 0.3],
  ['cv', 0.2, 0.15],
  ['kneeSwing0', -1.1, 0.3],
  ['kneeSwing1', -0.1, 0.2],
  ['kneeStance', -0.05, 0.1],
  ['ankleSwing', 0.1, 0.15],
  ['ankleStance', 0.0, 0.15],
  ['lean', -0.05, 0.08], // torso world angle: negative leans forward (CCW is positive, facing +x)
  ['split', 0.5, 0.15], // share of the step in the early state
  ['arm', 0.3, 0.3], // arm swing amplitude
];
const dims = SPEC.length;
function decode(v) {
  const p = {};
  SPEC.forEach(([name, base], i) => (p[name] = base + v[i]));
  p.T = Math.max(0.15, Math.min(0.8, p.T));
  p.split = Math.max(0.1, Math.min(0.9, p.split));
  return p;
}

const angleOf = (i) => Math.atan2(world.s[i], world.c[i]);
function command(jointIndex, target) {
  const joint = world.joints[jointIndex];
  const t = Math.max(joint.lo, Math.min(joint.hi, target));
  return t >= 0 ? (joint.hi > 0 ? t / joint.hi : 0) : joint.lo < 0 ? t / -joint.lo : 0;
}

// The controller's state lives on the episode. Returns the 14 joint commands in `action`.
export function simbicon(p, ep, action) {
  const st = ep.simbicon || (ep.simbicon = { stance: 0, t: 0 });
  const dt = 1 / config.physics.hz;
  st.t += dt;
  const stance = LEG[st.stance];
  const swing = LEG[1 - st.stance];
  const m = ep.measure;
  if ((st.t >= p.T && m.touch[swing.foot]) || st.t >= 2 * p.T) {
    st.stance = 1 - st.stance;
    st.t = 0;
    return simbicon(p, ep, action);
  }
  const early = st.t < p.split * p.T;
  const torso = angleOf(0);
  const d = (m.centerX || 0) - world.px[stance.foot];
  const v = m.centerVelocityX || 0;
  // Swing hip: world-frame thigh target with balance feedback; joint angle = thigh - torso.
  const swingWorld = (early ? p.swing0 : p.swing1) + p.cd * d + p.cv * v;
  action[swing.hip] = command(swing.hip, swingWorld - torso);
  // Stance hip: hold the torso at the lean. Moving the joint by (torso - lean) turns the torso back.
  const stanceHip = angleOf(stance.thigh) - torso;
  action[stance.hip] = command(stance.hip, stanceHip + (torso - p.lean));
  action[swing.knee] = command(swing.knee, early ? p.kneeSwing0 : p.kneeSwing1);
  action[stance.knee] = command(stance.knee, p.kneeStance);
  // Ankles: foot angle relative to the floor.
  action[swing.ankle] = command(swing.ankle, p.ankleSwing - angleOf(swing.shin));
  action[stance.ankle] = command(stance.ankle, p.ankleStance - angleOf(stance.shin));
  action[0] = 0; // waist
  action[1] = 0; // neck
  // Arms swing against the legs.
  action[swing.shoulder] = command(swing.shoulder, -p.arm);
  action[stance.shoulder] = command(stance.shoulder, p.arm);
  action[swing.elbow] = command(swing.elbow, 0.3);
  action[stance.elbow] = command(stance.elbow, 0.3);
  action[4] = 0;
  action[10] = 0;
  for (let j = J; j < action.length; j++) action[j] = -1;
}

function episode(p, seed) {
  const r = makeSeededRandom(3000 + seed);
  const start = sim.fillTrainingStart({}, r, evalMode(6));
  world.rebuild();
  const ep = new sim.Episode(world, weights, start.tilt || 0, start.drop || 0, Object.assign({ phase: seed % 6 }, start));
  const delay = Math.round((start.walk.delay || 0) * config.physics.hz);
  const orig = ep.applyMotors.bind(ep);
  ep.applyMotors = function () {
    if (this.frame >= delay) simbicon(p, this, this.action);
    orig();
  };
  const feet = [P.footLeft, P.footRight];
  const up = [false, false];
  const leftAt = [0, 0];
  const steps = [0, 0];
  let flight = 0;
  let frames = 0;
  while (!ep.advanceOneFrame()) {
    if (ep.frame < delay) continue;
    frames++;
    if (!ep.measure.touch[feet[0]] && !ep.measure.touch[feet[1]]) flight++;
    for (let k = 0; k < 2; k++) {
      const down = !!ep.measure.touch[feet[k]];
      const x = world.px[feet[k]];
      if (!down && !up[k]) {
        up[k] = true;
        leftAt[k] = x;
      } else if (down && up[k]) {
        up[k] = false;
        if (x - leftAt[k] > 0.04) steps[k]++;
      }
    }
  }
  const s = ep.stats;
  const wf = s.windowFrames || 0;
  const share = wf > 0 ? s.windowStood / wf : 0;
  const walked = s.walkCmd > 0 ? Math.max(0, Math.min(1.3, s.walkDist / s.walkCmd)) : 0;
  const pass = !s.headHit && share >= config.walk.standShare && s.walkCmd > 0 && s.walkDist >= config.walk.passShare * s.walkCmd;
  const flightShare = flight / Math.max(1, frames);
  const lo = Math.min(steps[0], steps[1]);
  const hi = Math.max(steps[0], steps[1]);
  const alt = hi > 0 ? (lo / hi) * Math.min(1, lo / 3) : 0;
  const alive = Math.min(1, (ep.measure.headHeight || 0) / config.body.standingHeadMeters);
  const score = 3 * share + 2 * Math.min(1, walked) * (0.3 + 0.7 * share) + 2 * alt * alive + 2 * alive - 2 * flightShare + (pass ? 3 : 0);
  const why = 'stood ' + share.toFixed(2) + ' walked ' + (s.walkDist || 0).toFixed(2) + '/' + (s.walkCmd || 0).toFixed(2) + ' steps ' + steps.join('/') + ' flight ' + flightShare.toFixed(2) + ' head ' + (ep.measure.headHeight || 0).toFixed(2);
  return { score, pass: pass ? 1 : 0, alt, why };
}

function evaluate(v) {
  const p = decode(v);
  let s = 0;
  let pass = 0;
  let alt = 0;
  for (let i = 0; i < seeds; i++) {
    const r = episode(p, i);
    s += r.score;
    pass += r.pass;
    alt += r.alt;
  }
  return { score: s / seeds, pass, alt: alt / seeds };
}

if (args.includes('--show')) {
  const p = decode(JSON.parse(fs.readFileSync(init, 'utf8')).best);
  for (let i = 0; i < seeds; i++) {
    const r = episode(p, 100 + i);
    console.log('seed', i, 'pass', r.pass, 'alt', r.alt.toFixed(2), r.why);
  }
  process.exit(0);
}

const rand = makeSeededRandom(66);
const normal = () => randomNormalSample(rand);
const mean = new Float64Array(dims);
if (init) {
  const saved = JSON.parse(fs.readFileSync(init, 'utf8')).best;
  for (let d = 0; d < dims; d++) mean[d] = saved[d];
}
const sigma = Float64Array.from(SPEC.map((s) => s[2]));
const floor = SPEC.map((s) => s[2] * 0.15);
let best = null;
for (let it = 0; it < iters; it++) {
  const cands = [];
  for (let c = 0; c < pop; c++) {
    const v = new Float64Array(dims);
    for (let d = 0; d < dims; d++) v[d] = mean[d] + sigma[d] * normal();
    cands.push({ v, ...evaluate(v) });
  }
  cands.sort((a, b) => b.score - a.score);
  if (!best || cands[0].score > best.score) best = cands[0];
  const top = cands.slice(0, elite);
  for (let d = 0; d < dims; d++) {
    let m = 0;
    for (const c of top) m += c.v[d];
    m /= top.length;
    let s = 0;
    for (const c of top) s += (c.v[d] - m) ** 2;
    s = Math.sqrt(s / top.length);
    mean[d] = 0.7 * m + 0.3 * mean[d];
    sigma[d] = Math.max(floor[d], 0.7 * s + 0.3 * sigma[d]);
  }
  if (it % 5 === 0 || it === iters - 1) {
    console.log('iter', it, 'best', best.score.toFixed(3), 'pass', best.pass + '/' + seeds, 'alt', best.alt.toFixed(2), 'gen best', cands[0].score.toFixed(3));
    fs.writeFileSync(out, JSON.stringify({ best: Array.from(best.v), params: decode(best.v), score: best.score, pass: best.pass, alt: best.alt }));
  }
}
let fresh = 0;
for (let i = 0; i < 12; i++) fresh += episode(decode(best.v), 100 + i).pass;
fs.writeFileSync(out, JSON.stringify({ best: Array.from(best.v), params: decode(best.v), score: best.score, pass: best.pass, alt: best.alt, freshPass: fresh }));
console.log('done fresh pass', fresh + '/12', '->', out);
sim.destroyWorld(world);
