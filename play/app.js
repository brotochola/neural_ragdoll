// PLAYGROUND: the physics runs live in the browser and you are the planner. Drag the five marks (hips,
// both hands, both feet); the pose executor (src/pose.js) turns them into joint angles with inverse
// kinematics and balance, and the motors move the body there. A hand with "grip" on closes on whatever it
// touches (the floor of the scene, a rung, the wall, the mountain). No external forces.
// Built into one page by play/build.sh.
import * as Box2D from 'https://cdn.jsdelivr.net/npm/phaser-box2d@1.1.0/dist/PhaserBox2D.min.js';
import { config, setSkill } from '../config.js';
import { createSim } from '../src/sim.js';
import { ElmanNetwork } from '../src/net.js';
import { makeSeededRandom } from '../src/utils.js';
import { evalMode } from '../src/training.js';
import * as pose from '../src/pose.js';
import { makeCrag, makeWall, makeRopeCourse } from '../src/terrain.js';
import standNet from '../trained/stand-and-walk.json';
import params from './params.json';
import { makeController as makeWalker } from '../src/simbicon.js';
import { installRunner } from '../src/run-selector.js';
import { installWallPlanner } from '../src/wall-planner.js';
import { installCragPlanner } from '../src/crag-planner.js';
import { installRopeRunner } from '../src/rope-swing.js';
import { installObstacleRunner } from '../src/obstacle-runner.js';
import * as simbMod from '../src/simbicon.js';

setSkill(6);
const sim = createSim(Box2D);
const world = sim.build();
const HZ = config.physics.hz;
const J = config.net.jointCount;
const P = config.partIndex;
const weights = ElmanNetwork.copy(standNet.theta);

const $ = (id) => document.getElementById(id);
const cv = $('stage');
const ctx = cv.getContext('2d');
const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

const state = {
  scene: 'floor',
  mode: 'manual', // who drives: manual (the marks), stand (the network), walk, run, climb (a planner)
  speed: { walk: 0.9, run: 2.7 },
  grip: [false, false],
  autoHips: true,
  balance: true,
  lean: 0,
  paused: false,
  seed: 1,
};
let ep = null;
let terrain = null; // { kind, surface?: [[x,y]], wall?: {x0, top}, rungs?: [{x0,x1,y}] }
const targets = { hips: null, hands: [null, null], feet: [null, null] };
let tmp = new Float64Array(J + 2);
let camX = 0, camY = 0, startX = 0;

function groundAt(x) {
  let h = 0;
  if (terrain && terrain.blocks) for (const b of terrain.blocks) if (x >= b.x0 && x <= b.x1) h = Math.max(h, b.top);
  if (terrain && terrain.wall && x >= terrain.wall.x0) h = Math.max(h, terrain.wall.top);
  if (terrain && terrain.surface) {
    const S = terrain.surface;
    for (let i = 0; i < S.length - 1; i++) if (x >= S[i][0] && x <= S[i + 1][0]) {
      const [x1, y1] = S[i], [x2, y2] = S[i + 1];
      h = Math.max(h, x2 > x1 ? y1 + (y2 - y1) * (x - x1) / (x2 - x1) : Math.max(y1, y2));
    }
  }
  return h;
}

function reset() {
  planner = null; summitShown = false; chain = null; ropeHole = null;
  const r = makeSeededRandom(state.seed);
  const start = sim.fillTrainingStart({}, r, evalMode(6));
  start.tilt = 0;
  start.walk = null;
  start.frames = 1e9;
  start.shiftX = 0;
  terrain = { kind: state.scene };
  let crag = null, lift = 0, holesRel = [];
  ropeCourse = null;
  if (state.scene === 'wall') start.terrain = makeWall();
  else if (state.scene === 'mountain') { crag = makeCrag(r, 2.6, 0.55 + r() * 0.08); start.terrain = { statics: [], polys: crag.polys, surface: crag.surface }; }
  else if (state.scene === 'rope') {
    // a hole with a rope over it, the floor raised 0.6 m so the hole is a real drop
    ropeCourse = makeRopeCourse(r, { gapMin: 1.4, gapMax: 2.2, lift: 2.0, low: 1.25 });
    lift = ropeCourse.lift;
    start.terrain = { statics: ropeCourse.statics, ropes: ropeCourse.ropes };
  } else if (state.scene === 'obstacle') {
    // a 2 m rock 9-12 m ahead, a long top to run on
    crag = makeCrag(r, 2.0, 9 + r() * 3, { plateau: 14 });
    start.terrain = { statics: [], polys: crag.polys, surface: crag.surface };
  }
  else if (state.scene === 'holes') {
    // a floor 0.6 m up with holes 0.5-0.9 m wide every 3-5 m, as in parkour.js
    lift = 0.6;
    const statics = [];
    let x = -3, next = 4 + r() * 1.5;
    for (let i = 0; i < 10; i++) { statics.push({ dx: x, w: next - x, h: lift }); const w = 0.5 + r() * 0.4; holesRel.push({ x0: next, x1: next + w }); x = next + w; next = x + 3 + r() * 2; }
    statics.push({ dx: x, w: 40, h: lift });
    start.terrain = { statics };
  } else start.terrain = null;
  // running starts mid-stride (a recorded running state), lifted onto the floor of the scene
  let rs = null;
  if (state.mode === 'run') {
    rs = params.runStates[Math.floor(r() * params.runStates.length)];
    const pz = rs.pose.slice();
    for (let k = 0; k < world.parts.length; k++) pz[k * 3 + 1] += lift;
    start.startPose = pz;
    start.startVel = rs.vel.map((v, i) => (i % 3 === 0 ? v * state.speed.run / rs.speed : v));
    start.settle = 0; start.drop = 0;
  } else if (lift) {
    // standing on the raised floor: the standing pose lifted
    const pz = [];
    world.rebuild();
    const tmpEp = new sim.Episode(world, weights, 0, 0, Object.assign({ phase: 0 }, sim.fillTrainingStart({}, makeSeededRandom(1), evalMode(6)), { tilt: 0, shiftX: 0 }));
    for (let k = 0; k < world.parts.length; k++) pz.push(world.px[k], world.py[k] + lift, Math.atan2(world.s[k], world.c[k]));
    start.startPose = pz; start.settle = 0; start.drop = 0;
  }
  world.rebuild();
  ep = new sim.Episode(world, weights, 0, 0, Object.assign({ phase: 0 }, start));
  if (state.scene === 'rope') terrain.blocks = ep.terrain.statics.map((b) => ({ x0: b.x0, x1: b.x1, top: b.top }));
  if (state.scene === 'holes') {
    terrain.blocks = ep.terrain.statics.map((b) => ({ x0: b.x0, x1: b.x1, top: b.top }));
    const mid = (world.px[P.footLeft] + world.px[P.footRight]) / 2;
    terrain.holes = holesRel.map((h) => ({ x0: mid + h.x0, x1: mid + h.x1 }));
  }
  if (ep.terrain && ep.terrain.statics && ep.terrain.statics.length && state.scene === 'wall') {
    const w = ep.terrain.statics[0];
    terrain.wall = { x0: w.x0, top: w.top };
    terrain.rungs = ep.terrain.holds.map((h) => ({ x0: h.x0, x1: h.x1, y: h.y }));
  }
  if (ep.terrain && ep.terrain.surface) terrain.surface = ep.terrain.surface;
  if (crag) { const dx = ep.terrain.surface[1][0] - crag.surface[1][0]; terrain.knobs = crag.knobs.map((k) => ({ lip: k.lip + dx, back: k.back + dx, y: k.y })); }
  const pv = pose.pelvisOf(world);
  targets.hips = { x: pv.x, y: pv.y };
  targets.feet = [P.footLeft, P.footRight].map((q) => { const s = pose.soleOf(world, q); return { x: s.x, y: s.y }; });
  targets.hands = [null, null];
  state.grip = [false, false];
  state.autoHips = true;
  syncButtons();
  camX = pv.x; camY = lift;
  startX = world.px[0];
  const pctx = { config, pose, Box2D };
  if (state.mode === 'manual') ep.updateGrip = ((orig) => function () { drive(this); orig(); })(ep.updateGrip.bind(ep));
  else if (state.mode === 'walk') {
    const walker = makeWalker(world);
    const wtmp = new Float64Array(J + 2);
    ep.updateGrip = ((orig) => function () {
      const m = this.measure; m.walkSpeed = state.speed.walk; m.walkDir = 1;
      walker(params.walk, this, wtmp);
      for (let j = 0; j < wtmp.length; j++) this.action[j] = j < J ? wtmp[j] : -1;
      orig();
    })(ep.updateGrip.bind(ep));
  } else if (state.mode === 'run') {
    const mid0 = (world.px[P.footLeft] + world.px[P.footRight]) / 2;
    if (state.scene === 'rope') {
      const hole = { x0: mid0 + ropeCourse.hole.x0, x1: mid0 + ropeCourse.hole.x1 };
      ropeHole = hole;
      terrain.blocks = ep.terrain.statics.map((b) => ({ x0: b.x0, x1: b.x1, top: b.top }));
      chain = installRopeRunner(ep, world, params.rope, { config, pose, Box2D, runnerP: params.runner, speed: () => 2.4, hole, rope: ep.terrain.ropes[0], lift, walkP: params.walkFast }); // 2.4 m/s: the speed it crosses best at
    } else if (state.scene === 'obstacle') {
      chain = installObstacleRunner(ep, world, { config, pose, Box2D, ElmanNetwork, crag, H: 2.0, speed: () => state.speed.run, stopAt: 0.6, vswitch: 1.3, params: { runP: params.runner, walkP: params.walkFast, slowP: params.slow, stopP: params.stop, stopK: params.stopK, cragP: params.cragObstacle, getup: params.getup } });
    } else installRunner(ep, world, params.runner, { config, speed: () => state.speed.run, obstacles: terrain.holes || [] });
    ep.runner = { stance: rs.stance, t: -0.1, flight: false, jumping: false, jumpNext: false, strides: 0 };
  } else if (state.mode === 'climb') {
    if (state.scene === 'wall') installWallPlanner(ep, world, params.wall, Object.assign({ wallH: 3.3 }, pctx));
    else if (state.scene === 'mountain') planner = installCragPlanner(ep, world, params.crag, Object.assign({ crag }, pctx));
    else ep.updateGrip = ((orig) => function () { drive(this); orig(); })(ep.updateGrip.bind(ep));
  }
  // stand: the stand-and-walk network drives (nothing to install)
  $('modeNote').textContent = (state.mode === 'run' && state.scene === 'rope' ? 'Corre, salta en el borde, agarra la soga en el aire, cruza el agujero, cae y sigue corriendo. En la simulación cruza y sigue corriendo cerca de la mitad de las veces. Si se cae, vuelve a empezar solo con otro agujero; si cruza, cuenta el cruce y prueba con otro.' : state.mode === 'run' && state.scene === 'obstacle' ? 'Corre, frena antes de la roca de 2 m, trepa, se arrastra sobre el borde, se levanta arriba y vuelve a correr. Tarda casi un minuto y en la simulación sale 4 de cada 10 veces (a veces se traba trepando).' : MODE_NOTE[state.mode]) + (state.mode === 'climb' && !['wall', 'mountain'].includes(state.scene) ? ' Elegí la pared o la montaña.' : '');
}
const MODE_NOTE = {
  manual: 'Vos sos el planificador: arrastrá las marcas.',
  stand: 'La red neuronal de pararse mantiene el equilibrio.',
  walk: 'El controlador de caminata (estilo SIMBICON) camina hacia la derecha.',
  run: 'El controlador de carrera corre y salta los agujeros solo. Arranca ya corriendo.',
  climb: 'El planificador elige dónde va cada mano y cada pie.',
};

// One control step: goal from the marks, joint commands from the pose executor, grips from the toggles.
function drive(e) {
  const m = e.measure;
  const feet = targets.feet.map((f) => ({ x: f.x, y: f.y, angle: 0 }));
  // a foot counts as standing when its mark is on the ground and it is not being dragged
  const planted = targets.feet.map((f, k) => Math.abs(f.y - groundAt(f.x)) < 0.06 && dragging !== 'feet' + k);
  const held = [0, 1].map((k) => state.grip[k] && !!(e.grips[k] && e.grips[k].joint));
  if (state.autoHips) {
    // hips over the planted feet (between them), pulled toward hands that are holding on, at standing
    // height over the ground, no higher than the legs reach
    const pf = [0, 1].filter((k) => planted[k]);
    let x = pf.length ? pf.reduce((a, k) => a + targets.feet[k].x, 0) / pf.length : targets.hips.x;
    const hh = [0, 1].filter((k) => held[k] && targets.hands[k]);
    if (hh.length) x += 0.35 * (hh.reduce((a, k) => a + targets.hands[k].x, 0) / hh.length - x);
    let y = groundAt(x) + 0.98; // soles read 5 cm above the ground, so this is standing
    for (const k of [0, 1]) y = Math.min(y, targets.feet[k].y + 0.95);
    if (hh.length) y = Math.min(y, Math.max(...hh.map((k) => targets.hands[k].y)) - 0.3);
    targets.hips.x += (x - targets.hips.x) * 0.05;
    targets.hips.y += (y - targets.hips.y) * 0.05;
  }
  const goal = { pelvis: targets.hips, torso: state.lean, feet, hands: targets.hands, gripping: held, planted, torsoGain: 0.4 };
  const actual = pose.actualOf(world);
  const jt = held[0] || held[1] || !state.balance ? pose.jointTargets(goal, actual) : pose.balancedTargets(goal, m, { planted, kHip: 0.3, kAnkle: 0.8, kAnkleVel: 0.4, kHipVel: 0.2 }, actual);
  pose.commandsFor(world, jt, tmp);
  for (let j = 0; j < J; j++) e.action[j] = tmp[j];
  // "agarra": the hand closes once it is at its mark (closing on the way caught whatever it brushed),
  // and stays closed while the button is on
  for (let k = 0; k < 2; k++) {
    const t = targets.hands[k];
    const hp = pose.handOf(world, k ? P.handRight : P.handLeft);
    const near = t && Math.hypot(hp.x - t.x, hp.y - t.y) < 0.15 && dragging !== 'hands' + k;
    e.action[J + k] = state.grip[k] && (held[k] || near) ? 1 : -1;
  }
}

// ---------- drawing ----------
let scale = 120; // px per metre
function toScreen(x, y) { return { x: cv.width / 2 + (x - camX) * scale, y: cv.height * 0.82 - (y - camY) * scale }; }
function toWorld(px, py) { return { x: (px - cv.width / 2) / scale + camX, y: (cv.height * 0.82 - py) / scale + camY }; }
function partColor(i) {
  if (i === 2 || i === 5 || i === 11) return css('--skin');
  if (i === 0 || i === 1 || i === 15) return css('--shirt');
  if (i === 8 || i === 14) return css('--ink');
  if (i === 6 || i === 7) return css('--jean-back');
  if (i === 12 || i === 13) return css('--jean');
  if (i === 3 || i === 4) return css('--shirt-back');
  return css('--shirt');
}
function drawPart(i) {
  const part = world.parts[i];
  const a = Math.atan2(world.s[i], world.c[i]);
  const p = toScreen(world.px[i], world.py[i]);
  const k = scale;
  ctx.save();
  ctx.translate(p.x, p.y); ctx.rotate(-a);
  ctx.fillStyle = ctx.strokeStyle = partColor(i);
  if (part.circle) { ctx.beginPath(); ctx.arc(0, 0, part.r * k, 0, Math.PI * 2); ctx.fill(); }
  else if (part.capsule) {
    ctx.lineCap = 'round'; ctx.lineWidth = part.r * 2 * k;
    ctx.beginPath(); ctx.moveTo(-(part.ax || 0) * k, (part.ay || 0) * k); ctx.lineTo((part.ax || 0) * k, -(part.ay || 0) * k); ctx.stroke();
    if (!part.ax && !part.ay) { ctx.beginPath(); ctx.arc(0, 0, part.r * k, 0, Math.PI * 2); ctx.fill(); }
  } else {
    const w = part.hx * 2 * k, h = part.hy * 2 * k;
    ctx.beginPath(); ctx.roundRect(-w / 2, -h / 2, w, h, Math.min(w, h) * 0.35); ctx.fill();
  }
  ctx.restore();
}
const MARKS = [
  { id: 'hips', label: 'cadera', get: () => targets.hips, color: '--mark-hips' },
  { id: 'hands0', label: 'mano izq', get: () => targets.hands[0], color: '--mark-hand' },
  { id: 'hands1', label: 'mano der', get: () => targets.hands[1], color: '--mark-hand' },
  { id: 'feet0', label: 'pie izq', get: () => targets.feet[0], color: '--mark-foot' },
  { id: 'feet1', label: 'pie der', get: () => targets.feet[1], color: '--mark-foot' },
];
const BODY_OF = { hips: () => pose.pelvisOf(world), hands0: () => pose.handOf(world, P.handLeft), hands1: () => pose.handOf(world, P.handRight), feet0: () => pose.soleOf(world, P.footLeft), feet1: () => pose.soleOf(world, P.footRight) };
function markPos(mk) { return mk.get() || BODY_OF[mk.id](); }

function draw() {
  const W = cv.width, H = cv.height;
  const pv = pose.pelvisOf(world);
  camX += (pv.x - camX) * 0.05;
  const gy = Math.max(0, Math.min(world.py[P.footLeft], world.py[P.footRight]) - 0.3);
  camY += (gy - camY) * 0.05;
  ctx.fillStyle = css('--stage'); ctx.fillRect(0, 0, W, H);
  // floor and ruler
  const g0 = toScreen(0, 0).y;
  ctx.fillStyle = css('--rock'); ctx.fillRect(0, g0, W, H - g0);
  ctx.strokeStyle = css('--rule'); ctx.lineWidth = 1; ctx.fillStyle = css('--muted');
  ctx.font = `${Math.max(10, Math.round(H * 0.022))}px ${css('--mono')}`; ctx.textAlign = 'center';
  const left = toWorld(0, 0).x, right = toWorld(W, 0).x;
  for (let m = Math.ceil(left); m <= right; m++) { const s = toScreen(m, 0); ctx.beginPath(); ctx.moveTo(s.x, g0); ctx.lineTo(s.x, g0 + 8); ctx.stroke(); ctx.fillText(m + ' m', s.x, g0 + 22); }
  // scene
  ctx.fillStyle = css('--rock'); ctx.strokeStyle = css('--rock-edge'); ctx.lineWidth = 2;
  if (terrain.blocks) for (const b of terrain.blocks) {
    const a = toScreen(b.x0, b.top), c = toScreen(b.x1, -1);
    ctx.fillRect(a.x, a.y, c.x - a.x, c.y - a.y); ctx.strokeRect(a.x, a.y, c.x - a.x, c.y - a.y);
  }
  if (terrain.wall) {
    const a = toScreen(terrain.wall.x0, terrain.wall.top), b = toScreen(terrain.wall.x0 + 3, 0);
    ctx.fillRect(a.x, a.y, b.x - a.x, b.y - a.y); ctx.strokeRect(a.x, a.y, b.x - a.x, b.y - a.y);
    ctx.fillStyle = css('--rock-edge');
    for (const h of terrain.rungs) { const s = toScreen(h.x0, h.y + 0.02); ctx.fillRect(s.x, s.y, (h.x1 - h.x0) * scale, 0.04 * scale); }
  }
  if (terrain.surface) {
    ctx.beginPath();
    const S = terrain.surface;
    let s0 = toScreen(S[0][0], -1); ctx.moveTo(s0.x, s0.y);
    for (const [x, y] of S) { const s = toScreen(x, y); ctx.lineTo(s.x, s.y); }
    const sl = toScreen(S[S.length - 1][0], -1); ctx.lineTo(sl.x, sl.y); ctx.closePath();
    ctx.fill(); ctx.stroke();
    if (terrain.knobs) {
      // the rock knobs (wedges sticking out of the face)
      for (const k of terrain.knobs) {
        const pts = [[k.lip, k.y], [k.lip + 0.03, k.y - 0.04], [k.back + 0.03, k.y - 0.07], [k.back + 0.03, k.y]];
        ctx.beginPath(); pts.forEach(([x, y], i) => { const q = toScreen(x, y); i ? ctx.lineTo(q.x, q.y) : ctx.moveTo(q.x, q.y); }); ctx.closePath(); ctx.fill(); ctx.stroke();
      }
    }
  }
  if (ep && ep.terrain && ep.terrain.ropes && ep.terrain.ropes.length) {
    // the rope: anchor, then the segment centres
    for (const rp of ep.terrain.ropes) {
      ctx.strokeStyle = css('--rope'); ctx.lineWidth = Math.max(2, 0.04 * scale); ctx.lineCap = 'round';
      ctx.beginPath(); let q = toScreen(rp.x, rp.top); ctx.moveTo(q.x, q.y);
      for (const id of rp.ids) { const v = Box2D.b2Body_GetPosition(id); q = toScreen(v.x, v.y); ctx.lineTo(q.x, q.y); }
      const last = Box2D.b2Body_GetPosition(rp.ids[rp.ids.length - 1]), rot = Box2D.b2Body_GetRotation ? Box2D.b2Body_GetRotation(rp.ids[rp.ids.length - 1]) : null;
      if (rot) { q = toScreen(last.x + rot.s * rp.seg / 2, last.y - rot.c * rp.seg / 2); ctx.lineTo(q.x, q.y); }
      ctx.stroke();
      const a = toScreen(rp.x - 0.15, rp.top + 0.06); ctx.fillStyle = css('--rock-edge'); ctx.fillRect(a.x, a.y, 0.3 * scale, 0.06 * scale);
    }
  }
  const showMarks = state.mode === 'manual' || (state.mode === 'climb' && !['wall', 'mountain'].includes(state.scene));
  // lines from each limb to its mark
  ctx.setLineDash([5, 5]); ctx.lineWidth = 1.5;
  if (showMarks)
  for (const mk of MARKS) {
    const t = mk.get(); if (!t) continue;
    const b = BODY_OF[mk.id](); const s1 = toScreen(b.x, b.y), s2 = toScreen(t.x, t.y);
    ctx.strokeStyle = css(mk.color); ctx.beginPath(); ctx.moveTo(s1.x, s1.y); ctx.lineTo(s2.x, s2.y); ctx.stroke();
  }
  ctx.setLineDash([]);
  // the body (far side first)
  for (const i of [6, 7, 8, 3, 4, 5, 15, 0, 1, 2, 12, 13, 14, 9, 10, 11]) drawPart(i);
  // grip rings
  for (const [k, part] of [[0, P.handLeft], [1, P.handRight]]) if (ep.grips[k] && ep.grips[k].joint) { const s = toScreen(world.px[part], world.py[part]); ctx.strokeStyle = css('--ok'); ctx.lineWidth = 3; ctx.beginPath(); ctx.arc(s.x, s.y, 0.07 * scale, 0, Math.PI * 2); ctx.stroke(); }
  // the marks
  ctx.textAlign = 'left';
  if (showMarks) for (const mk of MARKS) {
    const p = markPos(mk); const s = toScreen(p.x, p.y); const on = !!mk.get();
    ctx.fillStyle = on ? css(mk.color) : 'transparent'; ctx.strokeStyle = css(mk.color); ctx.lineWidth = 2.5;
    ctx.beginPath(); ctx.arc(s.x, s.y, mk.id === 'hips' ? 11 : 9, 0, Math.PI * 2); if (on) ctx.fill(); ctx.stroke();
    ctx.fillStyle = css('--ink'); ctx.fillText(mk.label, s.x + 13, s.y - 9);
  }
}

// ---------- input: drag the marks ----------
let dragging = null;
function pick(px, py) {
  let best = null, bd = 28;
  for (const mk of MARKS) { const p = markPos(mk); const s = toScreen(p.x, p.y); const d = Math.hypot(s.x - px, s.y - py); if (d < bd) { bd = d; best = mk.id; } }
  return best;
}
function setMark(id, w) {
  if (id === 'hips') { targets.hips = w; state.autoHips = false; syncButtons(); }
  else if (id.startsWith('hands')) targets.hands[+id.slice(5)] = w;
  else targets.feet[+id.slice(4)] = w;
}
function evPos(ev) { const r = cv.getBoundingClientRect(); return { x: (ev.clientX - r.left) * cv.width / r.width, y: (ev.clientY - r.top) * cv.height / r.height }; }
cv.addEventListener('pointerdown', (ev) => { if (state.mode !== 'manual') return; const p = evPos(ev); dragging = pick(p.x, p.y); if (dragging) { cv.setPointerCapture(ev.pointerId); ev.preventDefault(); } });
cv.addEventListener('pointermove', (ev) => { if (!dragging) return; const p = evPos(ev); setMark(dragging, toWorld(p.x, p.y)); });
const endDrag = () => { dragging = null; };
cv.addEventListener('pointerup', endDrag);
cv.addEventListener('pointercancel', endDrag);

// ---------- controls ----------
function syncButtons() {
  for (const k of [0, 1]) { const b = $('grip' + k); b.setAttribute('aria-pressed', state.grip[k]); b.textContent = (k ? 'Mano der: ' : 'Mano izq: ') + (state.grip[k] ? 'agarra' : 'suelta'); }
  $('autohips').setAttribute('aria-pressed', state.autoHips);
  $('balance').setAttribute('aria-pressed', state.balance);
  $('pause').textContent = state.paused ? 'Seguir' : 'Pausa';
  for (const b of document.querySelectorAll('[data-scene]')) b.setAttribute('aria-pressed', b.dataset.scene === state.scene);
  for (const b of document.querySelectorAll('[data-mode]')) b.setAttribute('aria-pressed', b.dataset.mode === state.mode);
  for (const el of document.querySelectorAll('.manual-only')) el.hidden = state.mode !== 'manual';
}
for (const k of [0, 1]) $('grip' + k).onclick = () => { state.grip[k] = !state.grip[k]; syncButtons(); };
$('autohips').onclick = () => { state.autoHips = !state.autoHips; syncButtons(); };
$('balance').onclick = () => { state.balance = !state.balance; syncButtons(); };
$('pause').onclick = () => { state.paused = !state.paused; syncButtons(); };
$('reset').onclick = () => reset();
$('dropHands').onclick = () => { targets.hands = [null, null]; state.grip = [false, false]; syncButtons(); };
$('newMountain').onclick = () => { state.seed++; state.scene = 'mountain'; reset(); };
$('lean').oninput = (ev) => { state.lean = -(+ev.target.value) * Math.PI / 180; $('leanVal').textContent = (+ev.target.value) + '°'; };
for (const b of document.querySelectorAll('[data-scene]')) b.onclick = () => {
  state.scene = b.dataset.scene; falls = 0; crossed = 0;
  // the rope and the 2 m rock are run courses: they only make sense with the runner driving
  if (['rope', 'obstacle'].includes(state.scene) && state.mode !== 'run') { state.mode = 'run'; falls = 0; }
  reset();
};
for (const b of document.querySelectorAll('[data-mode]')) b.onclick = () => {
  state.mode = b.dataset.mode;
  falls = 0;
  // the natural scene for each skill
  if (state.mode === 'run' && !['floor', 'holes', 'rope', 'obstacle'].includes(state.scene)) state.scene = 'holes';
  if (state.mode === 'climb' && !['wall', 'mountain'].includes(state.scene)) state.scene = 'mountain';
  if (state.mode === 'walk' && state.scene !== 'floor') state.scene = 'floor';
  reset();
};
window.addEventListener('keydown', (ev) => { if (ev.key === 'r' || ev.key === 'R') reset(); if (ev.key === ' ') { state.paused = !state.paused; syncButtons(); ev.preventDefault(); } });

function resize() {
  const r = cv.getBoundingClientRect();
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  cv.width = Math.round(r.width * dpr); cv.height = Math.round(r.height * dpr);
  scale = cv.height / 4.2;
}
window.addEventListener('resize', resize);

// ---------- loop ----------
let last = performance.now(), acc = 0, downFor = 0, holeFor = 0, falls = 0;
let planner = null, summitShown = false, ropeCourse = null, chain = null, ropeHole = null, crossed = 0;
function loop(now) {
  const dt = Math.min(0.1, (now - last) / 1000); last = now;
  if (!state.paused) {
    acc += dt * HZ;
    let n = 0;
    while (acc >= 1 && n < 6) { ep.advanceOneFrame(); acc -= 1; n++; }
    if (n === 6) acc = 0;
  }
  draw();
  // in the automatic modes a fall (head near the ground for 2 s) restarts the run
  const headUp = ep.measure.headHeight - groundAt(world.px[2]);
  downFor = headUp < 0.45 ? downFor + dt : 0;
  // crawling over the summit edge the head is low on purpose: not a fall
  if (planner && planner.phase) downFor = 0;
  if (chain && chain.planner && chain.planner.phase) downFor = 0;
  if (planner && planner.mantled && !summitShown) { summitShown = true; $('modeNote').textContent = '¡Llegó a la cima! Se arrastró sobre el borde y quedó acostado arriba. "Otra montaña" para probar con otra pared.'; }
  // hanging on the rope for good (no rule let go): counts as a fall too
  // or down in the hole (standing at the bottom is no fall by the head, but the run is over)
  const inHole = state.scene === 'rope' && ropeCourse && world.py[0] < ropeCourse.lift - 0.4 && !(ep.grips[0].joint || ep.grips[1].joint);
  holeFor = inHole ? holeFor + dt : 0;
  const stuck = (chain && chain.phase === 'swing' && chain.t > 4) || holeFor > 1;
  // across and running on 6 m past the hole: one more crossing, then a new hole (the floor ends at 30 m)
  if (ropeHole && chain && chain.phase === 'run2' && world.px[0] > ropeHole.x1 + 6) { crossed++; holeFor = 0; downFor = 0; state.seed++; reset(); }
  // a new course each restart (the same seed replayed the same fall exactly)
  if (state.mode !== 'manual' && (downFor > 2 || stuck)) { downFor = 0; holeFor = 0; falls++; state.seed++; reset(); }
  $('hud').textContent = `cabeza ${headUp.toFixed(2)} m sobre el suelo · agarres ${(ep.grips[0].joint ? 1 : 0) + (ep.grips[1].joint ? 1 : 0)} · avanzó ${(world.px[0] - startX).toFixed(1)} m · t ${(ep.frame / HZ).toFixed(1)} s` + (state.mode !== 'manual' && falls ? ` · caídas ${falls}` : '') + (state.scene === 'rope' && crossed ? ` · cruces ${crossed}` : '');
  requestAnimationFrame(loop);
}
resize();
reset();
$('loading').hidden = true;
// for poking at it from the console
window.ragdoll = { targets, state, world, toScreen: (x, y) => { const s = toScreen(x, y); const r = cv.getBoundingClientRect(); return { x: r.left + s.x * r.width / cv.width, y: r.top + s.y * r.height / cv.height }; }, markPos: (id) => markPos(MARKS.find((m) => m.id === id)) };
requestAnimationFrame(loop);
