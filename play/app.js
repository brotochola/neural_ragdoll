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
import { makeMountain, makeWall } from '../src/terrain.js';
import standNet from '../trained/stand-and-walk.json';

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
let camX = 0, camY = 0;

function groundAt(x) {
  let h = 0;
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
  const r = makeSeededRandom(state.seed);
  const start = sim.fillTrainingStart({}, r, evalMode(6));
  start.tilt = 0;
  start.walk = null;
  start.frames = 1e9;
  start.shiftX = 0;
  terrain = { kind: state.scene };
  if (state.scene === 'wall') start.terrain = makeWall();
  else if (state.scene === 'mountain') { const m = makeMountain(r, 3, 0.5); start.terrain = { statics: [], polys: m.polys, surface: m.surface }; }
  else start.terrain = null;
  world.rebuild();
  ep = new sim.Episode(world, weights, 0, 0, Object.assign({ phase: 0 }, start));
  if (ep.terrain && ep.terrain.statics && ep.terrain.statics.length && state.scene === 'wall') {
    const w = ep.terrain.statics[0];
    terrain.wall = { x0: w.x0, top: w.top };
    terrain.rungs = ep.terrain.holds.map((h) => ({ x0: h.x0, x1: h.x1, y: h.y }));
  }
  if (ep.terrain && ep.terrain.surface) terrain.surface = ep.terrain.surface;
  const pv = pose.pelvisOf(world);
  targets.hips = { x: pv.x, y: pv.y };
  targets.feet = [P.footLeft, P.footRight].map((q) => { const s = pose.soleOf(world, q); return { x: s.x, y: s.y }; });
  targets.hands = [null, null];
  state.grip = [false, false];
  state.autoHips = true;
  syncButtons();
  camX = pv.x; camY = 0;
  ep.updateGrip = ((orig) => function () { drive(this); orig(); })(ep.updateGrip.bind(ep));
}

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
  const jt = held[0] || held[1] || !state.balance ? pose.jointTargets(goal, actual) : pose.balancedTargets(goal, m, { planted, kHip: 1.0, kAnkle: 2.0, kAnkleVel: 0.15, kHipVel: 0.1 }, actual);
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
  }
  // lines from each limb to its mark
  ctx.setLineDash([5, 5]); ctx.lineWidth = 1.5;
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
  for (const mk of MARKS) {
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
cv.addEventListener('pointerdown', (ev) => { const p = evPos(ev); dragging = pick(p.x, p.y); if (dragging) { cv.setPointerCapture(ev.pointerId); ev.preventDefault(); } });
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
}
for (const k of [0, 1]) $('grip' + k).onclick = () => { state.grip[k] = !state.grip[k]; syncButtons(); };
$('autohips').onclick = () => { state.autoHips = !state.autoHips; syncButtons(); };
$('balance').onclick = () => { state.balance = !state.balance; syncButtons(); };
$('pause').onclick = () => { state.paused = !state.paused; syncButtons(); };
$('reset').onclick = () => reset();
$('dropHands').onclick = () => { targets.hands = [null, null]; state.grip = [false, false]; syncButtons(); };
$('newMountain').onclick = () => { state.seed++; state.scene = 'mountain'; reset(); };
$('lean').oninput = (ev) => { state.lean = -(+ev.target.value) * Math.PI / 180; $('leanVal').textContent = (+ev.target.value) + '°'; };
for (const b of document.querySelectorAll('[data-scene]')) b.onclick = () => { state.scene = b.dataset.scene; reset(); };
window.addEventListener('keydown', (ev) => { if (ev.key === 'r' || ev.key === 'R') reset(); if (ev.key === ' ') { state.paused = !state.paused; syncButtons(); ev.preventDefault(); } });

function resize() {
  const r = cv.getBoundingClientRect();
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  cv.width = Math.round(r.width * dpr); cv.height = Math.round(r.height * dpr);
  scale = cv.height / 4.2;
}
window.addEventListener('resize', resize);

// ---------- loop ----------
let last = performance.now(), acc = 0;
function loop(now) {
  const dt = Math.min(0.1, (now - last) / 1000); last = now;
  if (!state.paused) {
    acc += dt * HZ;
    let n = 0;
    while (acc >= 1 && n < 6) { ep.advanceOneFrame(); acc -= 1; n++; }
    if (n === 6) acc = 0;
  }
  draw();
  $('hud').textContent = `cabeza ${ep.measure.headHeight.toFixed(2)} m · agarres ${(ep.grips[0].joint ? 1 : 0) + (ep.grips[1].joint ? 1 : 0)} · t ${(ep.frame / HZ).toFixed(1)} s`;
  requestAnimationFrame(loop);
}
resize();
reset();
$('loading').hidden = true;
// for poking at it from the console
window.ragdoll = { targets, state, world, toScreen: (x, y) => { const s = toScreen(x, y); const r = cv.getBoundingClientRect(); return { x: r.left + s.x * r.width / cv.width, y: r.top + s.y * r.height / cv.height }; }, markPos: (id) => markPos(MARKS.find((m) => m.id === id)) };
requestAnimationFrame(loop);
