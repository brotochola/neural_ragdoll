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
import { makeCrag, makeWall, makeRopeCourse, makeDashCourse, makeTrailCourse, makePopLevel, buildSandbox, SANDBOX_PIECES } from '../src/terrain.js';
import { installPopDirector, installCourseDirector } from '../src/director.js';
import { installTrail } from '../src/trail.js';
import { makeTrailNet } from '../src/trail-net.js';
import standNet from '../trained/stand-and-walk.json';
import params from './params.json';
import { makeController as makeWalker } from '../src/simbicon.js';
import { installRunner } from '../src/run-selector.js';
import { makeDashNet } from '../src/dash-net.js';
import { installWallPlanner } from '../src/wall-planner.js';
import { installCragPlanner } from '../src/crag-planner.js';
import { installRopeRunner } from '../src/rope-swing.js';
import { installObstacleRunner } from '../src/obstacle-runner.js';
import * as simbMod from '../src/simbicon.js';
import { createInspector } from './inspect.js';

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
  scene: 'sandbox',
  mode: 'run', // who drives: manual (the marks), stand (the network), walk, run, climb (a planner)
  speed: { walk: 0.9, run: 2.7 },
  grip: [false, false],
  autoHips: true,
  balance: true,
  lean: 0,
  paused: false,
  seed: 1,
  timeScale: 1, // slow motion: 1, 0.5, 0.25, 0.1
  helpers: true, // draw what the controllers aim at on the stage
};
let ep = null;
let terrain = null; // { kind, surface?: [[x,y]], wall?: {x0, top}, rungs?: [{x0,x1,y}] }
const targets = { hips: null, hands: [null, null], feet: [null, null] };
// tele: what the controllers aimed at, for the inspector. The pose executor is wrapped so every planner that
// turns a goal (hips, hands, feet) into joint angles leaves that goal here; nothing else changes.
const tele = { goal: null, goalFrame: -1 };
const tapPose = Object.assign({}, pose, {
  jointTargets(goal, actual) { tele.goal = goal; tele.goalFrame = ep ? ep.frame : 0; return pose.jointTargets(goal, actual); },
  balancedTargets(goal, m, opts, actual) { tele.goal = goal; tele.goalFrame = ep ? ep.frame : 0; return pose.balancedTargets(goal, m, opts, actual); },
});
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
  const keepCam = state.scene === 'sandbox' && sb.editing ? [camX, camY] : null;
  sb.timeline = []; sb.tlLast = -1;
  planner = null; summitShown = false; chain = null; ropeHole = null; trailCourse = null; trailInfo = null; popLevel = null; popDir = null; stallX = -1e9; stallT = 0;
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
  } else if (state.scene === 'dash') {
    // random boxes and balls all the way (the course dash.js trains on, 200 m long)
    const dc = makeDashCourse(r, { length: 200 });
    start.terrain = { statics: dc.statics, balls: dc.balls };
  } else if (state.scene === 'pop') {
    // the Prince of Persia level: trail, a 2 m rock to leap at and climb, trail on top, a rope over a hole, trail
    popLevel = makePopLevel(r);
    lift = popLevel.lift; crag = popLevel.crag;
    start.terrain = { statics: popLevel.statics, balls: popLevel.balls, polys: popLevel.polys, surface: popLevel.surface, ropes: popLevel.ropes };
  } else if (state.scene === 'sandbox') {
    // your course (buildSandbox): the pieces in order of x, what they need put in by themselves
    sbLevel = buildSandbox(sb.pieces);
    sb.level = sbLevel;
    sb.pieces.forEach((p, i) => { if (sbLevel.spans[i]) p.x = +sbLevel.spans[i].x0.toFixed(2); });
    lift = sbLevel.lift;
    start.terrain = { statics: sbLevel.statics, balls: sbLevel.balls, polys: sbLevel.polys, surface: sbLevel.walls.length ? sbLevel.walls[0].crag.surface : null, ropes: sbLevel.ropes };
  } else if (state.scene === 'trail') {
    // the trail course: a raised floor that steps up and down, holes, boxes and balls (trail-es.js trains on it)
    trailCourse = makeTrailCourse(r, { length: 200 });
    lift = trailCourse.lift;
    start.terrain = { statics: trailCourse.statics, balls: trailCourse.balls };
  } else if (state.scene === 'leap') {
    // a 2 m rock 7-10 m ahead: run at it, jump and grab a knob in the air, climb on
    crag = makeCrag(r, 2.0, 7 + r() * 3, { plateau: 14 });
    start.terrain = { statics: [], polys: crag.polys, surface: crag.surface };
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
    const v0 = state.scene === 'dash' ? params.dash.vRun || state.speed.run : state.scene === 'trail' || state.scene === 'pop' || state.scene === 'sandbox' ? params.trail.vRun || 2.6 : state.scene === 'leap' && params.leap ? params.leap.vRun : state.speed.run;
    start.startVel = rs.vel.map((v, i) => (i % 3 === 0 ? v * v0 / rs.speed : v));
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
  if (state.scene === 'dash' || state.scene === 'trail' || state.scene === 'pop' || state.scene === 'sandbox') terrain.blocks = ep.terrain.statics.map((b) => ({ x0: b.x0, x1: b.x1, top: b.top }));
  // every rock polygon, absolute (the rocks, the knobs, the ledges in the pits)
  const lv0 = state.scene === 'pop' ? popLevel : state.scene === 'sandbox' ? sbLevel : null;
  if (lv0 && lv0.statics.length && ep.terrain.statics.length) {
    const mid = ep.terrain.statics[0].x0 - lv0.statics[0].dx;
    terrain.polys = lv0.polys.map((q) => q.map(([x, y]) => [x + mid, y]));
    if (state.scene === 'sandbox') { sb.mid = mid; sbNewAttempt(); }
  }
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
  if (ep.terrain && ep.terrain.surface && ep.terrain.surface.length && state.scene !== 'sandbox') terrain.surface = ep.terrain.surface;
  if (crag && terrain.surface) { const dx = ep.terrain.surface[1][0] - crag.surface[1][0]; terrain.knobs = crag.knobs.map((k) => ({ lip: k.lip + dx, back: k.back + dx, y: k.y })); }
  const pv = pose.pelvisOf(world);
  targets.hips = { x: pv.x, y: pv.y };
  targets.feet = [P.footLeft, P.footRight].map((q) => { const s = pose.soleOf(world, q); return { x: s.x, y: s.y }; });
  targets.hands = [null, null];
  state.grip = [false, false];
  state.autoHips = true;
  syncButtons();
  camX = pv.x; camY = lift;
  if (keepCam) [camX, camY] = keepCam;
  startX = world.px[0];
  const pctx = { config, pose: tapPose, Box2D };
  tele.goal = null; tele.goalFrame = -1; rockX = terrain.surface ? (terrain.surface.find((q) => q[1] > 0.1) || [1e9])[0] : null;
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
      // the landing run after the rope gets the trail network trained for it (rope-es.js)
      ep.floorAt = () => lift; ep.groundAt = (x) => (x > hole.x0 && x < hole.x1 ? 0 : lift);
      dashCorrect = params.ropeNet ? makeTrailNet(world, config, Box2D) : null;
      chain = installRopeRunner(ep, world, params.rope, { config, pose: tapPose, Box2D, runnerP: params.runner, speed: () => 2.4, hole, rope: ep.terrain.ropes[0], lift, walkP: params.walkFast, net: dashCorrect ? { correct: dashCorrect, theta: params.ropeNet } : null }); // 2.4 m/s: the speed it crosses best at
    } else if (state.scene === 'obstacle') {
      chain = installObstacleRunner(ep, world, { config, pose: tapPose, Box2D, ElmanNetwork, crag, H: 2.0, speed: () => state.speed.run, stopAt: 0.6, vswitch: 1.3, params: { runP: params.runner, walkP: params.walkFast, slowP: params.slow, stopP: params.stop, stopK: params.stopK, cragP: params.cragObstacle, getup: params.getup } });
    } else if (state.scene === 'pop') {
      dashCorrect = params.trailNet ? makeTrailNet(world, config, Box2D) : null;
      popDir = installPopDirector(ep, world, popLevel, { config, pose: tapPose, Box2D, ElmanNetwork, params, trailNet: dashCorrect ? { correct: dashCorrect, theta: params.trailNet } : null });
      trailInfo = popDir;
    } else if (state.scene === 'sandbox') {
      dashCorrect = params.trailNet ? makeTrailNet(world, config, Box2D) : null;
      popDir = installCourseDirector(ep, world, sbLevel, { config, pose: tapPose, Box2D, ElmanNetwork, params, trailNet: dashCorrect ? { correct: dashCorrect, theta: params.trailNet } : null });
      trailInfo = popDir;
    } else if (state.scene === 'trail') {
      trailInfo = installTrail(ep, trailCourse, Box2D);
      dashCorrect = params.trailNet ? makeTrailNet(world, config, Box2D) : null;
      const vt = params.trail.vRun || 2.6;
      installRunner(ep, world, params.trail, { config, obstacles: trailInfo.obstacles, floorAt: trailInfo.floorAt, speed: () => vt, net: dashCorrect ? { correct: dashCorrect, theta: params.trailNet } : null });
    } else if (state.scene === 'leap') {
      chain = installObstacleRunner(ep, world, { config, pose: tapPose, Box2D, ElmanNetwork, crag, H: 2.0, speed: () => state.speed.run, stopAt: 0.6, vswitch: 1.3, leapP: params.leap, params: { runP: params.runner, walkP: params.walkFast, slowP: params.slow, stopP: params.stop, stopK: params.stopK, cragP: params.cragLeap || params.cragObstacle, getup: params.getup } });
    } else if (state.scene === 'dash') {
      // its own speed (vRun, tuned with the rest) and jumps scaled by each box's or ball's height
      const obstacles = ep.terrain.statics.map((b) => ({ x0: b.x0, x1: b.x1, top: b.top, kind: 'caja' }))
        .concat(ep.terrain.balls.map((b) => ({ live: () => { const q = Box2D.b2Body_GetPosition(b.id); return { x0: q.x - b.r, x1: q.x + b.r, top: q.y + b.r, kind: 'bola' }; } })));
      dashCorrect = params.dashNet ? makeDashNet(world, config, Box2D) : null;
      installRunner(ep, world, params.dash, { config, obstacles, net: dashCorrect ? { correct: dashCorrect, theta: params.dashNet } : null });
    } else installRunner(ep, world, params.runner, { config, speed: () => state.speed.run, obstacles: (terrain.holes || []).map((h) => Object.assign({ kind: 'agujero' }, h)) });
    ep.runner = { stance: rs.stance, t: -0.1, flight: false, jumping: false, jumpNext: false, strides: 0 };
  } else if (state.mode === 'climb') {
    if (state.scene === 'wall') planner = installWallPlanner(ep, world, params.wall, Object.assign({ wallH: 3.3 }, pctx));
    else if (state.scene === 'mountain') planner = installCragPlanner(ep, world, params.crag, Object.assign({ crag }, pctx));
    else ep.updateGrip = ((orig) => function () { drive(this); orig(); })(ep.updateGrip.bind(ep));
  }
  // stand: the stand-and-walk network drives (nothing to install)
  $('modeNote').textContent = (state.mode === 'run' && state.scene === 'rope' ? 'Corre, salta en el borde, agarra la soga en el aire, cruza el agujero, cae y sigue corriendo. En la simulación cruza y sigue corriendo unas 6 de cada 10 veces (una red ayuda a aterrizar y retomar la carrera). Si se cae, vuelve a empezar solo con otro agujero; si cruza, cuenta el cruce y prueba con otro.' : state.mode === 'run' && state.scene === 'dash' ? DASH_NOTE : state.mode === 'run' && state.scene === 'pop' ? POP_NOTE : state.mode === 'run' && state.scene === 'sandbox' ? SANDBOX_NOTE : state.mode === 'run' && state.scene === 'trail' ? TRAIL_NOTE : state.mode === 'run' && state.scene === 'leap' ? LEAP_NOTE : state.mode === 'run' && state.scene === 'obstacle' ? 'Corre, frena antes de la roca de 2 m, trepa, se arrastra sobre el borde, se levanta arriba y vuelve a correr. Tarda casi un minuto y en la simulación sale 4 de cada 10 veces (a veces se traba trepando).' : MODE_NOTE[state.mode]) + (state.mode === 'climb' && !['wall', 'mountain'].includes(state.scene) ? ' Elegí la pared o la montaña.' : '');
}
const DASH_NOTE = 'Corre a la derecha lo más rápido que puede sobre cajas y bolas al azar, saltándolas. Las bolas están sueltas: si las patea, ruedan. El corredor (números ajustados por búsqueda) lleva encima una red neuronal chica, entrenada con estrategias evolutivas en miles de recorridos al azar, que corrige caderas, rodillas, tobillos y cintura según cómo viene el cuerpo y qué tiene adelante. Si se cae, vuelve a empezar con otro recorrido.';
const POP_NOTE = 'Nivel tipo Prince of Persia: terreno con escalones, agujeros, cajas y bolas; baja al piso, corre hacia una roca de 2 m, salta, se agarra y trepa; arriba sigue corriendo sobre más obstáculos (los agujeros acá son pozos de 2 m) y cruza un agujero grande colgado de una soga. Un director en código le pasa el control a cada habilidad según la zona, y si se cae en el terreno lo hace levantarse (con la red de levantarse), caminar y seguir corriendo. Cada intento es un nivel distinto.';
const SANDBOX_NOTE = 'Armá tu pista: elegí una pieza en «Poner» y tocá la escena donde la querés. Lo que una pieza necesita se agrega solo (escalones para bajar al piso antes de una roca o para subir antes de un pozo o una soga, y lugar para tomar carrera). Con «▶ Correr» la prueba: un director en código le pasa el control a cada habilidad (correr y saltar con la red, saltar a la roca y trepar, la soga, salir de un pozo, levantarse) y sobre cada pieza se cuentan las pasadas y las caídas.';
const TRAIL_NOTE = 'Terreno libre: el piso sube y baja en escalones, tiene agujeros, cajas y bolas. Corre lo más rápido que puede sin caerse. El corredor lleva encima una red correctora que ve el perfil del terreno hasta casi 3 m adelante; en recorridos nuevos pasa el 87% de los obstáculos (el corredor solo, el 64%) y la mitad de las veces corre 12 s sin caerse. Si se cae, empieza otro recorrido.';
const LEAP_NOTE = 'Corre hacia una roca de 2 m sin frenar, salta y se agarra en el aire del saliente más alto que alcanza, se cuelga, trepa con el planificador, pasa el borde, se levanta y sigue corriendo. Si no llega a agarrarse, vuelve al método lento: camina, frena y trepa desde parado. En rocas nuevas se agarra casi siempre y hace todo el recorrido unas 7 de cada 10 veces.';
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
  const jt = held[0] || held[1] || !state.balance ? tapPose.jointTargets(goal, actual) : tapPose.balancedTargets(goal, m, { planted, kHip: 0.3, kAnkle: 0.8, kAnkleVel: 0.4, kHipVel: 0.2 }, actual);
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
  if (!(state.scene === 'sandbox' && sb.editing)) {
    camX += (pv.x - camX) * 0.05;
    const gy = Math.max(0, Math.min(world.py[P.footLeft], world.py[P.footRight]) - 0.3);
    camY += (gy - camY) * 0.05;
  }
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
  if (terrain.polys) {
    // the rock polygons: big ones filled, the small ones (knobs, ledges) outlined too
    ctx.fillStyle = css('--rock'); ctx.strokeStyle = css('--rock-edge'); ctx.lineWidth = 2;
    for (const q of terrain.polys) {
      ctx.beginPath(); q.forEach(([x, y], i) => { const t = toScreen(x, y); i ? ctx.lineTo(t.x, t.y) : ctx.moveTo(t.x, t.y); }); ctx.closePath(); ctx.fill();
      const w = Math.max(...q.map((v) => v[0])) - Math.min(...q.map((v) => v[0]));
      if (w < 0.3) ctx.stroke();
    }
    if (state.scene === 'sandbox') sbOutlineRocks();
  }
  if (state.scene === 'sandbox') sbDrawUnder();
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
  if (ep && ep.terrain && ep.terrain.balls && ep.terrain.balls.length) {
    // the loose balls
    ctx.fillStyle = css('--ball'); ctx.strokeStyle = css('--rock-edge'); ctx.lineWidth = 2;
    for (const b of ep.terrain.balls) {
      const q = Box2D.b2Body_GetPosition(b.id), rot = Box2D.b2Body_GetRotation(b.id), s = toScreen(q.x, q.y);
      ctx.beginPath(); ctx.arc(s.x, s.y, b.r * scale, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(s.x, s.y); ctx.lineTo(s.x + rot.c * b.r * scale, s.y - rot.s * b.r * scale); ctx.stroke();
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
  if (state.scene === 'sandbox') sbDrawOver();
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
cv.addEventListener('pointerdown', (ev) => { if (state.scene === 'sandbox') { if (sbDown(evPos(ev))) { cv.setPointerCapture(ev.pointerId); ev.preventDefault(); } return; } if (state.mode !== 'manual') return; const p = evPos(ev); dragging = pick(p.x, p.y); if (dragging) { cv.setPointerCapture(ev.pointerId); ev.preventDefault(); } });
cv.addEventListener('pointermove', (ev) => { if (state.scene === 'sandbox') { sbMove(evPos(ev)); return; } if (!dragging) return; const p = evPos(ev); setMark(dragging, toWorld(p.x, p.y)); });
const endDrag = (ev) => { dragging = null; if (state.scene === 'sandbox' && ev) sbUp(evPos(ev)); };
cv.addEventListener('pointerup', endDrag);
cv.addEventListener('pointercancel', () => { dragging = null; sb.drag = null; sb.sling = null; });
cv.addEventListener('pointerleave', () => { sb.hover = null; });
cv.addEventListener('wheel', (ev) => { if (state.scene === 'sandbox' && sb.editing) { camX += (ev.deltaY + ev.deltaX) / scale; ev.preventDefault(); } }, { passive: false });

// ---------- controls ----------
function syncButtons() {
  for (const k of [0, 1]) { const b = $('grip' + k); b.setAttribute('aria-pressed', state.grip[k]); b.textContent = (k ? 'Mano der: ' : 'Mano izq: ') + (state.grip[k] ? 'agarra' : 'suelta'); }
  $('autohips').setAttribute('aria-pressed', state.autoHips);
  $('balance').setAttribute('aria-pressed', state.balance);
  $('pause').textContent = state.paused ? 'Seguir' : 'Pausa';
  for (const b of document.querySelectorAll('[data-scene]')) b.setAttribute('aria-pressed', b.dataset.scene === state.scene);
  for (const b of document.querySelectorAll('[data-mode]')) b.setAttribute('aria-pressed', b.dataset.mode === state.mode);
  for (const el of document.querySelectorAll('.manual-only')) el.hidden = state.mode !== 'manual';
  if (typeof sbSyncUI === 'function' && sb) sbSyncUI();
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
  state.scene = b.dataset.scene; falls = 0; crossed = 0; bestDash = 0;
  // the rope and the 2 m rock are run courses: they only make sense with the runner driving
  if (['rope', 'obstacle', 'dash', 'trail', 'leap', 'pop', 'sandbox'].includes(state.scene) && state.mode !== 'run') { state.mode = 'run'; falls = 0; }
  if (state.scene !== 'sandbox' && sb.editing) { sb.editing = false; sb.zoom = 1; state.paused = false; resize(); }
  reset();
};
for (const b of document.querySelectorAll('[data-mode]')) b.onclick = () => {
  state.mode = b.dataset.mode;
  falls = 0;
  // the natural scene for each skill
  if (state.mode === 'run' && !['floor', 'holes', 'rope', 'obstacle', 'dash', 'trail', 'leap', 'pop', 'sandbox'].includes(state.scene)) state.scene = 'holes';
  if (state.mode !== 'run' && state.scene === 'sandbox') { state.scene = 'floor'; sb.editing = false; sb.zoom = 1; state.paused = false; resize(); }
  if (state.mode === 'climb' && !['wall', 'mountain'].includes(state.scene)) state.scene = 'mountain';
  if (state.mode === 'walk' && state.scene !== 'floor') state.scene = 'floor';
  reset();
};
for (const b of document.querySelectorAll('[data-speed]')) b.onclick = () => { state.timeScale = +b.dataset.speed; syncSpeed(); };
function syncSpeed() { for (const b of document.querySelectorAll('[data-speed]')) b.setAttribute('aria-pressed', +b.dataset.speed === state.timeScale); $('helpers').setAttribute('aria-pressed', state.helpers); }
$('step').onclick = () => { state.paused = true; syncButtons(); ep.advanceOneFrame(); };
$('helpers').onclick = () => { state.helpers = !state.helpers; syncSpeed(); };
window.addEventListener('keydown', (ev) => { if (state.scene === 'sandbox' && sbKey(ev)) return; if (ev.key === 'r' || ev.key === 'R') reset(); if (ev.key === ' ') { state.paused = !state.paused; syncButtons(); ev.preventDefault(); } });

function resize() {
  const r = cv.getBoundingClientRect();
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  cv.width = Math.round(r.width * dpr); cv.height = Math.round(r.height * dpr);
  scale = cv.height / (4.2 * (sb ? sb.zoom : 1));
}
window.addEventListener('resize', resize);

// ---------- inspector: who does what, right now ----------
const f2 = (v) => (v >= 0 ? '+' : '') + v.toFixed(2);
const m1 = (v) => v.toFixed(1) + ' m';
const SIDE = ['izquierdo', 'derecho'];
const MOTORS = { who: '14 motores', kind: 'body', detail: 'Cada articulación gira hacia su objetivo con fuerza y velocidad limitadas. Nada empuja al cuerpo desde afuera.' };
const RAIBERT = { who: 'Regla de colocación del pie', kind: 'tuned', detail: 'En el aire, la pierna adelantada apunta a un ángulo fijo más una corrección por la velocidad y por dónde está el cuerpo (estilo Raibert).' };
const RUNNER = { who: 'Corredor', kind: 'tuned', detail: 'Postura de cada fase de la zancada (apoyo, empuje, vuelo) y el tronco sostenido por la cadera de apoyo. 33 números ajustados por búsqueda automática.' };
const POSE_EXEC = (who) => ({ who: who || 'Ejecutor de poses', kind: 'rules', detail: 'Con la cadera, las manos y los pies que le piden, calcula cada ángulo con geometría de dos segmentos (cinemática inversa) y corrige el equilibrio con cadera y tobillos.' });
const NO_LIMBS = { who: 'Nadie', kind: 'none', detail: 'Esta parte no fija dónde van manos ni pies: los ángulos salen directo.' };
const ROPE_PHASES = [['run', 'Corre'], ['reach', 'Salta y estira los brazos'], ['swing', 'Colgado de la soga'], ['plant', 'Apoya los pies'], ['land', 'Aterriza corriendo'], ['drop', 'Cae'], ['steady', 'Se estabiliza'], ['walk', 'Camina'], ['run2', 'Vuelve a correr']].map(([id, label]) => ({ id, label }));
const OBST_PHASES = [['run', 'Corre'], ['walk', 'Camina'], ['stop', 'Frena'], ['hold', 'Quieto'], ['climb', 'Trepa'], ['walk2', 'Camina arriba'], ['run2', 'Vuelve a correr']].map(([id, label]) => ({ id, label }));
const LEAP_PHASES = [['run', 'Corre'], ['leap', 'Salta y estira los brazos'], ['hang', 'Colgado'], ['climb', 'Trepa'], ['walk2', 'Camina arriba'], ['run2', 'Vuelve a correr']].map(([id, label]) => ({ id, label }));
const MANTLE = { press: 'se apoya en el borde', over: 'pasa el pecho por encima', hand: 'estira una mano', pull: 'tira y se arrastra', lie: 'queda acostado', getup: 'se levanta (red)', stand: 'se para (red)', stepup: 'sube un pie', rise: 'se levanta sobre el pie', step2: 'sube el otro pie' };
const RUN_PHASES = [{ id: 'stance', label: 'Pie apoyado' }, { id: 'flight', label: 'En el aire' }, { id: 'jump', label: 'Salto' }];

function elmanNet(title, purpose) {
  const n = ep.network;
  const L = ElmanNetwork.layout;
  return {
    title,
    note: purpose + ` Tiene memoria (recurrente): ${L.inputCount} entradas → ${L.ff.map((f) => f.units).join(' → ')} → ${L.recurrent} con memoria → ${n.action.length} salidas (14 motores y 2 manos). Decide 10 veces por segundo.`,
    layers: [
      { name: 'entradas', values: ep.input },
      ...n.ff.map((a, i) => ({ name: 'capa ' + (i + 1), values: a })),
      { name: 'memoria', values: n.hidden },
      { name: 'salidas', values: n.action },
    ],
  };
}
function dashNetInfo(trail) {
  const c = dashCorrect && dashCorrect.last;
  if (!c) return null;
  if (trail) return {
    title: 'Red de terreno libre',
    note: 'Mira el cuerpo, los dos obstáculos que vienen y el perfil del piso (38 números) y suma una corrección a 7 objetivos del corredor. Entrenada con estrategias evolutivas en recorridos al azar. Corre en cada cuadro.',
    layers: [
      { name: 'entradas', values: c.x },
      { name: 'capa', values: c.h },
      { name: 'correcciones', values: c.y, gain: 2, labels: ['cadera apoyo', 'rodilla apoyo', 'tobillo apoyo', 'cadera vuelo', 'rodilla vuelo', 'tobillo vuelo', 'cintura'] },
    ],
  };
  return {
    title: 'Red correctora de "Cajas y bolas"',
    note: 'Mira el cuerpo y los dos obstáculos que vienen (28 números) y suma una corrección a 7 objetivos del corredor. Entrenada con estrategias evolutivas en miles de recorridos al azar. Corre en cada cuadro (60 veces por segundo).',
    layers: [
      { name: 'entradas', values: c.x },
      { name: 'capa', values: c.h },
      { name: 'correcciones', values: c.y, gain: 2, labels: ['cadera apoyo', 'rodilla apoyo', 'tobillo apoyo', 'cadera vuelo', 'rodilla vuelo', 'tobillo vuelo', 'cintura'] },
    ],
  };
}
function holdName(i, TOP) { return i === 0 ? 'nada (todavía)' : i === TOP ? 'la cima' : 'saliente ' + i; }
function climbFacts(pl) {
  const out = [];
  if (!pl || !pl.lim) return out;
  const mv = pl.move;
  if (mv && (mv.kind === 'hand' || mv.kind === 'foot')) out.push(['Moviendo ahora', (mv.kind === 'hand' ? 'mano ' : 'pie ') + (mv.k ? 'der' : 'izq') + ' → ' + holdName(mv.to, pl.TOP), true]);
  else if (mv && mv.kind === 'mantle') out.push(['Moviendo ahora', 'pasa el borde de arriba', true]);
  for (const [nm, arr] of [['Mano', pl.lim.hand], ['Pie', pl.lim.foot]]) arr.forEach((l, k) => out.push([nm + (k ? ' der' : ' izq'), holdName(l.rung, pl.TOP) + (nm === 'Mano' && ep.grips[k].joint ? ' · agarrada' : '')]));
  out.push(['Regla', 'siempre tres apoyos en la roca: mueve un solo miembro por vez']);
  return out;
}
function runFacts(info, vTarget) {
  const sel = ep.sel, st = ep.runner || {}, m = ep.measure, f = info.facts;
  f.push(['Velocidad', (m.centerVelocityX || 0).toFixed(1) + ' m/s (pide ' + vTarget.toFixed(1) + ')']);
  f.push(['Pie de apoyo', st.flight ? 'ninguno: en el aire' : 'el ' + SIDE[st.stance || 0]]);
  if (!sel) return;
  const nx = sel.next;
  if (nx) {
    const d = nx.x0 - world.px[0];
    f.push(['Próximo obstáculo', nx.kind === 'agujero' ? `agujero de ${m1(nx.x1 - nx.x0)}, a ${m1(Math.max(0, d))}` : `${nx.kind || 'caja'} de ${(nx.top || 0).toFixed(2)} m de alto y ${(nx.x1 - nx.x0).toFixed(2)} m de ancho, a ${m1(Math.max(0, d))}`]);
  } else f.push(['Próximo obstáculo', 'ninguno a la vista']);
  const since = ep.frame - sel.jumpAt;
  if (since < 36) f.push(['Decisión', `¡salta! (fuerza del salto ×${sel.jumpScale.toFixed(2)}: más alto el obstáculo, más fuerte)`, true]);
  else if (Math.abs(sel.placeAdj) > 0.005) f.push(['Decisión', `alarga o acorta el paso (${f2(sel.placeAdj)}) para despegar en el punto justo`]);
  else f.push(['Decisión', 'corre normal']);
  if (sel.dropH > 0) f.push(['Bajando de', `una caja de ${sel.dropH.toFixed(2)} m`, true]);
  if (sel.net) {
    let a = 0, b = 0;
    for (const j of [0, 5, 6, 7, 11, 12, 13]) { a += Math.abs(sel.corr[j]); b += Math.abs(sel.base[j]); }
    f.push(['Aporte de la red', `cambia los objetivos de piernas y cintura en ${(a / 7).toFixed(3)} en promedio (sobre un rango de ±1), un ${(100 * a / Math.max(1e-6, b)).toFixed(0)}% de lo que pide el corredor`]);
  }
}

const RECOVER_PHASES = [['settle', 'Se queda quieto'], ['getup', 'Se levanta (red)'], ['stand', 'Se para (red)'], ['walk', 'Camina'], ['done', 'Vuelve a correr']].map(([id, label]) => ({ id, label }));
function describeRecover(R) {
  const info = { stages: {}, phases: RECOVER_PHASES, phase: R.phase, facts: [], net: null, sel: null };
  const st = info.stages;
  st.motors = MOTORS;
  const net = R.phase === 'getup' || R.phase === 'stand';
  st.sense = net ? { who: R.phase === 'getup' ? 'Red "levantarse"' : 'Red "pararse"', kind: 'net', detail: 'Ángulos y velocidades de las articulaciones, inclinación, rayos de visión y contactos, con el piso a la altura donde cayó.' } : { who: 'Código', kind: 'rules', detail: 'Mira si el cuerpo ya está quieto y qué pie toca el piso.' };
  st.decide = { who: 'Director del nivel → levantarse y seguir', kind: 'rules', detail: 'Se cayó en el terreno: en vez de empezar de nuevo, se levanta, camina unos pasos y vuelve a correr.' };
  st.limbs = { who: 'Nadie', kind: 'none', detail: 'Para levantarse no hay objetivos de manos ni pies: la red saca los ángulos directo.' };
  st.joints = R.phase === 'getup' ? { who: 'Red "levantarse"', kind: 'net', detail: 'Aprendió a levantarse imitando a un controlador maestro y después se afinó con estrategias evolutivas.' } : R.phase === 'stand' ? { who: 'Red "pararse"', kind: 'net', detail: 'Mantiene el equilibrio parada un momento.' } : R.phase === 'walk' ? { who: 'Caminata SIMBICON', kind: 'tuned', detail: 'Unos pasos acelerando de 0,3 a 1,2 m/s antes de volver a correr.' } : { who: 'Cuerpo suelto', kind: 'rules', detail: 'Motores relajados hasta quedar quieto.' };
  if (net) info.net = R.phase === 'getup' ? elmanNet('Red "levantarse"', 'Se levanta desde el piso.') : elmanNet('Red "pararse"', 'Mantiene el equilibrio parada.');
  info.facts.push(['Zona del nivel', 'levantándose después de una caída', true], ['Fase', (RECOVER_PHASES.find((q) => q.id === R.phase) || { label: R.phase }).label + ' · ' + R.t.toFixed(1) + ' s'], ['Levantadas en este nivel', String(R.count)], ['Piso', 'a ' + R.floor.toFixed(2) + ' m']);
  return info;
}
function describe() {
  // in the level, describe whichever skill the director has on, as its own scene would
  const zn = popDir ? popDir.zone() : null;
  if (zn === 'recover') return describeRecover(popDir.rec);
  // (a pit: the rock's chain on the pit's far wall: catch the top ledge, hang, climb out)
  const sc = zn ? (zn === 'trail' ? 'trail' : zn === 'wall' || zn === 'pit' ? 'leap' : 'rope') : state.scene;
  const ch = popDir ? (zn === 'rope' ? popDir.rope : zn === 'pit' && popDir.pit ? popDir.pit.OR : popDir.chain) : chain;
  const ropeHoleNow = popDir ? popDir.hole : ropeHole;
  const st = {};
  const info = { stages: st, phases: [], phase: null, sub: null, facts: [], net: null, sel: null, cyclic: false };
  st.motors = MOTORS;
  const m = ep.measure;
  if (state.mode === 'manual' || (state.mode === 'climb' && !['wall', 'mountain'].includes(sc))) {
    st.sense = { who: 'Vos', kind: 'you', detail: 'Mirás la escena.' };
    st.decide = { who: 'Vos', kind: 'you', detail: 'Elegís qué hacer moviendo las marcas.' };
    st.limbs = { who: 'Tus marcas', kind: 'you', detail: 'Cadera, manos y pies van adonde arrastrás cada marca.' };
    st.joints = POSE_EXEC();
    info.facts.push(['Equilibrio', state.balance ? 'activado: cadera y tobillos mantienen el peso sobre los pies apoyados' : 'apagado'], ['Manos', ['izq', 'der'].map((s, k) => s + ': ' + (ep.grips[k].joint ? 'agarrada' : state.grip[k] ? 'agarra al llegar' : 'suelta')).join(' · ')]);
  } else if (state.mode === 'stand') {
    st.sense = { who: 'Red "pararse"', kind: 'net', detail: `Recibe ${config.net.inputCount} números: ángulos y velocidades de las articulaciones, inclinación, posición de cada parte, rayos de visión y contactos.` };
    st.decide = { who: 'Nadie', kind: 'none', detail: 'No elige: siempre intenta quedarse parada.' };
    st.limbs = NO_LIMBS;
    st.joints = { who: 'Red "pararse"', kind: 'net', detail: 'Saca directo los 14 objetivos de los motores. Entrenada con estrategias evolutivas.' };
    info.net = elmanNet('Red "pararse"', 'Mantiene el equilibrio parada.');
    info.facts.push(['Inclinación del tronco', (Math.acos(Math.max(-1, Math.min(1, m.torsoCos || 1))) * 180 / Math.PI).toFixed(0) + '°'], ['Altura de la cabeza', m1(m.headHeight)]);
  } else if (state.mode === 'walk') {
    const sb = ep.simbicon || { stance: 0, t: 0 };
    st.sense = { who: 'Código', kind: 'rules', detail: 'Lee qué pie toca el piso, la velocidad y la inclinación del tronco.' };
    st.decide = { who: 'Nadie', kind: 'none', detail: 'Camina siempre hacia la derecha.' };
    st.limbs = { who: 'Regla de colocación del pie', kind: 'tuned', detail: 'El pie que avanza apunta a un ángulo fijo más una corrección por velocidad (estilo SIMBICON).' };
    st.joints = { who: 'Caminata SIMBICON', kind: 'tuned', detail: 'Una postura por fase del paso y el tronco sostenido por la cadera de apoyo. 14 números ajustados por búsqueda.' };
    info.phases = [{ id: 0, label: 'Apoya el pie izquierdo' }, { id: 1, label: 'Apoya el pie derecho' }]; info.phase = sb.stance; info.cyclic = true;
    info.facts.push(['Velocidad', (m.centerVelocityX || 0).toFixed(2) + ' m/s (pide ' + state.speed.walk + ')'], ['Tiempo en este paso', (sb.t || 0).toFixed(2) + ' s']);
  } else if (state.mode === 'run' && (sc === 'dash' || sc === 'holes' || sc === 'floor' || sc === 'trail')) {
    const trail = sc === 'trail';
    const dash = sc === 'dash' || trail;
    info.sel = ep.sel;
    st.sense = trail
      ? { who: 'Código + red de terreno', kind: 'mix', detail: 'El código mide dónde están los escalones, agujeros, cajas y bolas. La red recibe 38 números: el cuerpo, los dos obstáculos que vienen y el perfil del piso desde 0,4 m atrás hasta 2,8 m adelante.' }
      : dash
      ? { who: 'Código + red correctora', kind: 'mix', detail: 'El código mide directo dónde está cada caja y bola. La red recibe 28 números: tronco, velocidad, pies, ángulos de las piernas y los dos obstáculos que vienen.' }
      : { who: 'Código', kind: 'rules', detail: 'Mide directo dónde empiezan y terminan los agujeros, y la velocidad.' };
    st.decide = { who: 'Selector de saltos', kind: 'tuned', detail: 'En cada apoyo mira el próximo obstáculo: si el paso normal caería cerca del borde, salta ahora; si no, alarga o acorta el paso para llegar justo.' + (dash ? ' La fuerza del salto crece con la altura.' : '') };
    st.limbs = RAIBERT;
    st.joints = trail && params.trailNet ? { who: 'Corredor + red de terreno', kind: 'mix', detail: 'El corredor pide una postura; la red le suma correcciones a caderas, rodillas, tobillos y cintura según el cuerpo y el piso que viene.' } : trail ? RUNNER : dash && params.dashNet ? { who: 'Corredor + red correctora', kind: 'mix', detail: 'El corredor pide una postura; la red le suma correcciones a caderas, rodillas, tobillos y cintura según cómo viene el cuerpo y qué tiene adelante.' } : RUNNER;
    const r = ep.runner || {};
    info.phases = RUN_PHASES; info.cyclic = true;
    info.phase = r.jumping ? 'jump' : r.flight ? 'flight' : 'stance';
    runFacts(info, trail ? params.trail.vRun || 2.6 : dash ? params.dash.vRun || 2.4 : state.speed.run);
    if (trail && trailInfo) info.facts.push(['Piso', 'a ' + trailInfo.floorAt(world.px[0]).toFixed(2) + ' m; un paso adelante ' + trailInfo.floorAt(world.px[0] + 1.2).toFixed(2) + ' m']);
    if (trail && params.trailNet) info.net = dashNetInfo(true);
    else if (dash && !trail && params.dashNet) info.net = dashNetInfo();
  } else if (state.mode === 'run' && sc === 'rope' && ch) {
    const ph = ch.phase;
    info.phases = ROPE_PHASES; info.phase = ph;
    st.sense = { who: 'Código', kind: 'rules', detail: 'Mide la distancia a la soga y a los bordes del agujero, los contactos y la velocidad.' };
    st.decide = { who: 'Secuencia de la soga', kind: 'tuned', detail: 'Pasa de una fase a otra por contactos y distancias: salta en el borde, agarra cuando la mano llega, suelta cuando el vuelo libre caería del otro lado.' };
    st.limbs = ph === 'reach' || ph === 'swing' ? { who: 'Manos a la soga', kind: 'rules', detail: 'Las manos apuntan a la soga con geometría de dos segmentos y se cierran al tocarla.' } : ph === 'run' || ph === 'land' || ph === 'run2' ? RAIBERT : ph === 'walk' ? { who: 'Regla del paso (SIMBICON)', kind: 'tuned', detail: 'El pie que avanza apunta a un ángulo fijo más una corrección por velocidad.' } : { who: 'Pies donde apoyaron', kind: 'rules', detail: 'Los pies se quedan donde tocaron y la cadera sube encima de ellos.' };
    st.joints = ph === 'run' || ph === 'land' || ph === 'run2' ? RUNNER : ph === 'reach' ? { who: 'Corredor + brazos', kind: 'tuned', detail: 'Las piernas siguen al corredor; los brazos se estiran hacia la soga.' } : ph === 'swing' ? { who: 'Postura de balanceo', kind: 'tuned', detail: 'Brazo que agarra a un ángulo fijo, piernas recogidas y después estiradas para aterrizar. Números ajustados por búsqueda.' } : ph === 'walk' ? { who: 'Caminata SIMBICON', kind: 'tuned', detail: 'Unos pasos caminando antes de volver a correr.' } : POSE_EXEC();
    if ((ph === 'land' || ph === 'run2') && params.ropeNet) {
      st.joints = { who: 'Corredor + red de aterrizaje', kind: 'mix', detail: 'El corredor aterriza y retoma la carrera; una red entrenada para eso (la misma forma que la de terreno libre) le suma correcciones a piernas y cintura.' };
      const ni = dashNetInfo(true);
      if (ni) info.net = Object.assign(ni, { title: 'Red de aterrizaje de la soga', note: 'Corrige al corredor al aterrizar y retomar la carrera después de la soga. Mira el cuerpo y el perfil del piso (38 números). Entrenada con estrategias evolutivas en saltos con soga al azar.' });
    }
    const rp = ep.terrain.ropes[0];
    info.facts.push(['Fase', (ROPE_PHASES.find((q) => q.id === ph) || { label: ph }).label + ' · ' + (ch.t || 0).toFixed(1) + ' s']);
    info.facts.push(['Soga', 'a ' + m1(Math.abs(rp.x - world.px[0])) + (rp.x > world.px[0] ? ' adelante' : ' atrás')]);
    if (ropeHoleNow) info.facts.push(['Agujero', world.px[0] < ropeHoleNow.x0 ? 'empieza a ' + m1(ropeHoleNow.x0 - world.px[0]) : world.px[0] < ropeHoleNow.x1 ? 'está encima' : 'ya lo pasó (' + m1(world.px[0] - ropeHoleNow.x1) + ')']);
    info.facts.push(['Manos agarradas', String((ep.grips[0].joint ? 1 : 0) + (ep.grips[1].joint ? 1 : 0))]);
  } else if (state.mode === 'run' && (sc === 'obstacle' || sc === 'leap') && ch) {
    const md = ch.mode, pl = ch.planner, mp = pl && pl.phase;
    const lp = sc === 'leap' && !(ch.leap && ch.leap.missed);
    info.phases = lp ? LEAP_PHASES : OBST_PHASES; info.phase = md; info.sub = md === 'climb' && mp ? MANTLE[mp] || mp : null;
    st.sense = { who: 'Código', kind: 'rules', detail: 'Mide la distancia a la roca, la velocidad y dónde están los salientes.' };
    st.decide = lp ? { who: 'Secuencia del salto a la pared', kind: 'tuned', detail: 'Corre sin frenar, ajusta los pasos para despegar a la distancia justa, salta, agarra el saliente más alto que alcanza, se cuelga y le pasa el control al planificador de agarres.' } : { who: 'Secuencia del obstáculo', kind: 'tuned', detail: 'Cambia de habilidad por distancia y velocidad: corre, camina, frena cerca de la roca, trepa, se levanta arriba y vuelve a correr.' };
    const netNow = md === 'climb' && (mp === 'getup' || mp === 'stand');
    st.limbs = md === 'climb' ? (netNow ? NO_LIMBS : { who: 'Planificador de agarres', kind: 'tuned', detail: 'Elige en qué saliente va cada mano y cada pie, uno por vez, con tres apoyos siempre en la roca.' }) : md === 'run' || md === 'run2' ? RAIBERT : { who: 'Regla del paso', kind: 'tuned', detail: 'Pie que avanza a un ángulo fijo más corrección por velocidad.' };
    st.joints = netNow ? { who: mp === 'getup' ? 'Red "levantarse"' : 'Red "pararse"', kind: 'net', detail: 'Saca directo los 14 objetivos de los motores.' } : md === 'climb' || md === 'hold' ? POSE_EXEC() : md === 'run' || md === 'run2' ? RUNNER : md === 'stop' ? { who: 'Freno', kind: 'tuned', detail: 'Baja la velocidad hasta quedar parado frente a la roca.' } : { who: 'Caminata SIMBICON', kind: 'tuned', detail: 'Camina despacio hacia la roca.' };
    if (netNow) info.net = mp === 'getup' ? elmanNet('Red "levantarse"', 'Aprendió a levantarse imitando a un controlador maestro y después se afinó con estrategias evolutivas.') : elmanNet('Red "pararse"', 'Mantiene el equilibrio parada.');
    if (rockX != null) info.facts.push(['Roca', world.px[0] < rockX ? 'a ' + m1(rockX - world.px[0]) : 'encima']);
    info.facts.push(['Velocidad', (m.centerVelocityX || 0).toFixed(2) + ' m/s']);
    if (md === 'climb') info.facts.push(...climbFacts(pl));
    if (md === 'leap' || md === 'hang') {
      st.limbs = { who: 'Manos al saliente', kind: 'rules', detail: 'Las dos manos apuntan al saliente elegido con geometría de dos segmentos y se cierran al llegar; las piernas se recogen en el aire.' };
      st.joints = md === 'leap' ? { who: 'Corredor + brazos', kind: 'tuned', detail: 'El tronco sigue la fase de vuelo del corredor; brazos al saliente y piernas en una postura fija, ajustada por búsqueda.' } : { who: 'Postura colgado', kind: 'tuned', detail: 'La mano que agarró sostiene; la otra va al mismo saliente; rodillas y caderas dobladas para acercar los pies a la roca.' };
    }
    if (sc === 'leap' && ch.leap) {
      const L = ch.leap;
      info.facts.push(['Saliente elegido', L.aimI ? 'el ' + L.aimI + ' (el más alto que alcanza)' : 'ninguno al alcance']);
      if (L.caughtY != null) info.facts.push(['Agarre', 'a ' + L.caughtY.toFixed(2) + ' m de altura', md === 'hang']);
      if (L.missed) info.facts.push(['Salto', 'no llegó a agarrarse: sigue con el método lento', true]);
    }
  } else if (state.mode === 'climb') {
    const mp = planner && planner.phase;
    const netNow = mp === 'getup' || mp === 'stand';
    st.sense = { who: 'Código', kind: 'rules', detail: 'Conoce dónde está cada saliente de la roca y dónde están manos y pies.' };
    st.decide = { who: 'Planificador de agarres', kind: 'tuned', detail: 'Decide qué miembro mover y a qué saliente, y cuándo pasar por encima del borde.' };
    st.limbs = netNow ? NO_LIMBS : { who: 'Planificador de agarres', kind: 'tuned', detail: 'Elige el saliente de cada mano y cada pie, uno por vez, con tres apoyos siempre en la roca.' };
    st.joints = netNow ? { who: mp === 'getup' ? 'Red "levantarse"' : 'Red "pararse"', kind: 'net', detail: 'Saca directo los 14 objetivos de los motores.' } : POSE_EXEC();
    if (netNow) info.net = mp === 'getup' ? elmanNet('Red "levantarse"', 'Aprendió a levantarse imitando a un controlador maestro y después se afinó con estrategias evolutivas.') : elmanNet('Red "pararse"', 'Mantiene el equilibrio parada.');
    if (mp) { info.phases = [{ id: 'climb', label: 'Trepa' }, { id: 'mantle', label: 'Pasa el borde' }]; info.phase = 'mantle'; info.sub = MANTLE[mp] || mp; }
    else { info.phases = [{ id: 'climb', label: 'Trepa' }, { id: 'mantle', label: 'Pasa el borde' }]; info.phase = 'climb'; }
    info.facts.push(...climbFacts(planner));
  } else {
    st.sense = st.decide = st.limbs = st.joints = { who: '—', kind: 'none', detail: '' };
  }
  if (zn) {
    info.stages.decide = Object.assign({}, info.stages.decide, { who: 'Director del nivel → ' + info.stages.decide.who, detail: 'El director (código) elige la habilidad por zona: terreno, la roca o la soga. Ahora: ' + (zn === 'trail' ? 'terreno' : zn === 'wall' ? 'la roca' : 'la soga') + '. ' + info.stages.decide.detail });
    info.facts.unshift(['Zona del nivel', zn === 'trail' ? 'terreno' : zn === 'wall' ? 'la roca de 2 m' : zn === 'pit' ? 'cayó en un pozo: se agarra del borde de enfrente y sale trepando' : 'el agujero con soga', true]);
  }
  return info;
}

// what the controllers aim at, drawn on the stage
function drawHelpers(info) {
  if (!state.helpers || (state.scene === 'sandbox' && sb.editing)) return;
  ctx.save();
  ctx.font = `${Math.max(11, Math.round(cv.height * 0.024))}px ${css('--mono')}`;
  ctx.textAlign = 'center';
  const acc = css('--accent');
  // the goal the pose executor got (planners, the rope's landing, your marks): rings where each part should go
  const g = tele.goal;
  if (g && ep.frame - tele.goalFrame <= 3 && state.mode !== 'manual') {
    const items = [];
    if (g.pelvis) items.push([g.pelvis, pose.pelvisOf(world), '--mark-hips', 'cadera']);
    (g.hands || []).forEach((h, k) => h && items.push([h, pose.handOf(world, k ? P.handRight : P.handLeft), '--mark-hand', k ? 'mano der' : 'mano izq']));
    (g.feet || []).forEach((f, k) => f && items.push([f, pose.soleOf(world, k ? P.footRight : P.footLeft), '--mark-foot', k ? 'pie der' : 'pie izq']));
    for (const [t, b, col, lab] of items) {
      const s1 = toScreen(b.x, b.y), s2 = toScreen(t.x, t.y);
      ctx.strokeStyle = css(col); ctx.lineWidth = 2; ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(s1.x, s1.y); ctx.lineTo(s2.x, s2.y); ctx.stroke(); ctx.setLineDash([]);
      ctx.beginPath(); ctx.arc(s2.x, s2.y, 8, 0, Math.PI * 2); ctx.stroke();
      if (Math.hypot(s1.x - s2.x, s1.y - s2.y) > 24) { ctx.fillStyle = css(col); ctx.fillText(lab, s2.x, s2.y - 13); }
    }
  }
  // the jump selector: the next obstacle outlined, the take-off zone, the jump
  const sel = info.sel;
  if (sel && sel.next) {
    const o = sel.next, top = o.top != null ? o.top : groundAt(o.x0 - 0.05);
    const a = toScreen(o.x0, top + 0.02), b = toScreen(o.x1, o.top != null ? 0 : top - 0.6);
    ctx.strokeStyle = acc; ctx.lineWidth = 3; ctx.setLineDash([6, 4]);
    ctx.strokeRect(a.x - 3, a.y - 3, b.x - a.x + 6, b.y - a.y + 6); ctx.setLineDash([]);
    ctx.fillStyle = acc; ctx.fillText('próximo', (a.x + b.x) / 2, a.y - 10);
    const zx = toScreen(o.x0 - sel.win, top).x, zx1 = toScreen(o.x0, top).x, zy = toScreen(0, groundAt(o.x0 - 0.3)).y;
    ctx.globalAlpha = 0.16; ctx.fillRect(zx, zy - 6, zx1 - zx, 6); ctx.globalAlpha = 1;
    ctx.fillText('si apoya acá, salta', (zx + zx1) / 2, zy + 18);
  }
  if (sel && ep.frame - sel.jumpAt < 30) {
    const h = toScreen(world.px[2], world.py[2] + 0.35);
    ctx.fillStyle = acc; ctx.font = `700 ${Math.max(14, Math.round(cv.height * 0.035))}px ${css('--display')}`;
    ctx.fillText('¡salta!', h.x, h.y);
  }
  ctx.restore();
}


// ---------- sandbox: build your own course ----------
// Pieces in order of x (src/terrain.js buildSandbox puts in what each needs); the course director
// (src/director.js) hands over to the right skill on each. You place, move, resize and delete pieces while it
// is paused (✎ Editar); ▶ Correr runs it, again and again, counting on each piece how often it got past and
// where it fell. The course lives in the link (#pista=...) and in this browser.
const SB_KEY = 'ragdoll-pista-v1';
const SB_CODE = { box: 'b', ball: 'o', up: 'u', down: 'd', pit: 'p', rock: 'r', rope: 's' };
const SB_PRESETS = {
  pop: [['box', 4], ['ball', 8], ['up', 11, { d: 0.25 }], ['pit', 14, { w: 0.6 }], ['rock', 20], ['box', 34, { h: 0.2 }], ['pit', 38, { w: 0.8 }], ['rope', 44, { w: 1.8 }], ['ball', 54], ['pit', 58, { w: 0.7 }]],
  carrera: [['box', 4, { h: 0.15 }], ['ball', 7], ['box', 10, { h: 0.3, w: 0.6 }], ['up', 13, { d: 0.3 }], ['ball', 16, { r: 0.15 }], ['down', 19, { d: 0.4 }], ['box', 22, { h: 0.25 }], ['ball', 25, { r: 0.09 }], ['box', 28, { h: 0.35, w: 0.3 }], ['up', 31, { d: 0.2 }], ['box', 34]],
  pozos: [['pit', 4, { w: 0.5 }], ['pit', 8, { w: 0.8 }], ['up', 11, { d: 0.3 }], ['pit', 14, { w: 1.0 }], ['pit', 18, { w: 0.6 }], ['down', 21, { d: 0.3 }], ['pit', 24, { w: 0.9 }], ['pit', 28, { w: 0.7 }]],
  rocas: [['box', 4], ['rock', 8], ['ball', 22], ['down', 25, { d: 0.3 }], ['rock', 28, { H: 2.2 }], ['rope', 45, { w: 1.9 }], ['box', 55]],
  vacia: [],
};
const SB_TL = { trail: ['--k-tuned', 'corre y salta (corredor + red)'], wall: ['--k-mix', 'roca: salta, se agarra y trepa'], rope: ['--rope', 'soga'], pit: ['--mark-hips', 'pozo: se agarra y sale'], recover: ['--accent', 'se levanta'] };
const sb = {
  pieces: [], level: null, mid: 0, editing: false, tool: null, sel: -1, drag: null, hover: null, repeat: true, ghost: true,
  stats: [], attempts: 0, finishes: 0, best: null, att: null, prevPath: null, thrown: [], sling: null, timeline: [], tlLast: -1,
  banner: null, confetti: [], zoom: 1, timer: 0,
};
function sbMake(type, x, size) {
  const p = { type, x: +x.toFixed(2), seed: 1 + Math.floor(Math.random() * 997) };
  for (const [k, , , , def] of SANDBOX_PIECES[type].size) p[k] = size && size[k] != null ? size[k] : def;
  return p;
}
function sbRandom() {
  const W = [['box', 0.22], ['ball', 0.14], ['up', 0.1], ['down', 0.1], ['pit', 0.2], ['rock', 0.12], ['rope', 0.12]];
  const out = []; let x = 4, rocks = 0, ropesN = 0;
  const n = 8 + Math.floor(Math.random() * 5);
  for (let i = 0; i < n; i++) {
    let u = Math.random(), t = 'box';
    for (const [k, w] of W) { if ((u -= w) <= 0) { t = k; break; } }
    if ((t === 'rock' && rocks >= 2) || (t === 'rope' && ropesN >= 2)) t = 'box';
    if (t === 'rock') rocks++; if (t === 'rope') ropesN++;
    const size = {};
    for (const [k, , lo, hi] of SANDBOX_PIECES[t].size) size[k] = +(lo + Math.random() * (hi - lo)).toFixed(2);
    out.push(sbMake(t, x, size));
    x += 3 + Math.random() * 3;
  }
  return out;
}
function sbEncode() {
  const a = sb.pieces.map((p) => [SB_CODE[p.type], +p.x.toFixed(2)].concat(SANDBOX_PIECES[p.type].size.map(([k]) => +(+p[k]).toFixed(2)), p.seed || 1));
  return btoa(JSON.stringify(a)).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}
function sbDecode(code) {
  try {
    const inv = Object.fromEntries(Object.entries(SB_CODE).map(([k, v]) => [v, k]));
    const a = JSON.parse(atob(code.replace(/-/g, '+').replace(/_/g, '/')));
    return a.map((q) => {
      const type = inv[q[0]]; if (!type) return null;
      const p = { type, x: +q[1], seed: +q[q.length - 1] || 1 };
      SANDBOX_PIECES[type].size.forEach(([k], i) => (p[k] = +q[2 + i]));
      return p;
    }).filter(Boolean);
  } catch (e) { return null; }
}
function sbInit() {
  let pieces = null;
  try { const m = (location.hash || '').match(/pista=([A-Za-z0-9_-]+)/); if (m) pieces = sbDecode(m[1]); } catch (e) {}
  if (!pieces) { try { const c = localStorage.getItem(SB_KEY); if (c) pieces = sbDecode(c); } catch (e) {} }
  sb.pieces = pieces || SB_PRESETS.pop.map(([t, x, z]) => sbMake(t, x, z));
  sbResetStats();
  // the stage's buttons (made here: the shell only has their places)
  $('sbRun').onclick = () => sbSetEditing(false);
  $('sbEdit').onclick = () => sbSetEditing(!sb.editing);
  $('sbRepeat').onclick = () => { sb.repeat = !sb.repeat; sbSyncUI(); };
  $('sbGhost').onclick = () => { sb.ghost = !sb.ghost; sbSyncUI(); };
  for (const b of document.querySelectorAll('[data-tool]')) b.onclick = () => {
    const t = b.dataset.tool;
    sb.tool = sb.tool === t ? null : t;
    if (t === 'throw') { if (sb.editing && sb.tool) sbSetEditing(false); }
    else if (sb.tool && !sb.editing) sbSetEditing(true);
    sbSyncUI();
  };
  for (const b of document.querySelectorAll('[data-preset]')) b.onclick = () => {
    const n = b.dataset.preset;
    sb.pieces = n === 'azar' ? sbRandom() : (SB_PRESETS[n] || []).map(([t, x, z]) => sbMake(t, x, z));
    sb.sel = -1;
    sbChanged();
    if (!sb.editing) { state.paused = false; syncButtons(); }
  };
  $('sbLink').onclick = () => {
    const url = location.href.split('#')[0] + '#pista=' + sbEncode();
    const done = () => { $('sbLink').textContent = '¡Link copiado!'; setTimeout(() => ($('sbLink').textContent = 'Copiar link de la pista'), 1800); };
    try { navigator.clipboard.writeText(url).then(done, () => window.prompt('Link de la pista:', url)); } catch (e) { window.prompt('Link de la pista:', url); }
  };
}
function sbResetStats() { sb.stats = sb.pieces.map(() => ({ ok: 0, fail: 0 })); sb.attempts = 0; sb.finishes = 0; sb.best = null; sb.prevPath = null; }
function sbSave() {
  const code = sbEncode();
  try { localStorage.setItem(SB_KEY, code); } catch (e) {}
  try { history.replaceState(null, '', '#pista=' + code); } catch (e) {}
}
// the course changed: rebuild it (the stats start over: they were for the other course)
function sbChanged() { sbResetStats(); sbSave(); reset(); sbSyncUI(); }
function sbSoon() { clearTimeout(sb.timer); sb.timer = setTimeout(sbChanged, 140); }
function sbSetEditing(on) {
  sb.editing = on;
  if (on) { state.paused = true; sb.zoom = 1.7; camY = 0; if (sb.tool === 'throw') sb.tool = null; }
  else { sb.zoom = 1; if (sb.tool !== 'throw') sb.tool = null; sb.sel = -1; state.paused = false; state.seed++; reset(); }
  resize(); syncButtons(); sbSyncUI();
}
function sbNewAttempt() {
  sb.att = { passed: new Set(), path: [], done: false, endAt: 0 };
  sb.thrown = []; sb.sling = null; sb.timeline = []; sb.tlLast = -1;
}
// where a piece is (world coordinates), for drawing and picking
function sbBox(i) {
  const s = sb.level && sb.level.spans[i], p = sb.pieces[i];
  if (!s || !p) return null;
  let x0 = s.x0, x1 = s.x1;
  if (p.type === 'up' || p.type === 'down') { x0 -= 0.25; x1 += 0.25; }
  return { x0: x0 + sb.mid, x1: x1 + sb.mid, y0: p.type === 'pit' || p.type === 'rope' ? Math.max(0, s.top - 1.2) : s.base, y1: p.type === 'rope' ? s.top + 4.35 : s.top + 0.12 };
}
function sbHit(w) {
  let best = -1, ba = 1e9;
  for (let i = 0; i < sb.pieces.length; i++) {
    const b = sbBox(i); if (!b) continue;
    if (w.x > b.x0 - 0.15 && w.x < b.x1 + 0.15 && w.y > b.y0 - 0.2 && w.y < b.y1 + 0.4) { const a = (b.x1 - b.x0) * (b.y1 - b.y0); if (a < ba) { ba = a; best = i; } }
  }
  return best;
}
const sbWidth = (p) => (p.type === 'box' ? p.w : p.type === 'ball' ? 2 * p.r : p.type === 'pit' || p.type === 'rope' ? p.w : p.type === 'rock' ? 1.2 : 0.02);
function sbDown(sp) {
  const w = toWorld(sp.x, sp.y);
  if (!sb.editing) {
    if (sb.tool === 'throw') { sb.sling = { a: w, b: w }; return true; }
    return false;
  }
  const hit = sbHit(w);
  if (hit >= 0) { sb.drag = { i: hit, dx: (w.x - sb.mid) - sb.pieces[hit].x, sx: sp.x, moved: false }; sb.sel = hit; sbSyncUI(); return true; }
  if (sb.tool && sb.tool !== 'throw') {
    const p = sbMake(sb.tool, Math.max(1, (w.x - sb.mid) - sbWidth({ type: sb.tool, ...Object.fromEntries(SANDBOX_PIECES[sb.tool].size.map(([k, , , , d]) => [k, d])) }) / 2));
    sb.pieces.push(p); sb.sel = sb.pieces.length - 1;
    sbChanged();
    return true;
  }
  sb.drag = { pan: true, sx: sp.x, cam0: camX }; sb.sel = -1; sbSyncUI();
  return true;
}
function sbMove(sp) {
  const w = toWorld(sp.x, sp.y);
  sb.hover = w;
  if (sb.sling) sb.sling.b = w;
  const d = sb.drag;
  if (!d) return;
  if (d.pan) camX = d.cam0 - (sp.x - d.sx) / scale;
  else { if (Math.abs(sp.x - d.sx) > 4) d.moved = true; if (d.moved) sb.pieces[d.i].x = +Math.max(1, (w.x - sb.mid) - d.dx).toFixed(2); }
}
function sbUp(sp) {
  if (sb.sling) {
    // a slingshot: pull back from where the ball goes, let go
    const a = sb.sling.a, b = sb.sling.b; sb.sling = null;
    let vx = (a.x - b.x) * 6, vy = (a.y - b.y) * 6;
    const v = Math.hypot(vx, vy), cap = 14;
    if (v > cap) { vx *= cap / v; vy *= cap / v; }
    if (v > 0.3 && ep) { const id = sim.spawnThrown(world, { radius: 0.11, density: 40 }, a.x, a.y, vx, vy); sb.thrown.push({ id, r: 0.11 }); }
  }
  const d = sb.drag; sb.drag = null;
  if (d && d.i != null && d.moved) sbChanged();
}
function sbKey(ev) {
  if (ev.target && ev.target.tagName === 'INPUT') return false;
  if (sb.editing && (ev.key === 'Delete' || ev.key === 'Backspace') && sb.sel >= 0) { sb.pieces.splice(sb.sel, 1); sb.sel = -1; sbChanged(); ev.preventDefault(); return true; }
  if (ev.key === 'Escape') { sb.tool = null; sb.sel = -1; sbSyncUI(); return true; }
  if (ev.key === 'e' || ev.key === 'E') { sbSetEditing(!sb.editing); return true; }
  return false;
}
// one attempt, as it goes: pieces passed, the path, the finish line
function sbTick() {
  const L = sb.level, att = sb.att; if (!L || !popDir) return;
  const px = world.px[0] - sb.mid;
  if (ep.frame % 4 === 0 && att.path.length < 6000) att.path.push([world.px[0], world.py[0]]);
  const z = popDir.zone(), up = z !== 'recover' && z !== 'pit';
  for (let i = 0; i < sb.pieces.length; i++) {
    const sp = L.spans[i]; if (!sp || att.passed.has(i)) continue;
    const past = sb.pieces[i].type === 'rock' ? sp.x1 + 3 : sp.x1 + 0.6;
    if (px > past && up && !att.done) { att.passed.add(i); sb.stats[i].ok++; }
  }
  if (!att.done && px > L.finishX) {
    att.done = true; att.endAt = performance.now() + 2600;
    const t = ep.frame / HZ;
    sb.attempts++; sb.finishes++;
    const best = sb.best == null || t < sb.best; if (best) sb.best = t;
    sb.prevPath = att.path;
    sb.banner = { text: '¡Llegó!', sub: t.toFixed(1) + ' s' + (best && sb.finishes > 1 ? ' · ¡récord!' : ''), until: performance.now() + 2600, good: true };
    sbConfetti();
    sbSyncUI();
  }
  if (att.done && att.endAt && performance.now() > att.endAt) {
    att.endAt = 0;
    if (sb.repeat) { state.seed++; reset(); } else { state.paused = true; syncButtons(); }
  }
}
// an attempt ended in a fall or stuck: count it on the piece where it was; returns whether to start another
function sbFail(why) {
  const att = sb.att;
  if (att && !att.done) {
    att.done = true; sb.attempts++;
    const px = world.px[0] - sb.mid;
    let bi = -1, bx = 1e9;
    for (let i = 0; i < sb.pieces.length; i++) { const sp = sb.level.spans[i]; if (!sp || att.passed.has(i)) continue; if (sp.x0 < px + 3 && sp.x0 < bx) { bx = sp.x0; bi = i; } }
    if (bi >= 0) sb.stats[bi].fail++;
    sb.prevPath = att.path;
    sb.banner = { text: why === 'stuck' ? 'Se trabó' : 'Se cayó', sub: bi >= 0 ? SANDBOX_PIECES[sb.pieces[bi].type].label.toLowerCase() + ' n.º ' + (bi + 1) : 'en el piso', until: performance.now() + 1800, good: false };
    sbSyncUI();
  }
  if (!sb.repeat) { state.paused = true; syncButtons(); return false; }
  return true;
}
function sbConfetti() {
  const s = toScreen(world.px[2], world.py[2] + 0.3);
  const cols = ['--accent', '--k-tuned', '--ok', '--mark-hips', '--skin', '--k-mix'].map(css);
  for (let i = 0; i < 90; i++) sb.confetti.push({ x: s.x, y: s.y, vx: (Math.random() - 0.5) * 9, vy: -3 - Math.random() * 8, a: Math.random() * 6, va: (Math.random() - 0.5) * 0.4, c: cols[i % cols.length], life: 1.6 + Math.random() * 1.2 });
}
// the rocks' outline (their tiers are separate polygons)
function sbOutlineRocks() {
  for (const w of sb.level.walls) {
    ctx.strokeStyle = css('--rock-edge'); ctx.lineWidth = 2; ctx.beginPath();
    w.crag.surface.slice(1).forEach(([x, y], i) => { const t = toScreen(x + sb.mid, y); i ? ctx.lineTo(t.x, t.y) : ctx.moveTo(t.x, t.y); });
    ctx.stroke();
  }
}
function sbDrawUnder() {
  const L = sb.level; if (!L) return;
  const M = sb.mid, fs = Math.max(11, Math.round(cv.height * 0.022));
  ctx.save();
  ctx.font = `${fs}px ${css('--mono')}`; ctx.textAlign = 'center';
  // the finish line: a chequered flag
  const fx = L.finishX + M, fy = popDir ? popDir.floorAt(fx) : L.lift;
  const f0 = toScreen(fx, fy), f1 = toScreen(fx, fy + 1.7), q = 0.09 * scale;
  ctx.strokeStyle = css('--ink'); ctx.lineWidth = Math.max(2, 0.03 * scale);
  ctx.beginPath(); ctx.moveTo(f0.x, f0.y); ctx.lineTo(f1.x, f1.y); ctx.stroke();
  for (let r = 0; r < 3; r++) for (let c = 0; c < 5; c++) { ctx.fillStyle = (r + c) % 2 ? css('--ink') : css('--stage'); ctx.fillRect(f1.x + c * q, f1.y + r * q, q, q); }
  ctx.strokeRect(f1.x, f1.y, 5 * q, 3 * q);
  ctx.fillStyle = css('--ink'); ctx.fillText('LLEGADA', f1.x + 2.5 * q, f1.y - 8);
  if (sb.editing) {
    // where each big piece needs room: the run-up before rocks and ropes, the top of a rock to get up on
    for (let i = 0; i < sb.pieces.length; i++) {
      const sp = L.spans[i]; if (!sp) continue;
      const band = (a, b, y, label) => { const s0 = toScreen(a + M, y), s1 = toScreen(b + M, y); ctx.globalAlpha = 0.18; ctx.fillStyle = css('--accent'); ctx.fillRect(s0.x, s0.y - 5, s1.x - s0.x, 5); ctx.globalAlpha = 0.8; ctx.fillText(label, (s0.x + s1.x) / 2, s0.y + fs + 4); ctx.globalAlpha = 1; };
      if (sp.runFrom != null) band(sp.runFrom, sp.x0, sb.pieces[i].type === 'rock' ? 0 : sp.top, 'carrera');
      if (sp.landing != null) band(sp.x1, sp.landing, sp.top, 'arriba: se levanta y sigue');
    }
    ctx.fillStyle = css('--muted');
    for (const a of L.autos) { const t = toScreen(a.x0, Math.max(a.from, a.to)); ctx.fillText('auto', toScreen(a.x0 + M, 0).x, t.y - 6); }
    // every piece outlined and named; the one you picked in the accent colour
    for (let i = 0; i < sb.pieces.length; i++) {
      const b = sbBox(i); if (!b) continue;
      const a = toScreen(b.x0, b.y1), c = toScreen(b.x1, b.y0), sel = i === sb.sel;
      ctx.strokeStyle = sel ? css('--accent') : css('--muted'); ctx.lineWidth = sel ? 2.5 : 1.2; ctx.setLineDash(sel ? [] : [4, 4]);
      ctx.strokeRect(a.x - 4, a.y - 4, c.x - a.x + 8, c.y - a.y + 8); ctx.setLineDash([]);
      ctx.fillStyle = sel ? css('--accent') : css('--muted');
      ctx.fillText((i + 1) + ' · ' + SANDBOX_PIECES[sb.pieces[i].type].label, (a.x + c.x) / 2, a.y - 10);
    }
    // a piece being dragged: where it goes
    const d = sb.drag;
    if (d && d.i != null && d.moved) {
      const p = sb.pieces[d.i], b = sbBox(d.i);
      if (b) { const w = Math.max(0.2, b.x1 - b.x0), a = toScreen(p.x + M - (p.type === 'up' || p.type === 'down' ? 0.25 : 0), b.y1); ctx.strokeStyle = css('--accent'); ctx.setLineDash([6, 4]); ctx.lineWidth = 2; ctx.strokeRect(a.x, a.y, w * scale, (b.y1 - b.y0) * scale); ctx.setLineDash([]); }
    }
    // the piece you are about to put down
    if (sb.tool && sb.tool !== 'throw' && sb.hover && !sb.drag) {
      const h = toScreen(sb.hover.x, 0), top = toScreen(0, (popDir ? popDir.floorAt(sb.hover.x) : L.lift) + 1).y;
      ctx.strokeStyle = css('--accent'); ctx.setLineDash([3, 5]); ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(h.x, top); ctx.lineTo(h.x, toScreen(0, 0).y); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = css('--accent'); ctx.fillText('+ ' + SANDBOX_PIECES[sb.tool].label, h.x, top - 8);
    }
  }
  // how each piece is going: got past ✓, failed there ✗ (over all the attempts on this course)
  if (sb.attempts > 0 || sb.att) {
    ctx.font = `600 ${fs}px ${css('--mono')}`;
    for (let i = 0; i < sb.pieces.length; i++) {
      const b = sbBox(i), st = sb.stats[i]; if (!b || !st || (!st.ok && !st.fail)) continue;
      const t = toScreen((b.x0 + b.x1) / 2, b.y1 + 0.3);
      const txt1 = '✓' + st.ok, txt2 = st.fail ? ' ✗' + st.fail : '';
      const w1 = ctx.measureText(txt1).width, w2 = ctx.measureText(txt2).width, x0 = t.x - (w1 + w2) / 2;
      ctx.globalAlpha = 0.85; ctx.fillStyle = css('--card'); ctx.fillRect(x0 - 5, t.y - fs, w1 + w2 + 10, fs + 7); ctx.globalAlpha = 1;
      ctx.textAlign = 'left'; ctx.fillStyle = css('--ok'); ctx.fillText(txt1, x0, t.y); ctx.fillStyle = css('--accent'); ctx.fillText(txt2, x0 + w1, t.y); ctx.textAlign = 'center';
    }
  }
  // the previous attempt's path (its hips), faint
  if (sb.ghost && sb.prevPath && sb.prevPath.length > 1) {
    ctx.strokeStyle = css('--mark-hips'); ctx.globalAlpha = 0.35; ctx.lineWidth = 2; ctx.setLineDash([2, 5]);
    ctx.beginPath(); sb.prevPath.forEach(([x, y], i) => { const t = toScreen(x, y); i ? ctx.lineTo(t.x, t.y) : ctx.moveTo(t.x, t.y); }); ctx.stroke();
    ctx.setLineDash([]); ctx.globalAlpha = 1;
  }
  ctx.restore();
}
function sbDrawOver() {
  ctx.save();
  // the balls you threw
  ctx.fillStyle = css('--accent'); ctx.strokeStyle = css('--rock-edge'); ctx.lineWidth = 2;
  for (const b of sb.thrown) { const q = Box2D.b2Body_GetPosition(b.id), s = toScreen(q.x, q.y); ctx.beginPath(); ctx.arc(s.x, s.y, b.r * scale, 0, Math.PI * 2); ctx.fill(); ctx.stroke(); }
  // the slingshot being pulled: where it will fly
  if (sb.sling) {
    const a = toScreen(sb.sling.a.x, sb.sling.a.y), b = toScreen(sb.sling.b.x, sb.sling.b.y);
    ctx.strokeStyle = css('--accent'); ctx.lineWidth = 2; ctx.setLineDash([5, 4]);
    ctx.beginPath(); ctx.moveTo(b.x, b.y); ctx.lineTo(a.x, a.y); ctx.stroke(); ctx.setLineDash([]);
    ctx.beginPath(); ctx.arc(a.x, a.y, 0.11 * scale, 0, Math.PI * 2); ctx.stroke();
    // the arc it would fly (no drag), for a second
    let vx = (sb.sling.a.x - sb.sling.b.x) * 6, vy = (sb.sling.a.y - sb.sling.b.y) * 6; const v = Math.hypot(vx, vy); if (v > 14) { vx *= 14 / v; vy *= 14 / v; }
    ctx.globalAlpha = 0.5; ctx.beginPath();
    for (let t = 0; t <= 1; t += 0.05) { const s = toScreen(sb.sling.a.x + vx * t, sb.sling.a.y + vy * t - 4.9 * t * t); t ? ctx.lineTo(s.x, s.y) : ctx.moveTo(s.x, s.y); }
    ctx.stroke(); ctx.globalAlpha = 1;
  }
  // confetti and the banner
  const dt = 1 / 60;
  sb.confetti = sb.confetti.filter((c) => (c.life -= dt) > 0);
  for (const c of sb.confetti) { c.vy += 14 * dt; c.x += c.vx * dt * scale * 0.25; c.y += c.vy * dt * scale * 0.25; c.a += c.va; ctx.save(); ctx.translate(c.x, c.y); ctx.rotate(c.a); ctx.fillStyle = c.c; ctx.globalAlpha = Math.min(1, c.life); ctx.fillRect(-4, -2.5, 8, 5); ctx.restore(); }
  const bn = sb.banner;
  if (bn && performance.now() < bn.until) {
    ctx.textAlign = 'center';
    ctx.font = `800 ${Math.round(cv.height * 0.08)}px ${css('--display')}`; ctx.fillStyle = bn.good ? css('--ok') : css('--accent');
    ctx.fillText(bn.text, cv.width / 2, cv.height * 0.2);
    ctx.font = `500 ${Math.round(cv.height * 0.035)}px ${css('--mono')}`; ctx.fillStyle = css('--ink');
    ctx.fillText(bn.sub, cv.width / 2, cv.height * 0.2 + cv.height * 0.055);
  }
  if (sb.editing) {
    ctx.textAlign = 'left'; ctx.font = `600 ${Math.max(12, Math.round(cv.height * 0.026))}px ${css('--display')}`; ctx.fillStyle = css('--accent');
    ctx.fillText('✎ Editando — la física está en pausa', 14, 26);
  }
  ctx.restore();
}
// who drives, a strip over the attempt's time
function sbTimeline() {
  if (!popDir) return;
  if ((state.scene === 'pop' || state.scene === 'sandbox') && !state.paused) { const tt = Math.floor(ep.frame / (HZ * 0.1)); if (tt !== sb.tlLast) { sb.tlLast = tt; sb.timeline.push(popDir.zone()); } }
  const c = $('timeline'); if (!c || c.offsetParent === null) return;
  const r = c.getBoundingClientRect(), dpr = Math.min(2, window.devicePixelRatio || 1);
  const W = Math.round(r.width * dpr), H = Math.round(r.height * dpr);
  if (c.width !== W || c.height !== H) { c.width = W; c.height = H; }
  const g = c.getContext('2d');
  g.fillStyle = css('--stage'); g.fillRect(0, 0, W, H);
  const n = sb.timeline.length, span = Math.max(300, n), bw = W / span;
  for (let i = 0; i < n; i++) { const k = SB_TL[sb.timeline[i]]; if (!k) continue; g.fillStyle = css(k[0]); g.fillRect(i * bw, 6 * dpr, Math.ceil(bw), H - 22 * dpr); }
  g.fillStyle = css('--muted'); g.font = `${10 * dpr}px ${css('--mono')}`; g.textAlign = 'center';
  const step = span > 1200 ? 300 : span > 600 ? 200 : 100;
  for (let i = 0; i <= span; i += step) g.fillText((i / 10).toFixed(0) + ' s', Math.min(W - 14 * dpr, Math.max(14 * dpr, i * bw)), H - 4 * dpr);
}
function sbSyncUI() {
  const on = state.scene === 'sandbox';
  $('sbPanel').hidden = !on;
  $('tlBox').hidden = !(on || state.scene === 'pop');
  if ($('marksLegend')) $('marksLegend').hidden = on;
  if (!$('tlKeys').childElementCount) for (const [, [col, label]] of Object.entries(SB_TL)) { const sp = document.createElement('span'); sp.innerHTML = `<i style="background:var(${col})"></i>`; sp.append(label); $('tlKeys').append(sp); }
  if (!on) return;
  $('sbEdit').setAttribute('aria-pressed', sb.editing);
  $('sbRun').setAttribute('aria-pressed', !sb.editing && !state.paused);
  $('sbRepeat').setAttribute('aria-pressed', sb.repeat);
  $('sbGhost').setAttribute('aria-pressed', sb.ghost);
  for (const b of document.querySelectorAll('[data-tool]')) b.setAttribute('aria-pressed', sb.tool === b.dataset.tool);
  $('sbHint').textContent = sb.editing
    ? (sb.tool ? `Tocá la escena donde querés ${SANDBOX_PIECES[sb.tool] ? 'la ' + SANDBOX_PIECES[sb.tool].label.toLowerCase() : 'la pieza'} (podés poner varias; Esc para dejar de poner).` : 'Elegí una pieza en «Poner» y tocá la escena para ponerla. Arrastrá una pieza para moverla, tocála para cambiarle el tamaño o borrarla. Arrastrá el fondo (o usá la rueda) para recorrer la pista.') + ' Lo que una pieza necesita (escalones, lugar para tomar carrera) se agrega solo.'
    : sb.tool === 'throw' ? 'Arrastrá en la escena como una gomera: tirá para atrás desde donde sale la pelota y soltá. ¿Lo aguanta?' : 'Corriendo tu pista. Sobre cada pieza se van contando las pasadas (✓) y las caídas (✗); abajo, quién maneja en cada momento.';
  const hard = sb.stats.map((st, i) => [st, i]).filter(([st]) => st.fail).sort((a, b) => b[0].fail - a[0].fail)[0];
  $('sbStats').innerHTML = sb.attempts ? `Intentos: <b>${sb.attempts}</b> · llegó al final: <b>${sb.finishes}</b>` + (sb.best != null ? ` · mejor tiempo: <b>${sb.best.toFixed(1)} s</b>` : '') + (hard ? ` · la más difícil: <b>${SANDBOX_PIECES[sb.pieces[hard[1]].type].label.toLowerCase()} n.º ${hard[1] + 1}</b> (${hard[0].fail} caídas)` : '') : sb.pieces.length + ' piezas en la pista.';
  // the piece you picked: its sizes and a delete button
  const el = $('sbSel'), p = sb.pieces[sb.sel];
  if (!p || !sb.editing) { el.hidden = true; el.dataset.for = ''; return; }
  const key = sb.sel + ':' + p.type;
  if (el.dataset.for === key && !el.hidden) return;
  el.dataset.for = key; el.hidden = false; el.innerHTML = '';
  const t = document.createElement('b'); t.textContent = (sb.sel + 1) + ' · ' + SANDBOX_PIECES[p.type].label; el.append(t);
  for (const [k, label, lo, hi] of SANDBOX_PIECES[p.type].size) {
    const lab = document.createElement('label'), inp = document.createElement('input'), val = document.createElement('span');
    inp.type = 'range'; inp.min = lo; inp.max = hi; inp.step = 0.01; inp.value = p[k];
    const show = () => (val.textContent = Math.round(+inp.value * 100) + ' cm');
    show();
    inp.oninput = () => { p[k] = +inp.value; show(); sbSoon(); };
    lab.append(label + ' ', inp, val); el.append(lab);
  }
  if (p.type === 'rock') { const b = document.createElement('button'); b.textContent = 'Otra forma'; b.onclick = () => { p.seed = 1 + Math.floor(Math.random() * 997); sbChanged(); }; el.append(b); }
  const del = document.createElement('button'); del.textContent = 'Borrar'; del.onclick = () => { sb.pieces.splice(sb.sel, 1); sb.sel = -1; sbChanged(); }; el.append(del);
}

// ---------- loop ----------
let last = performance.now(), acc = 0, downFor = 0, holeFor = 0, falls = 0;
let sbLevel = null;
let planner = null, summitShown = false, ropeCourse = null, chain = null, ropeHole = null, crossed = 0, bestDash = 0, rockX = null, dashCorrect = null, trailCourse = null, trailInfo = null, popLevel = null, popDir = null, stallX = 0, stallT = 0;
function loop(now) {
  const dt = Math.min(0.1, (now - last) / 1000); last = now;
  if (!state.paused) {
    acc += dt * HZ * state.timeScale;
    let n = 0;
    while (acc >= 1 && n < 6) { ep.advanceOneFrame(); acc -= 1; n++; }
    if (n === 6) acc = 0;
  }
  draw();
  const info = describe();
  drawHelpers(info);
  inspector.update(ep, info);
  // in the automatic modes a fall (head near the ground for 2 s) restarts the run
  const headUp = ep.measure.headHeight - (trailInfo ? trailInfo.floorAt(world.px[2]) : groundAt(world.px[2]));
  // on the dash course sitting or crawling counts too (as dash.js scores it: head under 0.95 m or the trunk past ~70°)
  const popBusy = !!popDir && ((popDir.zone() === 'pit' && popDir.pit && popDir.pit.OR.mode !== 'run2') || popDir.zone() === 'wall' && popDir.chain.mode !== 'run' && popDir.chain.mode !== 'run2' || (ep.grips[0].joint || ep.grips[1].joint) || (popDir.rec && popDir.rec.busy()));
  const down = popDir ? !popBusy && (headUp < 0.95 || (ep.measure.torsoCos || 0) < 0.3 || world.py[0] < popDir.floorAt(world.px[0]) - 0.1) : state.scene === 'dash' || state.scene === 'trail' ? headUp < 0.95 || (ep.measure.torsoCos || 0) < 0.3 || (trailInfo && world.py[0] < trailInfo.floorAt(world.px[0]) - 0.1) : headUp < 0.45;
  downFor = down && !state.paused ? downFor + dt : 0;
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
  // over the rock and running on 5 m past its edge: one more pass, then a new rock (its top ends at 14 m)
  if ((state.scene === 'leap' || state.scene === 'obstacle') && chain && chain.mode === 'run2' && terrain.surface && world.px[0] > terrain.surface[terrain.surface.length - 2][0] + 5) { crossed++; holeFor = 0; downFor = 0; state.seed++; reset(); }
  // a new course each restart (the same seed replayed the same fall exactly)
  if ((state.scene === 'dash' || state.scene === 'trail') && downFor > 1.2) downFor = 2.01;
  if ((state.scene === 'pop' || state.scene === 'sandbox') && downFor > 3) downFor = 2.01; // the level's director gets up on its own first
  // stuck in the level (no progress for 25 s): a new level
  if (popDir && !state.paused) { if (world.px[0] > stallX + 0.5 || popBusy) { stallX = world.px[0]; stallT = 0; } else stallT += dt; }
  const stalled = popDir && stallT > 25;
  if (state.scene === 'sandbox' && !state.paused && sb.att) sbTick();
  if ((state.scene === 'dash' || state.scene === 'trail' || state.scene === 'pop') && downFor === 0) bestDash = Math.max(bestDash, world.px[0] - startX); // the farthest it got while up
  if (state.mode !== 'manual' && (downFor > 2 || stuck || stalled)) {
    stallT = 0; stallX = -1e9; downFor = 0; holeFor = 0;
    // your course: count the attempt (and where it failed); without "repeat" it stops there
    const goOn = state.scene !== 'sandbox' || sbFail(stalled ? 'stuck' : 'fell');
    if (goOn) { falls++; state.seed++; reset(); }
  }
  $('hud').textContent = `cabeza ${headUp.toFixed(2)} m sobre el suelo · agarres ${(ep.grips[0].joint ? 1 : 0) + (ep.grips[1].joint ? 1 : 0)} · avanzó ${(world.px[0] - startX).toFixed(1)} m · t ${(ep.frame / HZ).toFixed(1)} s` + (state.mode !== 'manual' && falls ? ` · caídas ${falls}` : '') + ((state.scene === 'rope' || state.scene === 'leap' || state.scene === 'obstacle') && crossed ? ` · ${state.scene === 'rope' ? 'cruces' : 'pasadas'} ${crossed}` : '') + (popDir && popDir.rec && popDir.rec.count ? ` · levantadas ${popDir.rec.count}` : '') + ((state.scene === 'dash' || state.scene === 'trail' || state.scene === 'pop') ? ` · ${((world.px[0] - startX) / Math.max(0.1, ep.frame / HZ)).toFixed(1)} m/s · récord ${bestDash.toFixed(0)} m` : '');
  if (state.scene === 'sandbox' || state.scene === 'pop') sbTimeline();
  requestAnimationFrame(loop);
}
const inspector = createInspector({ $, css, Box2D, world });
sbInit();
resize();
syncSpeed();
reset();
sbSyncUI();
$('loading').hidden = true;
// for poking at it from the console
window.ragdoll = { targets, state, world, toScreen: (x, y) => { const s = toScreen(x, y); const r = cv.getBoundingClientRect(); return { x: r.left + s.x * r.width / cv.width, y: r.top + s.y * r.height / cv.height }; }, markPos: (id) => markPos(MARKS.find((m) => m.id === id)) };
requestAnimationFrame(loop);
