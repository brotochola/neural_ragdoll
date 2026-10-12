// Mide si la red está usando los cinco targets, y cuánto vale cada uno.
//
//   node probe-targets.js                       red guardada, skill 2, 24 episodios
//   node probe-targets.js --skill 1 --n 30
//   node probe-targets.js --random               pesos al azar, para ver la línea de base
//   node probe-targets.js --net backups/mi-red.json
//
// Por cada target: cuánto se aleja de su marca, cuánto vale su fila en el premio, y qué pasa
// con el fitness cuando se le apagan sus cinco columnas de entrada. Una caída grande en la
// última columna significa que la red lee ese target. Caída cero significa que lo ignora.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { config, setSkill } from './config.js';
import { makeSeededRandom } from './src/utils.js';
import { targetGain, targetSpan, targetPositionWeight, scoreByTarget } from './src/reward.js';
import { ElmanNetwork } from './src/net.js';

const root = path.dirname(fileURLToPath(import.meta.url));

function readArgs() {
  const out = { skill: config.skill.level, n: 24, net: null, random: false, seed: config.eval.seed, commands: null, oracle: 0, legOracle: false };
  const argv = process.argv.slice(2);
  for (let i = 0; i < argv.length; i++) {
    const key = argv[i];
    const val = argv[i + 1];
    if (key === '--skill') { out.skill = +val; i++; }
    else if (key === '--n') { out.n = +val; i++; }
    else if (key === '--net') { out.net = val; i++; }
    else if (key === '--seed') { out.seed = +val; i++; }
    else if (key === '--commands') { out.commands = val; i++; }
    else if (key === '--random') out.random = true;
    else if (key === '--oracle') out.oracle = +val;
    else if (key === '--leg-oracle') out.legOracle = true;
  }
  return out;
}

const opts = readArgs();
setSkill(opts.skill);

const Box2D = await import(pathToFileURL(path.join(root, 'vendor/PhaserBox2D.min.js')).href);
const { createSim } = await import('./src/sim.js');
const sim = createSim(Box2D);

function loadTheta() {
  if (opts.random) {
    const random = makeSeededRandom(opts.seed + 1);
    const theta = new Float32Array(ElmanNetwork.weightCount);
    for (let i = 0; i < theta.length; i++) theta[i] = (random() * 2 - 1) * 0.15;
    return { theta, source: 'random' };
  }
  const file = opts.net || path.join(root, 'backups', 'train-state.json');
  const saved = JSON.parse(fs.readFileSync(file, 'utf8'));
  const theta = Float32Array.from(saved.theta);
  if (theta.length !== ElmanNetwork.weightCount)
    throw new Error(file + ' tiene ' + theta.length + ' pesos, la red pide ' + ElmanNetwork.weightCount);
  return { theta, source: path.basename(file) };
}

const { theta, source } = loadTheta();

// Magnitud de la orden, en el marco del torso para el cuerpo y en metros para los pies.
// El numero clave: si el error es comparable a la orden, el net esta ignorando la orden.
// Si el error es mucho mas chico que la orden, la esta obediguiendo.
function commandedMagnitude(start, t) {
  const spec = config.targets[t];
  const target = start.targets[t];
  if (spec.ride) {
    if (target.lx == null) return 0;
    const neutral = sim.neutralLocal(spec);
    return Math.hypot(target.lx - neutral.x, target.ly - neutral.y);
  }
  return Math.abs(target.dx || 0);
}

// Un controlador de referencia: NO es la red, es un piloto que lee el error y mueve la
// articulacion. Sirve para una sola pregunta: ¿la orden es fisicamente alcanzable?
//
// Antes este piloto solo tocaba el CUELLO (action[1]). Con el, los pies daban "corrige 0%"
// SIEMPRE, y eso no significaba que fueran imposibles: significaba que a nadie se le ocurrio
// mover una pierna. Un 0% de correccion tiene que significar "imposible", no "no medido".
//
// El piloto de piernas usa los joints 9..13 (cadera, rodilla, tobillo, una por pierna) y
// busca el mejor par de angulos con una rejilla, una sola vez, al principio del episodio:
// mueve la cadera si el pie va para el otro lado, la rodilla para levantarlo del piso.
const LEG_HIPS = [9, 11];
const LEG_KNEES = [10, 12];
const LEG_ANKLES = [13, 12]; // el tobillo derecho shares slot: se resuelve abajo

function oracleLegs(episode) {
  const original = episode.applyMotors.bind(episode);
  const part = config.partIndex;
  const feet = [
    { slot: config.targets.findIndex((s) => s.id === "footR"), joint: LEG_HIPS[0], knee: LEG_KNEES[0], ankle: 12 },
    { slot: config.targets.findIndex((s) => s.id === "footL"), joint: LEG_HIPS[1], knee: LEG_KNEES[1], ankle: 13 },
  ];
  let solved = false;
  episode.applyMotors = function () {
    if (!solved) {
      solved = true;
      const m = this.measure;
      for (const leg of feet) {
        const target = m.targets && m.targets[leg.slot];
        if (!target) continue;
        const footX = m.partX[leg.slot >= 0 ? part.footRight : part.footLeft] || 0;
        const err = (target.x || 0) - footX;
        // El pie va a la derecha: la cadera carga ese lado. A la izquierda, al reves.
        this.action[leg.joint] = Math.max(-1, Math.min(1, err * 6));
        // Hay que despegarlo: sin rodilla el pie raspa el piso y la friccion lo frena.
        this.action[leg.knee] = Math.max(-1, Math.min(1, -Math.abs(err) * 8 - 0.3));
        this.action[leg.ankle] = 0;
      }
    }
    original();
  };
  return () => {
    episode.applyMotors = original;
  };
}

// Un episode por semilla, siempre el mismo: el comparativo entre ablationes es pareado.
function episodeSpecs(n) {
  const list = [];
  for (let i = 0; i < n; i++) {
    const rand = makeSeededRandom(opts.seed + opts.skill * 10007 + i * 17);
    list.push(sim.fillTrainingStart({}, rand, {
      full: true,
      level: opts.skill,
      commands: opts.commands == null ? null : opts.commands,
    }));
  }
  return list;
}

const specs = episodeSpecs(opts.n);

// Los mundos se crean una sola vez y se reciclan con reset(). Box2D no aguanta
// que se le creen cientos en una corrida, y además es más rápido.
const pool = [];
for (let w = 0; w < Math.min(8, specs.length); w++)
  pool.push(new sim.Episode(sim.build(), theta, 0, 0, {}));

// El oraculo: el net hace TODO lo que sabe, y un controlador de una sola articulacion,
// proportional al error de angulo de la cabeza, reemplaza al cuello. Si con esto la cabeza
// obedece y el doll sigue parado, la tarea es aprendible y el problema es la busqueda.
// Si no, la tarea es imposible y hay que tocar la fisica, no el entrenamiento.
function oracleNeck(episode, gain) {
  const original = episode.applyMotors.bind(episode);
  const part = config.partIndex;
  episode.applyMotors = function () {
    const spec = config.targets[0];
    const measure = this.measure;
    const targets = measure.targets;
    if (targets && targets[0]) {
      const limit = 0.5; // El cuello va de -0.5 a 0.5 rad.
      const wanted = (targets[0].angle || 0) - (measure.torsoSin ? Math.atan2(measure.torsoSin, measure.torsoCos) : 0);
      this.action[1] = Math.max(-1, Math.min(1, (gain * wanted) / limit));
    }
    void part;
    original();
  };
  return () => {
    episode.applyMotors = original;
  };
}

function runPool(count) {
  const rows = [];
  for (let i = 0; i < count; i++) {
    const episode = pool[i % pool.length];
    const start = specs[i];
    const restore = opts.oracle ? oracleNeck(episode, opts.oracle) : null;
    const restoreLegs = opts.oracle === 2 || opts.legOracle ? oracleLegs(episode) : null;
    episode.reset(theta, start.tilt || 0, start.drop || 0, start);
    while (!episode.step());
    if (restore) restore();
    if (restoreLegs) restoreLegs();
    const stats = episode.stats;
    const measure = episode.measure;
    const targets = measure.targets;
    const marks = {};
    const aligns = {};
    const share = {};
    const commands = {};
    for (let t = 0; t < config.targets.length; t++) {
      const spec = config.targets[t];
      const index = config.partIndex[spec.part];
      const target = targets[t];
      const dx = (target.x || 0) - (measure.partX[index] || 0);
      const dy = (target.y || 0) - (measure.partY[index] || 0);
      marks[spec.id] = Math.hypot(dx, dy);
      commands[spec.id] = commandedMagnitude(start, t);
      const tsin = Math.sin(target.angle || 0);
      const tcos = Math.cos(target.angle || 0);
      aligns[spec.id] = tcos * (measure.partCos[index] || 1) + tsin * (measure.partSin[index] || 0);
      share[spec.id] = 0;
    }
    // Lo que cada fila aporta al premio en el frame final. Es el número que se tiene que mover.
    for (const row of scoreByTarget(measure)) share[row.id] = row.position + row.rotation;
    rows.push({
      fitness: stats.fitness,
      stood: stats.stoodUp ? 1 : 0,
      marks,
      aligns,
      share,
      commands,
      body: stats.bodyTargets || 0,
      foot: stats.foot || 0,
    });
    episode.cleanup();
    if (restore) restore();
  }
  return rows;
}

function mean(list) {
  return list.length ? list.reduce((a, b) => a + b, 0) / list.length : 0;
}

function report(title, rows) {
  const fit = mean(rows.map((row) => row.fitness));
  const stood = mean(rows.map((row) => row.stood));
  const foot = mean(rows.map((row) => row.foot));
  const body = mean(rows.map((row) => row.body));
  const cells = config.targets.map((spec) => {
    const mark = mean(rows.map((row) => row.marks[spec.id]));
    const share = mean(rows.map((row) => row.share[spec.id]));
    const asked = mean(rows.map((row) => row.commands[spec.id]));
    const best = targetPositionWeight(spec) + config.reward.targetRotation;
    const pct = best > 0 ? (share / best) * 100 : 0;
    const tol = spec.ride ? config.skill.targetReachMeters : config.skill.footReachMeters;
    const obey = rows.filter((row) => row.marks[spec.id] <= tol).length / (rows.length || 1);
    // Fraccion de la orden que el net realmente corrigio. 0 = la ignora, 1 = la obedece.
    const gain = asked > 1e-4 ? Math.max(0, 1 - mark / asked) : null;
    return (
      spec.id.padEnd(5) +
      ' orden ' + asked.toFixed(3) +
      '  err ' + mark.toFixed(3) +
      '  corrige ' + (gain == null ? '  -' : (gain * 100).toFixed(0) + '%') +
      '  obeye ' + (obey * 100).toFixed(0) + '%' +
      '  fila ' + (pct >= 0 ? '+' : '') + pct.toFixed(0) + '%'
    );
  });
  console.log(
    title.padEnd(16) +
      'fit ' + fit.toFixed(3).padStart(7) +
      '  stood ' + stood.toFixed(2) +
      '  foot ' + foot.toFixed(3) +
      '  body ' + body.toFixed(2) +
      '  | ' + cells.join(' | '),
  );
  return fit;
}

console.log('probe  skill ' + opts.skill + '  n ' + opts.n + '  comanda ' + (opts.commands || config.commands[opts.skill] || 'none') + '  red ' + source);
console.log('gains  ' + config.targets.map((s) => s.id + ' ' + targetGain(s.id)).join('  '));
console.log('spans  ' + config.targets.map((s) => s.id + ' ' + targetSpan(s)).join('  '));
console.log('');

const base = report('completa', runPool(specs.length));
console.log('');

const drops = {};
for (const spec of config.targets) {
  if (!targetGain(spec.id)) continue;
  config.net.mutedTargets = [spec.id];
  const fit = report('sin ' + spec.id, runPool(specs.length));
  drops[spec.id] = base - fit;
  config.net.mutedTargets = [];
}
config.net.mutedTargets = [];

console.log('');
console.log('caída de fitness al apagar cada target (positivo = la red lo usaba)');
const ranked = Object.keys(drops).sort((a, b) => drops[b] - drops[a]);
for (const id of ranked) {
  const share = base ? (drops[id] / base) * 100 : 0;
  const bar = '#'.repeat(Math.max(0, Math.min(40, Math.round(share))));
  console.log('  ' + id.padEnd(6) + drops[id].toFixed(3).padStart(8) + '  ' + share.toFixed(1).padStart(6) + '%  ' + bar);
}
const unused = ranked.filter((id) => drops[id] <= 1e-6);
console.log('');
console.log(unused.length ? 'sin uso: ' + unused.join(', ') : 'los cinco targets mueven el fitness');

// Cuánto pesa cada grupo de columnas en la red: RMS de los pesos de la primera capa que
// leen ese target, contra el RMS de toda esa capa. Alto = la red lo tiene cableado.
console.log('');
console.log('cableado en la primera capa (RMS de los pesos que leen esas columnas)');
const layout = ElmanNetwork.layout;
const stride = layout.feedforwardStride;
const firstTargetColumn = config.net.targetInputIndex;
let allSum = 0;
let allN = 0;
for (let u = 0; u < layout.feedforward; u++) {
  for (let c = 0; c < layout.inputCount; c++) {
    const w = theta[u * stride + 1 + c];
    allSum += w * w;
    allN++;
  }
}
const allRms = Math.sqrt(allSum / (allN || 1));
for (let t = 0; t < config.targets.length; t++) {
  const spec = config.targets[t];
  const use = rms(firstTargetColumn + t * 5);
  const err = rms(firstTargetColumn + t * 5 + 1, 4);
  // El atajo: los pesos que llevan el error del target directo a cada articulacion.
  const stride = layout.outputStride;
  const skipWeight = (joint, col) => theta[layout.outputBase + joint * stride + 1 + layout.recurrent + col];
  let skipSum = 0;
  let skipN = 0;
  for (let joint = 0; joint < 14; joint++) {
    for (let c = 1; c <= 4; c++) {
      const w = skipWeight(joint, t * 5 + c);
      skipSum += w * w;
      skipN++;
    }
  }
  const skipRms = Math.sqrt(skipSum / (skipN || 1));
  console.log(
    '  ' + spec.id.padEnd(6) +
      'use ' + (use / allRms).toFixed(2).padStart(5) + 'x' +
      '  error ' + (err / allRms).toFixed(2).padStart(5) + 'x' +
      '  atajo ' + (skipRms / allRms).toFixed(2).padStart(5) + 'x' +
      '   ' + barOf(skipRms / allRms),
  );
}

function rms(from, count) {
  let sum = 0;
  let n = 0;
  for (let u = 0; u < layout.feedforward; u++) {
    for (let c = from; c < from + (count == null ? 1 : count); c++) {
      const w = theta[u * stride + 1 + c];
      sum += w * w;
      n++;
    }
  }
  return Math.sqrt(sum / (n || 1));
}

function barOf(ratio) {
  return '#'.repeat(Math.max(0, Math.min(30, Math.round(ratio * 10))));
}