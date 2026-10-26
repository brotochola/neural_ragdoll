// One headless loop: same sim, net, reward, and config as the page.
// The ladder is commands, not steps: 1 stand, 2 the head to the command it is given,
// 3 the hands, 4 the feet, 5 all five at once. A level passes when the net is standing
// and that group's targets are inside their tolerance.
// Rates are config.eval.passRate. Official when --eval-n >= config.eval.officialMinN.
import { Worker, isMainThread, parentPort } from 'node:worker_threads';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { config, setTrainingPhase, setSkill } from './config.js';
import { isStanding, markError, targetGain } from './src/reward.js';

const root = path.dirname(fileURLToPath(import.meta.url));

if (!isMainThread) {
  await runWorker();
} else {
  await runMain();
}

async function runWorker() {
  const boxUrl = pathToFileURL(path.join(root, 'vendor/PhaserBox2D.min.js')).href;
  const Box2D = await import(boxUrl);
  const { createSim } = await import('./src/sim.js');
  const { isStanding } = await import('./src/reward.js');
  const sim = createSim(Box2D);
  let world = null;
  parentPort.postMessage({ type: 'ready' });
  parentPort.on('message', (msg) => {
    if (msg.type === 'phase') {
      setTrainingPhase(msg.phase);
      parentPort.postMessage({ type: 'phase', phase: msg.phase });
      return;
    }
    if (msg.type === 'scale') {
      config.skill.placeScale = msg.scale;
      parentPort.postMessage({ type: 'scale' });
      return;
    }
    if (msg.type === 'gains') {
      Object.assign(config.targetGains, msg.gains || {});
      parentPort.postMessage({ type: 'gains' });
      return;
    }
    if (msg.type === 'stop') {
      if (workerEpisode) workerEpisode.cleanup();
      workerEpisode = null;
      if (workerWorld) sim.destroyWorld(workerWorld);
      workerWorld = null;
      parentPort.postMessage({ type: 'stopped' });
      return;
    }
    if (msg.type !== 'run') return;
    try {
      const results = msg.jobs.map((job) => runEpisode(sim, job, isStanding));
      parentPort.postMessage({ type: 'done', id: msg.id, results });
    } catch (error) {
      parentPort.postMessage({ type: 'done', id: msg.id, error: String(error && error.stack || error) });
    }
  });
}

let workerEpisode = null;
let workerWorld = null;

function runEpisode(sim, job, isStanding) {
  const options = job.options || {};
  const weights = job.weights instanceof Float32Array ? job.weights : Float32Array.from(job.weights);
  // ONE world per worker for its whole life, rebuilt before every episode. A reset alone is not
  // enough: Box2D keeps per-body solver state across a reset, so a reused world answered
  // differently depending on how many episodes had run in it, and the official evaluation
  // changed with the worker count. See sim.runOneEpisode.
  if (!workerWorld) workerWorld = sim.build();
  else workerWorld.rebuild();
  workerEpisode = new sim.Episode(workerWorld, weights, options.tilt || 0, options.drop || 0, options);
  const startGap = options.goalX == null ? null : Math.abs(options.goalX - (workerEpisode.measure.centerX || 0));
  while (!workerEpisode.advanceOneFrame());
  const stats = workerEpisode.stats;
  const measure = workerEpisode.measure;
  const standing = isStanding(measure);
  const headHeight = measure.headHeight || 0;
  // Each target's error against its command, so obedience can be measured and not only "it was standing".
  const errors = {};
  for (let t = 0; t < config.targets.length; t++) {
    const error = markError(measure, t);
    errors[config.targets[t].id] = { dist: error.dist, align: error.align };
  }
  workerEpisode.cleanup();
  return {
    index: job.index,
    kind: job.kind,
    fitness: stats.fitness,
    start: stats.start,
    end: stats.end,
    stoodUp: stats.stoodUp,
    headImpact: stats.headImpact || 0,
    gap: stats.gap,
    standing,
    headHeight,
    headHigh: headHeight > config.body.standingHeadMeters,
    stance: measure.stanceWidth || 0,
    startGap,
    endGap: options.goalX == null ? null : stats.gap,
    goalX: options.goalX ?? null,
    windowStood: stats.windowStood || 0,
    windowFrames: stats.windowFrames || 0,
    steps: stats.steps || 0,
    footL: stats.footL,
    footR: stats.footR,
    alignL: stats.alignL,
    alignR: stats.alignR,
    cleared: stats.cleared || 0,
    alive: stats.alive || 0,
    foot: stats.foot || 0,
    headMark: stats.headMark,
    alignHead: stats.alignHead,
    handLMark: stats.handLMark,
    alignHandL: stats.alignHandL,
    handRMark: stats.handRMark,
    alignHandR: stats.alignHandR,
    bodyTargets: stats.bodyTargets || 0,
    errors,
    reach: stats.reach || 0,
    closed: stats.closed || 0,
    leaned: stats.leaned || 0,
    headHit: workerEpisode.headHit ? 1 : 0,
    walkDist: stats.walkDist || 0,
    walkCmd: stats.walkCmd || 0,
    climbOnTop: stats.climbOnTop || 0,
    climbHeight: stats.climbHeight || 0,
    onBoard: stats.onBoard || 0,
    isGetup: stats.isGetup || 0,
    isBoard: stats.isBoard || 0,
    trace: options.trace ? workerEpisode.trace : undefined,
  };
}

function readArgs() {
  const out = {
    fresh: false,
    gens: 6000,
    minutes: 45,
    evalN: config.eval.headlessN,
    quickN: config.eval.quickN,
    quickEvery: config.eval.quickEvery,
    pairs: config.workers.pairs,
    workers: Math.max(config.workers.headlessWorkerMin, Math.min(config.workers.headlessWorkerMax, os.cpus().length - config.workers.headlessWorkerReserve)),
    skill: config.skill.level || 1,
    evalOnly: false,
    resetMoments: false,
    gains: null,
    residualSearch: null,
    gateTargets: null,
    targetMean: null,
    from: null,
  };
  const argv = process.argv.slice(2);
  for (let i = 0; i < argv.length; i++) {
    const key = argv[i];
    const val = argv[i + 1];
    if (key === '--fresh') out.fresh = true;
    else if (key === '--gens') { out.gens = +val; i++; }
    else if (key === '--minutes') { out.minutes = +val; i++; }
    else if (key === '--eval-n') { out.evalN = +val; i++; }
    else if (key === '--quick-n') { out.quickN = +val; i++; }
    else if (key === '--quick-every') { out.quickEvery = +val; i++; }
    else if (key === '--pairs') { out.pairs = +val; i++; }
    else if (key === '--workers') { out.workers = +val; i++; }
    else if (key === '--skill') { out.skill = +val; out.skillSet = true; i++; }
    else if (key === '--eval-only') out.evalOnly = true;
    else if (key === '--trace') out.trace = true;
    else if (key === '--reset-moments') out.resetMoments = true;
    // --gains hand=0.5, or --gains body=0 to drop the head and hands, or feet=0 for the feet.
    else if (key === '--gains') {
      const aliases = {
        head: ['head'],
        handR: ['handR'],
        handL: ['handL'],
        hand: ['handR', 'handL'],
        footR: ['footR'],
        footL: ['footL'],
        foot: ['footR', 'footL'],
        feet: ['footR', 'footL'],
        body: ['head', 'handR', 'handL'],
        all: ['head', 'handR', 'handL', 'footR', 'footL'],
      };
      const gains = {};
      for (const piece of String(val || '').split(',')) {
        const [name, value] = piece.split('=');
        const key2 = String(name || '').trim();
        if (!key2) continue;
        for (const id of aliases[key2] || [key2]) gains[id] = +value;
      }
      out.gains = gains;
      i++;
    } else if (key === '--residual-search') out.residualSearch = true;
    else if (key === '--from') { out.from = val; i++; }
    else if (key === '--gate-targets') out.gateTargets = true;
    else if (key === '--target-mean') { out.targetMean = +val; i++; }
    else if (key === '--no-ramp') out.noRamp = true; // Full-length training episodes from the first generation.
    else if (key === '--board-cap') { out.boardCap = +val; i++; } // Resume the board amplitude curriculum here.
    else if (key === '--getup-assist') { out.getupAssist = +val; i++; } // Resume the helping-hand curriculum at this level.
    else if (key === '--getup-cap') { out.getupCap = +val; i++; } // Resume the get-up curriculum at this depth.
    else if (key === '--climb-cap') { out.climbCap = +val; i++; } // Resume a box-height curriculum at this height.
    else if (key === '--only') out.only = true; // Stop after the starting skill instead of climbing the ladder.
  }
  return out;
}

function makePool(count) {
  let seq = 1;
  const workers = [];
  const queue = [];
  function pump() {
    for (const slot of workers) {
      if (slot.busy || !slot.ready || !queue.length) continue;
      const task = queue.shift();
      slot.busy = true;
      slot.current = task;
      slot.worker.postMessage({ type: 'run', id: task.id, jobs: task.jobs });
    }
  }
  const ready = [];
  for (let i = 0; i < count; i++) {
    const worker = new Worker(new URL(import.meta.url));
    const slot = { worker, busy: false, ready: false, current: null, phaseWait: null };
    worker.on('message', (msg) => {
      if (msg.type === 'ready') {
        slot.ready = true;
        if (ready[i]) ready[i]();
        pump();
        return;
      }
      if (msg.type === 'phase' || msg.type === 'scale' || msg.type === 'gains') {
        if (slot.phaseWait) { const done = slot.phaseWait; slot.phaseWait = null; done(); }
        return;
      }
      if (msg.type === 'done' && slot.current && slot.current.id === msg.id) {
        const task = slot.current;
        slot.current = null;
        slot.busy = false;
        if (msg.error) task.reject(new Error(msg.error));
        else task.resolve(msg.results);
        pump();
      }
    });
    worker.on('error', (error) => {
      if (slot.current) slot.current.reject(error);
    });
    workers.push(slot);
  }
  function whenReady() {
    return Promise.all(workers.map((slot, i) => slot.ready ? null : new Promise((resolve) => { ready[i] = resolve; })));
  }
  function run(jobs) {
    const groups = Array.from({ length: count }, () => []);
    jobs.forEach((job, index) => groups[index % count].push(job));
    const tasks = groups.filter((group) => group.length).map((group) => new Promise((resolve, reject) => {
      queue.push({ id: seq++, jobs: group, resolve, reject });
    }));
    pump();
    const done = Promise.all(tasks).then((parts) => parts.flat().sort((a, b) => a.index - b.index));
    const timeout = new Promise((_resolve, reject) => setTimeout(() => reject(new Error('worker batch timed out')), 300000));
    return Promise.race([done, timeout]);
  }
  function setPhase(phase) {
    return Promise.all(workers.map((slot) => new Promise((resolve) => {
      slot.phaseWait = resolve;
      slot.worker.postMessage({ type: 'phase', phase });
    })));
  }
  function setScale(scale) {
    return Promise.all(workers.map((slot) => new Promise((resolve) => {
      slot.phaseWait = resolve;
      slot.worker.postMessage({ type: 'scale', scale });
    })));
  }
  function setGains(gains) {
    return Promise.all(workers.map((slot) => new Promise((resolve) => {
      slot.phaseWait = resolve;
      slot.worker.postMessage({ type: 'gains', gains });
    })));
  }
  async function stop() {
    await Promise.race([
      Promise.all(workers.map((slot) => new Promise((resolve) => {
        const timer = setTimeout(resolve, 2000);
        function onStop(msg) {
          if (!msg || msg.type !== 'stopped') return;
          slot.worker.off('message', onStop);
          clearTimeout(timer);
          resolve();
        }
        slot.worker.on('message', onStop);
        slot.worker.postMessage({ type: 'stop' });
      }))),
      new Promise((resolve) => setTimeout(resolve, 2000)),
    ]);
    await Promise.all(workers.map((slot) => Promise.resolve(slot.worker.terminate()).catch(() => {})));
  }
  return { whenReady, run, setPhase, setScale, setGains, stop };
}

// Which groups this level commands. 'none' is standing, 'all' is the final exam.
function commandGroup(skill) {
  return config.commands[skill | 0] || 'none';
}

// Which group each target is. 'hand' is both arms.
function groupOfId(id) {
  return id === 'head' ? 'head' : id === 'footR' || id === 'footL' ? 'foot' : 'hand';
}

// A target "obeys" when it is inside its tolerance. The tolerances are the same idea for
// all five. The ones that receive no command sit on the neutral, so they pass on their
// own: each level's bar is the new group, and 'all' is all five.
function obeys(row, spec) {
  if (!targetGain(spec.id)) return true;
  const error = row.errors && row.errors[spec.id];
  if (!error) return false;
  const useReach = spec.ride ? config.skill.targetReachMeters : config.skill.footReachMeters;
  const useRadians = spec.ride ? config.skill.targetReachRadians : config.skill.footReachRadians;
  return error.dist <= useReach && error.align >= Math.cos(useRadians);
}

function skillPass(skill, row) {
  const group = commandGroup(skill);
  if (group === 'highbox') {
    // Both feet on top of the tall box at the end, upright over it, standing half the last window.
    const frames = row.windowFrames || 0;
    if (!(frames > 0) || !row.climbOnTop) return false;
    if ((row.headHeight || 0) - (row.climbHeight || 0) < config.body.standingHeadMeters) return false;
    return row.windowStood / frames >= config.highbox.standShare;
  }
  if (group === 'board') {
    // On the plank: up at the end, standing for most of the last window, head never down.
    const frames = row.windowFrames || 0;
    if (!(frames > 0) || !row.stoodUp || row.headHit) return false;
    return row.windowStood / frames >= config.board.standShare;
  }
  if (group === 'getup') {
    // From lying: up at the end and standing for most of the last window. No headHit check:
    // the head starts on the floor.
    const frames = row.windowFrames || 0;
    if (!(frames > 0) || !row.stoodUp) return false;
    return row.windowStood / frames >= config.getup.standShare;
  }
  if (group === 'climb') {
    // Both feet on top of the box at the end, upright over it, standing for half the last window, and
    // the head never touched anything on the way.
    const frames = row.windowFrames || 0;
    if (!(frames > 0) || row.headHit || !row.climbOnTop) return false;
    if ((row.headHeight || 0) - (row.climbHeight || 0) < config.body.standingHeadMeters) return false;
    return row.windowStood / frames >= 0.5;
  }
  if (group === 'walk') {
    // Walked without falling: the head never touched the floor, it is high at the end, the doll
    // was standing for most of the last window, and the torso kept up with the marks.
    // Not row.stoodUp (15 straight standing frames at the end): a stride leans past 15 degrees
    // for a frame or two, and that broke the streak in 80% of episodes that were walking fine
    // (measured: standing 86% of the last 0.5 s, the misses mostly a brief tilt).
    const frames = row.windowFrames || 0;
    if (!(frames > 0) || row.headHit || !row.headHigh) return false;
    if (row.windowStood / frames < config.walk.standShare) return false;
    return row.walkCmd > 0 && row.walkDist >= config.walk.passShare * row.walkCmd;
  }
  if (group === 'none') {
    // "Standing" is a FRACTION of the final window, not all 180 frames with none missing.
    // It used to be windowStood === windowFrames: one stumble in three seconds failed the
    // episode, so the bar measured "never blinks" rather than "stands". See config.eval.standWindowShare.
    const frames = row.windowFrames || 0;
    if (!(frames > 0) || !row.stoodUp) return false;
    return row.windowStood / frames >= config.eval.standWindowShare;
  }
  if (!row.stoodUp) return false;
  for (const spec of config.targets) {
    if (group !== 'all' && groupOfId(spec.id) !== group) continue;
    if (!obeys(row, spec)) return false;
  }
  return true;
}

function skillKey(skill) {
  return ['', 'stand', 'head', 'hand', 'foot', 'all', 'walk', 'mix', 'climb', 'getup', 'board', 'highbox', 'sequence'][skill] || 'stand';
}

function targetGainText() {
  return config.targets
    .map((spec) => spec.id + ' ' + (config.targetGains[spec.id] || 0))
    .join(' ');
}

function skillOk(kind, row) {
  const quiet = (row.headImpact || 0) < config.search.headImpactQuiet;
  const stood = !!row.standing && !!row.headHigh;
  if (kind === 'stand' || kind === 'drop') return stood && quiet;
  if (kind === 'ball' || kind === 'supine' || kind === 'prone') return !!row.standing;
  if (kind === 'walk') {
    const closed = row.startGap != null && row.endGap < row.startGap - config.eval.walkClosingMeters;
    return !!row.standing && closed;
  }
  return false;
}

function wilson(ok, n) {
  if (!n) return [0, 0];
  const z = 1.96;
  const p = ok / n;
  const den = 1 + (z * z) / n;
  const center = (p + (z * z) / (2 * n)) / den;
  const margin = (z * Math.sqrt((p * (1 - p) + (z * z) / (4 * n)) / n)) / den;
  return [Math.max(0, center - margin), Math.min(1, center + margin)];
}

function summarize(kind, rows) {
  let ok = 0;
  let standing = 0;
  let headSum = 0;
  let impactSum = 0;
  let closed = 0;
  let stanceSum = 0;
  let headMarkSum = 0;
  let handMarkSum = 0;
  let bodySum = 0;
  let bodyN = 0;
  for (const row of rows) {
    if (row.standing) standing++;
    if (skillOk(kind, row)) ok++;
    headSum += row.headHeight || 0;
    impactSum += row.headImpact || 0;
    stanceSum += row.stance || 0;
    headMarkSum += row.headMark == null ? 0 : row.headMark;
    handMarkSum += (row.handLMark || 0) + (row.handRMark || 0);
    if (row.bodyTargets) {
      bodySum += row.bodyTargets;
      bodyN++;
    }
    if (row.startGap != null && row.endGap < row.startGap - config.eval.walkClosingMeters) closed++;
  }
  const n = rows.length;
  const band = wilson(ok, n);
  return {
    n,
    ok,
    rate: n ? ok / n : 0,
    standingRate: n ? standing / n : 0,
    closedRate: n ? closed / n : 0,
    meanHead: n ? headSum / n : 0,
    meanImpact: n ? impactSum / n : 0,
    meanStance: n ? stanceSum / n : 0,
    meanHeadMark: n ? headMarkSum / n : 0,
    meanHandMark: n ? handMarkSum / (2 * n) : 0,
    bodyRate: bodyN ? bodySum / bodyN : 0,
    wilsonLow: band[0],
    wilsonHigh: band[1],
  };
}

function watchKinds(phase) {
  if (phase <= 1) return ['stand', 'ball', 'drop'];
  if (phase === 2) return ['stand', 'ball', 'drop', 'supine', 'prone'];
  return ['stand', 'ball', 'drop', 'supine', 'prone', 'walk'];
}

function gateKeys(phase) {
  if (phase <= 1) return ['stand', 'ball', 'drop'];
  if (phase === 2) return ['supine', 'prone'];
  return ['walk'];
}

function passed(phase, ev) {
  return gateKeys(phase).every((key) => ev[key] && ev[key].rate >= config.eval.passRate[key]);
}

function nearPass(phase, ev) {
  return gateKeys(phase).every((key) => ev[key] && ev[key].rate >= config.eval.passRate[key] - config.eval.nearPassSlack);
}

function phaseScore(phase, ev) {
  return gateKeys(phase).reduce((sum, key) => sum + (ev[key] ? ev[key].rate : 0), 0);
}

function fmtEval(ev) {
  return Object.entries(ev).map(([key, row]) => key + ' ' + row.ok + '/' + row.n + '=' + row.rate.toFixed(2) + ' st' + (row.meanStance || 0).toFixed(2)).join(' ');
}

async function runMain() {
  const opts = readArgs();
  // The five targets: how much each group weighs in the reward, and how much the whole
  // group weighs in the fitness. --gains tries a subset without editing the file.
  if (opts.gains) Object.assign(config.targetGains, opts.gains);
  if (opts.residualSearch) config.skill.residualSearch = true;
  if (opts.gateTargets) config.eval.gateTargets = true;
  if (opts.targetMean != null && Number.isFinite(opts.targetMean)) config.reward.targetMean = opts.targetMean;
  if (opts.noRamp) config.skill.rampGenerations = 0;
  const backupDir = path.join(root, 'backups');
  fs.mkdirSync(backupDir, { recursive: true });
  const logPath = path.join(backupDir, 'train-log.txt');
  const statePath = path.join(backupDir, 'train-state.json');
  const statusPath = path.join(backupDir, 'status.json');
  const { ElmanNetwork, EvolutionSearch, ResidualSearch, setOutputJoints } = await import('./src/net.js');
  const { makeSeededRandom, randomNormalSample } = await import('./src/utils.js');
  const boxUrl = pathToFileURL(path.join(root, 'vendor/PhaserBox2D.min.js')).href;
  const Box2D = await import(boxUrl);
  const { createSim } = await import('./src/sim.js');
  const { runGeneration, trainingMode, evalMode } = await import('./src/training.js');
  const sim = createSim(Box2D);
  const random = makeSeededRandom(7);
  // The recorded falls for the get-up reverse curriculum, if the file is there.
  const fallsPath = path.join(root, config.getup.fallsFile || 'getup-falls.json');
  const getupFalls = fs.existsSync(fallsPath) ? JSON.parse(fs.readFileSync(fallsPath, 'utf8')).falls : null;
  const randomNormal = () => randomNormalSample(random);
  const weightCount = ElmanNetwork.weightCount;
  const dropMeters = (config.start.fallDropMinMeters + config.start.fallDropMaxMeters) / 2;
  const pool = makePool(opts.workers);
  const evalLog = [];
  const curve = [];
  const bestFiles = [];
  let logFile = logPath;
  let reportedParentMoves = 0;

  function log(line) {
    console.log(line);
    fs.appendFileSync(logFile, line + '\n');
  }

  function saveBackup(role, state, extra) {
    const score = Number.isFinite(state.bestEverScore) ? state.bestEverScore : 0;
    const safe = score.toFixed(3).replace(/-/g, 'm').replace('.', 'p');
    let name = 'g' + state.gen + '_s' + safe + '_' + role + '_' + Date.now() + '.json';
    let full = path.join(backupDir, name);
    let n = 2;
    while (fs.existsSync(full)) {
      name = 'g' + state.gen + '_s' + safe + '_' + role + '_' + Date.now() + '_' + n + '.json';
      full = path.join(backupDir, name);
      n++;
    }
    const payload = Object.assign({
      role,
      file: name,
      gen: state.gen,
      bestEverScore: state.bestEverScore,
      trainingPhase: state.phase,
      inputCount: config.net.inputCount,
      theta: Array.from(state.theta),
      bestEverTheta: Array.from(state.bestEverTheta),
      residual: state.residualSearch ? Array.from(state.residualSearch.theta) : null,
      shiftMeters: state.shiftMeters || 0,
    }, extra || {});
    fs.writeFileSync(full, JSON.stringify(payload));
    if (role === 'best-ever' || role === 'phase') bestFiles.push(name);
    return name;
  }

  function saveTrainState(state) {
    const pack = (weights) => Array.from(weights);
    fs.writeFileSync(statePath, JSON.stringify({
      gen: state.gen,
      k: state.k || 0,
      phase: state.phase,
      drill: state.drill || 0,
      fitnessVersion: config.reward.version,
      bestEverScore: state.bestEverScore,
      bestSmooth: Number.isFinite(state.bestSmooth) ? state.bestSmooth : null,
      eliteScore: state.eliteScore,
      parentMoves: state.parentMoves || 0,
      parentMoveNote: state.parentMoveNote || "",
      flatGens: state.flatGens || 0,
      skill: state.skill || 1,
      climbMax: state.climbMax || 0,
      boardMax: state.boardMax || 0,
      getupMax: state.getupMax || 0,
      getupAssist: state.getupAssist == null ? null : state.getupAssist,
      placeScale: state.placeScale || 0,
      footRatio: state.footRatio > 0 ? state.footRatio : 0,
      heldFoot: state.heldFoot > 0 ? state.heldFoot : 0,
      shiftMeters: state.shiftMeters == null ? null : state.shiftMeters,
      residual: state.residualSearch ? pack(state.residualSearch.theta) : null,
      residualM: state.residualSearch ? pack(state.residualSearch.m) : null,
      residualV: state.residualSearch ? pack(state.residualSearch.v) : null,
      residualK: state.residualSearch ? state.residualSearch.k : 0,
      bestResidual: state.bestResidual || null,
      bestFootRate: state.bestFootRate || 0,
      bestFootErr: state.bestFootErr || 0,
      theta: pack(state.theta),
      bestEverTheta: pack(state.bestEverTheta),
      m: pack(state.m),
      v: pack(state.v),
      curve: curve.slice(-2000),
      evalLog,
      bestFiles,
    }));
  }

  function loadPolicy() {
    // --from loads an explicit checkpoint. That is how a run continues from a net that
    // already passed a bar, instead of starting fresh or resuming the last train-state.
    if (opts.from) {
      const saved = JSON.parse(fs.readFileSync(opts.from, 'utf8'));
      const raw = saved.theta;
      const theta = ElmanNetwork.copy(raw);
      if (!theta) throw new Error(opts.from + ' tiene ' + (raw ? raw.length : 0) + ' pesos, la red pide ' + ElmanNetwork.weightCount);
      const state = new EvolutionSearch(randomNormal);
      state.theta = theta;
      state.bestEverTheta = theta.slice();
      state.bestGenTheta = theta.slice();
      state.gen = 0;
      state.phase = config.curriculum.phases[0].id;
      state.drill = 0;
      state.bestEverScore = -1e9;
      state.eliteScore = -1e9;
      return { state, source: path.basename(opts.from), rawLength: raw.length };
    }
    const file = fs.existsSync(path.join(root, 'checkpoint.json')) ? 'checkpoint.json' : 'weights.json';
    const data = JSON.parse(fs.readFileSync(path.join(root, file), 'utf8'));
    const raw = data.bestTheta || data.theta;
    const theta = ElmanNetwork.copy(raw);
    if (!theta) {
      log('weights length ' + (raw && raw.length) + ' in ' + file + ' does not match this net, fresh weights');
      const fresh = new EvolutionSearch(randomNormal);
      fresh.phase = config.curriculum.phases[0].id;
      fresh.skill = opts.skill || 1;
      fresh.bestEverScore = -1e9;
      fresh.eliteScore = -1e9;
      return { state: fresh, source: 'fresh', rawLength: raw ? raw.length : 0 };
    }
    const state = new EvolutionSearch(randomNormal);
    state.theta = theta;
    state.bestEverTheta = theta.slice();
    state.bestGenTheta = theta.slice();
    state.gen = 0;
    state.phase = config.curriculum.phases[0].id;
    state.drill = 0;
    state.bestEverScore = -1e9;
    state.eliteScore = -1e9;
    return { state, source: file, rawLength: raw.length };
  }

  function loadTrainState() {
    if (!fs.existsSync(statePath)) return null;
    const saved = JSON.parse(fs.readFileSync(statePath, 'utf8'));
    if (!saved || !saved.theta || saved.theta.length !== weightCount) return null;
    const state = new EvolutionSearch(randomNormal);
    state.theta = Float32Array.from(saved.theta);
    state.bestEverTheta = Float32Array.from(saved.bestEverTheta || saved.theta);
    state.bestGenTheta = state.theta.slice();
    state.m = Float32Array.from(saved.m || []);
    state.v = Float32Array.from(saved.v || []);
    if (state.m.length !== weightCount) state.m = new Float32Array(weightCount);
    if (state.v.length !== weightCount) state.v = new Float32Array(weightCount);
    state.k = saved.k || 0;
    state.gen = saved.gen || 0;
    state.phase = saved.phase || config.curriculum.phases[0].id;
    state.skill = saved.skill || opts.skill || 1;
    state.climbMax = saved.climbMax || 0;
    state.boardMax = saved.boardMax || 0;
    state.getupMax = saved.getupMax || 0;
    state.getupAssist = saved.getupAssist == null ? null : saved.getupAssist;
    state.placeScale = saved.placeScale || 0;
    state.footRatio = saved.footRatio > 0 ? saved.footRatio : 0;
    state.heldFoot = saved.heldFoot > 0 ? saved.heldFoot : 0;
    state.shiftMeters = Number.isFinite(+saved.shiftMeters) ? +saved.shiftMeters : null;
    if (saved.residual && saved.residual.length) {
      state.residualSearch = new ResidualSearch(randomNormal);
      if (saved.residual.length === state.residualSearch.theta.length) {
        state.residualSearch.theta.set(Float32Array.from(saved.residual));
        if (saved.residualM && saved.residualM.length === state.residualSearch.m.length)
          state.residualSearch.m.set(Float32Array.from(saved.residualM));
        if (saved.residualV && saved.residualV.length === state.residualSearch.v.length)
          state.residualSearch.v.set(Float32Array.from(saved.residualV));
        state.residualSearch.k = saved.residualK || 0;
      }
    }
    if (saved.bestResidual && saved.bestResidual.length) {
      state.bestResidual = saved.bestResidual;
      state.bestFootRate = saved.bestFootRate || 0;
      state.bestFootErr = saved.bestFootErr || 0;
    }
    state.drill = saved.drill || 0;
    state.bestEverScore = saved.bestEverScore ?? -1e9;
    if (Number.isFinite(+saved.bestSmooth)) state.bestSmooth = +saved.bestSmooth;
    state.eliteScore = saved.eliteScore ?? -1e9;
    state.parentMoves = saved.parentMoves || 0;
    state.parentMoveNote = saved.parentMoveNote || '';
    reportedParentMoves = state.parentMoves;
    if (Number.isFinite(+saved.flatGens) && +saved.flatGens > 0) state.flatGens = Math.floor(+saved.flatGens);
    state.refreshMutationBoost();
    if (Array.isArray(saved.curve)) curve.push(...saved.curve);
    if (Array.isArray(saved.evalLog)) evalLog.push(...saved.evalLog);
    if (Array.isArray(saved.bestFiles)) bestFiles.push(...saved.bestFiles);
    // The fitness changed scale: a saved bestEverScore is no longer comparable.
    // It is thrown out and reseeded, or the parent never beats its own record.
    const version = saved.fitnessVersion;
    if (version !== config.reward.version) {
      log('fitness version ' + (version == null ? 'none' : version) + ' -> ' + config.reward.version
        + ', previous record (' + state.bestEverScore + ') is not comparable, reseeding');
      state.bestEverScore = -1e9;
      state.bestSmooth = null;
      state.eliteScore = -1e9;
      state.flatGens = 0;
      state.reseed = true;
      state.m.fill(0);
      state.v.fill(0);
      state.k = 0;
    }
    return state;
  }

  function evalJobs(weights, kinds, n) {
    const jobs = [];
    let index = 0;
    for (const kind of kinds) {
      for (let i = 0; i < n; i++) {
        const rand = makeSeededRandom(config.eval.seed + kind.length * 1000 + i * 17);
        const jitter = () => rand() * 2 - 1;
        const side = i % 2 === 0 ? 1 : -1;
        const options = {
          amp: 0,
          period: config.body.platformPeriodSeconds,
          phaseShift: 0,
          projectiles: [],
          frames: config.episode.frames,
          phase: i % config.cadence.actionEveryNFrames,
          goalX: side * config.eval.walkGoalMeters,
          goalY: 0,
          shiftX: kind === 'walk' ? 0 : jitter() * config.eval.shiftMeters,
          // Without this the episode plants nothing and the reward rows look at the
          // blueprint, not the start pose. The eval measures the same way, but the fitness came out odd.
          failOnHead: true,
          skill: 0,
          // Flat marks, nothing drawn: the list is built by hand so rand is not advanced
          // and the eval set stays the one it was.
          targets: config.targets.map(() => ({ on: true, x: 0, y: 0, angle: 0, dx: 0, dy: 0, turn: 0 })),
        };
        if (kind === 'supine' || kind === 'prone') {
          const sign = kind === 'prone' ? -1 : 1;
          options.tilt = sign * (config.start.downTiltMinRadians + rand() * config.start.downTiltSpanRadians);
          options.drop = config.start.downDropMeters;
          options.settle = config.start.downSettleFrames;
        } else {
          options.tilt = jitter() * config.start.uprightTiltRadians;
          options.drop = kind === 'drop' ? dropMeters : config.start.uprightDropMeters;
          options.settle = 0;
        }
        if (kind === 'ball') {
          options.ballFrame = config.start.ballThrowFrame;
          options.ballSide = side;
          options.ballSpeed = config.start.ballSpeed;
        }
        jobs.push({ index, kind, weights, options });
        index++;
      }
    }
    return jobs;
  }

  async function evaluate(weights, kinds, n) {
    const rows = await pool.run(evalJobs(weights, kinds, n));
    if (rows.some((row) => !row || !Number.isFinite(row.fitness))) throw new Error('eval returned an empty score');
    const out = {};
    for (const kind of kinds) out[kind] = summarize(kind, rows.filter((row) => row.kind === kind));
    return out;
  }

  async function oneGeneration(state) {
    const before = state.bestEverScore;
    // residualSearch freezes the main net and only tunes the small leg net. That net reads
    // twelve hand-built numbers and never sees a target column, so it can never be the one
    // learning to aim a hand. config.skill.residualSearch turns the old behaviour back on.
    const useResidual = config.skill.residualSearch && (state.skill | 0) >= 2 && state.residualSearch;
    const { generation, scores } = await runGeneration({
      ask: () => useResidual
        ? state.residualSearch.ask(opts.pairs, randomNormal)
        : state.ask(null, opts.pairs, randomNormal),
      random,
      mode: Object.assign(trainingMode(state.skill | 0, state.gen, state.shiftMeters || 0), {
        climbMax: state.climbMax || 0,
        getupMax: state.getupMax || 0,
        getupAssist: state.getupAssist || 0,
        boardMax: state.boardMax || 0,
        getupFalls,
      }),
      runJobs: (jobs) => pool.run(jobs),
      decorate(job, child, asked) {
        if (useResidual) {
          job.weights = state.theta;
          job.options.residual = asked.weights[child];
        } else if (state.residualSearch) job.options.residual = state.residualSearch.theta;
      },
      tell(asked, rows) {
        if (!useResidual) {
          state.tell(asked, rows);
          return;
        }
        state.residualSearch.tell(asked, rows);
        let mean = 0;
        let best = -Infinity;
        let stoodN = 0;
        for (let i = 0; i < rows.length; i++) {
          mean += rows[i].fitness;
          if (rows[i].fitness > best) best = rows[i].fitness;
          if (rows[i].stoodUp) stoodN++;
        }
        mean /= rows.length || 1;
        state.hist.push({ mean, best, stood: stoodN / (rows.length || 1), start: 0 });
        if (state.hist.length > 400) state.hist.shift();
        state.gen++;
      },
    });
    if (state.anchorNote) {
      log(state.anchorNote);
      state.anchorNote = "";
    }
    if ((state.mutationBoost || 1) > (generation.boost || 1)) {
      const pct = Math.round((state.mutationFraction || 0) * 100);
      const share = pct > 0 ? ' ' + pct + '%' : '';
      log('mutation x' + state.mutationBoost + share + ' after ' + state.flatGens + ' generations without a better score');
    }
    const last = state.hist[state.hist.length - 1];
    const meanStance = scores.reduce((sum, row) => sum + (row.stance || 0), 0) / scores.length;
    const meanAlive = scores.reduce((sum, row) => sum + (row.alive || 0), 0) / scores.length;
    const meanFoot = scores.reduce((sum, row) => sum + (row.foot || 0), 0) / scores.length;
    const meanClosed = scores.reduce((sum, row) => sum + (row.closed || 0), 0) / scores.length;
    let meanLeaned = 0;
    let stoodLeanN = 0;
    let bestLean = 0;
    let bestClosed = 0;
    let bestSwing = 0;
    for (let i = 0; i < scores.length; i++) {
      const row = scores[i];
      meanLeaned += row.leaned || 0;
      if ((row.closed || 0) > bestSwing) bestSwing = row.closed || 0;
      if (!row.stoodUp) continue;
      stoodLeanN++;
      if ((row.leaned || 0) > bestLean) bestLean = row.leaned || 0;
      if ((row.closed || 0) > bestClosed) bestClosed = row.closed || 0;
    }
    meanLeaned /= scores.length;
    const meanSteps = scores.reduce((sum, row) => sum + (row.steps || 0), 0) / scores.length;
    let meanWalk = 0;
    let bestWalk = -Infinity;
    for (const row of scores) {
      meanWalk += row.walkDist || 0;
      if ((row.walkDist || 0) > bestWalk) bestWalk = row.walkDist || 0;
    }
    meanWalk /= scores.length;
    // BOARD curriculum: a larger tilt amplitude once the doll stays up on the current one.
    if ((state.skill | 0) === 10) {
      const b = config.board;
      if (!(state.boardMax > 0)) state.boardMax = b.startCap;
      let tried = 0;
      let made = 0;
      for (const row of scores) {
        if (!row.isBoard) continue;
        tried++;
        if (row.stoodUp && !row.headHit) made++;
      }
      state.boardLog = state.boardLog || [];
      state.boardLog.push([made, tried]);
      if (state.boardLog.length > b.capWindow) state.boardLog.shift();
      const sumMade = state.boardLog.reduce((a, r) => a + r[0], 0);
      const sumTried = state.boardLog.reduce((a, r) => a + r[1], 0);
      state.boardRate = sumTried ? sumMade / sumTried : 0;
      if (state.boardLog.length >= b.capWindow && state.boardRate >= b.capPassRate && state.boardMax < b.ampMax) {
        state.boardMax = Math.min(b.ampMax, state.boardMax + b.capStep);
        state.boardLog.length = 0;
        log('board amp cap -> ' + state.boardMax.toFixed(2) + ' rad (recent board stands ' + Math.round(state.boardRate * 100) + '%)');
      }
    }
    // GET-UP curriculum: start further down the recorded falls when get-ups succeed often enough.
    if ((state.skill | 0) === 9 && (getupFalls || config.getup.useAssist)) {
      const g = config.getup;
      if (!(state.getupMax > 0)) state.getupMax = g.startCap;
      if (state.getupAssist == null) state.getupAssist = g.useAssist ? g.assistStart : 0;
      let tried = 0;
      let made = 0;
      for (const row of scores) {
        if (!row.isGetup) continue;
        tried++;
        // Harness: a get-up counts when the head ends high, so the help can fade while the net
        // keeps getting up with less of it. Hung in the harness the doll never met the strict
        // standing test (both feet down, under 15 degrees), and the help never faded. The eval,
        // with no help, still asks for real standing.
        if (g.useAssist ? (row.headHeight || 0) >= g.harnessHeadMeters : row.stoodUp) made++;
      }
      state.getupLog = state.getupLog || [];
      state.getupLog.push([made, tried]);
      if (state.getupLog.length > g.capWindow) state.getupLog.shift();
      const sumMade = state.getupLog.reduce((a, r) => a + r[0], 0);
      const sumTried = state.getupLog.reduce((a, r) => a + r[1], 0);
      state.getupRate = sumTried ? sumMade / sumTried : 0;
      if (g.useAssist) {
        if (state.getupLog.length >= g.capWindow && state.getupRate >= g.capPassRate && state.getupAssist > 0) {
          state.getupAssist = Math.max(0, state.getupAssist - g.assistStep);
          state.getupLog.length = 0;
          log('getup assist -> ' + state.getupAssist.toFixed(2) + ' x body weight (recent get-ups ' + Math.round(state.getupRate * 100) + '%)');
        }
      } else if (state.getupLog.length >= g.capWindow && state.getupRate >= g.capPassRate && state.getupMax < g.maxCap) {
        state.getupMax = Math.min(g.maxCap, state.getupMax + g.capStep);
        state.getupLog.length = 0;
        log('getup cap -> ' + state.getupMax.toFixed(2) + ' of the way down (recent get-ups ' + Math.round(state.getupRate * 100) + '%)');
      }
    }
    // CLIMB curriculum: raise the box cap when the recent climb episodes succeed often enough.
    if ((state.skill | 0) === 8 || (state.skill | 0) === 11) {
      const c = (state.skill | 0) === 11 ? config.highbox : config.climb;
      if (!(state.climbMax > 0)) state.climbMax = c.startCap;
      let tried = 0;
      let made = 0;
      for (const row of scores) {
        if (!(row.climbHeight > 0)) continue;
        tried++;
        if (row.climbOnTop && !row.headHit) made++;
      }
      state.climbLog = state.climbLog || [];
      state.climbLog.push([made, tried]);
      if (state.climbLog.length > c.capWindow) state.climbLog.shift();
      const sumMade = state.climbLog.reduce((a, r) => a + r[0], 0);
      const sumTried = state.climbLog.reduce((a, r) => a + r[1], 0);
      state.climbRate = sumTried ? sumMade / sumTried : 0;
      if (state.climbLog.length >= c.capWindow && state.climbRate >= c.capPassRate && state.climbMax < c.maxCap) {
        state.climbMax = Math.min(c.maxCap, state.climbMax + c.capStep);
        state.climbLog.length = 0;
        log('climb cap -> ' + state.climbMax.toFixed(2) + ' m (recent climbs ' + Math.round(sumMade / sumTried * 100) + '%)');
      }
    }
    if (last) curve.push({ gen: state.gen, mean: last.mean, best: last.best, start: last.start, stood: last.stood, gap: last.gap, stance: meanStance, alive: meanAlive, foot: meanFoot, steps: meanSteps, walk: meanWalk, bestWalk, closed: meanClosed, leaned: meanLeaned, bestLean, bestClosed, bestSwing, stoodLeanN });
    if (state.bestEverScore > before) {
      const file = saveBackup('best-ever', state);
      log('best-ever gen ' + state.gen + ' score ' + state.bestEverScore.toFixed(3) + ' ' + file);
    }
    // Did the parent change this generation? If this stays 0 for a long stretch, the
    // population is exploring around a fixed parent and the average rises from noise, not learning.
    if (state.parentMoves > reportedParentMoves) {
      log('parent ' + state.parentMoves + ' changes (last: ' + (state.parentMoveNote || '?') + ')');
      reportedParentMoves = state.parentMoves;
    } else if (state.gen % 25 === 0 && state.parentMoves === 0) {
      log('WARNING: the parent has never changed. ' + (state.parentMoveNote || ''));
    }
    if (state.gen % 100 === 0) saveBackup('parent', state);
  }

  async function seedBest(state) {
    const jobs = [];
    let index = 0;
    for (let i = 0; i < config.eval.seedBestCount; i++) {
      for (const start of sim.trainingStarts(random)) {
        jobs.push({
          index: index++,
          kind: 'train',
          weights: state.theta,
          options: Object.assign({}, start, { phase: i % config.cadence.actionEveryNFrames }),
        });
      }
    }
    const rows = await pool.run(jobs);
    const mean = rows.reduce((sum, row) => sum + row.fitness, 0) / rows.length;
    state.bestEverTheta = state.theta.slice();
    state.bestEverScore = mean;
    state.bestSmooth = mean;
    log('seed bestEverScore ' + mean.toFixed(3) + ' phase ' + state.phase);
  }

  function writeStatus(state, extra) {
    fs.writeFileSync(statusPath, JSON.stringify(Object.assign({
      gen: state.gen,
      phase: state.phase,
      bestEverScore: state.bestEverScore,
      parentMoves: state.parentMoves || 0,
      parentMoveNote: state.parentMoveNote || "",
      eliteScore: state.eliteScore,
      thresholds: config.eval.passRate,
      evalN: opts.evalN,
      dropMeters,
      bestFiles,
      evalLog: evalLog.slice(-30),
      curveTail: curve.slice(-12),
    }, extra || {}), null, 2));
  }

  // The eval set: the same seeds for each level, the whole level, without the old placeScale ramp.
  // The command comes from config.commands[skill].
  //
  // secondsHere is the ONLY difference between eval and training: the eval ALWAYS runs the
  // whole episode, even while the training ramp is at 1.5 s. If the eval shortened the
  // episode we would be passing a net that can only stand for a second and a half.
function skillJobs(weights, skill, n, secondsHere, trace) {
    const mode = evalMode(skill);
    if (secondsHere > 0) mode.seconds = secondsHere;
    const jobs = [];
    for (let i = 0; i < n; i++) {
      const rand = makeSeededRandom(config.eval.seed + skill * 10007 + i * 17);
      const start = sim.fillTrainingStart({}, rand, mode);
      if (trace) start.trace = true;
      if (state.residualSearch) start.residual = state.residualSearch.theta;
      jobs.push({
        index: i,
        kind: 's' + skill,
        weights,
        options: Object.assign({ phase: i % config.cadence.actionEveryNFrames }, start),
      });
    }
    return jobs;
  }

  async function evaluateSkill(weights, skill, n, trace) {
    // MIX: run the stand eval and the walk eval. Its rate is the worse of the two, each as a
    // share of its own bar, so 1 means both passed.
    if (skill === 7) {
      const stand = await evaluateSkill(weights, 1, n, trace);
      const walk = await evaluateSkill(weights, 6, n, trace);
      const rate = Math.min(1, stand.rate / config.eval.passRate.stand, walk.rate / config.eval.passRate.walk);
      return Object.assign({}, walk, {
        skill: 7, key: 'mix', rate, ok: Math.round(rate * n), n,
        fitness: (stand.fitness + walk.fitness) / 2, stand, walk,
      });
    }
    const rows = await pool.run(skillJobs(weights, skill, n, 0, trace));
    if (rows.some((row) => !row || !Number.isFinite(row.fitness))) throw new Error('eval returned an empty score');
    if (trace) {
      const samples = rows.map((row) => row.trace || []);
      fs.writeFileSync(path.join(backupDir, 'trace.json'), JSON.stringify(samples));
      log('trace ' + samples.length + ' episodes -> backups/trace.json');
    }
    let ok = 0;
    let stood = 0;
    let fitnessSum = 0;
    let windowSum = 0;
    let footSum = 0;
    let stepSum = 0;
    let clearSum = 0;
    let headSum = 0;
    let stanceSum = 0;
    let headMarkSum = 0;
    let handMarkSum = 0;
    let bodySum = 0;
    let bodyN = 0;
    let walkDistSum = 0;
    let walkCmdSum = 0;
    let climbTopSum = 0;
    let climbHeightSum = 0;
    const obey = { head: [0, 0], hand: [0, 0], foot: [0, 0] };
    for (const row of rows) {
      if (skillPass(skill, row)) ok++;
      if (row.stoodUp) stood++;
      walkDistSum += row.walkDist || 0;
      walkCmdSum += row.walkCmd || 0;
      climbTopSum += row.climbOnTop || 0;
      climbHeightSum += row.climbHeight || 0;
      fitnessSum += row.fitness;
      windowSum += row.windowFrames ? row.windowStood / row.windowFrames : 0;
      footSum += ((row.footL || 0) + (row.footR || 0)) / 2;
      stepSum += row.steps || 0;
      clearSum += row.cleared || 0;
      headSum += row.headHeight || 0;
      stanceSum += row.stance || 0;
      headMarkSum += row.headMark == null ? 0 : row.headMark;
      handMarkSum += ((row.handLMark || 0) + (row.handRMark || 0)) / 2;
      if (row.bodyTargets) {
        bodySum += row.bodyTargets;
        bodyN++;
      }
      // Obedience per group: what fraction of that group's targets are on their mark.
      for (const spec of config.targets) {
        const group = groupOfId(spec.id);
        const error = row.errors && row.errors[spec.id];
        if (!error) continue;
        const tol = spec.ride ? config.skill.targetReachMeters : config.skill.footReachMeters;
        obey[group][1]++;
        if (error.dist <= tol) obey[group][0]++;
      }
    }
    const count = rows.length;
    return {
      skill,
      key: skillKey(skill),
      n: count,
      ok,
      fitness: count ? fitnessSum / count : 0,
      rate: count ? ok / count : 0,
      stood: count ? stood / count : 0,
      window: count ? windowSum / count : 0,
      foot: count ? footSum / count : 0,
      steps: count ? stepSum / count : 0,
      cleared: count ? clearSum / count : 0,
      head: count ? headSum / count : 0,
      stance: count ? stanceSum / count : 0,
      headMark: count ? headMarkSum / count : 0,
      handMark: count ? handMarkSum / count : 0,
      body: bodyN ? bodySum / bodyN : 0,
      headObey: obey.head[1] ? obey.head[0] / obey.head[1] : 0,
      handObey: obey.hand[1] ? obey.hand[0] / obey.hand[1] : 0,
      footObey: obey.foot[1] ? obey.foot[0] / obey.foot[1] : 0,
      walkDist: count ? walkDistSum / count : 0,
      walkCmd: count ? walkCmdSum / count : 0,
      climbTop: count ? climbTopSum / count : 0,
      climbHeight: count ? climbHeightSum / count : 0,
    };
  }

  function fmtSkill(row) {
    if (row.key === 'mix') return 'mix ' + row.rate.toFixed(2) + ' [ ' + fmtSkill(row.stand) + ' ] [ ' + fmtSkill(row.walk) + ' ]';
    return row.key + ' ' + row.ok + '/' + row.n + '=' + row.rate.toFixed(2)
      + ' fitness ' + (row.fitness || 0).toFixed(4)
      + ' stood ' + row.stood.toFixed(2)
      + ' window ' + row.window.toFixed(2)
      + ' foot ' + row.foot.toFixed(3)
      + ' steps ' + row.steps.toFixed(2)
      + ' boxes ' + row.cleared.toFixed(2)
      + ' head ' + (row.head || 0).toFixed(2)
      + ' stance ' + (row.stance || 0).toFixed(2)
      + ' headMark ' + (row.headMark == null ? 0 : row.headMark).toFixed(3)
      + ' handMark ' + (row.handMark == null ? 0 : row.handMark).toFixed(3)
      + ' obey h' + Math.round((row.headObey || 0) * 100)
      + ' m' + Math.round((row.handObey || 0) * 100)
      + ' p' + Math.round((row.footObey || 0) * 100)
      + (row.walkCmd ? ' walked ' + (row.walkDist || 0).toFixed(2) + '/' + row.walkCmd.toFixed(2) + 'm' : '')
      + (row.climbHeight ? ' ontop ' + Math.round(row.climbTop * 100) + '% box ' + row.climbHeight.toFixed(2) + 'm' : '');
  }

  function skillReached(row) {
    return row && opts.evalN >= config.eval.officialMinN && row.rate >= config.eval.passRate[row.key];
  }

  let gateEval = null;

  // The best held-out eval of this run, by pass rate and then fitness. Saved on its own file
  // so a later wobble in the parent never loses it: backups/best-skill<N>.json.
  const bestEval = {};
  function noteBestEval(state, row) {
    const prev = bestEval[row.skill];
    const better = !prev || row.rate > prev.rate || (row.rate === prev.rate && row.fitness > prev.fitness);
    if (!better) return;
    bestEval[row.skill] = { rate: row.rate, fitness: row.fitness, gen: state.gen, theta: state.theta.slice() };
    const file = path.join(backupDir, 'best-skill' + row.skill + '.json');
    fs.writeFileSync(file, JSON.stringify({ role: 'best-eval', skill: row.skill, gen: state.gen, eval: row, inputCount: config.net.inputCount, theta: Array.from(state.theta) }));
    log('best eval skill ' + row.skill + ' gen ' + state.gen + ' rate ' + row.rate.toFixed(2) + ' -> ' + path.basename(file));
  }

  function reloadStand(state) {
    const saved = JSON.parse(fs.readFileSync(path.join(backupDir, 'g200_s7p368_phase_1791767817432.json'), 'utf8'));
    state.theta.set(Float32Array.from(saved.theta));
    state.bestEverTheta.set(Float32Array.from(saved.bestEverTheta || saved.theta));
    state.zeroMoments();
    state.flatGens = 0;
    state.heldFoot = 0.215;
    state.footRatio = 0.97;
    log('reloaded g200');
  }

  // Eight fresh shifts, shared by the child and the parent. One nominating episode does not count.
  async function considerChild(state, weights, index, ratio) {
    const jobs = [];
    for (let i = 0; i < 8; i++) {
      const start = sim.fillTrainingStart({}, random, { full: true, level: 2 });
      const phase = i % config.cadence.actionEveryNFrames;
      jobs.push({ index: i, kind: 'confirm', weights: state.theta, options: Object.assign({ phase }, JSON.parse(JSON.stringify(start))) });
      jobs.push({ index: i + 8, kind: 'confirm', weights, options: Object.assign({ phase }, JSON.parse(JSON.stringify(start))) });
    }
    const rows = await pool.run(jobs);
    let parentFoot = 0;
    let childFoot = 0;
    let childStood = 0;
    for (let i = 0; i < 8; i++) {
      parentFoot += rows[i].foot || 0;
      childFoot += rows[i + 8].foot || 0;
      if (rows[i + 8].stoodUp) childStood++;
    }
    parentFoot /= 8;
    childFoot /= 8;
    const stood = childStood / 8;
    const ok = childFoot < parentFoot && stood >= 0.8;
    log('check gen ' + state.gen + ' child ' + index + ' ratio ' + ratio.toFixed(3) + ' parent ' + parentFoot.toFixed(3) + ' child ' + childFoot.toFixed(3) + ' stood ' + stood.toFixed(2) + (ok ? ' PASS' : ' FAIL'));
    if (!ok) return;
    state.theta.set(weights);
    state.bestEverTheta.set(state.theta);
    state.zeroMoments();
    const exam = await evaluateSkill(state.theta, 2, opts.evalN);
    gateEval = exam;
    evalLog.push({ gen: state.gen, skill: 2, who: 'adopt', eval: exam });
    log('EVAL adopt ' + fmtSkill(exam));
    if (!(exam.foot < state.heldFoot) || exam.stood < 0.8) {
      reloadStand(state);
      gateEval = null;
      return;
    }
    state.heldFoot = exam.foot;
    log('kept gen ' + state.gen + ' held foot ' + exam.foot.toFixed(3) + ' ' + saveBackup('adopt', state, { heldFoot: exam.foot }));
  }

  let state;
  try {
    log('=== ' + (opts.fresh ? 'fresh' : 'resume') + ' workers ' + opts.workers + ' pairs ' + opts.pairs + ' evalN ' + opts.evalN + ' ===');
    log('thresholds ' + JSON.stringify(config.eval.passRate) + ' dropM ' + dropMeters.toFixed(2) + ' impact<' + config.search.headImpactQuiet + ' walkClose ' + config.eval.walkClosingMeters);
    log('targets gains ' + targetGainText() + ' targetMean ' + config.reward.targetMean
      + ' residualSearch ' + (config.skill.residualSearch ? 'on (main net frozen)' : 'off (main net trains)')
      + ' gateTargets ' + (config.eval.gateTargets ? 'on' : 'off'));
    await Promise.race([
      pool.whenReady(),
      new Promise((_resolve, reject) => setTimeout(() => reject(new Error('workers did not start')), 60000)),
    ]);
    await pool.setGains(config.targetGains);
    const startedAt = Date.now();
    const timedOut = () => opts.minutes > 0 && Date.now() - startedAt > opts.minutes * 60000;
    if (opts.fresh) {
      if (!opts.from) fs.rmSync(statePath, { force: true });
      setTrainingPhase(config.curriculum.phases[0].id);
      const loaded = loadPolicy();
      state = loaded.state;
      log('loaded ' + loaded.source + ' raw ' + loaded.rawLength + ' padded ' + weightCount);
    } else {
      state = opts.from ? null : loadTrainState();
      if (!state) {
        const loaded = loadPolicy();
        state = loaded.state;
        log('no train-state, loaded ' + loaded.source + ' raw ' + loaded.rawLength);
      } else {
        log('resumed gen ' + state.gen + ' skill ' + (state.skill || 1));
      }
    }
    if (opts.boardCap > 0) state.boardMax = opts.boardCap;
    if (opts.getupCap > 0) state.getupMax = opts.getupCap;
    if (opts.getupAssist != null) state.getupAssist = opts.getupAssist;
    if (opts.climbCap > 0) state.climbMax = opts.climbCap;
    if (opts.skillSet) state.skill = opts.skill;
    else if (!(state.skill > 0)) state.skill = opts.skill || 1;
    setSkill(state.skill);
    log('skill ' + state.skill + ' (' + skillKey(state.skill) + ') commands ' + commandGroup(state.skill));
    if (state.reseed) {
      state.reseed = false;
      await seedBest(state);
      saveTrainState(state);
    }
    if (opts.resetMoments) {
      state.m = new Float32Array(weightCount);
      state.v = new Float32Array(weightCount);
      state.k = 0;
      state.flatGens = 0;
      state.bestEverScore = -1e9;
      state.bestSmooth = null;
      log('cleared Adam moments');
      await seedBest(state);
    }
    if (opts.evalOnly) {
      const only = await evaluateSkill(state.theta, state.skill, opts.evalN, opts.trace);
      log('EVAL-ONLY ' + fmtSkill(only) + (skillReached(only) ? ' PASS' : ' FAIL'));
      fs.writeFileSync(path.join(backupDir, 'result.json'), JSON.stringify({ evalOnly: true, eval: only }, null, 2));
    } else {
    if (opts.fresh || state.gen === 0) {
      const baseline = await evaluateSkill(state.theta, state.skill, opts.evalN);
      evalLog.push({ gen: state.gen, skill: state.skill, who: 'baseline', eval: baseline });
      log('BASELINE ' + fmtSkill(baseline));
      await seedBest(state);
      saveBackup('parent', state, { note: 'start' });
      saveTrainState(state);
      writeStatus(state, { baseline });
    }

    const lastSkill = opts.only ? state.skill : state.skill >= 6 ? 6 : 5;
    for (let skill = state.skill; skill <= lastSkill; skill++) {
      state.skill = skill;
      setSkill(skill);
      const gen0 = state.gen;
      let latest = null;
      setOutputJoints(null);
      // The frozen residual is created once and kept as a leg prior. With the command
      // ladder the scripted gait is off, so the residual barely does anything: the net
      // handles the legs itself.
      if (skill >= 2 && !state.residualSearch) {
        state.residualSearch = new ResidualSearch(randomNormal);
        state.residualSearch.sigma = 0.012;
        state.residualSearch.lr = 0.01;
        if (state.shiftMeters == null) state.shiftMeters = 0;
        if (!state.bestResidual) state.bestResidual = Array.from(state.residualSearch.theta);
      }
      log('SKILL ' + skill + ' ' + skillKey(skill) + ' commands ' + commandGroup(skill) +
        ' start gen ' + state.gen + ' budget ' + opts.gens +
        ' barra ' + config.eval.passRate[skillKey(skill)] +
        (config.skill.scriptedLegs ? ' scripted gait ON' : ' scripted gait OFF') +
        ' gains ' + targetGainText());
      while (state.gen - gen0 < opts.gens && !timedOut()) {
        const tick = Date.now();
        await oneGeneration(state);
        if (gateEval && skillReached(gateEval)) {
          latest = gateEval;
          gateEval = null;
          break;
        }
        gateEval = null;
        if (state.gen % 5 === 0 || state.gen - gen0 <= 3) {
          const last = curve[curve.length - 1];
          const sec = ((Date.now() - tick) / 1000).toFixed(2);
          const alive = last && last.alive != null ? ' alive ' + last.alive.toFixed(2) : '';
          const foot = last && last.foot != null ? ' foot ' + last.foot.toFixed(3) : '';
          const closed = last && last.closed != null ? ' closed ' + last.closed.toFixed(3) : '';
          const leaned = last && last.leaned != null ? ' leaned ' + last.leaned.toFixed(3) + ' swing ' + (last.bestSwing || 0).toFixed(3) + ' land ' + (last.bestClosed || 0).toFixed(3) + '/' + (last.stoodLeanN || 0) : '';
          const steps = last && last.steps != null ? ' steps ' + last.steps.toFixed(2) : '';
          const walked = last && last.walk ? ' walked ' + last.walk.toFixed(3) + ' best ' + last.bestWalk.toFixed(2) : '';
          const boarded = (state.skill | 0) === 10 ? ' board amp ' + (state.boardMax || 0).toFixed(2) + ' rate ' + (state.boardRate || 0).toFixed(2) : '';
          const gotUp = (state.skill | 0) === 9 ? (config.getup.useAssist ? ' assist ' + (state.getupAssist || 0).toFixed(2) : ' getup cap ' + (state.getupMax || 0).toFixed(2)) + ' rate ' + (state.getupRate || 0).toFixed(2) : '';
          const climbed = (state.skill | 0) === 8 || (state.skill | 0) === 11 ? ' climb cap ' + (state.climbMax || 0).toFixed(2) + ' rate ' + (state.climbRate || 0).toFixed(2) : '';
          log('gen ' + state.gen + ' skill ' + skill + ' mean ' + (last ? last.mean.toFixed(3) : '?') + ' stood ' + (last ? last.stood.toFixed(2) : '?') + alive + foot + closed + leaned + steps + walked + climbed + gotUp + boarded + ' ever ' + state.bestEverScore.toFixed(3) + ' ' + sec + 's');
          saveTrainState(state);
        }
        if (opts.quickEvery > 0 && state.gen % opts.quickEvery === 0) {
          latest = await evaluateSkill(state.theta, skill, opts.evalN);
          evalLog.push({ gen: state.gen, skill, who: 'theta', eval: latest });
          log('EVAL gen ' + state.gen + ' ' + fmtSkill(latest));
          noteBestEval(state, latest);
          writeStatus(state, { latest });
          saveTrainState(state);
          if (skillReached(latest)) break;
        }
      }
      if (!(latest && skillReached(latest))) {
        latest = await evaluateSkill(state.theta, skill, opts.evalN);
        log('FINAL skill ' + skill + ' ' + fmtSkill(latest));
      }
      if (latest) noteBestEval(state, latest);
      const kept = bestEval[skill];
      if (!skillReached(latest) && kept && kept.rate > latest.rate) {
        state.theta.set(kept.theta);
        latest = await evaluateSkill(state.theta, skill, opts.evalN);
        log('restored best eval of gen ' + kept.gen + ': ' + fmtSkill(latest));
      }
      let ever = null;
      const frozenMain = config.skill.residualSearch && state.residualSearch && skill >= 2;
      if (!skillReached(latest) && !frozenMain) {
        ever = await evaluateSkill(state.bestEverTheta, skill, opts.evalN);
        log('BESTEVER skill ' + skill + ' ' + fmtSkill(ever));
        if (skillReached(ever)) {
          state.theta.set(state.bestEverTheta);
          latest = ever;
          log('adopted best-ever for skill ' + skill);
        }
      }
      saveTrainState(state);
      writeStatus(state, { latest, bestEver: ever });
      const result = {
        skill,
        key: skillKey(skill),
        passed: skillReached(latest),
        thresholds: config.eval.passRate,
        evalN: opts.evalN,
        official: opts.evalN >= config.eval.officialMinN,
        theta: latest,
        bestEver: ever,
        gen: state.gen,
        bestFiles,
      };
      fs.writeFileSync(path.join(backupDir, 'result.json'), JSON.stringify(result, null, 2));
      if (!skillReached(latest)) {
        log('gate ' + skillKey(skill) + ' not reached ' + fmtSkill(latest));
        break;
      }
      const file = saveBackup('phase', state, { eval: latest, passed: true, skill });
      log('PASS skill ' + skill + ' ' + skillKey(skill) + ' ' + fmtSkill(latest) + ' ' + file);
      if (skill === lastSkill) {
        log(skill === 5 ? 'DONE the whole ladder: all five targets under command' : 'DONE skill ' + skill);
        break;
      }
      state.m = new Float32Array(weightCount);
      state.v = new Float32Array(weightCount);
      state.k = 0;
      state.flatGens = 0;
      state.eliteScore = -1e9;
      setSkill(skill + 1);
      log('next skill ' + (skill + 1) + ' at gen ' + state.gen);
      await seedBest(state);
      saveTrainState(state);
    }
    }
  } catch (error) {
    log('ERROR ' + (error && error.stack || error));
    if (state) saveTrainState(state);
    process.exitCode = 1;
  } finally {
    await pool.stop();
    // The workers are threads, not processes: one that does not exit keeps the event
    // loop alive and the script never ends. This is a CLI, so it exits on purpose.
    // The delay gives stdout time to flush.
    setTimeout(() => process.exit(process.exitCode || 0), 300);
  }
}

