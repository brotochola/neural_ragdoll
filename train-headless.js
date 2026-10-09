// One headless loop: same sim, net, reward, and config as the page.
// Pass/fail is the search center (theta) on a fixed held-out set, not the training batch.
// Stand and drop need isStanding, head above config.body.standingHeadMeters, and headImpact < config.search.headImpactQuiet.
// Ball needs isStanding at the end. Get-up needs isStanding at the end.
// Walk needs isStanding at the end and the gap to config.eval.walkGoalMeters down by at least config.eval.walkClosingMeters.
// Official gate (only when --eval-n >= config.eval.officialMinN): see config.eval.passRate in config.js.
import { Worker, isMainThread, parentPort } from 'node:worker_threads';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { config, setTrainingPhase, setTrainingByPhases } from './config.js';

const root = path.dirname(fileURLToPath(import.meta.url));

if (!isMainThread) {
  await runWorker();
} else {
  await runMain();
}

async function runWorker() {
  const boxUrl = pathToFileURL(path.join(root, 'node_modules/phaser-box2d/dist/PhaserBox2D.min.js')).href;
  const Box2D = await import(boxUrl);
  const { createSim } = await import('./src/sim.js');
  const { isStanding } = await import('./src/reward.js');
  const sim = createSim(Box2D);
  const world = sim.build();
  parentPort.postMessage({ type: 'ready' });
  parentPort.on('message', (msg) => {
    if (msg.type === 'phase') {
      setTrainingPhase(msg.phase);
      parentPort.postMessage({ type: 'phase', phase: msg.phase });
      return;
    }
    if (msg.type !== 'run') return;
    try {
      const results = msg.jobs.map((job) => runEpisode(sim, world, job, isStanding));
      parentPort.postMessage({ type: 'done', id: msg.id, results });
    } catch (error) {
      parentPort.postMessage({ type: 'done', id: msg.id, error: String(error && error.stack || error) });
    }
  });
}

function runEpisode(sim, world, job, isStanding) {
  const options = job.options || {};
  const weights = job.weights instanceof Float32Array ? job.weights : Float32Array.from(job.weights);
  const episode = new sim.Episode(world, weights, options.tilt || 0, options.drop || 0, options);
  const startGap = options.goalX == null ? null : Math.abs(options.goalX - (episode.measure.centerX || 0));
  while (!episode.step());
  const stats = episode.stats;
  const measure = episode.measure;
  const standing = isStanding(measure);
  const headHeight = measure.headHeight || 0;
  episode.cleanup();
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
    pairs: config.workers.headlessPairs,
    workers: Math.max(config.workers.headlessWorkerMin, Math.min(config.workers.headlessWorkerMax, os.cpus().length - config.workers.headlessWorkerReserve)),
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
      if (msg.type === 'phase') {
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
    const timeout = new Promise((_resolve, reject) => setTimeout(() => reject(new Error('worker batch timed out')), 120000));
    return Promise.race([done, timeout]);
  }
  function setPhase(phase) {
    return Promise.all(workers.map((slot) => new Promise((resolve) => {
      slot.phaseWait = resolve;
      slot.worker.postMessage({ type: 'phase', phase });
    })));
  }
  async function stop() {
    await Promise.all(workers.map((slot) => slot.worker.terminate()));
  }
  return { whenReady, run, setPhase, stop };
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
  for (const row of rows) {
    if (row.standing) standing++;
    if (skillOk(kind, row)) ok++;
    headSum += row.headHeight || 0;
    impactSum += row.headImpact || 0;
    stanceSum += row.stance || 0;
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
  const backupDir = path.join(root, 'backups');
  fs.mkdirSync(backupDir, { recursive: true });
  const logPath = path.join(backupDir, 'train-log.txt');
  const statePath = path.join(backupDir, 'train-state.json');
  const statusPath = path.join(backupDir, 'status.json');
  const { ElmanNetwork, EvolutionSearch } = await import('./src/net.js');
  const { makeSeededRandom, randomNormalSample } = await import('./src/utils.js');
  const boxUrl = pathToFileURL(path.join(root, 'node_modules/phaser-box2d/dist/PhaserBox2D.min.js')).href;
  const Box2D = await import(boxUrl);
  const { createSim } = await import('./src/sim.js');
  const sim = createSim(Box2D);
  const random = makeSeededRandom(7);
  const randomNormal = () => randomNormalSample(random);
  const weightCount = ElmanNetwork.weightCount;
  const dropMeters = (config.start.fallDropMinMeters + config.start.fallDropMaxMeters) / 2;
  const pool = makePool(opts.workers);
  const evalLog = [];
  const curve = [];
  const bestFiles = [];
  let logFile = logPath;

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
      bestEverScore: state.bestEverScore,
      bestSmooth: Number.isFinite(state.bestSmooth) ? state.bestSmooth : null,
      eliteScore: state.eliteScore,
      flatGens: state.flatGens || 0,
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
    const file = fs.existsSync(path.join(root, 'checkpoint.json')) ? 'checkpoint.json' : 'weights.json';
    const data = JSON.parse(fs.readFileSync(path.join(root, file), 'utf8'));
    const raw = data.bestTheta || data.theta;
    const theta = ElmanNetwork.copy(raw);
    if (!theta) throw new Error('weights length ' + (raw && raw.length) + ' in ' + file);
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
    state.drill = saved.drill || 0;
    state.bestEverScore = saved.bestEverScore ?? -1e9;
    if (Number.isFinite(+saved.bestSmooth)) state.bestSmooth = +saved.bestSmooth;
    state.eliteScore = saved.eliteScore ?? -1e9;
    if (Number.isFinite(+saved.flatGens) && +saved.flatGens > 0) state.flatGens = Math.floor(+saved.flatGens);
    state.refreshMutationBoost();
    if (Array.isArray(saved.curve)) curve.push(...saved.curve);
    if (Array.isArray(saved.evalLog)) evalLog.push(...saved.evalLog);
    if (Array.isArray(saved.bestFiles)) bestFiles.push(...saved.bestFiles);
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
    const generation = state.ask(null, opts.pairs, randomNormal);
    const jobs = [];
    const trios = [];
    let index = 0;
    for (let child = 0; child < generation.weights.length; child++) {
      if (child % 2 === 0) trios.push(sim.trainingStarts(random));
      for (const start of trios[Math.floor(child / 2)]) {
        jobs.push({
          index: index++,
          kind: 'train',
          weights: generation.weights[child],
          options: Object.assign({}, start, { phase: child % config.cadence.actionEveryNFrames }),
        });
      }
    }
    const raw = await pool.run(jobs);
    if (raw.length !== jobs.length || raw.some((score) => !score || !Number.isFinite(score.fitness))) {
      throw new Error('training returned an empty score');
    }
    const scores = [];
    for (let child = 0; child < generation.weights.length; child++) scores.push(sim.poseScore(raw.slice(child * 3, child * 3 + 3)));
    state.tell(generation, scores);
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
    if (last) curve.push({ gen: state.gen, mean: last.mean, best: last.best, start: last.start, stood: last.stood, gap: last.gap, stance: meanStance });
    if (state.bestEverScore > before) {
      const file = saveBackup('best-ever', state);
      log('best-ever gen ' + state.gen + ' score ' + state.bestEverScore.toFixed(3) + ' ' + file);
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
      thresholds: config.eval.passRate,
      evalN: opts.evalN,
      dropMeters,
      bestFiles,
      evalLog: evalLog.slice(-30),
      curveTail: curve.slice(-12),
    }, extra || {}), null, 2));
  }

  let state;
  try {
    log('=== ' + (opts.fresh ? 'fresh' : 'resume') + ' workers ' + opts.workers + ' pairs ' + opts.pairs + ' evalN ' + opts.evalN + ' ===');
    log('thresholds ' + JSON.stringify(config.eval.passRate) + ' dropM ' + dropMeters.toFixed(2) + ' impact<' + config.search.headImpactQuiet + ' walkClose ' + config.eval.walkClosingMeters);
    await Promise.race([
      pool.whenReady(),
      new Promise((_resolve, reject) => setTimeout(() => reject(new Error('workers did not start')), 60000)),
    ]);
    setTrainingByPhases(true);
    if (opts.fresh) {
      fs.rmSync(statePath, { force: true });
      setTrainingPhase(config.curriculum.phases[0].id);
      const loaded = loadPolicy();
      state = loaded.state;
      log('loaded ' + loaded.source + ' raw ' + loaded.rawLength + ' padded ' + weightCount);
    } else {
      state = loadTrainState();
      if (!state) {
        const loaded = loadPolicy();
        state = loaded.state;
        log('no train-state, loaded ' + loaded.source + ' raw ' + loaded.rawLength);
      } else {
        log('resumed gen ' + state.gen + ' phase ' + state.phase);
      }
      setTrainingPhase(state.phase);
    }
    await pool.setPhase(state.phase);
    if (opts.fresh || state.gen === 0) {
      const baseline = await evaluate(state.theta, watchKinds(3), opts.evalN);
      evalLog.push({ gen: state.gen, phase: 0, full: true, who: 'baseline', eval: baseline });
      log('BASELINE ' + fmtEval(baseline));
      await seedBest(state);
      saveBackup('parent', state, { note: 'start' });
      saveTrainState(state);
      writeStatus(state, { baseline });
    }

    const phaseCursor = Math.max(0, config.curriculum.phases.findIndex((item) => item.id === state.phase));
    for (let phaseIndex = phaseCursor; phaseIndex < config.curriculum.phases.length; phaseIndex++) {
      const phaseDef = config.curriculum.phases[phaseIndex];
      const phase = phaseDef.id;
      state.phase = phase;
      setTrainingPhase(phase);
      await pool.setPhase(phase);
      const gen0 = state.gen;
      const budgetGens = phaseDef.generations;
      let latest = null;
      log('PHASE ' + phaseDef.name + ' start gen ' + state.gen + ' for ' + budgetGens);
      while (true) {
        const doneGens = state.gen - gen0;
        if (doneGens >= budgetGens) break;
        const tick = Date.now();
        await oneGeneration(state);
        if (state.gen % 10 === 0) {
          const last = curve[curve.length - 1];
          const sec = ((Date.now() - tick) / 1000).toFixed(2);
          const pct = Math.round((state.mutationFraction || 0) * 100);
          const mut = (state.mutationBoost || 1) > 1 ? ' mut x' + state.mutationBoost + (pct > 0 ? ' ' + pct + '%' : '') : '';
          const began = last && Number.isFinite(last.start) ? last.start.toFixed(3) + ' → ' : '';
          log('gen ' + state.gen + ' drill ' + (state.drill || 0) + ' mean ' + began + (last ? last.mean.toFixed(3) : '?') + ' best ' + (last ? last.best.toFixed(3) : '?') + ' stood ' + (last ? last.stood.toFixed(2) : '?') + ' stance ' + (last && last.stance != null ? last.stance.toFixed(2) : '?') + ' ever ' + state.bestEverScore.toFixed(3) + mut + ' ' + sec + 's');
        }
        if (opts.quickEvery > 0 && state.gen % opts.quickEvery === 0) {
          const quickKinds = gateKeys(phase);
          const quick = await evaluate(state.theta, quickKinds, opts.quickN);
          evalLog.push({ gen: state.gen, phase, full: false, who: 'theta', eval: quick });
          log('QUICK gen ' + state.gen + ' ' + fmtEval(quick));
          const due = state.gen % (opts.quickEvery * config.eval.fullMultiple) === 0;
          if (due || nearPass(phase, quick) || passed(phase, quick)) {
            latest = await evaluate(state.theta, watchKinds(phase), opts.evalN);
            evalLog.push({ gen: state.gen, phase, full: true, who: 'theta', eval: latest });
            log('EVAL gen ' + state.gen + ' ' + fmtEval(latest));
            writeStatus(state, { latest });
            if (phase === 1 && opts.evalN >= config.eval.officialMinN) {
              let nextDrill = state.drill || 0;
              if (latest.stand && latest.stand.rate >= config.eval.passRate.stand) nextDrill = Math.max(nextDrill, 1);
              if (nextDrill >= 1 && latest.ball && latest.ball.rate >= config.eval.passRate.ball) nextDrill = Math.max(nextDrill, 2);
              if (nextDrill !== (state.drill || 0)) {
                state.drill = nextDrill;
                state.m = new Float32Array(weightCount);
                state.v = new Float32Array(weightCount);
                state.k = 0;
                log('drill ' + state.drill);
              }
            }
            if (opts.evalN >= config.eval.officialMinN && passed(phase, latest)) {
              const file = saveBackup('phase', state, { eval: latest, passed: true, phase });
              log('PASS phase ' + phase + ' ' + file);
              latest.passed = true;
            }
          }
          saveTrainState(state);
        }
      }
      if (!(latest && latest.passed)) {
        latest = await evaluate(state.theta, watchKinds(phase), opts.evalN);
        evalLog.push({ gen: state.gen, phase, full: true, who: 'theta-final', eval: latest });
        log('FINAL phase ' + phase + ' ' + fmtEval(latest));
        if (opts.evalN >= config.eval.officialMinN && passed(phase, latest)) {
          const file = saveBackup('phase', state, { eval: latest, passed: true, phase });
          log('PASS phase ' + phase + ' ' + file);
          latest.passed = true;
        }
      }
      const ever = await evaluate(state.bestEverTheta, watchKinds(phase), opts.evalN);
      evalLog.push({ gen: state.gen, phase, full: true, who: 'best-ever', eval: ever });
      log('BESTEVER phase ' + phase + ' ' + fmtEval(ever));
      saveTrainState(state);
      writeStatus(state, { latest, bestEver: ever });
      if (!(opts.evalN >= config.eval.officialMinN && latest && latest.passed)) {
        const all = await evaluate(state.theta, watchKinds(3), opts.evalN);
        evalLog.push({ gen: state.gen, phase, full: true, who: 'theta-all', eval: all });
        log('ALL phase ' + phase + ' ' + fmtEval(all));
        const result = {
          passedPhase: phase - 1,
          stoppedAt: phase,
          trainingPhase: phase,
          thresholds: config.eval.passRate,
          evalN: opts.evalN,
          dropMeters,
          official: opts.evalN >= config.eval.officialMinN,
          theta: all,
          bestEver: ever,
          bestFiles,
          curveTail: curve.filter((_row, index) => index % 10 === 0).slice(-80),
        };
        fs.writeFileSync(path.join(backupDir, 'result.json'), JSON.stringify(result, null, 2));
        log('phase ' + phaseDef.name + ' generations done, gate not met');
      }
      if (phaseIndex === config.curriculum.phases.length - 1) {
        if (opts.evalN >= config.eval.officialMinN && latest && latest.passed) {
          const all = await evaluate(state.theta, watchKinds(3), opts.evalN);
          log('ALL phase 3 ' + fmtEval(all));
          const result = {
            passedPhase: 3,
            stoppedAt: 3,
            trainingPhase: 3,
            thresholds: config.eval.passRate,
            evalN: opts.evalN,
            dropMeters,
            official: opts.evalN >= config.eval.officialMinN,
            theta: all,
            bestEver: ever,
            bestFiles,
          };
          fs.writeFileSync(path.join(backupDir, 'result.json'), JSON.stringify(result, null, 2));
          log('DONE all phases');
        }
        break;
      }
      state.theta = state.theta.slice();
      state.m = new Float32Array(weightCount);
      state.v = new Float32Array(weightCount);
      state.k = 0;
      state.eliteScore = -1e9;
      const nextPhase = config.curriculum.phases[phaseIndex + 1];
      state.phase = nextPhase.id;
      setTrainingPhase(state.phase);
      await pool.setPhase(state.phase);
      log('By phases: ' + nextPhase.name + ' (gen ' + state.gen + ')');
      await seedBest(state);
      saveBackup('phase', state, { note: 'enter', phase: state.phase });
      saveTrainState(state);
    }
  } catch (error) {
    log('ERROR ' + (error && error.stack || error));
    if (state) saveTrainState(state);
    process.exitCode = 1;
  } finally {
    await pool.stop();
  }
}
