// Runs the search and the simulation workers. The page thread only draws the doll.
import { EvolutionSearch } from "./net.js";
import { config, setTrainingPhase, setSkill, applySearchNumbers } from "../config.js";
import { bytesFor, writeTasks, readScore, resultIndex } from "./task-buffer.js";
import { makeSeededRandom, randomNormalSample } from "./utils.js";
import { runGeneration, trainingMode, generationPairs } from "./training.js";

const random = makeSeededRandom(7);
const randomNormal = () => randomNormalSample(random);

let paused = true;
let epoch = 0;
let phase = config.curriculum.phase;
let trainSpec = null;
let state = null;
const slots = [];
const scorePool = [];

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function snapshotStart(options) {
  if (!options) return null;
  const source = options.targets || [];
  const targets = [];
  for (let i = 0; i < source.length; i++) {
    const target = source[i] || {};
    targets.push({
      on: target.on !== false,
      dx: target.dx || 0,
      dy: target.dy || 0,
      turn: target.turn || 0,
      cx: target.cx || 0,
      cy: target.cy || 0,
      ca: target.ca || 0,
      x: target.x || 0,
      y: target.y || 0,
      angle: target.angle || 0,
    });
  }
  return {
    tilt: options.tilt || 0,
    drop: options.drop || 0,
    shiftX: options.shiftX || 0,
    stance: options.stance || 0,
    failOnHead: !!options.failOnHead,
    commands: options.commands || "",
    placed: !!options.placed,
    targets,
  };
}

function browserMode(search) {
  const mode = trainingMode(config.skill.level, search.gen || 0, 0);
  if (trainSpec && Array.isArray(trainSpec.targets)) {
    mode.placed = trainSpec.targets;
    mode.jitter = +trainSpec.jitter || 0;
    mode.randomStart = trainSpec.randomStart !== false;
  }
  return mode;
}

function scoreRow(index) {
  let row = scorePool[index];
  if (row) return row;
  row = { index: 0, fitness: 0, f: 0, start: 0, end: 0, stoodUp: false, up: false, hy: 0, headImpact: 0, gap: 0, stance: 0 };
  scorePool[index] = row;
  return row;
}

function adopt(packet) {
  const search = new EvolutionSearch(randomNormal);
  const length = search.theta.length;
  if (!packet || !packet.theta || packet.theta.length !== length) return search;
  search.theta.set(packet.theta);
  if (packet.bestGenTheta && packet.bestGenTheta.length === length) search.bestGenTheta.set(packet.bestGenTheta);
  else search.bestGenTheta.set(search.theta);
  if (packet.bestEverTheta && packet.bestEverTheta.length === length) search.bestEverTheta.set(packet.bestEverTheta);
  else search.bestEverTheta.set(search.theta);
  if (packet.m && packet.m.length === length) search.m.set(packet.m);
  if (packet.v && packet.v.length === length) search.v.set(packet.v);
  search.k = packet.k || 0;
  search.gen = packet.gen || 0;
  search.flatGens = packet.flatGens || 0;
  search.bestEverScore = packet.bestEverScore ?? -1e9;
  search.bestGenScore = packet.bestGenScore ?? -1e9;
  search.bestSmooth = packet.bestSmooth ?? null;
  search.eliteScore = packet.eliteScore ?? -1e9;
  search.hist = Array.isArray(packet.hist) ? packet.hist : [];
  search.refreshMutationBoost();
  return search;
}

function bootWorkers() {
  const cores = navigator.hardwareConcurrency || config.workers.browserHardwareFallback;
  const count = Math.max(config.workers.browserMin, Math.min(config.workers.browserMax, cores - config.workers.browserReserve - 1));
  return Promise.all(Array.from({ length: count }, () => new Promise((resolve, reject) => {
    const worker = new Worker(new URL("./browser-worker.js", import.meta.url), { type: "module" });
    const slot = { worker, buffers: [null, null], next: 0, list: [] };
    const timer = setTimeout(() => reject(new Error("A worker did not answer in 15 s")), 15000);
    worker.onmessage = (event) => { if (event.data === "ready") { clearTimeout(timer); resolve(); } };
    worker.onerror = () => { clearTimeout(timer); reject(new Error("Could not start a worker")); };
    slots.push(slot);
  })));
}

function evaluatePopulation(items) {
  for (let workerIndex = 0; workerIndex < slots.length; workerIndex++) {
    const slot = slots[workerIndex];
    if (!slot.bucket) slot.bucket = [];
    slot.bucket.length = 0;
  }
  for (let index = 0; index < items.length; index++) slots[index % slots.length].bucket.push(items[index]);
  const weightCount = items.length ? items[0].weights.length : 0;
  const sendStart = performance.now();
  const pending = slots.map((slot) => {
    const bucket = slot.bucket;
    if (!bucket.length) return Promise.resolve(null);
    const bytes = bytesFor(bucket.length, weightCount);
    const sent = slot.next;
    let buffer = slot.buffers[sent];
    if (!buffer || buffer.byteLength < bytes) {
      buffer = new ArrayBuffer(bytes);
      slot.buffers[sent] = buffer;
    }
    writeTasks(buffer, bucket, weightCount);
    slot.next ^= 1;
    const sentAt = performance.now();
    return new Promise((resolve, reject) => {
      const worker = slot.worker;
      worker.onmessage = (event) => {
        const arrived = performance.now();
        try {
          const data = event.data;
          if (!data || !data.buffer) throw new Error("Worker returned no buffer");
          slot.buffers[sent] = data.buffer;
          const f32 = new Float32Array(data.buffer);
          const list = slot.list;
          list.length = 0;
          for (let i = 0; i < bucket.length; i++) {
            const taskIndex = f32[resultIndex(bucket.length, weightCount, i)];
            list.push(readScore(f32, bucket.length, weightCount, i, scoreRow(taskIndex)));
          }
          resolve({
            results: list,
            ms: data.ms,
            steps: data.steps,
            handoff: Math.max(0, arrived - sentAt - data.ms),
            count: bucket.length,
          });
        } catch (error) { reject(error); }
      };
      worker.onerror = () => reject(new Error("Worker failed"));
      worker.postMessage({ buffer, trainingPhase: phase }, [buffer]);
    });
  });
  const sendMs = performance.now() - sendStart;
  return Promise.all(pending).then((groups) => {
    const timings = [];
    const ordered = [];
    ordered.length = items.length;
    for (let g = 0; g < groups.length; g++) {
      const group = groups[g];
      if (!group) continue;
      timings.push({ ms: group.ms, steps: group.steps, handoff: group.handoff, count: group.count });
      const results = group.results;
      for (let i = 0; i < results.length; i++) ordered[results[i].index] = results[i];
    }
    return { ordered, timings, sendMs };
  });
}

async function oneGeneration(search, runEpoch) {
  let run = null;
  let tellMs = 0;
  let preview = null;
  await runGeneration({
    ask: () => search.ask(null, generationPairs(), randomNormal),
    random,
    mode: browserMode(search),
    runJobs: async (jobs) => {
      preview = snapshotStart(jobs[0] && jobs[0].options);
      run = await evaluatePopulation(jobs);
      return run.ordered;
    },
    tell: (generation, scores) => {
      const tellStart = performance.now();
      search.tell(generation, scores);
      tellMs = performance.now() - tellStart;
    },
  });
  const row = search.hist[search.hist.length - 1];
  const theta = search.theta.slice();
  const bestGenTheta = search.bestGenTheta.slice();
  const bestEverTheta = search.bestEverTheta.slice();
  const m = search.m.slice();
  const v = search.v.slice();
  const note = search.anchorNote;
  search.anchorNote = "";
  const scouted = !!search.lastScouted;
  search.lastScouted = false;
  return {
    type: "generation",
    epoch: runEpoch,
    gen: search.gen,
    k: search.k || 0,
    flatGens: search.flatGens || 0,
    mutationBoost: search.mutationBoost || 1,
    mutationFraction: search.mutationFraction || 0,
    bestEverScore: search.bestEverScore,
    bestGenScore: search.bestGenScore,
    bestSmooth: search.bestSmooth,
    eliteScore: search.eliteScore,
    anchorNote: note || "",
    row,
    timings: run.timings,
    sendMs: run.sendMs,
    tellMs,
    preview,
    scouted,
    theta,
    bestGenTheta,
    bestEverTheta,
    m,
    v,
    transfer: [theta.buffer, bestGenTheta.buffer, bestEverTheta.buffer, m.buffer, v.buffer],
  };
}

onmessage = (event) => {
  const data = event.data;
  if (!data) return;
  if (data.type === "pause") { paused = !!data.paused; return; }
  if (data.type === "targets") {
    trainSpec = {
      targets: Array.isArray(data.targets) ? data.targets : [],
      jitter: +data.jitter || 0,
      randomStart: data.randomStart !== false,
    };
    return;
  }
  if (data.type === "skill") {
    setSkill(data.level);
    return;
  }
  if (data.type === "scout") {
    if (state) state.handScout = true;
    return;
  }
  if (data.type === "search") {
    applySearchNumbers(data.values);
    return;
  }
  if (data.type === "phase") {
    phase = data.phase;
    setTrainingPhase(phase);
    return;
  }
  if (data.type === "replace") {
    epoch = data.epoch;
    paused = !!data.paused;
    phase = data.phase;
    setTrainingPhase(phase);
    if (data.search) applySearchNumbers(data.search);
    if (Array.isArray(data.targets)) {
      trainSpec = {
        targets: data.targets,
        jitter: +data.jitter || 0,
        randomStart: data.randomStart !== false,
      };
    }
    state = adopt(data.state);
  }
};

let booted = false;
const ready = bootWorkers().then(() => {
  booted = true;
  postMessage({ type: "ready" });
}, (error) => {
  postMessage({ type: "error", message: error.message });
});

ready.then(async () => {
  if (!booted) return;
  while (slots.length) {
    if (paused || !state) { await sleep(150); continue; }
    const runEpoch = epoch;
    const search = state;
    try {
      const report = await oneGeneration(search, runEpoch);
      if (runEpoch !== epoch || search !== state) continue;
      const transfer = report.transfer;
      report.transfer = undefined;
      postMessage(report, transfer);
    } catch (error) {
      postMessage({ type: "error", message: error.message || String(error) });
      paused = true;
    }
  }
});
