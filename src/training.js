// The generation cycle both runners share. Checkpoints, the worker pool, and the page stay outside.
import { config } from "../config.js";
import { fillTrainingStart, summariseEpisode } from "./sim.js";

// Training episodes grow from rampStartSeconds to episode.seconds. 0 turns the ramp off.
// The held-out eval does not use this. It always runs the full episode.
export function trainingSeconds(gen) {
  const skill = config.skill;
  const full = config.episode.seconds;
  const start = skill.rampStartSeconds || 0;
  const gens = skill.rampGenerations | 0;
  if (!(start > 0) || !(gens > 0) || gen >= gens) return full;
  return start + ((full - start) * gen) / gens;
}

// What fillTrainingStart needs for one training generation.
// full is false only while a shift is being searched, matching the headless start.
export function trainingMode(level, gen, shiftMeters) {
  const shift = shiftMeters || 0;
  return {
    level: level | 0,
    shiftMeters: shift,
    full: !(shift > 0),
    seconds: trainingSeconds(gen),
  };
}

// Held-out eval: the whole episode, no duration ramp.
export function evalMode(level) {
  return { eval: true, full: true, level: level | 0 };
}

export function generationPairs() {
  return config.workers.pairs;
}

function copyStart(shared, phase) {
  const options = Object.assign({}, shared, { phase });
  if (shared.targets) options.targets = shared.targets.map((target) => Object.assign({}, target));
  if (shared.steps) options.steps = shared.steps.map((step) => Object.assign({}, step));
  if (shared.boxes) options.boxes = shared.boxes.map((box) => Object.assign({}, box));
  return options;
}

// One shared draw per antithetic pair, then a separate options object per child.
export function jobsFor(generation, random, mode, decorate) {
  const jobs = [];
  let shared = null;
  const count = generation.weights.length;
  for (let child = 0; child < count; child++) {
    if ((child & 1) === 0) shared = fillTrainingStart({}, random, mode);
    const phase = child % config.cadence.actionEveryNFrames;
    const job = {
      index: child,
      phase,
      kind: "train",
      weights: generation.weights[child],
      options: copyStart(shared, phase),
    };
    if (decorate) decorate(job, child, generation);
    jobs.push(job);
  }
  return jobs;
}

export function scoreGeneration(raw, childCount) {
  const scores = [];
  for (let child = 0; child < childCount; child++) scores.push(summariseEpisode(raw, child, 1, {}));
  return scores;
}

// ask() is supplied by the caller because the main search and the residual search
// do not take the same arguments. tell(generation, scores) writes the result back.
export async function runGeneration({ ask, random, mode, runJobs, decorate, tell }) {
  const generation = ask();
  const jobs = jobsFor(generation, random, mode, decorate);
  const raw = await runJobs(jobs);
  if (!raw || raw.length !== jobs.length || raw.some((row) => !row || !Number.isFinite(row.fitness)))
    throw new Error("training returned an empty score");
  const scores = scoreGeneration(raw, generation.weights.length);
  tell(generation, scores);
  return { generation, scores };
}
