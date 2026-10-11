import * as Box2D from "https://cdn.jsdelivr.net/npm/phaser-box2d@1.1.0/dist/PhaserBox2D.min.js";
import { createSim } from "./sim.js";
import { config, setTrainingPhase } from "../config.js";
import { HEADER, TASK_HEADER, resultIndex } from "./task-buffer.js";

const sim = createSim(Box2D);
const world = sim.build();
let episode = null;
const targets = [];
for (let i = 0; i < config.targets.length; i++) targets.push({ on: false, x: 0, y: 0, angle: 0 });
const options = {
  push: 0,
  amp: 0,
  period: config.body.platformPeriodSeconds,
  phaseShift: 0,
  projectiles: [],
  frames: config.episode.frames,
  task: "live",
  pose: "up",
  phase: 0,
  tilt: 0,
  drop: 0,
  settle: 0,
  shiftX: 0,
  failOnHead: false,
  targets,
  goalX: null,
  ballFrame: null,
  ballSide: 1,
  ballSpeed: 0,
};

function readOptions(f32, cursor) {
  options.phase = f32[cursor + 1];
  options.tilt = f32[cursor + 2];
  options.drop = f32[cursor + 3];
  options.settle = f32[cursor + 4];
  options.shiftX = f32[cursor + 5];
  options.failOnHead = f32[cursor + 6] !== 0;
  options.frames = f32[cursor + 7];
  options.amp = f32[cursor + 8];
  options.period = f32[cursor + 9];
  options.phaseShift = f32[cursor + 10];
  options.goalX = f32[cursor + 11] !== 0 ? f32[cursor + 12] : null;
  if (f32[cursor + 13] !== 0) {
    options.ballFrame = f32[cursor + 14];
    options.ballSide = f32[cursor + 15];
    options.ballSpeed = f32[cursor + 16];
  } else options.ballFrame = null;
  let field = cursor + 17;
  for (let t = 0; t < targets.length; t++) {
    const target = targets[t];
    target.on = f32[field++] !== 0;
    target.x = f32[field++];
    target.y = f32[field++];
    target.angle = f32[field++];
  }
}

postMessage("ready");
onmessage = (event) => {
  const data = event.data;
  if (!data || !data.buffer) return;
  if (data.trainingPhase != null) setTrainingPhase(data.trainingPhase);
  const buffer = data.buffer;
  const f32 = new Float32Array(buffer);
  const taskCount = f32[0];
  const weightCount = f32[1];
  const started = performance.now();
  let steps = 0;
  let cursor = HEADER;
  for (let index = 0; index < taskCount; index++) {
    const taskIndex = f32[cursor];
    readOptions(f32, cursor);
    const weights = new Float32Array(buffer, (cursor + TASK_HEADER) * 4, weightCount);
    if (!episode) episode = new sim.Episode(world, weights, options.tilt || 0, options.drop || 0, options);
    else episode.reset(weights, options.tilt || 0, options.drop || 0, options);
    while (!episode.step());
    steps += episode.frame;
    const at = resultIndex(taskCount, weightCount, index);
    const fitness = episode.fitnessValue;
    const goal = episode.options.goalX;
    f32[at] = taskIndex;
    f32[at + 1] = fitness;
    f32[at + 2] = episode.windowMean(false);
    f32[at + 3] = episode.windowMean(true);
    f32[at + 4] = episode.endedUp ? 1 : 0;
    f32[at + 5] = episode.headImpact || 0;
    f32[at + 6] = goal == null ? 0 : Math.abs(goal - (episode.measure.centerX || 0));
    f32[at + 7] = episode.sensedFrames ? episode.heightSum / episode.sensedFrames : 0;
    episode.cleanup();
    cursor += TASK_HEADER + weightCount;
  }
  postMessage({ buffer, ms: performance.now() - started, steps }, [buffer]);
};
