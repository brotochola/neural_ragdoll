import * as Box2D from "https://cdn.jsdelivr.net/npm/phaser-box2d@1.1.0/dist/PhaserBox2D.min.js";
import { createSim } from "./sim.js";
import { config, setTrainingPhase } from "../config.js";

const sim = createSim(Box2D);
const world = sim.build();

postMessage("ready");
onmessage = (event) => {
  if (event.data && event.data.trainingPhase != null) setTrainingPhase(event.data.trainingPhase);
  const results = [];
  const tasks = event.data.tasks || [];
  for (let index = 0; index < tasks.length; index++) {
    const task = tasks[index];
    const weights = task.weights instanceof Float32Array ? task.weights : Float32Array.from(task.weights);
    const options = Object.assign({}, task.options, { phase: (task.phase != null ? task.phase : index) % config.cadence.actionEveryNFrames });
    const episode = new sim.Episode(world, weights, options.tilt || 0, options.drop || 0, options);
    while (!episode.step());
    const stats = episode.stats;
    episode.cleanup();
    results.push({ index: task.index, ...stats });
  }
  postMessage(results);
};
