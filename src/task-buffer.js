// One ArrayBuffer per send. The header, the weights, and the scores share it.
// The main thread transfers it and the worker sends it back. Neither side touches it
// while the other owns it.

import { config } from "../config.js";

export const HEADER = 3;
// 17 situation floats, 20 for the five marks, the step plan, and one foot placement
// (index, meters, turn). config.skill.planSteps is 4. A longer plan keeps the first four.
export const STEP_SLOTS = 4;
export const BOX_SLOTS = 4;
// command index, stance, then cx/cy/ca for each of the five marks.
export const COMMAND_FLOATS = 2 + 5 * 3;
export const TASK_HEADER = 17 + 20 + 2 + STEP_SLOTS * 4 + 1 + BOX_SLOTS * 4 + 3 + COMMAND_FLOATS;
export const RESULT_STRIDE = 8;

export function bytesFor(taskCount, weightCount) {
  const floats = HEADER + taskCount * (TASK_HEADER + weightCount) + taskCount * RESULT_STRIDE;
  return floats * 4;
}

export function resultIndex(taskCount, weightCount, index) {
  return HEADER + taskCount * (TASK_HEADER + weightCount) + index * RESULT_STRIDE;
}

export function writeTasks(buffer, tasks, weightCount) {
  const f32 = new Float32Array(buffer);
  const taskCount = tasks.length;
  f32[0] = taskCount;
  f32[1] = weightCount;
  f32[2] = TASK_HEADER;
  let cursor = HEADER;
  for (let i = 0; i < taskCount; i++) {
    const task = tasks[i];
    const options = task.options || {};
    f32[cursor] = task.index;
    f32[cursor + 1] = task.phase || 0;
    f32[cursor + 2] = options.tilt || 0;
    f32[cursor + 3] = options.drop || 0;
    f32[cursor + 4] = options.settle || 0;
    f32[cursor + 5] = options.shiftX || 0;
    f32[cursor + 6] = options.failOnHead ? 1 : 0;
    f32[cursor + 7] = options.frames || 0;
    f32[cursor + 8] = options.amp || 0;
    f32[cursor + 9] = options.period || 0;
    f32[cursor + 10] = options.phaseShift || 0;
    const hasGoal = options.goalX != null && Number.isFinite(+options.goalX);
    f32[cursor + 11] = hasGoal ? 1 : 0;
    f32[cursor + 12] = hasGoal ? options.goalX : 0;
    const hasBall = options.ballFrame != null;
    f32[cursor + 13] = hasBall ? 1 : 0;
    f32[cursor + 14] = hasBall ? options.ballFrame : 0;
    f32[cursor + 15] = options.ballSide || 0;
    f32[cursor + 16] = options.ballSpeed || 0;
    const targets = options.targets || [];
    let field = cursor + 17;
    for (let t = 0; t < 5; t++) {
      const target = targets[t] || {};
      f32[field++] = target.on ? 1 : 0;
      f32[field++] = options.failOnHead ? (target.dx || 0) : (target.x || 0);
      f32[field++] = options.failOnHead ? (target.dy || 0) : (target.y || 0);
      f32[field++] = options.failOnHead ? (target.turn || 0) : (target.angle || 0);
    }
    f32[cursor + 37] = options.skill || 0;
    const steps = options.steps || [];
    const stepCount = Math.min(steps.length, STEP_SLOTS);
    f32[cursor + 38] = stepCount;
    let plan = cursor + 39;
    for (let s = 0; s < STEP_SLOTS; s++) {
      const step = s < stepCount ? steps[s] : {};
      f32[plan++] = step.foot || 0;
      f32[plan++] = step.dx || 0;
      f32[plan++] = step.dy || 0;
      f32[plan++] = step.turn || 0;
    }
    const boxes = options.boxes || [];
    const boxCount = Math.min(boxes.length, BOX_SLOTS);
    f32[plan++] = boxCount;
    for (let b = 0; b < BOX_SLOTS; b++) {
      const box = b < boxCount ? boxes[b] : {};
      f32[plan++] = box.dx || 0;
      f32[plan++] = box.halfW || 0;
      f32[plan++] = box.halfH || 0;
      f32[plan++] = box.dir || 0;
    }
    const place = options.place;
    f32[plan++] = place ? place.foot : -1;
    f32[plan++] = place ? place.extra || 0 : 0;
    f32[plan++] = place ? place.turn || 0 : 0;
    const command = options.placed ? 6 : config.commands.indexOf(options.commands || "");
    f32[plan++] = command < 0 ? 0 : command;
    f32[plan++] = options.stance || 0;
    for (let t = 0; t < 5; t++) {
      const target = targets[t] || {};
      if (options.placed) {
        f32[plan++] = target.x || 0;
        f32[plan++] = target.y || 0;
        f32[plan++] = target.angle || 0;
      } else {
        f32[plan++] = target.cx || 0;
        f32[plan++] = target.cy || 0;
        f32[plan++] = target.ca || 0;
      }
    }
    f32.set(task.weights, cursor + TASK_HEADER);
    cursor += TASK_HEADER + weightCount;
  }
  return f32;
}

export function readScore(f32, taskCount, weightCount, index, row) {
  const at = resultIndex(taskCount, weightCount, index);
  const fitness = f32[at + 1];
  row.index = f32[at];
  row.fitness = fitness;
  row.f = fitness;
  row.start = f32[at + 2];
  row.end = f32[at + 3];
  row.stoodUp = f32[at + 4] !== 0;
  row.up = row.stoodUp;
  row.headImpact = f32[at + 5];
  row.gap = f32[at + 6];
  row.hy = f32[at + 7];
  row.stance = 0;
  return row;
}
