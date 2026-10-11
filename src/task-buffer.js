// Un ArrayBuffer por envío. La cabecera, los pesos y los scores van en el mismo buffer.
// El main lo transfiere, el worker lo devuelve. Nadie lo toca mientras está del otro lado.

export const HEADER = 3;
export const TASK_HEADER = 37;
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
      f32[field++] = target.x || 0;
      f32[field++] = target.y || 0;
      f32[field++] = target.angle || 0;
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
