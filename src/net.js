// Elman net and the one-parent search. h_t keeps a leak of h_{t-1}. a_t = tanh(W h_t).
import {
  hiddenUnitCount,
  jointCount,
  inputCount,
  legacyInputCount,
  hiddenLeak,
  explorationScale,
  explorationFloor,
  explorationHalfLifeGenerations,
  limbExplorationScale,
  learningRate,
  learningRateFloor,
  learningRateHalfLifeGenerations,
  trainableJointIndexes,
  goalCloseEnoughMeters,
} from "../config.js";

export function weightCountOf(hiddenCount) {
  return hiddenCount * (inputCount + hiddenCount + 1) + jointCount * (hiddenCount + 1);
}
export function hiddenCountOf(length) {
  const span = inputCount + 1 + jointCount;
  const discriminant = span * span - 4 * (jointCount - length);
  if (!(discriminant >= 0)) return 0;
  const hiddenCount = Math.round((-span + Math.sqrt(discriminant)) / 2);
  return hiddenCount >= 1 && weightCountOf(hiddenCount) === length ? hiddenCount : 0;
}
export const weightCount = weightCountOf(hiddenUnitCount);

const hiddenStride = inputCount + hiddenUnitCount + 1;
const hiddenBlock = hiddenUnitCount * hiddenStride;

function isGoalInputWeight(index) {
  if (index >= hiddenBlock) return false;
  const local = index % hiddenStride;
  return local === inputCount - 1 || local === inputCount;
}
function explorationOf(index, scale) {
  if (isGoalInputWeight(index)) return scale;
  if (index < hiddenBlock) return 0;
  const joint = Math.floor((index - hiddenBlock) / (hiddenUnitCount + 1));
  return trainableJointIndexes.includes(joint) ? limbExplorationScale : 0;
}

// Older saves have fewer inputs. New columns are inserted as zeros so the old policy is unchanged.
export function padWeights(source) {
  if (!source || !source.length) return null;
  if (hiddenCountOf(source.length)) return Float32Array.from(source);
  const oldInputCount = (source.length - jointCount * (hiddenUnitCount + 1)) / hiddenUnitCount - hiddenUnitCount - 1;
  if (oldInputCount !== (oldInputCount | 0) || oldInputCount < 1 || oldInputCount >= inputCount) return null;
  const extra = inputCount - oldInputCount;
  const padded = new Float32Array(weightCount);
  let read = 0;
  let write = 0;
  for (let unit = 0; unit < hiddenUnitCount; unit++) {
    padded[write++] = source[read++];
    for (let i = 0; i < oldInputCount; i++) padded[write++] = source[read++];
    write += extra;
    for (let i = 0; i < hiddenUnitCount; i++) padded[write++] = source[read++];
  }
  for (let i = 0; i < jointCount * (hiddenUnitCount + 1); i++) padded[write++] = source[read++];
  return padded;
}

export function netsFromCheckpoint(checkpoint) {
  const seed = padWeights(checkpoint.bestTheta) || padWeights(checkpoint.thetaStand) || padWeights(checkpoint.theta);
  if (!seed) throw new Error("checkpoint size does not match this network");
  const take = (weights) => padWeights(weights) || seed;
  return { parent: take(checkpoint.theta), bestEver: take(checkpoint.bestTheta) || seed.slice() };
}

// Writes the next hidden state and the action into caller-owned buffers.
export function forwardInto(weights, input, hidden, nextHidden, action) {
  const hiddenSize = hidden.length;
  let cursor = 0;
  for (let unit = 0; unit < hiddenSize; unit++) {
    let sum = weights[cursor++];
    for (let i = 0; i < inputCount; i++) sum += weights[cursor++] * input[i];
    for (let i = 0; i < hiddenSize; i++) sum += weights[cursor++] * hidden[i];
    nextHidden[unit] = hiddenLeak * hidden[unit] + (1 - hiddenLeak) * Math.tanh(sum);
  }
  for (let joint = 0; joint < jointCount; joint++) {
    let sum = weights[cursor++];
    for (let unit = 0; unit < hiddenSize; unit++) sum += weights[cursor++] * nextHidden[unit];
    action[joint] = Math.tanh(sum);
  }
}

export function makeRandom(seed) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let mixed = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    mixed = (mixed + Math.imul(mixed ^ (mixed >>> 7), 61 | mixed)) ^ mixed;
    return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296;
  };
}

export function freshState(randomNormal) {
  const parent = new Float32Array(weightCount);
  const hiddenEnd = hiddenBlock;
  for (let i = 0; i < hiddenEnd; i++) parent[i] = randomNormal() * 0.12;
  for (let i = hiddenEnd; i < weightCount; i++) parent[i] = randomNormal() * 0.04;
  return {
    theta: parent,
    gen: 0,
    hist: [],
    k: 0,
    m: new Float32Array(weightCount),
    v: new Float32Array(weightCount),
    bestGenTheta: parent.slice(),
    bestGenScore: -1e9,
    bestEverTheta: parent.slice(),
    bestEverScore: -1e9,
    eliteScore: -1e9,
  };
}

// One shared situation for the generation, plus a phase so agents do not decide on the same frame.
export function askGeneration(state, situation, count, randomNormal) {
  const parent = state.theta;
  const length = parent.length;
  const scale = Math.max(explorationScale, explorationFloor * Math.pow(0.5, (state.gen || 0) / explorationHalfLifeGenerations));
  const noise = [];
  const weights = [];
  const phases = [];
  for (let individual = 0; individual < count; individual++) {
    const direction = new Float32Array(length);
    for (let i = 0; i < length; i++) direction[i] = randomNormal();
    noise.push(direction);
    for (const sign of [1, -1]) {
      const child = new Float32Array(length);
      for (let i = 0; i < length; i++) child[i] = parent[i] + sign * explorationOf(i, scale) * direction[i];
      weights.push(child);
      phases.push(weights.length - 1);
    }
  }
  return { situation, noise, weights, phases, count, scale };
}

export function tellGeneration(state, generation, scores) {
  const count = generation.count;
  const values = scores.map((score) => score.fitness);
  const length = state.theta.length;
  if (!state.m || state.m.length !== length) state.m = new Float32Array(length);
  if (!state.v || state.v.length !== length) state.v = new Float32Array(length);
  state.k = state.k || 0;
  const order = values.map((_, index) => index).sort((a, b) => values[a] - values[b]);
  const rank = new Array(values.length);
  order.forEach((index, place) => { rank[index] = place / (values.length - 1) - 0.5; });
  const gradient = new Float32Array(length);
  const scale = generation.scale || explorationScale;
  for (let pair = 0; pair < count; pair++) {
    const difference = rank[2 * pair] - rank[2 * pair + 1];
    for (let i = 0; i < length; i++) if (explorationOf(i, scale)) gradient[i] += difference * generation.noise[pair][i];
  }
  const stepIndex = ++state.k;
  const firstMoment = 0.9;
  const secondMoment = 0.999;
  const firstFix = 1 - Math.pow(firstMoment, stepIndex);
  const secondFix = 1 - Math.pow(secondMoment, stepIndex);
  const stepSize = Math.max(learningRateFloor, learningRate * Math.pow(0.5, (state.gen || 0) / learningRateHalfLifeGenerations));
  for (let i = 0; i < length; i++) {
    const coordinateScale = explorationOf(i, scale);
    if (!coordinateScale) continue;
    const estimate = gradient[i] / (count * coordinateScale);
    state.m[i] = firstMoment * state.m[i] + (1 - firstMoment) * estimate;
    state.v[i] = secondMoment * state.v[i] + (1 - secondMoment) * estimate * estimate;
    state.theta[i] += (stepSize * (state.m[i] / firstFix)) / (Math.sqrt(state.v[i] / secondFix) + 1e-8);
  }
  const bestIndex = order[order.length - 1];
  const bestWeights = generation.weights[bestIndex];
  state.bestGenTheta = Float32Array.from(bestWeights);
  state.bestGenScore = values[bestIndex];
  if (values[bestIndex] > (state.bestEverScore ?? -1e9)) {
    state.bestEverTheta = Float32Array.from(bestWeights);
    state.bestEverScore = values[bestIndex];
  }
  // Only replace the parent when someone actually arrived. A high score from standing still does not.
  if (values[bestIndex] > (state.eliteScore ?? -1e9) && (scores[bestIndex].gap ?? 9) < goalCloseEnoughMeters) {
    state.theta.set(bestWeights);
    state.eliteScore = values[bestIndex];
    state.m = new Float32Array(length);
    state.v = new Float32Array(length);
    state.k = 0;
  }
  state.gen++;
  const stood = scores.filter((score) => score.stoodUp && (score.headImpact || 0) < 1.5).length / scores.length;
  state.hist.push({
    mean: values.reduce((sum, value) => sum + value, 0) / values.length,
    best: Math.max(...values),
    stood,
    gap: scores.reduce((sum, score) => sum + (score.gap || 0), 0) / scores.length,
  });
  if (state.hist.length > 400) state.hist = state.hist.slice(-400);
}

export { legacyInputCount };
