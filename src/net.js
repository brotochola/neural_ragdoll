// Elman network and the single-parent search.
// The hidden state keeps a fraction of the previous step. The previous action enters that memory through a feedback layer.
// The action is the tanh of a linear layer. The weights, the hidden state, and the action live in Float32Array. There are no tensors.
import { config } from "../config.js";

const net = config.net;
const search = config.search;

// Positive widths from config.net.hidden. A non-positive entry is dropped. Empty falls back to one unit.
function hiddenSizes() {
  const raw = net.hidden;
  const sizes = [];
  if (raw && raw.length) {
    for (let i = 0; i < raw.length; i++) {
      const count = raw[i] | 0;
      if (count > 0) sizes.push(count);
    }
  }
  if (!sizes.length) sizes.push(1);
  return sizes;
}

function buildLayout() {
  const inputCount = net.inputCount;
  const sizes = hiddenSizes();
  const recurrent = sizes[sizes.length - 1];
  const ff = [];
  let cursor = 0;
  let incoming = inputCount;
  for (let layer = 0; layer < sizes.length - 1; layer++) {
    const units = sizes[layer];
    const stride = 1 + incoming;
    const block = units * stride;
    ff.push({ units, in: incoming, stride, base: cursor, block });
    cursor += block;
    incoming = units;
  }
  // First tanh layer, or 0 when the only hidden layer is the recurrent one. probe-targets.js reads these two.
  const feedforward = ff.length ? ff[0].units : 0;
  const feedforwardStride = 1 + inputCount;
  // Weights that multiply the raw observation. Later tanh layers read the layer before them.
  const inputBlock = ff.length ? ff[0].block : 0;
  const feedback = net.feedbackUnits > 0 ? net.feedbackUnits : 0;
  const feedbackStride = 1 + net.jointCount;
  const feedbackBlock = feedback * feedbackStride;
  const feedbackBase = cursor;
  cursor += feedbackBlock;
  const recurrentIn = ff.length ? ff[ff.length - 1].units : inputCount;
  const recurrentStride = 1 + recurrentIn + feedback + recurrent;
  const recurrentBlock = recurrent * recurrentStride;
  const recurrentBase = cursor;
  // Direct path from the target-error columns to each joint's output.
  // Without it, moving the neck to a head command requires the error to survive a
  // compression 99 -> 50 -> 25 and then the output layer has to find it again, all from
  // ONE fitness number. With the shortcut, that linear relation is learned in one step.
  const skip = net.skipTargetInputs ? net.targetInputCount : 0;
  const skipFrom = net.targetInputIndex;
  // Walk skip: each joint output also reads the walk inputs (gait clock, command) directly.
  const walkSkip = net.walkSkip ? net.walkInputCount || 0 : 0;
  const walkFrom = net.walkInputIndex;
  const outputStride = 1 + recurrent + skip + walkSkip;
  return {
    inputCount,
    ff,
    feedforward,
    feedback,
    recurrent,
    recurrentIn,
    skip,
    skipFrom,
    walkSkip,
    walkFrom,
    outputStride,
    feedforwardStride,
    feedbackStride,
    recurrentStride,
    feedforwardBlock: inputBlock,
    inputBlock,
    feedbackBlock,
    feedbackBase,
    recurrentBase,
    outputBase: recurrentBase + recurrentBlock,
  };
}

// One object. ask() and tell() read it once per weight.
const layout = buildLayout();

export class ElmanNetwork {
  // The last hidden layer is the Elman, with memory. Every layer before it is tanh of the previous layer, with no memory. One hidden layer means that layer is the Elman.
  static get layout() {
    return layout;
  }

  // Weights of the current net. With one layer this is the same packing as before.
  static get weightCount() {
    const layout = ElmanNetwork.layout;
    return layout.outputBase + net.jointCount * layout.outputStride;
  }

  // Copies the vector when it has exactly that length. Otherwise null.
  static copy(source) {
    if (!source || source.length !== ElmanNetwork.weightCount) return null;
    return Float32Array.from(source);
  }

  // Weights saved with fewer inputs, widened to this net: every extra input column, at the end
  // of each first-layer row, gets weight 0. The widened net behaves exactly like the old one
  // as long as the new inputs are 0. Returns null when the shapes cannot be matched.
  static widen(source, oldInputCount) {
    const layout = ElmanNetwork.layout;
    const extra = layout.inputCount - oldInputCount;
    if (!source || !(extra > 0) || !layout.ff.length) return null;
    const first = layout.ff[0];
    const oldStride = 1 + oldInputCount;
    if (source.length !== ElmanNetwork.weightCount - extra * first.units) return null;
    const out = new Float32Array(ElmanNetwork.weightCount);
    let from = 0;
    let to = 0;
    for (let unit = 0; unit < first.units; unit++) {
      for (let i = 0; i < oldStride; i++) out[to++] = source[from++];
      to += extra;
    }
    while (from < source.length) out[to++] = source[from++];
    return out;
  }

  // Weights saved with an older observation (fewer walk inputs) and/or a narrower walk skip,
  // converted to this net. New input columns and new walk-skip weights are 0, so the converted
  // net behaves exactly like the old one while the new inputs are 0.
  // oldWalkSkip is how many walk-skip weights each joint row had (0 before the skip existed).
  static convert(source, oldInputCount, oldWalkSkip) {
    const layout = ElmanNetwork.layout;
    if (!source || !layout.ff.length) return null;
    const first = layout.ff[0];
    const oldFirstStride = 1 + oldInputCount;
    const newFirstStride = 1 + layout.inputCount;
    const oldOutStride = layout.outputStride - layout.walkSkip + oldWalkSkip;
    const middle = layout.outputBase - first.block;
    const expected = first.units * oldFirstStride + middle + net.jointCount * oldOutStride;
    if (source.length !== expected || oldInputCount > layout.inputCount || oldWalkSkip > layout.walkSkip) return null;
    const out = new Float32Array(ElmanNetwork.weightCount);
    let from = 0;
    for (let unit = 0; unit < first.units; unit++)
      for (let i = 0; i < oldFirstStride; i++) out[unit * newFirstStride + i] = source[from++];
    for (let i = 0; i < middle; i++) out[first.block + i] = source[from++];
    for (let joint = 0; joint < net.jointCount; joint++)
      for (let i = 0; i < oldOutStride; i++) out[layout.outputBase + joint * layout.outputStride + i] = source[from++];
    return out;
  }

  // Weights saved before the walk skip existed: every joint output row gets walkSkip zero
  // weights at its end. Behaviour is unchanged while the walk inputs are 0.
  static widenOutputs(source) {
    const layout = ElmanNetwork.layout;
    const extra = layout.walkSkip;
    if (!source || !(extra > 0)) return null;
    const oldStride = layout.outputStride - extra;
    if (source.length !== layout.outputBase + net.jointCount * oldStride) return null;
    const out = new Float32Array(ElmanNetwork.weightCount);
    out.set(source.subarray ? source.subarray(0, layout.outputBase) : source.slice(0, layout.outputBase));
    for (let joint = 0; joint < net.jointCount; joint++) {
      for (let i = 0; i < oldStride; i++)
        out[layout.outputBase + joint * layout.outputStride + i] = source[layout.outputBase + joint * oldStride + i];
    }
    return out;
  }

  constructor(weights) {
    const layout = ElmanNetwork.layout;
    this.weights = weights && weights.length === ElmanNetwork.weightCount
      ? weights
      : new Float32Array(ElmanNetwork.weightCount);
    this.hidden = new Float32Array(layout.recurrent);
    this.nextHidden = new Float32Array(layout.recurrent);
    this.ff = [];
    for (let layer = 0; layer < layout.ff.length; layer++)
      this.ff.push(new Float32Array(layout.ff[layer].units));
    this.compressed = this.ff.length ? this.ff[0] : new Float32Array(0);
    this.feedback = new Float32Array(layout.feedback);
    this.feedbackFrom = new Float32Array(net.jointCount);
    this.action = new Float32Array(net.jointCount);
  }

  // One step forward. Writes the new action and leaves `hidden` at this step's state.
  // Each layer before the last is tanh(bias + weights · previous layer), with no memory.
  // The last layer is leak * previous + (1 - leak) * tanh(bias + weights · incoming + weights · feedback + weights · hidden).
  // The feedback layer is tanh(bias + weights · previous action), with no memory. It enters only the last layer.
  // Each joint: tanh(bias + weights · new hidden).
  forward(input) {
    const layout = ElmanNetwork.layout;
    const weights = this.weights;
    const leak = net.hiddenLeak;
    let cursor = 0;
    let incoming = input;
    const ff = layout.ff;
    for (let layer = 0; layer < ff.length; layer++) {
      const spec = ff[layer];
      const acts = this.ff[layer];
      for (let unit = 0; unit < spec.units; unit++) {
        let sum = weights[cursor++];
        for (let i = 0; i < spec.in; i++) sum += weights[cursor++] * incoming[i];
        acts[unit] = Math.tanh(sum);
      }
      incoming = acts;
    }
    this.feedbackFrom.set(this.action);
    for (let unit = 0; unit < layout.feedback; unit++) {
      let sum = weights[cursor++];
      for (let i = 0; i < net.jointCount; i++) sum += weights[cursor++] * this.feedbackFrom[i];
      this.feedback[unit] = Math.tanh(sum);
    }
    const incomingCount = ff.length ? layout.recurrentIn : layout.inputCount;
    for (let unit = 0; unit < layout.recurrent; unit++) {
      let sum = weights[cursor++];
      for (let i = 0; i < incomingCount; i++) sum += weights[cursor++] * incoming[i];
      for (let i = 0; i < layout.feedback; i++) sum += weights[cursor++] * this.feedback[i];
      for (let i = 0; i < layout.recurrent; i++) sum += weights[cursor++] * this.hidden[i];
      this.nextHidden[unit] = leak * this.hidden[unit] + (1 - leak) * Math.tanh(sum);
    }
    for (let joint = 0; joint < net.jointCount; joint++) {
      let sum = weights[cursor++];
      for (let unit = 0; unit < layout.recurrent; unit++) sum += weights[cursor++] * this.nextHidden[unit];
      // Shortcut: the target error goes straight to the joint, not through the hidden-layer bottleneck.
      for (let c = 0; c < layout.skip; c++) sum += weights[cursor++] * input[layout.skipFrom + c];
      for (let c = 0; c < layout.walkSkip; c++) sum += weights[cursor++] * input[layout.walkFrom + c];
      this.action[joint] = Math.tanh(sum);
    }
    if (cursor !== weights.length) return;
    const previousHidden = this.hidden;
    this.hidden = this.nextHidden;
    this.nextHidden = previousHidden;
  }
}

// Zeros the shortcut weights of each joint. All 14 stay trainable: the neck and the
// wrists are back on the list, or the head and hand rotation rows could not move.
export function zeroSkip(weights) {
  const layout = ElmanNetwork.layout;
  if (!layout.skip) return weights;
  for (let joint = 0; joint < net.jointCount; joint++) {
    const from = layout.outputBase + joint * layout.outputStride + 1 + layout.recurrent;
    for (let i = 0; i < layout.skip; i++) weights[from + i] = 0;
  }
  return weights;
}

// 1 when that output joint is trained. The list is config.search.trainableJointIndexes.
// All 14 are on: the neck and the wrists are back, or the head and hand rotation rows
// had no way to move. Built once.
const limbJoint = new Uint8Array(net.jointCount);
for (let jointIndex = 0; jointIndex < search.trainableJointIndexes.length; jointIndex++) {
  const joint = search.trainableJointIndexes[jointIndex];
  if (joint >= 0 && joint < limbJoint.length) limbJoint[joint] = 1;
}

// When set, the noise and Adam touch only those outputs. The rest of the vector stays at the checkpoint.
let outputJointMask = null;

export function setOutputJoints(indexes) {
  if (!indexes) {
    outputJointMask = null;
    return;
  }
  outputJointMask = new Uint8Array(net.jointCount);
  for (let i = 0; i < indexes.length; i++) {
    const joint = indexes[i];
    if (joint >= 0 && joint < outputJointMask.length) outputJointMask[joint] = 1;
  }
}

// How much noise each weight gets. The body barely moves; the limbs and the targets do.
// The large noise on the targets and on the head contact lands only on the weights that multiply that input.
// `boost` grows that noise when the score is not improving. sigmaCap clips it.
function explorationOf(index, scale, boost) {
  const extra = boost > 0 ? boost : 1;
  let sigma = search.bodyExplorationScale * extra;
  if (outputJointMask && index < layout.outputBase) return 0;
  if (index < layout.outputBase) {
    const inFeedback = index >= layout.feedbackBase && index < layout.feedbackBase + layout.feedbackBlock;
    const readsInput = !inFeedback && (layout.inputBlock ? index < layout.inputBlock : index >= layout.recurrentBase);
    if (readsInput) {
      const stride = layout.inputBlock ? layout.feedforwardStride : layout.recurrentStride;
      const origin = layout.inputBlock ? 0 : layout.recurrentBase;
      const local = (index - origin) % stride;
      const targetLocal = net.targetInputIndex + 1;
      if (local >= targetLocal && local < targetLocal + net.targetInputCount) sigma = scale * extra;
      else if (local > net.targetInputIndex + net.targetInputCount && local <= net.sensedInputCount) sigma = scale * extra;
      else if (local > net.sensedInputCount && local <= net.inputCount) sigma = net.freshInputExplorationScale * extra;
    }
  } else {
    const stride = layout.outputStride;
    const joint = Math.floor((index - layout.outputBase) / stride);
    const local = (index - layout.outputBase) % stride;
    const allowed = outputJointMask ? outputJointMask[joint] : limbJoint[joint];
    if (!allowed) sigma = 0;
    // The shortcut starts at zero, so it needs a little more noise than a normal weight
    // for the gradient to wake it, but not the large target noise: that changed the
    // action of every joint and broke balance.
    else if (local > layout.recurrent + layout.skip) sigma = search.walkSkipExplorationScale * extra;
    else if (local > layout.recurrent) sigma = search.skipExplorationScale * extra;
    else sigma = search.limbExplorationScale * extra;
  }
  const cap = search.sigmaCap;
  return sigma > 0 && cap > 0 ? Math.min(cap, sigma) : sigma;
}

// x1 below the threshold, then stagnationBoost, and again every so many generations, up to the cap.
function explorationBoost(flatGens) {
  const every = search.stagnationGenerations;
  if (!(every > 0) || !(flatGens >= every)) return 1;
  const steps = Math.floor(flatGens / every);
  return Math.min(search.stagnationBoostCap, Math.pow(search.stagnationBoost, steps));
}

// 0 below the threshold. After that, a fixed fraction of the pairs. The rest keep the small noise.
function mutationFraction(flatGens) {
  const every = search.stagnationGenerations;
  const cap = search.stagnationMutateCap;
  if (!(every > 0) || !(cap > 0) || !(flatGens >= every)) return 0;
  return Math.min(1, cap);
}

// One-parent search. Each generation is antithetic pairs: the same noise, plus and minus.
// The gradient comes from the fitness rank, not from a derivative. Adam applies it to the weight vector.
export class EvolutionSearch {
  constructor(randomNormal) {
    const parent = new Float32Array(ElmanNetwork.weightCount);
    const outputBase = ElmanNetwork.layout.outputBase;
    for (let i = 0; i < outputBase; i++) parent[i] = randomNormal() * search.hiddenInitScale;
    for (let i = outputBase; i < ElmanNetwork.weightCount; i++) parent[i] = randomNormal() * search.outputInitScale;
    // The shortcut starts at ZERO. Otherwise it puts 25 x noise x 14 joints of perturbation
    // into the action from generation 0, and the doll always falls: skill 1 stopped learning to stand.
    // At zero it is extra capacity, not a change to the initial behaviour: it grows where it helps.
    zeroSkip(parent);
    this.theta = parent;
    this.gen = 0;
    this.hist = [];
    this.k = 0;
    // Adam first and second moments.
    this.m = new Float32Array(ElmanNetwork.weightCount);
    this.v = new Float32Array(ElmanNetwork.weightCount);
    this.bestGenTheta = parent.slice();
    this.bestGenScore = -1e9;
    this.bestEverTheta = parent.slice();
    this.bestEverScore = -1e9;
    this.bestSmooth = null;
    this.eliteScore = -1e9;
    this.flatGens = 0;
    this.mutationBoost = 1;
    this.mutationFraction = 0;
    this.anchorNote = "";
    // How many times the parent changed. If this stays at zero through a plateau, the
    // population is exploring around a fixed parent and the average is not really rising.
    // Look at THIS number before blaming the reward.
    this.parentMoves = 0;
    this.parentMoveNote = "";
    const length = ElmanNetwork.weightCount;
    this.sigmaQuiet = new Float32Array(length);
    this.sigmaHot = new Float32Array(length);
    this.gradient = new Float32Array(length);
    this.noiseStore = [];
    this.childStore = [];
    this.noiseRefs = [];
    this.weightRefs = [];
    this.pairBoosts = [];
    this.phases = [];
    this.values = [];
    this.order = [];
    this.rank = [];
    this.generation = {
      situation: null,
      noise: this.noiseRefs,
      weights: this.weightRefs,
      phases: this.phases,
      count: 0,
      scale: 0,
      boost: 1,
      fraction: 0,
      pairBoosts: this.pairBoosts,
      anchorPairs: 0,
      carryPairs: 0,
      sigmaQuiet: this.sigmaQuiet,
      sigmaHot: this.sigmaHot,
    };
    const self = this;
    this.compareFitness = (a, b) => self.values[a] - self.values[b];
  }

  // Grows the pool only when this generation has more pairs than the last.
  growFloat(store, count, length) {
    for (let i = store.length; i < count; i++) store.push(new Float32Array(length));
  }

  fillSigma(out, scale, boost) {
    for (let i = 0; i < out.length; i++) out[i] = explorationOf(i, scale, boost);
  }

  // Writes one antithetic pair into buffers that already exist.
  writePair(pairIndex, parent, sigma, randomNormal) {
    const length = parent.length;
    const direction = this.noiseStore[pairIndex];
    const plus = this.childStore[pairIndex * 2];
    const minus = this.childStore[pairIndex * 2 + 1];
    for (let i = 0; i < length; i++) {
      direction[i] = randomNormal();
      const step = sigma[i] * direction[i];
      plus[i] = parent[i] + step;
      minus[i] = parent[i] - step;
    }
    this.noiseRefs[pairIndex] = direction;
    this.weightRefs[pairIndex * 2] = plus;
    this.weightRefs[pairIndex * 2 + 1] = minus;
    this.phases[pairIndex * 2] = pairIndex * 2;
    this.phases[pairIndex * 2 + 1] = pairIndex * 2 + 1;
  }

  zeroMoments() {
    this.m.fill(0);
    this.v.fill(0);
    this.k = 0;
  }

  // The bars belong to the previous task. Weights stay. Used when Train marks move.
  forgetRecords() {
    this.eliteScore = -1e9;
    this.bestEverScore = -1e9;
    this.bestSmooth = null;
    this.flatGens = 0;
    this.hist.length = 0;
  }

  // Extra pairs copied from Best ever. Zero until the record exists and the vector matches.
  anchorPairCount() {
    const pairs = search.bestEverPairs | 0;
    if (!(pairs > 0)) return 0;
    const score = this.bestEverScore;
    if (!(Number.isFinite(score) && score > -1e8)) return 0;
    if (!this.bestEverTheta || this.bestEverTheta.length !== this.theta.length) return 0;
    return pairs;
  }

  // Last generation's best child, re-measured. Zero until tell() has written a real score.
  carryPairCount() {
    const pairs = search.bestGenPairs | 0;
    if (!(pairs > 0)) return 0;
    if (!(Number.isFinite(this.bestGenScore) && this.bestGenScore > -1e8)) return 0;
    if (!this.bestGenTheta || this.bestGenTheta.length !== this.theta.length) return 0;
    return pairs;
  }

  // The factor and the fraction the next generation will use. Useful when resuming a run that is already stuck.
  refreshMutationBoost() {
    this.mutationBoost = explorationBoost(this.flatGens);
    this.mutationFraction = mutationFraction(this.flatGens);
  }

  // One situation for the whole generation. Each individual decides on a different frame.
  // Returns the same object every time. It is valid until the next ask.
  ask(situation, count, randomNormal) {
    const parent = this.theta;
    const length = parent.length;
    const scale = Math.max(search.explorationScale, search.explorationFloor * Math.pow(0.5, (this.gen || 0) / search.explorationHalfLifeGenerations));
    let boost = explorationBoost(this.flatGens);
    let fraction = mutationFraction(this.flatGens);
    this.lastScouted = false;
    if (this.handScout) {
      this.handScout = false;
      this.lastScouted = true;
      const forced = search.stagnationBoost > 0 ? search.stagnationBoost : 1;
      boost = Math.max(boost, forced);
      if (search.stagnationBoostCap > 0) boost = Math.min(search.stagnationBoostCap, boost);
      const share = search.stagnationMutateCap > 0 ? Math.min(1, search.stagnationMutateCap) : 0;
      fraction = share > 0 ? share : 1 / Math.max(1, count);
    }
    this.mutationBoost = boost;
    this.mutationFraction = fraction;
    const hot = fraction > 0 ? Math.min(count, Math.max(1, Math.round(count * fraction))) : 0;
    const anchor = this.anchorPairCount();
    const carry = this.carryPairCount();
    const pairTotal = count + anchor + carry;
    this.growFloat(this.noiseStore, pairTotal, length);
    this.growFloat(this.childStore, pairTotal * 2, length);
    this.fillSigma(this.sigmaQuiet, scale, 1);
    this.fillSigma(this.sigmaHot, scale, boost);
    this.pairBoosts.length = pairTotal;
    this.noiseRefs.length = pairTotal;
    this.weightRefs.length = pairTotal * 2;
    this.phases.length = pairTotal * 2;
    for (let individual = 0; individual < count; individual++) {
      const pairBoost = individual < hot ? boost : 1;
      this.pairBoosts[individual] = pairBoost;
      this.writePair(individual, parent, pairBoost > 1 ? this.sigmaHot : this.sigmaQuiet, randomNormal);
    }
    // Light copies of Best ever, at the end. count stays just the parent's pairs.
    const anchorParent = this.bestEverTheta;
    for (let individual = 0; individual < anchor; individual++) {
      const pairIndex = count + individual;
      this.pairBoosts[pairIndex] = 1;
      this.writePair(pairIndex, anchorParent, this.sigmaQuiet, randomNormal);
    }
    const carryParent = this.bestGenTheta;
    for (let individual = 0; individual < carry; individual++) {
      const pairIndex = count + anchor + individual;
      this.pairBoosts[pairIndex] = 1;
      this.writePair(pairIndex, carryParent, this.sigmaQuiet, randomNormal);
    }
    const generation = this.generation;
    generation.situation = situation;
    generation.count = count;
    generation.scale = scale;
    generation.boost = boost;
    generation.fraction = fraction;
    generation.anchorPairs = anchor;
    generation.carryPairs = carry;
    return generation;
  }

  tell(generation, scores) {
    const count = generation.count;
    const values = this.values;
    values.length = scores.length;
    for (let i = 0; i < scores.length; i++) values[i] = scores[i].fitness;
    const length = this.theta.length;
    const thetaChildren = count * 2;
    if (!this.m || this.m.length !== length) this.m = new Float32Array(length);
    if (!this.v || this.v.length !== length) this.v = new Float32Array(length);
    this.k = this.k || 0;
    const order = this.order;
    order.length = thetaChildren;
    for (let i = 0; i < thetaChildren; i++) order[i] = i;
    order.sort(this.compareFitness);
    const rank = this.rank;
    rank.length = thetaChildren;
    const span = Math.max(1, thetaChildren - 1);
    for (let place = 0; place < order.length; place++) rank[order[place]] = place / span - 0.5;
    let low = Infinity;
    let high = -Infinity;
    for (let i = 0; i < thetaChildren; i++) {
      const value = values[i];
      if (value < low) low = value;
      if (value > high) high = value;
    }
    // A total tie: the rank says nothing and Adam would still step. The parent is left alone.
    const boost = generation.boost || 1;
    if (high > low) {
      const gradient = this.gradient;
      gradient.fill(0);
      const sigmaQuiet = generation.sigmaQuiet || this.sigmaQuiet;
      const sigmaHot = generation.sigmaHot || this.sigmaHot;
      const pairBoosts = generation.pairBoosts;
      for (let pair = 0; pair < count; pair++) {
        const plus = values[2 * pair];
        const minus = values[2 * pair + 1];
        if (plus === minus) continue;
        const pairBoost = pairBoosts && pairBoosts[pair] > 0 ? pairBoosts[pair] : boost;
        const sigma = pairBoost > 1 ? sigmaHot : sigmaQuiet;
        const difference = rank[2 * pair] - rank[2 * pair + 1];
        const direction = generation.noise[pair];
        for (let i = 0; i < length; i++) {
          const scale = sigma[i];
          if (!scale) continue;
          gradient[i] += difference * direction[i] / scale;
        }
      }
      const stepIndex = ++this.k;
      const firstMoment = search.adamBeta1;
      const secondMoment = search.adamBeta2;
      const firstFix = 1 - Math.pow(firstMoment, stepIndex);
      const secondFix = 1 - Math.pow(secondMoment, stepIndex);
      const stepSize = Math.max(search.learningRateFloor, search.learningRate * Math.pow(0.5, (this.gen || 0) / search.learningRateHalfLifeGenerations));
      for (let i = 0; i < length; i++) {
        if (!sigmaQuiet[i]) continue;
        const estimate = gradient[i] / count;
        this.m[i] = firstMoment * this.m[i] + (1 - firstMoment) * estimate;
        this.v[i] = secondMoment * this.v[i] + (1 - secondMoment) * estimate * estimate;
        this.theta[i] += (stepSize * (this.m[i] / firstFix)) / (Math.sqrt(this.v[i] / secondFix) + search.adamEpsilon);
      }
    }
    let bestIndex = 0;
    for (let i = 1; i < values.length; i++) if (values[i] > values[bestIndex]) bestIndex = i;
    this.bestGenTheta.set(generation.weights[bestIndex]);
    this.bestGenScore = values[bestIndex];
    // The parent is replaced only when one of ITS children reaches the marker.
    // `order` is sorted over EVERY child, including the Best ever copies, so the index of
    // the best one does not say who it came from. With bestEverPairs > 0 a copy of the
    // record (an average of 16 generations ago, and on another scale if the reward changed)
    // could become the parent. The guard also requires the child to come from the parent.
    // The parent's pairs are the first `count`, indexes [0, count); the anchor's come after.
    let parentBestIndex = order.length ? order[order.length - 1] : -1;
    if (parentBestIndex < (generation.anchorPairs | 0)) parentBestIndex = -1;
    // The marker decides whether a child can become the parent. The guard used to be
    // (gap ?? 9) < goalCloseEnoughMeters with goalCloseEnoughMeters at -1, and a distance
    // is NEVER less than -1. Elitism was CLOSED: the parent never changed, and the only
    // path left was adoptAnchor, which asks for 1000 generations of plateau. Measured:
    // 400 generations of skill 1 and the parent did not move ONCE. The average rose from
    // the Adam step, not from selection, which is why the eval was 7/20 at gen 300 and
    // 1/20 at 400 while the record climbed. Elitism is measured by parentMoves, which is logged.
    //
    // If the situation has NO marker (the five target levels), there is no closeness to
    // require: the child only has to beat the parent's marker. If it has a marker, the
    // guard still asks for closeness, and goalCloseEnoughMeters has to be a positive distance.
    const row = parentBestIndex >= 0 ? scores[parentBestIndex] || {} : {};
    const hasGoal = row.goalX != null;
    const goalGate = row.gap ?? search.eliteMissingGapMeters;
    const goalOpen = !hasGoal || goalGate < search.goalCloseEnoughMeters;
    // search.elitism off: a child scored on ONE episode never replaces the parent. Measured
    // on a fresh skill-1 run: with it on (plus adoptAnchor), the parent changed 97 times in
    // 200 generations on lucky single episodes and the eval went 14/20 -> 0/20. The Adam
    // step on the ranked antithetic gradient is what learns; the held-out eval keeps the best.
    if (
      search.elitism !== false &&
      parentBestIndex >= 0 &&
      values[parentBestIndex] > (this.eliteScore ?? -1e9) &&
      goalOpen
    ) {
      this.theta.set(generation.weights[parentBestIndex]);
      this.eliteScore = values[parentBestIndex];
      this.parentMoves++;
      this.parentMoveNote =
        (hasGoal ? "elite(meta) " : "elite ") +
        values[parentBestIndex].toFixed(3) +
        " gap " + Number(goalGate).toFixed(2);
      this.zeroMoments();
    } else if (parentBestIndex >= 0) {
      // Why it was not promoted. "goal closed" means the situation has a marker and the
      // child did not arrive: that is correct. A frozen parent is what to look at.
      this.parentMoveNote =
        (goalOpen
          ? "elite "
          : "meta " + Number(goalGate).toFixed(2) + " > " + search.goalCloseEnoughMeters + ", ") +
        "child " + values[parentBestIndex].toFixed(3) +
        " vs elite " + (this.eliteScore ?? -1e9).toFixed(3);
    }
    this.gen++;
    let stoodCount = 0;
    let startSum = 0;
    let startCount = 0;
    let endSum = 0;
    let endCount = 0;
    let gapSum = 0;
    let parentSum = 0;
    for (let i = 0; i < thetaChildren; i++) {
      const score = scores[i];
      parentSum += values[i];
      gapSum += score.gap || 0;
      if (score.stoodUp && (score.headImpact || 0) < search.headImpactQuiet) stoodCount++;
      if (Number.isFinite(score.start)) {
        startSum += score.start;
        startCount++;
      }
      if (Number.isFinite(score.end)) {
        endSum += score.end;
        endCount++;
      }
    }
    const parentMean = thetaChildren ? parentSum / thetaChildren : 0;
    let bestValue = parentMean;
    for (let i = 0; i < values.length; i++) if (values[i] > bestValue) bestValue = values[i];
    this.hist.push({
      mean: parentMean,
      best: values.length ? bestValue : parentMean,
      start: startCount ? startSum / startCount : null,
      end: endCount ? endSum / endCount : null,
      stood: stoodCount / (thetaChildren || 1),
      gap: gapSum / (thetaChildren || 1),
      boost,
      fraction: generation.fraction || 0,
    });
    if (this.hist.length > 400) this.hist.shift();
    this.noteSmoothRecord();
    this.adoptAnchor(generation, values, thetaChildren);
    this.mutationBoost = explorationBoost(this.flatGens);
    this.mutationFraction = mutationFraction(this.flatGens);
  }

  // On a plateau, a small-noise child becomes the parent when it beats its twin and the average.
  // A Best ever copy gets in even if it ties its twin. A scout does not. The record is not touched.
  adoptAnchor(generation, values, thetaChildren) {
    this.anchorNote = "";
    if (!((this.flatGens || 0) >= search.stagnationGenerations)) return;
    const margin = search.bestEverAdoptMargin;
    if (!(margin > 0) || !values.length) return;
    let parentSum = 0;
    for (let i = 0; i < thetaChildren; i++) parentSum += values[i];
    const parentMean = thetaChildren ? parentSum / thetaChildren : 0;
    const boosts = generation.pairBoosts || [];
    let bestIndex = -1;
    let bestValue = -Infinity;
    const pairCount = Math.floor(values.length / 2);
    for (let pair = 0; pair < pairCount; pair++) {
      const boost = boosts[pair] > 0 ? boosts[pair] : 1;
      if (boost > 1) continue;
      const left = pair * 2;
      const right = left + 1;
      const winner = values[left] >= values[right] ? left : right;
      const loserValue = values[winner === left ? right : left];
      const fromBestEver = left >= thetaChildren;
      if (!(values[winner] > parentMean + margin)) continue;
      if (!fromBestEver && !(values[winner] > loserValue + margin)) continue;
      if (values[winner] > bestValue) {
        bestValue = values[winner];
        bestIndex = winner;
      }
    }
    if (bestIndex < 0) return;
    this.theta.set(generation.weights[bestIndex]);
    this.parentMoves++;
    this.zeroMoments();
    const source = bestIndex >= thetaChildren ? "best ever child " : "child ";
    this.anchorNote = source + bestValue.toFixed(2) + " became the parent";
  }

  // The record is the average of the endings of the last generations, and the saved weights are the parent.
  // A lucky child that was already standing does not qualify.
  noteSmoothRecord() {
    const window = search.recordWindow;
    if (!(window > 0) || this.hist.length < window) return;
    const start = this.hist.length - window;
    let sum = 0;
    for (let i = start; i < this.hist.length; i++) {
      const row = this.hist[i];
      if (!Number.isFinite(row.start)) return;
      sum += row.mean;
    }
    const smooth = sum / window;
    if (!Number.isFinite(this.bestSmooth)) {
      this.bestSmooth = smooth;
      return;
    }
    if (smooth > this.bestSmooth + search.recordMargin) {
      this.bestSmooth = smooth;
      this.bestEverTheta = this.theta.slice();
      this.bestEverScore = smooth;
      this.bestEverGen = this.gen;
      this.flatGens = Math.max(0, (this.flatGens || 0) - search.stagnationGenerations);
    } else {
      this.flatGens = (this.flatGens || 0) + 1;
      const after = search.revertAfterGenerations | 0;
      const saved = this.bestEverTheta;
      if (after > 0 && this.flatGens >= after && saved && saved.length === this.theta.length && this.bestEverScore > -1e8) {
        this.theta.set(saved);
        this.zeroMoments();
        this.flatGens = 0;
      }
    }
  }
}

// Small leg net. The balance prior is not touched. A few hundred weights, not thousands.
export const residualInputs = 12;
export const residualHidden = 16;
export const residualOutputs = 6;

export function residualWeightCount() {
  return residualHidden * (residualInputs + 1) + residualOutputs * (residualHidden + 1);
}

export class ResidualNet {
  constructor(weights) {
    const count = residualWeightCount();
    this.weights = weights && weights.length === count ? weights : new Float32Array(count);
    this.hidden = new Float32Array(residualHidden);
    this.out = new Float32Array(residualOutputs);
  }

  forward(input) {
    const weights = this.weights;
    let cursor = 0;
    for (let unit = 0; unit < residualHidden; unit++) {
      let sum = weights[cursor++];
      for (let i = 0; i < residualInputs; i++) sum += weights[cursor++] * input[i];
      this.hidden[unit] = Math.tanh(sum);
    }
    for (let joint = 0; joint < residualOutputs; joint++) {
      let sum = weights[cursor++];
      for (let unit = 0; unit < residualHidden; unit++) sum += weights[cursor++] * this.hidden[unit];
      this.out[joint] = Math.tanh(sum);
    }
  }
}

// The same antithetic search and Adam, only on the small net. Uniform sigma.
export class ResidualSearch {
  constructor(randomNormal) {
    const length = residualWeightCount();
    this.theta = new Float32Array(length);
    this.m = new Float32Array(length);
    this.v = new Float32Array(length);
    this.k = 0;
    this.gen = 0;
    this.sigma = 0.08;
    this.lr = 0.02;
    this.length = length;
    this.noiseStore = [];
    this.childStore = [];
    this.noiseRefs = [];
    this.weightRefs = [];
    this.values = [];
    this.order = [];
    this.rank = [];
    this.gradient = new Float32Array(length);
    const self = this;
    this.compareFitness = (a, b) => self.values[a] - self.values[b];
    if (randomNormal) this.randomNormal = randomNormal;
  }

  zeroMoments() {
    this.m.fill(0);
    this.v.fill(0);
    this.k = 0;
  }

  ask(count, randomNormal) {
    const length = this.length;
    const draw = randomNormal || this.randomNormal;
    while (this.noiseStore.length < count) this.noiseStore.push(new Float32Array(length));
    while (this.childStore.length < count * 2) this.childStore.push(new Float32Array(length));
    this.noiseRefs.length = count;
    this.weightRefs.length = count * 2;
    const sigma = this.sigma;
    for (let pair = 0; pair < count; pair++) {
      const direction = this.noiseStore[pair];
      const plus = this.childStore[pair * 2];
      const minus = this.childStore[pair * 2 + 1];
      for (let i = 0; i < length; i++) {
        direction[i] = draw();
        const step = sigma * direction[i];
        plus[i] = this.theta[i] + step;
        minus[i] = this.theta[i] - step;
      }
      this.noiseRefs[pair] = direction;
      this.weightRefs[pair * 2] = plus;
      this.weightRefs[pair * 2 + 1] = minus;
    }
    return { count, weights: this.weightRefs, noise: this.noiseRefs };
  }

  tell(generation, scores) {
    const count = generation.count;
    const values = this.values;
    values.length = scores.length;
    for (let i = 0; i < scores.length; i++) values[i] = scores[i].fitness;
    const length = this.length;
    const children = count * 2;
    const order = this.order;
    order.length = children;
    for (let i = 0; i < children; i++) order[i] = i;
    order.sort(this.compareFitness);
    const rank = this.rank;
    rank.length = children;
    const span = Math.max(1, children - 1);
    for (let place = 0; place < children; place++) rank[order[place]] = place / span - 0.5;
    let low = Infinity;
    let high = -Infinity;
    for (let i = 0; i < children; i++) {
      if (values[i] < low) low = values[i];
      if (values[i] > high) high = values[i];
    }
    if (high > low) {
      const gradient = this.gradient;
      gradient.fill(0);
      const sigma = this.sigma;
      for (let pair = 0; pair < count; pair++) {
        const plus = values[2 * pair];
        const minus = values[2 * pair + 1];
        if (plus === minus) continue;
        const difference = rank[2 * pair] - rank[2 * pair + 1];
        const direction = generation.noise[pair];
        for (let i = 0; i < length; i++) gradient[i] += difference * direction[i] / sigma;
      }
      const stepIndex = ++this.k;
      const b1 = 0.9;
      const b2 = 0.999;
      const fix1 = 1 - Math.pow(b1, stepIndex);
      const fix2 = 1 - Math.pow(b2, stepIndex);
      for (let i = 0; i < length; i++) {
        const estimate = gradient[i] / count;
        this.m[i] = b1 * this.m[i] + (1 - b1) * estimate;
        this.v[i] = b2 * this.v[i] + (1 - b2) * estimate * estimate;
        this.theta[i] += (this.lr * (this.m[i] / fix1)) / (Math.sqrt(this.v[i] / fix2) + 1e-8);
      }
    }
    this.gen++;
  }
}
