// Red de Elman y la búsqueda de un solo padre.
// El estado oculto guarda una fracción del paso anterior. La acción anterior entra por una capa de feedback a esa memoria.
// La acción es la tanh de una capa lineal. Los pesos, el estado oculto y la acción viven en Float32Array. No hay tensores.
import { config } from "../config.js";

const net = config.net;
const search = config.search;

// Unidades de una capa. Ratio 0 apaga la capa. Si no, es la fracción de las entradas, redondeada.
function unitsFromRatio(ratio) {
  if (!(ratio > 0)) return 0;
  return Math.round(net.inputCount * ratio);
}

function buildLayout() {
  const inputCount = net.inputCount;
  const first = Math.max(1, unitsFromRatio(net.hidden1Ratio));
  const second = unitsFromRatio(net.hidden2Ratio);
  const feedforward = second > 0 ? first : 0;
  const recurrent = second > 0 ? second : first;
  const feedback = net.feedbackUnits > 0 ? net.feedbackUnits : 0;
  const feedforwardStride = 1 + inputCount;
  const feedbackStride = 1 + net.jointCount;
  const recurrentStride = 1 + (feedforward || inputCount) + feedback + recurrent;
  const feedforwardBlock = feedforward * feedforwardStride;
  const feedbackBlock = feedback * feedbackStride;
  const recurrentBlock = recurrent * recurrentStride;
  const feedbackBase = feedforwardBlock;
  const recurrentBase = feedforwardBlock + feedbackBlock;
  return {
    inputCount,
    feedforward,
    feedback,
    recurrent,
    feedforwardStride,
    feedbackStride,
    recurrentStride,
    feedforwardBlock,
    feedbackBlock,
    feedbackBase,
    recurrentBase,
    outputBase: recurrentBase + recurrentBlock,
  };
}

// Un solo objeto. ask() y tell() lo leen una vez por peso.
const layout = buildLayout();

export class ElmanNetwork {
  // Una capa: la primera es la Elman, con memoria. Dos capas: la primera comprime sin memoria y la segunda es la única que recuerda.
  static get layout() {
    return layout;
  }

  // Pesos de la red actual. Con una sola capa es el mismo empaquetado de antes.
  static get weightCount() {
    const layout = ElmanNetwork.layout;
    return layout.outputBase + net.jointCount * (layout.recurrent + 1);
  }

  // Copia el vector si tiene exactamente ese largo. Si no, null.
  static copy(source) {
    if (!source || source.length !== ElmanNetwork.weightCount) return null;
    return Float32Array.from(source);
  }

  constructor(weights) {
    const layout = ElmanNetwork.layout;
    this.weights = weights && weights.length === ElmanNetwork.weightCount
      ? weights
      : new Float32Array(ElmanNetwork.weightCount);
    this.hidden = new Float32Array(layout.recurrent);
    this.nextHidden = new Float32Array(layout.recurrent);
    this.compressed = new Float32Array(layout.feedforward);
    this.feedback = new Float32Array(layout.feedback);
    this.feedbackFrom = new Float32Array(net.jointCount);
    this.action = new Float32Array(net.jointCount);
  }

  // Un paso adelante. Escribe la acción nueva y deja `hidden` en el estado de este paso.
  // La capa de feedback es tanh(sesgo + pesos · acción anterior), sin memoria. Entra solo a la capa recurrente.
  // Con una capa, cada unidad oculta es leak * anterior + (1 - leak) * tanh(sesgo + pesos · entrada + pesos · feedback + pesos · oculto).
  // Con dos, la primera es tanh(sesgo + pesos · entrada), sin memoria. La segunda es la Elman.
  // Cada articulación: tanh(sesgo + pesos · oculto nuevo).
  forward(input) {
    const layout = ElmanNetwork.layout;
    const weights = this.weights;
    const leak = net.hiddenLeak;
    let cursor = 0;
    if (layout.feedforward) {
      for (let unit = 0; unit < layout.feedforward; unit++) {
        let sum = weights[cursor++];
        for (let i = 0; i < layout.inputCount; i++) sum += weights[cursor++] * input[i];
        this.compressed[unit] = Math.tanh(sum);
      }
    }
    this.feedbackFrom.set(this.action);
    for (let unit = 0; unit < layout.feedback; unit++) {
      let sum = weights[cursor++];
      for (let i = 0; i < net.jointCount; i++) sum += weights[cursor++] * this.feedbackFrom[i];
      this.feedback[unit] = Math.tanh(sum);
    }
    const incoming = layout.feedforward ? this.compressed : input;
    const incomingCount = layout.feedforward || layout.inputCount;
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
      this.action[joint] = Math.tanh(sum);
    }
    if (cursor !== weights.length) return;
    const previousHidden = this.hidden;
    this.hidden = this.nextHidden;
    this.nextHidden = previousHidden;
  }
}

// 1 si esa articulación de salida se entrena. Cuello y muñecas quedan en 0. Se arma una vez.
const limbJoint = new Uint8Array(net.jointCount);
for (let jointIndex = 0; jointIndex < search.trainableJointIndexes.length; jointIndex++) {
  const joint = search.trainableJointIndexes[jointIndex];
  if (joint >= 0 && joint < limbJoint.length) limbJoint[joint] = 1;
}

// Si está puesto, el ruido y Adam solo tocan esas salidas. El resto del vector queda en el checkpoint.
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

// Cuánto ruido le toca a cada peso. El cuerpo casi no se mueve; los miembros y los targets, sí.
// El ruido grande de los targets y del contacto de la cabeza solo cae en los pesos que multiplican esa entrada.
// `boost` agranda ese ruido cuando el puntaje no mejora. sigmaCap lo corta. Cuello y muñecas siguen en 0.
function explorationOf(index, scale, boost) {
  const extra = boost > 0 ? boost : 1;
  let sigma = search.bodyExplorationScale * extra;
  if (outputJointMask && index < layout.outputBase) return 0;
  if (index < layout.outputBase) {
    const inFeedback = index >= layout.feedbackBase && index < layout.feedbackBase + layout.feedbackBlock;
    const readsInput = !inFeedback && (layout.feedforward ? index < layout.feedforwardBlock : index >= layout.recurrentBase);
    if (readsInput) {
      const stride = layout.feedforward ? layout.feedforwardStride : layout.recurrentStride;
      const origin = layout.feedforward ? 0 : layout.recurrentBase;
      const local = (index - origin) % stride;
      const targetLocal = net.targetInputIndex + 1;
      if (local >= targetLocal && local < targetLocal + net.targetInputCount) sigma = scale * extra;
      else if (local === net.headContactInputIndex + 1) sigma = scale * extra;
      else if (local > net.sensedInputCount && local <= net.inputCount) sigma = net.freshInputExplorationScale * extra;
    }
  } else {
    const joint = Math.floor((index - layout.outputBase) / (layout.recurrent + 1));
    const allowed = outputJointMask ? outputJointMask[joint] : limbJoint[joint];
    sigma = allowed ? search.limbExplorationScale * extra : 0;
  }
  const cap = search.sigmaCap;
  return sigma > 0 && cap > 0 ? Math.min(cap, sigma) : sigma;
}

// x1 por debajo del umbral, luego stagnationBoost, y otra vez cada tantas generaciones, hasta el tope.
function explorationBoost(flatGens) {
  const every = search.stagnationGenerations;
  if (!(every > 0) || !(flatGens >= every)) return 1;
  const steps = Math.floor(flatGens / every);
  return Math.min(search.stagnationBoostCap, Math.pow(search.stagnationBoost, steps));
}

// 0 por debajo del umbral. Después, una fracción fija de los pares. El resto se queda con el ruido chico.
function mutationFraction(flatGens) {
  const every = search.stagnationGenerations;
  const cap = search.stagnationMutateCap;
  if (!(every > 0) || !(cap > 0) || !(flatGens >= every)) return 0;
  return Math.min(1, cap);
}

// Búsqueda de un padre. Cada generación son pares antitéticos: el mismo ruido, con signo más y con signo menos.
// El gradiente sale del rango del fitness, no de una derivada. Adam lo aplica sobre el vector de pesos.
export class EvolutionSearch {
  constructor(randomNormal) {
    const parent = new Float32Array(ElmanNetwork.weightCount);
    const outputBase = ElmanNetwork.layout.outputBase;
    for (let i = 0; i < outputBase; i++) parent[i] = randomNormal() * search.hiddenInitScale;
    for (let i = outputBase; i < ElmanNetwork.weightCount; i++) parent[i] = randomNormal() * search.outputInitScale;
    this.theta = parent;
    this.gen = 0;
    this.hist = [];
    this.k = 0;
    // Primer y segundo momento de Adam.
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
      sigmaQuiet: this.sigmaQuiet,
      sigmaHot: this.sigmaHot,
    };
    const self = this;
    this.compareFitness = (a, b) => self.values[a] - self.values[b];
  }

  // Agranda el pool solo si esta generación tiene más pares que la anterior.
  growFloat(store, count, length) {
    for (let i = store.length; i < count; i++) store.push(new Float32Array(length));
  }

  fillSigma(out, scale, boost) {
    for (let i = 0; i < out.length; i++) out[i] = explorationOf(i, scale, boost);
  }

  // Escribe un par antitético en buffers que ya existen.
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

  // Pares extra copiados de Best ever. Cero hasta que el récord existe y el vector coincide.
  anchorPairCount() {
    const pairs = search.bestEverPairs | 0;
    if (!(pairs > 0)) return 0;
    const score = this.bestEverScore;
    if (!(Number.isFinite(score) && score > -1e8)) return 0;
    if (!this.bestEverTheta || this.bestEverTheta.length !== this.theta.length) return 0;
    return pairs;
  }

  // El factor y la fracción que va a usar la próxima generación. Sirve al retomar una sesión ya estancada.
  refreshMutationBoost() {
    this.mutationBoost = explorationBoost(this.flatGens);
    this.mutationFraction = mutationFraction(this.flatGens);
  }

  // Una situación para toda la generación. Cada individuo decide en un frame distinto.
  // Devuelve el mismo objeto cada vez. Vale hasta el próximo ask.
  ask(situation, count, randomNormal) {
    const parent = this.theta;
    const length = parent.length;
    const scale = Math.max(search.explorationScale, search.explorationFloor * Math.pow(0.5, (this.gen || 0) / search.explorationHalfLifeGenerations));
    const boost = explorationBoost(this.flatGens);
    const fraction = mutationFraction(this.flatGens);
    this.mutationBoost = boost;
    this.mutationFraction = fraction;
    const hot = fraction > 0 ? Math.min(count, Math.max(1, Math.round(count * fraction))) : 0;
    const anchor = this.anchorPairCount();
    const pairTotal = count + anchor;
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
    // Copias leves de Best ever, al final. count sigue siendo solo los pares del padre.
    const anchorParent = this.bestEverTheta;
    for (let individual = 0; individual < anchor; individual++) {
      const pairIndex = count + individual;
      this.pairBoosts[pairIndex] = 1;
      this.writePair(pairIndex, anchorParent, this.sigmaQuiet, randomNormal);
    }
    const generation = this.generation;
    generation.situation = situation;
    generation.count = count;
    generation.scale = scale;
    generation.boost = boost;
    generation.fraction = fraction;
    generation.anchorPairs = anchor;
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
    // Empate total: el rango no dice nada y Adam igual caminaría. No se toca al padre.
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
    // El padre solo se reemplaza si un hijo suyo llegó al marcador. Las copias de Best ever no entran acá.
    const parentBestIndex = order.length ? order[order.length - 1] : -1;
    if (parentBestIndex >= 0 && values[parentBestIndex] > (this.eliteScore ?? -1e9) && (scores[parentBestIndex].gap ?? search.eliteMissingGapMeters) < search.goalCloseEnoughMeters) {
      this.theta.set(generation.weights[parentBestIndex]);
      this.eliteScore = values[parentBestIndex];
      this.zeroMoments();
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

  // En el plateau, un hijo de ruido chico pasa a ser el padre si le gana a su gemelo y al promedio.
  // Una copia de Best ever entra aunque empate con su gemelo. Un scout no entra. El récord no se toca.
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
    this.zeroMoments();
    const source = bestIndex >= thetaChildren ? "best ever child " : "child ";
    this.anchorNote = source + bestValue.toFixed(2) + " became the parent";
  }

  // El récord es el promedio de finales de las últimas generaciones, y el peso guardado es el padre.
  // Un hijo con suerte, que ya arrancó parado, no alcanza.
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

// Red chica de las piernas. El prior de equilibrio no se toca. Unos cientos de pesos, no miles.
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

// La misma búsqueda antitética y Adam, solo sobre la red chica. Sigma uniforme.
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
