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

export class ElmanNetwork {
  // Una capa: la primera es la Elman, con memoria. Dos capas: la primera comprime sin memoria y la segunda es la única que recuerda.
  static get layout() {
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
    this.weights = weights;
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
    if (cursor !== weights.length) throw new Error("weight cursor " + cursor);
    const previousHidden = this.hidden;
    this.hidden = this.nextHidden;
    this.nextHidden = previousHidden;
  }
}

// Cuánto ruido le toca a cada peso. El cuerpo casi no se mueve; los miembros y el goal, sí.
// El ruido grande del goal y de standing solo cae en los pesos que multiplican la entrada.
// `boost` agranda ese ruido cuando el puntaje no mejora. sigmaCap lo corta. Cuello y muñecas siguen en 0.
function explorationOf(index, scale, boost) {
  const extra = boost > 0 ? boost : 1;
  const layout = ElmanNetwork.layout;
  let sigma = search.bodyExplorationScale * extra;
  if (index < layout.outputBase) {
    const inFeedback = index >= layout.feedbackBase && index < layout.feedbackBase + layout.feedbackBlock;
    const readsInput = !inFeedback && (layout.feedforward ? index < layout.feedforwardBlock : index >= layout.recurrentBase);
    if (readsInput) {
      const stride = layout.feedforward ? layout.feedforwardStride : layout.recurrentStride;
      const origin = layout.feedforward ? 0 : layout.recurrentBase;
      const local = (index - origin) % stride;
      if (local === net.goalInputIndex + 1 || local === net.goalInputIndex + 2 || local === net.standingInputIndex + 1) sigma = scale * extra;
      else if (local > net.sensedInputCount && local <= net.inputCount) sigma = net.freshInputExplorationScale * extra;
    }
  } else {
    const joint = Math.floor((index - layout.outputBase) / (layout.recurrent + 1));
    sigma = search.trainableJointIndexes.includes(joint) ? search.limbExplorationScale * extra : 0;
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
  }

  // El factor y la fracción que va a usar la próxima generación. Sirve al retomar una sesión ya estancada.
  refreshMutationBoost() {
    this.mutationBoost = explorationBoost(this.flatGens);
    this.mutationFraction = mutationFraction(this.flatGens);
  }

  // Una situación para toda la generación. Cada individuo decide en un frame distinto.
  ask(situation, count, randomNormal) {
    const parent = this.theta;
    const length = parent.length;
    const scale = Math.max(search.explorationScale, search.explorationFloor * Math.pow(0.5, (this.gen || 0) / search.explorationHalfLifeGenerations));
    const boost = explorationBoost(this.flatGens);
    const fraction = mutationFraction(this.flatGens);
    this.mutationBoost = boost;
    this.mutationFraction = fraction;
    const hot = fraction > 0 ? Math.min(count, Math.max(1, Math.round(count * fraction))) : 0;
    const noise = [];
    const weights = [];
    const phases = [];
    const pairBoosts = [];
    for (let individual = 0; individual < count; individual++) {
      const pairBoost = individual < hot ? boost : 1;
      pairBoosts.push(pairBoost);
      const direction = new Float32Array(length);
      for (let i = 0; i < length; i++) direction[i] = randomNormal();
      noise.push(direction);
      for (const sign of [1, -1]) {
        const child = new Float32Array(length);
        for (let i = 0; i < length; i++) child[i] = parent[i] + sign * explorationOf(i, scale, pairBoost) * direction[i];
        weights.push(child);
        phases.push(weights.length - 1);
      }
    }
    return { situation, noise, weights, phases, count, scale, boost, fraction, pairBoosts };
  }

  tell(generation, scores) {
    const count = generation.count;
    const values = scores.map((score) => score.fitness);
    const length = this.theta.length;
    if (!this.m || this.m.length !== length) this.m = new Float32Array(length);
    if (!this.v || this.v.length !== length) this.v = new Float32Array(length);
    this.k = this.k || 0;
    const order = values.map((_, index) => index).sort((a, b) => values[a] - values[b]);
    const rank = new Array(values.length);
    order.forEach((index, place) => { rank[index] = place / (values.length - 1) - 0.5; });
    const gradient = new Float32Array(length);
    const scale = generation.scale || search.explorationScale;
    const boost = generation.boost || 1;
    const pairBoosts = generation.pairBoosts;
    for (let pair = 0; pair < count; pair++) {
      const pairBoost = pairBoosts && pairBoosts[pair] > 0 ? pairBoosts[pair] : boost;
      const difference = rank[2 * pair] - rank[2 * pair + 1];
      for (let i = 0; i < length; i++) {
        const sigma = explorationOf(i, scale, pairBoost);
        if (!sigma) continue;
        gradient[i] += difference * generation.noise[pair][i] / sigma;
      }
    }
    const stepIndex = ++this.k;
    const firstMoment = search.adamBeta1;
    const secondMoment = search.adamBeta2;
    const firstFix = 1 - Math.pow(firstMoment, stepIndex);
    const secondFix = 1 - Math.pow(secondMoment, stepIndex);
    const stepSize = Math.max(search.learningRateFloor, search.learningRate * Math.pow(0.5, (this.gen || 0) / search.learningRateHalfLifeGenerations));
    for (let i = 0; i < length; i++) {
      if (!explorationOf(i, scale, 1)) continue;
      const estimate = gradient[i] / count;
      this.m[i] = firstMoment * this.m[i] + (1 - firstMoment) * estimate;
      this.v[i] = secondMoment * this.v[i] + (1 - secondMoment) * estimate * estimate;
      this.theta[i] += (stepSize * (this.m[i] / firstFix)) / (Math.sqrt(this.v[i] / secondFix) + search.adamEpsilon);
    }
    const bestIndex = order[order.length - 1];
    const bestWeights = generation.weights[bestIndex];
    this.bestGenTheta = Float32Array.from(bestWeights);
    this.bestGenScore = values[bestIndex];
    // El padre solo se reemplaza si alguien llegó al marcador. Un puntaje alto por quedarse parado no alcanza.
    if (values[bestIndex] > (this.eliteScore ?? -1e9) && (scores[bestIndex].gap ?? search.eliteMissingGapMeters) < search.goalCloseEnoughMeters) {
      this.theta.set(bestWeights);
      this.eliteScore = values[bestIndex];
      this.m = new Float32Array(length);
      this.v = new Float32Array(length);
      this.k = 0;
    }
    this.gen++;
    const stood = scores.filter((score) => score.stoodUp && (score.headImpact || 0) < search.headImpactQuiet).length / scores.length;
    let startSum = 0;
    let startCount = 0;
    let endSum = 0;
    let endCount = 0;
    for (const score of scores) {
      if (Number.isFinite(score.start)) {
        startSum += score.start;
        startCount++;
      }
      if (Number.isFinite(score.end)) {
        endSum += score.end;
        endCount++;
      }
    }
    this.hist.push({
      mean: values.reduce((sum, value) => sum + value, 0) / values.length,
      best: Math.max(...values),
      start: startCount ? startSum / startCount : null,
      end: endCount ? endSum / endCount : null,
      stood,
      gap: scores.reduce((sum, score) => sum + (score.gap || 0), 0) / scores.length,
      boost,
      fraction: generation.fraction || 0,
    });
    if (this.hist.length > 400) this.hist = this.hist.slice(-400);
    this.noteSmoothRecord();
    this.mutationBoost = explorationBoost(this.flatGens);
    this.mutationFraction = mutationFraction(this.flatGens);
  }

  // El récord es el promedio de finales de las últimas generaciones, y el peso guardado es el padre.
  // Un hijo con suerte, que ya arrancó parado, no alcanza.
  noteSmoothRecord() {
    const window = search.recordWindow;
    if (!(window > 0) || this.hist.length < window) return;
    const tail = this.hist.slice(-window);
    if (tail.some((row) => !Number.isFinite(row.start))) return;
    let sum = 0;
    for (const row of tail) sum += row.mean;
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
    }
  }
}
