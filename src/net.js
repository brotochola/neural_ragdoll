// Red de Elman y la búsqueda de un solo padre.
// El estado oculto guarda una fracción del paso anterior. La acción es la tanh de una capa lineal.
// Los pesos, el estado oculto y la acción viven en Float32Array. No hay tensores.
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
    const feedforwardStride = 1 + inputCount;
    const recurrentStride = 1 + (feedforward || inputCount) + recurrent;
    const feedforwardBlock = feedforward * feedforwardStride;
    const recurrentBlock = recurrent * recurrentStride;
    return {
      inputCount,
      feedforward,
      recurrent,
      feedforwardStride,
      recurrentStride,
      feedforwardBlock,
      outputBase: feedforwardBlock + recurrentBlock,
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
    this.action = new Float32Array(net.jointCount);
  }

  // Un paso adelante. Escribe la acción nueva y deja `hidden` en el estado de este paso.
  // Con una capa, cada unidad oculta es leak * anterior + (1 - leak) * tanh(sesgo + pesos · entrada + pesos · oculto).
  // Con dos, la primera es tanh(sesgo + pesos · entrada), sin memoria. La segunda es la Elman, y mira a la primera y a sí misma.
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
    const incoming = layout.feedforward ? this.compressed : input;
    const incomingCount = layout.feedforward || layout.inputCount;
    for (let unit = 0; unit < layout.recurrent; unit++) {
      let sum = weights[cursor++];
      for (let i = 0; i < incomingCount; i++) sum += weights[cursor++] * incoming[i];
      for (let i = 0; i < layout.recurrent; i++) sum += weights[cursor++] * this.hidden[i];
      this.nextHidden[unit] = leak * this.hidden[unit] + (1 - leak) * Math.tanh(sum);
    }
    for (let joint = 0; joint < net.jointCount; joint++) {
      let sum = weights[cursor++];
      for (let unit = 0; unit < layout.recurrent; unit++) sum += weights[cursor++] * this.nextHidden[unit];
      this.action[joint] = Math.tanh(sum);
    }
    const previousHidden = this.hidden;
    this.hidden = this.nextHidden;
    this.nextHidden = previousHidden;
  }
}

// Cuánto ruido le toca a cada peso. El cuerpo casi no se mueve; los miembros y el goal, sí.
// El ruido grande del goal y de standing solo cae en los pesos que multiplican la entrada.
function explorationOf(index, scale) {
  const layout = ElmanNetwork.layout;
  if (index < layout.outputBase) {
    const readsInput = !layout.feedforward || index < layout.feedforwardBlock;
    if (readsInput) {
      const stride = layout.feedforward ? layout.feedforwardStride : layout.recurrentStride;
      const local = index % stride;
      if (local === net.goalInputIndex + 1 || local === net.goalInputIndex + 2 || local === net.standingInputIndex + 1) return scale;
      if (local > net.sensedInputCount && local <= net.inputCount) return net.freshInputExplorationScale;
    }
    return search.bodyExplorationScale;
  }
  const joint = Math.floor((index - layout.outputBase) / (layout.recurrent + 1));
  return search.trainableJointIndexes.includes(joint) ? search.limbExplorationScale : 0;
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
    this.eliteScore = -1e9;
  }

  // Una situación para toda la generación. Cada individuo decide en un frame distinto.
  ask(situation, count, randomNormal) {
    const parent = this.theta;
    const length = parent.length;
    const scale = Math.max(search.explorationScale, search.explorationFloor * Math.pow(0.5, (this.gen || 0) / search.explorationHalfLifeGenerations));
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
    for (let pair = 0; pair < count; pair++) {
      const difference = rank[2 * pair] - rank[2 * pair + 1];
      for (let i = 0; i < length; i++) if (explorationOf(i, scale)) gradient[i] += difference * generation.noise[pair][i];
    }
    const stepIndex = ++this.k;
    const firstMoment = search.adamBeta1;
    const secondMoment = search.adamBeta2;
    const firstFix = 1 - Math.pow(firstMoment, stepIndex);
    const secondFix = 1 - Math.pow(secondMoment, stepIndex);
    const stepSize = Math.max(search.learningRateFloor, search.learningRate * Math.pow(0.5, (this.gen || 0) / search.learningRateHalfLifeGenerations));
    for (let i = 0; i < length; i++) {
      const coordinateScale = explorationOf(i, scale);
      if (!coordinateScale) continue;
      const estimate = gradient[i] / (count * coordinateScale);
      this.m[i] = firstMoment * this.m[i] + (1 - firstMoment) * estimate;
      this.v[i] = secondMoment * this.v[i] + (1 - secondMoment) * estimate * estimate;
      this.theta[i] += (stepSize * (this.m[i] / firstFix)) / (Math.sqrt(this.v[i] / secondFix) + search.adamEpsilon);
    }
    const bestIndex = order[order.length - 1];
    const bestWeights = generation.weights[bestIndex];
    this.bestGenTheta = Float32Array.from(bestWeights);
    this.bestGenScore = values[bestIndex];
    if (values[bestIndex] > (this.bestEverScore ?? -1e9)) {
      this.bestEverTheta = Float32Array.from(bestWeights);
      this.bestEverScore = values[bestIndex];
    }
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
    this.hist.push({
      mean: values.reduce((sum, value) => sum + value, 0) / values.length,
      best: Math.max(...values),
      stood,
      gap: scores.reduce((sum, score) => sum + (score.gap || 0), 0) / scores.length,
    });
    if (this.hist.length > 400) this.hist = this.hist.slice(-400);
  }
}
