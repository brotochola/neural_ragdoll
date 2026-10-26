// Backpropagation through time for the ElmanNetwork in net.js, for supervised training (behaviour
// cloning). The forward pass here is the same arithmetic as ElmanNetwork.forward, step by step,
// keeping what the backward pass needs. Supports the layouts this project uses: feed-forward tanh
// layers, no feedback units, one leaky recurrent layer, tanh outputs reading the recurrent layer,
// the (off) target skip and the walk skip.
import { config } from "../config.js";
import { ElmanNetwork, outputCountOf } from "./net.js";

const L = ElmanNetwork.layout;
const leak = config.net.hiddenLeak;
const OUT = outputCountOf();
const R = L.recurrent;
if (L.feedback) throw new Error("bptt.js: feedback units are not supported");
if (L.skip) throw new Error("bptt.js: the target skip is not supported");

// Runs one sequence. inputs: array of Float32Array(inputCount). Returns the cache for backward.
export function forwardSequence(weights, inputs) {
  const steps = inputs.length;
  const ff = L.ff;
  const acts = ff.map((layer) => new Float64Array(steps * layer.units));
  const g = new Float64Array(steps * R); // tanh of the recurrent pre-activation
  const h = new Float64Array((steps + 1) * R); // h[0] is the zero start state
  const out = new Float64Array(steps * OUT);
  for (let t = 0; t < steps; t++) {
    const x = inputs[t];
    let incoming = x;
    let inOff = 0;
    for (let l = 0; l < ff.length; l++) {
      const layer = ff[l];
      const a = acts[l];
      for (let u = 0; u < layer.units; u++) {
        let c = layer.base + u * layer.stride;
        let sum = weights[c++];
        for (let i = 0; i < layer.in; i++) sum += weights[c++] * incoming[inOff + i];
        a[t * layer.units + u] = Math.tanh(sum);
      }
      incoming = a;
      inOff = t * layer.units;
    }
    const last = acts[ff.length - 1];
    const lastOff = t * L.recurrentIn;
    for (let u = 0; u < R; u++) {
      let c = L.recurrentBase + u * L.recurrentStride;
      let sum = weights[c++];
      for (let i = 0; i < L.recurrentIn; i++) sum += weights[c++] * last[lastOff + i];
      for (let i = 0; i < R; i++) sum += weights[c++] * h[t * R + i];
      const gu = Math.tanh(sum);
      g[t * R + u] = gu;
      h[(t + 1) * R + u] = leak * h[t * R + u] + (1 - leak) * gu;
    }
    for (let j = 0; j < OUT; j++) {
      let c = L.outputBase + j * L.outputStride;
      let sum = weights[c++];
      for (let u = 0; u < R; u++) sum += weights[c++] * h[(t + 1) * R + u];
      for (let k = 0; k < L.walkSkip; k++) sum += weights[c++] * x[L.walkFrom + k];
      out[t * OUT + j] = Math.tanh(sum);
    }
  }
  return { steps, inputs, acts, g, h, out };
}

// Squared-error loss 0.5 * sum c_j (out - target)^2 over the sequence, gradient added into grad.
// targets: array of arrays (OUT values, or fewer: missing outputs are not trained).
// costs: per-output weight. Returns the loss.
export function backwardSequence(weights, cache, targets, costs, grad) {
  const { steps, inputs, acts, g, h, out } = cache;
  const ff = L.ff;
  let loss = 0;
  const dh = new Float64Array(R); // gradient on h_t flowing in from step t+1
  const dzRec = new Float64Array(R);
  const dLast = new Float64Array(L.recurrentIn);
  const dOut = new Float64Array(OUT);
  const layerGrads = ff.map((layer) => new Float64Array(layer.units));
  for (let t = steps - 1; t >= 0; t--) {
    const x = inputs[t];
    const target = targets[t];
    // outputs
    for (let j = 0; j < OUT; j++) {
      const y = target[j];
      const o = out[t * OUT + j];
      if (y == null || !costs[j]) {
        dOut[j] = 0;
        continue;
      }
      const e = o - y;
      loss += 0.5 * costs[j] * e * e;
      dOut[j] = costs[j] * e * (1 - o * o);
    }
    const hNow = (t + 1) * R;
    for (let j = 0; j < OUT; j++) {
      const d = dOut[j];
      if (!d) continue;
      let c = L.outputBase + j * L.outputStride;
      grad[c++] += d;
      for (let u = 0; u < R; u++) {
        grad[c] += d * h[hNow + u];
        dh[u] += d * weights[c];
        c++;
      }
      for (let k = 0; k < L.walkSkip; k++) grad[c++] += d * x[L.walkFrom + k];
    }
    // recurrent layer: h_t = leak h_{t-1} + (1 - leak) tanh(z_t)
    for (let u = 0; u < R; u++) {
      const gu = g[t * R + u];
      dzRec[u] = dh[u] * (1 - leak) * (1 - gu * gu);
    }
    const last = acts[ff.length - 1];
    const lastOff = t * L.recurrentIn;
    dLast.fill(0);
    const prev = new Float64Array(R);
    for (let u = 0; u < R; u++) prev[u] = leak * dh[u]; // through the leak
    for (let u = 0; u < R; u++) {
      const d = dzRec[u];
      if (!d) continue;
      let c = L.recurrentBase + u * L.recurrentStride;
      grad[c++] += d;
      for (let i = 0; i < L.recurrentIn; i++) {
        grad[c] += d * last[lastOff + i];
        dLast[i] += d * weights[c];
        c++;
      }
      for (let i = 0; i < R; i++) {
        grad[c] += d * h[t * R + i];
        prev[i] += d * weights[c];
        c++;
      }
    }
    dh.set(prev);
    // feed-forward layers, last to first
    let dAct = dLast;
    for (let l = ff.length - 1; l >= 0; l--) {
      const layer = ff[l];
      const a = acts[l];
      const off = t * layer.units;
      const below = l > 0 ? acts[l - 1] : x;
      const belowOff = l > 0 ? t * ff[l - 1].units : 0;
      const dBelow = l > 0 ? layerGrads[l - 1] : null;
      if (dBelow) dBelow.fill(0);
      for (let u = 0; u < layer.units; u++) {
        const au = a[off + u];
        const d = dAct[u] * (1 - au * au);
        if (!d) continue;
        let c = layer.base + u * layer.stride;
        grad[c++] += d;
        for (let i = 0; i < layer.in; i++) {
          grad[c] += d * below[belowOff + i];
          if (dBelow) dBelow[i] += d * weights[c];
          c++;
        }
      }
      dAct = dBelow;
    }
  }
  return loss;
}

// Finite-difference check on a few weights. Returns the worst relative error.
export function gradientCheck(weights, inputs, targets, costs, picks) {
  const grad = new Float64Array(weights.length);
  backwardSequence(weights, forwardSequence(weights, inputs), targets, costs, grad);
  const lossOf = (w) => {
    const g = new Float64Array(w.length);
    return backwardSequence(w, forwardSequence(w, inputs), targets, costs, g);
  };
  let worst = 0;
  for (const i of picks) {
    const w = Float64Array.from(weights);
    const eps = 1e-5;
    w[i] += eps;
    const up = lossOf(w);
    w[i] -= 2 * eps;
    const down = lossOf(w);
    const num = (up - down) / (2 * eps);
    const rel = Math.abs(num - grad[i]) / Math.max(1e-8, Math.abs(num) + Math.abs(grad[i]));
    worst = Math.max(worst, rel);
  }
  return worst;
}
