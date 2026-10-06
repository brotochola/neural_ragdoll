import { parentPort, workerData } from 'worker_threads';
import * as B from './phys.js';
import { createSim } from './core.js';

const S = createSim(B, { seed: (workerData && workerData.seed) || 1 });
const R = S.build();

parentPort.postMessage({ type: 'ready' });
parentPort.on('message', msg => {
  if (msg.type === 'eval') {
    const out = [];
    for (const t of msg.tasks) {
      const w = t.w instanceof Float32Array ? t.w : Float32Array.from(t.w);
      const ep = new S.Ep(R, w, t.o.tilt, t.o.drop, t.o);
      while (!ep.step());
      const s = ep.stats;
      ep.cleanup();
      out.push({ i: t.i, ...s });
    }
    parentPort.postMessage({ type: 'done', id: msg.id, out });
  }
});
