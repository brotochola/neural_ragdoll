import * as B from 'https://cdn.jsdelivr.net/npm/phaser-box2d@1.1.0/dist/PhaserBox2D.js';
import { createSim } from './core.js';

const S = createSim(B);
const R = S.build();
postMessage('ready');
onmessage = e => {
  const out = [];
  for (const t of e.data.tasks) {
    const w = t.w instanceof Float32Array ? t.w : Float32Array.from(t.w);
    const ep = new S.Ep(R, w, t.o.tilt, t.o.drop, t.o);
    while (!ep.step());
    const s = ep.stats;
    ep.cleanup();
    out.push({ i: t.i, f: s.f, d: s.d });
  }
  postMessage(out);
};
