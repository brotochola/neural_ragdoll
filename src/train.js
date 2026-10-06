import { Worker } from 'worker_threads';
import { cpus } from 'os';
import { fileURLToPath } from 'url';
import path from 'path';
import fs from 'fs';
import * as B from './phys.js';
import { createSim, NW, H, NI, O, MAXTILT, rngMk } from './core.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(__dirname, '..');
const CKPT = path.join(ROOT, 'checkpoint.json');
const WEIGHTS = path.join(ROOT, 'weights.json');
const LOG = path.join(ROOT, 'train.log');

const args = process.argv.slice(2);
const flags = new Set(args);
const evalOnly = flags.has('--eval');
const resume = !flags.has('--fresh') && fs.existsSync(CKPT);
const maxGenArg = args.find((a, i) => args[i - 1] === '--gens');
const maxGens = maxGenArg ? +maxGenArg : 8000;
const K = Math.max(1, Math.min(cpus().length - 2, +(process.env.WORKERS || (cpus().length - 2))));
const NP = 24;

const S = createSim(B, { seed: 7 });
S.setRng(Date.now() & 0xffff);

function log(s) {
  const line = `[${new Date().toISOString()}] ${s}`;
  console.log(line);
  fs.appendFileSync(LOG, line + '\n');
}

function pack(st, extra = {}) {
  return {
    H, NI, O, NW,
    gen: st.gen, A: st.A, Pm: st.Pm, Q: st.Q, Pa: st.Pa, G: st.G || 0,
    sig: st.sig, lr: st.lr, k: st.k || 0,
    hist: (st.hist || []).slice(-80),
    bestF: st.bestF, bestGen: st.bestGen,
    theta: Array.from(st.theta, x => +x.toFixed(5)),
    bestTheta: st.bestTheta ? Array.from(st.bestTheta, x => +x.toFixed(5)) : null,
    m: st.m ? Array.from(st.m, x => +x.toFixed(6)) : null,
    v: st.v ? Array.from(st.v, x => +x.toFixed(6)) : null,
    eval: st.lastEval || null,
    ...extra
  };
}

function save(st) {
  const full = pack(st);
  fs.writeFileSync(CKPT, JSON.stringify(full));
  const w = {
    H, NI, O, NW,
    gen: st.gen, bestGen: st.bestGen,
    eval: st.lastEval || null,
    hist: (st.hist || []).slice(-80),
    theta: Array.from(st.theta, x => +x.toFixed(5)),
    bestTheta: st.bestTheta ? Array.from(st.bestTheta, x => +x.toFixed(5)) : null
  };
  fs.writeFileSync(WEIGHTS, JSON.stringify(w));
}

function loadCkpt() {
  const c = JSON.parse(fs.readFileSync(CKPT, 'utf8'));
  if (c.NW !== NW || !c.theta || c.theta.length !== NW) throw new Error('checkpoint NW mismatch');
  return {
    theta: Float32Array.from(c.theta),
    bestTheta: c.bestTheta && c.bestTheta.length === NW ? Float32Array.from(c.bestTheta) : Float32Array.from(c.theta),
    gen: c.gen || 0, hist: c.hist || [], A: c.A ?? 0.12, Pm: c.Pm ?? 0, Q: c.Q ?? 0, Pa: c.Pa ?? 0,
    G: c.G ?? 0, sig: c.sig ?? 0.1, lr: c.lr ?? 0.01, k: c.k || 0,
    m: c.m && c.m.length === NW ? Float32Array.from(c.m) : new Float32Array(NW),
    v: c.v && c.v.length === NW ? Float32Array.from(c.v) : new Float32Array(NW),
    bestF: c.bestF, bestGen: c.bestGen, lastEval: c.eval || null
  };
}

function makePool() {
  const workers = [];
  const wait = new Map();
  let nextId = 1;
  function spawn(i) {
    return new Promise((res, rej) => {
      const w = new Worker(path.join(__dirname, 'worker.js'), { workerData: { seed: 1000 + i } });
      const to = setTimeout(() => rej(new Error('worker timeout')), 25000);
      w.on('message', m => {
        if (m.type === 'ready') { clearTimeout(to); res({ w }); return; }
        if (m.type === 'done') {
          const p = wait.get(m.id);
          if (p) { wait.delete(m.id); p.res(m.out); }
        }
      });
      w.on('error', e => { clearTimeout(to); rej(e); });
    });
  }
  async function ready() {
    workers.push(...await Promise.all(Array.from({ length: K }, (_, i) => spawn(i))));
  }
  function evalItems(items) {
    if (!items.length) return Promise.resolve([]);
    const buckets = Array.from({ length: K }, () => []);
    items.forEach((it, i) => buckets[i % K].push({ i, w: Array.from(it.w), o: it.o }));
    return Promise.all(buckets.map((b, k) => {
      if (!b.length) return [];
      const id = nextId++;
      return new Promise((res, rej) => {
        const to = setTimeout(() => { wait.delete(id); rej(new Error('eval timeout')); }, 180000);
        wait.set(id, { res: out => { clearTimeout(to); res(out); }, rej });
        workers[k].w.postMessage({ type: 'eval', id, tasks: b });
      });
    })).then(rs => {
      const out = new Array(items.length);
      for (const r of rs.flat()) if (r) out[r.i] = r;
      if (out.some(x => !x || !Number.isFinite(x.f))) throw new Error('incomplete eval batch');
      return out;
    });
  }
  function kill() { for (const { w } of workers) w.terminate(); }
  return { ready, evalItems, kill };
}

function evalSuite() {
  const er = rngMk(4242);
  const stand = [], plat = [], proj = [], getup = [], recover = [];
  for (let i = 0; i < 6; i++) {
    stand.push({ tilt: (er() * 2 - 1) * 0.2, drop: 0.14, push: (er() * 2 - 1) * 0.4, amp: 0, T: 8, ph: 0, proj: [], task: 'stand', steps: 480 });
    plat.push({ tilt: (er() * 2 - 1) * 0.15, drop: 0.14, push: 0, amp: MAXTILT * (0.6 + 0.4 * er()), T: 6 + 2 * er(), ph: 6.28 * er(), proj: [], task: 'stand', steps: 540 });
    const o = { tilt: (er() * 2 - 1) * 0.12, drop: 0.14, push: 0, amp: 0, T: 8, ph: 0, proj: [], task: 'stand', steps: 540 };
    S.addProjs(er, o, 0.9);
    proj.push(o);
    const p = S.fallenPose(er, 1);
    getup.push({ tilt: p.tilt, drop: p.drop, push: 0, amp: 0, T: 8, ph: 0, proj: [], task: 'getup', steps: 780 });
    recover.push({ tilt: (er() * 2 - 1) * 0.08, drop: 0.12, push: (er() < 0.5 ? 1 : -1) * 6.2, amp: 0, T: 8, ph: 0, proj: [], task: 'recover', steps: 780 });
  }
  return { stand, plat, proj, getup, recover };
}

function summarize(label, rows) {
  const n = rows.length || 1;
  return {
    f: rows.reduce((a, b) => a + b.f, 0) / n,
    rec: rows.filter(r => r.rec).length / n,
    up: rows.filter(r => r.up).length / n,
    hy: rows.reduce((a, b) => a + b.hy, 0) / n,
    stood: rows.reduce((a, b) => a + b.stood, 0) / n,
    label
  };
}

function composite(ev) {
  return 0.22 * ev.stand.stood + 0.16 * ev.plat.stood + 0.16 * ev.proj.stood
    + 0.16 * ev.getup.rec + 0.16 * ev.recover.rec + 0.06 * ev.getup.up + 0.08 * ev.recover.up;
}

const EV = evalSuite();

async function runEval(pool, theta) {
  const packE = (arr) => arr.map(o => ({ w: theta, o }));
  const stand = await pool.evalItems(packE(EV.stand));
  const plat = await pool.evalItems(packE(EV.plat));
  const proj = await pool.evalItems(packE(EV.proj));
  const getup = await pool.evalItems(packE(EV.getup));
  const recover = await pool.evalItems(packE(EV.recover));
  const ev = {
    stand: summarize('stand', stand),
    plat: summarize('plat', plat),
    proj: summarize('proj', proj),
    getup: summarize('getup', getup),
    recover: summarize('recover', recover)
  };
  ev.score = composite(ev);
  return ev;
}

function fmtEv(ev) {
  const p = (x) => (100 * x).toFixed(0) + '%';
  return `score ${ev.score.toFixed(3)}  stand ${p(ev.stand.stood)} hy ${ev.stand.hy.toFixed(2)}  plat ${p(ev.plat.stood)}  proj ${p(ev.proj.stood)}  getup ${p(ev.getup.rec)} hy ${ev.getup.hy.toFixed(2)}  recov ${p(ev.recover.rec)}`;
}

const pool = makePool();
let st = resume ? loadCkpt() : S.freshState();
if (resume && st.A >= 0.35) {
  st.G = Math.max(st.G || 0, 0.45);
  st.Pa = Math.max(st.Pa || 0, 0.85);
}
let stopping = false;
process.on('SIGINT', () => { stopping = true; log('SIGINT: saving after this gen'); });
process.on('SIGTERM', () => { stopping = true; });

await pool.ready();
log(`pool ${K} workers  NP ${NP}  NW ${NW}  max ${maxGens}  ${resume ? 'resume gen ' + st.gen : 'fresh'}`);
if (!resume) save(st);

if (evalOnly) {
  const ev = await runEval(pool, st.bestTheta || st.theta);
  log('eval ' + fmtEv(ev));
  pool.kill();
  process.exit(0);
}

const t0 = Date.now();
let bestScore = -1;

{
  const ev = await runEval(pool, st.theta);
  st.lastEval = ev;
  log('start ' + fmtEv(ev));
  bestScore = ev.score;
  st.bestF = ev.score; st.bestGen = st.gen; st.bestTheta = st.theta.slice();
  save(st);
}

while (!stopping && st.gen < maxGens) {
  const tGen = Date.now();
  const pop = S.esAsk(st, NP);
  const fits = await pool.evalItems(pop.ws.map(w => ({ w, o: pop.o })));
  S.esTell(st, pop, fits);
  const last = st.hist[st.hist.length - 1];
  const dt = ((Date.now() - tGen) / 1000).toFixed(1);
  if (st.gen % 5 === 0) {
    log(`gen ${st.gen}  ${pop.o.task}  fit ${last.m.toFixed(2)}/${last.b.toFixed(2)}  up ${(100 * last.s).toFixed(0)}%  A ${st.A.toFixed(2)} Q ${st.Q.toFixed(2)} Pa ${st.Pa.toFixed(2)} G ${(st.G || 0).toFixed(2)}  ${dt}s`);
    save(st);
  }
  if (st.gen % 15 === 0) {
    const ev = await runEval(pool, st.theta);
    st.lastEval = ev;
    log('eval ' + fmtEv(ev));
    if (ev.score >= bestScore - 0.002) {
      bestScore = Math.max(bestScore, ev.score);
      st.bestF = bestScore; st.bestGen = st.gen; st.bestTheta = st.theta.slice();
    }
    st.sig = Math.max(0.055, 0.1 * Math.pow(0.5, st.gen / 5500));
    save(st);
    if (ev.stand.stood >= 0.85 && ev.plat.stood >= 0.7 && ev.proj.stood >= 0.6 && ev.recover.rec >= 0.5 && ev.getup.rec >= 0.3) {
      log('target reached, stopping');
      break;
    }
  }
}

save(st);
log(`done gen ${st.gen} best ${st.bestGen} score ${bestScore.toFixed(3)}  ${(Date.now() - t0) / 60000} min`);
pool.kill();
process.exit(0);
