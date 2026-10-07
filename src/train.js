import { Worker } from 'worker_threads';
import { cpus } from 'os';
import { fileURLToPath } from 'url';
import path from 'path';
import fs from 'fs';
import * as B from './phys.js';
import { createSim, NW, H, NI, O, MAXTILT, rngMk, netsFromCkpt, padWeights } from './core.js';

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
const maxGens = maxGenArg && +maxGenArg > 0 ? +maxGenArg : Infinity;
const K = Math.max(1, Math.min(cpus().length - 2, +(process.env.WORKERS || (cpus().length - 2))));
const NP = 24;

const S = createSim(B, { seed: 7 });
S.setRng(Date.now() & 0xffff);

function log(s) {
  const line = `[${new Date().toISOString()}] ${s}`;
  console.log(line);
  fs.appendFileSync(LOG, line + '\n');
}

const arr5 = a => a ? Array.from(a, x => +x.toFixed(5)) : null;
const arr6 = a => a && a.length === NW ? Array.from(a, x => +x.toFixed(6)) : null;

function pack(st, extra = {}) {
  const thetaStand = st.thetaStand || st.theta;
  const bestStand = st.bestStand || st.bestTheta;
  return {
    H, NI, O, NW, nets: 2,
    gen: st.gen, A: st.A, Pm: st.Pm, Q: st.Q, Pa: st.Pa, G: st.G || 0,
    sig: st.sig, lr: st.lr,
    kStand: st.kStand || 0, kGetup: st.kGetup || 0,
    hist: (st.hist || []).slice(-80),
    bestStandF: st.bestStandF, bestStandGen: st.bestStandGen,
    bestGetupF: st.bestGetupF, bestGetupGen: st.bestGetupGen,
    bestF: st.bestStandF, bestGen: st.bestStandGen,
    theta: arr5(thetaStand),
    bestTheta: arr5(bestStand),
    thetaStand: arr5(thetaStand),
    thetaGetup: arr5(st.thetaGetup),
    bestStand: arr5(bestStand),
    bestGetup: arr5(st.bestGetup),
    mStand: arr6(st.mStand), vStand: arr6(st.vStand),
    mGetup: arr6(st.mGetup), vGetup: arr6(st.vGetup),
    eval: st.lastEval || null,
    ...extra
  };
}

function save(st) {
  const full = pack(st);
  fs.writeFileSync(CKPT, JSON.stringify(full));
  const w = {
    H, NI, O, NW, nets: 2,
    gen: st.gen, bestGen: st.bestStandGen, bestStandGen: st.bestStandGen, bestGetupGen: st.bestGetupGen,
    eval: st.lastEval || null,
    hist: (st.hist || []).slice(-80),
    theta: full.theta, bestTheta: full.bestTheta,
    thetaStand: full.thetaStand, thetaGetup: full.thetaGetup,
    bestStand: full.bestStand, bestGetup: full.bestGetup
  };
  fs.writeFileSync(WEIGHTS, JSON.stringify(w));
}

function loadCkpt() {
  const c = JSON.parse(fs.readFileSync(CKPT, 'utf8'));
  const n = netsFromCkpt(c);
  const adam = (a) => padWeights(a) || new Float32Array(NW);
  return {
    thetaStand: n.thetaStand, thetaGetup: n.thetaGetup,
    theta: n.thetaStand,
    bestStand: n.bestStand, bestGetup: n.bestGetup,
    bestTheta: n.bestStand,
    gen: c.gen || 0, hist: c.hist || [], A: c.A ?? 0.12, Pm: c.Pm ?? 0, Q: c.Q ?? 0, Pa: c.Pa ?? 0,
    G: c.G ?? 0, sig: c.sig ?? 0.1, lr: c.lr ?? 0.01,
    kStand: n.migrated ? 0 : (c.kStand || 0),
    kGetup: n.migrated || n.padded ? 0 : (c.kGetup || 0),
    mStand: n.migrated ? new Float32Array(NW) : adam(c.mStand),
    vStand: n.migrated ? new Float32Array(NW) : adam(c.vStand),
    mGetup: n.migrated || n.padded ? new Float32Array(NW) : adam(c.mGetup),
    vGetup: n.migrated || n.padded ? new Float32Array(NW) : adam(c.vGetup),
    bestStandF: c.bestStandF ?? (n.migrated ? c.bestF : undefined),
    bestGetupF: c.bestGetupF,
    bestStandGen: c.bestStandGen ?? (n.migrated ? c.bestGen : undefined),
    bestGetupGen: c.bestGetupGen,
    bestF: c.bestStandF ?? c.bestF, bestGen: c.bestStandGen ?? c.bestGen,
    lastEval: c.eval || null,
    migrated: n.migrated, padded: n.padded
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
    const p = i < 3 ? S.supinePose(er) : S.pronePose(er);
    getup.push({ tilt: p.tilt, drop: p.drop, push: 0, amp: 0, T: 8, ph: 0, proj: [], task: 'getup', steps: 960 });
    recover.push({ tilt: (er() * 2 - 1) * 0.08, drop: 0.12, push: (er() < 0.5 ? 1 : -1) * 6.2, amp: 0, T: 8, ph: 0, proj: [], task: 'recover', steps: 780 });
  }
  return { stand, plat, proj, getup, recover };
}

function summarize(label, rows) {
  const n = rows.length || 1;
  const avg = k => rows.reduce((a, b) => a + (b[k] || 0), 0) / n;
  return {
    f: avg('f'),
    rec: rows.filter(r => r.rec).length / n,
    up: rows.filter(r => r.up).length / n,
    hy: avg('hy'),
    stood: avg('stood'),
    ang: avg('ang'),
    hand: avg('hand'),
    label
  };
}

function composite(ev) {
  return 0.22 * ev.stand.stood + 0.16 * ev.plat.stood + 0.16 * ev.proj.stood
    + 0.16 * ev.getup.rec + 0.16 * ev.recover.rec + 0.06 * ev.getup.up + 0.08 * ev.recover.up;
}
function standScore(ev) {
  return 0.5 * ev.stand.stood + 0.25 * ev.plat.stood + 0.25 * ev.proj.stood;
}
function getupScore(ev) {
  return ev.getup.rec;
}

const EV = evalSuite();

async function runEval(pool, nets) {
  const standW = nets.thetaStand || nets.theta;
  const getupW = nets.thetaGetup || nets.theta;
  const packE = (arr, w) => arr.map(o => ({ w, o }));
  const stand = await pool.evalItems(packE(EV.stand, standW));
  const plat = await pool.evalItems(packE(EV.plat, standW));
  const proj = await pool.evalItems(packE(EV.proj, standW));
  const getup = await pool.evalItems(packE(EV.getup, getupW));
  const recover = await pool.evalItems(packE(EV.recover, standW));
  const ev = {
    stand: summarize('stand', stand),
    plat: summarize('plat', plat),
    proj: summarize('proj', proj),
    getup: summarize('getup', getup),
    recover: summarize('recover', recover)
  };
  ev.score = composite(ev);
  ev.standScore = standScore(ev);
  ev.getupScore = getupScore(ev);
  return ev;
}

function fmtEv(ev) {
  const p = (x) => (100 * x).toFixed(0) + '%';
  const g = ev.getup;
  return `score ${ev.score.toFixed(3)}  stand ${p(ev.stand.stood)} hy ${ev.stand.hy.toFixed(2)}  plat ${p(ev.plat.stood)}  proj ${p(ev.proj.stood)}  getup ${p(g.rec)} fit ${g.f.toFixed(2)} |a| ${g.ang.toFixed(0)}° hand ${p(g.hand)} hy ${g.hy.toFixed(2)}  recov ${p(ev.recover.rec)}`;
}

const pool = makePool();
let st = resume ? loadCkpt() : S.freshState();
if (resume && st.A >= 0.35) {
  st.Pa = Math.max(st.Pa || 0, 0.85);
}
if (resume && st.bestStand && (st.lastEval?.stand?.stood ?? 1) < 0.75) {
  st.thetaStand = st.bestStand.slice();
  st.theta = st.thetaStand;
}
if (resume && st.padded) {
  st.thetaGetup = S.initTheta();
  st.bestGetup = null;
  st.bestGetupF = undefined;
  st.bestGetupGen = undefined;
  st.mGetup = new Float32Array(NW);
  st.vGetup = new Float32Array(NW);
  st.kGetup = 0;
  st.G = 0;
}
let stopping = false;
process.on('SIGINT', () => { stopping = true; log('SIGINT: saving after this gen'); });
process.on('SIGTERM', () => { stopping = true; });

await pool.ready();
log(`pool ${K} workers  NP ${NP}  NW ${NW}  max ${maxGens}  ${resume ? 'resume gen ' + st.gen : 'fresh'}${st.migrated ? '  migrate two-net from bestTheta' : ''}${st.padded ? '  pad NI ' + NI + '  getup reset' : ''}`);
if (!resume) save(st);

if (evalOnly) {
  const ev = await runEval(pool, {
    thetaStand: st.bestStand || st.thetaStand,
    thetaGetup: st.bestGetup || st.thetaGetup
  });
  log('eval ' + fmtEv(ev));
  pool.kill();
  process.exit(0);
}

const t0 = Date.now();
let bestStandScore = -1, bestGetupScore = -1;

{
  const ev = await runEval(pool, st);
  st.lastEval = ev;
  log('start ' + fmtEv(ev) + (st.bestStandGen != null ? `  bestStand gen ${st.bestStandGen}` : ''));
  if (resume && st.bestStand && st.bestStandF != null) {
    bestStandScore = st.bestStandF;
    if (ev.standScore >= bestStandScore - 0.002) {
      bestStandScore = Math.max(bestStandScore, ev.standScore);
      st.bestStandF = bestStandScore; st.bestStandGen = st.gen; st.bestStand = st.thetaStand.slice();
    }
  } else {
    bestStandScore = ev.standScore;
    st.bestStandF = ev.standScore; st.bestStandGen = st.gen; st.bestStand = st.thetaStand.slice();
  }
  if (resume && st.bestGetup && st.bestGetupF != null) {
    bestGetupScore = st.bestGetupF;
    if (ev.getupScore >= bestGetupScore - 0.002) {
      bestGetupScore = Math.max(bestGetupScore, ev.getupScore);
      st.bestGetupF = bestGetupScore; st.bestGetupGen = st.gen; st.bestGetup = st.thetaGetup.slice();
    }
  } else {
    bestGetupScore = ev.getupScore;
    st.bestGetupF = ev.getupScore; st.bestGetupGen = st.gen; st.bestGetup = st.thetaGetup.slice();
  }
  st.bestF = st.bestStandF; st.bestGen = st.bestStandGen; st.bestTheta = st.bestStand;
  st.theta = st.thetaStand;
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
    const g = pop.o.task === 'getup' ? `  |a| ${(last.ang || 0).toFixed(0)}° hand ${(100 * (last.hand || 0)).toFixed(0)}% hy ${(last.hy || 0).toFixed(2)}` : '';
    log(`gen ${st.gen}  ${pop.o.task}  fit ${last.m.toFixed(2)}/${last.b.toFixed(2)}  up ${(100 * last.s).toFixed(0)}%${g}  A ${st.A.toFixed(2)} Q ${st.Q.toFixed(2)} Pa ${st.Pa.toFixed(2)} G ${(st.G || 0).toFixed(2)}  ${dt}s`);
    save(st);
  }
  if (st.gen % 15 === 0) {
    const ev = await runEval(pool, st);
    st.lastEval = ev;
    log('eval ' + fmtEv(ev));
    if (ev.standScore >= bestStandScore - 0.002) {
      bestStandScore = Math.max(bestStandScore, ev.standScore);
      st.bestStandF = bestStandScore; st.bestStandGen = st.gen; st.bestStand = st.thetaStand.slice();
    }
    if (ev.getupScore >= bestGetupScore - 0.002) {
      bestGetupScore = Math.max(bestGetupScore, ev.getupScore);
      st.bestGetupF = bestGetupScore; st.bestGetupGen = st.gen; st.bestGetup = st.thetaGetup.slice();
    }
    st.bestF = st.bestStandF; st.bestGen = st.bestStandGen; st.bestTheta = st.bestStand;
    st.theta = st.thetaStand;
    st.sig = Math.max(0.055, 0.1 * Math.pow(0.5, st.gen / 5500));
    save(st);
    if (ev.stand.stood >= 0.85 && ev.plat.stood >= 0.7 && ev.proj.stood >= 0.6 && ev.recover.rec >= 0.5 && ev.getup.rec >= 0.3) {
      log('target reached, stopping');
      break;
    }
  }
}

save(st);
log(`done gen ${st.gen} bestStand ${st.bestStandGen} ${bestStandScore.toFixed(3)} bestGetup ${st.bestGetupGen} ${bestGetupScore.toFixed(3)}  ${(Date.now() - t0) / 60000} min`);
pool.kill();
process.exit(0);
