import { Worker } from 'worker_threads';
import { cpus } from 'os';
import { fileURLToPath } from 'url';
import path from 'path';
import fs from 'fs';
import * as B from './phys.js';
import { createSim, H, NI, O, rngMk } from './core.js';

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

const arr = (a, n) => a && a.length === n ? Array.from(a, x => +(+x).toFixed(5)) : null;

function pack(st) {
  const n = st.theta.length;
  return {
    H: st.H || H, NI, O, NW: n, nets: 1,
    gen: st.gen, sig: st.sig, lr: st.lr, k: st.k || 0,
    hist: (st.hist || []).slice(-80),
    bestF: st.bestF, bestGen: st.bestGen, eliteF: st.eliteF || 0,
    theta: arr(st.theta, n), bestTheta: arr(st.bestTheta, n),
    m: arr(st.m, n), v: arr(st.v, n),
    eval: st.lastEval || null
  };
}

function save(st) {
  const full = pack(st);
  fs.writeFileSync(CKPT, JSON.stringify(full));
  fs.writeFileSync(WEIGHTS, JSON.stringify({
    H: full.H, NI, O, NW: full.NW, nets: 1,
    gen: st.gen, bestGen: st.bestGen, bestF: st.bestF,
    eval: st.lastEval || null, hist: full.hist,
    theta: full.theta, bestTheta: full.bestTheta
  }));
}

function loadCkpt() {
  const c = JSON.parse(fs.readFileSync(CKPT, 'utf8'));
  const theta = Float32Array.from(c.theta);
  const n = theta.length;
  const adam = a => (a && a.length === n) ? Float32Array.from(a) : new Float32Array(n);
  return {
    theta, H: c.H || H, gen: c.gen || 0, hist: c.hist || [],
    sig: c.sig ?? 0.12, lr: c.lr ?? 0.01, k: c.k || 0,
    m: adam(c.m), v: adam(c.v),
    bestTheta: c.bestTheta && c.bestTheta.length === n ? Float32Array.from(c.bestTheta) : theta.slice(),
    bestF: c.bestF, bestGen: c.bestGen, eliteF: c.eliteF || 0,
    lastEval: c.eval || null
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
  const supine = [], prone = [], fallen = [], up = [];
  const base = { push: 0, amp: 0, T: 8, ph: 0, proj: [], steps: 960, full: true, cut: false, task: 'live' };
  for (let i = 0; i < 4; i++) {
    const s = S.supinePose(er), p = S.pronePose(er), f = S.fallenPose(er);
    supine.push({ ...base, tilt: s.tilt, drop: s.drop, pose: 'supine', settle: 50 });
    prone.push({ ...base, tilt: p.tilt, drop: p.drop, pose: 'prone', settle: 50 });
    fallen.push({ ...base, tilt: f.tilt, drop: f.drop, pose: 'fallen', settle: 50 });
    up.push({ ...base, tilt: (er() * 2 - 1) * 0.08, drop: 0.12, pose: 'up', settle: 0, push: (er() < 0.5 ? -1 : 1) * 0.3 });
  }
  return { supine, prone, fallen, up };
}

function summarize(rows) {
  const n = rows.length || 1;
  const avg = k => rows.reduce((a, b) => a + (b[k] || 0), 0) / n;
  return {
    f: avg('f'), hy: avg('hy'),
    up: rows.filter(r => r.up && (r.headImpact || 0) < 1.5).length / n,
    head: avg('headImpact')
  };
}

const EV = evalSuite();
async function runEval(pool, theta) {
  const roll = (list) => pool.evalItems(list.map(o => ({ w: theta, o })));
  const ev = {
    supine: summarize(await roll(EV.supine)),
    prone: summarize(await roll(EV.prone)),
    fallen: summarize(await roll(EV.fallen)),
    up: summarize(await roll(EV.up))
  };
  ev.score = (ev.supine.f + ev.prone.f + ev.fallen.f + ev.up.f) / 4;
  return ev;
}

function fmtEv(ev) {
  const p = x => (100 * x).toFixed(0) + '%';
  const bit = (n, s) => `${n} up ${p(s.up)} hy ${s.hy.toFixed(2)}`;
  return `${bit('arriba', ev.supine)}  ${bit('abajo', ev.prone)}  ${bit('costado', ev.fallen)}  ${bit('parado', ev.up)}  fit ${ev.score.toFixed(2)}`;
}

const pool = makePool();
let st = resume ? loadCkpt() : S.freshState();
let stopping = false;
process.on('SIGINT', () => { stopping = true; log('SIGINT: saving after this gen'); });
process.on('SIGTERM', () => { stopping = true; });

await pool.ready();
log(`pool ${K} workers  NP ${NP}  H ${st.H || H}  max ${maxGens}  one net  ${resume ? 'resume gen ' + st.gen : 'fresh'}`);
if (!resume) save(st);

if (evalOnly) {
  const ev = await runEval(pool, st.bestTheta || st.theta);
  log('eval ' + fmtEv(ev));
  pool.kill();
  process.exit(0);
}

const t0 = Date.now();
let bestScore = st.bestF != null ? st.bestF : -1e9;
{
  const ev = await runEval(pool, st.theta);
  st.lastEval = ev;
  if (ev.score > bestScore) {
    bestScore = ev.score; st.bestF = ev.score; st.bestGen = st.gen; st.bestTheta = st.theta.slice();
  }
  log('start ' + fmtEv(ev));
  save(st);
}

while (!stopping && st.gen < maxGens) {
  const tGen = Date.now();
  const pop = S.esAsk(st, NP);
  const fits = await pool.evalItems(pop.ws.map(w => ({ w, o: pop.o })));
  S.esTell(st, pop, fits);
  const last = st.hist[st.hist.length - 1];
  const dt = ((Date.now() - tGen) / 1000).toFixed(1);
  log(`gen ${st.gen}  ${pop.o.pose}  fit ${last.m.toFixed(2)}/${last.b.toFixed(2)}  up ${(100 * last.s).toFixed(0)}%  hy ${(last.hy || 0).toFixed(2)} head ${(last.head || 0).toFixed(2)}  H ${st.H || H}${st.mut ? '  ' + st.mut : ''}  ${dt}s`);
  if (st.gen % 15 === 0) {
    const ev = await runEval(pool, st.theta);
    st.lastEval = ev;
    log('eval ' + fmtEv(ev));
    if (ev.score > bestScore) {
      bestScore = ev.score; st.bestF = ev.score; st.bestGen = st.gen; st.bestTheta = st.theta.slice();
    }
  }
  save(st);
}

save(st);
log(`done gen ${st.gen} best ${st.bestGen} ${(bestScore).toFixed(3)}  ${(Date.now() - t0) / 60000} min`);
pool.kill();
process.exit(0);
