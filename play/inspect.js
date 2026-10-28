// INSPECTOR: the panel under the stage that shows what is going on inside, live.
//   pipe     who does each job right now: what it senses, what it decides to do, where the hands and feet
//            go, the angle of every joint, the motors. Each one tagged by kind: neural network, code with
//            numbers tuned by search, code with fixed rules, you.
//   phases   the steps of the current skill, the current one lit
//   facts    what it is deciding (the next obstacle, the jump, the hold it reaches for, ...)
//   motors   the 14 joints: the target the motor turns to and where the joint really is; on the dash course
//            the runner's target and the network's correction apart
//   net      the network that is driving, its units lit by activation
// app.js builds an `info` object each frame (describe()) and calls update / draw.

export const KINDS = {
  net: { label: 'Red neuronal', cls: 'k-net' },
  tuned: { label: 'Código · números ajustados por búsqueda', cls: 'k-tuned' },
  rules: { label: 'Código · reglas fijas', cls: 'k-rules' },
  you: { label: 'Vos', cls: 'k-you' },
  mix: { label: 'Código + red neuronal', cls: 'k-mix' },
  none: { label: 'Nadie', cls: 'k-none' },
  body: { label: 'Cuerpo', cls: 'k-body' },
};
const STAGES = [
  { id: 'sense', title: 'Qué mira' },
  { id: 'decide', title: 'Qué decide hacer' },
  { id: 'limbs', title: 'Dónde van manos y pies' },
  { id: 'joints', title: 'Ángulo de cada articulación' },
  { id: 'motors', title: 'Qué lo mueve' },
];
export const JOINT_NAMES = ['Cintura', 'Cuello', 'Hombro izq', 'Codo izq', 'Muñeca izq', 'Cadera izq', 'Rodilla izq', 'Tobillo izq', 'Hombro der', 'Codo der', 'Muñeca der', 'Cadera der', 'Rodilla der', 'Tobillo der'];
const JOINT_ORDER = [0, 1, 5, 6, 7, 11, 12, 13, 2, 3, 4, 8, 9, 10];

// every brain in the project, by kind; lit when it is working right now (its name is in the pipeline)
const PIECES = [
  ['Red "pararse"', 'net', 'Mantiene el equilibrio parada'],
  ['Red "levantarse"', 'net', 'Se levanta del piso'],
  ['Red correctora', 'net', 'Corrige al corredor en Cajas y bolas'],
  ['Corredor', 'tuned', 'Corre con fase de vuelo'],
  ['Selector de saltos', 'tuned', 'Cuándo saltar y cómo ajustar el paso'],
  ['Caminata SIMBICON', 'tuned', 'Camina'],
  ['Freno', 'tuned', 'Frena hasta quedar quieto'],
  ['Planificador de agarres', 'tuned', 'Elige salientes para manos y pies'],
  ['Secuencia de la soga', 'tuned', 'Fases de saltar con la soga'],
  ['Secuencia del obstáculo', 'tuned', 'Encadena correr, frenar, trepar y seguir'],
  ['Ejecutor de poses', 'rules', 'Convierte cadera, manos y pies en ángulos'],
];
const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' })[c]);

export function createInspector({ $, css, Box2D, world }) {
  const pipe = $('pipe'), phasesEl = $('phases'), factsEl = $('facts');
  const mc = $('motors'), mctx = mc.getContext('2d');
  const nc = $('netCanvas'), nctx = nc.getContext('2d');
  let lastKey = '', lastPhaseKey = '', tick = 0;

  function sizeCanvas(c, h) {
    const r = c.getBoundingClientRect();
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.max(200, Math.round(r.width * dpr)), hh = Math.round(h * dpr);
    if (c.width !== w || c.height !== hh) { c.width = w; c.height = hh; }
    return dpr;
  }

  // the pipeline cards: rebuilt only when who-does-what changes
  function renderPipe(info) {
    const key = STAGES.map((s) => { const v = info.stages[s.id]; return v.who + '|' + v.kind + '|' + v.detail; }).join('#');
    if (key === lastKey) return;
    lastKey = key;
    const whos = STAGES.map((st) => info.stages[st.id].who).join(' | ');
    const piecesEl = document.getElementById('pieces');
    if (piecesEl) piecesEl.innerHTML = PIECES.map(([name, kind, what]) => {
      const short = name.replace(/^Red /, '');
      const on = whos.includes(name) || (kind === 'net' && whos.includes(short)) || (name === 'Red correctora' && whos.includes('red correctora')) || (name === 'Caminata SIMBICON' && whos.includes('SIMBICON'));
      return `<li class="piece ${KINDS[kind].cls}${on ? ' on' : ''}"><b>${esc(name)}</b><span>${esc(what)}</span></li>`;
    }).join('');
    pipe.innerHTML = STAGES.map((s, i) => {
      const v = info.stages[s.id], k = KINDS[v.kind] || KINDS.none;
      return `<li class="stage ${k.cls}"><span class="stage-n">${i + 1}</span><span class="stage-t">${s.title}</span><b class="stage-who">${esc(v.who)}</b><span class="badge">${k.label}</span><span class="stage-d">${esc(v.detail)}</span></li>`;
    }).join('');
  }
  function renderPhases(info) {
    const key = (info.phases || []).map((p) => p.id + p.label).join(',') + '@' + info.phase + '@' + (info.sub || '');
    if (key === lastPhaseKey) return;
    lastPhaseKey = key;
    if (!info.phases || !info.phases.length) { phasesEl.innerHTML = '<span class="small">Esta habilidad no tiene fases: hace siempre lo mismo.</span>'; return; }
    phasesEl.innerHTML = info.phases.map((p) => `<span class="chip${p.id === info.phase ? ' on' : ''}">${esc(p.label)}${p.id === info.phase && info.sub ? ' · ' + esc(info.sub) : ''}</span>`).join(info.cyclic ? '<span class="arrow" aria-hidden="true">·</span>' : '<span class="arrow" aria-hidden="true">→</span>');
  }
  function renderFacts(info) {
    factsEl.innerHTML = (info.facts || []).map(([k, v, hot]) => `<dt>${esc(k)}</dt><dd${hot ? ' class="hot"' : ''}>${esc(v)}</dd>`).join('');
  }

  // the 14 joints: track = the joint's range (left = its lower limit, right = upper, tick = straight);
  // ring = where the joint is; bar = the target the motor turns to (from the centre tick)
  function drawMotors(ep, info) {
    const dpr = sizeCanvas(mc, 14 * 22 + 30);
    const W = mc.width, H = mc.height;
    mctx.clearRect(0, 0, W, H);
    const rowH = 22 * dpr, labW = 92 * dpr, x0 = labW, x1 = W - 10 * dpr, y0 = 6 * dpr;
    mctx.font = `${11 * dpr}px ${css('--mono')}`; mctx.textBaseline = 'middle';
    const ink = css('--ink'), muted = css('--muted'), rule = css('--rule'), acc = css('--accent'), base = css('--k-tuned'), net = css('--k-net');
    const kindColor = css('--' + (KINDS[info.stages.joints.kind] || KINDS.none).cls) || ink;
    const sel = info.sel;
    $('legNet').hidden = !(sel && sel.net);
    JOINT_ORDER.forEach((j, r) => {
      const jt = world.joints[j];
      const y = y0 + r * rowH + rowH / 2 + (r >= 8 ? 6 * dpr : 0);
      const X = (a) => x0 + (Math.max(jt.lo, Math.min(jt.hi, a)) - jt.lo) / (jt.hi - jt.lo) * (x1 - x0);
      const toAngle = (c) => (c >= 0 ? c * jt.hi : c * -jt.lo);
      mctx.fillStyle = muted; mctx.textAlign = 'left'; mctx.fillText(JOINT_NAMES[j], 0, y);
      mctx.strokeStyle = rule; mctx.lineWidth = 4 * dpr; mctx.lineCap = 'round';
      mctx.beginPath(); mctx.moveTo(x0, y); mctx.lineTo(x1, y); mctx.stroke();
      const z = X(0);
      mctx.lineWidth = 1 * dpr; mctx.beginPath(); mctx.moveTo(z, y - 7 * dpr); mctx.lineTo(z, y + 7 * dpr); mctx.stroke();
      const cmd = ep.action[j] || 0;
      if (sel && sel.net) {
        // runner target (blue) and the network's correction on top of it (orange arrow)
        const b = X(toAngle(sel.base[j])), f = X(toAngle(Math.max(-1, Math.min(1, sel.base[j] + sel.corr[j]))));
        mctx.strokeStyle = base; mctx.lineWidth = 6 * dpr; mctx.lineCap = 'butt';
        mctx.beginPath(); mctx.moveTo(z, y); mctx.lineTo(b, y); mctx.stroke();
        if (Math.abs(f - b) > 0.5 * dpr) {
          mctx.strokeStyle = net; mctx.lineWidth = 6 * dpr;
          mctx.beginPath(); mctx.moveTo(b, y); mctx.lineTo(f, y); mctx.stroke();
        }
      } else {
        mctx.strokeStyle = kindColor; mctx.lineWidth = 6 * dpr; mctx.lineCap = 'butt';
        mctx.beginPath(); mctx.moveTo(z, y); mctx.lineTo(X(toAngle(cmd)), y); mctx.stroke();
      }
      const a = Box2D.b2RevoluteJoint_GetAngle(jt.id);
      mctx.fillStyle = css('--stage'); mctx.strokeStyle = ink; mctx.lineWidth = 2 * dpr;
      mctx.beginPath(); mctx.arc(X(a), y, 4.5 * dpr, 0, Math.PI * 2); mctx.fill(); mctx.stroke();
    });
    void acc;
  }

  // the network: columns of units, each square lit by its activation (orange +, blue -)
  function drawNet(info) {
    const dpr = sizeCanvas(nc, 230);
    const W = nc.width, H = nc.height;
    nctx.clearRect(0, 0, W, H);
    const n = info.net;
    $('netCard').classList.toggle('idle', !n);
    $('netTitle').textContent = n ? n.title : 'Red neuronal';
    $('netNote').textContent = n ? n.note : 'Ahora no maneja ninguna red neuronal: todo lo que se mueve sale de código.';
    if (!n) return;
    const pos = css('--k-net'), neg = css('--k-tuned'), rule = css('--rule'), muted = css('--muted');
    const layers = n.layers;
    const colW = (W - 20 * dpr) / layers.length;
    nctx.font = `${10.5 * dpr}px ${css('--mono')}`; nctx.textAlign = 'center'; nctx.textBaseline = 'top';
    layers.forEach((L, li) => {
      const vals = L.values, cnt = vals.length;
      const cols = Math.max(1, Math.ceil(cnt / Math.floor((H - 40 * dpr) / (7 * dpr))));
      const rows = Math.ceil(cnt / cols);
      const cell = Math.min(14 * dpr, (H - 46 * dpr) / rows, (colW - 16 * dpr - (L.labels ? 70 * dpr : 0)) / cols);
      const gx = 10 * dpr + li * colW + (colW - cols * cell - (L.labels ? 70 * dpr : 0)) / 2, gy = 26 * dpr + (H - 46 * dpr - rows * cell) / 2;
      nctx.fillStyle = muted; nctx.font = `${10 * dpr}px ${css('--mono')}`;
      nctx.fillText(L.name, 10 * dpr + li * colW + colW / 2, 0); nctx.fillText(String(cnt), 10 * dpr + li * colW + colW / 2, 11 * dpr);
      for (let i = 0; i < cnt; i++) {
        const v = Math.max(-1, Math.min(1, vals[i] * (L.gain || 1)));
        const c = i % cols, r = Math.floor(i / cols);
        nctx.globalAlpha = 0.12 + 0.88 * Math.abs(v);
        nctx.fillStyle = Math.abs(v) < 0.02 ? rule : v > 0 ? pos : neg;
        nctx.fillRect(gx + c * cell + 1, gy + r * cell + 1, cell - 2, cell - 2);
      }
      nctx.globalAlpha = 1;
      if (L.labels && cell >= 10 * dpr) {
        nctx.textAlign = 'left'; nctx.textBaseline = 'middle'; nctx.fillStyle = muted;
        L.labels.forEach((lab, i) => { if (i < cnt) nctx.fillText(lab, gx + cols * cell + 4 * dpr, gy + i * cell + cell / 2); });
        nctx.textAlign = 'center'; nctx.textBaseline = 'top';
      }
      if (li < layers.length - 1) {
        nctx.fillStyle = muted; nctx.fillText('→', 10 * dpr + (li + 1) * colW, H / 2 - 6 * dpr);
      }
    });
  }

  return {
    update(ep, info) {
      tick++;
      renderPipe(info);
      renderPhases(info);
      if (tick % 6 === 0) renderFacts(info);
      drawMotors(ep, info);
      if (tick % 2 === 0) drawNet(info);
    },
    reset() { lastKey = ''; lastPhaseKey = ''; },
  };
}
