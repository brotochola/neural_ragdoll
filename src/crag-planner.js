// The crag planner (crag.js): the wall planner (src/wall-planner.js) on an irregular rock face. The
// "rungs" are the ledges of the face, each at its own height and its own distance out: three points stay
// on the rock while the fourth moves to the next ledge up; hands go on the lip of a ledge, feet in on its
// floor, the hips hang a set distance out from the rock at chest height (the face steps back as it
// rises). Then over the summit edge. Installs itself on an episode (wraps ep.updateGrip).
const ENV = globalThis.process?.env || {};
const smooth = (s) => s * s * (3 - 2 * s);

export function installCragPlanner(ep, world, p, ctx) {
  const { config, pose } = ctx;
  const HZ = config.physics.hz, J = config.net.jointCount, OUT = J + config.net.gripOutputs, P = config.partIndex;
  // the face, from the absolute top line (rises are faces); the knobs from ctx.crag (relative to the start,
  // shifted like the terrain)
  const S = ep.terrain.surface;
  const dx = S[1][0] - ctx.crag.surface[1][0];
  const faces = [];
  for (let i = 0; i < S.length - 1; i++) { const [x1, y1] = S[i], [x2, y2] = S[i + 1]; if (y2 > y1 + 1e-6) faces.push({ x1, y1, x2, y2 }); }
  const rungs = ctx.crag.knobs.map((k) => ({ lip: k.lip + dx, back: k.back + dx, y: k.y, top: k.y, x0: k.lip + dx, x1: k.back + dx }));
  const wallH = faces[faces.length - 1].y2;
  const wallX = faces[faces.length - 1].x2; // the summit edge
  const faceAt = (y) => {
    for (const f of faces) if (y <= f.y2) return y <= f.y1 ? f.x1 : f.x1 + (f.x2 - f.x1) * (y - f.y1) / (f.y2 - f.y1);
    return wallX;
  };
  // index N+1 is the TOP of the wall: hands on its top surface past the edge, then the feet
  const N = rungs.length;
  const TOP = N + 1;
  const ry = (i) => (i <= 0 ? 0 : i === TOP ? wallH : rungs[i - 1].top);
  // limb state: rung index (0 = floor) and current world target
  const feet0 = [P.footLeft, P.footRight].map((q) => pose.soleOf(world, q));
  const lim = {
    hand: [{ rung: 0, pos: null, grip: false }, { rung: 0, pos: null, grip: false }],
    foot: [{ rung: 0, pos: { x: feet0[0].x, y: feet0[0].y } }, { rung: 0, pos: { x: feet0[1].x, y: feet0[1].y } }],
  };
  const handTarget = (i) => (i === TOP ? { x: wallX + p.topHand, y: wallH + 0.03 } : { x: rungs[i - 1].back - 0.05 + p.handIn, y: ry(i) + p.handUp });
  const footTarget = (i) => {
    if (i <= 0) return null;
    if (i === TOP) return { x: wallX + 0.22, y: wallH + 0.005 };
    const L = rungs[i - 1];
    return { x: L.back - 0.07 + p.footIn, y: L.y + 0.005 };
  };
  const handsTop = () => lim.hand[0].rung === TOP && lim.hand[1].rung === TOP;
  const feetTop = () => lim.foot.filter((f) => f.rung === TOP).length;
  // first rung the hands can reach standing
  let reach = 0;
  const sh0 = { x: pose.pelvisOf(world).x + 0.05, y: 1.45 };
  for (let i = 1; i <= rungs.length; i++) { const h = handTarget(i); if (h.y < 1.6 && Math.hypot(h.x - sh0.x, h.y - sh0.y) < (p.reachR || 0.62)) reach = i; }
  if (ENV.DEBUG_CW) console.log('  sh0', sh0.x.toFixed(2), sh0.y, 'holds', rungs.map((L, i) => handTarget(i + 1)).map((h) => h.x.toFixed(2) + ',' + h.y.toFixed(2)).join(' '));
  if (!reach) { let bd = 9; for (let i = 1; i <= rungs.length; i++) { const h = handTarget(i); const d = Math.hypot(h.x - sh0.x, h.y - sh0.y); if (h.y < 1.75 && d < bd) { bd = d; reach = i; } } }
  if (!reach) reach = 1; // about shoulder height: an arm raised overhead next to the wall hit the rungs
  // the move list is generated as the climb goes
  let move = null;
  let retry = 0, lastKey = '';
  const pv0 = pose.pelvisOf(world);
  let pelvisGoal = { x: pv0.x, y: pv0.y };
  let pelvisFrom = { ...pelvisGoal };
  const next = () => {
    // highest rung for hands, feet
    const hMin = Math.min(lim.hand[0].rung, lim.hand[1].rung);
    const fMax = Math.max(lim.foot[0].rung, lim.foot[1].rung);
    const fMin = Math.min(lim.foot[0].rung, lim.foot[1].rung);
    if (lim.hand[0].rung === 0) return { kind: 'hand', k: 0, to: reach };
    if (lim.hand[1].rung === 0) return { kind: 'hand', k: 1, to: reach };
    // feet climb while they stay a body length below the hands; then the pelvis rises, then hands
    const handsY = Math.min(ry(lim.hand[0].rung), ry(lim.hand[1].rung));
    const feetY = Math.min(ry(lim.foot[0].rung), ry(lim.foot[1].rung));
    // hands first while they can still reach from the feet, the lower hand first; then the lower foot
    const hk = lim.hand[0].rung <= lim.hand[1].rung ? 0 : 1;
    const hTo = lim.hand[hk].rung + 1;
    if (hTo < TOP && ry(hTo) - feetY <= p.spanMax) return { kind: 'hand', k: hk, to: hTo };
    // onto the top only once the feet are high enough for the hips to bring the shoulders up to the edge
    if (hTo === TOP && wallH - feetY <= p.spanTop) return { kind: 'hand', k: hk, to: hTo };
    const fk = lim.foot[0].rung <= lim.foot[1].rung ? 0 : 1;
    const fTo = lim.foot[fk].rung + 1;
    // MANTLE: both hands on top; the feet walk up the last rungs, then step onto the top
    if (handsTop()) {
      // BELLY MANTLE (p.belly): press up on the arms, tip the chest over the edge and crawl onto the top
      // on the hands (stepping a foot up onto the top from a knob below the edge never worked)
      const footHi = Math.max(ry(lim.foot[0].rung), ry(lim.foot[1].rung));
      if ((p.belly || p.stepMantle) && (footHi >= wallH - p.mantleH || retry >= 3)) return { kind: 'mantle' };
      if (feetTop() === 1) { const k2 = lim.foot[0].rung === TOP ? 1 : 0; return { kind: 'foot', k: k2, to: TOP }; }
      if (fTo <= N) return { kind: 'foot', k: fk, to: fTo };
      if (fTo === TOP || Math.max(lim.foot[0].rung, lim.foot[1].rung) >= N) { const k1 = lim.foot[0].rung >= lim.foot[1].rung ? 0 : 1; if (lim.foot[k1].rung !== TOP) return { kind: 'foot', k: k1, to: TOP }; }
      return null;
    }
    if (fTo <= rungs.length && handsY - ry(fTo) >= p.spanMin) return { kind: 'foot', k: fk, to: fTo };
    // stuck (a hand caught a lower rung than planned): a foot goes up anyway while there is room
    if (fTo <= rungs.length && handsY - ry(fTo) >= p.spanLow) return { kind: 'foot', k: fk, to: fTo };
    return null;
  };
  let leanNow = p.lean;
  const pelvisFor = () => {
    const fMax = Math.max(lim.foot[0].pos.y, lim.foot[1].pos.y);
    leanNow = p.lean;
    const ft = feetTop();
    if (ft === 2) return { x: wallX + 0.28, y: wallH + 0.98 };
    if (ft === 1) { leanNow = -0.5; return { x: wallX + 0.12, y: wallH + p.mantleUp }; }
    if (handsTop()) {
      // as the feet near the top the hips come in over the edge and the chest leans over the top
      const frac = Math.max(0, Math.min(1, (fMax - (wallH - 1.3)) / 1.0));
      leanNow = p.lean + (p.mantleLean - p.lean) * frac;
      const x0 = Math.min(wallX - p.pelvisOffTop, faceAt(Math.min(wallH, fMax + p.pelvisAbove + p.chestUp)) - p.pelvisOff);
      return { x: x0 + (wallX - 0.12 - x0) * frac, y: Math.min(fMax + p.pelvisAbove, wallH + 0.3) };
    }
    const hands = lim.hand.filter((h) => h.pos).map((h) => h.pos.y);
    const onRung = lim.foot.some((f) => f.rung > 0);
    let y = onRung ? fMax + p.pelvisAbove : 0.98; // sole heights read 5 cm high, so 0.98 is standing
    if (hands.length) y = Math.min(y, Math.min(...hands) - p.armDown);
    const onWall = lim.foot.some((f) => f.rung > 0) || lim.hand.some((h) => h.rung > 0);
    // near the top (a hand on the last rung or over the edge): hips in closer so the arms reach over it
    const nearTop = lim.hand.some((h) => h.rung >= N);
    const yy = Math.max(0.6, Math.min(y, 0.95 + Math.max(0, fMax)));
    return { x: onWall ? (nearTop ? wallX - p.pelvisOffTop : faceAt(Math.min(wallH, yy + p.chestUp)) - p.pelvisOff) : pv0.x, y: yy };
  };
  let mt = null;
  const startMove = () => {
    const m = next();
    if (m && m.kind === 'mantle') { mt = { phase: 'press', t: 0, from: { ...pelvisGoal }, k: 0, lean0: leanNow }; move = { kind: 'mantle', t: 0, dur: 99 }; return; }
    if (!m) { move = { kind: 'hold', t: 0, dur: 99 }; return; }
    const limb = m.kind === 'hand' ? lim.hand[m.k] : lim.foot[m.k];
    const from = limb.pos || pose.handOf(world, m.k === 0 ? P.handLeft : P.handRight);
    const to = Object.assign({}, m.kind === 'hand' ? handTarget(m.to) : footTarget(m.to));
    // a move that failed before is tried again a little differently: a hand lower and closer in, a foot
    // with the hips lower and closer to the rock (retry counts per limb and target)
    const key = m.kind + m.k + ':' + m.to;
    if (key !== lastKey) { lastKey = key; retry = 0; }
    if (m.kind === 'hand') { to.y -= (p.retryHandY || 0.03) * Math.min(3, retry); to.x += 0.015 * Math.min(3, retry); }
    move = Object.assign(m, { t: 0, dur: m.kind === 'hand' ? p.tHand : p.tFoot, from: { x: from.x, y: from.y }, target: to, wait: 0 });
    if (m.kind === 'hand') limb.grip = false;
    pelvisFrom = { ...pelvisGoal };
  };
  const tmp = new Float64Array(OUT);
  const lerp = (a, b, q) => ({ x: a.x + (b.x - a.x) * q, y: a.y + (b.y - a.y) * q });
  function runMantle(e) {
    mt.t += 1 / HZ;
    const pv = pose.pelvisOf(world);
    const hands = lim.hand.map((h) => (h.pos ? { ...h.pos } : null));
    const gripping = lim.hand.map((h) => h.grip);
    let feet = lim.foot.map((f) => ({ x: f.pos.x, y: f.pos.y, angle: 0 }));
    let planted = [true, true];
    let pel, lean;
    // trailing legs: hang down the face (toes on the rock) until the hips are well over the edge, then
    // follow along the top (stretched out behind in the air they pulled the body back off the edge)
    const trail = () => {
      planted = [false, false];
      const over = pv.x - wallX;
      // while the knees are still off the edge the shins hang down the face (knees bent over the edge)
      feet = [0, 1].map((k) => (over < 0.35 ? { x: Math.min(pv.x - 0.1, wallX - 0.06), y: pv.y - 0.72 + 0.08 * k, angle: 0.4 }
        : pv.x - 0.45 < wallX ? { x: wallX - 0.08, y: wallH - 0.35, angle: 0.4 } : { x: Math.max(wallX + 0.06, pv.x - 0.75 - 0.08 * k), y: wallH + 0.06, angle: 0 }));
    };
    if (p.stepMantle > 0 && ['press', 'stepup', 'rise', 'step2', 'stand'].includes(mt.phase)) {
      // PRESS-OUT MANTLE: straight arms push the hips up above the edge (trunk leaning over the hands),
      // a foot steps up onto the top, the body rises over it, the other foot follows, the hands let go
      const top = { x: wallX + p.psFoot, y: wallH + 0.05 };
      if (mt.phase === 'press') {
        const q = smooth(Math.min(1, mt.t / p.tPress));
        pel = lerp(mt.from, { x: wallX - p.psOff, y: wallH + p.psUp }, q);
        lean = mt.lean0 + (p.psLean - mt.lean0) * q;
        if (mt.t >= p.tPress + 0.2) { mt.phase = 'stepup'; mt.t = 0; mt.from = { ...pel }; mt.k = lim.foot[0].pos.y >= lim.foot[1].pos.y ? 0 : 1; mt.footFrom = { ...lim.foot[mt.k].pos }; }
      } else if (mt.phase === 'stepup') {
        pel = { ...mt.from }; lean = p.psLean;
        const q = Math.min(1, mt.t / p.tFoot), a = mt.footFrom;
        const wp1 = { x: Math.min(a.x, wallX - 0.15), y: wallH + p.psClear }, wp2 = { x: top.x, y: wallH + p.psClear };
        const pts = [a, wp1, wp2, top], cuts = [0, 0.45, 0.8, 1];
        let si = 0; while (si < 2 && q > cuts[si + 1]) si++;
        const e2 = smooth(Math.min(1, (q - cuts[si]) / (cuts[si + 1] - cuts[si])));
        feet[mt.k] = { x: pts[si].x + (pts[si + 1].x - pts[si].x) * e2, y: pts[si].y + (pts[si + 1].y - pts[si].y) * e2, angle: 0 };
        planted[mt.k] = false;
        if (q >= 1) {
          const so = pose.soleOf(world, mt.k === 0 ? P.footLeft : P.footRight);
          if (Math.abs(so.y - top.y) < 0.08 && Math.abs(so.x - top.x) < 0.15) { lim.foot[mt.k] = { rung: TOP, pos: { x: so.x, y: wallH + 0.05 } }; mt.phase = 'rise'; mt.t = 0; mt.from = { ...pel }; }
          else if (mt.t > p.tFoot + 0.6) { mt.phase = 'press'; mt.t = 0; mt.from = { ...pel }; mt.tries = (mt.tries || 0) + 1; }
        }
      } else if (mt.phase === 'rise') {
        const q = smooth(Math.min(1, mt.t / p.tRise2));
        const f = lim.foot[mt.k].pos;
        pel = lerp(mt.from, { x: f.x + 0.02, y: wallH + p.riseH }, q);
        lean = p.psLean + (p.riseLean - p.psLean) * q;
        // the other leg hangs out of the way, knee bent, until the hips are over the top
        planted[1 - mt.k] = false;
        feet[1 - mt.k] = { x: wallX - 0.12, y: pel.y - 0.6, angle: 0 };
        if (q > 0.7) for (let k = 0; k < 2; k++) { lim.hand[k].grip = false; }
        if (mt.t >= p.tRise2 + 0.3) { mt.phase = 'step2'; mt.t = 0; mt.from = { ...pel }; mt.footFrom = pose.soleOf(world, mt.k === 1 ? P.footLeft : P.footRight); }
      } else if (mt.phase === 'step2') {
        const k2 = 1 - mt.k, f = lim.foot[mt.k].pos;
        pel = { x: f.x + 0.02, y: wallH + p.riseH }; lean = p.riseLean;
        const q = Math.min(1, mt.t / p.tFoot), a = mt.footFrom, b = { x: f.x + 0.22, y: wallH + 0.05 };
        const wp = { x: Math.max(a.x, wallX + 0.05), y: wallH + p.psClear };
        feet[k2] = q < 0.5 ? lerp(a, wp, smooth(q * 2)) : lerp(wp, b, smooth(q * 2 - 1));
        feet[k2].angle = 0; planted[k2] = false;
        if (q >= 1 && mt.t > p.tFoot + 0.3) { lim.foot[k2] = { rung: TOP, pos: { x: b.x, y: b.y } }; mt.phase = 'stand'; mt.t = 0; }
      } else {
        const f0 = lim.foot[0].pos, f1 = lim.foot[1].pos;
        pel = { x: (f0.x + f1.x) / 2, y: wallH + 0.93 }; lean = 0;
        stood = true; mantled = true;
      }
      const handsFree = mt.phase === 'step2' || mt.phase === 'stand' || (mt.phase === 'rise' && !lim.hand[0].grip);
      for (let k = 0; k < 2; k++) if (handsFree) { hands[k] = null; gripping[k] = false; }
      pelvisGoal = pel; leanNow = lean;
      const goal = { pelvis: pel, torso: lean, feet, hands, gripping, planted, torsoGain: p.torsoGain };
      const onTwo = planted[0] && planted[1];
      const jt = handsFree ? pose.balancedTargets(goal, e.measure, { planted, support: onTwo ? undefined : lim.foot[mt.k].pos.x, kHip: 0.3, kAnkle: 0.8, kAnkleVel: 0.4, kHipVel: 0.2 }, pose.actualOf(world)) : pose.jointTargets(goal, pose.actualOf(world));
      pose.commandsFor(world, jt, tmp);
      for (let j = 0; j < J; j++) e.action[j] = tmp[j];
      for (let k = 0; k < 2; k++) e.action[J + k] = gripping[k] ? 1 : -1;
      return;
    }
    if (mt.phase === 'press') {
      const q = smooth(Math.min(1, mt.t / p.tPress));
      pel = lerp(mt.from, { x: wallX - p.pressOff, y: wallH + p.pressUp }, q);
      lean = mt.lean0 + (p.pressLean - mt.lean0) * q;
      if (mt.t >= p.tPress + 0.2) { mt.phase = 'over'; mt.t = 0; mt.from = { ...pel }; }
    } else if (mt.phase === 'over') {
      const q = smooth(Math.min(1, mt.t / p.tOver));
      // up over the edge first, then forward onto the top (diagonally the hips ran into the corner)
      const up = { x: mt.from.x, y: wallH + 0.2 };
      pel = q < 0.5 ? lerp(mt.from, up, q * 2) : lerp(up, { x: wallX + p.overX, y: wallH + 0.2 }, q * 2 - 1);
      lean = p.pressLean + (-1.35 - p.pressLean) * q;
      // the feet stay on their knobs and push the hips up over the edge
      if (mt.t >= p.tOver + 0.2) { mt.phase = 'hand'; mt.t = 0; mt.from = { x: pv.x, y: wallH + 0.17 }; }
    } else if (mt.phase === 'hand' || mt.phase === 'pull') {
      trail();
      // chest up on the arms (tipped flat or past it, the shoulders cannot reach forward along the top)
      lean = p.crawlLean;
      const k = mt.k;
      if (mt.phase === 'hand') {
        pel = { ...mt.from };
        if (!mt.hand) { const other = lim.hand[1 - k].pos; mt.hand = { from: { ...lim.hand[k].pos }, to: { x: pv.x + p.crawlReach, y: wallH + 0.03 } }; lim.hand[k].grip = false; }
        const q = Math.min(1, mt.t / p.tHand);
        const a = mt.hand.from, b = mt.hand.to, mid = { x: (a.x + b.x) / 2, y: wallH + 0.15 };
        const pt = q < 0.5 ? lerp(a, mid, smooth(q * 2)) : lerp(mid, b, smooth(q * 2 - 1));
        if (q >= 1 && mt.t > p.tHand) pt.y -= Math.min(0.05, (mt.t - p.tHand) * 0.2);
        hands[k] = pt; gripping[k] = false;
        const hp = pose.handOf(world, k === 0 ? P.handLeft : P.handRight);
        // close on the top wherever the hand comes down ahead of the hips (it may not reach the full way)
        e.action[J + k] = q > 0.6 && hp.x > pv.x + 0.15 && hp.y < wallH + 0.12 ? 1 : -1;
        if (ENV.DEBUG_MANTLE && e.frame % 15 === 0) console.log('   crawl hand', k, 'q', q.toFixed(2), 'hand', hp.x.toFixed(2), hp.y.toFixed(2), 'goal', pt.x.toFixed(2), pt.y.toFixed(2), 'target', b.x.toFixed(2), b.y.toFixed(2), 'pelvis', pv.x.toFixed(2), pv.y.toFixed(2), pv.angle.toFixed(2), 'edge', wallX.toFixed(2), wallH.toFixed(2));
        const g = e.grips[k];
        if (q > 0.6 && g && g.joint) { lim.hand[k] = { rung: TOP, pos: { x: hp.x, y: hp.y }, grip: true }; mt.phase = 'pull'; mt.t = 0; mt.hand = null; mt.from = { x: pv.x, y: wallH + 0.17 }; }
        else if (mt.t > p.tHand + 1.0) { mt.hand = null; mt.t = 0; } // try again
      } else {
        const q = smooth(Math.min(1, mt.t / p.tPull));
        pel = lerp(mt.from, { x: mt.from.x + p.pullStep, y: wallH + 0.15 }, q);
        if (mt.t >= p.tPull + 0.2) { mt.k = 1 - mt.k; mt.phase = 'hand'; mt.t = 0; mt.from = { x: pv.x, y: wallH + 0.17 }; }
      }
      if (pv.x > wallX + p.doneX) { mt.phase = 'lie'; mt.t = 0; }
    }
    if (mt.phase === 'getup' || mt.phase === 'stand') {
      // the get-up expert network (as trained: from rest, with its get-up clock), then the stand network
      // (the episode's own), both told the floor is the summit (groundOffset)
      const t = mt.t;
      const hand = (ctx.nets.handover && ctx.nets.handover[mt.side]) || 3;
      // as the expert was trained: it is the episode's network until the handover, then the stand net
      const src = mt.phase === 'getup' && t >= hand && mt.stand ? mt.stand.action : e.network.action;
      if (src !== e.network.action) for (let j = 0; j < OUT; j++) e.network.action[j] = src[j];
      for (let k = 0; k < 2; k++) e.action[J + k] = -1;
      const up = e.measure.headHeight - wallH > 1.35 && e.measure.torsoCos > 0.8;
      if (mt.phase === 'getup' && t > hand + 1 && up) { mt.phase = 'stand'; e.options.getup = false; e.options.startTime = 0; stood = true; e.network = mt.stand; mt.stand = null; mt.getup = null; }
      else if (mt.phase === 'getup' && t > hand + 6) { mt.phase = 'lie'; mt.t = 0; e.options.getup = false; e.options.startTime = 0; e.network = mt.stand; mt.stand = null; mt.getup = null; }
      return;
    }
    if (mt.phase === 'lie') {
      // on the top: let go, lie flat (legs along the top)
      // the hands stay where they are, open (dropped to the default pose at once, the arms swept along the
      // rock and threw the body back off the edge)
      for (let k = 0; k < 2; k++) { lim.hand[k].grip = false; const hp = pose.handOf(world, k === 0 ? P.handLeft : P.handRight); if (!mt.restHands) mt.restHands = []; if (!mt.restHands[k]) mt.restHands[k] = { x: hp.x, y: hp.y }; hands[k] = mt.restHands[k]; gripping[k] = false; }
      planted = [false, false];
      feet = [0, 1].map((k) => ({ x: Math.max(wallX + 0.06, pv.x - 0.85), y: wallH + 0.06, angle: 0 }));
      pel = { x: pv.x, y: wallH + 0.13 }; lean = -1.5;
      mantled = true;
      // still for a moment: get up (when the caller gave the networks)
      let still = true;
      for (let q = 0; q < world.parts.length; q++) if (Math.abs(world.vx[q]) > 0.15 || Math.abs(world.vy[q]) > 0.15) still = false;
      if (ctx.nets && ((mt.t > 0.8 && still) || mt.t > 3)) {
        mt.phase = 'getup'; mt.t = 0;
        mt.side = Math.atan2(world.s[0], world.c[0]) > 0 ? 'up' : 'down';
        mt.stand = e.network; // the stand net keeps reading the inputs (decide hook)
        mt.getup = new ctx.nets.ElmanNetwork(Float32Array.from(ctx.nets.getupW));
        e.network = mt.getup;
        e.groundOffset = wallH;
        if (!e.options.targets || e.options.targets.length !== config.targets.length) e.options.targets = config.targets.map((t) => Object.assign({}, t));
        e.options.getup = true;
        e.options.startTime = -e.frame / HZ;
        return;
      }
    }
    pelvisGoal = pel;
    leanNow = lean;
    const goal = { pelvis: pel, torso: lean, feet, hands, gripping, planted, torsoGain: p.torsoGain };
    const jt = pose.jointTargets(goal, pose.actualOf(world));
    // lying over the edge the leg IK (made for an upright body) asked the knees to bend the wrong way and
    // the legs stuck out straight into the air: set them directly instead, hanging down the face with
    // the knees bent while the knees are off the edge, then straight along the top
    if (['hand', 'pull', 'lie'].includes(mt.phase) || (mt.phase === 'over' && pv.x > wallX - 0.05)) {
      const kneesOver = pv.x - 0.45 > wallX;
      // hips still near the edge: thighs along the body, heels folded up toward the seat (compact, nothing
      // hanging off the edge to drag the body back); once the knees are over the top, legs out along it
      const hipsOver = pv.x > wallX + 0.05;
      const flat = mt.phase === 'lie';
      for (const [h, kn, an] of [[5, 6, 7], [11, 12, 13]]) { jt[h] = flat ? 0.1 : hipsOver ? p.foldHip : p.hangHip; jt[kn] = flat ? -0.2 : hipsOver ? -2.2 : p.hangKnee; jt[an] = 0.3; }
    }
    pose.commandsFor(world, jt, tmp);
    for (let j = 0; j < J; j++) e.action[j] = tmp[j];
    for (let k = 0; k < 2; k++) if (!(mt.phase === 'hand' && mt.k === k)) e.action[J + k] = gripping[k] ? 1 : -1;
  }
  let mantled = false, stood = false;
  if (ctx.nets) {
    // the get-up network reads the same inputs as the episode's own (stand) network
    const origDecide = ep.decide.bind(ep);
    ep.decide = function () {
      // on the summit the networks' marks sit on the summit floor (they were at y = 0, 2.6 m below, and
      // the get-up threw the body off the edge reaching for them)
      if (mt && (mt.phase === 'getup' || mt.phase === 'stand') && this.options.targets) {
        const t = this.options.targets, px = world.px[0];
        for (let i = 0; i < config.targets.length; i++) {
          const spec = config.targets[i], mk = t[i];
          if (!mk) continue;
          if (spec.ride) { mk.x = px + (spec.x || 0); mk.y = wallH + (spec.y || 0); if (mk.walkBase != null) mk.walkBase = mk.x - (this.walkOffset ? this.walkOffset() : 0); }
          else { mk.x = world.px[spec.id === 'footR' ? P.footRight : P.footLeft]; mk.y = wallH + config.skill.footGroundMeters; }
        }
      }
      origDecide();
      if (mt && mt.stand) mt.stand.forward(this.input);
    };
  }
  let handedOff = false;
  ep.updateGrip = ((orig) => function () {
    const m = this.measure;
    if (handedOff) { orig(); return; }
    if (mt) { runMantle(this); orig(); return; }
    if (!move) startMove();
    if (mt) { runMantle(this); orig(); return; }
    move.t += 1 / HZ;
    const s = Math.min(1, move.t / move.dur);
    const e = smooth(s);
    // limb targets
    const hands = lim.hand.map((h) => h.pos);
    const feet = lim.foot.map((f) => ({ x: f.pos.x, y: f.pos.y, angle: 0 }));
    const gripping = lim.hand.map((h) => h.grip);
    const planted = lim.foot.map(() => true);
    if (move.kind === 'hand' || move.kind === 'foot') {
      // path: to a point in front of the hold (out from the wall, a little above it) by 70% of the move,
      // then in onto it; straight lines dragged the hand up the rungs and pushed the body off backward
      const wp = { x: move.target.x - p.clear - 0.08, y: move.target.y + (move.kind === 'foot' ? 0.08 : 0.03) };
      // a hand first comes up close in front of the shoulder, elbow bent: swung up straight, the arm's
      // arc hit the rungs and pushed the body over backward
      const pts = [move.from];
      if (move.kind === 'hand') { const pv = pose.pelvisOf(world); pts.push({ x: Math.min(wp.x, pv.x + 0.15), y: Math.max(move.from.y, pv.y + 0.75) }); }
      // a foot first rises where it is (knee up, away from the wall), then goes in above the rung and down:
      // going forward first slid it under the rung below
      else { pts.push({ x: Math.min(move.from.x, wp.x) - 0.04, y: move.from.y + 0.04 }); pts.push({ x: Math.min(move.from.x, wp.x) - 0.04, y: wp.y }); }
      pts.push(wp, move.target);
      const segs = pts.length - 1;
      const cuts = segs === 4 ? [0, 0.2, 0.55, 0.8, 1] : segs === 3 ? [0, 0.4, 0.75, 1] : [0, 0.7, 1];
      let si = 0;
      while (si < segs - 1 && s > cuts[si + 1]) si++;
      const q = smooth(Math.min(1, (s - cuts[si]) / (cuts[si + 1] - cuts[si])));
      const pt = { x: pts[si].x + (pts[si + 1].x - pts[si].x) * q, y: pts[si].y + (pts[si + 1].y - pts[si].y) * q };
      // waiting for the grip: the hand presses down and in onto the knob (it hovered 6 cm over it)
      if (move.kind === 'hand' && s >= 1 && move.wait > 0) { const w = Math.min(1, move.wait / 0.4); pt.y -= (p.press || 0.06) * w; pt.x += 0.02 * w; }
      if (move.kind === 'hand') { hands[move.k] = pt; gripping[move.k] = false; }
      else { feet[move.k] = { x: pt.x, y: pt.y, angle: 0 }; planted[move.k] = false; }
    }
    // pelvis: rise after a foot is placed (towards the pose the new holds allow)
    const goalNow = pelvisFor();
    // a foot going up needs room under the hips for the knee: they hang at most footBelow above its target
    if (move.kind === 'foot' && p.footBelow && move.target && move.to !== TOP && !handsTop()) goalNow.y = Math.min(goalNow.y, move.target.y + p.footBelow);
    if (move.kind === 'foot' && retry > 0) { goalNow.y -= (p.retryPelY || 0.06) * Math.min(3, retry); goalNow.x += (p.retryPelX || 0.04) * Math.min(3, retry); }
    const pr = Math.min(1, move.t / p.tRise);
    pelvisGoal = { x: pelvisFrom.x + (goalNow.x - pelvisFrom.x) * smooth(pr), y: pelvisFrom.y + (goalNow.y - pelvisFrom.y) * smooth(pr) };
    // both feet on top: let go and stand up
    if (feetTop() === 2) { hands[0] = hands[1] = null; gripping[0] = gripping[1] = false; lim.hand[0].grip = lim.hand[1].grip = false; }
    const goal = { pelvis: pelvisGoal, torso: leanNow, feet, hands, gripping, planted, torsoGain: p.torsoGain };
    // on the floor with no hand on the wall yet: keep the balance over the feet (ankles and hips)
    const floorOnly = lim.foot.every((f) => f.rung === 0) && !lim.hand.some((h) => h.grip);
    const jt = floorOnly && ENV.FLOOR_BAL ? pose.balancedTargets(goal, m, { planted }, pose.actualOf(world)) : pose.jointTargets(goal, pose.actualOf(world));
    pose.commandsFor(world, jt, tmp);
    if (ENV.DEBUG_LEG && move.kind === 'foot' && Math.round(move.t * HZ) % 10 === 0) {
      const L = move.k === 0 ? [5, 6, 7] : [11, 12, 13];
      const so = pose.soleOf(world, move.k === 0 ? P.footLeft : P.footRight);
      const pv = pose.pelvisOf(world);
      const S2 = move.k === 0 ? [11, 12, 13] : [5, 6, 7];
      const ang = (j) => ctx.Box2D.b2RevoluteJoint_GetAngle(world.joints[j].id);
      console.log('  stance foot pos', JSON.stringify(lim.foot[1 - move.k].pos), 'pelvis goal', JSON.stringify(pelvisGoal));
      console.log('  stance hip/knee/ankle target', jt[S2[0]].toFixed(2), jt[S2[1]].toFixed(2), jt[S2[2]].toFixed(2), 'actual', ang(S2[0]).toFixed(2), ang(S2[1]).toFixed(2), ang(S2[2]).toFixed(2), 'arms tgt', jt[2].toFixed(2), jt[3].toFixed(2), 'act', ang(2).toFixed(2), ang(3).toFixed(2));
      console.log('  foot', move.k, 't', move.t.toFixed(2), 'tgt', feet[move.k].x.toFixed(2), feet[move.k].y.toFixed(2), 'sole', so.x.toFixed(2), so.y.toFixed(2), 'pelvis', pv.x.toFixed(2), pv.y.toFixed(2), 'goal pel', pelvisGoal.x.toFixed(2), pelvisGoal.y.toFixed(2), 'hip/knee target', jt[L[0]].toFixed(2), jt[L[1]].toFixed(2), 'actual', world.joints[L[0]].angle != null ? '' : '', 'cmd', tmp[L[0]].toFixed(2), tmp[L[1]].toFixed(2));
    }
    for (let j = 0; j < J; j++) this.action[j] = tmp[j];
    // grips: held hands stay closed; a moving hand closes once it is at its rung
    for (let k = 0; k < 2; k++) {
      const moving = move.kind === 'hand' && move.k === k;
      // a moving hand closes only once it is at its rung (closing on the way caught lower rungs)
      let near = false;
      if (moving && s > 0.6) { const hp = pose.handOf(world, k === 0 ? P.handLeft : P.handRight); near = Math.hypot(hp.x - move.target.x, hp.y - move.target.y) < (s >= 1 ? 0.12 : 0.07); }
      this.action[J + k] = lim.hand[k].grip || near ? 1 : -1;
    }
    // end of move: hand gripped (or give up after a wait), foot down
    if (s >= 1) {
      if (move.kind === 'hand') {
        const g = this.grips[move.k];
        if (g && g.joint) {
          // the rung actually held: the one nearest the hand (it may have caught a lower one on the way)
          const hp = pose.handOf(world, move.k === 0 ? P.handLeft : P.handRight);
          let bi = move.to, bd = 9;
          for (let i = 1; i <= rungs.length; i++) { const d = Math.hypot(hp.x - (rungs[i - 1].back - 0.05), hp.y - rungs[i - 1].y); if (d < bd) { bd = d; bi = i; } }
          if (move.to === TOP && hp.y > wallH - 0.03) bi = TOP;
          lim.hand[move.k] = { rung: bi, pos: { x: hp.x, y: hp.y }, grip: true };
          move = null;
        }
        else if ((move.wait += 1 / HZ) > 0.6) { if (ENV.DEBUG_CW) { const hp = pose.handOf(world, move.k === 0 ? P.handLeft : P.handRight); console.log('  grip miss hand', move.k, 'at', hp.x.toFixed(2), hp.y.toFixed(2), 'target', move.target.x.toFixed(2), move.target.y.toFixed(2)); } move = null; retry++; } // retry the same move
      } else if (move.kind === 'foot') {
        // standing on the rung? (sole within 6 cm of its top and over it); else try again
        const so = pose.soleOf(world, move.k === 0 ? P.footLeft : P.footRight);
        const tgt = move.target;
        if (Math.abs(so.y - tgt.y) < 0.07 && Math.abs(so.x - tgt.x) < 0.12) { lim.foot[move.k] = { rung: move.to, pos: tgt }; move = null; }
        else if ((move.wait += 1 / HZ) > 0.5) {
          if (ENV.DEBUG_CW) console.log('  foot miss', move.k, 'at', so.x.toFixed(2), so.y.toFixed(2), 'target', tgt.x.toFixed(2), tgt.y.toFixed(2));
          move = null; retry++;
        }
      }
    }
    orig();
  })(ep.updateGrip.bind(ep));
  return { lim, feetTop, handsTop, rungs, wallX, wallH, N, TOP, faceAt, get mantled() { return mantled; }, get stood() { return stood; }, handOff() { handedOff = true; }, get phase() { return mt ? mt.phase : null; }, get move() { return move; } };
}
