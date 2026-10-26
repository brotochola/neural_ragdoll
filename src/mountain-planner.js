// The mountain planner (mountain.js): looks at the ground ahead, picks footholds and handholds, steps on the
// feet where it can and scrambles with the hands where it must. Installs itself on an episode (wraps
// ep.updateGrip). Shared by the trainer and the playground.
const ENV = globalThis.process?.env || {};
const smooth = (s) => s * s * (3 - 2 * s);

export function installMountainPlanner(ep, world, p, ctx) {
  const { config, pose } = ctx;
  const HZ = config.physics.hz, J = config.net.jointCount, OUT = J + config.net.gripOutputs, P = config.partIndex;
  const S = ep.terrain.surface; // absolute
  const summitX = S[S.length - 2][0];
  const H = ctx.H != null ? ctx.H : S[S.length - 1][1];
  const ground = (x) => {
    if (x <= S[0][0]) return 0;
    for (let i = 0; i < S.length - 1; i++) if (x <= S[i + 1][0]) { const [x1, y1] = S[i], [x2, y2] = S[i + 1]; return x2 > x1 ? y1 + (y2 - y1) * (x - x1) / (x2 - x1) : Math.max(y1, y2); }
    return S[S.length - 1][1];
  };
  const groundMax = (a, b) => { let m = Math.max(ground(a), ground(b)); for (const [x, y] of S) if (x > a && x < b) m = Math.max(m, y); return m; };
  // a foothold: the ground within 7 cm either side is level to 5 cm and the foot is not against a rise
  const flatAt = (x) => { const a = ground(x - 0.07), b = ground(x), c = ground(x + 0.07); return Math.max(a, b, c) - Math.min(a, b, c) < 0.05; };
  // a foothold on ground no steeper than maxSlope (the foot lies along it; friction 2 holds ~40 degrees),
  // not more than maxRise above the foot it steps from; the farthest such spot within a step
  const slopeAt = (x) => Math.atan2(ground(x + 0.06) - ground(x - 0.06), 0.12);
  const footholdAfter = (fromX, fromY, maxRise, maxSlope, limitX = Infinity) => {
    let best = null;
    const shrink = 1 / (1 + 0.5 * footFails); // after a failed step, shorter and lower ones
    for (let x = fromX + 0.08; x <= fromX + p.stepMax * shrink; x += 0.02) {
      const y = ground(x) + SOLE;
      const a = slopeAt(x);
      if (x > limitX || y - fromY > maxRise * shrink || Math.abs(a) > maxSlope) continue;
      best = { x, y, angle: a };
    }
    return best;
  };
  const SOLE = 0.05; // pose.soleOf reads a standing sole 5 cm above the ground
  const feet0 = [P.footLeft, P.footRight].map((q) => pose.soleOf(world, q));
  const lim = {
    foot: feet0.map((f) => ({ x: f.x, y: f.y })),
    hand: [null, null], // {x, y} while on the rock
    grip: [false, false],
  };
  const pv0 = pose.pelvisOf(world);
  let pelvisGoal = { x: pv0.x, y: pv0.y };
  let pelvisFrom = { ...pelvisGoal };
  let leanNow = 0;
  let leanCur = 0;
  let move = null;
  let stuck = 0;
  let footFails = 0; // foot moves in a row that did not land on their hold
  let handFails = 0; // hand moves in a row that ended without a grip
  // SCRAMBLE mode: set when the ground ahead rises steeply (hips down, trunk leaning in, hands to the
  // rock), cleared when it eases and the hands have been released
  let scramble = false;
  const scrambling = () => scramble || lim.grip[0] || lim.grip[1];
  const lead = () => (lim.foot[0].x >= lim.foot[1].x ? 0 : 1);
  const riseAhead = (x, y) => groundMax(x, x + 0.6) - y;
  const slopeAhead = (x) => Math.atan2(ground(x + 0.5) - ground(x - 0.1), 0.6);
  const shoulderAt = (pg, lean) => ({ x: pg.x + Math.sin(-lean) * 0.52, y: pg.y + Math.cos(lean) * 0.52 });
  const handholdFrom = (sh, minX = -1e9) => {
    let best = null;
    for (let x = Math.max(minX, sh.x - 0.1); x <= sh.x + p.handReach; x += 0.02) {
      const y = ground(x) + 0.03;
      if (Math.hypot(x - sh.x, y - sh.y) > p.handReach) continue;
      if (!best || x > best.x) best = { x, y };
    }
    return best;
  };
  const next = () => {
    const L = lead(), T = 1 - L;
    const lf = lim.foot[L], tf = lim.foot[T];
    // hands go to the rock when the feet alone find no foothold ahead (too high or too steep)
    const footOnly = footholdAfter(lf.x, lf.y, p.stepUp, p.footSlope);
    // where the ground starts to rise ahead of the leading foot: the feet walk up to it before the hands go down
    let xr = lf.x;
    while (xr < lf.x + 1.5 && ground(xr) < lf.y - SOLE + 0.06) xr += 0.02;
    const far = xr - lf.x > (p.baseGap || 0.3) && !scrambling();
    const steep = !far && (!footOnly || riseAhead(lf.x, lf.y - SOLE) > p.handRise);
    if (steep) scramble = true;
    // the planned hips and shoulders for this stance, to see what the hands can reach
    const pg = pelvisFor(null);
    const sh = shoulderAt(pg, leanNow);
    if (ENV.DEBUG_MT) console.log('    steep', steep, 'rise', riseAhead(lf.x, lf.y - SOLE).toFixed(2), 'pelvis plan', pg.x.toFixed(2), pg.y.toFixed(2), 'shoulder', sh.x.toFixed(2), sh.y.toFixed(2), 'lean', leanNow.toFixed(2), 'ground ahead', [0.1, 0.3, 0.5, 0.7].map((d) => ground(lf.x + d).toFixed(2)).join(' '));
    if (steep) {
      // hands: the one behind (or not down) goes forward first, while it can move ahead
      // a handhold on the rock ahead of the leading foot (handAhead), the trunk turns toward it
      // a hand that just failed to grip waits until a foot has moved up (the body closer to the rock),
      // and a hand goes no farther than the shoulder reaches
      const reach = handholdFrom(sh);
      const handsGripped = lim.grip[0] || lim.grip[1];
      if (!(handFails > 0 && handsGripped)) {
        for (const k of [0, 1].sort((a, b) => (lim.hand[a] ? lim.hand[a].x : -9) - (lim.hand[b] ? lim.hand[b].x : -9))) {
          const h = lim.hand[k];
          // hands stay a set distance ahead of the leading foot (they used to leapfrog on alone and
          // stretch the body out flat)
          let hx = Math.max(xr + 0.06, lf.x + p.handAhead) + (k === lead() ? 0.04 : 0);
          if (reach) hx = Math.min(hx, reach.x);
          const hold = { x: hx, y: ground(hx) + 0.03 };
          if (!h || !lim.grip[k] || hold.x > h.x + 0.12) return { kind: 'hand', k, target: hold };
        }
      }
    } else if (scrambling() && riseAhead(lf.x, lf.y - SOLE) < p.handRise * 0.5 && lim.hand.every((h) => !h || h.x < pg.x + 0.1)) {
      scramble = false;
      return { kind: 'release' };
    }
    // feet: the trailing foot steps past the leading one onto the next foothold
    const handsOn = lim.grip[0] || lim.grip[1];
    const maxRise = handsOn ? p.stepUp + 0.2 : p.stepUp;
    const maxSlope = handsOn ? p.footSlope + 0.25 : p.footSlope;
    if (scramble && !handsOn) return null; // wait for a hand on the rock first
    // the moving foot travels at most stride (from where it is) — a trailing foot sent 0.86 m ahead fell
    let hold = footholdAfter(lf.x, lf.y, maxRise, maxSlope, tf.x + (p.stride || 0.65) / (1 + 0.5 * footFails));
    if (hold) return { kind: 'foot', k: T, target: hold };
    // no foothold past the lead foot: bring the trailing foot up next to it
    hold = footholdAfter(tf.x, tf.y, maxRise, maxSlope);
    if (hold && hold.x < lf.x + 0.02) return { kind: 'foot', k: T, target: hold };
    return null;
  };
  // hips: over the standing foot when a foot is in the air, between the feet otherwise; pulled toward the
  // hands when scrambling; at a set height above the ground, no higher than the legs reach.
  function pelvisFor(m) {
    const planted = [0, 1].filter((k) => !(m && m.kind === 'foot' && m.k === k && m.lifted));
    let x;
    if (m && m.kind === 'foot') x = lim.foot[1 - m.k].x + 0.02;
    else x = (lim.foot[0].x + lim.foot[1].x) / 2;
    const hands = lim.hand.filter((h, k) => h && lim.grip[k]);
    if (hands.length) {
      const hx = hands.reduce((a, h) => a + h.x, 0) / hands.length;
      x = x + p.handPull * (hx - x);
    }
    leanNow = -Math.max(p.leanMin || -0.1, Math.min(p.leanCap || 0.3, p.leanK * slopeAhead(x))); // on the feet alone, a little lean into the slope at most
    let y = ground(x) + (scrambling() ? p.hipHands : p.hipH) + SOLE;
    if (scrambling()) {
      // trunk toward the hands (or where they are going): lean = angle of hips->hands from vertical, less a bit
      const goingTo = m && m.kind === 'hand' ? m.target : null;
      const hs = lim.hand.filter((h) => h).concat(goingTo ? [goingTo] : []);
      if (hs.length) {
        const hx = hs.reduce((a, h) => a + h.x, 0) / hs.length, hy = hs.reduce((a, h) => a + h.y, 0) / hs.length;
        // the shoulders go above and a little behind the hands (arms hanging onto the rock, not under a
        // trunk lying flat on it): the lean is the angle from the hips to there
        leanNow = -Math.max(0, Math.min(lim.grip[0] || lim.grip[1] ? p.leanMax || 1.0 : p.leanFree || 0.6, Math.atan2(hx - (p.shBack != null ? p.shBack : 0.15) - x, hy + (p.shUp != null ? p.shUp : 0.45) - y)));
      }
    }
    for (const k of planted) y = Math.min(y, lim.foot[k].y + 0.93);
    return { x, y };
  }
  const startMove = () => {
    // feet that slipped: plan from where they are
    for (let k = 0; k < 2; k++) {
      const so = pose.soleOf(world, k === 0 ? P.footLeft : P.footRight);
      if (Math.hypot(so.x - lim.foot[k].x, so.y - lim.foot[k].y) > 0.05) lim.foot[k] = { x: so.x, y: ground(so.x) + SOLE, angle: slopeAt(so.x) };
    }
    const m = next();
    if (ENV.DEBUG_MT) console.log('  t', (ep.frame / HZ).toFixed(1), m ? m.kind + (m.k != null ? m.k : '') + (m.target ? ' -> ' + m.target.x.toFixed(2) + ',' + m.target.y.toFixed(2) : '') : 'none', 'feet', lim.foot.map((f) => f.x.toFixed(2) + ',' + f.y.toFixed(2)).join(' '), 'hands', lim.hand.map((h, k) => (h ? h.x.toFixed(2) + ',' + h.y.toFixed(2) : '-') + (lim.grip[k] ? 'G' : '')).join(' '));
    if (!m) { stuck += 1; move = { kind: 'hold', t: 0, dur: 0.5 }; return; }
    move = Object.assign(m, { t: 0, wait: 0 });
    if (m.kind === 'foot') { move.dur = p.tFoot; move.from = { ...lim.foot[m.k] }; }
    else if (m.kind === 'hand') { move.dur = p.tHand; const hp = lim.hand[m.k] || pose.handOf(world, m.k === 0 ? P.handLeft : P.handRight); move.from = { x: hp.x, y: hp.y }; lim.grip[m.k] = false; }
    else move.dur = 0.5;
    pelvisFrom = { ...pelvisGoal };
  };
  const tmp = new Float64Array(OUT);
  ep.updateGrip = ((orig) => function () {
    const m = this.measure;
    if (!move) startMove();
    // a foot lifts only once the centre of mass is over the other foot (lifted on a timer, the body
    // was still 20 cm behind its standing heel and fell back); with a hand on the rock the hands hold it
    if (p.comGate && move.kind === 'foot' && !move.putDown && !move.lifted && move.t + 1 / HZ > move.dur * p.shift && !(lim.grip[0] || lim.grip[1])) {
      const sx = pose.soleOf(world, move.k === 0 ? P.footRight : P.footLeft).x;
      move.shiftWait = (move.shiftWait || 0) + 1 / HZ;
      if (Math.abs((m.centerX || 0) - sx) > (p.comTol || 0.06)) {
        if (move.shiftWait > (p.shiftMax || 2)) { stuck++; footFails++; move = null; orig(); return; }
      } else move.t += 1 / HZ;
    } else move.t += 1 / HZ;
    const s = Math.min(1, move.t / move.dur);
    const feet = lim.foot.map((f) => ({ x: f.x, y: f.y, angle: f.angle || 0 }));
    const hands = lim.hand.map((h) => (h ? { x: h.x, y: h.y } : null));
    const gripping = lim.grip.slice();
    const planted = [true, true];
    if (move.kind === 'foot') {
      // weight shift first, then lift: up over the highest ground in between, across, and down
      move.lifted = s > p.shift;
      if (move.lifted) {
        const q = (s - p.shift) / (1 - p.shift);
        const top = Math.max(groundMax(Math.min(move.from.x, move.target.x), Math.max(move.from.x, move.target.x)), move.from.y, move.target.y) + p.clear;
        const pts = [move.from, { x: move.from.x - 0.02, y: top }, { x: move.target.x, y: top }, move.target];
        const cuts = [0, 0.3, 0.75, 1];
        let si = 0;
        while (si < 2 && q > cuts[si + 1]) si++;
        const e = smooth(Math.min(1, (q - cuts[si]) / (cuts[si + 1] - cuts[si])));
        feet[move.k] = { x: pts[si].x + (pts[si + 1].x - pts[si].x) * e, y: pts[si].y + (pts[si + 1].y - pts[si].y) * e, angle: si === 2 ? (move.target.angle || 0) * e : 0 };
        planted[move.k] = false;
      }
    } else if (move.kind === 'hand') {
      // a hand goes forward and a little above its hold, then down onto it
      const pts = [move.from, { x: (move.from.x + move.target.x) / 2, y: Math.max(move.from.y, move.target.y) + 0.2 }, { x: move.target.x - 0.03, y: move.target.y + 0.1 }, move.target];
      const cuts = [0, 0.3, 0.75, 1];
      let si = 0;
      while (si < 2 && s > cuts[si + 1]) si++;
      const e = smooth(Math.min(1, (s - cuts[si]) / (cuts[si + 1] - cuts[si])));
      hands[move.k] = { x: pts[si].x + (pts[si + 1].x - pts[si].x) * e, y: pts[si].y + (pts[si + 1].y - pts[si].y) * e };
      gripping[move.k] = false;
    } else if (move.kind === 'release') {
      for (let k = 0; k < 2; k++) { lim.hand[k] = null; lim.grip[k] = false; hands[k] = null; gripping[k] = false; }
    }
    const goalNow = pelvisFor(move);
    const pr = smooth(Math.min(1, move.t / Math.max(0.2, move.kind === 'foot' ? move.dur * p.shift : move.dur * 0.6)));
    pelvisGoal = { x: pelvisFrom.x + (goalNow.x - pelvisFrom.x) * pr, y: pelvisFrom.y + (goalNow.y - pelvisFrom.y) * pr };
    // the trunk turns toward its lean at a limited rate (leaning over at once toppled it before the
    // hands were out)
    leanCur += Math.max(-p.leanRate / HZ, Math.min(p.leanRate / HZ, leanNow - leanCur));
    // the pelvis is led on a leash: a goal far from where the pelvis is turns into leg targets that kick
    // the planted feet off their holds (they slid back 30 cm while the hips lagged behind the plan)
    const pvA = pose.pelvisOf(world);
    const dxp = pelvisGoal.x - pvA.x, dyp = pelvisGoal.y - pvA.y, dp = Math.hypot(dxp, dyp), LEASH = p.leash || +(ENV.LEASH || 0.06);
    const pelvisLed = (lim.grip[0] || lim.grip[1]) && dp > LEASH ? { x: pvA.x + dxp * LEASH / dp, y: pvA.y + dyp * LEASH / dp } : pelvisGoal;
    // planted feet are held where they really are (a stale foot target in the loop feet-hands-rock made
    // the legs fight the arms: the planted foot slid, or lifted off)
    if (lim.grip[0] || lim.grip[1]) for (let k = 0; k < 2; k++) if (planted[k]) {
      const so = pose.soleOf(world, k === 0 ? P.footLeft : P.footRight);
      if (Math.hypot(so.x - feet[k].x, so.y - feet[k].y) > 0.03) { feet[k] = { x: so.x, y: so.y, angle: feet[k].angle }; if (move.kind !== 'foot' || move.k !== k) lim.foot[k] = { x: so.x, y: ground(so.x) + SOLE, angle: slopeAt(so.x) }; }
    }
    const goal = { pelvis: pelvisLed, torso: leanCur, feet, hands, gripping, planted, torsoGain: p.torsoGain };
    const actual = pose.actualOf(world);
    const support = move.kind === 'foot' ? lim.foot[1 - move.k].x : undefined;
    const jt = lim.grip[0] || lim.grip[1] ? pose.jointTargets(goal, actual) : pose.balancedTargets(goal, m, { planted, support, kHip: p.kHip, kAnkle: p.kAnkle, kAnkleVel: 0.4, kHipVel: 0.2 }, actual);
    pose.commandsFor(world, jt, tmp);
    if (ENV.DEBUG_PEL && ep.frame % 6 === 0) { const pv = pose.pelvisOf(world); console.log('   f', (ep.frame / HZ).toFixed(2), move.kind + (move.k != null ? move.k : ''), 's', s.toFixed(2), 'pel goal', pelvisGoal.x.toFixed(2), pelvisGoal.y.toFixed(2), 'act', pv.x.toFixed(2), pv.y.toFixed(2), 'torso', pv.angle.toFixed(2), 'lean', leanCur.toFixed(2), 'com', (m.centerX || 0).toFixed(2), 'feet', feet.map((f) => f.x.toFixed(2) + ',' + f.y.toFixed(2)).join(' '), 'planted', planted.join(','), 'sole', [P.footLeft, P.footRight].map((q) => { const so = pose.soleOf(world, q); return so.x.toFixed(2) + ',' + so.y.toFixed(2); }).join(' '), 'hand', [P.handLeft, P.handRight].map((q) => { const h = pose.handOf(world, q); return h.x.toFixed(2) + ',' + h.y.toFixed(2); }).join(' ')); }
    for (let j = 0; j < J; j++) this.action[j] = tmp[j];
    for (let k = 0; k < 2; k++) {
      let near = false;
      if (move.kind === 'hand' && move.k === k && s > 0.4) { const hp = pose.handOf(world, k === 0 ? P.handLeft : P.handRight); near = Math.hypot(hp.x - move.target.x, hp.y - move.target.y) < (p.gripNear || 0.18); }
      this.action[J + k] = lim.grip[k] || near ? 1 : -1;
    }
    if (move.kind === 'hand' && s > 0.4 && this.grips[move.k] && this.grips[move.k].joint) {
      // gripped on the way (the hand touched the rock near its hold): it stays there
      const hp = pose.handOf(world, move.k === 0 ? P.handLeft : P.handRight);
      lim.hand[move.k] = { x: hp.x, y: hp.y }; lim.grip[move.k] = true; move = null; stuck = 0; handFails = 0;
      orig(); return;
    }
    if (s >= 1) {
      if (move.kind === 'foot') {
        const so = pose.soleOf(world, move.k === 0 ? P.footLeft : P.footRight);
        if (Math.abs(so.y - move.target.y) < 0.08 && Math.abs(so.x - move.target.x) < 0.12) { lim.foot[move.k] = { x: so.x, y: ground(so.x) + SOLE, angle: slopeAt(so.x) }; if (!move.putDown) { stuck = 0; footFails = 0; } handFails = 0; move = null; }
        else if ((move.wait += 1 / HZ) > 0.4) {
          stuck++; footFails++;
          if (so.y - ground(so.x) - SOLE > 0.04 && !move.putDown) {
            // left hanging in the air (blocked by the rock): put it down where it is, then try a shorter step
            const k = move.k, x = Math.max(so.x - 0.05, Math.min(lim.foot[1 - k].x, so.x));
            move = { kind: 'foot', k, target: { x, y: ground(x) + SOLE, angle: slopeAt(x) }, putDown: true, t: 0, wait: 0, dur: 0.5, from: { x: so.x, y: so.y } };
            pelvisFrom = { ...pelvisGoal };
          } else { lim.foot[move.k] = { x: so.x, y: ground(so.x) + SOLE, angle: slopeAt(so.x) }; move = null; }
        }
      } else if (move.kind === 'hand') {
        const g = this.grips[move.k];
        if (g && g.joint) { const hp = pose.handOf(world, move.k === 0 ? P.handLeft : P.handRight); lim.hand[move.k] = { x: hp.x, y: hp.y }; lim.grip[move.k] = true; move = null; stuck = 0; handFails = 0; }
        else if ((move.wait += 1 / HZ) > 0.5) { lim.hand[move.k] = null; move = null; stuck++; handFails++; }
      } else move = null;
    }
    orig();
  })(ep.updateGrip.bind(ep));
  return { lim, ground, summitX, H, S, SOLE, get stuck() { return stuck; } };
}
