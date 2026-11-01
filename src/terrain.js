// Terrain generators shared by the trainers and the in-browser playground.

// A jagged line from flat ground up to a summit plateau, x strictly increasing (each piece a convex
// trapezoid down to below the floor), relative to the start of the feet.
export function makeMountain(r, H = 3, d = 0.5) {
  const pts = [[-4, 0], [1.2, 0]];
  let x = 1.2, y = 0;
  while (y < H) {
    const u = r();
    const pSteep = 0.1 + 0.2 * d, pStep = 0.2 + 0.2 * d, pSlope = 0.35 - 0.1 * d;
    let dx, dy;
    if (u < pSteep) { const a = (55 + r() * 20) * Math.PI / 180; const L = 0.25 + r() * 0.3; dx = L * Math.cos(a); dy = L * Math.sin(a); }
    else if (u < pSteep + pStep) { dx = 0.04 + r() * 0.04; dy = 0.12 + r() * (0.15 + 0.15 * d); }
    else if (u < pSteep + pStep + pSlope) { const a = (18 + r() * 22) * Math.PI / 180; const L = 0.25 + r() * 0.35; dx = L * Math.cos(a); dy = L * Math.sin(a); }
    else { dx = 0.25 + r() * 0.35; dy = dx * Math.tan((-4 + r() * 12) * Math.PI / 180); }
    if (y + dy > H) dy = H - y;
    x += dx; y = Math.max(0, y + dy);
    pts.push([+x.toFixed(3), +y.toFixed(3)]);
  }
  pts.push([x + 5, y]);
  const polys = [];
  for (let i = 1; i < pts.length - 1; i++) {
    const [x1, y1] = pts[i], [x2, y2] = pts[i + 1];
    if (y1 <= 0.001 && y2 <= 0.001) continue;
    polys.push([[x1, -0.2], [x2, -0.2], [x2, y2], [x1, y1]]);
  }
  return { surface: pts, polys, summitX: x, H: y };
}

// The climbing wall: a block with rungs on its face (as in climb-wall.js), relative to the feet.
export function makeWall(wallDx = 0.65, wallH = 3.3, spacing = 0.3) {
  const holds = [];
  for (let y = spacing; y < wallH - 0.25; y += spacing) holds.push({ dx: wallDx - 0.1, y: +y.toFixed(3), w: 0.1, h: 0.04 });
  return { statics: [{ dx: wallDx, w: 3, h: wallH }], holds };
}

// THE CRAG: a steep, irregular rock face (relative to the feet): a jagged near-vertical line leaning back
// a little as it rises (each 0.22-0.36 m tier shifted -0.02..+0.06 m), with a rock knob sticking out at
// the top of each tier (0.08-0.12 m out, flat on top) for hands and feet, up to a summit plateau at H.
// Tiers and knobs are convex quads (overlapping, rigid, grippable anywhere). Returns the top line
// (surface), polys, knobs ({lip, back, y}: lip = outer end, back = the face behind it) and the summit edge.
export function makeCrag(r, H = 2.6, x0 = 0.6, opts = {}) {
  const tMin = opts.tierMin || 0.22, tMax = opts.tierMax || 0.36;
  const polys = [], knobs = [], tiers = [];
  let x = x0, y = 0;
  while (y < H - 1e-6) {
    let h = tMin + r() * (tMax - tMin);
    if (H - (y + h) < tMin * 0.7) h = H - y;
    const lean = -0.01 + r() * 0.04;
    tiers.push({ xb: x, xt: x + lean, y0: y, y1: y + h });
    y += h;
    x += lean + (y < H - 1e-6 ? -0.02 + r() * 0.06 : 0);
  }
  const end = x + (opts.plateau || 9);
  const surface = [[-4, 0]];
  for (let i = 0; i < tiers.length; i++) {
    const t = tiers[i];
    polys.push([[t.xb, -0.2], [end, -0.2], [end, t.y1], [t.xt, t.y1]]);
    surface.push([t.xb, t.y0], [t.xt, t.y1]);
    if (i < tiers.length - 1) {
      // the knob: on the face of the next tier, just above this tier's top, sticking out toward the climber
      const back = Math.max(t.xt, tiers[i + 1].xb), ky = t.y1, out = 0.08 + r() * 0.04;
      polys.push([[back - out, ky], [back - out + 0.03, ky - 0.04], [back + 0.03, ky - 0.07], [back + 0.03, ky]]);
      knobs.push({ lip: back - out, back, y: ky });
    }
  }
  surface.push([end, H]);
  const last = tiers[tiers.length - 1];
  return { surface: surface.map(([a, b]) => [+a.toFixed(3), +b.toFixed(3)]), polys, knobs, edgeX: last.xt, H, summitX: last.xt };
}

// the course, relative to the feet: floor, a hole W wide 3-4 m ahead, a rope over its middle (the
// anchor ropeTop above the floor, the rope hanging to ropeLow above it), floor on after it
export function makeRopeCourse(r, opts = {}) {
  const LIFT = opts.lift || 0.6;
  const gmin = opts.gapMin || 1.6, gmax = opts.gapMax || 2.4;
  const x0 = 3 + r() * 1.0, w = gmin + r() * (gmax - gmin);
  const top = 4.2 + r() * 0.3, low = (opts.low || 1.65) + r() * 0.15;
  return {
    statics: [{ dx: -4, w: 4 + x0, h: LIFT }, { dx: x0 + w, w: 30, h: LIFT }],
    hole: { x0, x1: x0 + w },
    ropes: [{ dx: x0 + w * (opts.ropeAt || 0.5), top: LIFT + top, length: top - low, seg: opts.seg || 0.3, density: opts.ropeDensity || 120 }],
    lift: LIFT,
  };
}


// DASH course, relative to the feet: flat floor, then random obstacles every gapMin-gapMax metres from
// `start` on, out to `length`: boxes (static, hMin-hMax tall, wMin-wMax wide) or balls (loose, rMin-rMax
// radius) with probability ballP. Returns { statics: [{dx, w, h}], balls: [{dx, r, density}] }.
export function makeDashCourse(r, opts = {}) {
  const o = Object.assign({ start: 4, length: 70, ballP: 0.35, hMin: 0.08, hMax: 0.35, wMin: 0.2, wMax: 1.0, rMin: 0.08, rMax: 0.18, gapMin: 2.5, gapMax: 5, density: 150 }, opts);
  const statics = [], balls = [];
  let x = o.start + r() * 1.5;
  while (x < o.length) {
    if (r() < o.ballP) {
      const rad = o.rMin + r() * (o.rMax - o.rMin);
      balls.push({ dx: +(x + rad).toFixed(3), r: +rad.toFixed(3), density: o.density });
      x += 2 * rad;
    } else {
      const w = o.wMin + r() * (o.wMax - o.wMin), h = o.hMin + r() * (o.hMax - o.hMin);
      statics.push({ dx: +x.toFixed(3), w: +w.toFixed(3), h: +h.toFixed(3) });
      x += w;
    }
    x += o.gapMin + r() * (o.gapMax - o.gapMin);
  }
  return { statics, balls };
}

// TRAIL course ("terreno libre"), relative to the feet: a raised floor whose level changes, broken by holes,
// with boxes and balls on it. Starts at level `start` (0.6 m) with `lead` metres of flat floor, then random
// features every gapMin-gapMax metres out to `length`:
//   hole   no floor for holeMin-holeMax metres (the level after it within +-0.15)
//   up     the floor steps up upMin-upMax (capped at maxLevel)
//   down   the floor steps down downMin-downMax (not under minLevel)
//   box    a static box hMin-hMax tall, wMin-wMax wide on the floor
//   ball   a loose ball rMin-rMax on the floor
// Returns { statics: [{dx, w, h}] (floor pieces as columns from the ground, boxes as columns from the ground
// to their top), balls: [{dx, r, density, y}], features: [{type, x0, x1, top?, from, to}] (x relative,
// heights absolute; top = what to clear, from/to = floor level before and after), levels: [[x0, x1, level]]
// (the floor, holes left out) }.
export function makeTrailCourse(r, opts = {}) {
  const o = Object.assign({ start: 0.6, lead: 4, length: 70, minLevel: 0.3, maxLevel: 1.6, gapMin: 2.5, gapMax: 4.5,
    pHole: 0.25, pUp: 0.2, pDown: 0.2, pBox: 0.2, pBall: 0.15,
    holeMin: 0.4, holeMax: 1.0, upMin: 0.12, upMax: 0.4, downMin: 0.15, downMax: 0.5,
    hMin: 0.08, hMax: 0.3, wMin: 0.2, wMax: 0.8, rMin: 0.08, rMax: 0.16, density: 150, tail: 40, from: -4 }, opts);
  const statics = [], balls = [], features = [], levels = [], polys = [];
  let lv = o.start, x = o.from, segStart = o.from;
  // a floor piece (none needed on the ground itself, level 0)
  const close = (xe) => { if (xe > segStart + 1e-3) { if (lv > 0.005) statics.push({ dx: +segStart.toFixed(3), w: +(xe - segStart).toFixed(3), h: +lv.toFixed(3) }); levels.push([segStart, xe, lv]); } };
  x = o.lead + r() * 1.0;
  const pick = () => { const t = o.pHole + o.pUp + o.pDown + o.pBox + o.pBall; let u = r() * t; for (const [k, pk] of [['hole', o.pHole], ['up', o.pUp], ['down', o.pDown], ['box', o.pBox], ['ball', o.pBall]]) { if ((u -= pk) <= 0) return k; } return 'ball'; };
  while (x < o.length) {
    let type = pick();
    if (type === 'up' && lv + o.upMin > o.maxLevel) type = 'down';
    if (type === 'down' && lv - o.downMin < o.minLevel) type = 'up';
    if (type === 'hole') {
      const w = o.holeMin + r() * (o.holeMax - o.holeMin);
      close(x);
      const to = Math.max(o.minLevel, Math.min(o.maxLevel, lv + (r() * 2 - 1) * 0.15));
      const f = { type, x0: x, x1: x + w, from: lv, to };
      // o.knobs: ledges on the pit's far wall every 0.3 m from its foot (a fall into the pit can catch one and
      // climb out the way the rock is climbed); no random draws, so the course is the same either way
      if (o.knobs) {
        f.knobs = [];
        const back = x + w;
        // spaced like the rock's ledges, the highest a quarter metre under the edge (room for the hands over it)
        for (let ky = to - 0.25 - 0.29 * Math.floor((to - 0.5) / 0.29); ky <= to - 0.25 + 1e-6; ky += 0.29) {
          polys.push([[back - 0.1, ky], [back - 0.07, ky - 0.04], [back + 0.03, ky - 0.07], [back + 0.03, ky]]);
          f.knobs.push({ lip: back - 0.1, back, y: +ky.toFixed(3) });
        }
      }
      features.push(f);
      x += w; lv = to; segStart = x;
    } else if (type === 'up' || type === 'down') {
      const d = type === 'up' ? o.upMin + r() * (o.upMax - o.upMin) : -(o.downMin + r() * (o.downMax - o.downMin));
      const to = Math.max(o.minLevel, Math.min(o.maxLevel, lv + d));
      close(x);
      features.push({ type, x0: x, x1: x + 0.02, top: Math.max(lv, to), from: lv, to });
      lv = to; segStart = x;
    } else if (type === 'box') {
      const w = o.wMin + r() * (o.wMax - o.wMin), h = o.hMin + r() * (o.hMax - o.hMin);
      statics.push({ dx: +x.toFixed(3), w: +w.toFixed(3), h: +(lv + h).toFixed(3) });
      features.push({ type, x0: x, x1: x + w, top: lv + h, from: lv, to: lv });
      x += w;
    } else {
      const rad = o.rMin + r() * (o.rMax - o.rMin);
      balls.push({ dx: +(x + rad).toFixed(3), r: +rad.toFixed(3), density: o.density, y: +lv.toFixed(3) });
      features.push({ type, x0: x, x1: x + 2 * rad, top: lv + 2 * rad, from: lv, to: lv, ball: balls.length - 1 });
      x += 2 * rad;
    }
    x += o.gapMin + r() * (o.gapMax - o.gapMin);
  }
  close(x + o.tail);
  return { statics, balls, features, levels, polys, lift: o.start, end: x + o.tail, endLevel: lv };
}

// PRINCE OF PERSIA level, relative to the feet: trail A (raised floor, holes, steps, boxes, balls), a drop to
// the ground, a 2 m rock to leap at, grab and climb, then on its top trail B at ~2 m (holes are 2 m pits),
// with a wide hole crossed on a rope, and trail C after it. Returns the pieces in one terrain plus the zones
// the director (src/director.js) switches skills on:
//   { statics, balls, polys, surface, ropes, crag, features, levels, zones: { wall: {x0, summit, plateauEnd},
//     rope: {x0, x1, lift} }, lift }
export function makePopLevel(r, opts = {}) {
  const o = Object.assign({ lenA: 10, lenB: 10, lenC: 30, wallH: 2.0, knobs: true }, opts);
  const shift = (c, dx) => {
    c.statics.forEach((b) => (b.dx = +(b.dx + dx).toFixed(3)));
    c.balls.forEach((b) => (b.dx = +(b.dx + dx).toFixed(3)));
    c.features.forEach((f) => { f.x0 += dx; f.x1 += dx; if (f.knobs) f.knobs.forEach((k) => { k.lip += dx; k.back += dx; }); });
    c.polys.forEach((q) => q.forEach((pt) => (pt[0] += dx)));
    c.levels.forEach((l) => { l[0] += dx; l[1] += dx; });
    c.end += dx;
    return c;
  };
  // A on the ground: boxes, balls and low steps, no holes (they need a raised floor), so the way down to the rock is short
  const A = makeTrailCourse(r, { start: 0, minLevel: 0, maxLevel: 0.45, pHole: 0, pBox: 0.3, pBall: 0.2, length: o.lenA, tail: 3 });
  const statics = A.statics.slice(), balls = A.balls.slice(), features = A.features.slice(), levels = A.levels.slice();
  // down to the ground in steps of at most 0.3 m every 2.5 m (the trail runner learned drops up to 0.5 m),
  // a few metres of ground, then the rock
  let downX = A.end, lvD = A.endLevel;
  while (lvD > 0.32) {
    const to = +(lvD - 0.3).toFixed(3);
    features.push({ type: 'down', x0: downX, x1: downX + 0.02, top: lvD, from: lvD, to });
    statics.push({ dx: +downX.toFixed(3), w: 2.5, h: to }); levels.push([downX, downX + 2.5, to]);
    downX += 2.5; lvD = to;
  }
  features.push({ type: 'down', x0: downX, x1: downX + 0.02, top: lvD, from: lvD, to: 0 });
  const wallX0 = downX + 7.5 + r() * 2; // the zone of the rock starts 4 m before it: the drop is the trail runner's
  // a 10 m top: crawling over the edge and getting up can take several metres (5 m left a 2 m pit in reach)
  const crag = makeCrag(r, o.wallH, wallX0, { plateau: 10 });
  levels.push([downX, wallX0, 0]);
  const summit = crag.summitX, plateauEnd = summit + 10;
  // the rock as a floor piece for the level's floor line: its top from the summit edge on
  levels.push([summit, plateauEnd, o.wallH]);
  // trail B on the top, a rope over a wide hole, trail C
  // B ends in 6 m of flat floor before the rope's hole: the rope runner's run-up (it knows no obstacles)
  const B = shift(makeTrailCourse(r, { start: o.wallH, minLevel: o.wallH - 0.3, maxLevel: o.wallH + 0.5, lead: 2, length: o.lenB, tail: 6, pHole: 0.2, from: 0, knobs: o.knobs }), plateauEnd);
  const holeW = 1.4 + r() * 0.8, hx0 = B.end, hx1 = hx0 + holeW, lvR = B.endLevel;
  const ropeTop = lvR + 4.2 + r() * 0.3, ropeLow = lvR + 1.25 + r() * 0.15;
  const C = shift(makeTrailCourse(r, { start: lvR, minLevel: o.wallH - 0.3, maxLevel: o.wallH + 0.5, lead: 4, length: o.lenC, tail: 40, from: 0, knobs: o.knobs }), hx1);
  for (const c of [B, C]) { statics.push(...c.statics); balls.push(...c.balls); features.push(...c.features); levels.push(...c.levels); }
  const rh = { type: 'ropehole', x0: hx0, x1: hx1, from: lvR, to: C.levels.length ? C.levels[0][2] : lvR };
  // ledges on the rope hole's far wall too: a body that falls in can climb out (o.knobs)
  const ropeKnobPolys = [];
  if (o.knobs) {
    rh.knobs = [];
    for (let ky = rh.to - 0.25 - 0.29 * Math.floor((rh.to - 0.5) / 0.29); ky <= rh.to - 0.25 + 1e-6; ky += 0.29) {
      ropeKnobPolys.push([[hx1 - 0.1, ky], [hx1 - 0.07, ky - 0.04], [hx1 + 0.03, ky - 0.07], [hx1 + 0.03, ky]]);
      rh.knobs.push({ lip: hx1 - 0.1, back: hx1, y: +ky.toFixed(3) });
    }
  }
  features.push(rh);
  features.sort((a, b) => a.x0 - b.x0);
  levels.sort((a, b) => a[0] - b[0]);
  const ropes = [{ dx: hx0 + holeW / 2, top: ropeTop, length: ropeTop - ropeLow, seg: 0.3, density: 120 }];
  return {
    statics, balls, polys: crag.polys.concat(B.polys, C.polys, ropeKnobPolys), surface: crag.surface, ropes, crag, features, levels, lift: A.lift,
    zones: { wall: { x0: wallX0, summit, plateauEnd, H: o.wallH }, rope: { x0: hx0, x1: hx1, lift: lvR }, down: downX },
  };
}
