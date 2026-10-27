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
