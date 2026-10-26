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
