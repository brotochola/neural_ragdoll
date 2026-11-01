// The trail course once it is in the world: obstacles for the jump selector (src/run-selector.js), the floor
// level (holes bridged) and the real ground height anywhere. Shared by trail-es.js and the playground.
//   course: makeTrailCourse(...) (x relative to the feet at the start); ep: the episode it was spawned in.
// Sets ep.floorAt and ep.groundAt (src/trail-net.js reads them) and returns { obstacles, floorAt, groundAt,
// features } with absolute x.
export function installTrail(ep, course, Box2D) {
  const mid = ep.terrain.statics[0].x0 - course.statics[0].dx;
  const levels = course.levels.map(([a, b, l]) => [a + mid, b + mid, l]);
  const features = course.features.map((f) => Object.assign({}, f, { x0: f.x0 + mid, x1: f.x1 + mid }));
  const balls = ep.terrain.balls || [];
  // the floor: the level of the piece under x; in a hole, the piece before it
  function floorAt(x) {
    let lv = levels.length ? levels[0][2] : 0;
    for (const [a, b, l] of levels) { if (x >= a) lv = l; if (x < b) break; }
    return lv;
  }
  // the ground: the floor where there is floor (0 in a hole), boxes and balls on it
  function groundAt(x) {
    let g = 0;
    for (const [a, b, l] of levels) if (x >= a && x <= b) { g = l; break; }
    for (const f of features) if (f.type === 'box' && x >= f.x0 && x <= f.x1) g = Math.max(g, f.top);
    for (const b of balls) { const q = Box2D.b2Body_GetPosition(b.id); if (Math.abs(x - q.x) < b.r) g = Math.max(g, q.y + Math.sqrt(b.r * b.r - (x - q.x) ** 2)); }
    return g;
  }
  const obstacles = [];
  for (const f of features) {
    if (f.type === 'hole') obstacles.push({ x0: f.x0, x1: f.x1, kind: 'agujero' });
    else if (f.type === 'up') obstacles.push({ x0: f.x0, x1: f.x0 + 0.02, top: f.to, kind: 'escalón' });
    else if (f.type === 'box') obstacles.push({ x0: f.x0, x1: f.x1, top: f.top, kind: 'caja' });
    else if (f.type === 'ball') { const b = balls[f.ball]; if (b) obstacles.push({ live: () => { const q = Box2D.b2Body_GetPosition(b.id); return { x0: q.x - b.r, x1: q.x + b.r, top: q.y + b.r, kind: 'bola' }; } }); }
  }
  ep.floorAt = floorAt; ep.groundAt = groundAt;
  return { obstacles, floorAt, groundAt, features };
}
