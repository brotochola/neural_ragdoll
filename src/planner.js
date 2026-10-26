// PLANNER (code, not a network): turns an obstacle into a sequence of body goals for the pose
// executor (src/pose.js): where the pelvis, each foot and each hand should be, and when to grip.
// Positions are relative to the obstacle, so every step is a reason a person could give
// ("hands on the edge", "right foot on top", "hips over the box"), and there is no room for the
// tricks the free searches found (diving head-first, lying with the feet on the box).
//
// A plan is a list of keys: { dur, pelvis:{x,y}, torso, feet:[L,R], hands:[L,R], planted:[L,R], grip }.
// A limb spec is {x,y} (absolute), 'hold' (stay where it is when the key starts), or null (hands only:
// let the arm hang). Keys blend smoothly (cosine) from the previous goal.
import { soleOf, handOf, pelvisOf, PARTS } from "./pose.js";

// CLIMB onto a box of height h whose near edge is at e, facing +x. p: tunable numbers (offsets).
// Geometry kept consistent: in the "support" position the shoulders are above the hands (arms ~0.5 m)
// and the pelvis hangs 0.52 m below the shoulders along the torso lean.
// Tall boxes (h > mantleAbove): hands above the edge, down onto the top (grip closes there, not on the
// side face), crouch, push with the legs into support on the arms, lead foot on top, rise, trail foot,
// stand. Low boxes: the same without the push: lead foot straight onto the top.
export function climbPlan(h, e, p = {}) {
  const q = Object.assign({
    settleT: 0.5,
    handX: 0.12, handAbove: 0.15, lean: -0.35, standBack: 0.30, reachT: 0.7, downT: 0.3,
    crouchY: 0.72, crouchT: 0.4,
    supportArm: 0.42, supportLean: -0.6, pushT: 0.45,
    tuckX: -0.10, footX: 0.12, tuckT: 0.5,
    liftX: -0.05, liftY: 0.15, liftT: 0.8, placeT: 0.6,
    riseX: 0.15, riseY: 0.78, riseLean: -0.45, riseT: 1.0,
    trailX: 0.33, trailLift: 0.2, trailT: 0.8,
    standX: 0.26, standT: 1.0,
    mantleAbove: 0.62,
  }, p);
  const top = h;
  const hand = { x: e + q.handX, y: top + 0.02 };
  const above = { x: hand.x, y: top + q.handAbove };
  const torsoLen = 0.52;
  const keys = [];
  // 0 SETTLE: stand where it is.
  keys.push({ dur: q.settleT, pelvis: 'hold', torso: 0, feet: ['hold', 'hold'], hands: [null, null], planted: [true, true], grip: false });
  // 1 REACH above the edge, 2 DOWN onto the top surface: the grip closes only there.
  keys.push({ dur: q.reachT, pelvis: { x: e - q.standBack, y: 0.9 }, torso: q.lean, feet: ['hold', 'hold'], hands: [above, above], planted: [true, true], grip: false });
  keys.push({ dur: q.downT, pelvis: { x: e - q.standBack, y: 0.9 }, torso: q.lean, feet: ['hold', 'hold'], hands: [hand, hand], planted: [true, true], grip: true });
  if (h > q.mantleAbove) {
    // 3 CROUCH to push.
    keys.push({ dur: q.crouchT, pelvis: { x: e - q.standBack + 0.05, y: q.crouchY }, torso: q.lean, feet: ['hold', 'hold'], hands: ['hold', 'hold'], planted: [true, true], grip: true });
    // 4 PUSH into support: shoulders above the hands, legs extend off the floor.
    const sh = { x: hand.x - 0.05, y: hand.y + q.supportArm };
    const supportPelvis = { x: sh.x + Math.sin(q.supportLean) * torsoLen, y: sh.y - Math.cos(q.supportLean) * torsoLen };
    keys.push({ dur: q.pushT, pelvis: supportPelvis, torso: q.supportLean, feet: ['hold', 'hold'], hands: ['hold', 'hold'], planted: [false, false], grip: true });
    // 5 TUCK the lead (right) foot onto the top, the other leg hangs.
    keys.push({ dur: q.tuckT, pelvis: { x: supportPelvis.x + 0.03, y: supportPelvis.y + 0.03 }, torso: q.supportLean, feet: [{ x: e + q.tuckX - 0.1, y: top - 0.35 }, { x: e + q.footX, y: top }], hands: ['hold', 'hold'], planted: [false, false], grip: true });
  } else {
    // 3 LIFT the lead foot over the edge, weight on the back foot and the hands; 4 PLACE it on top.
    keys.push({ dur: q.liftT, pelvis: { x: e - q.standBack - 0.05, y: 0.9 }, torso: q.lean, feet: ['hold', { x: e + q.liftX, y: top + q.liftY }], hands: ['hold', 'hold'], planted: [true, false], grip: true });
    keys.push({ dur: q.placeT, pelvis: { x: e - q.standBack + 0.05, y: 0.9 }, torso: q.lean, feet: ['hold', { x: e + q.footX, y: top }], hands: ['hold', 'hold'], planted: [true, false], grip: true });
  }
  // RISE: hips up over the lead foot; the hands press down (mantle).
  keys.push({ dur: q.riseT, pelvis: { x: e + q.riseX, y: top + q.riseY }, torso: q.riseLean, feet: ['hold', 'hold'], hands: ['hold', 'hold'], planted: [false, true], grip: true });
  // TRAIL foot over the edge and onto the top.
  keys.push({ dur: q.trailT * 0.5, pelvis: { x: e + q.riseX + 0.05, y: top + q.riseY + 0.05 }, torso: q.riseLean * 0.6, feet: [{ x: e + 0.05, y: top + q.trailLift }, 'hold'], hands: ['hold', 'hold'], planted: [false, true], grip: true });
  keys.push({ dur: q.trailT * 0.5, pelvis: { x: e + q.standX, y: top + 0.88 }, torso: q.riseLean * 0.3, feet: [{ x: e + q.trailX - 0.1, y: top }, 'hold'], hands: ['hold', 'hold'], planted: [false, true], grip: true });
  // STAND on top, let go, arms down.
  keys.push({ dur: q.standT, pelvis: { x: e + q.standX, y: top + 0.92 }, torso: 0, feet: ['hold', 'hold'], hands: [null, null], planted: [true, true], grip: false });
  return keys;
}

// Runs a plan on an episode: call step(ep, world) every frame; returns the current goal (or null when
// the plan is over) and the grip flag.
export function makePlanRunner(keys, world, hz) {
  const P = PARTS();
  let k = -1;
  let t = 0;
  let from = null;
  let to = null;
  const lerp = (a, b, u) => a + (b - a) * u;
  const resolve = (spec, current) => (spec === 'hold' ? { x: current.x, y: current.y, angle: 0 } : spec ? { x: spec.x, y: spec.y, angle: 0 } : null);
  function now() {
    const pv = pelvisOf(world);
    return {
      pelvis: { x: pv.x, y: pv.y }, torso: pv.angle,
      feet: P.feet.map((f) => { const s = soleOf(world, f); return { x: s.x, y: s.y, angle: 0 }; }),
      hands: P.hands.map((hd) => handOf(world, hd)),
    };
  }
  function start(i) {
    const cur = now();
    from = to || cur;
    const key = keys[i];
    to = {
      pelvis: key.pelvis === 'hold' ? cur.pelvis : key.pelvis, torso: key.torso,
      feet: key.feet.map((f, j) => resolve(f, cur.feet[j])),
      hands: key.hands.map((hd, j) => resolve(hd, cur.hands[j])),
      planted: key.planted, grip: key.grip,
    };
    k = i;
    t = 0;
  }
  return {
    get key() { return k; },
    step() {
      if (k < 0) start(0);
      t += 1 / hz;
      if (t > keys[k].dur) {
        if (k + 1 >= keys.length) return null;
        start(k + 1);
      }
      const u = 0.5 - 0.5 * Math.cos(Math.PI * Math.min(1, t / keys[k].dur));
      const g = {
        pelvis: { x: lerp(from.pelvis.x, to.pelvis.x, u), y: lerp(from.pelvis.y, to.pelvis.y, u) },
        torso: lerp(from.torso, to.torso, u),
        feet: [0, 1].map((j) => ({ x: lerp(from.feet[j].x, to.feet[j].x, u), y: lerp(from.feet[j].y, to.feet[j].y, u), angle: 0 })),
        hands: [0, 1].map((j) => (to.hands[j] ? { x: lerp((from.hands[j] || to.hands[j]).x, to.hands[j].x, u), y: lerp((from.hands[j] || to.hands[j]).y, to.hands[j].y, u) } : null)),
        planted: to.planted,
        grip: to.grip,
      };
      return g;
    },
  };
}
