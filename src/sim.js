// Ragdoll body, sensing, and one episode. The network and the reward live in their own files.
import { config, activeTrainingPhase } from "../config.js";
import {
  scoreDecision,
  scoreTotal,
  rememberAxisErrors,
  isStanding,
  onMark,
  markError,
  targetGain,
  targetSpan,
} from "./reward.js";
import { ElmanNetwork, ResidualNet } from "./net.js";
import { clamp, makeSeededRandom, randomNormalSample } from "./utils.js";
import { RagDoll2D, faceRaySpecs, facePartName } from "./ragdoll.js";

export const jointNames = [
  "Waist",
  "Neck",
  "Shoulder L",
  "Elbow L",
  "Wrist L",
  "Hip L",
  "Knee L",
  "Ankle L",
  "Shoulder R",
  "Elbow R",
  "Wrist R",
  "Hip R",
  "Knee R",
  "Ankle R",
];

export const observationLabels = buildObservationLabels();

function episodeTargets(options) {
  const source = options && options.targets;
  return source && source.length === config.targets.length
    ? source
    : config.targets;
}

function buildObservationLabels() {
  const labels = [];
  for (const name of jointNames)
    labels.push({
      name: "angle " + name,
      tip: "Joint angle of " + name + ", scaled by its range.",
    });
  for (const name of jointNames)
    labels.push({
      name: "rate " + name,
      tip: "How fast " + name + " is bending.",
    });
  labels.push(
    {
      name: "torso sin",
      tip: "Sine of the torso angle. The sign is face up versus face down.",
    },
    {
      name: "torso cos",
      tip: "Cosine of the torso. +1 upright, 0 on its side, negative upside down.",
    },
    { name: "head sin", tip: "Sine of the head angle." },
    { name: "head cos", tip: "Cosine of the head. +1 is upright." },
    { name: "head height", tip: "Head height, scaled by a standing reach." },
    {
      name: "torso height",
      tip: "Torso height, scaled by the standing blueprint.",
    },
    {
      name: "mass x",
      tip: "Center of mass relative to the midpoint of the feet.",
    },
    { name: "mass y", tip: "Height of the center of mass." },
    { name: "mass vx", tip: "Horizontal speed of the center of mass." },
    { name: "mass vy", tip: "Vertical speed of the center of mass." },
  );
  // Speed of each commanded part. Position error alone cannot tell a part flying
  // toward its mark from one crawling toward it, so the net cannot brake.
  for (const spec of config.targets) {
    labels.push({
      name: spec.id + " vx",
      tip:
        "Horizontal speed of the " +
        spec.id +
        " in meters per second. Full scale at " +
        config.body.velocityScale +
        " m/s.",
    });
    labels.push({
      name: spec.id + " vy",
      tip: "Vertical speed of the " + spec.id + ", same scale.",
    });
  }
  const partNames = [];
  for (const name of Object.keys(config.partIndex))
    partNames[config.partIndex[name]] = name;
  for (let i = 0; i < config.net.partCount; i++) {
    if (i === config.partIndex.torso) continue;
    labels.push({
      name: partNames[i] + " x",
      tip:
        "Horizontal position of the " +
        partNames[i] +
        " in the torso frame, in meters. The pelvis is the origin.",
    });
    labels.push({
      name: partNames[i] + " y",
      tip:
        "Vertical position of the " +
        partNames[i] +
        " in the torso frame, in meters. The pelvis is the origin.",
    });
  }
  for (const spec of faceRaySpecs) {
    const reach = spec.vision ? spec.reach : config.body.bodyRayMeters;
    labels.push({
      name: "ray " + facePartName[spec.part] + " " + spec.name,
      tip: spec.vision
        ? "Head view. 1 is clear out to " +
          reach +
          " m, 0 is a hit on the head. The floor and thrown boxes and balls all count."
        : "Hit fraction out of the " +
          spec.name +
          " face, within " +
          reach +
          " m. 1 is clear, 0 is a hit at the face.",
    });
  }
  for (const spec of config.targets) {
    labels.push(
      {
        name: "use " + spec.id,
        tip:
          "1 when the " +
          spec.id +
          " target is on, 0 when it is off. Its own input. Not packed with the other targets.",
      },
      {
        name: spec.id + " dx",
        tip:
          "Horizontal distance from the " +
          spec.id +
          " to its target, divided by " +
          (spec.id === "footR" || spec.id === "footL"
            ? "the foot span"
            : "the target span") +
          ". 0 while that target is off.",
      },
      {
        name: spec.id + " dy",
        tip:
          "Vertical distance from the " +
          spec.id +
          " to its target, divided by " +
          (spec.id === "footR" || spec.id === "footL"
            ? "the foot span"
            : "the target span") +
          ". 0 while that target is off.",
      },
      {
        name: spec.id + " esin",
        tip:
          "Sine of the angle error of the " +
          spec.id +
          ". 0 while that target is off, and 0 when it already faces the target.",
      },
      {
        name: spec.id + " ecos",
        tip:
          "Cosine of the angle error of the " +
          spec.id +
          ". 1 when it faces the target. 0 while that target is off.",
      },
      {
        name: spec.id + " wide dx",
        tip:
          "The same horizontal error, divided by " +
          config.reward.wideSpanMeters +
          " m, so a fall does not saturate it. Clamped at 2.",
      },
      {
        name: spec.id + " wide dy",
        tip:
          "The same vertical error, divided by " +
          config.reward.wideSpanMeters +
          " m. Clamped at 2.",
      },
    );
  }
  labels.push(
    {
      name: "head contact",
      tip: "1 when the head is touching the floor or a thrown object. A corner counts. Otherwise 0.",
    },
    {
      name: "footL contact",
      tip: "How hard the left foot is pressed, from 0 to 1. Not a penalty.",
    },
    {
      name: "footR contact",
      tip: "How hard the right foot is pressed, from 0 to 1. Not a penalty.",
    },
  );
  for (const group of config.contact.groups)
    labels.push({
      name: group.id + " contact",
      tip:
        "How hard " +
        group.parts.join(", ") +
        " is pressed into the floor or a thrown object, from 0 to 1. The same amount as that reward row, before the weight. Hands and feet are not in any group.",
    });
  if (config.net.rewardTotalNow)
    labels.push({
      name: "reward total",
      tip: "Sum of the reward rows at this decision, scaled like the other reward inputs.",
    });
  if (config.net.rewardTotalPrevious)
    labels.push({
      name: "reward total previous",
      tip: "Reward total from the previous decision, one action interval ago. Zero before the first decision.",
    });
  if (config.net.rewardTotalNext)
    labels.push({
      name: "reward total next",
      tip: "Reward total if every part kept its current speed for one physics frame. No collision or motor.",
    });
  if (config.net.walkInputCount) {
    labels.push({ name: "gait clock sin", tip: "Walk level only: sine of a clock with period config.walk.periodSeconds. Zero otherwise." });
    labels.push({ name: "gait clock cos", tip: "Walk level only: cosine of the same clock. Zero otherwise." });
    labels.push({ name: "walk command", tip: "Walk level only: commanded speed with its direction, over config.walk.maxSpeed. Zero otherwise." });
    labels.push({ name: "box ahead", tip: "Climb level only: 1 while there is a box. Zero otherwise." });
    labels.push({ name: "box distance", tip: "Climb level only: near edge of the box minus the feet midpoint, in metres, clamped to -1..1." });
    labels.push({ name: "box height", tip: "Climb level only: height of the box top over 0.5 m." });
    labels.push({ name: "grip left", tip: "1 while the left hand is holding something." });
    labels.push({ name: "grip right", tip: "1 while the right hand is holding something." });
  }
  if (labels.length !== config.net.inputCount)
    throw new Error("observation labels " + labels.length);
  return labels;
}

export const thrownKinds = [
  { name: "Ball 4.5 kg", radius: 0.12, density: 100, color: 0xff6b6b },
  { name: "Box 9 kg", box: 1, half: 0.15, density: 100, color: 0xc08a4b },
  { name: "Weight 25 kg", radius: 0.16, density: 310, color: 0x9aa3b8 },
];

function targetSlot(id) {
  for (let i = 0; i < config.targets.length; i++)
    if (config.targets[i].id === id) return i;
  return 0;
}

function signed(randomSource, span) {
  return (randomSource() * 2 - 1) * span;
}

function ensureTargets(out) {
  let list = out.targets;
  if (!list || list.length !== config.targets.length) {
    list = [];
    for (let i = 0; i < config.targets.length; i++) {
      list.push({ on: true, x: 0, y: 0, angle: 0, dx: 0, dy: 0, turn: 0 });
    }
    out.targets = list;
  }
  return list;
}

// The feet are born on top of each other. The mark used to separate them so the reward and standing agreed.
function fillStandFeet(list, randomSource) {
  const half = config.skill.stanceMeters;
  const slip = config.start.targetJitterMeters;
  const turn = config.start.targetJitterRadians;
  const right = list[targetSlot("footR")];
  const left = list[targetSlot("footL")];
  right.dx = half + signed(randomSource, slip);
  left.dx = -half + signed(randomSource, slip);
  right.dy = 0;
  left.dy = 0;
  right.turn = signed(randomSource, turn);
  left.turn = signed(randomSource, turn);
}

// A short step of both feet, the same draw for both twins. The head follows the midpoint.
function fillPlaceFeet(out, list, randomSource, scale, shiftMeters) {
  const skill = config.skill;
  const half = skill.placeStanceMeters || skill.stanceMeters;
  // Both feet shift together. scale < 1 shortens the step. The official eval passes 1.
  const s = scale == null ? 1 : scale;
  const extra =
    shiftMeters > 0
      ? shiftMeters
      : s *
        (skill.placeMinMeters +
          randomSource() * (skill.placeMaxMeters - skill.placeMinMeters));
  const rolled = randomSource() < 0.5 ? -1 : 1;
  // The official long step only goes toward +x. A fixed shift keeps its sign, so 0.12 and 0.16 stay put.
  const sign = shiftMeters > 0 ? rolled : 1;
  const right = list[targetSlot("footR")];
  const left = list[targetSlot("footL")];
  right.dx = half;
  left.dx = -half;
  right.dy = 0;
  left.dy = 0;
  right.turn = 0;
  left.turn = 0;
  out.place = {
    foot: targetSlot(sign < 0 ? "footL" : "footR"),
    extra: sign * extra,
    turn: 0,
  };
}

// Alternating steps. The plan is fixed at the start: reset does not draw another.
function fillWalk(out, list, randomSource, boxes) {
  const skill = config.skill;
  const dir = randomSource() < 0.5 ? -1 : 1;
  const step =
    skill.stepMinMeters +
    randomSource() * (skill.stepMaxMeters - skill.stepMinMeters);
  const half = skill.placeStanceMeters || skill.stanceMeters;
  const rightIndex = targetSlot("footR");
  const leftIndex = targetSlot("footL");
  let posR = half;
  let posL = -half;
  const firstRight = randomSource() < 0.5;
  const count = Math.min(skill.planSteps | 0, 4);
  const steps = [];
  const boxList = [];
  for (let i = 0; i < count; i++) {
    const right = firstRight ? i % 2 === 0 : i % 2 === 1;
    const prev = right ? posR : posL;
    if (right) posR += dir * step;
    else posL += dir * step;
    const next = right ? posR : posL;
    steps.push({
      foot: right ? rightIndex : leftIndex,
      dx: next,
      dy: 0,
      turn: signed(randomSource, skill.stepTurnRadians),
    });
    if (boxes) {
      boxList.push({
        dx: prev + (next - prev) * skill.boxAlong,
        halfW: skill.boxHalfWidthMeters,
        halfH: skill.boxHeightMeters / 2,
        dir,
      });
    }
  }
  out.steps = steps;
  out.boxes = boxes ? boxList : null;
  const right = list[rightIndex];
  const left = list[leftIndex];
  right.dx = firstRight ? steps[0].dx : half;
  left.dx = firstRight ? -half : steps[0].dx;
  right.dy = 0;
  left.dy = 0;
  right.turn = firstRight ? steps[0].turn : 0;
  left.turn = firstRight ? 0 : steps[0].turn;
}

function maybeCrouch(out, mode, randomSource, wantsFoot) {
  if (!wantsFoot) return;
  if (mode && mode.eval) return;
  if (mode && mode.full && !(mode.seconds > 0)) return;
  if (!(config.skill.crouchChance > 0) || randomSource() >= config.skill.crouchChance)
    return;
  out.crouchKnee = config.skill.crouchKneeRadians;
  out.settle = Math.max(out.settle || 0, 1);
}

export function fillTrainingStart(out, randomSource, mode) {
  const level = mode && mode.level ? mode.level | 0 : config.skill.level | 0;
  const full = !!(mode && mode.full);
  // mode.seconds wins. That is the training duration ramp: start short and stretch.
  // config.episode.seconds is not changed globally, because a global that moves every
  // generation leaks into another level.
  const seconds =
    mode && mode.seconds > 0
      ? mode.seconds
      : full
        ? config.episode.seconds
        : config.skill.trainSeconds || config.episode.seconds;
  out.push = 0;
  out.amp = 0;
  out.period = config.body.platformPeriodSeconds;
  out.phaseShift = 0;
  out.frames = Math.max(1, Math.round(seconds * config.physics.hz));
  out.task = "live";
  out.pose = "up";
  out.failOnHead = true;
  out.settle = 0;
  out.crouchKnee = 0;
  out.ballFrame = null;
  out.goalX = null;
  out.skill = level;
  out.steps = null;
  out.boxes = null;
  out.place = null;
  out.placeAssist =
    mode && mode.placeAssist != null
      ? mode.placeAssist
      : full
        ? 0
        : config.skill.placeAssist || 0;
  out.tilt = signed(randomSource, config.start.uprightTiltRadians);
  out.drop = config.start.uprightDropMeters;
  out.tilt += signed(randomSource, config.episode.startTiltJitterRadians);
  out.drop += randomSource() * config.episode.startDropJitterMeters;
  out.shiftX = signed(randomSource, config.episode.startShiftMeters);
  // Stance width is on the POSE, not in the reward. The blueprint puts both feet on the
  // same x, and that gap used to be invented on the MARK (fillStandFeet put +-stance on
  // dx), which leaked in as a step command: skill 1 is called STAND, and the right foot
  // was asked to step 12 cm in the same episode it was asked not to fall. The feet are
  // worth 6 of 12 points. The feet are now born apart, and the mark is planted where they
  // are: on skill 1 the foot row means "do not move the foot", and on skill 4 it means the step.
  out.stance = config.skill.stanceMeters || 0;
  const placed = mode && Array.isArray(mode.placed) ? mode.placed : null;
  if (placed && mode.randomStart === false) {
    out.tilt = 0;
    out.drop = config.start.uprightDropMeters;
    out.shiftX = 0;
  }
  if (placed) {
    out.placed = true;
    out.commands = "all";
    const list = ensureTargets(out);
    const jitter = mode.jitter > 0 ? mode.jitter : 0;
    const turn = jitter > 0 ? config.start.targetJitterRadians : 0;
    for (let i = 0; i < list.length; i++) {
      const mark = placed[i] || {};
      const target = list[i];
      target.on = !!mark.on;
      target.x = (mark.x || 0) + (jitter ? signed(randomSource, jitter) : 0);
      target.y = (mark.y || 0) + (jitter ? signed(randomSource, jitter) : 0);
      target.angle = (mark.angle || 0) + (turn ? signed(randomSource, turn) : 0);
      target.cx = 0;
      target.cy = 0;
      target.ca = target.angle;
    }
    let footOn = false;
    for (let i = 0; i < list.length; i++) {
      if (list[i].on && !config.targets[i].ride) footOn = true;
    }
    maybeCrouch(out, mode, randomSource, footOn);
    return out;
  }
  out.placed = false;
  // Which target group gets a real command at this rung of the ladder.
  out.commands =
    mode && mode.commands != null
      ? mode.commands
      : config.commands[level] || "none";
  // MIX: a stand or a walk, drawn here so both twins of a pair get the same task.
  if (out.commands === "mix")
    out.commands = randomSource() < (config.walk.mixWalkShare ?? 0.5) ? "walk" : "none";
  // CLIMB training mixes in walks and stands, so the earlier skills are not forgotten.
  // The eval (mode.eval) is always a climb.
  if (out.commands === "highbox" && !(mode && mode.eval)) {
    const roll = randomSource();
    const h = config.highbox;
    out.commands = roll < h.mixHighbox ? "highbox" : roll < h.mixHighbox + h.mixWalk ? "walk" : "none";
  }
  if (out.commands === "board" && !(mode && mode.eval)) {
    const roll = randomSource();
    const b = config.board;
    out.commands = roll < b.mixBoard ? "board" : roll < b.mixBoard + b.mixWalk ? "walk" : "none";
  }
  if (out.commands === "getup" && !(mode && mode.eval)) {
    const roll = randomSource();
    const g = config.getup;
    out.commands = roll < g.mixGetup ? "getup" : roll < g.mixGetup + g.mixWalk ? "walk" : "none";
  }
  if (out.commands === "climb" && !(mode && mode.eval)) {
    const roll = randomSource();
    const c = config.climb;
    out.commands = roll < c.mixClimb ? "climb" : roll < c.mixClimb + c.mixWalk ? "walk" : "none";
  }
  // The sandbox does not come in here. The level decides the marks, and the pair shares this draw.
  const list = ensureTargets(out);
  const meters = config.start.targetJitterMeters;
  const radians = config.start.targetJitterRadians;
  const commands = out.commands;
  const asksFoot = commands === "foot" || commands === "all";
  // WALK: the body marks slide at a commanded speed (see moveWalkMarks). The feet carry no
  // mark at all: a frozen foot mark would charge every step the doll takes.
  out.walk = null;
  out.climb = null;
  if (commands === "walk") {
    const w = config.walk;
    out.walk = {
      dir: w.dirs && w.dirs.length ? w.dirs[(randomSource() * w.dirs.length) | 0] : 1,
      speed: w.minSpeed + randomSource() * (w.maxSpeed - w.minSpeed),
      delay: w.delaySeconds,
    };
  }
  // CLIMB: a walk toward a box. The marks rise by the box height as they pass its near edge.
  if (commands === "climb") {
    const c = config.climb;
    let height;
    if (mode && mode.climbMax > 0) {
      const cap = Math.max(c.minHeight, mode.climbMax);
      const low = Math.max(c.minHeight, cap - (c.band || cap));
      height = low + randomSource() * (cap - low);
    } else {
      const list = c.evalHeights;
      height = list[(randomSource() * list.length) | 0];
    }
    out.climb = {
      near: c.nearMin + randomSource() * (c.nearMax - c.nearMin),
      height,
    };
    out.walk = {
      dir: 1,
      speed: c.speedMin + randomSource() * (c.speedMax - c.speedMin),
      delay: config.walk.delaySeconds,
    };
    out.frames = Math.round(c.seconds * config.physics.hz);
  }
  const walking = commands === "walk" || commands === "climb";
  // HIGHBOX: a tall box on either side. No walk: the marks are the standing pose on top of it,
  // feet included (highboxMarks in reset).
  if (commands === "highbox") {
    const h = config.highbox;
    let height = h.evalHeight;
    if (mode && mode.climbMax > 0) {
      const cap = Math.min(h.maxCap, mode.climbMax);
      const low = Math.max(0.05, cap - h.band);
      height = low + randomSource() * (cap - low);
    }
    out.climb = {
      near: h.nearMin + randomSource() * (h.nearMax - h.nearMin),
      height,
      side: randomSource() < 0.5 ? -1 : 1,
      half: h.halfWidth,
      high: true,
    };
    out.frames = Math.round(h.seconds * config.physics.hz);
  }
  // GETUP: lying face up or face down, settled on the floor before the clock starts. The marks are
  // moved to the standing pose in reset (getupMarks), and the feet carry none.
  // BOARD: start standing in the middle of a plank on a pivot. The plank follows a slow tilt wave.
  out.board = null;
  if (commands === "board") {
    const b = config.board;
    out.board = {
      // Training draws in [bandLow x cap, cap]: at the frontier. Drawn from 0, the easy half kept the
      // success rate up and the cap climbed to 0.19 rad while the 0.15 rad eval stayed 0/20.
      amp: mode && mode.eval
        ? b.evalAmp
        : (() => {
            const cap = Math.min(b.ampMax, mode && mode.boardMax > 0 ? mode.boardMax : b.ampMax);
            const low = (b.bandLow || 0) * cap;
            return low + randomSource() * (cap - low);
          })(),
      period: b.periodMin + randomSource() * (b.periodMax - b.periodMin),
      phase: randomSource() * 2 * Math.PI,
    };
    out.drop = b.pivotHeight + 2 * b.halfThick + 0.02;
    out.settle = Math.max(out.settle || 0, 10);
    out.frames = Math.round(b.seconds * config.physics.hz);
  }
  // SWING: start standing in the middle of the hanging plank, which gets a push.
  out.swing = null;
  if (commands === "swing") {
    const w = config.swing;
    const cap = mode && mode.swingMax > 0 ? Math.min(w.pushMax, mode.swingMax) : w.pushMax;
    out.swing = {
      push: (mode && mode.eval ? w.evalPush : w.pushMin + randomSource() * Math.max(0, cap - w.pushMin)) * (randomSource() < 0.5 ? -1 : 1),
    };
    out.drop = w.top + 0.02;
    // Start upright: the random start tilt of the other levels, on a plank that moves away underfoot,
    // tipped every controller over within a second, even with no push.
    out.tilt = 0;
    out.settle = Math.max(out.settle || 0, 10);
    out.frames = Math.round(w.seconds * config.physics.hz);
  }
  out.getup = false;
  out.startPose = null;
  if (commands === "getup") {
    const sign = randomSource() < 0.5 ? -1 : 1;
    out.getup = true;
    out.tilt = sign * (config.start.downTiltMinRadians + randomSource() * config.start.downTiltSpanRadians);
    out.drop = config.start.downDropMeters;
    out.settle = config.start.downSettleFrames;
    // Training with the fall library: start somewhere along a recorded fall (reverse curriculum).
    out.assist = 0;
    out.strength = 1;
    if (config.getup.useAssist && mode && mode.getupAssist > 0) out.assist = mode.getupAssist;
    if (config.getup.useStrength && mode && mode.getupStrength > 1) out.strength = mode.getupStrength;
    const falls = mode && mode.getupFalls;
    if (config.getup.useFalls && falls && falls.length && mode.getupMax > 0) {
      const g = config.getup;
      const cap = Math.min(g.maxCap, mode.getupMax);
      const low = Math.max(0, cap - g.band);
      const frac = low + randomSource() * (cap - low);
      const fall = falls[(randomSource() * falls.length) | 0];
      out.startPose = fall[Math.round(frac * (fall.length - 1))];
      out.startFrac = frac;
      out.settle = 2; // A mid-fall pose would collapse in a passive 40-frame settle.
      out.tilt = 0;
    }
    out.frames = Math.round(config.getup.seconds * config.physics.hz);
    // Imitation: the reference for this side (face up = positive tilt), and sometimes a start on it.
    out.reference = null;
    out.startTime = 0;
    out.startVel = null;
    const refs = mode && mode.getupReference;
    if (refs && !(mode && mode.eval)) {
      const side = out.tilt > 0 ? "up" : "down";
      out.reference = refs[side] || null;
      const states = mode.getupStates && mode.getupStates[side];
      if (states && states.length && randomSource() < (config.getup.rsiShare || 0)) {
        const st = states[(randomSource() * states.length) | 0];
        out.startPose = st.pose;
        out.startVel = st.vel;
        out.startTime = st.t;
        out.settle = 0;
        out.tilt = side === "up" ? 1e-3 : -1e-3; // keeps the side for the reference and the clock
      }
    }
  }
  for (let i = 0; i < list.length; i++) {
    const target = list[i];
    const spec = config.targets[i];
    if ((walking || commands === "getup") && !spec.ride) {
      target.on = false;
      target.gain = 0;
      target.dx = 0;
      target.dy = 0;
      target.turn = 0;
      continue;
    }
    if (commands === "foot" && spec.ride) {
      target.on = false;
      target.gain = 0;
      continue;
    }
    const order = sampleCommandFor(spec, commands, randomSource);
    target.on = true;
    target.cx = order.x;
    target.cy = order.y;
    target.ca = order.angle;
    target.dx = signed(randomSource, meters);
    target.dy = signed(randomSource, meters);
    target.turn = spec.cmdArc || !spec.ride ? 0 : signed(randomSource, radians);
  }
  // Foot ladder first, then the foot command. The other way around, fillStandFeet overwrites
  // the command and the foot level stops commanding the feet.
  // The old ladder (place, walk) runs only while the scripted gait is on. Without it, the
  // feet stay at the neutral stance and the command is in charge. Otherwise the place
  // stance (0.32 m) leaks into the head and hand levels.
  if (asksFoot) {
    const feet = [];
    for (let i = 0; i < list.length; i++) {
      if (!config.targets[i].ride) feet.push(i);
    }
    const swing =
      config.skill.oneFoot && feet.length
        ? feet[(randomSource() * feet.length) | 0]
        : -1;
    const cap = config.skill.cmdFootMeters;
    const floor = Math.min(cap, (config.skill.footReachMeters || 0) + 0.02);
    for (let i = 0; i < feet.length; i++) {
      const slot = feet[i];
      if (swing >= 0 && slot !== swing) {
        list[slot].dx = 0;
      } else {
        const mag = floor + randomSource() * Math.max(0, cap - floor);
        list[slot].dx = (randomSource() < 0.5 ? -1 : 1) * mag;
      }
      list[slot].turn = 0;
    }
    maybeCrouch(out, mode, randomSource, true);
  } else if (config.skill.scriptedLegs && level === 2) {
    const scale = full
      ? 1
      : mode && mode.placeScale != null
        ? mode.placeScale
        : config.skill.placeScale == null
          ? 1
          : config.skill.placeScale;
    fillPlaceFeet(out, list, randomSource, scale, mode && mode.shiftMeters);
  } else if (config.skill.scriptedLegs && level >= 3) {
    fillWalk(out, list, randomSource, level >= 4);
  } else {
    fillStandFeet(list, randomSource);
  }
  return out;
}

// Anatomical neutral of each part, in the torso frame: the blueprint minus the hip.
// A command is this point plus an offset drawn inside the part's workspace.
export function neutralLocal(spec) {
  return { x: spec.x || 0, y: (spec.y || 0) - config.body.torsoHeightMeters };
}

// The arc the head can describe. The neck is headRadius+headAxis from the head center,
// so the center can only sit on that circle.
function headArc(theta) {
  const r = config.body.headRadius + config.body.headAxis;
  const neckY = config.body.neckOffsetMeters || 0.42;
  return { x: -r * Math.sin(theta), y: neckY + r * Math.cos(theta) };
}

// Midpoint of the two feet, in the world. The body marks are anchored here.
function footMidX(measure) {
  return (
    ((measure.partX[config.partIndex.footLeft] || 0) +
      (measure.partX[config.partIndex.footRight] || 0)) /
    2
  );
}

// Keep the point inside the shoulder or hip reach. The arm and the leg start fully
// extended, so a command "a little further" is physically impossible and the net
// learns to ignore targets it can never hit.
function clampToReach(point, pivotY, reach) {
  const dy = point.y - pivotY;
  const d = Math.hypot(point.x, dy);
  if (d <= reach || d === 0) return point;
  const k = reach / d;
  return { x: point.x * k, y: pivotY + dy * k };
}

// Draw one part's command. Returns the torso-frame offset and the asked rotation.
// groups is what this rung enables: 'none', 'head', 'hand', 'foot', 'all'.
function sampleCommandFor(spec, groups, randomSource) {
  const neutral = neutralLocal(spec);
  if (spec.cmdArc) {
    const theta = signed(randomSource, spec.cmdAngle || 0);
    const arc = headArc(theta);
    return { x: arc.x, y: arc.y, angle: theta };
  }
  if (!spec.cmdAngle && !spec.cmdX && !spec.cmdY)
    return { x: neutral.x, y: neutral.y, angle: 0 };
  let point = {
    x: neutral.x + signed(randomSource, spec.cmdX || 0),
    y: neutral.y + signed(randomSource, spec.cmdY || 0),
  };
  if (spec.ride) {
    const shoulderY =
      config.body.shoulderMeters - config.body.torsoHeightMeters;
    point = clampToReach(point, shoulderY, config.body.armReachMeters);
  }
  return {
    x: point.x,
    y: point.y,
    angle: signed(randomSource, spec.cmdAngle || 0),
  };
}

function groupOf(spec) {
  const id = spec.id;
  return id === "head"
    ? "head"
    : id === "footR" || id === "footL"
      ? "foot"
      : "hand";
}

function commandedGroup(groups) {
  if (!groups || groups === "none") return "";
  return groups;
}

// TRAINING: planted ONCE. The foot anchor is used only at that instant, to take the
// standing torso height. After that the marks are FROZEN in the world: the torso has to
// stay up to reach them, so falling is a target error and needs no extra term. This used
// to run every frame and the mark chased the feet, which is why a dragged mark in the
// sandbox "came back".
// The asked rotation is the COMMAND (ca), NOT the start jitter (turn): the neck and the
// wrists have to serve a world command, not copy the start pose.
function plantBodyMarks(measure, options) {
  if (!measure || !options || !options.failOnHead) return;
  const targets = options.targets;
  if (!targets) return;
  const mid = footMidX(measure);
  for (let i = 0; i < config.targets.length; i++) {
    const spec = config.targets[i];
    if (!spec.ride) continue;
    const target = targets[i];
    if (!target || target.on === false || target.lx == null) continue;
    target.x = mid + target.lx;
    // ly is the world height: the part's height plus how far the command sits from neutral.
    target.y = target.ly;
    // The asked rotation is the command (ca), NOT the start jitter (turn).
    target.angle = target.ca || 0;
    // Frozen: leave it alone from here on.
    target.frozen = true;
  }
}

// Deliberate: the feet moved a little when the stage changed, and the head mark stays
// where it was. The feet do change marks, and only when the walk script advances a step.
function refreezeBodyTargets(options) {
  const targets = options && options.targets;
  if (!targets) return;
  for (let i = 0; i < config.targets.length; i++) {
    const spec = config.targets[i];
    if (!spec.ride) continue;
    const target = targets[i];
    if (target) target.frozen = true;
  }
}

function applyStep(measure, options, step) {
  const target = options.targets[step.foot];
  target.x = (target.baseX || 0) + step.dx;
  target.y = config.skill.footGroundMeters;
  target.angle = step.turn || 0;
  target.on = true;
}

// The next foot appears once this one is on its mark and the body is still standing.
// Once it is still, one foot stays and the other receives the drawn step.
function armPlace(episode) {
  const place = episode.options.place;
  if (!place || episode.placeArmed) return;
  if ((episode.upStreak || 0) < (episode.options.armAt || 30)) return;
  episode.placeArmed = true;
  episode.placeArmFrame = episode.frame;
  const measure = episode.measure;
  const targets = episode.options.targets;
  const feet = [
    [targetSlot("footL"), config.partIndex.footLeft],
    [targetSlot("footR"), config.partIndex.footRight],
  ];
  for (let i = 0; i < feet.length; i++) {
    const slot = feet[i][0];
    const part = feet[i][1];
    const target = targets[slot];
    target.x = measure.partX[part] || 0;
    target.baseX = target.x;
    target.y = config.skill.footGroundMeters;
    target.angle = Math.atan2(
      measure.partSin[part] || 0,
      measure.partCos[part] || 0,
    );
    target.on = true;
  }
  const extra = place.extra || 0;
  targets[targetSlot("footL")].x += extra;
  targets[targetSlot("footR")].x += extra;
  episode.placeExtra = extra;
  episode.phaseFrame = episode.frame;
  // The feet moved a little: the body mark does NOT follow them. It stays frozen.
  refreezeBodyTargets(episode.options);
}

function maybeAdvance(episode) {
  const steps = episode.options.steps;
  if (!steps || !steps.length) return;
  const cursor = episode.stepCursor || 0;
  if (cursor >= steps.length) return;
  if (!isStanding(episode.measure)) return;
  if (!onMark(episode.measure, steps[cursor].foot)) return;
  episode.stepCursor = cursor + 1;
  episode.stepsDone = episode.stepCursor;
  const next = steps[episode.stepCursor];
  if (next) {
    applyStep(episode.measure, episode.options, next);
    episode.placeExtra = next.foot === targetSlot("footR") ? 1 : -1;
  }
  episode.phaseFrame = episode.frame;
}

function armWalk(episode) {
  if ((episode.options.skill | 0) < 3 || episode.phaseFrame > 0) return;
  const steps = episode.options.steps;
  if (!steps || !steps.length || (episode.upStreak || 0) < 30) return;
  const step = steps[episode.stepCursor || 0];
  if (!step) return;
  episode.phaseFrame = episode.frame;
  episode.placeExtra = step.foot === targetSlot("footR") ? 1 : -1;
}

function boxesCleared(episode) {
  const live = episode.liveBoxes;
  if (!live || !live.length) return 0;
  const measure = episode.measure;
  const left = measure.partX[config.partIndex.footLeft];
  const right = measure.partX[config.partIndex.footRight];
  let count = 0;
  for (let i = 0; i < live.length; i++) {
    const box = live[i];
    const past =
      box.dir > 0
        ? left > box.far || right > box.far
        : left < box.far || right < box.far;
    if (past) count++;
  }
  return count;
}

// Writes the world mark.
// Feet sit at floor height, in world coordinates: a foot command is a world command,
// "put that foot 20 cm further forward".
// Body parts are in the torso frame, and the mark does NOT come from where the doll
// already is: it comes from the anatomical neutral plus the offset the ladder drew.
// That is the difference between a net that holds a pose and a net that obeys.
function stampTrainTargets(measure, options) {
  const source = options && options.targets;
  if (!source) return;
  if (options.placed) {
    for (let i = 0; i < source.length && i < config.targets.length; i++) {
      const target = source[i];
      const spec = config.targets[i];
      if (!target || !target.on) continue;
      target.span = spec.spanCmd || spec.span;
      // gainCmd is for the ladder, when that part is the task. Placed marks keep the
      // manual gain: a head left on while training a step must not be worth 3.
      target.gain = targetGain(spec.id);
    }
    return;
  }
  const groups = options.commands;
  for (let i = 0; i < source.length && i < config.targets.length; i++) {
    const target = source[i];
    if (!target.on) continue;
    const spec = config.targets[i];
    const index = config.partIndex[spec.part];
    const group = groupOf(spec);
    const asked =
      commandedGroup(groups) === "all" || commandedGroup(groups) === group;
    if (spec.ride) {
      if (commandedGroup(groups) === "foot") {
        target.on = false;
        target.gain = 0;
        continue;
      }
      // Span and gain depend on whether this part is being commanded: generous when it only
      // has to be roughly there, tight when it has to obey.
      target.span = asked ? spec.spanCmd || spec.span : spec.span;
      target.gain = targetGain(spec.id) * (asked ? spec.gainCmd || 1 : 1);
      if (asked) {
        // cx and cy are torso-frame positions. The mark is the part, plus how far the
        // command sits from that part's neutral. At rest the delta is zero, so the mark
        // starts on the part instead of a neck-length above the head.
        const feetMid = footMidX(measure);
        const neutral = neutralLocal(spec);
        target.lx = (measure.partX[index] || 0) - feetMid + ((target.cx || 0) - neutral.x);
        target.ly = (measure.partY[index] || 0) + ((target.cy || 0) - neutral.y);
        target.angle = target.ca || 0;
      } else {
        const feetMid = footMidX(measure);
        target.lx = (measure.partX[index] || 0) - feetMid;
        target.ly = measure.partY[index] || 0;
        target.angle = 0;
      }
    } else {
      target.span = asked ? spec.spanCmd || spec.span : spec.span;
      target.gain = targetGain(spec.id) * (asked ? spec.gainCmd || 1 : 1);
      // THE FEET STAY FROZEN IN THE WORLD TOO. Anchoring the mark to the hip was tried so
      // the row would measure the step and not the drift. It does that, and it ruins skill 1.
      //
      // check-collapse cannot see why. It COMPARES FROZEN POSES, and there the hip anchor
      // wins: standing 7.31 against lying -4.99, a gap of 10.31 where the world anchor gave
      // 3.18. In training, the hip-anchored foot was 4/20 at gen 300 against 9/20 for the
      // world anchor, then it fell to 0/20 and stayed there. While the torso goes to the
      // floor the hip drops WITH the feet, so the mark follows and the foot scores the
      // point without the body being up. The feet are worth 6 of 12 points.
      //
      // The drift that was measured (0.298 m against a 0.089 m command) is not fixed by
      // the anchor: it is what happens while the net still cannot stand. Once it can, the
      // drift is zero and the frozen mark measures the command alone. Skill 1 first, skill 4 after.
      target.baseX = measure.partX[index] || 0;
      // IF THIS FOOT IS NOT BEING COMMANDED, THE MARK GOES WHERE THE FOOT IS. It used to
      // not: the foot separation (fillStandFeet) was put on the target dx, that dx was
      // added to the mark, and on skill 1 — STAND — the right foot was asked to step 12 cm
      // in the same episode it was asked not to fall. The feet are worth 6 of 12 points.
      // The separation is now on the start pose (options.stance). There is nothing to command here.
      target.x = target.baseX + (asked ? target.dx || 0 : 0);
      target.y = config.skill.footGroundMeters;
      target.angle = target.turn || 0;
    }
  }
  plantBodyMarks(measure, options);
}

// How many body targets (head and hands) that are on are inside their tolerance.
// The eval uses this to see whether the net is using them, not only the feet.
function placedBodyTargets(measure) {
  let on = 0;
  let placed = 0;
  for (let i = 0; i < config.targets.length; i++) {
    const spec = config.targets[i];
    if (!spec.ride) continue;
    const target = measure.targets && measure.targets[i];
    if (!target || !target.on || !targetGain(spec.id)) continue;
    on++;
    if (onMark(measure, i)) placed++;
  }
  return on ? placed / on : 0;
}

// HIGHBOX: every mark is the standing pose on top of the box, standInset from the facing edge.
// The feet marks sit on the box top, one stance apart.
function highboxMarks(options, box) {
  const targets = options && options.targets;
  if (!targets) return;
  const inset = config.highbox.standInset;
  const standX = box.side > 0 ? box.x0 + inset : box.x1 - inset;
  const stance = config.skill.stanceMeters || 0.11;
  for (let i = 0; i < config.targets.length; i++) {
    const spec = config.targets[i];
    const target = targets[i];
    if (!target) continue;
    target.on = true;
    target.angle = 0;
    target.frozen = true;
    if (spec.ride) {
      target.x = standX + (spec.x || 0);
      target.y = box.top + (spec.y || 0);
    } else {
      target.x = standX + (spec.id === "footR" ? stance : -stance);
      target.y = box.top + config.skill.footGroundMeters;
    }
  }
}

// GETUP: the head and hand marks go to the STANDING pose over the pelvis, not where the parts
// lie. stampTrainTargets planted them on the lying body, which would pay for staying down.
function getupMarks(measure, options) {
  const targets = options && options.targets;
  if (!targets) return;
  const pelvisX = measure.partX[config.partIndex.torso] || 0;
  for (let i = 0; i < config.targets.length; i++) {
    const spec = config.targets[i];
    if (!spec.ride) continue;
    const target = targets[i];
    if (!target || target.on === false) continue;
    target.x = pelvisX + (spec.x || 0);
    target.y = spec.y || 0;
    target.angle = 0;
    target.frozen = true;
  }
}

// How many feet are on top of the box: centre inside it in x and at least top - slack high.
export function feetOnBox(measure, box) {
  if (!box || !measure || !measure.partX) return 0;
  const slack = config.climb.onTopSlack;
  const P = config.partIndex;
  let n = 0;
  for (const part of [P.footLeft, P.footRight]) {
    const x = measure.partX[part] || 0;
    const y = measure.partY[part] || 0;
    if (x > box.x0 && x < box.x1 && y > box.top - slack) n++;
  }
  return n;
}

export function summariseEpisode(rows, offset, count, out) {
  let fitness = 0;
  let start = 0;
  let end = 0;
  let gap = 0;
  let stance = 0;
  let headImpact = 0;
  let stoodUp = true;
  let alive = 0;
  let steps = 0;
  let foot = 0;
  let reach = 0;
  let closed = 0;
  let leaned = 0;
  let walkDist = 0;
  let climbOnTop = 0;
  let climbHeight = 0;
  let headHit = 0;
  let isGetup = 0;
  let isBoard = 0;
  let trackErr = 0;
  let trackN = 0;
  let headHeight = 0;
  const last = offset + count;
  for (let i = offset; i < last; i++) {
    const row = rows[i];
    fitness += row.fitness;
    start += row.start || 0;
    end += row.end || 0;
    gap += row.gap || 0;
    stance += row.stance || 0;
    alive += row.alive || 0;
    steps += row.steps || 0;
    foot += row.foot || 0;
    reach += row.reach || 0;
    closed += row.closed || 0;
    leaned += row.leaned || 0;
    walkDist += row.walkDist || 0;
    climbOnTop += row.climbOnTop || 0;
    climbHeight += row.climbHeight || 0;
    headHit += row.headHit || 0;
    isGetup += row.isGetup || 0;
    isBoard += row.isBoard || 0;
    if (row.trackErr >= 0) { trackErr += row.trackErr; trackN++; }
    headHeight += row.headHeight || 0;
    if ((row.headImpact || 0) > headImpact) headImpact = row.headImpact;
    if (!row.stoodUp) stoodUp = false;
  }
  const n = count || 1;
  const mean = fitness / n;
  out.fitness = mean;
  out.f = mean;
  out.start = start / n;
  out.end = end / n;
  out.gap = gap / n;
  out.stance = stance / n;
  out.headImpact = headImpact;
  out.stoodUp = stoodUp;
  out.up = stoodUp;
  out.alive = alive / n;
  out.steps = steps / n;
  out.foot = foot / n;
  out.reach = reach / n;
  out.closed = closed / n;
  out.leaned = leaned / n;
  out.walkDist = walkDist / n;
  out.climbOnTop = climbOnTop / n;
  out.climbHeight = climbHeight / n;
  out.headHit = headHit / n;
  out.isGetup = isGetup / n;
  out.isBoard = isBoard / n;
  out.trackErr = trackN ? trackErr / trackN : -1;
  out.headHeight = headHeight / n;
  return out;
}

export function createSim(box) {
  const {
    CreateBoxPolygon,
    b2Vec2,
    b2BodyType,
    b2World_Step,
    b2Body_SetTransform,
    b2MakeRot,
    b2Body_SetLinearVelocity,
    b2Body_GetLinearVelocity,
    b2Body_SetAngularVelocity,
    b2Body_SetAwake,
    b2Body_ApplyForceToCenter,
    b2Body_ApplyTorque,
    b2RevoluteJoint_GetAngle,
    b2RevoluteJoint_SetMaxMotorTorque,
    b2RevoluteJoint_SetMotorSpeed,
    b2RevoluteJoint_EnableMotor,
    b2RevoluteJoint_EnableSpring,
    CreateCircle,
    b2DestroyBody,
    b2Body_GetContactCapacity,
    b2Body_GetContactData,
    b2Shape_GetBody,
    b2Body_GetPosition: Box2D_GetPosition,
  } = box;
  const dynamicBody = b2BodyType.b2_dynamicBody;
  const staticBody = b2BodyType.b2_staticBody;
  const vector = (x, y) => new b2Vec2(x, y);
  const placed = [];
  let random = makeSeededRandom(7);
  // Reads `random` on every call, so setRandom changes the sequence.
  const randomNormal = () => randomNormalSample(random);

  function build(options) {
    return new RagDoll2D(box, config, options);
  }

  // A world must be destroyed, not just dropped. Box2D keeps a fixed array of worlds and the
  // slot is only released by DestroyWorld, so building worlds without destroying them exhausts
  // the array and the next CreateBody reads undefined.
  function destroyWorld(world) {
    if (world) world.destroy();
  }

  // Runs one episode and returns the finished Episode.
  //
  // Every scored episode must come through here. Resetting a world is not enough: Box2D keeps
  // per-body solver state that a transform reset does not clear, so a world reused for a second
  // episode answered differently depending on how many episodes had run in it before. That is
  // what made the official evaluation change when only the worker count changed.
  //
  // Passing a world in reuses it and calls rebuild(), which destroys the body tree and builds it
  // again in place. That is the only way to get there, because b2CreateWorldArray caps a process
  // at 32 worlds and b2DestroyWorld does not return the slot.
  function runOneEpisode(weights, options, world) {
    const wd = world || build();
    if (world) wd.rebuild();
    const episode = new Episode(
      wd,
      weights,
      (options && options.tilt) || 0,
      (options && options.drop) || 0,
      options || {},
    );
    while (!episode.advanceOneFrame());
    return episode;
  }

  function sameBody(a, b) {
    if (a === b) return true;
    if (!a || !b || a.index1 == null || b.index1 == null) return false;
    return a.index1 === b.index1 && a.world0 === b.world0;
  }
  function isDollBody(world, bodyId) {
    const parts = world.parts;
    for (let i = 0; i < parts.length; i++)
      if (sameBody(parts[i].id, bodyId)) return true;
    return false;
  }
  function otherBody(contact, bodyId) {
    const shapeA = contact.shapeIdA;
    const shapeB = contact.shapeIdB;
    if (!shapeA || !shapeB) return null;
    const bodyA = b2Shape_GetBody(shapeA);
    const bodyB = b2Shape_GetBody(shapeB);
    if (sameBody(bodyA, bodyId)) return bodyB;
    if (sameBody(bodyB, bodyId)) return bodyA;
    return null;
  }
  // A thigh is allowed to meet the chest. That is not "touching the world".
  function selfContact(world, bodyId, contact) {
    const other = otherBody(contact, bodyId);
    return !!(other && isDollBody(world, other));
  }

  // Face rays miss a corner. Box2D's manifold has that point. Speculative contacts reach 2 cm with no load, so those do not count.
  const contactData = [];
  function freshContact() {
    return { manifold: null, shapeIdA: null, shapeIdB: null };
  }
  function bodyTouching(world, bodyId) {
    const capacity = b2Body_GetContactCapacity(bodyId);
    if (!capacity) return false;
    while (contactData.length < capacity) contactData.push(freshContact());
    const count = b2Body_GetContactData(bodyId, contactData, capacity);
    const gap = config.body.touchGapMeters;
    for (let i = 0; i < count; i++) {
      if (selfContact(world, bodyId, contactData[i])) continue;
      const manifold = contactData[i].manifold;
      if (!manifold) continue;
      for (let p = 0; p < manifold.pointCount; p++) {
        const point = manifold.points[p];
        if (point.separation <= gap || point.normalImpulse > 0) return true;
      }
    }
    return false;
  }

  // How hard a body is pressed into the world, from 0 to 1. 0 when it is clear. Separation
  // reaches 1 at contact.fullMeters, and impulse rises as 1 - exp(-impulse / impulseFull).
  //
  // This reads Box2D's contact manifolds, not the proximity rays, and that is deliberate.
  // ragdoll.js gives thigh, shin, torso and upper arm only HORIZONTAL rays, so a thigh lying flat
  // on the floor never lowers measure.clearance and a ray cannot see it at all. The manifold can:
  // it knows the real gap and the real load.
  //
  // Thighs can meet the chest. Those contacts are skipped: the penalty is the floor and
  // thrown objects, not the doll touching itself.
  function contactSample(world, bodyId) {
    const capacity = b2Body_GetContactCapacity(bodyId);
    if (!capacity) return { amount: 0, impulse: 0 };
    while (contactData.length < capacity) contactData.push(freshContact());
    const count = b2Body_GetContactData(bodyId, contactData, capacity);
    const reach = config.body.touchGapMeters + config.contact.fullMeters;
    const impulseFull = config.contact.impulseFull;
    let amount = 0;
    let impulse = 0;
    for (let i = 0; i < count; i++) {
      if (selfContact(world, bodyId, contactData[i])) continue;
      const manifold = contactData[i].manifold;
      if (!manifold) continue;
      for (let p = 0; p < manifold.pointCount; p++) {
        const point = manifold.points[p];
        let fromGap = (config.body.touchGapMeters - point.separation) / reach;
        if (fromGap < 0) fromGap = 0;
        else if (fromGap > 1) fromGap = 1;
        const load = point.normalImpulse || 0;
        const fromImpulse = load > 0 && impulseFull > 0 ? 1 - Math.exp(-load / impulseFull) : 0;
        const pointAmount = fromGap > fromImpulse ? fromGap : fromImpulse;
        if (pointAmount > amount) amount = pointAmount;
        if (load > impulse) impulse = load;
      }
    }
    return { amount: amount > 1 ? 1 : amount, impulse };
  }

  function layoutStartPose(world, tilt, drop, options) {
    const cosine = Math.cos(tilt);
    const sine = Math.sin(tilt);
    const parts = world.parts;
    const count = parts.length;
    let lowest = Infinity;
    const shiftX = (options && options.shiftX) || 0;
    // Stance width is applied to the POSE, not the mark. See fillTrainingStart: that gap
    // used to be invented on the target dx, which leaked in as a step command.
    const stance = (options && options.stance) || 0;
    const crouch = (options && options.crouchKnee) || 0;
    const footRight = config.partIndex.footRight;
    const footLeft = config.partIndex.footLeft;
    const thighLeft = config.partIndex.thighLeft;
    const thighRight = config.partIndex.thighRight;
    const shinLeft = config.partIndex.shinLeft;
    const shinRight = config.partIndex.shinRight;
    const hipY = config.body.hipMeters;
    const kneeY0 = 0.52;
    for (let i = 0; i < count; i++) {
      const part = parts[i];
      let slot = placed[i];
      if (!slot) {
        slot = { part: null, x: 0, y: 0, extra: 0 };
        placed[i] = slot;
      }
      let localX = part.cx;
      let localY = part.cy;
      let extra = 0;
      if (i === footRight) localX = stance || part.cx;
      else if (i === footLeft) localX = -(stance || part.cx);
      if (crouch > 0) {
        const leg =
          i === thighLeft ||
          i === shinLeft ||
          i === footLeft ||
          i === thighRight ||
          i === shinRight ||
          i === footRight;
        if (leg) {
          const phi = crouch;
          const c = Math.cos(phi);
          const s = Math.sin(phi);
          if (i === thighLeft || i === thighRight) {
            const dx = localX;
            const dy = localY - hipY;
            localX = c * dx - s * dy;
            localY = hipY + s * dx + c * dy;
            extra = phi;
          } else {
            const newKneeX = (hipY - kneeY0) * s;
            const newKneeY = hipY - (hipY - kneeY0) * c;
            localX = newKneeX + localX;
            localY = newKneeY + (localY - kneeY0);
          }
        }
      }
      const x = cosine * localX - sine * (localY - 0.04) + shiftX;
      const y = 0.04 + drop + sine * localX + cosine * (localY - 0.04);
      slot.part = part;
      slot.x = x;
      slot.y = y;
      slot.extra = extra;
      world.support(part, -sine, -cosine);
      lowest = Math.min(lowest, y + sine * world.skinX + cosine * world.skinY);
    }
    const lift = Math.max(0, 0.05 - lowest);
    for (let i = 0; i < count; i++) placed[i].y += lift;
  }

  function resetPose(world, tilt, drop, options) {
    layoutStartPose(world, tilt, drop, options);
    const count = world.parts.length;
    for (let i = 0; i < count; i++) {
      const slot = placed[i];
      const part = world.parts[i];
      // PhaserBox2D stores these by reference. A reused vec/rot glues bodies together
      // or parks one at the origin (the pelvis at 0,0 crush).
      b2Body_SetTransform(
        part.id,
        vector(slot.x, slot.y),
        b2MakeRot(tilt + (slot.extra || 0)),
      );
      b2Body_SetLinearVelocity(part.id, vector(0, 0));
      b2Body_SetAngularVelocity(part.id, 0);
      b2Body_SetAwake(part.id, true);
    }
    const joints = world.joints;
    for (let i = 0; i < joints.length; i++)
      b2RevoluteJoint_SetMotorSpeed(joints[i].id, 0);
  }

  function readBodyState(world, measure) {
    world.readPose();
    let mass = 0;
    let momentX = 0;
    let momentY = 0;
    let momentVx = 0;
    let momentVy = 0;
    const count = world.parts.length;
    const partX = measure.partX;
    const partY = measure.partY;
    const partCos = measure.partCos;
    const partSin = measure.partSin;
    for (let i = 0; i < count; i++) {
      const part = world.parts[i];
      mass += part.m;
      momentX += part.m * partX[i];
      momentY += part.m * partY[i];
      momentVx += part.m * measure.partVelocityX[i];
      momentVy += part.m * measure.partVelocityY[i];
    }
    const torso = config.partIndex.torso;
    const head = config.partIndex.head;
    const headTouch = bodyTouching(world, world.parts[head].id);
    measure.touch[head] = headTouch;
    measure.headContact = headTouch ? 1 : 0;
    const footLeft = config.partIndex.footLeft;
    const footRight = config.partIndex.footRight;
    if (!measure.footContact) measure.footContact = new Float32Array(2);
    measure.footContact[0] = contactSample(world, world.parts[footLeft].id).amount;
    measure.footContact[1] = contactSample(world, world.parts[footRight].id).amount;
    measure.headX = partX[head];
    measure.headWorldY = partY[head];
    measure.headVelocityY = measure.partVelocityY[head];
    measure.headVelocityX = measure.partVelocityX[head];
    measure.footXLeft = partX[footLeft];
    measure.footXRight = partX[footRight];
    measure.torsoHeight = partY[torso];
    measure.headHeight = partY[head];
    measure.centerX = momentX / mass;
    measure.centerY = momentY / mass;
    measure.centerVelocityX = momentVx / mass;
    measure.centerVelocityY = momentVy / mass;
    measure.torsoSin = partSin[torso];
    measure.torsoCos = partCos[torso];
    measure.headSin = partSin[head];
    measure.headCos = partCos[head];
    measure.torsoSpin = measure.partSpin[torso];
    measure.stanceWidth = Math.abs(measure.footXLeft - measure.footXRight);
    let elbow = 0;
    for (const joint of world.joints) {
      if (joint.mark !== "elbow") continue;
      measure.elbowY[elbow++] =
        partY[joint.a] +
        partSin[joint.a] * joint.ax +
        partCos[joint.a] * joint.ay;
    }
    const groups = config.contact.groups;
    let contact = measure.contact;
    let contactImpulse = measure.contactImpulse;
    if (!contact || contact.length !== groups.length) {
      contact = new Float32Array(groups.length);
      measure.contact = contact;
    }
    if (!contactImpulse || contactImpulse.length !== groups.length) {
      contactImpulse = new Float32Array(groups.length);
      measure.contactImpulse = contactImpulse;
    }
    for (let g = 0; g < groups.length; g++) {
      const names = groups[g].parts;
      let sum = 0;
      let impulse = 0;
      for (let p = 0; p < names.length; p++) {
        const sample = contactSample(
          world,
          world.parts[config.partIndex[names[p]]].id,
        );
        sum += sample.amount;
        if (sample.impulse > impulse) impulse = sample.impulse;
      }
      contact[g] = names.length ? sum / names.length : 0;
      contactImpulse[g] = impulse;
    }
    measure.handOn = !!(
      measure.touch[config.partIndex.handLeft] ||
      measure.touch[config.partIndex.handRight]
    );
    measure.bodyOn = false;
    for (let i = 0; i < count; i++) {
      if (i !== footLeft && i !== footRight && measure.touch[i])
        measure.bodyOn = true;
    }
    return measure;
  }

  // Rays only on the action frame. The head contact, already measured, is not overwritten.
  function measureProximity(world, measure) {
    const count = world.parts.length;
    for (let i = 0; i < count; i++)
      measure.clearance[i] = config.body.bodyRayMeters;
    world.throwRays(measure);
    const gap = config.body.touchGapMeters;
    for (let i = 0; i < count; i++)
      measure.touch[i] = measure.clearance[i] < gap;
    measure.touch[config.partIndex.head] = measure.headContact !== 0;
    // A foot also touches when Box2D says it is pressed on the floor. The sole ray starts at the
    // foot centre, so a foot resting on its heel with the toe tilted up 0.14 rad read 9 mm of
    // clearance while its contact amount was 0.75: planted, but "in the air" for isStanding.
    const feet = measure.footContact;
    if (feet) {
      const pressed = config.skill.stanceContact || 0.3;
      if (feet[0] > pressed) measure.touch[config.partIndex.footLeft] = true;
      if (feet[1] > pressed) measure.touch[config.partIndex.footRight] = true;
    }
  }

  // Pose one physics frame ahead if every part keeps its current speed. No collision, motor, or joint.
  function predictMeasure(world, measure, dt) {
    const count = world.parts.length;
    const predicted = {
      partCos: measure.partCos,
      partSin: measure.partSin,
      torsoCos: measure.torsoCos,
      partHeight: new Float32Array(count),
      partVelocityX: measure.partVelocityX,
      partVelocityY: measure.partVelocityY,
      partSpin: measure.partSpin,
      clearance: new Array(count),
      touch: new Array(count),
      sole: new Float32Array(2),
      torsoSpin: measure.torsoSpin,
      headVelocityY: measure.headVelocityY,
      centerVelocityX: measure.centerVelocityX,
      centerVelocityY: measure.centerVelocityY,
    };
    let mass = 0;
    let momentX = 0;
    let momentY = 0;
    const aheadX = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      const x = measure.partX[i] + (measure.partVelocityX[i] || 0) * dt;
      const rise = (measure.partVelocityY[i] || 0) * dt;
      aheadX[i] = x;
      predicted.partHeight[i] = (measure.partHeight[i] || 0) + rise;
      predicted.clearance[i] = (measure.clearance[i] || 0) + rise;
      predicted.touch[i] = predicted.clearance[i] < config.body.touchGapMeters;
      mass += world.parts[i].m;
      momentX += world.parts[i].m * x;
      momentY += world.parts[i].m * predicted.partHeight[i];
    }
    const footLeft = config.partIndex.footLeft;
    const footRight = config.partIndex.footRight;
    predicted.footXLeft = aheadX[footLeft];
    predicted.footXRight = aheadX[footRight];
    predicted.torsoHeight = predicted.partHeight[config.partIndex.torso];
    predicted.headHeight = predicted.partHeight[config.partIndex.head];
    predicted.centerX = mass ? momentX / mass : 0;
    predicted.centerY = mass ? momentY / mass : 0;
    predicted.stanceWidth = Math.abs(
      predicted.footXLeft - predicted.footXRight,
    );
    predicted.sole[0] = Math.max(
      0,
      (measure.sole[0] || 0) + (measure.partVelocityY[footLeft] || 0) * dt,
    );
    predicted.sole[1] = Math.max(
      0,
      (measure.sole[1] || 0) + (measure.partVelocityY[footRight] || 0) * dt,
    );
    return predicted;
  }

  function spawnStaticBox(world, x, halfW, halfH) {
    const created = CreateBoxPolygon({
      worldId: world.worldId,
      type: staticBody,
      position: vector(x, halfH),
      size: vector(halfW, halfH),
      friction: 2,
    });
    const id = created.bodyId;
    world.thrownIds.push(id);
    world.thrownShapes.push(world.shapeOf(id));
    return id;
  }

  // A grippable body the hand touches, or null. grippable is [{ id, cx, cy }] of static boxes.
  function touchingGrippable(world, handId, grippable) {
    const capacity = b2Body_GetContactCapacity(handId);
    if (!capacity) return null;
    while (contactData.length < capacity) contactData.push(freshContact());
    const count = b2Body_GetContactData(handId, contactData, capacity);
    for (let i = 0; i < count; i++) {
      const other = otherBody(contactData[i], handId);
      if (!other) continue;
      for (let g = 0; g < grippable.length; g++) if (sameBody(other, grippable[g].id)) return grippable[g];
    }
    return null;
  }

  // BOARD: a static pivot block and a dynamic plank hinged on it. Both go in objects, so cleanup
  // destroys them (the joint goes with the plank). The plank is registered with the rays.
  function spawnBoard(world, x, objects) {
    const b = config.board;
    const pivotHalf = b.pivotHeight / 2;
    const pivot = CreateBoxPolygon({
      worldId: world.worldId, type: staticBody,
      position: vector(x, pivotHalf), size: vector(0.05, pivotHalf), friction: 2,
    }).bodyId;
    const plankY = b.pivotHeight + b.halfThick;
    const plank = CreateBoxPolygon({
      worldId: world.worldId, type: dynamicBody,
      position: vector(x, plankY), size: vector(b.halfLength, b.halfThick),
      density: b.density, friction: 2,
    }).bodyId;
    for (const id of [pivot, plank]) {
      world.thrownIds.push(id);
      world.thrownShapes.push(world.shapeOf(id));
      objects.push(id);
    }
    const joint = world.CreateRevoluteJoint({
      worldId: world.worldId, bodyIdA: pivot, bodyIdB: plank,
      anchorA: vector(0, pivotHalf), anchorB: vector(0, -b.halfThick),
      lowerAngle: -b.limitRadians, upperAngle: b.limitRadians, enableLimit: true,
      enableMotor: true, motorSpeed: 0, maxMotorTorque: b.motorTorque,
    }).jointId;
    return { pivot, plank, joint };
  }

  // SWING: a static bar, two rope rods hinged to it and to the plank (a parallelogram, so the plank
  // stays level). The rods collide with nothing (own category, empty mask). All go in objects.
  function spawnSwing(world, x, objects) {
    const w = config.swing;
    const plankY = w.top - w.halfThick;
    const barY = w.top + w.ropeLength;
    const bar = CreateBoxPolygon({
      worldId: world.worldId, type: staticBody,
      position: vector(x, barY + 0.03), size: vector(w.halfLength + 0.2, 0.03), friction: 1,
    }).bodyId;
    const plank = CreateBoxPolygon({
      worldId: world.worldId, type: dynamicBody,
      position: vector(x, plankY), size: vector(w.halfLength, w.halfThick),
      density: w.density, friction: 2,
    }).bodyId;
    world.thrownIds.push(plank);
    world.thrownShapes.push(world.shapeOf(plank));
    objects.push(bar, plank);
    const ropes = [];
    // Each rope spans exactly from its bar anchor (y = barY) to its plank anchor (the plank centre
    // line): the first version was 4.5 cm short and pulled the plank 7 cm below its design height.
    const ropeHalf = (barY - plankY) / 2;
    for (const side of [-1, 1]) {
      const rx = x + side * (w.halfLength - w.ropeInset);
      const rope = CreateBoxPolygon({
        worldId: world.worldId, type: dynamicBody,
        position: vector(rx, (barY + plankY) / 2), size: vector(0.01, ropeHalf),
        // Own category, colliding with nothing: the doll's chest and thighs collide with the default
        // (floor) category, so a group index alone let the body lean on the ropes. Joints still hold them.
        density: config.swing.ropeDensity || 2, friction: 0, categoryBits: 0x10, maskBits: 0,
      }).bodyId;
      objects.push(rope);
      world.CreateRevoluteJoint({
        worldId: world.worldId, bodyIdA: bar, bodyIdB: rope,
        anchorA: vector(side * (w.halfLength - w.ropeInset), -0.03), anchorB: vector(0, ropeHalf),
      });
      world.CreateRevoluteJoint({
        worldId: world.worldId, bodyIdA: plank, bodyIdB: rope,
        anchorA: vector(side * (w.halfLength - w.ropeInset), 0), anchorB: vector(0, -ropeHalf),
      });
      ropes.push(rope);
    }
    return { bar, plank, ropes, x };
  }

  // A still box in the path of the step. dx is relative to the left foot at the start.
  function placeSkillBoxes(episode) {
    const boxes = episode.options.boxes;
    const live = episode.liveBoxes;
    live.length = 0;
    if (!boxes || !boxes.length) return;
    const origin = episode.measure.partX[config.partIndex.footLeft] || 0;
    for (let i = 0; i < boxes.length; i++) {
      const box = boxes[i];
      const x = origin + box.dx;
      episode.objects.push(
        spawnStaticBox(episode.world, x, box.halfW, box.halfH),
      );
      live.push({
        far: x + box.dir * box.halfW,
        dir: box.dir || 1,
        x,
        halfW: box.halfW,
      });
    }
  }

  class Episode {
    constructor(world, weights, tilt, drop, options = {}) {
      this.free = false;
      this.passive = false;
      this.world = world;
      this.network = new ElmanNetwork(weights);
      this.residual = new ResidualNet();
      this.residualIn = new Float32Array(12);
      this.legAdd = new Float32Array(config.net.jointCount);
      this.legBase = new Float32Array(config.net.jointCount);
      this.startSamples = rewardSamples(config.episode.startWindowSeconds);
      this.endRing = new Float32Array(
        rewardSamples(config.episode.endWindowSeconds),
      );
      this.input = new Float32Array(config.net.inputCount);
      this.previousJointAngle = new Float32Array(config.net.jointCount);
      this.objects = [];
      this.liveBoxes = [];
      this.measure = {
        partX: world.px,
        partY: world.py,
        partCos: world.c,
        partSin: world.s,
        partHeight: world.py,
        clearance: new Array(world.parts.length),
        touch: new Array(world.parts.length),
        sole: new Float32Array(2),
        elbowY: new Float32Array(2),
        faceRay: world.faceRay,
        partVelocityX: world.vx,
        partVelocityY: world.vy,
        partSpin: world.spin,
      };
      this.reset(weights, tilt, drop, options);
    }
    bindOptions(options) {
      if (options && options.frames != null && options.period != null) {
        this.options = options;
        return;
      }
      if (!this.optionScratch) {
        this.optionScratch = {
          push: 0,
          amp: 0,
          period: config.body.platformPeriodSeconds,
          phaseShift: 0,
          projectiles: [],
          frames: config.episode.frames,
          task: "stand",
          phase: 0,
        };
      }
      const scratch = this.optionScratch;
      scratch.push = 0;
      scratch.amp = 0;
      scratch.period = config.body.platformPeriodSeconds;
      scratch.phaseShift = 0;
      scratch.frames = config.episode.frames;
      scratch.task = "stand";
      scratch.phase = 0;
      scratch.failOnHead = false;
      scratch.settle = 0;
      scratch.shiftX = 0;
      scratch.tilt = 0;
      scratch.drop = 0;
      scratch.ballFrame = null;
      scratch.targets = null;
      scratch.trace = false;
      if (options) {
        const keys = Object.keys(options);
        for (let i = 0; i < keys.length; i++)
          scratch[keys[i]] = options[keys[i]];
      }
      this.options = scratch;
    }
    // Same body and the same buffers. Changes the weights and returns to the start pose.
    reset(weights, tilt, drop, options) {
      this.bindOptions(options || {});
      const network = this.network;
      network.weights = weights;
      network.hidden.fill(0);
      network.nextHidden.fill(0);
      const ff = network.ff;
      for (let layer = 0; layer < ff.length; layer++) ff[layer].fill(0);
      if (network.feedback.length) network.feedback.fill(0);
      network.action.fill(0);
      const rawResidual = this.options.residual;
      if (rawResidual && rawResidual.length === this.residual.weights.length)
        this.residual.weights.set(rawResidual);
      else this.residual.weights.fill(0);
      this.frame = 0;
      this.startSum = 0;
      this.startCount = 0;
      this.endCursor = 0;
      this.endFilled = 0;
      this.fallen = false;
      this.upStreak = 0;
      this.stoodFrames = 0;
      this.sensedFrames = 0;
      this.heightSum = 0;
      this.headImpact = 0;
      this.headHit = false;
      this.motionSum = 0;
      this.motionCount = 0;
      this.lifeSum = 0;
      this.lifeCount = 0;
      this.uprightSum = 0;
      this.stepCursor = 0;
      this.placeArmed = false;
      this.placeExtra = 0;
      this.placeArmFrame = 0;
      this.phaseFrame = 0;
      this.legLocked = false;
      this.gap0 = null;
      this.gapCloseSum = 0;
      this.gapCloseCount = 0;
      this.bodyGap0 = null;
      this.bodyCloseSum = 0;
      this.airPart = -1;
      this.airGap = 0;
      this.swingPeak = 0;
      this.swingClose = 0;
      this.footGap = 0;
      this.torsoDown = false;
      this.stepsDone = 0;
      this.windowStood = 0;
      this.windowFrames = 0;
      this.endedStanding = false;
      this.liveBoxes.length = 0;
      this.previousRewardTotal = 0;
      this.ballThrown = false;
      this.endedUp = false;
      this.previousJointAngle.fill(0);
      if (this.measure.prevAbsX) this.measure.prevAbsX.fill(NaN);
      if (this.measure.prevAbsY) this.measure.prevAbsY.fill(NaN);
      this.cleanup();
      const world = this.world;
      resetPose(world, tilt, drop, this.options);
      // A recorded whole-body pose (get-up curriculum): every part placed, at rest.
      const pose = this.options.startPose;
      if (pose && pose.length === world.parts.length * 3) {
        const shift = this.options.shiftX || 0;
        // A reference-state start may carry velocities (x, y, spin per part).
        const vel = this.options.startVel && this.options.startVel.length === pose.length ? this.options.startVel : null;
        for (let i = 0; i < world.parts.length; i++) {
          const part = world.parts[i];
          b2Body_SetTransform(part.id, vector(pose[i * 3] + shift, pose[i * 3 + 1] + 0.01), b2MakeRot(pose[i * 3 + 2]));
          b2Body_SetLinearVelocity(part.id, vector(vel ? vel[i * 3] : 0, vel ? vel[i * 3 + 1] : 0));
          b2Body_SetAngularVelocity(part.id, vel ? vel[i * 3 + 2] : 0);
          b2Body_SetAwake(part.id, true);
        }
      }
      // Grips from the last episode died with their boxes in cleanup.
      if (!this.grips) this.grips = [{ joint: null }, { joint: null }];
      this.grips[0].joint = null;
      this.grips[1].joint = null;
      this.grippable = [];
      this.gripCount = 0;
      this.harnessX = null;
      this.trackSum = 0;
      this.trackCount = 0;
      // STRENGTH: every motor torque is the built torque times options.strength (1 = the real
      // body). The get-up curriculum starts strong and weakens to 1 (Tao et al. 2022, "Learning to
      // Get Up": a strong character discovers the motion; at normal strength exploration gets stuck).
      const strength = this.options.strength > 0 ? this.options.strength : 1;
      if (b2RevoluteJoint_SetMaxMotorTorque && world.joints[0] && world.joints[0].torque) {
        for (let i = 0; i < world.joints.length; i++)
          b2RevoluteJoint_SetMaxMotorTorque(world.joints[i].id, world.joints[i].torque * strength);
      }
      this.strength = strength;
      this.swing = null;
      this.swingMarkBase = null;
      if (this.options.swing) {
        readBodyState(world, this.measure);
        this.swing = spawnSwing(world, this.measure.partX[config.partIndex.torso] || 0, this.objects);
      }
      this.board = null;
      if (this.options.board) {
        readBodyState(world, this.measure);
        this.boardX = this.measure.partX[config.partIndex.torso] || 0;
        this.board = spawnBoard(world, this.boardX, this.objects);
        this.boardAngle = 0;
      }
      const settleFrames =
        this.options.settle != null ? this.options.settle : 0;
      for (let i = 0; i < settleFrames; i++)
        b2World_Step(
          world.worldId,
          config.physics.timeStep,
          config.physics.solverSubsteps,
        );
      for (let i = 0; i < world.joints.length; i++)
        this.previousJointAngle[i] = b2RevoluteJoint_GetAngle(
          world.joints[i].id,
        );
      readBodyState(world, this.measure);
      // SWING: the push, after the doll has settled on the plank.
      if (this.swing && this.options.swing.push) b2Body_SetLinearVelocity(this.swing.plank, vector(this.options.swing.push, 0));
      if (this.options.failOnHead) {
        stampTrainTargets(this.measure, this.options);
        placeSkillBoxes(this);
      }
      this.walkStartX = this.measure.centerX || 0;
      // On the swing the whole body rides the plank, which moves at up to ~1.2 m/s: the 0.5 m/s
      // still-standing limit would fail every frame of a good balance.
      this.measure.standSpeedLimit = this.options.walk
        ? config.walk.maxStandSpeed
        : this.options.swing
          ? 3
          : 0.5;
      this.measure.plantedWeight = config.reward.planted || 0; // While walking: one foot down is enough (reward.js).
      this.measure.walking = !!this.options.walk;
      this.measure.getup = !!this.options.getup;
      this.measure.walkSpeed = 0;
      this.measure.walkDir = 0;
      if (this.options.walk && this.options.targets) {
        for (let i = 0; i < config.targets.length; i++) {
          const target = this.options.targets[i];
          if (target && config.targets[i].ride) {
            target.walkBase = target.x || 0;
            target.walkBaseY = target.y || 0;
          }
        }
      }
      this.climbBox = null;
      this.climbTopFrames = 0;
      const climb = this.options.climb;
      if (climb && climb.height > 0) {
        const mid =
          ((this.measure.partX[config.partIndex.footLeft] || 0) +
            (this.measure.partX[config.partIndex.footRight] || 0)) / 2;
        const half = climb.half || config.climb.boxHalfWidth;
        const side = climb.side || 1;
        const x0 = side > 0 ? mid + climb.near : mid - climb.near - 2 * half;
        const id = spawnStaticBox(world, x0 + half, half, climb.height / 2);
        this.objects.push(id);
        this.climbBox = { x0, x1: x0 + 2 * half, top: climb.height, side, high: !!climb.high };
        this.grippable.push({ id, cx: x0 + half, cy: climb.height / 2 });
        if (climb.high) highboxMarks(this.options, this.climbBox);
      }
      // STAIRS: n steps of rise x run starting near the feet, then a landing. Each step is a static box
      // from its front edge to the end of the landing, (i+1) x rise tall. this.climbBox is the first.
      // TERRAIN: extra static blocks (uneven ground, steps) and loose dynamic boxes, placed relative to
      // the feet at the start: options.terrain = { statics: [{dx, w, h}], dynamics: [{dx, w, h, density}] }.
      this.terrain = null;
      const terrain = this.options.terrain;
      if (terrain) {
        const mid =
          ((this.measure.partX[config.partIndex.footLeft] || 0) +
            (this.measure.partX[config.partIndex.footRight] || 0)) / 2;
        this.terrain = { statics: [], dynamics: [] };
        for (const b of terrain.statics || []) {
          const id = spawnStaticBox(world, mid + b.dx + b.w / 2, b.w / 2, b.h / 2);
          this.objects.push(id);
          this.terrain.statics.push({ x0: mid + b.dx, x1: mid + b.dx + b.w, top: b.h, side: 1 });
          this.grippable.push({ id, cx: mid + b.dx + b.w / 2, cy: b.h / 2 });
        }
        for (const b of terrain.dynamics || []) {
          const id = CreateBoxPolygon({
            worldId: world.worldId, type: dynamicBody,
            position: vector(mid + b.dx + b.w / 2, (b.y || 0) + b.h / 2), size: vector(b.w / 2, b.h / 2),
            density: b.density || 20, friction: 0.8,
          }).bodyId;
          world.thrownIds.push(id);
          world.thrownShapes.push(world.shapeOf(id));
          this.objects.push(id);
          this.terrain.dynamics.push(id);
        }
      }
      this.stairs = null;
      const stairs = this.options.stairs;
      if (stairs && stairs.n > 0) {
        const mid =
          ((this.measure.partX[config.partIndex.footLeft] || 0) +
            (this.measure.partX[config.partIndex.footRight] || 0)) / 2;
        const start = mid + (stairs.near || 0.3);
        const end = start + stairs.n * stairs.run + (stairs.landing || 1);
        this.stairs = [];
        for (let i = 0; i < stairs.n; i++) {
          const x0 = start + i * stairs.run;
          const top = (i + 1) * stairs.rise;
          const id = spawnStaticBox(world, (x0 + end) / 2, (end - x0) / 2, top / 2);
          this.objects.push(id);
          this.stairs.push({ x0, x1: end, top, side: 1 });
          this.grippable.push({ id, cx: (x0 + end) / 2, cy: top / 2 });
        }
        this.climbBox = this.stairs[0];
      }
      this.measure.climbBox = this.climbBox;
      if (this.options.getup) getupMarks(this.measure, this.options);
      // BOARD: the foot marks sit on the plank, not on the floor. Planted at floor height they were
      // 13 cm under the feet, and a net trained to put its feet on their marks stepped off a still
      // plank within 2 s.
      if ((this.board || this.swing) && this.options.targets) {
        const top = this.swing ? config.swing.top : config.board.pivotHeight + 2 * config.board.halfThick;
        for (let i = 0; i < config.targets.length; i++) {
          if (config.targets[i].ride) continue;
          const target = this.options.targets[i];
          if (target) target.y = top + config.skill.footGroundMeters;
        }
      }
      this.measure.targets = episodeTargets(this.options);
      measureProximity(world, this.measure);
    }
    cleanup() {
      const objects = this.objects;
      for (let i = 0; i < objects.length; i++)
        destroyThrown(this.world, objects[i]);
      objects.length = 0;
    }
    // Read the body into this.input and write a new action. Does not score.
    decide() {
      const world = this.world;
      this.measure.targets = episodeTargets(this.options);
      const measure = this.measure;
      const footMid = (measure.footXLeft + measure.footXRight) / 2;
      const input = this.input;
      let cursor = 0;
      for (let i = 0; i < config.net.jointCount; i++) {
        const joint = world.joints[i];
        const angle = b2RevoluteJoint_GetAngle(joint.id);
        input[cursor + i] = angle / joint.n;
        input[cursor + config.net.jointCount + i] = clamp(
          (angle - this.previousJointAngle[i]) /
            (config.cadence.actionEveryNFrames * config.physics.timeStep) /
            6,
          -1,
          1,
        );
        this.previousJointAngle[i] = angle;
      }
      cursor += 2 * config.net.jointCount;
      input[cursor++] = measure.torsoSin;
      input[cursor++] = measure.torsoCos;
      input[cursor++] = measure.headSin;
      input[cursor++] = measure.headCos;
      input[cursor++] = measure.headHeight / config.body.headHeightMeters;
      input[cursor++] = measure.torsoHeight / config.body.torsoHeightMeters;
      input[cursor++] = clamp((measure.centerX - footMid) / 0.3, -2, 2);
      input[cursor++] = measure.centerY - 0.85;
      input[cursor++] = measure.centerVelocityX / 2;
      input[cursor++] = measure.centerVelocityY / 2;
      // Speed of the five commanded parts. The net used to see the position error and not
      // how fast the part was moving: a hand flying toward the mark and one crawling
      // toward it looked almost the same. With this it can brake.
      for (let v = 0; v < config.targets.length; v++) {
        const vp = config.partIndex[config.targets[v].part];
        input[cursor++] = clamp(
          (measure.partVelocityX[vp] || 0) / config.body.velocityScale,
          -1,
          1,
        );
        input[cursor++] = clamp(
          (measure.partVelocityY[vp] || 0) / config.body.velocityScale,
          -1,
          1,
        );
      }
      // Every part except the pelvis, in the torso frame. The pelvis is the origin, so a
      // neck movement does not drag the feet across the input.
      const torso = config.partIndex.torso;
      const originX = measure.partX[torso] || 0;
      const originY = measure.partY[torso] || 0;
      const torsoCos = measure.torsoCos || 1;
      const torsoSin = measure.torsoSin || 0;
      const pelvisSpan = config.body.pelvisFrameMeters || 1;
      for (let i = 0; i < config.net.partCount; i++) {
        if (i === torso) continue;
        const dx = (measure.partX[i] || 0) - originX;
        const dy = (measure.partY[i] || 0) - originY;
        input[cursor++] = clamp((torsoCos * dx + torsoSin * dy) / pelvisSpan, -2, 2);
        input[cursor++] = clamp((-torsoSin * dx + torsoCos * dy) / pelvisSpan, -2, 2);
      }
      const face = measure.faceRay;
      for (let i = 0; i < face.length; i++) input[cursor++] = face[i];
      const targets = measure.targets;
      const muted = config.net.mutedTargets;
      // For the parts anchored to the feet, the error is in WORLD axes: the mark lives in
      // the world, so the reward and what the net sees are the same quantity. Rotating it
      // into the torso frame made the error look smaller when the doll was down, which is
      // exactly when it most needs to be seen.
      for (let t = 0; t < config.targets.length; t++) {
        const spec = config.targets[t];
        const target = targets[t];
        const index = config.partIndex[spec.part];
        // A target with gain 0 is out of the reward AND out of the input. Otherwise the net
        // would still see columns for a target that does not count, and the gain ramp would not be real.
        const on =
          target.on &&
          targetGain(spec.id, target) > 0 &&
          !(muted && muted.indexOf(spec.id) >= 0)
            ? 1
            : 0;
        input[cursor++] = on;
        if (!on) {
          input[cursor++] = 0;
          input[cursor++] = 0;
          input[cursor++] = 0;
          input[cursor++] = 0;
          input[cursor++] = 0;
          input[cursor++] = 0;
          continue;
        }
        // The normaliser is the EPISODE span, not the old 2 m: at 2 m a 20 cm command
        // enters as 0.1 and the net barely sees it.
        const span = targetSpan(spec, target);
        let ex = (target.x || 0) - measure.partX[index];
        let ey = (target.y || 0) - measure.partY[index];
        input[cursor++] = clamp(ex / span, -1, 1);
        input[cursor++] = clamp(ey / span, -1, 1);
        const targetSin = Math.sin(target.angle || 0);
        const targetCos = Math.cos(target.angle || 0);
        const partSin = measure.partSin[index];
        const partCos = measure.partCos[index];
        input[cursor++] = targetSin * partCos - targetCos * partSin;
        input[cursor++] = targetCos * partCos + targetSin * partSin;
        const wide = config.reward.wideSpanMeters || 1;
        input[cursor++] = clamp(ex / wide, -2, 2);
        input[cursor++] = clamp(ey / wide, -2, 2);
      }
      input[cursor++] = measure.headContact || 0;
      const feet = measure.footContact;
      input[cursor++] = feet ? feet[0] : 0;
      input[cursor++] = feet ? feet[1] : 0;
      // The same 0..1 amounts the reward penalises. Rays miss a thigh lying flat, and the
      // torso frame does not say which part is on the floor.
      const contact = measure.contact;
      const groups = config.contact.groups;
      for (let g = 0; g < groups.length; g++) {
        let amount = contact ? contact[g] || 0 : 0;
        if (amount < 0) amount = 0;
        else if (amount > 1) amount = 1;
        input[cursor++] = amount;
      }
      if (cursor !== config.net.sensedInputCount)
        throw new Error("base observation " + cursor);
      if (
        config.net.rewardTotalNow ||
        config.net.rewardTotalPrevious ||
        config.net.rewardTotalNext
      ) {
        const scored = scoreDecision(measure);
        const rewardColumn = (total) =>
          clamp(total / config.scale.rewardInput, -2, 2);
        if (config.net.rewardTotalNow)
          input[cursor++] = rewardColumn(scored.total);
        if (config.net.rewardTotalPrevious)
          input[cursor++] = rewardColumn(this.previousRewardTotal);
        if (config.net.rewardTotalNext) {
          const predicted = predictMeasure(
            world,
            measure,
            config.physics.timeStep,
          );
          input[cursor++] = rewardColumn(scoreDecision(predicted).total);
        }
        this.previousRewardTotal = scored.total;
      }
      // Walk inputs: a gait clock and the commanded speed. Zero when this is not a walk.
      const walk = this.options.walk;
      if (config.net.walkInputCount) {
        if (this.options.getup) {
          // GET-UP clock: time since the start of the get-up, as a half turn over getup.clockSeconds,
          // so the net knows where it is in the motion it imitates. Same slots as the gait clock.
          const t = this.getupTime();
          const phase = Math.PI * Math.min(1, t / (config.getup.clockSeconds || 5));
          input[cursor++] = Math.sin(phase);
          input[cursor++] = Math.cos(phase);
          input[cursor++] = 0;
        } else if (walk) {
          const phase = (2 * Math.PI * this.frame) / (config.physics.hz * (config.walk.periodSeconds || 1));
          input[cursor++] = Math.sin(phase);
          input[cursor++] = Math.cos(phase);
          input[cursor++] = (walk.dir * walk.speed) / (config.walk.maxSpeed || 1);
        } else {
          input[cursor++] = 0;
          input[cursor++] = 0;
          input[cursor++] = 0;
        }
        const box = this.climbBox;
        if (box) {
          input[cursor++] = 1;
          // Signed distance to the nearest edge facing the doll: positive ahead, negative behind.
          const near = box.side < 0 ? box.x1 : box.x0;
          input[cursor++] = clamp(near - footMid, -1, 1);
          input[cursor++] = box.top / 0.5;
        } else {
          input[cursor++] = 0;
          input[cursor++] = 0;
          input[cursor++] = 0;
        }
        input[cursor++] = this.grips[0].joint ? 1 : 0;
        input[cursor++] = this.grips[1].joint ? 1 : 0;
      }
      if (cursor !== config.net.inputCount)
        throw new Error("observation length " + cursor);
      this.network.forward(input);
    }
    sampleReward() {
      const tracing = !!(this.options && this.options.trace);
      const scored = tracing ? scoreDecision(this.measure) : null;
      const total = scored ? scored.total : scoreTotal(this.measure);
      rememberAxisErrors(this.measure);
      if (tracing) {
        if (!this.trace) this.trace = [];
        const src = this.action;
        const action = [];
        for (let i = 0; i < src.length; i++) action.push(src[i]);
        const termList = scored.terms;
        const rows = [];
        for (let i = 0; i < termList.length; i++) rows.push(termList[i].v);
        const amounts = this.measure.contact;
        const contact = [];
        const contactCount = amounts ? amounts.length : 0;
        for (let i = 0; i < contactCount; i++) contact.push(amounts[i]);
        this.trace.push({ action, rows, contact, total });
      }
      if (this.startCount < this.startSamples) {
        this.startSum += total;
        this.startCount++;
      }
      this.lifeSum += total;
      this.lifeCount++;
      this.endRing[this.endCursor] = total;
      this.endCursor++;
      if (this.endCursor >= this.endRing.length) this.endCursor = 0;
      if (this.endFilled < this.endRing.length) this.endFilled++;
    }
    windowMean(fromEnd) {
      if (!fromEnd)
        return this.startCount ? this.startSum / this.startCount : 0;
      const n = this.endFilled;
      if (!n) return 0;
      let sum = 0;
      for (let i = 0; i < n; i++) sum += this.endRing[i];
      return sum / n;
    }
    trackStandingStreak() {
      const measure = this.measure;
      const standing = isStanding(measure);
      this.fallen = measure.bodyOn || measure.torsoCos < 0.25;
      if ((measure.torsoCos || 0) < 0.25) this.torsoDown = true;
      if (standing) {
        this.upStreak++;
        this.stoodFrames++;
      } else this.upStreak = 0;
      this.sensedFrames++;
      this.heightSum += measure.headHeight;
      if (measure.touch[config.partIndex.head]) {
        this.headHit = true;
        this.headImpact = Math.max(
          this.headImpact || 0,
          -(measure.headVelocityY || 0),
        );
      }
      const height = clamp(
        measure.headHeight / config.body.standingHeadMeters,
        0,
        1.25,
      );
      this.uprightSum += Math.max(0, measure.torsoCos || 0) * height;
      const total = this.options.frames || config.episode.frames;
      const endFrames = Math.round(
        config.episode.endWindowSeconds * config.physics.hz,
      );
      if (this.frame >= total - endFrames) {
        this.windowFrames++;
        if (standing) this.windowStood++;
      }
    }
    advanceOneFrame() {
      const frame = this.frame + (this.options.phase || 0);
      const acting = frame % config.cadence.actionEveryNFrames === 0;
      this.measure.targets = episodeTargets(this.options);
      this.moveWalkMarks();
      this.moveSwingMarks();
      readBodyState(this.world, this.measure);
      // Marks are not replanted here: they are planted once in reset() and stay frozen in
      // the world, all five. Replanting every frame made the body marks chase the feet,
      // which is what made a dragged mark jump in the sandbox.
      this.trackStandingStreak();
      this.trackReference();
      armPlace(this);
      armWalk(this);
      maybeAdvance(this);
      if (
        this.options.ballFrame != null &&
        this.frame === this.options.ballFrame &&
        !this.ballThrown
      ) {
        this.ballThrown = true;
        const torso = config.partIndex.torso;
        const x = this.measure.partX[torso];
        const y = this.measure.partY[torso];
        const side = this.options.ballSide || 1;
        const id = spawnThrown(
          this.world,
          thrownKinds[0],
          x + side * config.start.ballStartMeters,
          Math.max(y, 0.6),
          -side * (this.options.ballSpeed || config.start.ballSpeed),
          0.4,
        );
        this.objects.push(id);
      }
      if (acting) {
        measureProximity(this.world, this.measure);
        if (!this.passive) this.decide();
      }
      if (
        !this.quiet &&
        frame % config.cadence.rewardEveryNFrames ===
          config.cadence.rewardFrameOffset
      )
        this.sampleReward();
      this.trackGoalProgress();
      this.driveBoard();
      this.updateGrip();
      this.applyMotors();
      this.stepPhysics();
      this.frame++;
      if (
        this.frame >= (this.options.frames || config.episode.frames) &&
        !this.free
      ) {
        this.endedStanding = isStanding(this.measure);
        this.endedUp = this.upStreak >= 15;
        return true;
      }
      return false;
    }
    // BOTH foot centres at least box top - slack, and inside the box in x. One foot was not
    // enough: the first climb run learned to lunge, front foot on the box and back foot on the
    // floor, and held it to the end.
    footOnTop() {
      return feetOnBox(this.measure, this.climbBox) === 2;
    }
    // GRIP: a hand whose grip output is above grip.on and that touches a grippable object (a
    // climb box) is pinned to it with a revolute joint at the hand centre, so the wrist can still
    // turn. It lets go when the output drops below grip.off. Joints die with the box at cleanup.
    updateGrip() {
      const world = this.world;
      const action = this.network.action;
      const base = config.net.jointCount;
      if (!(config.net.gripOutputs > 0) || action.length < base + 2) return;
      for (let k = 0; k < 2; k++) {
        const grip = this.grips[k];
        const out = action[base + k];
        if (grip.joint) {
          if (out < config.grip.off) {
            if (world.b2DestroyJoint) world.b2DestroyJoint(grip.joint);
            grip.joint = null;
          }
          continue;
        }
        if (out <= config.grip.on || !this.grippable.length) continue;
        const handPart = k === 0 ? config.partIndex.handLeft : config.partIndex.handRight;
        const hand = world.parts[handPart];
        const target = touchingGrippable(world, hand.id, this.grippable);
        if (!target) continue;
        const hx = this.measure.partX[handPart] || 0;
        const hy = this.measure.partY[handPart] || 0;
        grip.joint = world.CreateRevoluteJoint({
          worldId: world.worldId, bodyIdA: hand.id, bodyIdB: target.id,
          anchorA: vector(0, 0), anchorB: vector(hx - target.cx, hy - target.cy),
          enableLimit: false, enableMotor: false, collideConnected: false,
        }).jointId;
        this.gripCount = (this.gripCount || 0) + 1;
      }
    }
    // Seconds into the get-up: frames so far plus the start time of a reference-state start (RSI).
    getupTime() {
      return this.frame / config.physics.hz + (this.options.startTime || 0);
    }
    // GET-UP imitation (DeepMimic-style): the reference is a list of key poses (joint commands) with
    // durations, the same blending as getup-keyframes.js. measure.trackError is the mean joint-angle
    // error to the reference's target angles, in radians; the reward turns it into a row. After the
    // reference ends there is nothing to track and standing takes over.
    trackReference() {
      const ref = this.options.reference;
      const m = this.measure;
      m.trackError = -1;
      if (!ref || !ref.length) return;
      const t = this.getupTime();
      let start = 0;
      let prev = null;
      let found = null;
      let blend = 1;
      for (let k = 0; k < ref.length; k++) {
        const f = ref[k];
        if (t < start + f.dur) {
          found = f;
          blend = prev ? Math.min(1, (t - start) / (0.4 * f.dur)) : 1;
          break;
        }
        start += f.dur;
        prev = f.pose;
      }
      if (!found) return;
      const joints = this.world.joints;
      let err = 0;
      for (let j = 0; j < config.net.jointCount; j++) {
        const cmd = prev ? prev[j] + (found.pose[j] - prev[j]) * blend : found.pose[j];
        const target = cmd * (cmd >= 0 ? joints[j].hi : -joints[j].lo);
        err += Math.abs(target - b2RevoluteJoint_GetAngle(joints[j].id));
      }
      m.trackError = err / config.net.jointCount;
      this.trackSum = (this.trackSum || 0) + m.trackError;
      this.trackCount = (this.trackCount || 0) + 1;
    }
    // The plank's pivot motor chases a slow sine; its torque is capped, so the doll can push back.
    driveBoard() {
      const board = this.board;
      const spec = this.options.board;
      if (!board || !spec) return;
      const t = this.frame / config.physics.hz;
      const wave = spec.amp * Math.sin((2 * Math.PI * t) / spec.period + spec.phase);
      const angle = b2RevoluteJoint_GetAngle(board.joint);
      b2RevoluteJoint_SetMotorSpeed(board.joint, config.board.motorGain * (wave - angle));
      this.boardAngle = angle;
    }
    // WALK: the head and hand marks are still frozen in height, but their x slides at the
    // commanded speed once the delay is over. The same position rows that make the doll
    // stand now pay it to keep up, and the upright and contact rows still charge a fall.
    walkOffset() {
      const walk = this.options.walk;
      if (!walk) return 0;
      const t = this.frame / config.physics.hz - (walk.delay || 0);
      return t > 0 ? walk.dir * walk.speed * t : 0;
    }
    // SWING: every mark rides with the plank. Left where they started, the foot marks paid the doll
    // to step off the moving plank (standing fell from 29% to 15% in training).
    moveSwingMarks() {
      if (!this.swing || !this.options.targets) return;
      const t = Box2D_GetPosition(this.swing.plank);
      if (this.swingMarkBase == null) {
        this.swingMarkBase = { x: t.x, y: t.y, marks: this.options.targets.map((m) => (m ? { x: m.x, y: m.y } : null)) };
      }
      const dx = t.x - this.swingMarkBase.x;
      const dy = t.y - this.swingMarkBase.y;
      for (let i = 0; i < this.options.targets.length; i++) {
        const m = this.options.targets[i];
        const b = this.swingMarkBase.marks[i];
        if (!m || !b) continue;
        m.x = b.x + dx;
        m.y = b.y + dy;
      }
    }
    moveWalkMarks() {
      if (!this.options.walk || !this.options.targets) return;
      const walk = this.options.walk;
      // What the speed row asks for right now: nothing during the delay.
      this.measure.walkDir = walk.dir;
      this.measure.walkSpeed = this.frame / config.physics.hz > (walk.delay || 0) ? walk.speed : 0;
      // The same clock the net sees, for the periodic gait reward (reward.js).
      this.measure.gaitPhase = (2 * Math.PI * this.frame) / (config.physics.hz * (config.walk.periodSeconds || 1));
      this.measure.partVX = this.world.vx;
      const offset = this.walkOffset();
      const box = this.climbBox;
      const ramp = config.climb.liftRamp || 0.3;
      for (let i = 0; i < config.targets.length; i++) {
        if (!config.targets[i].ride) continue;
        const target = this.options.targets[i];
        if (target && target.on && target.walkBase != null) {
          target.x = target.walkBase + offset;
          // Climb: the mark rises by the box height as it passes the near edge.
          if (box && target.walkBaseY != null) {
            let k = (target.x - box.x0) / ramp;
            k = k < 0 ? 0 : k > 1 ? 1 : k;
            target.y = target.walkBaseY + k * box.top;
          }
        }
      }
    }
    // The episode owns the clock. The small net only says how to move hip, knee, and ankle.
    applyScriptedGait() {
      const add = this.legAdd;
      add.fill(0);
      if (!config.skill.scriptedLegs) return;
      if (!this.options || (this.options.skill | 0) < 2) return;
      if (!this.placeArmed) return;
      const farMark = Math.abs(this.placeExtra || 0) > 0.18;
      if (this.frame === this.phaseFrame) {
        this.legBase.set(this.action);
        this.legLocked = true;
        this.stepXLeft = this.measure.partX[config.partIndex.footLeft] || 0;
        this.rewind = 0;
        this.extraAt = 0;
        this.extraDecided = 0;
        this.extraElapsed = 0;
        this.extraWhich = 0;
        this.leadDownFrame = 0;
        this.leadWasUp = 0;
        this.pulse2Dx = null;
        this.pulse2FootL = null;
        this.pulse2FootR = null;
        this.pulse2LeanSign = 0;
        this.place3At = 0;
        this.place3Which = 0;
        this.place3Dx = null;
        this.place3FootL = null;
        this.place3FootR = null;
        if (this.options.replayRight) {
          this.swingX0 = this.measure.partX[config.partIndex.torso] || 0;
          this.swingL = null;
          this.swingR = null;
          this.restWait = 0;
          this.restOk = 0;
          this.place3In = 0;
          this.place3Err0 = 0;
          this.replayWhy = "";
          if (this.options.boxes) placeSkillBoxes(this);
        }
      }
      const single = (this.options.skill | 0) >= 3;
      const lead = (this.placeExtra || 0) >= 0 ? 1 : -1;
      const first = this.options.leadFlip ? -lead : lead;
      const extraPulses = this.options.retries | 0;
      const rightFar = farMark && (this.placeExtra || 0) > 0;
      const span0 = rightFar ? 40 : this.options.spanFrames || 80;
      const span1 = this.options.span2Frames || span0;
      const rewindSpan = 40;
      const wall = Math.max(0, this.frame - this.phaseFrame);
      if (!this.leadDownFrame && wall > 12) {
        const leadPart =
          first > 0 ? config.partIndex.footRight : config.partIndex.footLeft;
        const y = this.measure.partY[leadPart] || 0;
        if (y > 0.06) this.leadWasUp = 1;
        if (this.leadWasUp && y < 0.05) this.leadDownFrame = this.frame;
      }
      // The second slide starts when the first foot that lifted comes back to the floor.
      // Only if more than 8 cm is still missing and the mark is past 18 cm. 0.12 and 0.16 do not use it.
      const footDown =
        this.leadDownFrame === this.frame && wall >= span0 + span1;
      const before150 = farMark && this.frame === 145;
      if (
        !single &&
        !this.extraAt &&
        this.frame < 150 &&
        (footDown || before150)
      ) {
        const episodeFrames = this.options.frames || 360;
        const err = (slot, part) => {
          const target = this.options.targets && this.options.targets[slot];
          return (target ? target.x : 0) - (this.measure.partX[part] || 0);
        };
        const eL = err(targetSlot("footL"), config.partIndex.footLeft);
        const eR = err(targetSlot("footR"), config.partIndex.footRight);
        const gap = Math.max(Math.abs(eL), Math.abs(eR));
        const dir = (this.placeExtra || 0) >= 0 ? 1 : -1;
        const room = this.frame + 80 <= episodeFrames - 90;
        if (room && farMark && dir > 0 && gap > 0.08 && (eL + eR) * dir > 0) {
          this.extraAt = this.frame;
          this.extraElapsed = wall;
          this.extraWhich = 1;
          this.pulse2LeanSign = -1;
        }
      }
      let elapsed = wall;
      const slideSpan = 80;
      const placeSpan = 32;
      if (this.extraAt) {
        const into = this.frame - this.extraAt;
        const placeInto =
          this.place3At > 0 ? this.frame - this.place3At : placeSpan;
        const busy = into < slideSpan || placeInto < placeSpan;
        elapsed = busy ? this.extraElapsed : span0 + span1 + rewindSpan + 1;
      }
      let phase = 1;
      let which = first;
      let holding = false;
      let index = 0;
      if (single || elapsed < span0) {
        index = 0;
        phase = Math.min(1, elapsed / span0);
        which = first;
        if (single && elapsed >= span0) {
          which = 0;
          holding = true;
          this.legLocked = false;
        }
      } else if (elapsed < span0 + span1) {
        index = 1;
        phase = (elapsed - span0) / span1;
        which = -first;
      } else if (
        !single &&
        !this.extraAt &&
        (this.placeExtra || 0) < 0 &&
        elapsed < span0 + span1 + rewindSpan
      ) {
        if (!this.rewind) {
          const slide =
            (this.measure.partX[config.partIndex.footLeft] || 0) -
            (this.stepXLeft || 0);
          this.rewind = slide < 0.16 ? 1 : -1;
        }
        if (this.rewind > 0) {
          index = 2;
          phase = (elapsed - span0 - span1) / rewindSpan;
          which = 1;
          this.legLocked = true;
        } else {
          which = 0;
          phase = 1;
          holding = true;
          this.legLocked = false;
        }
      } else if (
        extraPulses > 0 &&
        elapsed < span0 + span1 + extraPulses * span0
      ) {
        index = 2;
        phase = ((elapsed - span0 - span1) % span0) / span0;
        const miss = (slot, part) => {
          const target = this.options.targets && this.options.targets[slot];
          return Math.abs(
            (target ? target.x : 0) - (this.measure.partX[part] || 0),
          );
        };
        const left = miss(targetSlot("footL"), config.partIndex.footLeft);
        const right = miss(targetSlot("footR"), config.partIndex.footRight);
        if (Math.max(left, right) <= 0.08) {
          which = 0;
          holding = true;
          this.legLocked = false;
        } else which = left >= right ? -1 : 1;
      } else {
        which = 0;
        phase = 1;
        holding = true;
        this.legLocked = false;
      }
      if (this.extraAt) {
        const into = this.frame - this.extraAt;
        const pelvis = this.measure.partX[config.partIndex.torso] || 0;
        if (into === 0) {
          this.pulse2X0 = pelvis;
          this.pulse2Foot = this.extraWhich || 0;
          const shift = (this.placeExtra || 0) >= 0 ? 1 : -1;
          this.pulse2Shift = shift;
          this.pulse2Lean = this.pulse2LeanSign || 0;
          this.pulse2Sin = 0;
          this.pulse2SinN = 0;
          this.pulse2FootL0 =
            this.measure.partX[config.partIndex.footLeft] || 0;
          this.pulse2FootR0 =
            this.measure.partX[config.partIndex.footRight] || 0;
        }
        if (into === slideSpan - 1) {
          const missOf = (slot, part) => {
            const target = this.options.targets && this.options.targets[slot];
            return Math.abs(
              (target ? target.x : 0) - (this.measure.partX[part] || 0),
            );
          };
          this.pulse2ErrL = missOf(
            targetSlot("footL"),
            config.partIndex.footLeft,
          );
          this.pulse2ErrR = missOf(
            targetSlot("footR"),
            config.partIndex.footRight,
          );
        }
        if (into === slideSpan && !this.place3At) {
          this.place3At = -1;
          if (
            this.extraWhich > 0 &&
            (this.pulse2ErrR || 0) > 0.08 &&
            (this.pulse2ErrL || 0) <= 0.06
          ) {
            this.place3At = this.frame;
            for (const i of [5, 6, 7, 11, 12, 13]) {
              const joint = this.world.joints[i];
              const angle = b2RevoluteJoint_GetAngle(joint.id);
              const span = angle >= 0 ? joint.hi : -joint.lo;
              this.legBase[i] = span ? clamp(angle / span, -1, 1) : 0;
            }
            this.place3Which = 1;
            this.place3Lean = 0;
            this.place3X0 = pelvis;
            this.place3L0 = this.measure.partX[config.partIndex.footLeft] || 0;
            this.place3R0 = this.measure.partX[config.partIndex.footRight] || 0;
          }
        }
        const placeInto = this.place3At > 0 ? this.frame - this.place3At : -1;
        if (into >= 0 && into < slideSpan) {
          index = 3;
          phase = into / slideSpan;
          which = this.extraWhich || 0;
          holding = false;
          this.legLocked = true;
          this.pulse2Sin += this.measure.torsoSin || 0;
          this.pulse2SinN = (this.pulse2SinN || 0) + 1;
          if (into === slideSpan - 1) {
            this.pulse2Dx = pelvis - (this.pulse2X0 || 0);
            this.pulse2FootL =
              (this.measure.partX[config.partIndex.footLeft] || 0) -
              (this.pulse2FootL0 || 0);
            this.pulse2FootR =
              (this.measure.partX[config.partIndex.footRight] || 0) -
              (this.pulse2FootR0 || 0);
          }
        } else if (
          placeInto >= 0 &&
          placeInto < placeSpan &&
          this.place3At > 0
        ) {
          const shoved = pelvis - (this.place3X0 || 0);
          if (shoved > 0.04 || shoved < -0.22) {
            this.place3Dx = shoved;
            this.place3FootL =
              (this.measure.partX[config.partIndex.footLeft] || 0) -
              (this.place3L0 || 0);
            this.place3FootR =
              (this.measure.partX[config.partIndex.footRight] || 0) -
              (this.place3R0 || 0);
            this.place3At = -1;
          } else {
            index = 4;
            phase = placeInto / placeSpan;
            which = this.place3Which || 0;
            holding = false;
            this.legLocked = true;
            if (placeInto === placeSpan - 1) {
              this.place3Dx = shoved;
              this.place3FootL =
                (this.measure.partX[config.partIndex.footLeft] || 0) -
                (this.place3L0 || 0);
              this.place3FootR =
                (this.measure.partX[config.partIndex.footRight] || 0) -
                (this.place3R0 || 0);
            }
          }
        }
      }
      // Second step: the right knee flexes 0.05 rad more, the same length. Then the right foot plants.
      if (
        this.options.replayRight &&
        !single &&
        (this.placeExtra || 0) > 0 &&
        !this.extraAt &&
        wall >= span0
      ) {
        const pelvis = this.measure.partX[config.partIndex.torso] || 0;
        const missOf = (slot, part) => {
          const target = this.options.targets && this.options.targets[slot];
          return Math.abs(
            (target ? target.x : 0) - (this.measure.partX[part] || 0),
          );
        };
        const left = missOf(targetSlot("footL"), config.partIndex.footLeft);
        const right = missOf(targetSlot("footR"), config.partIndex.footRight);
        if (this.swingL == null) {
          this.swingL = left;
          this.swingR = right;
          this.swingDx = pelvis - (this.swingX0 || 0);
        }
        const snapLegs = () => {
          for (const i of [5, 6, 7, 11, 12, 13]) {
            const joint = this.world.joints[i];
            const angle = b2RevoluteJoint_GetAngle(joint.id);
            const span = angle >= 0 ? joint.hi : -joint.lo;
            this.legBase[i] = span ? clamp(angle / span, -1, 1) : 0;
          }
        };
        const placeInto = this.place3At > 0 ? this.frame - this.place3At : -1;
        if (placeInto >= 0 && placeInto < placeSpan) {
          const shoved = pelvis - (this.place3X0 || 0);
          if (left <= 0.08) this.place3In = 1;
          const leftAway =
            this.place3In && left > (this.place3Err0 || 0) + 0.04;
          const back = shoved < -0.22;
          if (leftAway || back) {
            this.place3Dx = shoved;
            this.place3FootL =
              (this.measure.partX[config.partIndex.footLeft] || 0) -
              (this.place3L0 || 0);
            this.place3FootR =
              (this.measure.partX[config.partIndex.footRight] || 0) -
              (this.place3R0 || 0);
            this.replayWhy =
              leftAway && back ? "left+back" : leftAway ? "left" : "back";
            this.place3At = -1;
            which = 0;
            holding = true;
            this.legLocked = false;
          } else {
            index = 4;
            phase = placeInto / placeSpan;
            which = 1;
            holding = false;
            this.legLocked = true;
            if (placeInto === placeSpan - 1) {
              this.place3Dx = shoved;
              this.place3FootL =
                (this.measure.partX[config.partIndex.footLeft] || 0) -
                (this.place3L0 || 0);
              this.place3FootR =
                (this.measure.partX[config.partIndex.footRight] || 0) -
                (this.place3R0 || 0);
              this.replayWhy = "place";
              this.place3At = -1;
            }
          }
        } else if (this.place3At === -1) {
          which = 0;
          holding = true;
          this.legLocked = false;
        } else {
          const speed = Math.abs(
            this.measure.partVelocityX[config.partIndex.torso] || 0,
          );
          which = 0;
          holding = true;
          this.legLocked = false;
          if (speed < 0.15 || (this.restWait || 0) >= 90) {
            this.restOk = speed < 0.15 ? 1 : 0;
            this.restN = this.restWait || 0;
            if (right > 0.08) {
              this.place3At = this.frame;
              snapLegs();
              this.place3Which = 1;
              this.place3X0 = pelvis;
              this.place3L0 =
                this.measure.partX[config.partIndex.footLeft] || 0;
              this.place3R0 =
                this.measure.partX[config.partIndex.footRight] || 0;
              this.place3Err0 = left;
              this.place3In = left <= 0.08 ? 1 : 0;
              index = 4;
              phase = 0;
              which = 1;
              holding = false;
              this.legLocked = true;
            } else {
              this.replayWhy = "near";
              this.place3At = -1;
            }
          } else this.restWait = (this.restWait || 0) + 1;
        }
      }
      const swingGate = holding
        ? 0
        : Math.sin(Math.PI * Math.max(0, Math.min(1, phase)));
      const holdGate = holding ? 0.25 : 0;
      if (!holding && which !== 0) {
        const tuned = this.options.kneeAmp;
        let amp = tuned ? (which > 0 ? tuned.right : tuned.left) : 0.4;
        if (index === 4) amp = 0;
        else if (this.options.replayRight && which > 0) amp = 0.4 + 0.05 / 2.4;
        else if (!tuned && which < 0 && (this.placeExtra || 0) > 0) amp = 0.43;
        const support = tuned && tuned.support != null ? tuned.support : 0.1;
        add[which > 0 ? 12 : 6] = -amp * swingGate;
        const extraBend = this.options.extraBend || 0;
        if (extraBend && phase > 0.45) {
          const part =
            which > 0 ? config.partIndex.footRight : config.partIndex.footLeft;
          const slot = targetSlot(which > 0 ? "footR" : "footL");
          const target = this.options.targets && this.options.targets[slot];
          const miss = target
            ? Math.abs((target.x || 0) - (this.measure.partX[part] || 0))
            : 0;
          if (miss > (this.options.extraAt || 0.09))
            add[which > 0 ? 12 : 6] -= extraBend * swingGate;
        }
        if (support && index !== 4)
          add[which > 0 ? 6 : 12] = -support * swingGate;
        const ankleAmp = this.options.ankleAmp || 0;
        if (ankleAmp) add[which > 0 ? 13 : 7] = ankleAmp * swingGate;
        if (index === 4) add[13] = -swingGate;
        const hip = tuned ? (which > 0 ? tuned.hipR || 0 : tuned.hipL || 0) : 0;
        if (hip) add[which > 0 ? 11 : 5] = hip * swingGate;
        if (index >= 2 && tuned && tuned.retryHip)
          add[which > 0 ? 11 : 5] += tuned.retryHip * swingGate;
        const steer = this.options.steer || 0;
        if (steer) {
          const dir = (this.placeExtra || 0) >= 0 ? 1 : -1;
          add[which > 0 ? 11 : 5] += dir * steer * swingGate;
        }
        const waist = this.options.waist == null ? -0.05 : this.options.waist;
        if (waist && index !== 4) {
          const dir = (this.placeExtra || 0) >= 0 ? 1 : -1;
          const leanSign =
            index === 3 && this.pulse2LeanSign
              ? this.pulse2LeanSign
              : dir * (waist < 0 ? -1 : 1);
          add[0] += leanSign * Math.abs(waist) * swingGate;
        }
        const gain = this.options.hipGain || 0;
        if (gain) {
          const part =
            which > 0 ? config.partIndex.footRight : config.partIndex.footLeft;
          const slot = targetSlot(which > 0 ? "footR" : "footL");
          const target = this.options.targets && this.options.targets[slot];
          const err = target
            ? (target.x || 0) - (this.measure.partX[part] || 0)
            : 0;
          const sign = this.options.hipFlip && which < 0 ? -1 : 1;
          add[which > 0 ? 11 : 5] =
            clamp(sign * err * gain, -0.2, 0.2) * swingGate;
        }
      }
      if (holding && this.options.holdKnee) {
        const held = -0.4 * this.options.holdKnee;
        add[6] = held;
        add[12] = held;
      }
      // With a box, the left foot lifts only while it passes over the box. The right hip does not join in.
      if (this.options.boxes && this.options.replayRight) {
        const box = this.liveBoxes && this.liveBoxes[0];
        const footX = this.measure.partX[config.partIndex.footLeft] || 0;
        const half = (box && box.halfW) || 0.04;
        const pad = 0.04;
        const lift =
          this.options.boxLift == null ? 0.182 : this.options.boxLift;
        if (
          box &&
          lift &&
          footX > box.x - half - pad &&
          footX < box.x + half + pad
        )
          add[5] += lift;
      }
      const push = this.options.pushN || 0;
      if (push && swingGate > 0) {
        const dir = (this.placeExtra || 0) >= 0 ? 1 : -1;
        b2Body_ApplyForceToCenter(
          this.world.parts[config.partIndex.torso].id,
          vector(dir * push * swingGate, 0),
          true,
        );
      }
      if (!(this.options.residual && this.options.residual.length)) return;
      const measure = this.measure;
      const targets = this.options.targets;
      const swingRight = which > 0;
      const swingPart = swingRight
        ? config.partIndex.footRight
        : config.partIndex.footLeft;
      const stayPart = swingRight
        ? config.partIndex.footLeft
        : config.partIndex.footRight;
      const swingSlot = targetSlot(swingRight ? "footR" : "footL");
      const staySlot = targetSlot(swingRight ? "footL" : "footR");
      const swing = targets && targets[swingSlot];
      const stay = targets && targets[staySlot];
      const input = this.residualIn;
      input[0] = swing
        ? ((swing.x || 0) - (measure.partX[swingPart] || 0)) / 0.5
        : 0;
      input[1] = swing
        ? ((swing.y || 0) - (measure.partY[swingPart] || 0)) / 0.5
        : 0;
      input[2] = stay
        ? ((stay.x || 0) - (measure.partX[stayPart] || 0)) / 0.5
        : 0;
      input[3] = which;
      input[4] = phase;
      input[5] = measure.torsoSin || 0;
      input[6] = measure.torsoCos || 0;
      input[7] = measure.touch[config.partIndex.footLeft] ? 1 : 0;
      input[8] = measure.touch[config.partIndex.footRight] ? 1 : 0;
      input[9] = measure.stanceWidth || 0;
      input[10] = (measure.centerVelocityX || 0) / 2;
      input[11] = (measure.headHeight || 0) / config.body.standingHeadMeters;
      this.residual.forward(input);
      const legs =
        which > 0
          ? [11, 12, 13]
          : which < 0
            ? [5, 6, 7]
            : [5, 6, 7, 11, 12, 13];
      const from =
        which > 0 ? [3, 4, 5] : which < 0 ? [0, 1, 2] : [0, 1, 2, 3, 4, 5];
      const scale = swingGate + holdGate;
      if (!(scale > 0)) return;
      for (let i = 0; i < legs.length; i++)
        add[legs[i]] += 0.1 * scale * this.residual.out[from[i]];
    }
    applyMotors() {
      if (this.passive) {
        for (let i = 0; i < config.net.jointCount; i++) {
          const id = this.world.joints[i].id;
          b2RevoluteJoint_SetMotorSpeed(id, 0);
          b2RevoluteJoint_EnableMotor(id, false);
          b2RevoluteJoint_EnableSpring(id, false);
        }
        return;
      }
      this.applyScriptedGait();
      // A stronger body is also allowed to move its joints faster (up to 2x).
      const limit = (config.episode.motorSpeedLimit || 1) * Math.min(2, Math.sqrt(this.strength || 1));
      let speedSum = 0;
      for (let i = 0; i < config.net.jointCount; i++) {
        const joint = this.world.joints[i];
        b2RevoluteJoint_EnableMotor(joint.id, true);
        b2RevoluteJoint_EnableSpring(joint.id, true);
        const leg =
          this.legLocked &&
          (i === 5 || i === 6 || i === 7 || i === 11 || i === 12 || i === 13);
        const command = clamp(
          (leg ? this.legBase[i] : this.action[i]) + (this.legAdd[i] || 0),
          -1,
          1,
        );
        const target = command * (command >= 0 ? joint.hi : -joint.lo);
        const speed = clamp(
          config.episode.motorGain *
            (target - b2RevoluteJoint_GetAngle(joint.id)),
          -limit,
          limit,
        );
        speedSum += Math.abs(speed);
        b2RevoluteJoint_SetMotorSpeed(joint.id, speed);
      }
      this.motionSum += speedSum / (config.net.jointCount * limit);
      this.motionCount++;
    }
    // A foot in the air counts when the trunk and the head are still up and that foot is closing on its mark.
    trackGoalProgress() {
      if (!this.options.place || !this.placeArmed) return;
      const measure = this.measure;
      const targets = this.options.targets;
      const feet = [
        [targetSlot("footL"), config.partIndex.footLeft],
        [targetSlot("footR"), config.partIndex.footRight],
      ];
      let gap = 0;
      for (let i = 0; i < feet.length; i++) {
        const target = targets[feet[i][0]];
        const part = feet[i][1];
        const dx = (target.x || 0) - (measure.partX[part] || 0);
        const dy = (target.y || 0) - (measure.partY[part] || 0);
        gap += Math.hypot(dx, dy);
      }
      gap /= 2;
      this.footGap = gap;
      if (this.gap0 == null) this.gap0 = gap;
      this.gapCloseSum += this.gap0 - gap;
      this.gapCloseCount++;
      const mid =
        ((targets[targetSlot("footL")].x || 0) +
          (targets[targetSlot("footR")].x || 0)) /
        2;
      const bodyGap = Math.abs((measure.centerX || 0) - mid);
      if (this.bodyGap0 == null) this.bodyGap0 = bodyGap;
      this.bodyCloseSum += this.bodyGap0 - bodyGap;
      const upright =
        (measure.torsoCos || 0) >
          Math.cos(config.body.standingTorsoAngleRadians) &&
        measure.headHeight > config.body.standingHeadMeters &&
        !measure.touch[config.partIndex.head] &&
        !measure.bodyOn;
      const leftOn = !!measure.touch[config.partIndex.footLeft];
      const rightOn = !!measure.touch[config.partIndex.footRight];
      let air = -1;
      if (upright && leftOn && !rightOn) air = 1;
      else if (upright && rightOn && !leftOn) air = 0;
      const footDx = (index) => {
        const slot = feet[index][0];
        const part = feet[index][1];
        return Math.abs((targets[slot].x || 0) - (measure.partX[part] || 0));
      };
      if (air >= 0) {
        if (this.airPart !== air) {
          this.airPart = air;
          this.airGap = footDx(air);
        }
        const gained = this.airGap - footDx(air);
        if (gained > this.swingPeak) this.swingPeak = gained;
        return;
      }
      if (this.airPart < 0) return;
      const landed = !!measure.touch[feet[this.airPart][1]];
      if (landed && upright) {
        const gained = this.airGap - footDx(this.airPart);
        if (gained > this.swingPeak) this.swingPeak = gained;
        // Under 1 cm is the same noise as standing still. It does not separate the parent.
        if (gained > 0.01) this.swingClose += gained;
      }
      this.airPart = -1;
    }
    stepPhysics() {
      // Get-up harness: a damped spring takes the pelvis to standing height over the point where it
      // started, with gravity compensated, and another keeps the chest over the pelvis (torso
      // upright). Everything scales with options.assist (0 = off, 1 = it holds the body up).
      // The first helping hand was a plain upward force: it pivoted the body on feet left out in
      // front and swung like a pendulum, and it only worked at hand-picked strengths.
      const assist = this.options.assist || 0;
      if (assist > 0) {
        const world = this.world;
        const P = config.partIndex;
        const m = this.measure;
        if (!world.totalMass) world.totalMass = world.parts.reduce((sum, part) => sum + (part.m || 0), 0);
        if (this.harnessX == null) this.harnessX = m.partX[P.torso] || 0;
        const g = config.getup;
        const mass = world.totalMass;
        const w = 2 * Math.PI * (g.harnessHertz || 1);
        const kp = w * w;
        const kd = 2 * (g.harnessDamping || 1) * w;
        const tx = this.harnessX;
        // A little under standing height, so straight legs put the feet on the floor and load them.
        const ty = config.body.torsoHeightMeters - (config.getup.harnessDrop || 0);
        const px = m.partX[P.torso] || 0, py = m.partY[P.torso] || 0;
        const vx = m.partVelocityX[P.torso] || 0, vy = m.partVelocityY[P.torso] || 0;
        let fx = assist * mass * (kp * (tx - px) - kd * vx);
        let fy = assist * mass * (9.8 + kp * (ty - py) - kd * vy);
        const cap = assist * mass * 9.8 * 2;
        fx = fx < -cap ? -cap : fx > cap ? cap : fx;
        fy = fy < -cap ? -cap : fy > cap ? cap : fy;
        b2Body_ApplyForceToCenter(world.parts[P.torso].id, vector(fx, fy), true);
        // Upright: the chest is pulled over the pelvis, the pelvis pushed the other way (no net force).
        const cx = m.partX[P.chest] || 0;
        const cvx = m.partVelocityX[P.chest] || 0;
        const chestMass = world.parts[P.chest].m + world.parts[P.head].m;
        let ux = assist * chestMass * (kp * (px - cx) - kd * (cvx - vx)) * (g.harnessUpright || 1);
        ux = ux < -cap ? -cap : ux > cap ? cap : ux;
        b2Body_ApplyForceToCenter(world.parts[P.chest].id, vector(ux, 0), true);
        b2Body_ApplyForceToCenter(world.parts[P.torso].id, vector(-ux, 0), true);
        // Pelvis orientation: a damped torsional spring toward upright. Without it the body hung
        // rigid, leaning back 27 degrees with the feet out in front, and never counted as standing.
        if (b2Body_ApplyTorque && g.harnessTilt) {
          const angle = Math.atan2(m.partSin[P.torso] || 0, m.partCos[P.torso] || 1);
          const spin = m.partSpin[P.torso] || 0;
          const inertia = g.harnessInertia || 10;
          let tau = assist * inertia * g.harnessTilt * (kp * -angle - kd * spin);
          const tcap = assist * (g.harnessTorqueCap || 300);
          tau = tau < -tcap ? -tcap : tau > tcap ? tcap : tau;
          b2Body_ApplyTorque(world.parts[P.torso].id, tau, true);
        }
      }
      b2World_Step(
        this.world.worldId,
        config.physics.timeStep,
        config.physics.solverSubsteps,
      );
    }
    get fitnessValue() {
      // Fitness is the reward rows over the last endWindowSeconds, not the whole episode.
      // A short ramp episode has not filled that window, so the mean is every sample so far.
      // start is the opening window. startHoldShare charges only a drop from there to the
      // ending. A gain is not charged.
      const end = this.windowMean(true);
      const start = this.windowMean(false);
      if (!this.endFilled) return 0;
      const hold = config.reward.startHoldShare;
      // BEFORE: (1-hold)*life + hold*min(start, life), with life the whole-episode mean.
      // That charged a net that IMPROVES, and the saturated fall samples diluted the rest.
      // NOW: the drop from the opening window to the ending is charged, and improving is free.
      const drop = start > end ? start - end : 0;
      let blend = hold ? end - hold * drop : end;
      // GET-UP: the motion happens in the first seconds and the end window is the last 3 s, so the
      // imitation rows never reached the fitness (tracking error 0.69 -> 0.70 rad in 100 generations).
      // Here the whole episode counts too, and the start-window drop is not charged: lying still at
      // the start is the point.
      if (this.options.getup && this.lifeCount) {
        const share = config.getup.wholeShare == null ? 0.5 : config.getup.wholeShare;
        blend = (1 - share) * end + share * (this.lifeSum / this.lifeCount);
      }
      return config.reward.targetMean * blend;
    }
    get stats() {
      const end = this.windowMean(true);
      const fitness = this.fitnessValue;
      return {
        fitness,
        f: fitness,
        start: this.windowMean(false),
        end,
        stoodUp: !!this.endedUp,
        up: !!this.endedUp,
        hy: this.sensedFrames ? this.heightSum / this.sensedFrames : 0,
        headImpact: this.headImpact || 0,
        gap:
          this.options.goalX == null
            ? 0
            : Math.abs(this.options.goalX - (this.measure.centerX || 0)),
        steps: this.stepsDone || 0,
        windowStood: this.windowStood || 0,
        windowFrames: this.windowFrames || 0,
        cleared: boxesCleared(this),
        footL: markError(this.measure, targetSlot("footL")).dist,
        footR: markError(this.measure, targetSlot("footR")).dist,
        alignL: markError(this.measure, targetSlot("footL")).align,
        alignR: markError(this.measure, targetSlot("footR")).align,
        headMark: markError(this.measure, targetSlot("head")).dist,
        alignHead: markError(this.measure, targetSlot("head")).align,
        handLMark: markError(this.measure, targetSlot("handL")).dist,
        alignHandL: markError(this.measure, targetSlot("handL")).align,
        handRMark: markError(this.measure, targetSlot("handR")).dist,
        alignHandR: markError(this.measure, targetSlot("handR")).align,
        bodyTargets: placedBodyTargets(this.measure),
        alive: this.frame ? this.stoodFrames / this.frame : 0,
        foot:
          (markError(this.measure, targetSlot("footL")).dist +
            markError(this.measure, targetSlot("footR")).dist) /
          2,
        reach: Math.abs(this.placeExtra || 0),
        closed: this.swingPeak || 0,
        leaned: this.gapCloseCount ? this.bodyCloseSum / this.gapCloseCount : 0,
        // Walk: metres the centre of mass moved along the commanded direction, and the
        // distance the marks moved. Zero when the episode is not a walk.
        walkDist: this.options.walk
          ? this.options.walk.dir * ((this.measure.centerX || 0) - (this.walkStartX || 0))
          : 0,
        walkCmd: this.options.walk ? Math.abs(this.walkOffset()) : 0,
        // Climb: is a foot on top of the box at the end, and the box height.
        climbOnTop: this.climbBox ? (this.footOnTop() ? 1 : 0) : 0,
        climbHeight: this.climbBox ? this.climbBox.top : 0,
        onBoard: this.board ? 1 : 0,
        isGetup: this.options.getup ? 1 : 0,
        isBoard: this.board ? 1 : 0,
        // SWING: both feet still on the plank at the end (not stepped off onto the floor).
        onSwing: this.swing ? (Math.min(this.world.py[config.partIndex.footLeft], this.world.py[config.partIndex.footRight]) > Box2D_GetPosition(this.swing.plank).y + config.swing.halfThick - 0.05 ? 1 : 0) : 0,
        isSwing: this.swing ? 1 : 0,
        trackErr: this.trackCount ? this.trackSum / this.trackCount : -1,
      };
    }
    get weights() {
      return this.network.weights;
    }
    set weights(value) {
      this.network.weights = value;
    }
    get hidden() {
      return this.network.hidden;
    }
    get action() {
      return this.network.action;
    }
  }

  function physicsHz() {
    return 1 / config.physics.timeStep;
  }

  function rewardSamples(seconds) {
    return Math.max(
      1,
      Math.round(
        (seconds * config.physics.hz) / config.cadence.rewardEveryNFrames,
      ),
    );
  }

  function situation(randomSource, pose) {
    const side = randomSource() < 0.5 ? -1 : 1;
    const shared = {
      push: 0,
      amp: 0,
      period: config.body.platformPeriodSeconds,
      phaseShift: 0,
      projectiles: [],
      frames: config.episode.frames,
      task: "live",
      goalX:
        side *
        (config.episode.goalNearMeters +
          randomSource() * config.episode.goalFarExtraMeters),
      goalY: 0,
    };
    const phase = activeTrainingPhase();
    const explicit = pose === "up" || pose === "supine" || pose === "prone";
    let chosen = explicit ? pose : "mix";
    if (!explicit && (config.curriculum.byPhases || chosen === "mix"))
      chosen = phase.start === "up" ? "up" : "random";
    const falling =
      chosen === "up" &&
      phase.falls &&
      randomSource() < config.start.fallDropChance;
    const uprightDrop = falling
      ? config.start.fallDropMinMeters +
        randomSource() *
          (config.start.fallDropMaxMeters - config.start.fallDropMinMeters)
      : config.start.uprightDropMeters;
    const upright = () =>
      Object.assign(shared, {
        tilt: (randomSource() * 2 - 1) * config.start.uprightTiltRadians,
        drop: uprightDrop,
        settle: 0,
        pose: "up",
      });
    const down = (sign) =>
      Object.assign(shared, {
        tilt:
          sign *
          (config.start.downTiltMinRadians +
            randomSource() * config.start.downTiltSpanRadians),
        drop: config.start.downDropMeters,
        settle: config.start.downSettleFrames,
        pose: "down",
      });
    const randomPose = () =>
      Object.assign(shared, {
        tilt: (randomSource() * 2 - 1) * Math.PI,
        drop: randomSource() * config.start.fallDropMaxMeters,
        settle: 0,
        pose: "random",
      });
    const started =
      chosen === "up"
        ? upright()
        : chosen === "random"
          ? randomPose()
          : down(chosen === "prone" ? -1 : 1);
    if (phase.balls && randomSource() < config.start.ballChance) {
      started.ballFrame = config.start.ballThrowFrame;
      started.ballSide = randomSource() < 0.5 ? -1 : 1;
      started.ballSpeed = config.start.ballSpeed;
    }
    return started;
  }

  function trainingStarts(randomSource, mode) {
    return [fillTrainingStart({}, randomSource, mode || null)];
  }

  function spawnThrown(world, kind, x, y, vx, vy) {
    const position = vector(x, y);
    const created = kind.box
      ? CreateBoxPolygon({
          worldId: world.worldId,
          type: dynamicBody,
          position,
          size: vector(kind.half, kind.half),
          density: kind.density,
          friction: 0.6,
          restitution: 0.2,
        })
      : CreateCircle({
          worldId: world.worldId,
          type: dynamicBody,
          position,
          radius: kind.radius,
          density: kind.density,
          friction: 0.6,
          restitution: 0.3,
        });
    const id = created.bodyId;
    b2Body_SetLinearVelocity(id, vector(vx, vy));
    world.thrownIds.push(id);
    world.thrownShapes.push(world.shapeOf(id));
    return id;
  }

  function destroyThrown(world, id) {
    const index = world.thrownIds.indexOf(id);
    if (index >= 0) {
      world.thrownIds.splice(index, 1);
      world.thrownShapes.splice(index, 1);
    }
    b2DestroyBody(id);
  }

  // The training start, drawn on the page. Same placement as resetPose, then the same
  // marks stampTrainTargets plants. Does not move the live bodies.
  function previewStart(world, options) {
    const tilt = (options && options.tilt) || 0;
    const drop = (options && options.drop) || 0;
    layoutStartPose(world, tilt, drop, options);
    const count = world.parts.length;
    const x = new Float32Array(count);
    const y = new Float32Array(count);
    const poseA = [];
    for (let i = 0; i < count; i++) {
      x[i] = placed[i].x;
      y[i] = placed[i].y;
      poseA.push(tilt + (placed[i].extra || 0));
    }
    const source = (options && options.targets) || [];
    const targets = [];
    for (let i = 0; i < config.targets.length; i++) {
      const target = source[i] || {};
      targets.push({
        on: target.on !== false,
        dx: target.dx || 0,
        dy: target.dy || 0,
        turn: target.turn || 0,
        cx: target.cx || 0,
        cy: target.cy || 0,
        ca: target.ca || 0,
        x: target.x || 0,
        y: target.y || 0,
        angle: target.angle || 0,
      });
    }
    const scratch = {
      targets,
      failOnHead: !options || options.failOnHead !== false,
      commands: options && options.commands,
      placed: !!(options && options.placed),
    };
    stampTrainTargets({ partX: x, partY: y }, scratch);
    const marks = [];
    for (let i = 0; i < config.targets.length; i++) {
      const target = targets[i];
      marks.push({
        on: target.on !== false,
        x: target.x || 0,
        y: target.y || 0,
        angle: target.angle || 0,
        part: config.targets[i].part,
      });
    }
    const poseX = [];
    const poseY = [];
    for (let i = 0; i < count; i++) {
      poseX.push(x[i]);
      poseY.push(y[i]);
    }
    return { x: poseX, y: poseY, angle: poseA, marks };
  }

  return {
    build,
    destroyWorld,
    runOneEpisode,
    Episode,
    situation,
    trainingStarts,
    fillTrainingStart,
    neutralLocal,
    summariseEpisode,
    spawnThrown,
    destroyThrown,
    resetPose,
    previewStart,
    setRandom(seed) {
      random = makeSeededRandom(seed);
    },
    randomNormal,
  };
}
