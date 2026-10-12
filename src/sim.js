// Ragdoll body, sensing, and one episode. The network and the reward live in their own files.
import { config, activeTrainingPhase } from "../config.js";
import {
  scoreDecision,
  scoreTotal,
  isStanding,
  onMark,
  markError,
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
    );
  }
  labels.push({
    name: "head contact",
    tip: "1 when the head is touching the floor or a thrown object. A corner counts. Otherwise 0.",
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

// Los pies nacen uno encima del otro. La marca los separa para que el premio y pararse coincidan.
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

// Un paso corto de los dos pies, el mismo para los dos gemelos. La cabeza sigue al medio.
function fillPlaceFeet(out, list, randomSource, scale, shiftMeters) {
  const skill = config.skill;
  const half = skill.placeStanceMeters || skill.stanceMeters;
  // Los dos pies se corren juntos. scale < 1 acorta el paso. La eval oficial pasa 1.
  const s = scale == null ? 1 : scale;
  const extra =
    shiftMeters > 0
      ? shiftMeters
      : s *
        (skill.placeMinMeters +
          randomSource() * (skill.placeMaxMeters - skill.placeMinMeters));
  const rolled = randomSource() < 0.5 ? -1 : 1;
  // El paso largo oficial solo va hacia +x. Un shift fijo conserva el signo, así 0.12 y 0.16 no cambian.
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

// Pasos alternados. El plan queda en el arranque: reset no sortea otro.
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

export function fillTrainingStart(out, randomSource, mode) {
  const level = mode && mode.level ? mode.level | 0 : config.skill.level | 0;
  const full = !!(mode && mode.full);
  const seconds = full
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
  // El sandbox no entra. El nivel decide las marcas, y el par comparte este mismo sorteo.
  const list = ensureTargets(out);
  const meters = config.start.targetJitterMeters;
  const radians = config.start.targetJitterRadians;
  for (let i = 0; i < list.length; i++) {
    const target = list[i];
    target.on = true;
    target.dx = signed(randomSource, meters);
    target.dy = signed(randomSource, meters);
    target.turn = signed(randomSource, radians);
  }
  if (level <= 1) fillStandFeet(list, randomSource);
  else if (level === 2) {
    const scale = full
      ? 1
      : mode && mode.placeScale != null
        ? mode.placeScale
        : config.skill.placeScale == null
          ? 1
          : config.skill.placeScale;
    fillPlaceFeet(out, list, randomSource, scale, mode && mode.shiftMeters);
  } else fillWalk(out, list, randomSource, level >= 4);
  return out;
}

// La cabeza y las manos se paran arriba del medio de los pies, así caminar no las deja en el origen.
function plantBodyTargets(options) {
  const targets = options && options.targets;
  if (!targets) return;
  const right = targets[targetSlot("footR")];
  const left = targets[targetSlot("footL")];
  const mid = ((right.x || 0) + (left.x || 0)) / 2;
  for (let i = 0; i < 3; i++) targets[i].x = mid;
}

function applyStep(options, step) {
  const target = options.targets[step.foot];
  target.x = (target.baseX || 0) + step.dx;
  target.y = config.skill.footGroundMeters;
  target.angle = step.turn || 0;
  target.on = true;
  plantBodyTargets(options);
}

// El siguiente pie aparece cuando este ya está en la marca y el cuerpo sigue parado.
// Cuando ya está quieto, un pie se queda y el otro recibe el paso sorteado.
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
  plantBodyTargets(episode.options);
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
    applyStep(episode.options, next);
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

// Escribe la marca del mundo: la parte ya colocada, más el corrimiento que sorteó el par.
// Los pies quedan a la altura del suelo. El drop los deja en el aire y esa marca no se puede pisar.
function stampTrainTargets(measure, options) {
  const source = options && options.targets;
  if (!source) return;
  for (let i = 0; i < source.length && i < config.targets.length; i++) {
    const target = source[i];
    if (!target.on) continue;
    const spec = config.targets[i];
    const index = config.partIndex[spec.part];
    const foot = spec.id === "footR" || spec.id === "footL";
    if (foot) {
      target.baseX = measure.partX[index] || 0;
      target.x = target.baseX + (target.dx || 0);
      target.y = config.skill.footGroundMeters;
      target.angle = target.turn || 0;
    } else {
      target.x = (measure.partX[index] || 0) + (target.dx || 0);
      target.y = (measure.partY[index] || 0) + (target.dy || 0);
      target.angle =
        Math.atan2(measure.partSin[index] || 0, measure.partCos[index] || 0) +
        (target.turn || 0);
    }
  }
  if ((options.skill || 0) >= 2) plantBodyTargets(options);
}

export function poseScore(rows, offset, count, out) {
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
    b2RevoluteJoint_GetAngle,
    b2RevoluteJoint_SetMotorSpeed,
    CreateCircle,
    b2DestroyBody,
    b2Body_GetContactCapacity,
    b2Body_GetContactData,
  } = box;
  const dynamicBody = b2BodyType.b2_dynamicBody;
  const staticBody = b2BodyType.b2_staticBody;
  const vector = (x, y) => new b2Vec2(x, y);
  const placed = [];
  // Box2D guarda el vector y la rotación que le pasamos. Uno solo compartido pega todas las partes juntas.
  function partPose(part, x, y) {
    let pos = part.pose;
    if (!pos) {
      pos = new b2Vec2(x, y);
      part.pose = pos;
    } else {
      pos.x = x;
      pos.y = y;
    }
    return pos;
  }
  function partRot(part, tilt) {
    let rot = part.tiltRot;
    if (!rot) {
      rot = b2MakeRot(tilt);
      part.tiltRot = rot;
    } else {
      rot.c = Math.cos(tilt);
      rot.s = Math.sin(tilt);
    }
    return rot;
  }
  function partVel(part) {
    let vel = part.vel;
    if (!vel) {
      vel = new b2Vec2(0, 0);
      part.vel = vel;
    } else {
      vel.x = 0;
      vel.y = 0;
    }
    return vel;
  }
  let random = makeSeededRandom(7);
  // Lee `random` en cada llamada, así setRandom cambia la secuencia.
  const randomNormal = () => randomNormalSample(random);

  function build() {
    return new RagDoll2D(box);
  }

  // Face rays miss a corner. Box2D's manifold has that point. Speculative contacts reach 2 cm with no load, so those do not count.
  const contactData = [];
  function bodyTouching(bodyId) {
    const capacity = b2Body_GetContactCapacity(bodyId);
    if (!capacity) return false;
    while (contactData.length < capacity) contactData.push({ manifold: null });
    const count = b2Body_GetContactData(bodyId, contactData, capacity);
    const gap = config.body.touchGapMeters;
    for (let i = 0; i < count; i++) {
      const manifold = contactData[i].manifold;
      if (!manifold) continue;
      for (let p = 0; p < manifold.pointCount; p++) {
        const point = manifold.points[p];
        if (point.separation <= gap || point.normalImpulse > 0) return true;
      }
    }
    return false;
  }

  function resetPose(world, tilt, drop, options) {
    const cosine = Math.cos(tilt);
    const sine = Math.sin(tilt);
    const parts = world.parts;
    const count = parts.length;
    let lowest = Infinity;
    const shiftX = options.shiftX || 0;
    for (let i = 0; i < count; i++) {
      const part = parts[i];
      let slot = placed[i];
      if (!slot) {
        slot = { part: null, x: 0, y: 0 };
        placed[i] = slot;
      }
      const x = cosine * part.cx - sine * (part.cy - 0.04) + shiftX;
      const y = 0.04 + drop + sine * part.cx + cosine * (part.cy - 0.04);
      slot.part = part;
      slot.x = x;
      slot.y = y;
      world.support(part, -sine, -cosine);
      lowest = Math.min(lowest, y + sine * world.skinX + cosine * world.skinY);
    }
    const lift = Math.max(0, 0.05 - lowest);
    for (let i = 0; i < count; i++) {
      const slot = placed[i];
      const part = slot.part;
      b2Body_SetTransform(
        part.id,
        partPose(part, slot.x, slot.y + lift),
        partRot(part, tilt),
      );
      b2Body_SetLinearVelocity(part.id, partVel(part));
      b2Body_SetAngularVelocity(part.id, 0);
      b2Body_SetAwake(part.id, true);
    }
    const joints = world.joints;
    for (let i = 0; i < joints.length; i++)
      b2RevoluteJoint_SetMotorSpeed(joints[i].id, 0);
  }

  function fillMeasure(world, measure) {
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
    const headTouch = bodyTouching(world.parts[head].id);
    measure.touch[head] = headTouch;
    measure.headContact = headTouch ? 1 : 0;
    const footLeft = config.partIndex.footLeft;
    const footRight = config.partIndex.footRight;
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

  // Rayos solo en el frame de la acción. El contacto de la cabeza, ya medido, no se pisa.
  function castMeasureRays(world, measure) {
    const count = world.parts.length;
    for (let i = 0; i < count; i++)
      measure.clearance[i] = config.body.bodyRayMeters;
    world.throwRays(measure);
    const gap = config.body.touchGapMeters;
    for (let i = 0; i < count; i++)
      measure.touch[i] = measure.clearance[i] < gap;
    measure.touch[config.partIndex.head] = measure.headContact !== 0;
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

  // Caja quieta en el camino del paso. dx es relativo al pie izquierdo en el arranque.
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
      console.log("episode constructor", options, this);
      this.free = false;
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
      if (options) {
        const keys = Object.keys(options);
        for (let i = 0; i < keys.length; i++)
          scratch[keys[i]] = options[keys[i]];
      }
      this.options = scratch;
    }
    // Mismo cuerpo y mismos buffers. Cambia los pesos y vuelve a la pose inicial.
    reset(weights, tilt, drop, options) {
      this.bindOptions(options || {});
      const network = this.network;
      network.weights = weights;
      network.hidden.fill(0);
      network.nextHidden.fill(0);
      if (network.compressed.length) network.compressed.fill(0);
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
      this.cleanup();
      const world = this.world;
      resetPose(world, tilt, drop, this.options);
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
      fillMeasure(world, this.measure);
      if (this.options.failOnHead) {
        stampTrainTargets(this.measure, this.options);
        placeSkillBoxes(this);
      }
      this.measure.targets = episodeTargets(this.options);
      castMeasureRays(world, this.measure);
    }
    cleanup() {
      const objects = this.objects;
      for (let i = 0; i < objects.length; i++)
        destroyThrown(this.world, objects[i]);
      objects.length = 0;
    }
    // Read the body into this.input and write a new action. Does not score.
    act() {
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
      const face = measure.faceRay;
      for (let i = 0; i < face.length; i++) input[cursor++] = face[i];
      const targets = measure.targets;
      for (let t = 0; t < config.targets.length; t++) {
        const spec = config.targets[t];
        const target = targets[t];
        const index = config.partIndex[spec.part];
        const on = target.on ? 1 : 0;
        input[cursor++] = on;
        if (!on) {
          input[cursor++] = 0;
          input[cursor++] = 0;
          input[cursor++] = 0;
          input[cursor++] = 0;
          continue;
        }
        const span =
          spec.id === "footR" || spec.id === "footL"
            ? config.skill.footSpanMeters
            : config.body.targetSpanMeters;
        input[cursor++] = clamp(
          ((target.x || 0) - measure.partX[index]) / span,
          -1,
          1,
        );
        input[cursor++] = clamp(
          ((target.y || 0) - measure.partY[index]) / span,
          -1,
          1,
        );
        const targetSin = Math.sin(target.angle || 0);
        const targetCos = Math.cos(target.angle || 0);
        const partSin = measure.partSin[index];
        const partCos = measure.partCos[index];
        input[cursor++] = targetSin * partCos - targetCos * partSin;
        input[cursor++] = targetCos * partCos + targetSin * partSin;
      }
      input[cursor++] = measure.headContact || 0;
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
      if (cursor !== config.net.inputCount)
        throw new Error("observation length " + cursor);
      this.network.forward(input);
    }
    evaluate() {
      const total = scoreTotal(this.measure);
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
    notePose() {
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
    step() {
      const frame = this.frame + (this.options.phase || 0);
      const acting = frame % config.cadence.actionEveryNFrames === 0;
      this.measure.targets = episodeTargets(this.options);
      fillMeasure(this.world, this.measure);
      this.notePose();
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
        castMeasureRays(this.world, this.measure);
        this.act();
      }
      if (
        !this.quiet &&
        frame % config.cadence.rewardEveryNFrames ===
          config.cadence.rewardFrameOffset
      )
        this.evaluate();
      this.noteGap();
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
    // El reloj lo pone el episodio. La red chica solo dice cómo mover cadera, rodilla y tobillo.
    blendLegs() {
      const add = this.legAdd;
      add.fill(0);
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
      // El segundo desliz arranca cuando el primer pie que se alzó vuelve al suelo.
      // Solo si falta más de 8 cm y la marca pasa de 18 cm. 0.12 y 0.16 no lo usan.
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
      // Segundo paso: la rodilla derecha flexiona 0.05 rad más, el mismo largo. Luego el apoyo del pie derecho.
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
      // Con caja, el pie izquierdo se alza solo mientras pasa sobre ella. La cadera derecha no entra.
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
      this.blendLegs();
      const limit = config.episode.motorSpeedLimit || 1;
      let speedSum = 0;
      for (let i = 0; i < config.net.jointCount; i++) {
        const joint = this.world.joints[i];
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
    // Un pie en el aire cuenta si el tronco y la cabeza siguen arriba y ese pie se acerca a su marca.
    noteGap() {
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
        // Por debajo de 1 cm es el mismo ruido de estar quieto. No separa al padre.
        if (gained > 0.01) this.swingClose += gained;
      }
      this.airPart = -1;
    }
    stepPhysics() {
      b2World_Step(
        this.world.worldId,
        config.physics.timeStep,
        config.physics.solverSubsteps,
      );
    }
    get fitnessValue() {
      const end = this.windowMean(true);
      const start = this.windowMean(false);
      // El final de 3 s no separa a los que se caen antes. El promedio de todo el episodio sí.
      const life = this.lifeCount ? this.lifeSum / this.lifeCount : end;
      const body = this.options.failOnHead ? life : end;
      let fitness = body - Math.max(0, start - body);
      if (!this.options.failOnHead) return fitness;
      if (this.headHit) fitness -= config.reward.headTouch;
      if (this.motionCount)
        fitness -= config.reward.motion * (this.motionSum / this.motionCount);
      const frames = this.frame || 1;
      fitness += config.reward.upright * (this.uprightSum / frames);
      fitness += config.reward.alive * (this.stoodFrames / frames);
      if ((this.options.skill | 0) >= 2) {
        // Caer, cabeza en el piso, torso abajo, o no volver a los dos pies: por debajo de quedarse quieto.
        if (this.headHit || this.torsoDown || !this.endedUp) return 0;
        // Quietos en los dos pies vale 1. El pie que se acercó en el aire, y lo que quedó al final, valen más.
        const stayed =
          this.gap0 == null ? 0 : Math.max(0, this.gap0 - this.footGap - 0.01);
        fitness = 1 + 8 * stayed;
        if ((this.options.skill | 0) >= 3)
          fitness += config.reward.step * (this.stepsDone || 0);
      }
      return fitness;
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
        alive: this.frame ? this.stoodFrames / this.frame : 0,
        foot:
          (markError(this.measure, targetSlot("footL")).dist +
            markError(this.measure, targetSlot("footR")).dist) /
          2,
        reach: Math.abs(this.placeExtra || 0),
        closed: this.swingPeak || 0,
        leaned: this.gapCloseCount ? this.bodyCloseSum / this.gapCloseCount : 0,
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

  function varyStart(base, randomSource) {
    return Object.assign({}, base, {
      tilt:
        (base.tilt || 0) +
        (randomSource() * 2 - 1) * config.episode.startTiltJitterRadians,
      drop:
        (base.drop || 0) +
        randomSource() * config.episode.startDropJitterMeters,
      shiftX: (randomSource() * 2 - 1) * config.episode.startShiftMeters,
    });
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

  return {
    build,
    Episode,
    situation,
    varyStart,
    trainingStarts,
    fillTrainingStart,
    poseScore,
    spawnThrown,
    destroyThrown,
    resetPose,
    setRandom(seed) {
      random = makeSeededRandom(seed);
    },
    randomNormal,
  };
}
