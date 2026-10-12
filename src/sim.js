// Ragdoll body, sensing, and one episode. The network and the reward live in their own files.
import { config, activeTrainingPhase } from "../config.js";
import { scoreDecision, scoreTotal, isStanding } from "./reward.js";
import { ElmanNetwork } from "./net.js";
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
          " to its target, divided by the target span. 0 while that target is off.",
      },
      {
        name: spec.id + " dy",
        tip:
          "Vertical distance from the " +
          spec.id +
          " to its target, divided by the target span. 0 while that target is off.",
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

export function fillTrainingStart(out, randomSource) {
  out.push = 0;
  out.amp = 0;
  out.period = config.body.platformPeriodSeconds;
  out.phaseShift = 0;
  out.frames = config.episode.frames;
  out.task = "live";
  out.pose = "up";
  out.failOnHead = true;
  out.settle = 0;
  out.ballFrame = null;
  out.goalX = null;
  out.tilt = (randomSource() * 2 - 1) * config.start.uprightTiltRadians;
  out.drop = config.start.uprightDropMeters;
  out.tilt += (randomSource() * 2 - 1) * config.episode.startTiltJitterRadians;
  out.drop += randomSource() * config.episode.startDropJitterMeters;
  out.shiftX = (randomSource() * 2 - 1) * config.episode.startShiftMeters;
  // El sandbox no entra. Las cinco partes se premian cerca de donde nacen en este episodio.
  const meters = config.start.targetJitterMeters;
  const radians = config.start.targetJitterRadians;
  let list = out.targets;
  if (!list || list.length !== config.targets.length) {
    list = [];
    for (let i = 0; i < config.targets.length; i++) list.push({ on: true, x: 0, y: 0, angle: 0, dx: 0, dy: 0, turn: 0 });
    out.targets = list;
  }
  for (let i = 0; i < config.targets.length; i++) {
    const target = list[i];
    target.on = true;
    target.dx = (randomSource() * 2 - 1) * meters;
    target.dy = (randomSource() * 2 - 1) * meters;
    target.turn = (randomSource() * 2 - 1) * radians;
  }
  return out;
}

// Escribe la marca del mundo: la parte ya colocada, más el corrimiento que sorteó el par.
function stampTrainTargets(measure, options) {
  const source = options && options.targets;
  if (!source) return;
  for (let i = 0; i < source.length && i < config.targets.length; i++) {
    const target = source[i];
    if (!target.on) continue;
    const index = config.partIndex[config.targets[i].part];
    target.x = (measure.partX[index] || 0) + (target.dx || 0);
    target.y = (measure.partY[index] || 0) + (target.dy || 0);
    target.angle = Math.atan2(measure.partSin[index] || 0, measure.partCos[index] || 0) + (target.turn || 0);
  }
}

export function poseScore(rows, offset, count, out) {
  let fitness = 0;
  let start = 0;
  let end = 0;
  let gap = 0;
  let stance = 0;
  let headImpact = 0;
  let stoodUp = true;
  const last = offset + count;
  for (let i = offset; i < last; i++) {
    const row = rows[i];
    fitness += row.fitness;
    start += row.start || 0;
    end += row.end || 0;
    gap += row.gap || 0;
    stance += row.stance || 0;
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
    b2Body_SetAngularVelocity,
    b2Body_SetAwake,
    b2RevoluteJoint_GetAngle,
    b2RevoluteJoint_SetMotorSpeed,
    CreateCircle,
    b2DestroyBody,
    b2Body_GetContactCapacity,
    b2Body_GetContactData,
  } = box;
  const dynamicBody = b2BodyType.b2_dynamicBody;
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

  class Episode {
    constructor(world, weights, tilt, drop, options = {}) {
      this.free = false;
      this.world = world;
      this.network = new ElmanNetwork(weights);
      this.startSamples = rewardSamples(config.episode.startWindowSeconds);
      this.endRing = new Float32Array(
        rewardSamples(config.episode.endWindowSeconds),
      );
      this.input = new Float32Array(config.net.inputCount);
      this.previousJointAngle = new Float32Array(config.net.jointCount);
      this.objects = [];
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
      if (this.options.failOnHead) stampTrainTargets(this.measure, this.options);
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
        const span = config.body.targetSpanMeters;
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
    }
    step() {
      const frame = this.frame + (this.options.phase || 0);
      const acting = frame % config.cadence.actionEveryNFrames === 0;
      this.measure.targets = episodeTargets(this.options);
      fillMeasure(this.world, this.measure);
      this.notePose();
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
      this.applyMotors();
      this.stepPhysics();
      this.frame++;
      if (
        this.frame >= (this.options.frames || config.episode.frames) &&
        !this.free
      ) {
        this.endedUp = this.upStreak >= 15;
        return true;
      }
      return false;
    }
    applyMotors() {
      const limit = config.episode.motorSpeedLimit || 1;
      let speedSum = 0;
      for (let i = 0; i < config.net.jointCount; i++) {
        const joint = this.world.joints[i];
        const command = this.action[i];
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
      let fitness = end - Math.max(0, start - end);
      if (!this.options.failOnHead) return fitness;
      if (this.headHit) fitness -= config.reward.headTouch;
      if (this.motionCount)
        fitness -= config.reward.motion * (this.motionSum / this.motionCount);
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

  function trainingStarts(randomSource) {
    return [fillTrainingStart({}, randomSource)];
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
