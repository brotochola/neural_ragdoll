// Ragdoll body, sensing, and one episode. The network and the reward live in their own files.
import { config, activeTrainingPhase } from "../config.js";
import { scoreDecision, isStanding } from "./reward.js";
import { ElmanNetwork } from "./net.js";
import { clamp, makeSeededRandom, randomNormalSample } from "./utils.js";

export const jointNames = [
  "Neck", "Shoulder L", "Elbow L", "Wrist L", "Hip L", "Knee L", "Ankle L",
  "Shoulder R", "Elbow R", "Wrist R", "Hip R", "Knee R", "Ankle R",
];

// Same order as the observation vector filled in Episode.act.
// Local +X is the torso's front when standing. Local +Y is up. Each ray leaves the center of that face.
const faceRayMeters = 2;
const faceRayPull = 0.25;
const faceRaySpecs = [
  ["torso", "front", 1, 0], ["torso", "back", -1, 0],
  ["head", "front", 1, 0], ["head", "back", -1, 0], ["head", "top", 0, 1], ["head", "chin", 0, -1],
  ["upperArmLeft", "front", 1, 0], ["upperArmLeft", "back", -1, 0],
  ["upperArmRight", "front", 1, 0], ["upperArmRight", "back", -1, 0],
  ["forearmLeft", "front", 1, 0], ["forearmLeft", "back", -1, 0],
  ["forearmRight", "front", 1, 0], ["forearmRight", "back", -1, 0],
  ["handLeft", "front", 1, 0], ["handLeft", "back", -1, 0], ["handLeft", "tip", 0, -1], ["handLeft", "base", 0, 1],
  ["handRight", "front", 1, 0], ["handRight", "back", -1, 0], ["handRight", "tip", 0, -1], ["handRight", "base", 0, 1],
  ["thighLeft", "front", 1, 0], ["thighLeft", "back", -1, 0],
  ["thighRight", "front", 1, 0], ["thighRight", "back", -1, 0],
  ["shinLeft", "front", 1, 0], ["shinLeft", "back", -1, 0],
  ["shinRight", "front", 1, 0], ["shinRight", "back", -1, 0],
  ["footLeft", "sole", 0, -1], ["footLeft", "top", 0, 1], ["footLeft", "toe", 1, 0],
  ["footRight", "sole", 0, -1], ["footRight", "top", 0, 1], ["footRight", "toe", 1, 0],
  ["hip", "front", 1, 0], ["hip", "back", -1, 0], ["hip", "up", 0, 1], ["hip", "down", 0, -1],
].map(([part, name, x, y]) => ({ part, name, x, y, index: config.partIndex[part] }));
const soleRayIndex = { footLeft: -1, footRight: -1 };
faceRaySpecs.forEach((spec, index) => {
  if (spec.name === "sole") soleRayIndex[spec.part] = index;
});
const facePartName = {
  torso: "torso", head: "head", upperArmLeft: "arm L", upperArmRight: "arm R",
  forearmLeft: "elbow L", forearmRight: "elbow R", handLeft: "hand L", handRight: "hand R",
  thighLeft: "thigh L", thighRight: "thigh R", shinLeft: "shin L", shinRight: "shin R",
  footLeft: "foot L", footRight: "foot R", hip: "tail",
};

export const observationLabels = buildObservationLabels();

function buildObservationLabels() {
  const labels = [];
  for (const name of jointNames) labels.push({ name: "angle " + name, tip: "Joint angle of " + name + ", scaled by its range." });
  for (const name of jointNames) labels.push({ name: "rate " + name, tip: "How fast " + name + " is bending." });
  labels.push(
    { name: "torso sin", tip: "Sine of the torso angle. The sign is face up versus face down." },
    { name: "torso cos", tip: "Cosine of the torso. +1 upright, 0 on its side, negative upside down." },
    { name: "head sin", tip: "Sine of the head angle." },
    { name: "head cos", tip: "Cosine of the head. +1 is upright." },
    { name: "head height", tip: "Head height, scaled by a standing reach." },
    { name: "torso height", tip: "Torso height, scaled by the standing blueprint." },
    { name: "mass x", tip: "Center of mass relative to the midpoint of the feet." },
    { name: "mass y", tip: "Height of the center of mass." },
    { name: "mass vx", tip: "Horizontal speed of the center of mass." },
    { name: "mass vy", tip: "Vertical speed of the center of mass." },
  );
  for (const spec of faceRaySpecs) {
    labels.push({
      name: "ray " + facePartName[spec.part] + " " + spec.name,
      tip: "Hit fraction out of the " + spec.name + " face. 1 is clear, 0 is a hit at the face.",
    });
  }
  labels.push(
    { name: "goal x", tip: "Horizontal distance to the marker, relative to the body. 0 means arrived." },
    { name: "goal y", tip: "Height of the marker above the floor. 0 while the marker is on the floor." },
    { name: "standing", tip: "1 when the body is truly standing: both feet, nothing else down, upright, high head, a real stance, and quiet. Otherwise 0." },
    { name: "head contact", tip: "1 when the head is touching the floor, the platform, or a thrown object. A corner counts. Otherwise 0." },
  );
  for (const [name, index] of Object.entries(config.partIndex)) {
    if (index >= config.net.partCount) continue;
    labels.push({ name: "vx " + name, tip: "Horizontal speed of the " + name + "." });
    labels.push({ name: "vy " + name, tip: "Vertical speed of the " + name + "." });
    labels.push({ name: "spin " + name, tip: "How fast the " + name + " is rotating." });
  }
  if (config.net.rewardTotalNow) labels.push({ name: "reward total", tip: "Sum of the reward rows at this decision, scaled like the other reward inputs." });
  if (config.net.rewardTotalPrevious) labels.push({ name: "reward total previous", tip: "Reward total from the previous decision, one action interval ago. Zero before the first decision." });
  if (config.net.rewardTotalNext) labels.push({ name: "reward total next", tip: "Reward total if every part kept its current speed for one physics frame. No collision or motor." });
  if (labels.length !== config.net.inputCount) throw new Error("observation labels " + labels.length);
  return labels;
}

export const thrownKinds = [
  { name: "Ball 4.5 kg", radius: 0.12, density: 100, color: 0xff6b6b },
  { name: "Box 9 kg", box: 1, half: 0.15, density: 100, color: 0xc08a4b },
  { name: "Weight 25 kg", radius: 0.16, density: 310, color: 0x9aa3b8 },
];

export function createSim(box) {
  const {
    CreateWorld, CreateBoxPolygon, CreateRevoluteJoint, b2Vec2, b2BodyType, b2World_Step,
    b2Body_GetPosition, b2Body_GetRotation, b2Rot_GetAngle, b2Body_SetTransform, b2MakeRot,
    b2Body_SetLinearVelocity, b2Body_SetAngularVelocity, b2Body_SetAwake, b2Body_GetLinearVelocity,
    b2Body_GetAngularVelocity, b2RevoluteJoint_GetAngle, b2RevoluteJoint_SetMotorSpeed,
    CreateCircle, b2DestroyBody, b2Shape_RayCast, b2Body_GetShapes,
    b2Body_GetContactCapacity, b2Body_GetContactData,
  } = box;
  const dynamicBody = b2BodyType.b2_dynamicBody;
  const staticBody = b2BodyType.b2_staticBody;
  const kinematicBody = b2BodyType.b2_kinematicBody;
  const vector = (x, y) => new b2Vec2(x, y);
  const rayOrigin = vector(0, 0);
  const rayDelta = vector(0, 0);
  let random = makeSeededRandom(7);
  // Lee `random` en cada llamada, así setRandom cambia la secuencia.
  const randomNormal = () => randomNormalSample(random);

  function platformAngle(options, time) {
    return options.amp * Math.sin((6.2831853 / options.period) * time + options.phaseShift);
  }

  function build() {
    const { worldId } = CreateWorld({});
    const ground = CreateBoxPolygon({ worldId, type: staticBody, position: vector(0, -1), size: vector(800, 0.5), friction: 2 });
    const platform = CreateBoxPolygon({
      worldId, type: kinematicBody, position: vector(0, -config.body.platformHalfThickness),
      size: vector(config.body.platformHalfWidth, config.body.platformHalfThickness), friction: 2,
    });
    const parts = [];
    const joints = [];
    const addBox = (centerX, centerY, halfX, halfY, density = 70) => {
      const created = CreateBoxPolygon({
        worldId, type: dynamicBody, position: vector(centerX, centerY), size: vector(halfX, halfY),
        density, friction: 2, groupIndex: -1,
      });
      parts.push({
        id: created.bodyId, cx: centerX, cy: centerY, hx: halfX, hy: halfY,
        m: density * 4 * halfX * halfY, th: Math.max(halfX, halfY) + 0.04,
      });
      return parts.length - 1;
    };
    const addJoint = (partA, partB, worldX, worldY, lower, upper, torque) => {
      const a = parts[partA];
      const b = parts[partB];
      const created = CreateRevoluteJoint({
        worldId, bodyIdA: a.id, bodyIdB: b.id,
        anchorA: vector(worldX - a.cx, worldY - a.cy), anchorB: vector(worldX - b.cx, worldY - b.cy),
        lowerAngle: lower, upperAngle: upper, enableLimit: true, enableMotor: true, motorSpeed: 0, maxMotorTorque: torque,
      });
      joints.push({
        id: created.jointId, lo: lower, hi: upper, n: Math.max(-lower, upper),
        a: partA, ax: worldX - a.cx, ay: worldY - a.cy,
        b: partB, bx: worldX - b.cx, by: worldY - b.cy, mark: "",
      });
    };
    // Standing blueprint: torso, head, then left limbs, then right limbs. Indexes match config.partIndex.
    addBox(0, 1.26, 0.13, 0.3);
    addBox(0, 1.68, 0.11, 0.12, 40);
    for (let side = 0; side < 2; side++) {
      addBox(0, 1.34, 0.05, 0.14, 50);
      addBox(0, 1.07, 0.045, 0.13, 50);
      addBox(0, 0.88, 0.05, 0.06, 50);
      addBox(0, 0.74, 0.07, 0.22);
      addBox(0, 0.31, 0.06, 0.21);
      addBox(0.06, 0.04, 0.12, 0.04, 90);
    }
    addJoint(0, 1, 0, 1.56, -0.5, 0.5, 20);
    for (let side = 0; side < 2; side++) {
      const base = 2 + 6 * side;
      addJoint(0, base, 0, 1.48, -2, 2.5, 70);
      addJoint(base, base + 1, 0, 1.2, 0, 2.4, 50);
      addJoint(base + 1, base + 2, 0, 0.94, -1, 1, 25);
      addJoint(0, base + 3, 0, 0.96, -0.7, 2.4, 140);
      addJoint(base + 3, base + 4, 0, 0.52, -2.4, 0, 140);
      addJoint(base + 4, base + 5, 0, 0.1, -0.6, 0.6, 60);
      joints[joints.length - 5].mark = "elbow";
      joints[joints.length - 3].mark = "hip";
      joints[joints.length - 2].mark = "knee";
    }
    const hipRadius = 0.13 * 1.1;
    const hip = CreateCircle({
      worldId, type: dynamicBody, position: vector(0, 0.96), radius: hipRadius, density: 40, friction: 2, groupIndex: -1,
    });
    parts.push({
      id: hip.bodyId, cx: 0, cy: 0.96, hx: hipRadius, hy: hipRadius, r: hipRadius, circle: true,
      m: 40 * Math.PI * hipRadius * hipRadius, th: hipRadius + 0.04,
    });
    CreateRevoluteJoint({
      worldId, bodyIdA: parts[0].id, bodyIdB: hip.bodyId,
      anchorA: vector(0, 0.96 - parts[0].cy), anchorB: vector(0, 0),
      lowerAngle: 0, upperAngle: 0, enableLimit: true, enableMotor: false, motorSpeed: 0, maxMotorTorque: 0,
    });
    return {
      worldId, parts, joints, groundId: ground.bodyId, platformId: platform.bodyId,
      floorShapes: [firstShape(platform.bodyId), firstShape(ground.bodyId)],
    };
  }

  function firstShape(bodyId) {
    const shapes = [];
    b2Body_GetShapes(bodyId, shapes);
    return shapes[0];
  }

  // ponytail: two shapes, not b2World_CastRay. That cast keeps the first fixture and hides the platform. Upgrade path: a category bit if the floor becomes dynamic.
  function rayFraction(world, originX, originY, dirX, dirY) {
    rayOrigin.x = originX;
    rayOrigin.y = originY;
    rayDelta.x = dirX;
    rayDelta.y = dirY;
    let fraction = 1;
    const shapes = world.floorShapes;
    for (let i = 0; i < shapes.length; i++) {
      const cast = b2Shape_RayCast(shapes[i], rayOrigin, rayDelta);
      if (cast.hit && cast.fraction >= 0 && cast.fraction < fraction) fraction = cast.fraction;
    }
    return fraction;
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
    b2Body_SetTransform(world.platformId, vector(0, -config.body.platformHalfThickness), b2MakeRot(platformAngle(options, 0)));
    b2Body_SetLinearVelocity(world.platformId, vector(0, 0));
    b2Body_SetAngularVelocity(world.platformId, 0);
    let lowest = Infinity;
    const placed = [];
    for (const part of world.parts) {
      const x = cosine * part.cx - sine * (part.cy - 0.04) + (options.shiftX || 0);
      const y = 0.04 + drop + sine * part.cx + cosine * (part.cy - 0.04);
      placed.push({ part, x, y });
      lowest = Math.min(lowest, y - Math.abs(sine) * part.hx - Math.abs(cosine) * part.hy);
    }
    const lift = Math.max(0, 0.05 - lowest);
    for (const place of placed) {
      b2Body_SetTransform(place.part.id, vector(place.x, place.y + lift), b2MakeRot(tilt));
      b2Body_SetLinearVelocity(place.part.id, vector(0, 0));
      b2Body_SetAngularVelocity(place.part.id, 0);
      b2Body_SetAwake(place.part.id, true);
    }
    for (const joint of world.joints) b2RevoluteJoint_SetMotorSpeed(joint.id, 0);
  }

  function fillMeasure(world, measure) {
    let mass = 0;
    let momentX = 0;
    let momentY = 0;
    let momentVx = 0;
    let momentVy = 0;
    const rays = measure.recordRays ? (measure.rays = []) : null;
    const count = world.parts.length;
    const partX = measure.partX;
    const partY = measure.partY;
    const partCos = measure.partCos;
    const partSin = measure.partSin;
    for (let i = 0; i < count; i++) {
      const part = world.parts[i];
      const position = b2Body_GetPosition(part.id);
      const velocity = b2Body_GetLinearVelocity(part.id);
      const rot = b2Body_GetRotation(part.id);
      partX[i] = position.x;
      partY[i] = position.y;
      partCos[i] = rot.c;
      partSin[i] = rot.s;
      measure.partVelocityX[i] = velocity.x;
      measure.partVelocityY[i] = velocity.y;
      measure.partSpin[i] = b2Body_GetAngularVelocity(part.id);
      measure.partHeight[i] = position.y;
      measure.clearance[i] = faceRayMeters;
      mass += part.m;
      momentX += part.m * position.x;
      momentY += part.m * position.y;
      momentVx += part.m * velocity.x;
      momentVy += part.m * velocity.y;
    }
    const face = measure.faceRay;
    for (let i = 0; i < faceRaySpecs.length; i++) {
      const spec = faceRaySpecs[i];
      const part = world.parts[spec.index];
      const cosine = partCos[spec.index];
      const sine = partSin[spec.index];
      const localX = spec.x * (part.circle ? part.r : part.hx);
      const localY = spec.y * (part.circle ? part.r : part.hy);
      const faceX = partX[spec.index] + cosine * localX - sine * localY;
      const faceY = partY[spec.index] + sine * localX + cosine * localY;
      const dirX = cosine * spec.x - sine * spec.y;
      const dirY = sine * spec.x + cosine * spec.y;
      // Start behind the face. A sole sunk into the platform would otherwise begin inside it and miss.
      const castX = faceX - dirX * faceRayPull;
      const castY = faceY - dirY * faceRayPull;
      const reach = faceRayMeters + faceRayPull;
      const fraction = rayFraction(world, castX, castY, dirX * reach, dirY * reach);
      const distance = fraction >= 1 ? faceRayMeters : Math.max(0, fraction * reach - faceRayPull);
      face[i] = Math.min(1, distance / faceRayMeters);
      if (distance < measure.clearance[spec.index]) measure.clearance[spec.index] = distance;
      if (rays) rays.push({
        x0: faceX, y0: faceY,
        x1: faceX + dirX * distance, y1: faceY + dirY * distance,
        hit: distance < faceRayMeters, gap: distance,
      });
    }
    for (let i = 0; i < count; i++) measure.touch[i] = measure.clearance[i] < config.body.touchGapMeters;
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
    measure.sole[0] = face[soleRayIndex.footLeft] * faceRayMeters;
    measure.sole[1] = face[soleRayIndex.footRight] * faceRayMeters;
    measure.handOn = !!(measure.touch[config.partIndex.handLeft] || measure.touch[config.partIndex.handRight]);
    measure.bodyOn = false;
    for (let i = 0; i < count; i++) {
      if (i !== footLeft && i !== footRight && measure.touch[i]) measure.bodyOn = true;
    }
    return measure;
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
    predicted.stanceWidth = Math.abs(predicted.footXLeft - predicted.footXRight);
    predicted.sole[0] = Math.max(0, (measure.sole[0] || 0) + (measure.partVelocityY[footLeft] || 0) * dt);
    predicted.sole[1] = Math.max(0, (measure.sole[1] || 0) + (measure.partVelocityY[footRight] || 0) * dt);
    if (measure.goalDistanceMeters != null) {
      predicted.goalDistanceMeters = measure.goalDistanceMeters - (predicted.centerX - measure.centerX);
    }
    return predicted;
  }

  class Episode {
    constructor(world, weights, tilt, drop, options = {}) {
      this.options = Object.assign({
        push: 0, amp: 0, period: config.body.platformPeriodSeconds, phaseShift: 0, projectiles: [],
        frames: config.episode.frames, task: "stand", phase: 0,
      }, options);
      this.free = false;
      this.world = world;
      this.network = new ElmanNetwork(weights);
      this.frame = 0;
      this.startSamples = rewardSamples(config.episode.startWindowSeconds);
      this.startSum = 0;
      this.startCount = 0;
      this.endRing = new Float32Array(rewardSamples(config.episode.endWindowSeconds));
      this.endCursor = 0;
      this.endFilled = 0;
      this.input = new Float32Array(config.net.inputCount);
      this.previousJointAngle = new Float32Array(config.net.jointCount);
      this.fallen = false;
      this.upStreak = 0;
      this.stoodFrames = 0;
      this.sensedFrames = 0;
      this.heightSum = 0;
      this.headImpact = 0;
      this.hintMatchSum = 0;
      this.hintMatchCount = 0;
      this.previousRewardTotal = 0;
      this.objects = [];
      this.measure = {
        partX: new Float32Array(world.parts.length),
        partY: new Float32Array(world.parts.length),
        partCos: new Float32Array(world.parts.length),
        partSin: new Float32Array(world.parts.length),
        partHeight: new Float32Array(world.parts.length),
        clearance: new Array(world.parts.length),
        touch: new Array(world.parts.length),
        sole: new Float32Array(2),
        faceRay: new Float32Array(faceRaySpecs.length),
        partVelocityX: new Float32Array(world.parts.length),
        partVelocityY: new Float32Array(world.parts.length),
        partSpin: new Float32Array(world.parts.length),
      };
      this.ballThrown = false;
      resetPose(world, tilt, drop, this.options);
      const settleFrames = this.options.settle != null ? this.options.settle : 0;
      for (let i = 0; i < settleFrames; i++) b2World_Step(world.worldId, config.physics.timeStep, config.physics.solverSubsteps);
      for (let i = 0; i < world.joints.length; i++) this.previousJointAngle[i] = b2RevoluteJoint_GetAngle(world.joints[i].id);
      fillMeasure(world, this.measure);
    }
    cleanup() {
      for (const id of this.objects) b2DestroyBody(id);
      this.objects = [];
    }
    // Read the body into this.input and write a new action. Does not score.
    act() {
      const world = this.world;
      const measure = fillMeasure(world, this.measure);
      const footMid = (measure.footXLeft + measure.footXRight) / 2;
      if (this.options.goalX != null) {
        measure.goalDistanceMeters = this.options.goalX - measure.centerX;
        measure.goalDx = clamp(measure.goalDistanceMeters / 4, -1, 1);
        measure.goalDy = clamp((this.options.goalY || 0) / 2, -1, 1);
      }
      const input = this.input;
      let cursor = 0;
      for (let i = 0; i < config.net.jointCount; i++) {
        const joint = world.joints[i];
        const angle = b2RevoluteJoint_GetAngle(joint.id);
        input[cursor + i] = angle / joint.n;
        input[cursor + config.net.jointCount + i] = clamp((angle - this.previousJointAngle[i]) / (config.cadence.actionEveryNFrames * config.physics.timeStep) / 6, -1, 1);
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
      input[cursor++] = measure.goalDx || 0;
      input[cursor++] = measure.goalDy || 0;
      input[cursor++] = isStanding(measure) ? 1 : 0;
      input[cursor++] = measure.headContact || 0;
      if (cursor !== config.net.sensedInputCount) throw new Error("base observation " + cursor);
      for (let i = 0; i < config.net.partCount; i++) {
        input[cursor++] = clamp((measure.partVelocityX[i] || 0) / config.scale.partSpeed, -2, 2);
        input[cursor++] = clamp((measure.partVelocityY[i] || 0) / config.scale.partSpeed, -2, 2);
        input[cursor++] = clamp((measure.partSpin[i] || 0) / config.scale.partSpin, -2, 2);
      }
      if (config.net.rewardTotalNow || config.net.rewardTotalPrevious || config.net.rewardTotalNext) {
        const scored = scoreDecision(measure);
        const rewardColumn = (total) => clamp(total / config.scale.rewardInput, -2, 2);
        if (config.net.rewardTotalNow) input[cursor++] = rewardColumn(scored.total);
        if (config.net.rewardTotalPrevious) input[cursor++] = rewardColumn(this.previousRewardTotal);
        if (config.net.rewardTotalNext) {
          const predicted = predictMeasure(world, measure, config.physics.timeStep);
          input[cursor++] = rewardColumn(scoreDecision(predicted).total);
        }
        this.previousRewardTotal = scored.total;
      }
      if (cursor !== config.net.inputCount) throw new Error("observation length " + cursor);
      this.network.forward(input);
      this.applyHint();
    }
    // The forced sign is what the motor and the next decision's feedback see. The raw output is kept for the match bonus.
    applyHint() {
      const hint = this.options.hint;
      if (!hint || (hint.sign !== 1 && hint.sign !== -1)) return;
      const joint = hint.joint;
      if (joint < 0 || joint >= config.net.jointCount) return;
      this.hintMatchSum += this.action[joint] * hint.sign;
      this.hintMatchCount++;
      this.action[joint] = hint.sign;
    }
    evaluate() {
      const measure = this.measure;
      if (this.options.goalX != null) measure.goalDistanceMeters = this.options.goalX - measure.centerX;
      const scored = scoreDecision(measure);
      this.lastScore = scored;
      const total = scored.total;
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
      if (!fromEnd) return this.startCount ? this.startSum / this.startCount : 0;
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
      if (standing) { this.upStreak++; this.stoodFrames++; }
      else this.upStreak = 0;
      this.sensedFrames++;
      this.heightSum += measure.headHeight;
      if (measure.touch[config.partIndex.head]) {
        this.headImpact = Math.max(this.headImpact || 0, -(measure.headVelocityY || 0));
      }
    }
    step() {
      const frame = this.frame + (this.options.phase || 0);
      fillMeasure(this.world, this.measure);
      if (this.options.goalX != null) {
        this.measure.goalDistanceMeters = this.options.goalX - this.measure.centerX;
        this.measure.goalDx = clamp(this.measure.goalDistanceMeters / 4, -1, 1);
        this.measure.goalDy = clamp((this.options.goalY || 0) / 2, -1, 1);
      }
      this.notePose();
      if (this.options.ballFrame != null && this.frame === this.options.ballFrame && !this.ballThrown) {
        this.ballThrown = true;
        const torso = this.world.parts[config.partIndex.torso];
        const position = b2Body_GetPosition(torso.id);
        const x = position.x;
        const y = position.y;
        const side = this.options.ballSide || 1;
        const id = spawnThrown(this.world, thrownKinds[0], x + side * config.start.ballStartMeters, Math.max(y, 0.6), -side * (this.options.ballSpeed || config.start.ballSpeed), 0.4);
        this.objects.push(id);
      }
      if (frame % config.cadence.actionEveryNFrames === 0) this.act();
      if (!this.quiet && frame % config.cadence.rewardEveryNFrames === config.cadence.rewardFrameOffset) this.evaluate();
      this.applyMotors();
      this.stepPhysics();
      this.frame++;
      if (this.frame >= (this.options.frames || config.episode.frames) && !this.free) {
        this.endedUp = this.upStreak >= 15;
        return true;
      }
      return false;
    }
    applyMotors() {
      for (let i = 0; i < config.net.jointCount; i++) {
        const joint = this.world.joints[i];
        const command = this.action[i];
        const target = command * (command >= 0 ? joint.hi : -joint.lo);
        const speed = clamp(config.episode.motorGain * (target - b2RevoluteJoint_GetAngle(joint.id)), -config.episode.motorSpeedLimit, config.episode.motorSpeedLimit);
        b2RevoluteJoint_SetMotorSpeed(joint.id, speed);
      }
    }
    stepPhysics() {
      const options = this.options;
      const time = this.frame / physicsHz();
      const wave = 6.2831853 / options.period;
      const desired = options.amp * wave * Math.cos(wave * time + options.phaseShift);
      const current = b2Rot_GetAngle(b2Body_GetRotation(this.world.platformId));
      b2Body_SetAngularVelocity(this.world.platformId, desired + 6 * (platformAngle(options, time) - current));
      b2World_Step(this.world.worldId, config.physics.timeStep, config.physics.solverSubsteps);
    }
    get fitnessValue() {
      const end = this.windowMean(true);
      const start = this.windowMean(false);
      const pose = end - Math.max(0, start - end);
      if (!this.hintMatchCount) return pose;
      return pose + config.hint.matchWeight * (this.hintMatchSum / this.hintMatchCount);
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
        gap: this.options.goalX == null ? 0 : Math.abs(this.options.goalX - (this.measure.centerX || 0)),
      };
    }
    get weights() { return this.network.weights; }
    set weights(value) { this.network.weights = value; }
    get hidden() { return this.network.hidden; }
    get action() { return this.network.action; }
  }

  function physicsHz() { return 1 / config.physics.timeStep; }

  function rewardSamples(seconds) {
    return Math.max(1, Math.round(seconds * config.physics.hz / config.cadence.rewardEveryNFrames));
  }

  function situation(randomSource, pose) {
    const side = randomSource() < 0.5 ? -1 : 1;
    const shared = {
      push: 0, amp: 0, period: config.body.platformPeriodSeconds, phaseShift: 0, projectiles: [],
      frames: config.episode.frames, task: "live",
      goalX: side * (config.episode.goalNearMeters + randomSource() * config.episode.goalFarExtraMeters),
      goalY: 0,
    };
    const phase = activeTrainingPhase();
    const explicit = pose === "up" || pose === "supine" || pose === "prone";
    let chosen = explicit ? pose : "mix";
    if (!explicit && (config.curriculum.byPhases || chosen === "mix")) chosen = phase.start === "up" ? "up" : "random";
    const falling = chosen === "up" && phase.falls && randomSource() < config.start.fallDropChance;
    const uprightDrop = falling ? config.start.fallDropMinMeters + randomSource() * (config.start.fallDropMaxMeters - config.start.fallDropMinMeters) : config.start.uprightDropMeters;
    const upright = () => Object.assign(shared, { tilt: (randomSource() * 2 - 1) * config.start.uprightTiltRadians, drop: uprightDrop, settle: 0, pose: "up" });
    const down = (sign) => Object.assign(shared, {
      tilt: sign * (config.start.downTiltMinRadians + randomSource() * config.start.downTiltSpanRadians), drop: config.start.downDropMeters, settle: config.start.downSettleFrames, pose: "down",
    });
    const randomPose = () => Object.assign(shared, {
      tilt: (randomSource() * 2 - 1) * Math.PI, drop: randomSource() * config.start.fallDropMaxMeters, settle: 0, pose: "random",
    });
    const started = chosen === "up" ? upright() : chosen === "random" ? randomPose() : down(chosen === "prone" ? -1 : 1);
    if (phase.balls && randomSource() < config.start.ballChance) {
      started.ballFrame = config.start.ballThrowFrame;
      started.ballSide = randomSource() < 0.5 ? -1 : 1;
      started.ballSpeed = config.start.ballSpeed;
    }
    return started;
  }

  function trainingStarts(randomSource) {
    return ["up", "supine", "prone"].map((pose) => varyStart(situation(randomSource, pose), randomSource));
  }

  function poseScore(rows) {
    let fitness = 0;
    let start = 0;
    let end = 0;
    let gap = 0;
    let stance = 0;
    let headImpact = 0;
    let stoodUp = true;
    for (let i = 0; i < rows.length; i++) {
      const row = rows[i];
      fitness += row.fitness;
      start += row.start || 0;
      end += row.end || 0;
      gap += row.gap || 0;
      stance += row.stance || 0;
      if ((row.headImpact || 0) > headImpact) headImpact = row.headImpact;
      if (!row.stoodUp) stoodUp = false;
    }
    const n = rows.length || 1;
    const mean = fitness / n;
    return { fitness: mean, f: mean, start: start / n, end: end / n, gap: gap / n, stance: stance / n, headImpact, stoodUp, up: stoodUp };
  }

  function varyStart(base, randomSource) {
    return Object.assign({}, base, {
      tilt: (base.tilt || 0) + (randomSource() * 2 - 1) * config.episode.startTiltJitterRadians,
      drop: (base.drop || 0) + randomSource() * config.episode.startDropJitterMeters,
      shiftX: (randomSource() * 2 - 1) * config.episode.startShiftMeters,
    });
  }

  function spawnThrown(world, kind, x, y, vx, vy) {
    const position = vector(x, y);
    const created = kind.box
      ? CreateBoxPolygon({ worldId: world.worldId, type: dynamicBody, position, size: vector(kind.half, kind.half), density: kind.density, friction: 0.6, restitution: 0.2 })
      : CreateCircle({ worldId: world.worldId, type: dynamicBody, position, radius: kind.radius, density: kind.density, friction: 0.6, restitution: 0.3 });
    b2Body_SetLinearVelocity(created.bodyId, vector(vx, vy));
    return created.bodyId;
  }

  return {
    build, Episode, situation, varyStart, trainingStarts, poseScore, spawnThrown, resetPose,
    platformHalfWidth: config.body.platformHalfWidth,
    platformHalfThickness: config.body.platformHalfThickness,
    setRandom(seed) { random = makeSeededRandom(seed); },
    randomNormal,
  };
}
