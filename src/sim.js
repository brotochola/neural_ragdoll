// Ragdoll body, sensing, and one episode. The network and the reward live in their own files.
import {
  physicsTimeStep,
  solverSubsteps,
  actionEveryNFrames,
  rewardEveryNFrames,
  rewardFrameOffset,
  platformHalfWidth,
  platformHalfThickness,
  platformPeriodSeconds,
  touchGapMeters,
  episodeFrames,
  uprightStartChance,
  goalNearMeters,
  goalFarExtraMeters,
  motorGain,
  motorSpeedLimit,
  partIndex,
  jointCount,
  inputCount,
  hiddenUnitCount,
  headHeightMeters,
} from "../config.js";
import { scoreDecision, isStanding } from "./reward.js";
import { forwardInto, hiddenCountOf } from "./net.js";

export const jointNames = [
  "Neck", "Shoulder L", "Elbow L", "Wrist L", "Hip L", "Knee L", "Ankle L",
  "Shoulder R", "Elbow R", "Wrist R", "Hip R", "Knee R", "Ankle R",
];

// Same order as the observation vector filled in Episode.act.
export const observationLabels = buildObservationLabels();

function buildObservationLabels() {
  const labels = [];
  for (const name of jointNames) labels.push({ name: "angle " + name, tip: "Joint angle of " + name + ", scaled by its range." });
  for (const name of jointNames) labels.push({ name: "rate " + name, tip: "How fast " + name + " is bending." });
  labels.push(
    { name: "torso sin", tip: "Sine of the torso angle." },
    { name: "torso cos", tip: "Cosine of the torso. +1 upright, 0 on its side, negative upside down." },
    { name: "torso spin", tip: "How fast the torso is rotating." },
    { name: "torso vx", tip: "Horizontal speed of the torso." },
    { name: "torso vy", tip: "Vertical speed of the torso." },
    { name: "mass x", tip: "Center of mass relative to the midpoint of the feet." },
    { name: "mass y", tip: "Height of the center of mass." },
    { name: "mass vx", tip: "Horizontal speed of the center of mass." },
    { name: "mass vy", tip: "Vertical speed of the center of mass." },
    { name: "capture", tip: "Where the center of mass will land if it keeps its current speed." },
    { name: "head fall", tip: "How soon the head would hit if it kept falling." },
    { name: "torso fall", tip: "How soon the torso would hit if it kept falling." },
    { name: "head height", tip: "Head height, scaled by a standing reach." },
  );
  for (let i = 0; i < 4; i++) labels.push({ name: "sole " + (i + 1), tip: "How close one corner of a foot is to the floor." });
  labels.push(
    { name: "platform angle", tip: "Tilt of the platform. Zero while it is held flat." },
    { name: "platform rate", tip: "How fast the platform is tilting." },
  );
  for (const name of jointNames) labels.push({ name: "prev " + name, tip: "The action sent to " + name + " last time." });
  for (const name of ["torso", "head", "hand L", "hand R", "foot L", "foot R"]) {
    labels.push({ name: "touch " + name, tip: name + " is touching the floor." });
  }
  labels.push(
    { name: "ray torso", tip: "Downward ray from the torso. 1 is clear, 0 is a hit." },
    { name: "ray foot L", tip: "Downward ray from the left foot." },
    { name: "ray foot R", tip: "Downward ray from the right foot." },
    { name: "clear hand L", tip: "Gap under the left hand." },
    { name: "clear hand R", tip: "Gap under the right hand." },
    { name: "clear shin L", tip: "Gap under the left shin." },
    { name: "clear shin R", tip: "Gap under the right shin." },
    { name: "clear head", tip: "Gap under the head." },
    { name: "ray head", tip: "Downward ray from the head." },
    { name: "head sin", tip: "Sine of the head angle." },
    { name: "head cos", tip: "Cosine of the head. +1 is upright." },
    { name: "clear thigh L", tip: "Gap under the left thigh." },
    { name: "clear thigh R", tip: "Gap under the right thigh." },
    { name: "clear elbow L", tip: "Gap under the left forearm." },
    { name: "clear elbow R", tip: "Gap under the right forearm." },
    { name: "goal x", tip: "Horizontal distance to the marker, relative to the body. 0 means arrived." },
    { name: "goal y", tip: "Height of the marker above the floor. 0 while the marker is on the floor." },
  );
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
    CreateCircle, b2DestroyBody, b2World_CastRay, b2DefaultQueryFilter, b2Shape_GetFilter,
  } = box;
  const dynamicBody = b2BodyType.b2_dynamicBody;
  const staticBody = b2BodyType.b2_staticBody;
  const kinematicBody = b2BodyType.b2_kinematicBody;
  const vector = (x, y) => new b2Vec2(x, y);
  const queryFilter = b2DefaultQueryFilter();
  let random = makeLocalRandom(7);
  const randomNormal = () => Math.sqrt(-2 * Math.log(random() + 1e-12)) * Math.cos(6.2831853 * random());

  function makeLocalRandom(seed) {
    return () => {
      seed |= 0;
      seed = (seed + 0x6d2b79f5) | 0;
      let mixed = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      mixed = (mixed + Math.imul(mixed ^ (mixed >>> 7), 61 | mixed)) ^ mixed;
      return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296;
    };
  }

  function platformAngle(options, time) {
    return options.amp * Math.sin((6.2831853 / options.period) * time + options.phaseShift);
  }

  function build() {
    const { worldId } = CreateWorld({});
    const ground = CreateBoxPolygon({ worldId, type: staticBody, position: vector(0, -1), size: vector(800, 0.5), friction: 0.9 });
    const platform = CreateBoxPolygon({
      worldId, type: kinematicBody, position: vector(0, -platformHalfThickness),
      size: vector(platformHalfWidth, platformHalfThickness), friction: 0.9,
    });
    const parts = [];
    const joints = [];
    const addBox = (centerX, centerY, halfX, halfY, density = 70) => {
      const created = CreateBoxPolygon({
        worldId, type: dynamicBody, position: vector(centerX, centerY), size: vector(halfX, halfY),
        density, friction: 0.9, groupIndex: -1,
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
    // Standing blueprint: torso, head, then left limbs, then right limbs. Indexes match partIndex.
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
      addJoint(0, base + 3, 0, 0.96, -0.7, 2, 140);
      addJoint(base + 3, base + 4, 0, 0.52, -2.4, 0, 140);
      addJoint(base + 4, base + 5, 0, 0.1, -0.6, 0.6, 60);
      joints[joints.length - 5].mark = "elbow";
      joints[joints.length - 3].mark = "hip";
      joints[joints.length - 2].mark = "knee";
    }
    const hipRadius = 0.13 * 1.1;
    const hip = CreateCircle({
      worldId, type: dynamicBody, position: vector(0, 0.96), radius: hipRadius, density: 40, friction: 0.9, groupIndex: -1,
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
    return { worldId, parts, joints, groundId: ground.bodyId, platformId: platform.bodyId };
  }

  function rayFraction(worldId, originX, originY, directionX, directionY) {
    if (directionX * directionX + directionY * directionY < 1e-8) return 1;
    let fraction = 1;
    b2World_CastRay(worldId, vector(originX, originY), vector(directionX, directionY), queryFilter, (shapeId, _point, _normal, hitFraction) => {
      if (b2Shape_GetFilter(shapeId).groupIndex === -1) return -1;
      fraction = hitFraction;
      return hitFraction;
    }, 0);
    return fraction;
  }

  function clearanceOf(part, platformPosition, platformCos, platformSin) {
    const position = b2Body_GetPosition(part.id);
    const angle = b2Rot_GetAngle(b2Body_GetRotation(part.id));
    const cosine = Math.cos(angle);
    const sine = Math.sin(angle);
    let gap = 1e9;
    for (const [signX, signY] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
      const worldX = position.x + cosine * signX * part.hx - sine * signY * part.hy;
      const worldY = position.y + sine * signX * part.hx + cosine * signY * part.hy;
      const localX = platformCos * (worldX - platformPosition.x) + platformSin * (worldY - platformPosition.y);
      const localY = -platformSin * (worldX - platformPosition.x) + platformCos * (worldY - platformPosition.y);
      const outsideX = Math.abs(localX) - platformHalfWidth;
      const outsideY = Math.abs(localY) - platformHalfThickness;
      gap = Math.min(gap, outsideX > 0 && outsideY > 0 ? Math.hypot(outsideX, outsideY) : Math.max(outsideX, outsideY), worldY + 0.5);
    }
    return gap;
  }

  function resetPose(world, tilt, drop, options) {
    const cosine = Math.cos(tilt);
    const sine = Math.sin(tilt);
    b2Body_SetTransform(world.platformId, vector(0, -platformHalfThickness), b2MakeRot(platformAngle(options, 0)));
    b2Body_SetLinearVelocity(world.platformId, vector(0, 0));
    b2Body_SetAngularVelocity(world.platformId, 0);
    let lowest = Infinity;
    const placed = [];
    for (const part of world.parts) {
      const x = cosine * part.cx - sine * (part.cy - 0.04);
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
    const platformAngleNow = b2Rot_GetAngle(b2Body_GetRotation(world.platformId));
    const slope = Math.tan(platformAngleNow);
    const platformPosition = b2Body_GetPosition(world.platformId);
    const platformCos = Math.cos(platformAngleNow);
    const platformSin = Math.sin(platformAngleNow);
    let mass = 0;
    let momentX = 0;
    let momentY = 0;
    let momentVx = 0;
    let momentVy = 0;
    const torso = world.parts[partIndex.torso];
    const head = world.parts[partIndex.head];
    const footLeft = world.parts[partIndex.footLeft];
    const footRight = world.parts[partIndex.footRight];
    for (let i = 0; i < world.parts.length; i++) {
      const part = world.parts[i];
      const position = b2Body_GetPosition(part.id);
      const velocity = b2Body_GetLinearVelocity(part.id);
      const height = position.y - position.x * slope;
      measure.partHeight[i] = height;
      measure.partAngle[i] = b2Rot_GetAngle(b2Body_GetRotation(part.id));
      measure.clearance[i] = clearanceOf(part, platformPosition, platformCos, platformSin);
      measure.touch[i] = measure.clearance[i] < touchGapMeters;
      mass += part.m;
      momentX += part.m * position.x;
      momentY += part.m * height;
      momentVx += part.m * velocity.x;
      momentVy += part.m * velocity.y;
    }
    const torsoPosition = b2Body_GetPosition(torso.id);
    const torsoVelocity = b2Body_GetLinearVelocity(torso.id);
    const headPosition = b2Body_GetPosition(head.id);
    const headVelocity = b2Body_GetLinearVelocity(head.id);
    const leftFootPosition = b2Body_GetPosition(footLeft.id);
    const rightFootPosition = b2Body_GetPosition(footRight.id);
    measure.torsoX = torsoPosition.x;
    measure.torsoVelocityX = torsoVelocity.x;
    measure.torsoVelocityY = torsoVelocity.y;
    measure.headX = headPosition.x;
    measure.headWorldY = headPosition.y;
    measure.headVelocityY = headVelocity.y;
    measure.headVelocityX = headVelocity.x;
    measure.footXLeft = leftFootPosition.x;
    measure.footXRight = rightFootPosition.x;
    measure.torsoHeight = measure.partHeight[partIndex.torso];
    measure.headHeight = measure.partHeight[partIndex.head];
    measure.centerX = momentX / mass;
    measure.centerY = momentY / mass;
    measure.centerVelocityX = momentVx / mass;
    measure.centerVelocityY = momentVy / mass;
    measure.torsoAngle = measure.partAngle[partIndex.torso];
    measure.torsoSpin = b2Body_GetAngularVelocity(torso.id);
    measure.headAngle = measure.partAngle[partIndex.head];
    measure.platformAngle = platformAngleNow;
    measure.platformSpin = b2Body_GetAngularVelocity(world.platformId);
    measure.stanceWidth = Math.abs(measure.footXLeft - measure.footXRight);
    let soleCursor = 0;
    for (const foot of [footLeft, footRight]) {
      const position = b2Body_GetPosition(foot.id);
      const angle = b2Rot_GetAngle(b2Body_GetRotation(foot.id));
      const cosine = Math.cos(angle);
      const sine = Math.sin(angle);
      measure.sole[soleCursor++] = Math.max(0, position.y - sine * foot.hx - cosine * foot.hy - (position.x - cosine * foot.hx + sine * foot.hy) * slope);
      measure.sole[soleCursor++] = Math.max(0, position.y + sine * foot.hx - cosine * foot.hy - (position.x + cosine * foot.hx + sine * foot.hy) * slope);
    }
    measure.handOn = !!(measure.touch[partIndex.handLeft] || measure.touch[partIndex.handRight]);
    measure.bodyOn = false;
    for (let i = 0; i < world.parts.length; i++) {
      if (i !== partIndex.footLeft && i !== partIndex.footRight && measure.touch[i]) measure.bodyOn = true;
    }
    measure.rayTorso = rayFraction(world.worldId, torsoPosition.x, torsoPosition.y, 0, -2);
    measure.rayFootLeft = rayFraction(world.worldId, leftFootPosition.x, leftFootPosition.y, 0, -2);
    measure.rayFootRight = rayFraction(world.worldId, rightFootPosition.x, rightFootPosition.y, 0, -2);
    measure.rayHead = rayFraction(world.worldId, headPosition.x, headPosition.y, 0, -2);
    measure.rayHeadAhead = rayFraction(world.worldId, headPosition.x, headPosition.y, headVelocity.x * 0.2, headVelocity.y * 0.2);
    // Aliases the reward and the old drawing code still read.
    measure.a = measure.torsoAngle;
    measure.w = measure.torsoSpin;
    measure.ha = measure.headAngle;
    measure.hy = measure.headHeight;
    measure.ty = measure.torsoHeight;
    measure.cx = measure.centerX;
    measure.cy = measure.centerY;
    measure.cvx = measure.centerVelocityX;
    measure.cvy = measure.centerVelocityY;
    measure.hvy = measure.headVelocityY;
    measure.hx = measure.headX;
    measure.hyw = measure.headWorldY;
    measure.hvx = measure.headVelocityX;
    measure.fx0 = measure.footXLeft;
    measure.fx1 = measure.footXRight;
    measure.tx = measure.torsoX;
    measure.tvx = measure.torsoVelocityX;
    measure.tvy = measure.torsoVelocityY;
    measure.pa = measure.platformAngle;
    measure.pw = measure.platformSpin;
    measure.stance = measure.stanceWidth;
    measure.clear = measure.clearance;
    return measure;
  }

  function timeToFloor(height, velocityY) {
    const speed = velocityY + Math.sqrt(velocityY * velocityY + 19.62 * Math.max(height - 0.15, 0));
    return clamp01(speed / 9.81, 0, 1.5) / 1.5;
  }
  function clamp01(value, low, high) {
    return Math.max(low, Math.min(high, value));
  }

  class Episode {
    constructor(world, weights, tilt, drop, options = {}) {
      this.options = Object.assign({
        push: 0, amp: 0, period: platformPeriodSeconds, phaseShift: 0, projectiles: [],
        frames: episodeFrames, task: "stand", phase: 0,
      }, options);
      this.free = false;
      this.world = world;
      this.weights = weights;
      this.frame = 0;
      this.returnSum = 0;
      this.scoreCount = 0;
      this.hidden = new Float32Array(hiddenCountOf(weights.length) || hiddenUnitCount);
      this.nextHidden = new Float32Array(this.hidden.length);
      this.action = new Float32Array(jointCount);
      this.previousAction = new Float32Array(jointCount);
      this.input = new Float32Array(inputCount);
      this.previousJointAngle = new Float32Array(jointCount);
      this.fallen = false;
      this.upStreak = 0;
      this.stoodFrames = 0;
      this.sensedFrames = 0;
      this.heightSum = 0;
      this.headImpact = 0;
      this.objects = [];
      this.measure = {
        partHeight: new Float32Array(world.parts.length),
        partAngle: new Float32Array(world.parts.length),
        clearance: new Array(world.parts.length),
        touch: new Array(world.parts.length),
        sole: new Float32Array(4),
      };
      resetPose(world, tilt, drop, this.options);
      const settleFrames = this.options.settle != null ? this.options.settle : 0;
      for (let i = 0; i < settleFrames; i++) b2World_Step(world.worldId, physicsTimeStep, solverSubsteps);
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
        measure.goalDx = clamp01(measure.goalDistanceMeters / 4, -1, 1);
        measure.goalDy = clamp01((this.options.goalY || 0) / 2, -1, 1);
      }
      const fallTime = Math.sqrt(Math.max(measure.centerY, 0.3) / 9.81);
      const captureX = measure.centerX + measure.centerVelocityX * fallTime;
      const input = this.input;
      let cursor = 0;
      for (let i = 0; i < jointCount; i++) {
        const joint = world.joints[i];
        const angle = b2RevoluteJoint_GetAngle(joint.id);
        input[cursor + i] = angle / joint.n;
        input[cursor + jointCount + i] = clamp01((angle - this.previousJointAngle[i]) / (actionEveryNFrames * physicsTimeStep) / 6, -1, 1);
        this.previousJointAngle[i] = angle;
      }
      cursor += 2 * jointCount;
      input[cursor++] = Math.sin(measure.torsoAngle);
      input[cursor++] = Math.cos(measure.torsoAngle);
      input[cursor++] = measure.torsoSpin / 5;
      input[cursor++] = measure.torsoVelocityX / 3;
      input[cursor++] = measure.torsoVelocityY / 3;
      input[cursor++] = clamp01((measure.centerX - footMid) / 0.3, -2, 2);
      input[cursor++] = measure.centerY - 0.85;
      input[cursor++] = measure.centerVelocityX / 2;
      input[cursor++] = measure.centerVelocityY / 2;
      input[cursor++] = clamp01((captureX - footMid) / 0.3, -2, 2);
      input[cursor++] = timeToFloor(measure.headHeight, measure.headVelocityY);
      input[cursor++] = timeToFloor(measure.torsoHeight, measure.torsoVelocityY);
      input[cursor++] = measure.headHeight / headHeightMeters;
      for (let i = 0; i < 4; i++) input[cursor++] = clamp01(1 - measure.sole[i] / 0.06, 0, 1);
      input[cursor++] = measure.platformAngle / 0.2;
      input[cursor++] = clamp01(measure.platformSpin / 0.15, -2, 2);
      for (let i = 0; i < jointCount; i++) input[cursor++] = this.action[i];
      input[cursor++] = measure.touch[partIndex.torso] ? 1 : 0;
      input[cursor++] = measure.touch[partIndex.head] ? 1 : 0;
      input[cursor++] = measure.touch[partIndex.handLeft] ? 1 : 0;
      input[cursor++] = measure.touch[partIndex.handRight] ? 1 : 0;
      input[cursor++] = measure.touch[partIndex.footLeft] ? 1 : 0;
      input[cursor++] = measure.touch[partIndex.footRight] ? 1 : 0;
      input[cursor++] = measure.rayTorso;
      input[cursor++] = measure.rayFootLeft;
      input[cursor++] = measure.rayFootRight;
      const gap01 = (index) => clamp01((measure.clearance[index] || 0) / 0.6, 0, 1);
      input[cursor++] = gap01(partIndex.handLeft);
      input[cursor++] = gap01(partIndex.handRight);
      input[cursor++] = gap01(partIndex.shinLeft);
      input[cursor++] = gap01(partIndex.shinRight);
      input[cursor++] = gap01(partIndex.head);
      input[cursor++] = measure.rayHead;
      input[cursor++] = Math.sin(measure.headAngle);
      input[cursor++] = Math.cos(measure.headAngle);
      input[cursor++] = gap01(partIndex.thighLeft);
      input[cursor++] = gap01(partIndex.thighRight);
      input[cursor++] = gap01(partIndex.forearmLeft);
      input[cursor++] = gap01(partIndex.forearmRight);
      input[cursor++] = measure.goalDx || 0;
      input[cursor++] = measure.goalDy || 0;
      if (cursor !== inputCount) throw new Error("observation length " + cursor);
      forwardInto(this.weights, input, this.hidden, this.nextHidden, this.action);
      this.previousAction.set(this.action);
      const swap = this.hidden;
      this.hidden = this.nextHidden;
      this.nextHidden = swap;
    }
    evaluate() {
      const measure = this.measure;
      if (this.options.goalX != null) measure.goalDistanceMeters = this.options.goalX - measure.centerX;
      const scored = scoreDecision(measure);
      this.lastScore = scored;
      this.returnSum += scored.total;
      this.scoreCount++;
    }
    notePose() {
      const measure = this.measure;
      const standing = isStanding(measure);
      this.fallen = measure.bodyOn || Math.cos(measure.torsoAngle) < 0.25;
      if (standing) { this.upStreak++; this.stoodFrames++; }
      else this.upStreak = 0;
      this.sensedFrames++;
      this.heightSum += measure.headHeight;
      if (measure.clearance[partIndex.head] < 0.03) {
        this.headImpact = Math.max(this.headImpact || 0, -(measure.headVelocityY || 0));
      }
    }
    step() {
      const frame = this.frame + (this.options.phase || 0);
      fillMeasure(this.world, this.measure);
      if (this.options.goalX != null) {
        this.measure.goalDistanceMeters = this.options.goalX - this.measure.centerX;
        this.measure.goalDx = clamp01(this.measure.goalDistanceMeters / 4, -1, 1);
        this.measure.goalDy = clamp01((this.options.goalY || 0) / 2, -1, 1);
      }
      this.notePose();
      if (frame % actionEveryNFrames === 0) this.act();
      if (!this.quiet && frame % rewardEveryNFrames === rewardFrameOffset) this.evaluate();
      this.applyMotors();
      this.stepPhysics();
      this.frame++;
      if (this.frame >= (this.options.frames || episodeFrames) && !this.free) {
        this.endedUp = this.upStreak >= 15;
        return true;
      }
      return false;
    }
    applyMotors() {
      for (let i = 0; i < jointCount; i++) {
        const joint = this.world.joints[i];
        const command = this.action[i];
        const target = command * (command >= 0 ? joint.hi : -joint.lo);
        const speed = clamp01(motorGain * (target - b2RevoluteJoint_GetAngle(joint.id)), -motorSpeedLimit, motorSpeedLimit);
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
      b2World_Step(this.world.worldId, physicsTimeStep, solverSubsteps);
    }
    get fitnessValue() {
      return this.returnSum / Math.max(1, this.scoreCount);
    }
    get stats() {
      return {
        fitness: this.fitnessValue,
        f: this.fitnessValue,
        stoodUp: !!this.endedUp,
        up: !!this.endedUp,
        hy: this.sensedFrames ? this.heightSum / this.sensedFrames : 0,
        headImpact: this.headImpact || 0,
        gap: this.options.goalX == null ? 0 : Math.abs(this.options.goalX - (this.measure.centerX || 0)),
      };
    }
    // Names the page already draws with.
    get m() { return this.measure; }
    get o() { return this.options; }
    get w() { return this.weights; }
    set w(weights) { this.weights = weights; }
    get h() { return this.hidden; }
    get a() { return this.action; }
    get x() { return this.input; }
    get t() { return this.frame; }
  }

  function physicsHz() { return 1 / physicsTimeStep; }

  function situation(randomSource, pose) {
    const side = randomSource() < 0.5 ? -1 : 1;
    const shared = {
      push: 0, amp: 0, period: platformPeriodSeconds, phaseShift: 0, projectiles: [],
      frames: episodeFrames, task: "live",
      goalX: side * (goalNearMeters + randomSource() * goalFarExtraMeters),
      goalY: 0,
    };
    const upright = () => Object.assign(shared, { tilt: (randomSource() * 2 - 1) * 0.08, drop: 0.12, settle: 0, pose: "up" });
    const down = (sign) => Object.assign(shared, {
      tilt: sign * (1.15 + randomSource() * 0.55), drop: 0.05, settle: 40, pose: "down",
    });
    if (pose === "up") return upright();
    if (pose === "supine") return down(1);
    if (pose === "prone") return down(-1);
    if (randomSource() < uprightStartChance) return upright();
    return down(randomSource() < 0.5 ? -1 : 1);
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
    build, Episode, situation, spawnThrown, resetPose, platformHalfWidth, platformHalfThickness,
    setRandom(seed) { random = makeLocalRandom(seed); },
    randomNormal,
  };
}
