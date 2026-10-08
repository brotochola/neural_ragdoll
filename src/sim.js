// Ragdoll body, sensing, and one episode. The network and the reward live in their own files.
import { config, activeTrainingPhase } from "../config.js";
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
    { name: "standing", tip: "1 when the body is truly standing: both feet, nothing else down, upright, high head, a real stance, and quiet. Otherwise 0." },
  );
  for (const [name] of Object.entries(config.partIndex)) {
    labels.push({ name: "vx " + name, tip: "Horizontal speed of the " + name + "." });
    labels.push({ name: "vy " + name, tip: "Vertical speed of the " + name + "." });
    labels.push({ name: "spin " + name, tip: "How fast the " + name + " is rotating." });
  }
  for (const term of scoreDecision({ partAngle: [], touch: [], clearance: [] }).terms) {
    if (term.scoreOnly) continue;
    labels.push({ name: "reward " + term.n, tip: term.tip });
  }
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
      const outsideX = Math.abs(localX) - config.body.platformHalfWidth;
      const outsideY = Math.abs(localY) - config.body.platformHalfThickness;
      gap = Math.min(gap, outsideX > 0 && outsideY > 0 ? Math.hypot(outsideX, outsideY) : Math.max(outsideX, outsideY), worldY + 0.5);
    }
    return gap;
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
    const torso = world.parts[config.partIndex.torso];
    const head = world.parts[config.partIndex.head];
    const footLeft = world.parts[config.partIndex.footLeft];
    const footRight = world.parts[config.partIndex.footRight];
    for (let i = 0; i < world.parts.length; i++) {
      const part = world.parts[i];
      const position = b2Body_GetPosition(part.id);
      const velocity = b2Body_GetLinearVelocity(part.id);
      const vx = velocity.x;
      const vy = velocity.y;
      measure.partVelocityX[i] = vx;
      measure.partVelocityY[i] = vy;
      measure.partSpin[i] = b2Body_GetAngularVelocity(part.id);
      const height = position.y - position.x * slope;
      measure.partHeight[i] = height;
      measure.partAngle[i] = b2Rot_GetAngle(b2Body_GetRotation(part.id));
      measure.clearance[i] = clearanceOf(part, platformPosition, platformCos, platformSin);
      measure.touch[i] = measure.clearance[i] < config.body.touchGapMeters;
      mass += part.m;
      momentX += part.m * position.x;
      momentY += part.m * height;
      momentVx += part.m * vx;
      momentVy += part.m * vy;
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
    measure.torsoHeight = measure.partHeight[config.partIndex.torso];
    measure.headHeight = measure.partHeight[config.partIndex.head];
    measure.centerX = momentX / mass;
    measure.centerY = momentY / mass;
    measure.centerVelocityX = momentVx / mass;
    measure.centerVelocityY = momentVy / mass;
    measure.torsoAngle = measure.partAngle[config.partIndex.torso];
    measure.torsoSpin = b2Body_GetAngularVelocity(torso.id);
    measure.headAngle = measure.partAngle[config.partIndex.head];
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
    measure.handOn = !!(measure.touch[config.partIndex.handLeft] || measure.touch[config.partIndex.handRight]);
    measure.bodyOn = false;
    for (let i = 0; i < world.parts.length; i++) {
      if (i !== config.partIndex.footLeft && i !== config.partIndex.footRight && measure.touch[i]) measure.bodyOn = true;
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
        push: 0, amp: 0, period: config.body.platformPeriodSeconds, phaseShift: 0, projectiles: [],
        frames: config.episode.frames, task: "stand", phase: 0,
      }, options);
      this.free = false;
      this.world = world;
      this.weights = weights;
      this.frame = 0;
      this.returnSum = 0;
      this.scoreCount = 0;
      this.hidden = new Float32Array(hiddenCountOf(weights.length) || config.net.hiddenUnitCount);
      this.nextHidden = new Float32Array(this.hidden.length);
      this.action = new Float32Array(config.net.jointCount);
      this.previousAction = new Float32Array(config.net.jointCount);
      this.input = new Float32Array(config.net.inputCount);
      this.previousJointAngle = new Float32Array(config.net.jointCount);
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
        measure.goalDx = clamp01(measure.goalDistanceMeters / 4, -1, 1);
        measure.goalDy = clamp01((this.options.goalY || 0) / 2, -1, 1);
      }
      const fallTime = Math.sqrt(Math.max(measure.centerY, 0.3) / 9.81);
      const captureX = measure.centerX + measure.centerVelocityX * fallTime;
      const input = this.input;
      let cursor = 0;
      for (let i = 0; i < config.net.jointCount; i++) {
        const joint = world.joints[i];
        const angle = b2RevoluteJoint_GetAngle(joint.id);
        input[cursor + i] = angle / joint.n;
        input[cursor + config.net.jointCount + i] = clamp01((angle - this.previousJointAngle[i]) / (config.cadence.actionEveryNFrames * config.physics.timeStep) / 6, -1, 1);
        this.previousJointAngle[i] = angle;
      }
      cursor += 2 * config.net.jointCount;
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
      input[cursor++] = measure.headHeight / config.body.headHeightMeters;
      for (let i = 0; i < 4; i++) input[cursor++] = clamp01(1 - measure.sole[i] / 0.06, 0, 1);
      input[cursor++] = measure.platformAngle / 0.2;
      input[cursor++] = clamp01(measure.platformSpin / 0.15, -2, 2);
      for (let i = 0; i < config.net.jointCount; i++) input[cursor++] = this.action[i];
      input[cursor++] = measure.touch[config.partIndex.torso] ? 1 : 0;
      input[cursor++] = measure.touch[config.partIndex.head] ? 1 : 0;
      input[cursor++] = measure.touch[config.partIndex.handLeft] ? 1 : 0;
      input[cursor++] = measure.touch[config.partIndex.handRight] ? 1 : 0;
      input[cursor++] = measure.touch[config.partIndex.footLeft] ? 1 : 0;
      input[cursor++] = measure.touch[config.partIndex.footRight] ? 1 : 0;
      input[cursor++] = measure.rayTorso;
      input[cursor++] = measure.rayFootLeft;
      input[cursor++] = measure.rayFootRight;
      const gap01 = (index) => clamp01((measure.clearance[index] || 0) / 0.6, 0, 1);
      input[cursor++] = gap01(config.partIndex.handLeft);
      input[cursor++] = gap01(config.partIndex.handRight);
      input[cursor++] = gap01(config.partIndex.shinLeft);
      input[cursor++] = gap01(config.partIndex.shinRight);
      input[cursor++] = gap01(config.partIndex.head);
      input[cursor++] = measure.rayHead;
      input[cursor++] = Math.sin(measure.headAngle);
      input[cursor++] = Math.cos(measure.headAngle);
      input[cursor++] = gap01(config.partIndex.thighLeft);
      input[cursor++] = gap01(config.partIndex.thighRight);
      input[cursor++] = gap01(config.partIndex.forearmLeft);
      input[cursor++] = gap01(config.partIndex.forearmRight);
      input[cursor++] = measure.goalDx || 0;
      input[cursor++] = measure.goalDy || 0;
      input[cursor++] = isStanding(measure) ? 1 : 0;
      if (cursor !== config.net.sensedInputCount) throw new Error("base observation " + cursor);
      for (let i = 0; i < config.net.partCount; i++) {
        input[cursor++] = clamp01((measure.partVelocityX[i] || 0) / config.scale.partSpeed, -2, 2);
        input[cursor++] = clamp01((measure.partVelocityY[i] || 0) / config.scale.partSpeed, -2, 2);
        input[cursor++] = clamp01((measure.partSpin[i] || 0) / config.scale.partSpin, -2, 2);
      }
      for (const term of scoreDecision(measure).terms) {
        if (term.scoreOnly) continue;
        input[cursor++] = clamp01(term.v / config.scale.rewardInput, -2, 2);
      }
      if (cursor !== config.net.inputCount) throw new Error("observation length " + cursor);
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
      if (measure.clearance[config.partIndex.head] < 0.03) {
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
        const speed = clamp01(config.episode.motorGain * (target - b2RevoluteJoint_GetAngle(joint.id)), -config.episode.motorSpeedLimit, config.episode.motorSpeedLimit);
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

  function physicsHz() { return 1 / config.physics.timeStep; }

  function situation(randomSource, pose) {
    const side = randomSource() < 0.5 ? -1 : 1;
    const shared = {
      push: 0, amp: 0, period: config.body.platformPeriodSeconds, phaseShift: 0, projectiles: [],
      frames: config.episode.frames, task: "live",
      goalX: side * (config.episode.goalNearMeters + randomSource() * config.episode.goalFarExtraMeters),
      goalY: 0,
    };
    const phase = activeTrainingPhase();
    let chosen = pose;
    if (config.curriculum.byPhases || !chosen || chosen === "mix") {
      if (phase.start === "up") chosen = "up";
      else if (phase.start === "down") chosen = randomSource() < 0.5 ? "supine" : "prone";
      else chosen = randomSource() < config.start.uprightStartChance ? "up" : (randomSource() < 0.5 ? "supine" : "prone");
    }
    const falling = chosen === "up" && phase.falls && randomSource() < config.start.fallDropChance;
    const uprightDrop = falling ? config.start.fallDropMinMeters + randomSource() * (config.start.fallDropMaxMeters - config.start.fallDropMinMeters) : config.start.uprightDropMeters;
    const upright = () => Object.assign(shared, { tilt: (randomSource() * 2 - 1) * config.start.uprightTiltRadians, drop: uprightDrop, settle: 0, pose: "up" });
    const down = (sign) => Object.assign(shared, {
      tilt: sign * (config.start.downTiltMinRadians + randomSource() * config.start.downTiltSpanRadians), drop: config.start.downDropMeters, settle: config.start.downSettleFrames, pose: "down",
    });
    const started = chosen === "up" ? upright() : down(chosen === "prone" ? -1 : 1);
    if (phase.balls && randomSource() < config.start.ballChance) {
      started.ballFrame = config.start.ballThrowFrame;
      started.ballSide = randomSource() < 0.5 ? -1 : 1;
      started.ballSpeed = config.start.ballSpeed;
    }
    return started;
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
    build, Episode, situation, varyStart, spawnThrown, resetPose,
    platformHalfWidth: config.body.platformHalfWidth,
    platformHalfThickness: config.body.platformHalfThickness,
    setRandom(seed) { random = makeLocalRandom(seed); },
    randomNormal,
  };
}
