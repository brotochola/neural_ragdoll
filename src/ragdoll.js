// Cuerpo, articulaciones y, si hay una vista, los sprites. La pose de cada frame vive en typed arrays.
import { config } from "../config.js";
import { rotate } from "./utils.js";

const faceRayPull = 0.25;
const bodyRaySpecs = [
  ["torso", "front", 1, 0], ["torso", "back", -1, 0],
  ["chest", "front", 1, 0], ["chest", "back", -1, 0],
  ["upperArmLeft", "front", 1, 0], ["upperArmLeft", "back", -1, 0],
  ["upperArmRight", "front", 1, 0], ["upperArmRight", "back", -1, 0],
  ["forearmLeft", "front", 1, 0], ["forearmLeft", "back", -1, 0],
  ["forearmRight", "front", 1, 0], ["forearmRight", "back", -1, 0],
  ["handLeft", "tip", 0, -1],
  ["handRight", "tip", 0, -1],
  ["thighLeft", "front", 1, 0], ["thighLeft", "back", -1, 0],
  ["thighRight", "front", 1, 0], ["thighRight", "back", -1, 0],
  ["shinLeft", "front", 1, 0], ["shinLeft", "back", -1, 0],
  ["shinRight", "front", 1, 0], ["shinRight", "back", -1, 0],
  ["footLeft", "sole", 0, -1], ["footLeft", "toe", 1, 0],
  ["footRight", "sole", 0, -1], ["footRight", "toe", 1, 0],
  ["hip", "front", 1, 0], ["hip", "back", -1, 0], ["hip", "down", 0, -1],
].map(([part, name, x, y]) => ({
  part, name, x, y, index: config.partIndex[part], reach: config.body.bodyRayMeters, vision: false,
}));
const headRaySpecs = [];
function pushHeadFan(count, center, arc) {
  const reach = config.vision.headRayMeters;
  const step = arc / count;
  for (let i = 0; i < count; i++) {
    const angle = center - arc / 2 + (i + 0.5) * step;
    headRaySpecs.push({
      part: "head", name: "vision " + headRaySpecs.length, x: Math.cos(angle), y: Math.sin(angle),
      index: config.partIndex.head, reach, vision: true,
    });
  }
}
const vision = config.vision;
pushHeadFan(vision.headForwardCount, 0, vision.headForwardArcRadians);
pushHeadFan(vision.headBackCount, Math.PI, vision.headBackArcRadians);

export const faceRaySpecs = bodyRaySpecs.concat(headRaySpecs);
const soleRayIndex = { footLeft: -1, footRight: -1 };
faceRaySpecs.forEach((spec, index) => {
  if (spec.name === "sole") soleRayIndex[spec.part] = index;
});
export const facePartName = {
  torso: "torso", chest: "chest", head: "head", upperArmLeft: "arm L", upperArmRight: "arm R",
  forearmLeft: "elbow L", forearmRight: "elbow R", handLeft: "hand L", handRight: "hand R",
  thighLeft: "thigh L", thighRight: "thigh R", shinLeft: "shin L", shinRight: "shin R",
  footLeft: "foot L", footRight: "foot R", hip: "tail",
};

const skinColor = 0xffd166;
const jeanColor = 0x3f6f9f;
const shirtColor = 0xd24b4b;
const footColor = 0xc4a574;

function shadeColor(color, scale) {
  const red = (color >> 16) & 255;
  const green = (color >> 8) & 255;
  const blue = color & 255;
  return (Math.round(red * scale) << 16) | (Math.round(green * scale) << 8) | Math.round(blue * scale);
}

function darken(color) {
  return shadeColor(color, 0.62);
}

const torsoTint = shadeColor(shirtColor, 0.9);
const leftArmTint = shadeColor(shirtColor, 0.84);
const leftJeanTint = darken(jeanColor);

function partTint(index) {
  const id = config.partIndex;
  if (index === id.torso || index === id.chest) return torsoTint;
  if (index === id.upperArmLeft || index === id.forearmLeft) return leftArmTint;
  let color = shirtColor;
  if (index === id.head || index === id.handLeft || index === id.handRight) color = skinColor;
  else if (index === id.footLeft || index === id.footRight) color = footColor;
  else if (index === id.thighLeft || index === id.thighRight || index === id.shinLeft || index === id.shinRight) color = jeanColor;
  const leftLimb = index === id.thighLeft || index === id.shinLeft || index === id.footLeft || index === id.handLeft;
  return leftLimb ? darken(color) : color;
}

function makeSolidTexture(circle) {
  const PIXI = globalThis.PIXI;
  const size = 32;
  const textureCanvas = document.createElement("canvas");
  textureCanvas.width = size;
  textureCanvas.height = size;
  const context = textureCanvas.getContext("2d");
  context.fillStyle = "#ffffff";
  if (circle) {
    context.beginPath();
    context.arc(size / 2, size / 2, size / 2, 0, Math.PI * 2);
    context.fill();
  } else context.fillRect(0, 0, size, size);
  return PIXI.Texture.from(textureCanvas);
}

function makeCapsuleTexture(part) {
  const PIXI = globalThis.PIXI;
  const ppm = 400;
  const width = Math.max(2, Math.ceil(part.hx * 2 * ppm));
  const height = Math.max(2, Math.ceil(part.hy * 2 * ppm));
  const textureCanvas = document.createElement("canvas");
  textureCanvas.width = width;
  textureCanvas.height = height;
  const context = textureCanvas.getContext("2d");
  const originX = width / 2;
  const originY = height / 2;
  const radius = part.r * ppm;
  const axisX = part.ax * ppm;
  const axisY = -part.ay * ppm;
  context.fillStyle = "#ffffff";
  context.strokeStyle = "#ffffff";
  context.lineWidth = radius * 2;
  context.lineCap = "butt";
  context.beginPath();
  context.moveTo(originX - axisX, originY - axisY);
  context.lineTo(originX + axisX, originY + axisY);
  context.stroke();
  context.beginPath();
  context.arc(originX - axisX, originY - axisY, radius, 0, Math.PI * 2);
  context.arc(originX + axisX, originY + axisY, radius, 0, Math.PI * 2);
  context.fill();
  return PIXI.Texture.from(textureCanvas);
}

const drawOrder = [
  config.partIndex.thighLeft, config.partIndex.shinLeft, config.partIndex.footLeft,
  config.partIndex.upperArmLeft, config.partIndex.forearmLeft, config.partIndex.handLeft,
  config.partIndex.torso, config.partIndex.chest, config.partIndex.head,
  config.partIndex.thighRight, config.partIndex.shinRight, config.partIndex.footRight,
  config.partIndex.upperArmRight, config.partIndex.forearmRight, config.partIndex.handRight,
];

// Creada una vez. x, y, cos, sin, vx, vy y spin leen la pose, no una copia.
class Part {
  constructor(doll, index) {
    this.doll = doll;
    this.index = index;
  }
  get x() { return this.doll.px[this.index]; }
  get y() { return this.doll.py[this.index]; }
  get cos() { return this.doll.c[this.index]; }
  get sin() { return this.doll.s[this.index]; }
  get vx() { return this.doll.vx[this.index]; }
  get vy() { return this.doll.vy[this.index]; }
  get spin() { return this.doll.spin[this.index]; }
}

export class RagDoll2D {
  constructor(box, dollConfig = config) {
    const cfg = dollConfig;
    const {
      CreateWorld, CreateBoxPolygon, CreateRevoluteJoint, CreateCircle, CreateCapsule, b2Vec2, b2BodyType,
      b2Body_GetPosition, b2Body_GetRotation, b2Body_GetLinearVelocity, b2Body_GetAngularVelocity,
      b2Shape_RayCast, b2Body_GetShapes,
    } = box;
    const dynamicBody = b2BodyType.b2_dynamicBody;
    const staticBody = b2BodyType.b2_staticBody;
    const vector = (x, y) => new b2Vec2(x, y);
    this.getPosition = b2Body_GetPosition;
    this.getRotation = b2Body_GetRotation;
    this.getVelocity = b2Body_GetLinearVelocity;
    this.getSpin = b2Body_GetAngularVelocity;
    this.castRay = b2Shape_RayCast;
    this.getShapes = b2Body_GetShapes;
    this.rayOrigin = vector(0, 0);
    this.rayDelta = vector(0, 0);
    this.rot = { x: 0, y: 0 };
    this.skinX = 0;
    this.skinY = 0;
    this.shapeList = [];
    this.partSprites = null;

    const { worldId } = CreateWorld({});
    const ground = CreateBoxPolygon({ worldId, type: staticBody, position: vector(0, -0.5), size: vector(800, 0.5), friction: 2 });
    const parts = [];
    const joints = [];
    const addBox = (centerX, centerY, halfX, halfY, density = 70) => {
      const created = CreateBoxPolygon({
        worldId, type: dynamicBody, position: vector(centerX, centerY), size: vector(halfX, halfY),
        density, friction: 2, groupIndex: -1,
      });
      const part = new Part(this, parts.length);
      part.id = created.bodyId;
      part.cx = centerX;
      part.cy = centerY;
      part.hx = halfX;
      part.hy = halfY;
      part.m = density * 4 * halfX * halfY;
      part.th = Math.max(halfX, halfY) + 0.04;
      parts.push(part);
      return parts.length - 1;
    };
    const addCapsule = (centerX, centerY, axisX, axisY, radius, density = 70) => {
      const created = CreateCapsule({
        worldId, type: dynamicBody, position: vector(centerX, centerY),
        center1: vector(-axisX, -axisY), center2: vector(axisX, axisY),
        radius, density, friction: 2, groupIndex: -1,
      });
      const length = 2 * Math.hypot(axisX, axisY);
      const area = Math.PI * radius * radius + length * 2 * radius;
      const part = new Part(this, parts.length);
      part.id = created.bodyId;
      part.cx = centerX;
      part.cy = centerY;
      part.hx = Math.abs(axisX) + radius;
      part.hy = Math.abs(axisY) + radius;
      part.r = radius;
      part.ax = axisX;
      part.ay = axisY;
      part.capsule = true;
      part.m = density * area;
      part.th = Math.max(Math.abs(axisX), Math.abs(axisY)) + radius + 0.04;
      parts.push(part);
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
    const torsoRadius = 0.13;
    const torsoAxis = 0.05;
    addCapsule(0, 1.14, 0, torsoAxis, torsoRadius);
    addCapsule(0, 1.38, 0, torsoAxis, torsoRadius);
    addCapsule(0, cfg.body.headHeightMeters, 0, cfg.body.headAxis, cfg.body.headRadius, 40);
    for (let side = 0; side < 2; side++) {
      addBox(0, 1.34, 0.05, 0.14, 50);
      addBox(0, 1.07, 0.045, 0.13, 50);
      addCapsule(0, 0.91, 0, 0.045, 0.045, 50);
      addBox(0, 0.74, 0.07, 0.22);
      addBox(0, 0.31, 0.06, 0.21);
      addCapsule(0.06, 0.04, 0.08, 0, 0.04, 90);
    }
    addJoint(0, 1, 0, 1.26, -cfg.body.waistForwardRadians, cfg.body.waistBackRadians, 160);
    addJoint(1, 2, 0, 1.56, -0.5, 0.5, 20);
    for (let side = 0; side < 2; side++) {
      const base = 3 + 6 * side;
      addJoint(1, base, 0, 1.48, -2, 2.5, 70);
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
    const tail = new Part(this, parts.length);
    tail.id = hip.bodyId;
    tail.cx = 0;
    tail.cy = 0.96;
    tail.hx = hipRadius;
    tail.hy = hipRadius;
    tail.r = hipRadius;
    tail.circle = true;
    tail.m = 40 * Math.PI * hipRadius * hipRadius;
    tail.th = hipRadius + 0.04;
    parts.push(tail);
    CreateRevoluteJoint({
      worldId, bodyIdA: parts[0].id, bodyIdB: hip.bodyId,
      anchorA: vector(0, 0.96 - parts[0].cy), anchorB: vector(0, 0),
      lowerAngle: 0, upperAngle: 0, enableLimit: true, enableMotor: false, motorSpeed: 0, maxMotorTorque: 0,
    });

    const count = parts.length;
    this.worldId = worldId;
    this.parts = parts;
    this.joints = joints;
    this.groundId = ground.bodyId;
    this.floorShapes = [this.shapeOf(ground.bodyId)];
    this.thrownIds = [];
    this.thrownShapes = [];
    this.px = new Float32Array(count);
    this.py = new Float32Array(count);
    this.c = new Float32Array(count);
    this.s = new Float32Array(count);
    this.vx = new Float32Array(count);
    this.vy = new Float32Array(count);
    this.spin = new Float32Array(count);
    this.faceRay = new Float32Array(faceRaySpecs.length);
    this.rayCount = faceRaySpecs.length;
    this.rayX0 = new Float32Array(this.rayCount);
    this.rayY0 = new Float32Array(this.rayCount);
    this.rayX1 = new Float32Array(this.rayCount);
    this.rayY1 = new Float32Array(this.rayCount);
    this.rayGap = new Float32Array(this.rayCount);
    this.rayHit = new Uint8Array(this.rayCount);
  }

  shapeOf(bodyId) {
    const shapes = this.shapeList;
    shapes.length = 0;
    this.getShapes(bodyId, shapes);
    return shapes[0];
  }

  // Único lugar que lee la pose desde Box2D. Copia x e y antes de la llamada siguiente: el vector se reutiliza.
  readPose() {
    const parts = this.parts;
    const count = parts.length;
    const px = this.px;
    const py = this.py;
    const c = this.c;
    const s = this.s;
    const vx = this.vx;
    const vy = this.vy;
    const spin = this.spin;
    const getPosition = this.getPosition;
    const getVelocity = this.getVelocity;
    const getRotation = this.getRotation;
    const getSpin = this.getSpin;
    for (let i = 0; i < count; i++) {
      const id = parts[i].id;
      const position = getPosition(id);
      px[i] = position.x;
      py[i] = position.y;
      const velocity = getVelocity(id);
      vx[i] = velocity.x;
      vy[i] = velocity.y;
      const rot = getRotation(id);
      c[i] = rot.c;
      s[i] = rot.s;
      spin[i] = getSpin(id);
    }
  }

  // Punto de la piel en dirección local. Escribe skinX y skinY.
  support(part, dirX, dirY) {
    const len = Math.hypot(dirX, dirY) || 1;
    const ux = dirX / len;
    const uy = dirY / len;
    let x;
    let y;
    if (part.capsule) {
      const along = part.ax * ux + part.ay * uy;
      const sign = along > 1e-8 ? 1 : along < -1e-8 ? -1 : 0;
      x = sign * part.ax + part.r * ux;
      y = sign * part.ay + part.r * uy;
    } else if (part.circle) {
      x = part.r * ux;
      y = part.r * uy;
    } else {
      x = Math.sign(ux) * part.hx;
      y = Math.sign(uy) * part.hy;
    }
    this.skinX = x;
    this.skinY = y;
  }

  // ponytail: listed shapes, not b2World_CastRay. That cast keeps the first fixture. Upgrade path: a category bit if the floor becomes dynamic.
  rayFraction(shapes, originX, originY, dirX, dirY) {
    const origin = this.rayOrigin;
    const delta = this.rayDelta;
    origin.x = originX;
    origin.y = originY;
    delta.x = dirX;
    delta.y = dirY;
    const castRay = this.castRay;
    let fraction = 1;
    for (let i = 0; i < shapes.length; i++) {
      const cast = castRay(shapes[i], origin, delta);
      if (cast.hit && cast.fraction >= 0 && cast.fraction < fraction) fraction = cast.fraction;
    }
    return fraction;
  }

  // Escribe las fracciones en faceRay. Si measure.recordRays, también los segmentos de debug.
  throwRays(measure) {
    const specs = faceRaySpecs;
    const face = this.faceRay;
    const px = this.px;
    const py = this.py;
    const c = this.c;
    const s = this.s;
    const parts = this.parts;
    const record = !!measure.recordRays;
    const rayX0 = this.rayX0;
    const rayY0 = this.rayY0;
    const rayX1 = this.rayX1;
    const rayY1 = this.rayY1;
    const rayGap = this.rayGap;
    const rayHit = this.rayHit;
    const rot = this.rot;
    const floorShapes = this.floorShapes;
    const thrown = this.thrownShapes;
    const thrownCount = thrown.length;
    for (let i = 0; i < specs.length; i++) {
      const spec = specs[i];
      const index = spec.index;
      const cosine = c[index];
      const sine = s[index];
      this.support(parts[index], spec.x, spec.y);
      const skinX = this.skinX;
      const skinY = this.skinY;
      rotate(skinX, skinY, cosine, sine, rot);
      const faceX = px[index] + rot.x;
      const faceY = py[index] + rot.y;
      const dirLen = Math.hypot(spec.x, spec.y) || 1;
      rotate(spec.x, spec.y, cosine, sine, rot);
      const dirX = rot.x / dirLen;
      const dirY = rot.y / dirLen;
      const castX = faceX - dirX * faceRayPull;
      const castY = faceY - dirY * faceRayPull;
      const reach = spec.reach + faceRayPull;
      const dirReachX = dirX * reach;
      const dirReachY = dirY * reach;
      const floorFraction = this.rayFraction(floorShapes, castX, castY, dirReachX, dirReachY);
      const fraction = thrownCount
        ? Math.min(floorFraction, this.rayFraction(thrown, castX, castY, dirReachX, dirReachY))
        : floorFraction;
      const floorDistance = floorFraction >= 1 ? spec.reach : Math.max(0, floorFraction * reach - faceRayPull);
      const distance = fraction >= 1 ? spec.reach : Math.max(0, fraction * reach - faceRayPull);
      face[i] = Math.min(1, distance / spec.reach);
      if (!spec.vision && floorDistance < measure.clearance[index]) measure.clearance[index] = floorDistance;
      if (i === soleRayIndex.footLeft) measure.sole[0] = floorDistance;
      else if (i === soleRayIndex.footRight) measure.sole[1] = floorDistance;
      if (record) {
        rayX0[i] = faceX;
        rayY0[i] = faceY;
        rayX1[i] = faceX + dirX * distance;
        rayY1[i] = faceY + dirY * distance;
        rayHit[i] = distance < spec.reach ? 1 : 0;
        rayGap[i] = distance;
      }
    }
  }

  findJoint(mark, partIndex) {
    const joints = this.joints;
    for (let i = 0; i < joints.length; i++) {
      const joint = joints[i];
      if (joint.mark === mark && joint.a === partIndex) return joint;
    }
    return null;
  }

  addSprite(texture, tint) {
    const PIXI = globalThis.PIXI;
    const sprite = new PIXI.Sprite(texture);
    sprite.anchor.set(0.5);
    sprite.tint = tint;
    sprite.zIndex = this.spriteDepth++;
    this.viewContainer.addChild(sprite);
    return sprite;
  }

  addPartSprite(index, boxTexture) {
    const part = this.parts[index];
    const texture = part.capsule ? makeCapsuleTexture(part) : boxTexture;
    this.partSprites[index] = this.addSprite(texture, partTint(index));
  }

  attachView(container) {
    this.viewContainer = container;
    this.spriteDepth = 1;
    this.partSprites = [];
    const boxTexture = makeSolidTexture(false);
    const circleTexture = makeSolidTexture(true);
    const id = config.partIndex;
    const thighWidth = this.parts[id.thighLeft].hx * 2;
    const shinWidth = this.parts[id.shinLeft].hx * 2;
    const armWidth = this.parts[id.upperArmLeft].hx * 2;
    const forearmWidth = this.parts[id.forearmLeft].hx * 2;
    this.kneeDiameter = Math.max(thighWidth, shinWidth) * 1.1;
    this.kneeForward = this.kneeDiameter / 2 - this.parts[id.thighLeft].hx;
    this.elbowDiameter = Math.max(armWidth, forearmWidth) * 1.1;
    this.elbowBack = this.parts[id.upperArmLeft].hx - this.elbowDiameter / 2;
    this.addPartSprite(id.thighLeft, boxTexture);
    this.addPartSprite(id.shinLeft, boxTexture);
    this.addPartSprite(id.footLeft, boxTexture);
    this.leftKnee = this.findJoint("knee", id.thighLeft);
    this.leftKneeSprite = this.addSprite(circleTexture, leftJeanTint);
    this.addPartSprite(id.upperArmLeft, boxTexture);
    this.addPartSprite(id.forearmLeft, boxTexture);
    this.addPartSprite(id.handLeft, boxTexture);
    this.leftElbow = this.findJoint("elbow", id.upperArmLeft);
    this.leftElbowSprite = this.addSprite(circleTexture, leftArmTint);
    this.tailSprite = this.addSprite(circleTexture, jeanColor);
    this.hipMarks = [];
    for (let i = 0; i < this.joints.length; i++) {
      if (this.joints[i].mark === "hip") this.hipMarks.push(this.joints[i]);
    }
    this.addPartSprite(id.torso, boxTexture);
    this.addPartSprite(id.chest, boxTexture);
    this.addPartSprite(id.head, boxTexture);
    this.addPartSprite(id.thighRight, boxTexture);
    this.addPartSprite(id.shinRight, boxTexture);
    this.addPartSprite(id.footRight, boxTexture);
    this.rightKnee = this.findJoint("knee", id.thighRight);
    this.rightKneeSprite = this.addSprite(circleTexture, jeanColor);
    this.addPartSprite(id.upperArmRight, boxTexture);
    this.addPartSprite(id.forearmRight, boxTexture);
    this.addPartSprite(id.handRight, boxTexture);
    this.rightElbow = this.findJoint("elbow", id.upperArmRight);
    this.rightElbowSprite = this.addSprite(circleTexture, shirtColor);
  }

  placeKnob(joint, sprite, diameter, forward, scale, cx, groundY, half) {
    const index = joint.a;
    const cos = this.c[index];
    const sin = this.s[index];
    rotate(joint.ax, joint.ay, cos, sin, this.rot);
    let x = this.px[index] + this.rot.x;
    let y = this.py[index] + this.rot.y;
    if (forward) {
      x += cos * forward;
      y += sin * forward;
    }
    sprite.position.set((x - cx) * scale + half, groundY - y * scale);
    const size = diameter * scale;
    sprite.width = size;
    sprite.height = size;
  }

  // Coloca los sprites desde la pose. No lee Box2D y no hace nada si no hay vista.
  update(view) {
    const sprites = this.partSprites;
    if (!sprites) return;
    const scale = view.scale;
    const cx = view.cx;
    const groundY = view.groundY;
    const half = view.width * 0.5;
    const px = this.px;
    const py = this.py;
    const c = this.c;
    const s = this.s;
    const parts = this.parts;
    for (let n = 0; n < drawOrder.length; n++) {
      const index = drawOrder[n];
      const sprite = sprites[index];
      const part = parts[index];
      sprite.position.set((px[index] - cx) * scale + half, groundY - py[index] * scale);
      sprite.rotation = -Math.atan2(s[index], c[index]);
      sprite.width = part.hx * 2 * scale;
      sprite.height = part.hy * 2 * scale;
    }
    this.placeKnob(this.leftKnee, this.leftKneeSprite, this.kneeDiameter, this.kneeForward, scale, cx, groundY, half);
    this.placeKnob(this.rightKnee, this.rightKneeSprite, this.kneeDiameter, this.kneeForward, scale, cx, groundY, half);
    this.placeKnob(this.leftElbow, this.leftElbowSprite, this.elbowDiameter, this.elbowBack, scale, cx, groundY, half);
    this.placeKnob(this.rightElbow, this.rightElbowSprite, this.elbowDiameter, this.elbowBack, scale, cx, groundY, half);
    const hips = this.hipMarks;
    if (hips.length === 2) {
      let hipX = 0;
      let hipY = 0;
      for (let k = 0; k < 2; k++) {
        const joint = hips[k];
        const index = joint.a;
        rotate(joint.ax, joint.ay, c[index], s[index], this.rot);
        hipX += px[index] + this.rot.x;
        hipY += py[index] + this.rot.y;
      }
      const torso = config.partIndex.torso;
      const torsoPart = parts[torso];
      const tailRadius = torsoPart.hx * 1.1;
      const x = hipX * 0.5 + c[torso] * (torsoPart.hx - tailRadius);
      const y = hipY * 0.5 + s[torso] * (torsoPart.hx - tailRadius);
      this.tailSprite.position.set((x - cx) * scale + half, groundY - y * scale);
      const size = tailRadius * 2 * scale;
      this.tailSprite.width = size;
      this.tailSprite.height = size;
    }
  }
}
