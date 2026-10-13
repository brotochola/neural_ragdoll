// One frame's reward is the distance and rotation of the targets that are on. An off target adds nothing.
import { config } from "../config.js";

const part = config.partIndex;
const weight = config.reward;
const standingCos = Math.cos(config.body.standingTorsoAngleRadians);
const termCount = config.net.rewardRowCount;
const terms = [];
for (let i = 0; i < config.targets.length; i++) {
  const spec = config.targets[i];
  terms.push({
    n: spec.id + " position",
    tip:
      "Distance from the " +
      spec.id +
      " to its target. +" +
      targetPositionWeight(spec) +
      " on the mark, 0 at " +
      targetSpan(spec) +
      " m, and approaching -" +
      targetPositionWeight(spec) +
      " far away. It never flattens, so there is always a way back. Zero while this target is off or its gain is zero.",
    v: 0,
    scoreOnly: false,
  });
  terms.push({
    n: spec.id + " rotation",
    tip:
      "How well the " +
      spec.id +
      " faces its target. +" +
      weight.targetRotation +
      " when level, -" +
      weight.targetRotation +
      " when flipped. Zero while this target is off or its gain is zero.",
    v: 0,
    scoreOnly: false,
  });
}
for (let g = 0; g < config.contact.groups.length; g++) {
  const group = config.contact.groups[g];
  terms.push({
    n: group.id + " contact",
    tip:
      "How hard " +
      group.parts.join(", ") +
      " is pressed into the floor or a thrown object. 0 when clear, -" +
      config.contact.weight +
      " when fully pressed. Hands and feet are not in any group.",
    v: 0,
    scoreOnly: false,
  });
}
const scored = { total: 0, max: 0, onFeet: 0, terms };

// Each target carries its own span, weight, and gain. Gain 0 drops the row from the reward
// and its five input columns. The numbers live on config.targets.
export function targetGain(id, target) {
  // The episode can multiply the gain (gainCmd, the level that commands this part).
  if (target && typeof target.gain === "number") return target.gain > 0 ? target.gain : 0;
  const gains = config.targetGains;
  const g = gains ? gains[id] : 1;
  return typeof g === "number" && g > 0 ? g : 0;
}

// The span can come from the episode (target.span): it tightens when that part is commanded.
// Otherwise it is the spec's span.
export function targetSpan(spec, target) {
  if (target && target.span > 0) return target.span;
  if (spec.spanCmd > 0 && spec.span <= 0) return spec.spanCmd;
  if (spec.span > 0) return spec.span;
  return spec.id === "footR" || spec.id === "footL"
    ? config.skill.footSpanMeters
    : config.body.targetSpanMeters;
}

export function targetPositionWeight(spec) {
  if (spec.weight > 0) return spec.weight;
  return spec.id === "footR" || spec.id === "footL"
    ? weight.footPosition
    : weight.targetPosition;
}

// Standing: both feet down, nothing else touching, torso upright, head high, a real base, and still.
export function isStanding(measure) {
  if (!measure || !measure.touch || !measure.touch[part.footLeft] || !measure.touch[part.footRight]) return false;
  for (let i = 0; i < measure.touch.length; i++) {
    if (i !== part.footLeft && i !== part.footRight && measure.touch[i]) return false;
  }
  return (measure.torsoCos || 0) > standingCos
    && measure.headHeight > config.body.standingHeadMeters
    && measure.stanceWidth >= config.body.standingStanceMinMeters
    && measure.stanceWidth <= config.body.standingStanceMaxMeters
    && Math.abs(measure.torsoSpin) < 1.5
    && Math.abs(measure.centerVelocityX) < 0.5;
}

function angleErrorCos(targetAngle, partSin, partCos) {
  const targetSin = Math.sin(targetAngle || 0);
  const targetCos = Math.cos(targetAngle || 0);
  return targetCos * partCos + targetSin * partSin;
}

function footTarget(id) {
  return id === "footR" || id === "footL";
}

// Position row: +weight on the mark, -weight at infinity, and never a plateau.
//
// It used to be linear (1 - 2r) and it flattened at -weight past one span. A fall is about
// 1.5 m, so the whole fall was one number and there was no way back up.
// 1/(1+r^2) fixed the plateau but the floor became 0 instead of -weight, so lying down
// scored about 0 instead of about -weight. With the feet worth 6 of 12 points, lying down
// was almost as good as standing (measured: 10.92 standing against 8.00 down, where the
// linear version gave 7.49 against -1.48).
//
// The shape now is r / sqrt(1+r^2), with r = d/span:
//   - slope -2*weight/span on the mark, the same as the linear row. The 10 cm sensitivity,
//     which is where the net learns, is unchanged.
//   - it tends to -weight far away, so a fall still costs the whole reward.
//   - the derivative 1/(1+r^2)^1.5 is always positive: it always falls, and never flattens.
function positionRow(weight, distance, span) {
  const ratio = distance / span;
  return weight * (1 - 2 * (ratio / Math.sqrt(1 + ratio * ratio)));
}
export function markError(measure, targetIndex) {
  const spec = config.targets[targetIndex];
  const target = measure && measure.targets && measure.targets[targetIndex];
  const index = spec ? part[spec.part] : 0;
  if (!spec || !target || !measure || !measure.partX) return { dist: Infinity, align: -1 };
  const dx = (target.x || 0) - (measure.partX[index] || 0);
  const dy = (target.y || 0) - (measure.partY[index] || 0);
  const partCos = measure.partCos && measure.partCos[index];
  const align = angleErrorCos(target.angle, (measure.partSin && measure.partSin[index]) || 0, partCos == null ? 1 : partCos);
  return { dist: Math.hypot(dx, dy), align };
}

export function onMark(measure, targetIndex) {
  const spec = config.targets[targetIndex];
  const foot = spec && footTarget(spec.id);
  const reach = foot ? config.skill.footReachMeters : config.skill.targetReachMeters;
  const radians = foot ? config.skill.footReachRadians : config.skill.targetReachRadians;
  const error = markError(measure, targetIndex);
  return error.dist <= reach && error.align >= Math.cos(radians);
}

// Writes the total. When writeRows is set, also the fixed panel rows. Allocates nothing.
function writeRewardRows(measure, writeRows) {
  const targets = measure.targets && measure.targets.length === config.targets.length ? measure.targets : config.targets;
  let total = 0;
  let ceiling = 0;
  let row = 0;
  for (let i = 0; i < config.targets.length; i++) {
    const spec = config.targets[i];
    const target = targets[i] || spec;
    const index = part[spec.part];
    const gain = targetGain(spec.id, target);
    const on = gain > 0 && !!(target && target.on);
    let position = 0;
    let rotation = 0;
    if (on) {
      const dx = (target.x || 0) - ((measure.partX && measure.partX[index]) || 0);
      const dy = (target.y || 0) - ((measure.partY && measure.partY[index]) || 0);
      const span = targetSpan(spec, target);
      const positionWeight = targetPositionWeight(spec);
      position = gain * positionRow(positionWeight, Math.hypot(dx, dy), span);
      const partCos = measure.partCos && measure.partCos[index];
      rotation = gain * weight.targetRotation * angleErrorCos(target.angle, (measure.partSin && measure.partSin[index]) || 0, partCos == null ? 1 : partCos);
      ceiling += gain * (positionWeight + weight.targetRotation);
    }
    if (writeRows) {
      terms[row].v = position;
      terms[row + 1].v = rotation;
    }
    row += 2;
    total += position + rotation;
  }
  const contactWeight = config.contact.weight;
  const contact = measure.contact;
  for (let g = 0; g < config.contact.groups.length; g++) {
    let amount = contact ? contact[g] || 0 : 0;
    if (amount < 0) amount = 0;
    else if (amount > 1) amount = 1;
    const penalty = -contactWeight * amount;
    if (writeRows) terms[row].v = penalty;
    row += 1;
    total += penalty;
  }
  if (row !== termCount) throw new Error("reward rows " + row);
  scored.total = total;
  scored.max = ceiling;
  return total;
}

// The same total split per target. The probe and the diagnostic panel use it, not the reward.
const perTarget = config.targets.map((spec) => ({ id: spec.id, position: 0, rotation: 0 }));

export function scoreByTarget(measure) {
  writeRewardRows(measure, false);
  for (let i = 0; i < config.targets.length; i++) {
    const spec = config.targets[i];
    const target = measure.targets && measure.targets[i];
    const index = part[spec.part];
    const row = perTarget[i];
    row.position = 0;
    row.rotation = 0;
    if (!target || !target.on || !targetGain(spec.id, target)) continue;
    const span = targetSpan(spec, target);
    const dx = (target.x || 0) - ((measure.partX && measure.partX[index]) || 0);
    const dy = (target.y || 0) - ((measure.partY && measure.partY[index]) || 0);
    row.position = positionRow(targetPositionWeight(spec), Math.hypot(dx, dy), span);
    const partCos = measure.partCos && measure.partCos[index];
    row.rotation = weight.targetRotation * angleErrorCos(target.angle, (measure.partSin && measure.partSin[index]) || 0, partCos == null ? 1 : partCos);
  }
  return perTarget;
}

// Number only. The episode uses it on every reward sample.
export function scoreTotal(measure) {
  return writeRewardRows(measure, false);
}

// The same rows, rewritten. The panel reads them on the same turn.
export function scoreDecision(measure) {
  writeRewardRows(measure, true);
  return scored;
}
