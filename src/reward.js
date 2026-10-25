// One frame's reward is the targets that are on, plus upright and contact. An off target adds nothing.
import { config } from "../config.js";

const part = config.partIndex;
const weight = config.reward;
const standingCos = Math.cos(config.body.standingTorsoAngleRadians);
const termCount = config.net.rewardRowCount;
const terms = [];
for (let i = 0; i < config.targets.length; i++) {
  const spec = config.targets[i];
  const positionWeight = targetPositionWeight(spec);
  const onMarkValue = positionWeight * (spec.ride ? 1 + weight.wideWeight : 1);
  const axisTip = (axis) =>
    "Distance from the " +
    spec.id +
    " to its target along " +
    axis +
    ". +" +
    onMarkValue +
    " on the mark, approaching -" +
    onMarkValue +
    " far away." +
    (spec.ride
      ? " A wide copy keeps a slope out to " + weight.wideSpanMeters + " m."
      : "") +
    " Zero while this target is off or its gain is zero.";
  terms.push({ n: spec.id + " x", tip: axisTip("x"), v: 0, scoreOnly: false });
  terms.push({ n: spec.id + " y", tip: axisTip("y"), v: 0, scoreOnly: false });
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
  const closeTip = (axis) =>
    "How many meters the " +
    spec.id +
    " closed along " +
    axis +
    " since the previous reward sample, times " +
    weight.closeWeight +
    ". Positive when that axis got closer. Zero on the first sample, and while this target is off.";
  terms.push({ n: spec.id + " close x", tip: closeTip("x"), v: 0, scoreOnly: false });
  terms.push({ n: spec.id + " close y", tip: closeTip("y"), v: 0, scoreOnly: false });
}
for (let g = 0; g < config.contact.groups.length; g++) {
  const group = config.contact.groups[g];
  const share = group.share > 0 ? group.share : 1;
  const floor = config.contact.weight * share;
  terms.push({
    n: group.id + " contact",
    tip:
      (group.parts.length > 1 ? "Mean of how hard " : "How hard ") +
      group.parts.join(", ") +
      " is pressed into the floor or a thrown object. 0 when clear, -" +
      floor +
      " when fully pressed. Hands and feet are not in any group.",
    v: 0,
    scoreOnly: false,
  });
}
terms.push({
  n: "upright",
  tip:
    "Torso cosine plus head height over a standing head, minus one. +" +
    weight.upright +
    " standing, -" +
    weight.upright +
    " lying flat. Plus " + (weight.planted || 0) + " with both feet on the floor, minus that with one up (not while walking).",
  v: 0,
  scoreOnly: false,
});
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
// While walking (measure.walking) one foot down is enough and the stance width is not checked:
// a stride always has a foot in the air, and the feet cross each other.
export function isStanding(measure) {
  if (!measure || !measure.touch) return false;
  const left = measure.touch[part.footLeft];
  const right = measure.touch[part.footRight];
  if (measure.walking ? !(left || right) : !(left && right)) return false;
  for (let i = 0; i < measure.touch.length; i++) {
    if (i !== part.footLeft && i !== part.footRight && measure.touch[i]) return false;
  }
  if (measure.walking)
    return (measure.torsoCos || 0) > standingCos
      && measure.headHeight > config.body.standingHeadMeters
      && Math.abs(measure.torsoSpin) < 1.5
      && Math.abs(measure.centerVelocityX) < (measure.standSpeedLimit || 0.5);
  return (measure.torsoCos || 0) > standingCos
    && measure.headHeight > config.body.standingHeadMeters
    && measure.stanceWidth >= config.body.standingStanceMinMeters
    && measure.stanceWidth <= config.body.standingStanceMaxMeters
    && Math.abs(measure.torsoSpin) < 1.5
    && Math.abs(measure.centerVelocityX) < (measure.standSpeedLimit || 0.5);
}

function angleErrorCos(targetAngle, partSin, partCos) {
  const targetSin = Math.sin(targetAngle || 0);
  const targetCos = Math.cos(targetAngle || 0);
  return targetCos * partCos + targetSin * partSin;
}

function footTarget(id) {
  return id === "footR" || id === "footL";
}

function footContactOf(measure, id) {
  const feet = measure && measure.footContact;
  if (!feet) return 1;
  const value = id === "footR" ? feet[1] : id === "footL" ? feet[0] : 1;
  return value || 0;
}

// A step in the air: this foot is off, the other is planted, and the torso is still up.
// Then Y and close Y stay at zero. Both feet off, or a fall, still pays Y.
function isFootSwing(measure, spec) {
  if (!spec || !footTarget(spec.id)) return false;
  const other = spec.id === "footR" ? "footL" : "footR";
  return footContactOf(measure, spec.id) < config.skill.swingContact
    && footContactOf(measure, other) > config.skill.stanceContact
    && (measure.torsoCos || 0) > config.skill.swingTorsoCos;
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
// The shape now is r / (1+|r|), with r = |one axis| / span:
//   - slope -2*weight/span on the mark, the same as the linear row. The 10 cm sensitivity,
//     which is where the net learns, is unchanged.
//   - it tends to -weight far away, so a fall still costs the whole reward.
//   - the derivative 1/(1+|r|)^2 is always positive: it always falls, and never flattens.
function positionRow(weight, distance, span) {
  const ratio = Math.abs(distance / span);
  return weight * (1 - 2 * (ratio / (1 + ratio)));
}

// One axis. The wide copy stays inside the same row: the tight span is already flat by 1.5 m.
function axisPosition(spec, target, delta) {
  const positionWeight = targetPositionWeight(spec);
  const distance = Math.abs(delta);
  let value = positionRow(positionWeight, distance, targetSpan(spec, target));
  if (spec.ride)
    value += positionRow(positionWeight * weight.wideWeight, distance, weight.wideSpanMeters);
  return value;
}

// Meters closed since the previous reward sample. Positive when this axis got closer.
function axisClose(measure, targetIndex, vertical, distance) {
  const prev = measure && (vertical ? measure.prevAbsY : measure.prevAbsX);
  const before = prev ? prev[targetIndex] : NaN;
  if (!Number.isFinite(before)) return 0;
  return (weight.closeWeight || 0) * (before - distance);
}

function uprightRow(measure) {
  const standing = config.body.standingHeadMeters;
  let height = standing > 0 ? ((measure && measure.headHeight) || 0) / standing : 0;
  if (height < 0) height = 0;
  else if (height > 1) height = 1;
  let mix = ((measure && measure.torsoCos) || 0) + height - 1;
  if (mix < -1) mix = -1;
  else if (mix > 1) mix = 1;
  // Both feet planted: +planted, one foot in the air: -planted. Without it a fresh run learned
  // to stand on ONE foot (measured: right foot up 95% of the last 3 s, head at 1.68 m), which
  // the reward barely noticed and isStanding refuses. The walk level sets this to 0 through
  // measure.plantedWeight, since walking is lifting a foot.
  const planted = measure && measure.plantedWeight != null ? measure.plantedWeight : weight.planted || 0;
  let feet = 0;
  if (planted && measure && measure.touch) {
    const both = measure.touch[part.footLeft] && measure.touch[part.footRight];
    feet = planted * (both ? 1 : -1);
  }
  return weight.upright * mix + feet;
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
    let xRow = 0;
    let yRow = 0;
    let rotation = 0;
    let closeX = 0;
    let closeY = 0;
    if (on) {
      const dx = (target.x || 0) - ((measure.partX && measure.partX[index]) || 0);
      const dy = (target.y || 0) - ((measure.partY && measure.partY[index]) || 0);
      const positionWeight = targetPositionWeight(spec);
      xRow = gain * axisPosition(spec, target, dx);
      const swing = isFootSwing(measure, spec);
      yRow = swing ? 0 : gain * axisPosition(spec, target, dy);
      const partCos = measure.partCos && measure.partCos[index];
      rotation = gain * weight.targetRotation * angleErrorCos(target.angle, (measure.partSin && measure.partSin[index]) || 0, partCos == null ? 1 : partCos);
      closeX = gain * axisClose(measure, i, false, Math.abs(dx));
      closeY = swing ? 0 : gain * axisClose(measure, i, true, Math.abs(dy));
      ceiling += gain * (2 * positionWeight * (spec.ride ? 1 + weight.wideWeight : 1) + weight.targetRotation);
    }
    if (writeRows) {
      terms[row].v = xRow;
      terms[row + 1].v = yRow;
      terms[row + 2].v = rotation;
      terms[row + 3].v = closeX;
      terms[row + 4].v = closeY;
    }
    row += 5;
    total += xRow + yRow + rotation + closeX + closeY;
  }
  const contactWeight = config.contact.weight;
  const contact = measure.contact;
  for (let g = 0; g < config.contact.groups.length; g++) {
    const group = config.contact.groups[g];
    const share = group.share > 0 ? group.share : 1;
    let amount = contact ? contact[g] || 0 : 0;
    if (amount < 0) amount = 0;
    else if (amount > 1) amount = 1;
    const penalty = -contactWeight * share * amount;
    if (writeRows) terms[row].v = penalty;
    row += 1;
    total += penalty;
  }
  const upright = uprightRow(measure);
  if (writeRows) terms[row].v = upright;
  row += 1;
  total += upright;
  ceiling += weight.upright + (weight.planted || 0);
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
    const dx = (target.x || 0) - ((measure.partX && measure.partX[index]) || 0);
    const dy = (target.y || 0) - ((measure.partY && measure.partY[index]) || 0);
    row.position = axisPosition(spec, target, dx) + (isFootSwing(measure, spec) ? 0 : axisPosition(spec, target, dy));
    const partCos = measure.partCos && measure.partCos[index];
    row.rotation = weight.targetRotation * angleErrorCos(target.angle, (measure.partSin && measure.partSin[index]) || 0, partCos == null ? 1 : partCos);
  }
  return perTarget;
}

// Stores |dx| and |dy| for the next reward sample. Call this after the sample is scored.
export function rememberAxisErrors(measure) {
  const count = config.targets.length;
  if (!measure.prevAbsX || measure.prevAbsX.length !== count) {
    measure.prevAbsX = new Float32Array(count);
    measure.prevAbsY = new Float32Array(count);
  }
  const targets = measure.targets && measure.targets.length === count ? measure.targets : config.targets;
  for (let i = 0; i < count; i++) {
    const spec = config.targets[i];
    const target = targets[i] || spec;
    const index = part[spec.part];
    const on = targetGain(spec.id, target) > 0 && !!(target && target.on) && measure.partX;
    if (!on) {
      measure.prevAbsX[i] = NaN;
      measure.prevAbsY[i] = NaN;
      continue;
    }
    measure.prevAbsX[i] = Math.abs((target.x || 0) - (measure.partX[index] || 0));
    // A swing does not leave a Y error for the landing sample to charge.
    measure.prevAbsY[i] = isFootSwing(measure, spec)
      ? NaN
      : Math.abs((target.y || 0) - (measure.partY[index] || 0));
  }
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
