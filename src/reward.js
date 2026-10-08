// One row per thing the body can do. Weights live in config.js.
import {
  partIndex as part,
  rewardWeight as weight,
  headHeightMeters,
  torsoHeightMeters,
  handClearMeters,
  thighClearMeters,
  balanceSpanMeters,
  goalSpanMeters,
  goalWalkSpeed,
  standingHeadMeters,
  standingTorsoAngleRadians,
  standingStanceMinMeters,
  standingStanceMaxMeters,
  comfortableStanceMeters,
  thighSpreadMinRadians,
  thighSpreadMaxRadians,
} from "../config.js";

export const clamp = (value, low, high) => Math.max(low, Math.min(high, value));

// True only when both feet are down, nothing else touches, and the body is upright and quiet.
export function isStanding(measure) {
  if (!measure || !measure.touch || !measure.touch[part.footLeft] || !measure.touch[part.footRight]) return false;
  for (let i = 0; i < measure.touch.length; i++) {
    if (i !== part.footLeft && i !== part.footRight && measure.touch[i]) return false;
  }
  return Math.abs(measure.torsoAngle) < standingTorsoAngleRadians
    && measure.headHeight > standingHeadMeters
    && measure.stanceWidth >= standingStanceMinMeters
    && measure.stanceWidth <= standingStanceMaxMeters
    && Math.abs(measure.torsoSpin) < 1.5
    && Math.abs(measure.centerVelocityX) < 0.5;
}

function uprightRow(name, tip, angle, rowWeight) {
  return { name, tip, value: rowWeight * Math.cos(angle || 0) };
}

function heightRow(name, tip, height, fullHeight, rise, handsAreFree, rowWeight) {
  const height01 = clamp((height || 0) / fullHeight, 0, 1);
  // Floor value is -weight. Full height with a free torso reaches +weight.
  const value = -rowWeight + rise * handsAreFree * 2 * rowWeight * height01;
  return { name, tip, value };
}

// Score one sampled frame. `measure` is filled by the sim.
export function scoreDecision(measure) {
  const torsoFace = Math.cos(measure.torsoAngle || 0);
  const rise = clamp((torsoFace + 0.2) / 1.2, 0, 1);
  const handLeft = 1 - clamp((measure.clearance?.[part.handLeft] ?? 1) / handClearMeters, 0, 1);
  const handRight = 1 - clamp((measure.clearance?.[part.handRight] ?? 1) / handClearMeters, 0, 1);
  const handsAreFree = 1 - clamp(handLeft + handRight, 0, 1);
  const park = clamp((torsoFace - 0.55) / 0.35, 0, 1);
  const angle = measure.partAngle || [];

  const rows = [
    uprightRow("Torso angle", "Cosine of the torso. +1 is upright, 0 is lying on the side, -1 is upside down.", measure.torsoAngle, weight.torsoAngle),
    uprightRow("Head angle", "Cosine of the head. +1 is upright.", measure.headAngle, weight.headAngle),
    uprightRow("Left upper arm", "Cosine of the left upper arm. +1 is vertical.", angle[part.upperArmLeft], weight.upperArmAngle),
    uprightRow("Right upper arm", "Cosine of the right upper arm. +1 is vertical.", angle[part.upperArmRight], weight.upperArmAngle),
    uprightRow("Left forearm", "Cosine of the left forearm. +1 is vertical.", angle[part.forearmLeft], weight.forearmAngle),
    uprightRow("Right forearm", "Cosine of the right forearm. +1 is vertical.", angle[part.forearmRight], weight.forearmAngle),
    uprightRow("Left thigh", "Cosine of the left thigh. +1 is vertical, like standing.", angle[part.thighLeft], weight.thighAngle),
    uprightRow("Right thigh", "Cosine of the right thigh. +1 is vertical, like standing.", angle[part.thighRight], weight.thighAngle),
    uprightRow("Left shin", "Cosine of the left shin. +1 is vertical. A shin near the floor can still be vertical.", angle[part.shinLeft], weight.shinAngle),
    uprightRow("Right shin", "Cosine of the right shin. +1 is vertical.", angle[part.shinRight], weight.shinAngle),
    heightRow("Head height", "Head height against a standing reach. Stays at the floor value until the torso turns up, and stays there if a hand is holding the body.", measure.headHeight, headHeightMeters, rise, handsAreFree, weight.headHeight),
    heightRow("Torso height", "Torso height. Same gate as the head: no credit for a hand bridge.", measure.torsoHeight, torsoHeightMeters, rise, handsAreFree, weight.torsoHeight),
  ];

  const handTip = "Near the floor this subtracts a little. If the torso is already upright and the hand is still down, it subtracts much more.";
  rows.push({
    name: "Left hand",
    tip: handTip,
    value: -(weight.handOnFloor + weight.handPark * park) * handLeft,
  });
  rows.push({
    name: "Right hand",
    tip: handTip,
    value: -(weight.handOnFloor + weight.handPark * park) * handRight,
  });

  const thighTip = "Subtracts while the thigh is within 35 cm of the floor. The shin is not in this row.";
  const thighClear = (index) => 1 - clamp((measure.clearance?.[index] ?? 1) / thighClearMeters, 0, 1);
  rows.push({ name: "Left thigh clearance", tip: thighTip, value: -weight.thighOnFloor * thighClear(part.thighLeft) });
  rows.push({ name: "Right thigh clearance", tip: thighTip, value: -weight.thighOnFloor * thighClear(part.thighRight) });
  let spread = Math.abs((angle[part.thighLeft] || 0) - (angle[part.thighRight] || 0)) % (Math.PI * 2);
  if (spread > Math.PI) spread = Math.PI * 2 - spread;
  const spreadPenalty = spread < thighSpreadMinRadians
    ? (thighSpreadMinRadians - spread) / thighSpreadMinRadians
    : spread > thighSpreadMaxRadians
      ? (spread - thighSpreadMaxRadians) / (Math.PI - thighSpreadMaxRadians)
      : 0;
  rows.push({
    name: "Thigh spread",
    tip: "Angle between the thighs. Subtracts below 10° and above 60°, including on the floor. A step in between adds nothing. A full split subtracts the whole weight.",
    value: -weight.thighSpread * clamp(spreadPenalty, 0, 1),
  });

  const footTip = "Subtracts while that foot is off the floor. A planted foot adds nothing by itself.";
  rows.push({ name: "Left foot", tip: footTip, value: measure.touch?.[part.footLeft] ? 0 : -weight.footInAir });
  rows.push({ name: "Right foot", tip: footTip, value: measure.touch?.[part.footRight] ? 0 : -weight.footInAir });

  const footMid = ((measure.footXLeft || 0) + (measure.footXRight || 0)) / 2;
  const lean = clamp(Math.abs((measure.centerX || 0) - footMid) / balanceSpanMeters, 0, 1);
  rows.push({
    name: "Balance",
    tip: "Center of mass against the midpoint of the feet. Above the feet it adds. Hanging off to one side it subtracts.",
    value: weight.balance * (1 - 2 * lean),
  });

  const headDown = Math.max(0, -(measure.headVelocityY || 0));
  const headOnFloor = measure.touch?.[part.head] ? 1 : 0;
  const headResting = headOnFloor * weight.headImpact;
  const headStrike = headOnFloor && headDown > 1 ? weight.headImpact * clamp(headDown / 3, 0, 2) : 0;
  rows.push({
    name: "Head impact",
    tip: "Subtracts while the head is touching the floor. If it is also moving down faster than 1 m/s, it subtracts more, up to twice the weight again.",
    value: -(headResting + headStrike),
  });

  const bothFeet = (measure.touch?.[part.footLeft] ? 1 : 0) + (measure.touch?.[part.footRight] ? 1 : 0);
  let otherTouch = 0;
  if (measure.touch) {
    for (let i = 0; i < measure.touch.length; i++) {
      if (i !== part.footLeft && i !== part.footRight && measure.touch[i]) otherTouch = 1;
    }
  }
  const upright = clamp((torsoFace - 0.7) / 0.3, 0, 1);
  const onFeet = (bothFeet === 2 ? 1 : 0) * (1 - otherTouch) * upright * clamp((measure.headHeight || 0) / standingHeadMeters, 0, 1);
  const stanceComfort = Math.exp(-14 * ((measure.stanceWidth || 0) - comfortableStanceMeters) ** 2);
  rows.push({
    name: "Standing",
    tip: "The large bonus. Both feet down, nothing else touching, torso upright, head high. A stance near 29 cm adds a little more. A true stand, quiet and in a real stance, adds one more.",
    value: onFeet * (weight.standing + weight.stanceComfort * stanceComfort) + (isStanding(measure) ? weight.standingFlag : 0),
  });

  const hasGoal = measure.goalDistanceMeters != null;
  const closeness = hasGoal ? 1 - clamp(Math.abs(measure.goalDistanceMeters) / goalSpanMeters, 0, 1) : 0;
  const toward = hasGoal
    ? clamp((measure.goalDistanceMeters > 0 ? measure.centerVelocityX || 0 : -(measure.centerVelocityX || 0)) / goalWalkSpeed, 0, 1)
    : 0;
  rows.push({
    name: "Goal distance",
    tip: "Closer to the marker adds more. Worth more while standing than while on the floor, so diving at it does not pay.",
    value: hasGoal ? (onFeet ? weight.goalDistanceUp : weight.goalDistanceDown) * closeness : 0,
  });
  rows.push({
    name: "Goal speed",
    tip: "Adds while the center of mass moves toward the marker. Worth more while standing.",
    value: hasGoal ? (onFeet ? weight.goalSpeedUp : weight.goalSpeedDown) * toward : 0,
  });

  let total = 0;
  let ceiling = 0;
  for (const row of rows) {
    total += row.value;
    if (row.name === "Goal distance" && !hasGoal) continue;
    if (row.name === "Goal speed" && !hasGoal) continue;
    if (row.name === "Goal distance") ceiling += weight.goalDistanceUp;
    else if (row.name === "Goal speed") ceiling += weight.goalSpeedUp;
    else if (row.name === "Standing") ceiling += weight.standing + weight.stanceComfort + weight.standingFlag;
    else if (row.name === "Head height") ceiling += weight.headHeight;
    else if (row.name === "Torso height") ceiling += weight.torsoHeight;
    else if (row.value > 0 || row.name.endsWith("angle") || row.name.endsWith("arm") || row.name.endsWith("thigh") || row.name.endsWith("shin") || row.name === "Balance") {
      const match = {
        "Torso angle": weight.torsoAngle,
        "Head angle": weight.headAngle,
        "Left upper arm": weight.upperArmAngle,
        "Right upper arm": weight.upperArmAngle,
        "Left forearm": weight.forearmAngle,
        "Right forearm": weight.forearmAngle,
        "Left thigh": weight.thighAngle,
        "Right thigh": weight.thighAngle,
        "Left shin": weight.shinAngle,
        "Right shin": weight.shinAngle,
        Balance: weight.balance,
      };
      if (match[row.name]) ceiling += match[row.name];
    }
  }
  return { total, max: ceiling, onFeet, terms: rows.map((row) => ({ n: row.name, tip: row.tip, v: row.value })) };
}
