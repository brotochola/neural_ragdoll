// Una fila por cada cosa que el cuerpo puede hacer. Los pesos viven en config.reward.
// Puntuar un frame no guarda estado: las mismas medidas dan el mismo número.
import { config, activeTrainingPhase } from "../config.js";
import { clamp } from "./utils.js";

const part = config.partIndex;
const weight = config.reward;
const standingCos = Math.cos(config.body.standingTorsoAngleRadians);

// Parado de verdad: los dos pies abajo, nada más tocando, torso erguido, cabeza alta, base real y quieto.
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

function uprightRow(name, tip, cosine, rowWeight) {
  return { name, tip, value: rowWeight * (cosine || 0) };
}

function heightRow(name, tip, height, fullHeight, rise, rowWeight) {
  const height01 = clamp((height || 0) / fullHeight, 0, 1);
  // En el piso vale -peso. A altura completa, con el torso girando hacia arriba, llega a +peso.
  const value = -rowWeight + rise * 2 * rowWeight * height01;
  return { name, tip, value };
}

// Puntúa un frame ya medido. El total es la suma de las filas; las marcadas scoreOnly solo se muestran.
export function scoreDecision(measure) {
  const cosineOf = (index) => measure.partCos?.[index] || 0;
  const sineOf = (index) => measure.partSin?.[index] || 0;
  const torsoFace = measure.torsoCos || 0;
  const rise = clamp((torsoFace + 0.2) / 1.2, 0, 1);
  const upright = clamp((torsoFace - 0.7) / 0.3, 0, 1);
  const handLeft = 1 - clamp((measure.clearance?.[part.handLeft] ?? 1) / config.body.handClearMeters, 0, 1);
  const handRight = 1 - clamp((measure.clearance?.[part.handRight] ?? 1) / config.body.handClearMeters, 0, 1);
  const park = clamp((torsoFace - 0.55) / 0.35, 0, 1);

  const rows = [
    // Coseno del ángulo: +1 parado, 0 de costado, -1 cabeza abajo.
    uprightRow("Torso angle", "Cosine of the torso. +1 is upright, 0 is lying on the side, -1 is upside down.", cosineOf(part.torso), weight.torsoAngle),
    // La cabeza también quiere quedar vertical, con menos peso que el torso.
    uprightRow("Head angle", "Cosine of the head. +1 is upright.", cosineOf(part.head), weight.headAngle),
    // Cada brazo, antebrazo, muslo y pantorrilla suma su propio coseno. No es una pose de referencia.
    uprightRow("Left upper arm", "Cosine of the left upper arm. +1 is vertical.", cosineOf(part.upperArmLeft), weight.upperArmAngle),
    uprightRow("Right upper arm", "Cosine of the right upper arm. +1 is vertical.", cosineOf(part.upperArmRight), weight.upperArmAngle),
    uprightRow("Left forearm", "Cosine of the left forearm. +1 is vertical.", cosineOf(part.forearmLeft), weight.forearmAngle),
    uprightRow("Right forearm", "Cosine of the right forearm. +1 is vertical.", cosineOf(part.forearmRight), weight.forearmAngle),
    uprightRow("Left thigh", "Cosine of the left thigh. +1 is vertical, like standing.", cosineOf(part.thighLeft), weight.thighAngle),
    uprightRow("Right thigh", "Cosine of the right thigh. +1 is vertical, like standing.", cosineOf(part.thighRight), weight.thighAngle),
    uprightRow("Left shin", "Cosine of the left shin. +1 is vertical. A shin near the floor can still be vertical.", cosineOf(part.shinLeft), weight.shinAngle),
    uprightRow("Right shin", "Cosine of the right shin. +1 is vertical.", cosineOf(part.shinRight), weight.shinAngle),
    // La altura sube con el torso, también si una mano empuja el piso. La mano se cobra en su propia fila.
    heightRow("Head height", "Head height against a standing reach. Stays at the floor value until the torso turns up. A hand on the floor is scored on its own row.", measure.headHeight, config.body.headHeightMeters, rise, weight.headHeight),
    // Misma compuerta que la cabeza: sin crédito hasta que el torso gira hacia arriba.
    heightRow("Torso height", "Torso height. Same gate as the head: no credit until the torso turns up.", measure.torsoHeight, config.body.torsoHeightMeters, rise, weight.torsoHeight),
  ];

  // Mano cerca del piso resta poco. Si el torso ya está erguido y la mano sigue abajo, resta mucho más.
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

  // El muslo cerca del piso resta. La pantorrilla no entra en esta fila.
  const thighTip = "Subtracts while the thigh is within 35 cm of the floor. The shin is not in this row.";
  const thighClear = (index) => 1 - clamp((measure.clearance?.[index] ?? 1) / config.body.thighClearMeters, 0, 1);
  rows.push({ name: "Left thigh clearance", tip: thighTip, value: -weight.thighOnFloor * thighClear(part.thighLeft) });
  rows.push({ name: "Right thigh clearance", tip: thighTip, value: -weight.thighOnFloor * thighClear(part.thighRight) });
  // La cola resta aunque el torso siga tumbado. Sentarse en ella no es levantarse.
  const tailClose = 1 - clamp((measure.clearance?.[part.hip] ?? 1) / config.body.thighClearMeters, 0, 1);
  rows.push({
    name: "Tail",
    tip: "Subtracts while the tail is within 35 cm of the floor. Not gated by the torso, so sitting on it costs during a get-up.",
    value: -weight.tailOnFloor * tailClose,
  });
  const spread = Math.abs(Math.atan2(
    sineOf(part.thighLeft) * cosineOf(part.thighRight) - cosineOf(part.thighLeft) * sineOf(part.thighRight),
    cosineOf(part.thighLeft) * cosineOf(part.thighRight) + sineOf(part.thighLeft) * sineOf(part.thighRight),
  ));
  const spreadPenalty = spread < config.body.thighSpreadMinRadians
    ? (config.body.thighSpreadMinRadians - spread) / config.body.thighSpreadMinRadians
    : spread > config.body.thighSpreadMaxRadians
      ? (spread - config.body.thighSpreadMaxRadians) / (Math.PI - config.body.thighSpreadMaxRadians)
      : 0;
  // La apertura solo resta con el torso erguido. En el piso las piernas pueden moverse para levantarse.
  rows.push({
    name: "Thigh spread",
    tip: "Angle between the thighs. Subtracts below 10° and above 40° only while the torso is upright. On the floor the legs can move. A step in between adds nothing.",
    value: -weight.thighSpread * upright * clamp(spreadPenalty, 0, 1),
  });

  // Pie en el aire solo resta si el torso ya está erguido. Apoyado, esta fila vale cero.
  const footTip = "Subtracts while that foot is off the floor and the torso is upright. On the floor a lifted foot adds nothing. A planted foot adds nothing.";
  rows.push({ name: "Left foot", tip: footTip, value: upright * (measure.touch?.[part.footLeft] ? 0 : -weight.footInAir) });
  rows.push({ name: "Right foot", tip: footTip, value: upright * (measure.touch?.[part.footRight] ? 0 : -weight.footInAir) });

  const footMid = ((measure.footXLeft || 0) + (measure.footXRight || 0)) / 2;
  const lean = clamp(Math.abs((measure.centerX || 0) - footMid) / config.body.balanceSpanMeters, 0, 1);
  // El centro de masa sobre el punto medio de los pies suma. Colgado hacia un costado resta.
  rows.push({
    name: "Balance",
    tip: "Center of mass against the midpoint of the feet. Above the feet it adds. Hanging off to one side it subtracts.",
    value: weight.balance * (1 - 2 * lean),
  });

  const headDown = Math.max(0, -(measure.headVelocityY || 0));
  const headOnFloor = measure.touch?.[part.head] ? 1 : 0;
  const headStrike = headOnFloor && headDown > 1 ? weight.headImpact * clamp(headDown / 3, 0, 2) : 0;
  // Solo el golpe. Apoyar la cabeza no resta: eso valía más que pararse.
  rows.push({
    name: "Head impact",
    tip: "Subtracts while the head is touching the floor and moving down faster than 1 m/s, up to twice the weight. Resting the head on the floor does not subtract.",
    value: -headStrike,
  });

  const bothFeet = (measure.touch?.[part.footLeft] ? 1 : 0) + (measure.touch?.[part.footRight] ? 1 : 0);
  let otherTouch = 0;
  if (measure.touch) {
    for (let i = 0; i < measure.touch.length; i++) {
      if (i !== part.footLeft && i !== part.footRight && measure.touch[i]) otherTouch = 1;
    }
  }
  const onFeet = (bothFeet === 2 ? 1 : 0) * (1 - otherTouch) * upright * clamp((measure.headHeight || 0) / config.body.standingHeadMeters, 0, 1);
  // 1 when the sole ray hits at the face. A sole 6 cm up scores 0.
  const soleDown = (index) => clamp(1 - (measure.sole?.[index] || 0) / 0.06, 0, 1);
  const feetFlat = measure.sole ? (soleDown(0) + soleDown(1)) / 2 : 0;
  const bodyClear = 1 - otherTouch;
  const headUp = clamp((measure.headHeight || 0) / config.body.standingHeadMeters, 0, 1);
  // ponytail: linear falloff, not a narrow gaussian. The gaussian was ~0 past 0.5 m, so a 1.3 m stance had no rank signal. Upgrade path: a shaped kernel if 29 cm needs a flat top.
  const stanceOff = Math.abs((measure.stanceWidth || 0) - config.body.comfortableStanceMeters);
  const stanceComfort = 1 - clamp(stanceOff / 1.1, 0, 2);
  // Standing no es un producto. Los +5 se reparten: 40% plantas, 20% cuerpo libre, 20% torso, 20% cabeza.
  // El comfort de la base solo cuenta si las plantas están abajo. El flag de parado quieto se suma aparte.
  const soleShare = 0.4 * weight.standing * feetFlat;
  const clearShare = 0.2 * weight.standing * bodyClear;
  const uprightShare = 0.2 * weight.standing * upright;
  const headShare = 0.2 * weight.standing * headUp;
  // 0 recta. 1 - cos: 20° casi no resta, un ángulo recto paga el peso entero.
  const kneeFold = (shinIndex, thighIndex) => 1 - (cosineOf(shinIndex) * cosineOf(thighIndex) + sineOf(shinIndex) * sineOf(thighIndex));
  const kneeLeft = rise * kneeFold(part.shinLeft, part.thighLeft);
  const kneeRight = rise * kneeFold(part.shinRight, part.thighRight);
  const kneeCost = weight.kneeBend * (kneeLeft + kneeRight);
  const included = " Already included in Standing.";
  rows.push({
    name: "Stand soles",
    scoreOnly: true,
    tip: "How close each sole ray is to the floor. A sole in the air or on its side stays low. This is 40% of the standing weight." + included,
    value: soleShare,
  });
  rows.push({
    name: "Stand clear",
    scoreOnly: true,
    tip: "Full while nothing but the feet touches the floor. This is 20% of the standing weight." + included,
    value: clearShare,
  });
  rows.push({
    name: "Stand upright",
    scoreOnly: true,
    tip: "How upright the torso is. This is 20% of the standing weight." + included,
    value: uprightShare,
  });
  rows.push({
    name: "Stand head",
    scoreOnly: true,
    tip: "Head height against the standing height. This is 20% of the standing weight." + included,
    value: headShare,
  });
  const kneeTip = "Subtracts while that knee is bent and the torso is turning up. Straight is free. A small bend barely counts. A right angle spends the whole weight. On the floor, with the torso flat, bending to get up is free." + included;
  rows.push({ name: "Left knee", scoreOnly: true, tip: kneeTip, value: -weight.kneeBend * kneeLeft });
  rows.push({ name: "Right knee", scoreOnly: true, tip: kneeTip, value: -weight.kneeBend * kneeRight });
  rows.push({
    name: "Standing",
    tip: "Sum of the four stand rows, plus stance comfort while the soles are down, plus one more for a true quiet stand, minus a bent knee while the torso is turning up. Flat soles are 40%, a clear body 20%, an upright torso 20%, and a high head 20%. A stance near 29 cm adds more as the soles get flat; a very wide one subtracts.",
    value: soleShare + clearShare + uprightShare + headShare + weight.stanceComfort * feetFlat * stanceComfort + (isStanding(measure) ? weight.standingFlag : 0) - kneeCost,
  });

  const hasGoal = measure.goalDistanceMeters != null;
  const closeness = hasGoal ? 1 - clamp(Math.abs(measure.goalDistanceMeters) / config.body.goalSpanMeters, 0, 1) : 0;
  const toward = hasGoal
    ? clamp((measure.goalDistanceMeters > 0 ? measure.centerVelocityX || 0 : -(measure.centerVelocityX || 0)) / config.body.goalWalkSpeed, 0, 1)
    : 0;
  const goalOn = activeTrainingPhase().goal;
  const distanceUp = goalOn ? config.curriculum.phase3GoalDistanceUp : weight.goalDistanceUp;
  const distanceDown = goalOn ? config.curriculum.phase3GoalDistanceDown : weight.goalDistanceDown;
  const speedUp = goalOn ? config.curriculum.phase3GoalSpeedUp : weight.goalSpeedUp;
  const speedDown = goalOn ? config.curriculum.phase3GoalSpeedDown : weight.goalSpeedDown;
  // Cerca del marcador suma más, y más todavía si ya está sobre los pies, para que no convenga tirarse.
  rows.push({
    name: "Goal distance",
    tip: "Closer to the marker adds more. Worth more while standing than while on the floor, so diving at it does not pay.",
    value: hasGoal ? (onFeet ? distanceUp : distanceDown) * closeness : 0,
  });
  // Velocidad hacia el marcador. También vale más parado que en el piso.
  rows.push({
    name: "Goal speed",
    tip: "Adds while the center of mass moves toward the marker. Worth more while standing.",
    value: hasGoal ? (onFeet ? speedUp : speedDown) * toward : 0,
  });
  let motion = 0;
  for (let i = 0; i < config.net.partCount; i++) {
    motion += Math.abs(measure.partVelocityX?.[i] || 0) + Math.abs(measure.partVelocityY?.[i] || 0) + Math.abs(measure.partSpin?.[i] || 0);
  }
  // En el piso moverse es gratis. Erguido, una sacudida fuerte resta el peso entero.
  rows.push({
    name: "Motion",
    tip: "Subtracts while the torso is upright and any part is moving or spinning. On the floor, moving to get up is free. A quiet stand stays near zero. A violent shake pays the whole weight.",
    value: -weight.motion * upright * clamp(motion / config.scale.motionFull, 0, 1),
  });
  const inputRows = rows.filter((row) => !row.scoreOnly);
  if (inputRows.length !== config.net.rewardRowCount) throw new Error("reward rows " + inputRows.length);

  let total = 0;
  let ceiling = 0;
  for (const row of rows) {
    if (row.scoreOnly) continue;
    total += row.value;
    if (row.name === "Goal distance" && !hasGoal) continue;
    if (row.name === "Goal speed" && !hasGoal) continue;
    if (row.name === "Goal distance") ceiling += goalOn ? config.curriculum.phase3GoalDistanceUp : weight.goalDistanceUp;
    else if (row.name === "Goal speed") ceiling += goalOn ? config.curriculum.phase3GoalSpeedUp : weight.goalSpeedUp;
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
  return {
    total,
    max: ceiling,
    onFeet,
    terms: rows.map((row) => ({ n: row.name, tip: row.tip, v: row.value, scoreOnly: !!row.scoreOnly })),
  };
}
