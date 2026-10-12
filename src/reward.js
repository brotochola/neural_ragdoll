// El premio de un frame es la distancia y la rotación de los targets prendidos. Uno apagado no suma.
import { config } from "../config.js";
import { clamp } from "./utils.js";

const part = config.partIndex;
const weight = config.reward;
const standingCos = Math.cos(config.body.standingTorsoAngleRadians);
const termCount = config.net.rewardRowCount;
const terms = [];
for (let i = 0; i < config.targets.length; i++) {
  const id = config.targets[i].id;
  const foot = id === "footR" || id === "footL";
  const positionWeight = foot ? weight.footPosition : weight.targetPosition;
  const span = foot ? config.skill.footSpanMeters : config.body.targetSpanMeters;
  terms.push({
    n: id + " position",
    tip: "Distance from the " + id + " to its target. +" + positionWeight + " on the mark, -" + positionWeight + " at " + span + " m. Zero while this target is off.",
    v: 0,
    scoreOnly: false,
  });
  terms.push({
    n: id + " rotation",
    tip: "How well the " + id + " faces its target. +" + weight.targetRotation + " when aligned, -" + weight.targetRotation + " when flipped. Zero while this target is off.",
    v: 0,
    scoreOnly: false,
  });
}
const scored = { total: 0, max: 0, onFeet: 0, terms };

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

function angleErrorCos(targetAngle, partSin, partCos) {
  const targetSin = Math.sin(targetAngle || 0);
  const targetCos = Math.cos(targetAngle || 0);
  return targetCos * partCos + targetSin * partSin;
}

function footTarget(id) {
  return id === "footR" || id === "footL";
}

// Distancia al mark y coseno del error de ángulo. align 1 es alineado.
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
  const error = markError(measure, targetIndex);
  return error.dist <= config.skill.footReachMeters && error.align >= Math.cos(config.skill.footReachRadians);
}

// Escribe el total. Si writeRows, también las filas fijas del panel. No crea objetos.
function fillScore(measure, writeRows) {
  const targets = measure.targets && measure.targets.length === config.targets.length ? measure.targets : config.targets;
  let total = 0;
  let ceiling = 0;
  let row = 0;
  for (let i = 0; i < config.targets.length; i++) {
    const spec = config.targets[i];
    const target = targets[i] || spec;
    const index = part[spec.part];
    const on = !!(target && target.on);
    const foot = footTarget(spec.id);
    let position = 0;
    let rotation = 0;
    if (on) {
      const dx = (target.x || 0) - ((measure.partX && measure.partX[index]) || 0);
      const dy = (target.y || 0) - ((measure.partY && measure.partY[index]) || 0);
      const span = foot ? config.skill.footSpanMeters : config.body.targetSpanMeters;
      const positionWeight = foot ? weight.footPosition : weight.targetPosition;
      const closeness = 1 - 2 * clamp(Math.hypot(dx, dy) / span, 0, 1);
      position = positionWeight * closeness;
      const partCos = measure.partCos && measure.partCos[index];
      rotation = weight.targetRotation * angleErrorCos(target.angle, (measure.partSin && measure.partSin[index]) || 0, partCos == null ? 1 : partCos);
      ceiling += positionWeight + weight.targetRotation;
    }
    if (writeRows) {
      terms[row].v = position;
      terms[row + 1].v = rotation;
    }
    row += 2;
    total += position + rotation;
  }
  if (row !== termCount) throw new Error("reward rows " + row);
  scored.total = total;
  scored.max = ceiling;
  return total;
}

// Número solo. Lo usa el episodio, en cada muestra de reward.
export function scoreTotal(measure) {
  return fillScore(measure, false);
}

// Las mismas filas de siempre, reescritas. El panel las lee en la misma vuelta.
export function scoreDecision(measure) {
  fillScore(measure, true);
  return scored;
}
