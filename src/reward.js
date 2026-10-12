// El premio de un frame es la distancia y la rotación de los targets prendidos. Uno apagado no suma.
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
const scored = { total: 0, max: 0, onFeet: 0, terms };

// Cada target trae su span, su peso y su gain. El gain 0 saca la fila del premio y sus
// cinco columnas de la entrada. Nada de pie contra cuerpo: los numeros viven en config.targets.
export function targetGain(id, target) {
  // El episodio puede multiplicar el gain (gainCmd, el nivel que manda esa parte).
  if (target && typeof target.gain === "number") return target.gain > 0 ? target.gain : 0;
  const gains = config.targetGains;
  const g = gains ? gains[id] : 1;
  return typeof g === "number" && g > 0 ? g : 0;
}

// El span puede venir del episodio (target.span): cambia si a esa parte la estan mandando.
// Si no viene, el del spec.
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

// La fila de posicion: +weight en la marca, -weight en el infinito, y NUNCA una meseta.
//
// Antes era lineal (1 - 2r) y se APLANABA en -weight a partir de un span. Caerse son ~1.5 m,
// asi que toda la caida daba el mismo numero: no habia forma de volver a subir, era un pozo.
// El intento con 1/(1+r^2) arreglarlo pero rompio otra cosa: el fondo era 0, no -weight, asi
// que tirado el doll sacaba ~0 en vez de ~-peso. Con los pies valiendo 6 de 12 puntos, eso
// hacia que estar TIRADO fuera casi igual de rentable (medido: 10.92 parado contra 8.00
// tumbado, cuando la version lineal daba 7.49 contra -1.48). El termine se habia comido el
// premio de pararse.
//
// La forma de ahora es r / sqrt(1+r^2) con r = d/span:
//   - deriva -2*weight/span en la marca, IGUAL que la lineal. La sensibilidad de 10 cm,
//     que es donde la red aprende, no se toca.
//   - tiende a -weight muy lejos: caerse sigue costingo el premio entero.
//   - d/dr de r/sqrt(1+r^2) es 1/(1+r^2)^1.5 > 0: baja siempre, nunca sube ni se aplana.
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
  if (row !== termCount) throw new Error("reward rows " + row);
  scored.total = total;
  scored.max = ceiling;
  return total;
}

// El mismo total partido por target. Lo usa el probe y el panel de diagnóstico, no el premio.
const perTarget = config.targets.map((spec) => ({ id: spec.id, position: 0, rotation: 0 }));

export function scoreByTarget(measure) {
  fillScore(measure, false);
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

// Número solo. Lo usa el episodio, en cada muestra de reward.
export function scoreTotal(measure) {
  return fillScore(measure, false);
}

// Las mismas filas de siempre, reescritas. El panel las lee en la misma vuelta.
export function scoreDecision(measure) {
  fillScore(measure, true);
  return scored;
}
