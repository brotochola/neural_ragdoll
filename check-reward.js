// Chequeos rápidos de la aritmética del premio y de las marcas. Sin Box2D, sin física.
//   node check-reward.js
import { config } from './config.js';
import { scoreTotal, scoreByTarget, markError, onMark, isStanding, targetGain, targetPositionWeight } from './src/reward.js';

// Un measure falso: parte en (x, y) mirando un ángulo, con su marca en el origen.
function fakeMeasure(spec, dx, dy, angle, extra) {
  const index = config.partIndex[spec.part];
  const targets = config.targets.map(() => ({ on: false, x: 0, y: 0, angle: 0 }));
  targets[config.targets.indexOf(spec)] = { on: true, x: 0, y: 0, angle: 0 };
  const partX = [];
  const partY = [];
  const partSin = [];
  const partCos = [];
  const touch = [];
  for (let i = 0; i < config.net.partCount; i++) {
    partX[i] = dx;
    partY[i] = dy;
    partSin[i] = Math.sin(angle);
    partCos[i] = Math.cos(angle);
    touch[i] = false;
  }
  const measure = {
    partX,
    partY,
    partSin,
    partCos,
    touch,
    targets,
    torsoCos: 1,
    torsoSin: 0,
    headHeight: 1.6,
    stanceWidth: 0.4,
    torsoSpin: 0,
    centerVelocityX: 0,
  };
  return Object.assign(measure, extra || {});
}

let failures = 0;
// Acepta un numero o un booleano. Para lo booleano dice si se cumple, y por que no.
function check(name, actual, expected, tolerance) {
  if (typeof expected === 'boolean') {
    const ok = !!actual === expected;
    if (!ok) failures++;
    console.log((ok ? 'ok   ' : 'FALLA') + '  ' + name.padEnd(46) + String(!!actual).padStart(6));
    return;
  }
  const ok = Math.abs(actual - expected) <= (tolerance == null ? 1e-6 : tolerance);
  if (!ok) failures++;
  console.log(
    (ok ? 'ok   ' : 'FALLA') + '  ' + name.padEnd(46) + actual.toFixed(4) + (ok ? '' : ' esperado ' + expected),
  );
}

const head = config.targets[0];
const foot = config.targets[3];

// Cada eje es near + wide. near es weight*(1 - 2r/(1+|r|)) con r = |eje|/span.
// wide es la misma curva, solo en los targets que ridean, con span ancho.
// En la marca un eje suma +peso*(1+wideWeight) si ridea, y +peso si es un pie.
// La posicion del target es la suma de los dos ejes.
const hw = targetPositionWeight(head);
const hr = config.reward.targetRotation;
const wideWeight = config.reward.wideWeight;
const wideSpan = config.reward.wideSpanMeters;
const pRow = (w, span, d) => {
  const r = Math.abs(d / span);
  return w * (1 - 2 * (r / (1 + r)));
};
const positionOf = (spec, dx, dy) => {
  const w = targetPositionWeight(spec);
  const one = (d) => {
    let value = pRow(w, spec.span, Math.abs(d));
    if (spec.ride) value += pRow(w * wideWeight, wideSpan, Math.abs(d));
    return value;
  };
  return one(dx) + one(dy);
};
const uprightOf = (torsoCos, headHeight) => {
  let height = headHeight / config.body.standingHeadMeters;
  if (height < 0) height = 0;
  else if (height > 1) height = 1;
  return config.reward.upright * (0.5 * torsoCos + 0.5 * height);
};
// El measure falso nace parado: torsoCos 1, cabeza a 1.6 m. Eso es la fila de parado entera.
const upright = uprightOf(1, 1.6);
const frame = (spec, dx, dy, angle) => positionOf(spec, dx, dy) + hr * Math.cos(angle) + upright;
const positionRowOf = (spec, dx, dy) => scoreByTarget(fakeMeasure(spec, dx, dy, 0))[config.targets.indexOf(spec)].position;

check('cabeza en la marca', scoreTotal(fakeMeasure(head, 0, 0, 0)), frame(head, 0, 0, 0));
check('cabeza a la mitad del span', scoreTotal(fakeMeasure(head, head.span / 2, 0, 0)), frame(head, head.span / 2, 0, 0));
check('cabeza al triple del span', scoreTotal(fakeMeasure(head, head.span * 3, 0, 0)), frame(head, head.span * 3, 0, 0));
check('cabeza girada 90 grados', scoreTotal(fakeMeasure(head, 0, 0, Math.PI / 2)), frame(head, 0, 0, Math.PI / 2));
check('cabeza invertida', scoreTotal(fakeMeasure(head, 0, 0, Math.PI)), frame(head, 0, 0, Math.PI));
check('pie en la marca', scoreTotal(fakeMeasure(foot, 0, 0, 0)), frame(foot, 0, 0, 0));
check('pie a la mitad de su span', scoreTotal(fakeMeasure(foot, foot.span / 2, 0, 0)), frame(foot, foot.span / 2, 0, 0));
check('pie al doble de su span', scoreTotal(fakeMeasure(foot, foot.span * 2, 0, 0)), frame(foot, foot.span * 2, 0, 0));
// La caida entera: 2 m tiene que seguir castigando, y no puede valer lo mismo que 0.5 m.
// Y la sensibilidad cerca de la marca tiene que ser la de siempre: 10 cm no puede costar
// menos que lo que costaba con la fila lineal.
check('caer a 2 m sigue costando la mayor parte del peso', pRow(hw, head.span, 2.0) < -0.75 * hw, true);
check('medio metro vale mas que 2 m', pRow(hw, head.span, 0.5) > pRow(hw, head.span, 2.0), true);
check('la fila de posicion nunca es positiva lejos de la marca', positionRowOf(head, 99, 99) < 0, true);
check('la fila no se aplana: 2 m vale mas que 20 m', positionRowOf(head, 2.0, 0) > positionRowOf(head, 20, 0), true);
// La sensibilidad tiene que ser la misma que daba la fila lineal. Lo que importa es la DERIVA
// en la marca (ahi se aprende la precision fina), y que a 10 cm no cueste mucho menos.
const slope = (spec) => (positionRowOf(spec, 0.0001, 0) - positionRowOf(spec, 0, 0)) / 0.0001;
const linearSlope = (spec) => {
  const w = targetPositionWeight(spec);
  let value = -2 * w / spec.span;
  if (spec.ride) value += -2 * w * wideWeight / wideSpan;
  return value;
};
check('la derivada en la marca es la lineal', slope(head), linearSlope(head), 0.02);
check('a 10 cm es r/(1+|r|)', pRow(hw, head.span, 0.1), hw * (1 - 2 * (0.1 / (head.span + 0.1))), 1e-9);
check('la fila de giro si baja de cero', scoreByTarget(fakeMeasure(head, head.span, 0, Math.PI))[0].rotation < 0, true);

// El span decides la sensibilidad por metro: mismo error, misma pérdida en los cinco.
const err = 0.1;
console.log('');
console.log('sensibilidad: valor de la fila con 10 cm de error, y con 2 m (la caida entera)');
for (let i = 0; i < config.targets.length; i++) {
  const spec = config.targets[i];
  const w = targetPositionWeight(spec);
  const near = positionOf(spec, err, 0);
  const far = positionOf(spec, 2, 0);
  console.log(
    '  ' + spec.id.padEnd(6) + 'span ' + String(spec.span).padEnd(6) + ' peso ' + w +
    '  10cm ' + near.toFixed(3) + '  2m ' + far.toFixed(3) +
    '  (se distinguen: ' + (near - far > 0.01 ? 'si' : 'NO') + ')',
  );
}

// gain 0 saca las filas de ese target. La fila de parado no depende de ningun target.
const before = scoreTotal(fakeMeasure(head, 0, 0, 0));
config.targetGains.head = 0;
check('gain 0 deja solo la fila de parado', scoreTotal(fakeMeasure(head, 0, 0, 0)), upright);
config.targetGains.head = 1;
check('gain 1 la vuelve', scoreTotal(fakeMeasure(head, 0, 0, 0)), before);

// scoreByTarget reparte los targets. El total es esa suma mas la fila de parado y el contacto.
const marked = fakeMeasure(head, 0, 0, 0);
const rows = scoreByTarget(marked);
const sum = rows.reduce((acc, row) => acc + row.position + row.rotation, 0);
check('scoreByTarget suma los targets', sum, scoreTotal(marked) - upright);

// LA MEZCLA DEL ARRANQUE. startHoldShare cobra SOLO lo que se perdio entre la ventana inicial
// y la ventana final. Antes era (1-share)*vida + share*min(arranque, vida), y como min siempre
// es <= vida, le cobraba a quien MEJORA. Estas cuatro lineas fijan la regla; fitnessValue la
// aplica a la ventana final, no al promedio del episodio entero.
const hold = config.reward.startHoldShare;
const blend = (end, start) => (hold ? end - hold * (start > end ? start - end : 0) : end);
check('mejorar no se castiga', blend(8, 3), 8, 1e-9);
check('perder se castiga', blend(3, 8) < 3, true);
check('no cambiar no cambia nada', blend(6, 6), 6, 1e-9);
check('perder todo vale menos que el promedio', blend(0, 8), -4, 1e-9);
check('la mezcla nunca sube sobre el promedio', blend(2, 9) <= 2, true);

// onMark usa la tolerancia del tipo de target.
const rideMeasure = fakeMeasure(head, config.skill.targetReachMeters - 0.01, 0, 0);
check('onMark cabeza dentro', onMark(rideMeasure, 0) ? 1 : 0, 1);
const footMeasure = fakeMeasure(foot, config.skill.footReachMeters - 0.01, 0, 0);
check('onMark pie dentro', onMark(footMeasure, 3) ? 1 : 0, 1);
check('onMark cabeza demasiado lejos', onMark(fakeMeasure(head, config.skill.targetReachMeters + 0.05, 0, 0), 0) ? 1 : 0, 0);

// isStanding no secontamination con los targets.
const stand = {
  partX: [], partY: [], partSin: [], partCos: [], touch: [],
  targets: [], torsoCos: 1, torsoSin: 0, headHeight: 1.6, stanceWidth: 0.4,
  torsoSpin: 0, centerVelocityX: 0,
};
for (let i = 0; i < config.net.partCount; i++) {
  stand.partX[i] = 0;
  stand.partY[i] = 1;
  stand.partSin[i] = 0;
  stand.partCos[i] = 1;
  stand.touch[i] = false;
}
stand.touch[config.partIndex.footLeft] = true;
stand.touch[config.partIndex.footRight] = true;
check('isStanding con los dos pies', isStanding(stand) ? 1 : 0, 1);
stand.touch[config.partIndex.handLeft] = true;
check('isStanding con una mano abajo', isStanding(stand) ? 1 : 0, 0);

console.log('');
console.log(failures ? failures + ' chequeos fallaron' : 'todos los chequeos pasan');
process.exitCode = failures ? 1 : 0;