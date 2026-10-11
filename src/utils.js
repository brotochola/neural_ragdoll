// Funciones puras. No leen la simulación ni la red.

// Recorta un número al intervalo [low, high].
export function clamp(value, low, high) {
  return Math.max(low, Math.min(high, value));
}

// Generador uniforme en [0, 1) a partir de una semilla entera. Misma semilla, misma secuencia.
export function makeSeededRandom(seed) {
  let state = seed | 0;
  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let mixed = Math.imul(state ^ (state >>> 15), 1 | state);
    mixed = (mixed + Math.imul(mixed ^ (mixed >>> 7), 61 | mixed)) ^ mixed;
    return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296;
  };
}

// Una muestra normal estándar. `uniform` es una función que devuelve [0, 1).
export function randomNormalSample(uniform) {
  return Math.sqrt(-2 * Math.log(uniform() + 1e-12)) * Math.cos(6.2831853 * uniform());
}

// Atajo: una función que pide una normal nueva cada vez, usando el mismo uniforme.
export function randomNormalFrom(uniform) {
  return () => randomNormalSample(uniform);
}

// Copia los pesos redondeados a 4 decimales. Solo para el JSON de un archivo. No es el camino de cada generación.
export function quantizeWeights(weights) {
  if (!weights) return null;
  const out = new Array(weights.length);
  for (let i = 0; i < weights.length; i++) out[i] = +(+weights[i]).toFixed(4);
  return out;
}

// Gira (x, y) con el coseno y el seno de la parte. Escribe en `out`, que ya existe.
export function rotate(x, y, cos, sin, out) {
  const turnedX = cos * x - sin * y;
  const turnedY = sin * x + cos * y;
  out.x = turnedX;
  out.y = turnedY;
  return out;
}
