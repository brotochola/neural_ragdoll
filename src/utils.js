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
