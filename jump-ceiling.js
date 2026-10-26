// node jump-ceiling.js: how high the body can jump at all (feet off the ground, torso up), searched with the
// cross-entropy method over a crouch-extend-tuck controller. 0.22 m with the default joints; strength 1.5-2
// changes nothing; motorGain 40 (stiffer servo) 0.30 m. A ceiling for box jumping, not a skill.
// Capability test: best standing tuck jump the body can do (feet as high as possible with the torso up),
// searched by CEM over a crouch-extend-tuck controller. Not a skill; a ceiling for box jumping.
import fs from 'node:fs'; import { pathToFileURL } from 'node:url';
const root = '/home/claude/brotochola/neural_ragdoll';
const { config, setSkill } = await import(pathToFileURL(root + '/config.js').href);
const Box2D = await import(pathToFileURL(root + '/vendor/PhaserBox2D.min.js').href);
const { createSim } = await import(pathToFileURL(root + '/src/sim.js').href);
const { ElmanNetwork } = await import(pathToFileURL(root + '/src/net.js').href);
const { makeSeededRandom, randomNormalSample } = await import(pathToFileURL(root + '/src/utils.js').href);
const { evalMode } = await import(pathToFileURL(root + '/src/training.js').href);
setSkill(6);
const sim = createSim(Box2D); const world = sim.build();
const w = ElmanNetwork.copy(JSON.parse(fs.readFileSync(root + '/trained/stand-and-walk.json', 'utf8')).theta);
const J = config.net.jointCount, HZ = config.physics.hz;
const angle = (i) => Math.atan2(world.s[i], world.c[i]);
const cmd = (j, t) => { const jt = world.joints[j]; const x = Math.max(jt.lo, Math.min(jt.hi, t)); return x >= 0 ? x / jt.hi : x / -jt.lo; };
// params: crouch knee, crouch time, hip push, ankle push, tuck hip, tuck knee, arm swing, lean
const base = [-1.4, 0.45, -0.2, 0.5, 1.6, -2.2, 2.5, 0.0];
function run(v) {
  const s = sim.fillTrainingStart({}, makeSeededRandom(1), evalMode(6)); s.tilt = 0; s.frames = Math.round(2.2 * HZ);
  world.rebuild();
  const ep = new sim.Episode(world, w, 0, 0, Object.assign({ phase: 0 }, s));
  let phase = 0, t0 = 0, best = 0;
  ep.applyMotors = ((orig) => function () {
    this.speedCap = 20;
    const t = this.frame / HZ, torso = angle(0);
    for (let j = 0; j < this.action.length; j++) this.action[j] = j < J ? 0 : -1;
    const legs = [[5, 6, 7, 6, 7], [11, 12, 13, 12, 13]];
    if (t < 0.5) { orig(); return; }
    const air = !this.measure.touch[config.partIndex.footLeft] && !this.measure.touch[config.partIndex.footRight];
    if (phase === 0 && t > 0.5 + v[1]) phase = 1;
    if (phase === 1 && air && t > 0.6 + v[1]) phase = 2;
    for (const [h, k, a, thigh, shin] of legs) {
      if (phase === 0) { this.action[k] = cmd(k, v[0]); this.action[h] = cmd(h, angle(thigh + 0) - torso + (torso - v[7])); this.action[h] = cmd(h, -v[0] * 0.6 + v[7]); this.action[a] = cmd(a, 0 - angle(shin)); }
      else if (phase === 1) { this.action[k] = cmd(k, 0); this.action[h] = cmd(h, v[2]); this.action[a] = cmd(a, v[3]); }
      else { this.action[k] = cmd(k, v[5]); this.action[h] = cmd(h, v[4]); this.action[a] = cmd(a, 0); }
    }
    for (const sh of [2, 8]) this.action[sh] = cmd(sh, phase >= 1 ? v[6] : -0.5);
    orig();
  })(ep.applyMotors.bind(ep));
  while (!ep.advanceOneFrame()) {
    if (Math.cos(angle(0)) > 0.8) best = Math.max(best, Math.min(world.py[config.partIndex.footLeft], world.py[config.partIndex.footRight]) - 0.05);
  }
  return best;
}
const rng = makeSeededRandom(5); const N = base.length;
let mean = base.slice(), sig = [0.4, 0.2, 0.3, 0.3, 0.4, 0.4, 0.6, 0.2], bestV = null, bestS = -1;
for (let it = 0; it < 25; it++) {
  const c = [];
  for (let k = 0; k < 24; k++) { const v = mean.map((m, i) => m + sig[i] * randomNormalSample(rng)); v[1] = Math.max(0.15, v[1]); c.push({ v, s: run(v) }); }
  c.sort((a, b) => b.s - a.s);
  if (c[0].s > bestS) { bestS = c[0].s; bestV = c[0].v; }
  const top = c.slice(0, 6);
  for (let i = 0; i < N; i++) { const m = top.reduce((a, x) => a + x.v[i], 0) / 6; const sd = Math.sqrt(top.reduce((a, x) => a + (x.v[i] - m) ** 2, 0) / 6); mean[i] = 0.7 * m + 0.3 * mean[i]; sig[i] = Math.max(0.03, 0.7 * sd + 0.3 * sig[i]); }
  if (it % 5 === 4) console.log('iter', it, 'best foot height', bestS.toFixed(2));
}
console.log('best params', bestV.map((x) => x.toFixed(2)).join(' '));
