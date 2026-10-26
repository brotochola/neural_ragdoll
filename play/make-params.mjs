// Collects the controllers' tuned numbers into play/params.json for the playground.
//   node play/make-params.mjs
import fs from 'node:fs';
const simb = await import('../src/simbicon.js');
const rd = (f) => JSON.parse(fs.readFileSync(new URL('../' + f, import.meta.url), 'utf8'));
const walkSaved = rd('trained/walk-rough-controller.json');
const walk = Object.assign(simb.decode(walkSaved.best), { track: !!walkSaved.track });
const runner = rd('trained/runner-gap-12c.json').params;
// planner numbers: decoded the way climb-wall.js and mountain.js do (their SPEC defaults plus offsets)
const specOf = (file) => {
  const src = fs.readFileSync(new URL('../' + file, import.meta.url), 'utf8');
  const block = src.slice(src.indexOf('const SPEC = ['), src.indexOf('];', src.indexOf('const SPEC = [')) + 2);
  return eval(block.replace('const SPEC = ', ''));
};
const decodeWith = (spec, v = [], fix = {}) => { const p = {}; spec.forEach(([k, b], i) => (p[k] = b + (v[i] || 0))); return Object.assign(p, fix); };
const wallSpec = specOf('climb-wall.js');
const wall = decodeWith(wallSpec, [], { pelvisOffTop: 0.56 });
for (const k of ['tHand', 'tFoot', 'tRise']) wall[k] = Math.max(0.2, wall[k]);
const mtSpec = specOf('mountain.js');
let mtBest = [];
try { mtBest = rd(process.env.MOUNTAIN || 'trained/mountain-a.json').best; } catch (e) {}
const mountain = decodeWith(mtSpec, mtBest);
const cragSpec = specOf('crag.js');
let cragBest = [];
try { cragBest = rd(process.env.CRAG || 'trained/crag-c.json').best; } catch (e) {}
const crag = decodeWith(cragSpec, cragBest);
for (const k of ['tHand', 'tFoot', 'tRise']) crag[k] = Math.max(0.2, crag[k]);
const states = rd('trained/run-states.json');
const runStates = states.filter((_, i) => i % 6 === 0).slice(0, 36);
fs.writeFileSync(new URL('./params.json', import.meta.url), JSON.stringify({ walk, runner, wall, mountain, crag, runStates }));
console.log('params: walk', Object.keys(walk).length, 'runner', Object.keys(runner).length, 'wall', Object.keys(wall).length, 'mountain', Object.keys(mountain).length, 'crag', Object.keys(crag).length, 'run states', runStates.length);
