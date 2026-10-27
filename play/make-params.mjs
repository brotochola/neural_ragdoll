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
for (const k of ['tHand', 'tFoot', 'tRise', 'tPress', 'tOver', 'tPull']) crag[k] = Math.max(0.2, crag[k]);
crag.belly = 1;
// the slow walker, the stop controller and the get-up expert (the obstacle chain), the rope swing
const fastSaved = rd('trained/walk-fast-controller.json');
const walkFast = Object.assign(simb.decode(fastSaved.best), { track: !!fastSaved.track });
const slowSaved = rd('trained/walk-controller.json');
const slow = Object.assign(simb.decode(slowSaved.best), { track: !!slowSaved.track });
const stopK = rd('trained/stop-controller.json').params;
const stop = Object.assign({}, slow, stopK, { track: true });
const g = rd('trained/getup-expert-19.json');
const getup = { theta: g.theta.map((x) => +x.toFixed(5)), handover: g.handover };
// the obstacle chain uses the numbers tuned on 2 m rocks from varied start distances
let obV = [];
try { obV = rd(process.env.CRAG_OBSTACLE || 'trained/crag-2m-wide.json').best; } catch (e) {}
const cragObstacle = Object.assign(decodeWith(cragSpec, obV), { belly: 1, doneX: 3.0 });
for (const k of ['tHand', 'tFoot', 'tRise', 'tPress', 'tOver', 'tPull']) cragObstacle[k] = Math.max(0.2, cragObstacle[k]);
const ropeSpec = (await import('../src/rope-swing.js')).SPEC;
let ropeBest = [];
try { ropeBest = rd(process.env.ROPE || 'trained/rope-a.json').best; } catch (e) {}
const rope = decodeWith(ropeSpec, ropeBest);
const states = rd('trained/run-states.json');
const runStates = states.filter((_, i) => i % 6 === 0).slice(0, 36);
// the dash runner (random boxes and balls, dash.js): DASH env, its decoded params
let dash = null;
try { dash = rd(process.env.DASH || 'trained/dash.json').params; } catch (e) { dash = runner; }
fs.writeFileSync(new URL('./params.json', import.meta.url), JSON.stringify({ walk, runner, wall, mountain, crag, runStates, slow, stop, stopK, getup, cragObstacle, rope, walkFast, dash }));
console.log('params: walk', Object.keys(walk).length, 'runner', Object.keys(runner).length, 'wall', Object.keys(wall).length, 'mountain', Object.keys(mountain).length, 'crag', Object.keys(crag).length, 'run states', runStates.length);
