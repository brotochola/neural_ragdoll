// Builds the replay page from one or more recordings made by record-episodes.js.
//   node replay/build.js out.html "Stand|Standing ...|replay-stand.json" "Walk|Walking ...|replay-walk.json"
// Each argument after the output is label|one-line description|recording file.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const [out, ...specs] = process.argv.slice(2);
let shapes = null;
let fps = 30;
const sets = specs.map((spec) => {
  const [label, lede, file] = spec.split('|');
  const rec = JSON.parse(fs.readFileSync(file, 'utf8'));
  shapes = rec.shapes;
  fps = rec.fps;
  return { label, lede, fps: rec.fps, episodes: rec.episodes };
});
const json = JSON.stringify({ fps, shapes, sets }).replace(/</g, '\\u003c');
const html = fs.readFileSync(path.join(here, 'template.html'), 'utf8').replace('/*DATA*/', () => json);
fs.writeFileSync(out, html);
console.log('wrote', out, (html.length / 1024).toFixed(0) + ' KB');
