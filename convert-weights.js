// Converts saved weights from an older network layout to the current one, with zeros for every new
// input column, walk-skip weight and output row. The converted net behaves exactly like the old one
// while the new inputs are 0 and the new outputs are unused.
//   node convert-weights.js in.json out.json
// The old layout is detected by trying the input counts and widths this project has used.
import fs from 'node:fs';
import { config } from './config.js';
import { ElmanNetwork } from './src/net.js';

const [input, output] = process.argv.slice(2);
if (!input || !output) {
  console.error('usage: node convert-weights.js in.json out.json');
  process.exit(1);
}
const saved = JSON.parse(fs.readFileSync(input, 'utf8'));
const theta = saved.theta;
const sensed = config.net.walkInputIndex;
if (theta.length === ElmanNetwork.weightCount) {
  fs.writeFileSync(output, JSON.stringify(Object.assign({}, saved, { inputCount: config.net.inputCount })));
  console.log('already current:', theta.length, 'weights');
  process.exit(0);
}
// Walk inputs this project has had: 0 (none), 3 (clock + command), 6 (+ terrain), 8 (+ grip flags).
// Output rows: 14 motors, or 14 motors + 2 grips.
for (const walkInputs of [0, 3, 6, 8]) {
  for (const walkSkip of [0, 3, 6, 8]) {
    if (walkSkip > walkInputs) continue;
    for (const outputs of [config.net.jointCount, config.net.jointCount + 2]) {
      const converted = ElmanNetwork.convert(theta, sensed + walkInputs, walkSkip, outputs);
      if (!converted) continue;
      fs.writeFileSync(output, JSON.stringify(Object.assign({}, saved, {
        theta: Array.from(converted),
        bestEverTheta: undefined,
        inputCount: config.net.inputCount,
        convertedFrom: { weights: theta.length, walkInputs, walkSkip, outputs },
      })));
      console.log('converted', theta.length, '->', converted.length, 'from walk inputs', walkInputs, 'walk skip', walkSkip, 'outputs', outputs);
      process.exit(0);
    }
  }
}
console.error('no known layout matches', theta.length, 'weights');
process.exit(1);
