// Lee el estado del entrenamiento y lo resume en una pantalla. No toca nada.
//   node supervise.js            una vez
//   node supervise.js --watch 60 cada 60 segundos,Ctrl-C para salir
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { config } from './config.js';

const root = path.dirname(fileURLToPath(import.meta.url));
const statusPath = path.join(root, 'backups', 'status.json');
const logPath = path.join(root, 'backups', 'train-log.txt');

function readStatus() {
  if (!fs.existsSync(statusPath)) return null;
  try {
    return JSON.parse(fs.readFileSync(statusPath, 'utf8'));
  } catch {
    return null;
  }
}

function tailLog(lines) {
  if (!fs.existsSync(logPath)) return [];
  const all = fs.readFileSync(logPath, 'utf8').trimEnd().split('\n');
  return all.slice(-lines);
}

function bar(ratio, width) {
  const n = Math.max(0, Math.min(width, Math.round(ratio * width)));
  return '[' + '#'.repeat(n) + '.'.repeat(width - n) + ']';
}

function pct(v) {
  return v == null ? '-' : (v * 100).toFixed(0) + '%';
}

function show() {
  const s = readStatus();
  if (!s) {
    console.log('todavia no hay status.json');
    return;
  }
  const latest = s.latest || {};
  console.log('');
  console.log('generacion ' + s.gen + '   skill ' + (latest.skill || s.phase) + '   mejor ' + (s.bestEverScore == null ? '-' : s.bestEverScore.toFixed(3)));
  if (latest.key) {
    const gate = config.eval.passRate[latest.key];
    const ratio = (latest.rate || 0) / Math.max(0.01, gate || 1);
    console.log('  ' + latest.key.padEnd(6) + ' ' + (latest.ok || 0) + '/' + latest.n +
      ' = ' + (latest.rate || 0).toFixed(2) + '  barra ' + (gate || 0).toFixed(2));
    console.log('    ' + bar(ratio, 30) + '  ' + (ratio * 100).toFixed(0) + '% de la barra');
    console.log('    parado ' + (latest.stood || 0).toFixed(2) + '   pie ' + (latest.foot == null ? '-' : latest.foot.toFixed(3)) +
      '   pasos ' + (latest.steps == null ? '-' : latest.steps.toFixed(2)) +
      '   caja ' + (latest.cleared == null ? '-' : latest.cleared.toFixed(2)));
    if (latest.obey != null)
      console.log('    obedece  cabeza ' + pct(latest.headObey) + '  manos ' + pct(latest.handObey) + '  pies ' + pct(latest.footObey));
  } else {
    console.log('  todavia sin eval de skill');
  }
  const targets = [
    ['cabeza', latest.headMark],
    ['manos', latest.handMark],
    ['cuerpo', latest.body],
  ];
  console.log('    targets  ' + targets.map(([name, v]) => name + ' ' + (v == null ? '-' : v.toFixed(3))).join('   '));
  if (s.evalLog && s.evalLog.length) {
    const recent = s.evalLog.slice(-6);
    console.log('    ultimos evals');
    for (const row of recent) {
      const ev = row.eval || {};
      console.log('      gen ' + String(row.gen).padStart(6) + '  ' + String(ev.key || row.skill || '?').padEnd(6) +
        '  ' + (ev.rate == null ? '-' : ev.rate.toFixed(2)) +
        '  parado ' + (ev.stood == null ? '-' : ev.stood.toFixed(2)) +
        '  pie ' + (ev.foot == null ? '-' : ev.foot.toFixed(3)) +
        '  cabeza ' + (ev.headMark == null ? '-' : ev.headMark.toFixed(3)) +
        '  manos ' + (ev.handMark == null ? '-' : ev.handMark.toFixed(3)) +
        '  cuerpo ' + (ev.body == null ? '-' : ev.body.toFixed(2)));
    }
  }
  const tail = tailLog(6);
  if (tail.length) {
    console.log('    ultimas lineas del log');
    for (const line of tail) console.log('      ' + line);
  }
}

const watchIndex = process.argv.indexOf('--watch');
if (watchIndex >= 0) {
  const every = Number(process.argv[watchIndex + 1]) || 60;
  show();
  setInterval(show, every * 1000);
} else {
  show();
}