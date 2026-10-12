import http from 'http';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const PORT = +(process.env.PORT || 8765);
const BACKUPS = path.join(ROOT, 'backups');
const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json',
  '.css': 'text/css'
};

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    req.on('data', (chunk) => {
      size += chunk.length;
      if (size > 8_000_000) { reject(new Error('too big')); req.destroy(); return; }
      chunks.push(chunk);
    });
    req.on('end', () => resolve(Buffer.concat(chunks)));
    req.on('error', reject);
  });
}

function safeBackupName(file) {
  const base = path.basename(String(file || ''));
  if (!/^g\d+_s[0-9mp.]+_(parent|best-ever|phase)_\d+(?:_\d+)?\.json$/.test(base)) return null;
  return base;
}

function uniqueBackupName(dir, wanted) {
  let name = safeBackupName(wanted) || ('g0_s0_parent_' + Date.now() + '.json');
  let full = path.join(dir, name);
  let n = 2;
  while (fs.existsSync(full)) {
    name = name.replace(/\.json$/, '_' + n + '.json');
    full = path.join(dir, name);
    n++;
  }
  return name;
}

http.createServer((req, res) => {
  const u = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  if (req.method === 'POST' && u === '/api/backup') {
    readBody(req).then((buf) => {
      let data;
      try { data = JSON.parse(buf.toString('utf8')); }
      catch { res.writeHead(400); res.end('bad json'); return; }
      fs.mkdirSync(BACKUPS, { recursive: true });
      const name = uniqueBackupName(BACKUPS, data.file);
      data.file = name;
      fs.writeFileSync(path.join(BACKUPS, name), JSON.stringify(data));
      res.writeHead(200, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' });
      res.end(JSON.stringify({ ok: true, file: name }));
    }).catch(() => { if (!res.headersSent) { res.writeHead(413); res.end('too big'); } });
    return;
  }
  const rel = u === '/' ? 'index.html' : u.replace(/^[/\\]+/, '');
  const p = path.normalize(path.join(ROOT, rel));
  if (!p.startsWith(ROOT)) { res.writeHead(403); res.end(); return; }
  fs.readFile(p, (e, d) => {
    if (e) { res.writeHead(404); res.end('not found'); return; }
    res.writeHead(200, { 'Content-Type': MIME[path.extname(p)] || 'application/octet-stream', 'Cache-Control': 'no-store' });
    res.end(d);
  });
}).listen(PORT, () => console.log('viz http://localhost:' + PORT + '/'));
