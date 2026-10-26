#!/bin/bash
# Bundles play/app.js (sim, pose executor, terrain, stand net weights) into one page: play/ragdoll-live.html.
# Box2D stays an import from the CDN. Needs esbuild (ESBUILD env or npx esbuild).
set -e
cd "$(dirname "$0")/.."
ESB=${ESBUILD:-npx esbuild}
$ESB play/app.js --bundle --format=esm --minify --external:'https://*' --loader:.json=json --outfile=play/bundle.js --log-level=warning
python3 - <<'PY'
s = open('play/shell.html').read()
b = open('play/bundle.js').read().replace('</script', '<\\/script')
open('play/ragdoll-live.html', 'w').write(s.replace('/*BUNDLE*/', b))
print('wrote play/ragdoll-live.html', len(s) + len(b), 'bytes')
PY
