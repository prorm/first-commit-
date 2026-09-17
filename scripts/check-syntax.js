/**
 * Syntax check every source file, including the browser-only modules that no
 * Node test imports.
 *
 * There is no bundler in this project — a syntax error in a client module would
 * otherwise surface as a silently blank page in front of an audience.  This
 * catches it in one second.
 *
 *   node scripts/check-syntax.js
 */
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const ROOT = path.join(__dirname, '..');
const SKIP = new Set(['node_modules', '.git', 'recordings', 'graphify-out']);

function walk(dir, out = []) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    if (SKIP.has(entry.name)) continue;
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) walk(full, out);
    else if (/\.(mjs|js)$/.test(entry.name)) out.push(full);
  }
  return out;
}

const files = walk(ROOT);
let failures = 0;
let checked = 0;

for (const file of files) {
  const rel = path.relative(ROOT, file).replace(/\\/g, '/');
  const src = fs.readFileSync(file, 'utf8');
  // ESM and CJS parse under different rules, so pick by extension, then by
  // whether the file uses import/export syntax.
  const isModule = file.endsWith('.mjs')
    || /^\s*(import|export)\s/m.test(src)
    || /\bimport\s*\(/.test(src) && /^\s*export\b/m.test(src);
  try {
    if (isModule) {
      // eslint-disable-next-line no-new
      new vm.SourceTextModule(src, { identifier: rel });
    } else {
      new vm.Script(src, { filename: rel });
    }
    checked++;
  } catch (e) {
    failures++;
    console.error('FAIL  ' + rel);
    console.error('      ' + e.message.split('\n')[0]);
  }
}

// Also check the HTML pages reference files that exist.
const htmlFiles = walk(path.join(ROOT, 'public')).length ? [] : [];
for (const page of ['index.html', 'phone/index.html', 'map/index.html', 'diagnostics/index.html']) {
  const full = path.join(ROOT, 'public', page);
  if (!fs.existsSync(full)) { console.error('FAIL  missing page ' + page); failures++; continue; }
  const html = fs.readFileSync(full, 'utf8');
  const refs = [];
  const re = /(?:src|href)="([^"]+)"/g;
  let m;
  while ((m = re.exec(html))) {
    const href = m[1];
    if (/^(https?:|\/\/|#|mailto:)/.test(href)) continue;
    if (href.startsWith('/')) continue;          // server routes, checked by the browser check
    refs.push(href);
  }
  for (const ref of refs) {
    const target = path.resolve(path.dirname(full), ref.split('?')[0]);
    if (!fs.existsSync(target)) {
      console.error('FAIL  ' + page + ' references missing ' + ref);
      failures++;
    }
  }
}

console.log((failures ? 'SYNTAX CHECK FAILED' : 'syntax ok') + ' — ' + checked + ' files parsed, ' + failures + ' problem(s)');
process.exit(failures ? 1 : 0);
