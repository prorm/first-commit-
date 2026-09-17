/**
 * Static file serving with route aliases.
 *
 * Deliberately tiny and dependency-free: the demo must not depend on a build
 * step or a framework that could break on the morning of the presentation.
 * Everything under public/ is served as-is, plus the shared modules and the
 * classifier weights from src/, so the browser and Node load the identical
 * files.
 */
const fs = require('node:fs');
const path = require('node:path');
const { URL } = require('node:url');

const ROOT = path.join(__dirname, '..');
const PUBLIC = path.join(ROOT, 'public');

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.mjs': 'application/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.ico': 'image/x-icon',
  '.woff2': 'font/woff2',
  '.map': 'application/json',
  '.webmanifest': 'application/manifest+json',
  '.md': 'text/markdown; charset=utf-8',
  '.npz': 'application/octet-stream',
  '.onnx': 'application/octet-stream',
};

/** Friendly URLs -> files on disk. */
const ROUTES = {
  '/': 'index.html',
  '/phone/': 'phone/index.html',
  '/map/': 'map/index.html',
  '/diagnostics/': 'diagnostics/index.html',
};

/**
 * Directory URLs must carry the trailing slash.
 *
 * Without it the browser resolves `map.css` against `/` instead of `/map/`,
 * so the page loads with no stylesheet and no script — and it fails silently,
 * which is exactly the kind of thing that only shows up in front of an
 * audience.  Redirecting keeps the short URLs on the QR code and in the
 * startup banner working.
 */
const REDIRECTS = {
  '/phone': '/phone/',
  '/map': '/map/',
  '/diagnostics': '/diagnostics/',
};

/** Files outside public/ that the browser is allowed to load. */
const EXTRA = {
  '/favicon.ico': path.join(PUBLIC, 'assets', 'icon.svg'),
  '/vendor/echonet_weights.js': path.join(ROOT, 'src', 'classifier', 'echonet_weights.js'),
  '/legacy/gate-test': path.join(ROOT, 'gate_test.html'),
  '/docs/DEMO.md': path.join(ROOT, 'DEMO.md'),
  '/docs/ARCHITECTURE.md': path.join(ROOT, 'ARCHITECTURE.md'),
  '/docs/STATUS.md': path.join(ROOT, 'STATUS.md'),
  '/docs/PROTOCOL.md': path.join(ROOT, 'PROTOCOL.md'),
};

function resolvePath(urlPath) {
  if (EXTRA[urlPath]) return EXTRA[urlPath];
  const mapped = ROUTES[urlPath];
  if (mapped) return path.join(PUBLIC, mapped);

  // Normalise and confine to public/ — no traversal out of the web root.
  const clean = path.normalize(decodeURIComponent(urlPath)).replace(/^([/\\])+/, '');
  const candidate = path.join(PUBLIC, clean);
  if (!candidate.startsWith(PUBLIC)) return null;
  return candidate;
}

function serve(req, res, extraHandlers) {
  let urlPath = '/';
  try {
    urlPath = new URL(req.url, 'http://localhost').pathname;
  } catch (e) { /* fall back to / */ }

  if (extraHandlers) {
    for (const h of extraHandlers) {
      if (h.match(urlPath, req)) { h.handle(req, res, urlPath); return; }
    }
  }

  if (REDIRECTS[urlPath]) {
    const qs = req.url.indexOf('?') >= 0 ? req.url.slice(req.url.indexOf('?')) : '';
    res.writeHead(302, { Location: REDIRECTS[urlPath] + qs, 'Cache-Control': 'no-store' });
    res.end();
    return;
  }

  const filePath = resolvePath(urlPath);
  if (!filePath) { send(res, 400, 'text/plain', 'Bad path'); return; }

  fs.stat(filePath, (err, st) => {
    if (err || !st.isFile()) {
      // A clear 404 beats silently serving the wrong page — a mistyped URL
      // during a demo should be obvious, not mysterious.
      send(res, 404, 'text/html; charset=utf-8', notFoundPage(urlPath));
      return;
    }
    const ext = path.extname(filePath).toLowerCase();
    fs.readFile(filePath, (e2, buf) => {
      if (e2) { send(res, 500, 'text/plain', 'Read error: ' + e2.code); return; }
      send(res, 200, MIME[ext] || 'application/octet-stream', buf, {
        // No caching anywhere: an edit must show up on the next reload, and a
        // stale cached bundle on the phone is a demo-killer.
        'Cache-Control': 'no-store, must-revalidate',
        'Access-Control-Allow-Origin': '*',
      });
    });
  });
}

function send(res, code, type, body, headers) {
  res.writeHead(code, Object.assign({ 'Content-Type': type }, headers || {}));
  res.end(body);
}

function notFoundPage(urlPath) {
  return `<!doctype html><meta charset="utf-8">
<title>404 — SentryShield</title>
<style>body{background:#05070c;color:#e2e8f0;font:14px ui-monospace,Menlo,Consolas,monospace;padding:40px;line-height:1.7}
a{color:#22d3ee;text-decoration:none}a:hover{text-decoration:underline}code{color:#f59e0b}
h1{font-size:15px;letter-spacing:.18em;color:#22d3ee;font-weight:600}</style>
<h1>SENTRYSHIELD — 404</h1>
<p>No route for <code>${escapeHtml(urlPath)}</code>.</p>
<p>Available:</p>
<ul>
<li><a href="/map">/map</a> — command center</li>
<li><a href="/phone">/phone</a> — phone sensor</li>
<li><a href="/diagnostics">/diagnostics</a> — capability checks and self-tests</li>
<li><a href="/">/</a> — launcher</li>
</ul>`;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

module.exports = { serve, send, resolvePath, MIME, PUBLIC, ROOT };
