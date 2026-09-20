#!/usr/bin/env node
/**
 * SentryShield server entry point.
 *
 *   npm start
 *
 * Starts HTTP and (if certificates exist) HTTPS, mounts the WebSocket hub on
 * /ws, finds the LAN address, and prints the phone URL as a scannable QR code.
 * Mobile Chrome needs a secure context for getUserMedia, so the QR points at
 * HTTPS whenever a certificate is available.
 */
const http = require('node:http');
const https = require('node:https');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');

loadDotEnv();

const staticServer = require('./static');
const { Hub } = require('./hub');
const { SCENARIO_LIST } = require('./simulation');

const ROOT = path.join(__dirname, '..');
const HTTP_PORT = Number(process.env.PORT || 8000);
const HTTPS_PORT = Number(process.env.HTTPS_PORT || 8443);

// ---------------------------------------------------------------------------
// .env loading — no dependency, no overwriting of real environment variables
// ---------------------------------------------------------------------------
function loadDotEnv() {
  const file = path.join(__dirname, '..', '.env');
  if (!fs.existsSync(file)) return;
  try {
    for (const raw of fs.readFileSync(file, 'utf8').split(/\r?\n/)) {
      const line = raw.trim();
      if (!line || line.startsWith('#')) continue;
      const eq = line.indexOf('=');
      if (eq < 1) continue;
      const key = line.slice(0, eq).trim();
      let val = line.slice(eq + 1).trim();
      if ((val.startsWith('"') && val.endsWith('"')) || (val.startsWith("'") && val.endsWith("'"))) {
        val = val.slice(1, -1);
      }
      // A real environment variable always wins over the file.
      if (val && process.env[key] === undefined) process.env[key] = val;
    }
  } catch (e) {
    console.warn('[warn] could not read .env:', e.message);
  }
}

// ---------------------------------------------------------------------------
// LAN address
// ---------------------------------------------------------------------------
function getLocalIPs() {
  const out = [];
  const ifaces = os.networkInterfaces();
  for (const name of Object.keys(ifaces)) {
    for (const i of ifaces[name] || []) {
      if (i.family !== 'IPv4' || i.internal) continue;
      if (i.address.startsWith('169.254')) continue;         // link-local, useless for a phone
      out.push({ name, address: i.address });
    }
  }
  // Prefer a Wi-Fi adapter: that is the network the phone is actually on.
  out.sort((a, b) => score(b.name) - score(a.name));
  return out;
}

function score(name) {
  const n = name.toLowerCase();
  if (/wi-?fi|wlan|wireless/.test(n)) return 3;
  if (/ethernet|eth|en\d/.test(n)) return 2;
  if (/vethernet|virtual|vmware|vbox|hyper-v|docker|wsl|loopback|tailscale|zerotier/.test(n)) return 0;
  return 1;
}

// ---------------------------------------------------------------------------
// API routes (everything stateful goes over WebSocket; these are for tooling)
// ---------------------------------------------------------------------------
function apiHandlers(hub) {
  return [
    {
      match: (p) => p === '/api/status',
      handle: (req, res) => staticServer.send(res, 200, 'application/json', JSON.stringify(hub.statusPayload(), null, 2), { 'Access-Control-Allow-Origin': '*' }),
    },
    {
      match: (p) => p === '/api/health',
      handle: (req, res) => staticServer.send(res, 200, 'application/json', JSON.stringify({
        ok: true, uptimeS: Math.round(process.uptime()), mode: hub.mode,
        peers: hub.peerSummary(), classifier: hub.capabilities().classifier,
      }), { 'Access-Control-Allow-Origin': '*' }),
    },
    {
      match: (p) => p === '/api/scenarios',
      handle: (req, res) => staticServer.send(res, 200, 'application/json', JSON.stringify(SCENARIO_LIST), { 'Access-Control-Allow-Origin': '*' }),
    },
    {
      match: (p) => p === '/api/scenario',
      handle: (req, res) => staticServer.send(res, 200, 'application/json', JSON.stringify(hub.sim.info()), { 'Access-Control-Allow-Origin': '*' }),
    },
    {
      match: (p) => p === '/api/recordings',
      handle: (req, res) => staticServer.send(res, 200, 'application/json', JSON.stringify(hub.recorder.list()), { 'Access-Control-Allow-Origin': '*' }),
    },
    {
      match: (p) => p === '/api/snapshot',
      handle: (req, res) => staticServer.send(res, 200, 'application/json', JSON.stringify(hub.map.snapshot({ reconstruct: true })), { 'Access-Control-Allow-Origin': '*' }),
    },
    {
      match: (p) => p === '/api/aws',
      handle: (req, res) => staticServer.send(res, 200, 'application/json', JSON.stringify(hub.aws.status(), null, 2), { 'Access-Control-Allow-Origin': '*' }),
    },
  ];
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
const hub = new Hub({
  log: (tag, payload) => {
    if (process.env.SENTRY_QUIET === 'true') return;
    if (tag === 'error') console.error('[' + tag + ']', payload);
    else if (tag !== 'ws') console.log('[' + tag + ']', JSON.stringify(payload));
  },
});

const handlers = apiHandlers(hub);
const requestListener = (req, res) => staticServer.serve(req, res, handlers);

const httpServer = http.createServer(requestListener);
hub.attach(httpServer);

let httpsServer = null;
const keyPath = path.join(ROOT, 'key.pem');
const certPath = path.join(ROOT, 'cert.pem');
const haveCerts = fs.existsSync(keyPath) && fs.existsSync(certPath);

if (haveCerts) {
  try {
    httpsServer = https.createServer(
      { key: fs.readFileSync(keyPath), cert: fs.readFileSync(certPath) },
      requestListener
    );
    // A second WebSocketServer on the HTTPS listener, sharing the same hub —
    // the phone connects over WSS while the laptop's map uses WS.
    hub.attach(httpsServer);
  } catch (e) {
    console.warn('[warn] HTTPS disabled:', e.message);
    httpsServer = null;
  }
}

httpServer.on('error', (e) => fatalListen(e, HTTP_PORT, 'HTTP'));

httpServer.listen(HTTP_PORT, '0.0.0.0', () => {
  if (httpsServer) {
    httpsServer.on('error', (e) => {
      console.warn('[warn] HTTPS listen failed on ' + HTTPS_PORT + ':', e.message);
      printBanner(false);
    });
    httpsServer.listen(HTTPS_PORT, '0.0.0.0', () => printBanner(true));
  } else {
    printBanner(false);
  }
});

function fatalListen(e, port, label) {
  if (e.code === 'EADDRINUSE') {
    console.error(`\n[fatal] ${label} port ${port} is already in use.`);
    console.error(`        Another SentryShield may be running. Stop it, or start with a different port:`);
    console.error(`        PORT=${port + 1} npm start\n`);
    process.exit(1);
  }
  console.error('[fatal]', e.message);
  process.exit(1);
}

// ---------------------------------------------------------------------------
// Startup banner + QR
// ---------------------------------------------------------------------------
async function printBanner(httpsUp) {
  const ips = getLocalIPs();
  const lan = ips.length ? ips[0].address : 'localhost';
  // The phone needs a secure context for the microphone, so point it at HTTPS
  // when we have one.
  const phoneURL = httpsUp ? `https://${lan}:${HTTPS_PORT}/phone` : `http://${lan}:${HTTP_PORT}/phone`;
  const mapURL = `http://localhost:${HTTP_PORT}/map`;
  const diagURL = httpsUp ? `https://${lan}:${HTTPS_PORT}/diagnostics` : `http://${lan}:${HTTP_PORT}/diagnostics`;
  const aws = hub.aws.status();
  const caps = hub.capabilities();

  const bar = '='.repeat(64);
  console.log('\n' + bar);
  console.log('  SENTRYSHIELD READY        acoustic perception prototype');
  console.log(bar);
  console.log('');
  console.log('  COMMAND CENTER   ' + mapURL);
  console.log('  PHONE SENSOR     ' + phoneURL);
  console.log('  DIAGNOSTICS      ' + diagURL);
  if (!httpsUp) {
    console.log('');
    console.log('  ! No HTTPS: mobile Chrome blocks the microphone on plain HTTP.');
    console.log('    Simulation mode still works fully. To enable live audio, see DEMO.md');
    console.log('    (generate key.pem/cert.pem, then restart).');
  }
  console.log('');
  console.log('  MODE             ' + hub.mode.toUpperCase() + '   (no phone needed for simulation)');
  console.log('  CLASSIFIER       EchoNet ' + caps.classifier +
    (caps.classifierMeta
      ? '  (' + caps.classifierMeta.n_params + ' params, '
        + (caps.classifierMeta.classes_in_use || ['WALL', 'SOFT']).join('/') + ' only, '
        + (caps.classifierMeta.val_accuracy * 100).toFixed(0) + '% on real echoes vs 50% chance - experimental)'
      : ''));
  console.log('  VOICE            ' + aws.voice.label + '   (' + aws.voice.reason + ')');
  console.log('  BEDROCK          ' + (aws.bedrock.enabled ? 'ENABLED  ' + aws.bedrock.modelId : 'disabled  (' + aws.bedrock.reason + ')'));
  console.log('  S3 ARCHIVE       ' + (aws.s3.enabled ? 'ENABLED  s3://' + aws.s3.bucket + '/' + aws.s3.prefix : 'disabled  (' + aws.s3.reason + ')'));
  console.log('  SCENARIOS        ' + SCENARIO_LIST.map((s) => s.id).join(', '));
  if (ips.length > 1) {
    console.log('  OTHER ADDRESSES  ' + ips.slice(1).map((i) => i.address).join(', '));
  }
  console.log('');

  await printQR(phoneURL);

  console.log(bar);
  console.log('  Scan the QR with the phone, or open the phone URL directly.');
  console.log('  Self-signed certificate: tap Advanced -> Proceed on first visit.');
  console.log('  Ctrl+C to stop.');
  console.log(bar + '\n');
}

async function printQR(url) {
  try {
    const qrcode = require('qrcode');
    const art = await qrcode.toString(url, { type: 'terminal', small: true, errorCorrectionLevel: 'L' });
    console.log('  SCAN TO CONNECT PHONE');
    console.log(art.split('\n').map((l) => '  ' + l).join('\n'));
  } catch (e) {
    console.log('  (QR unavailable: ' + e.message + ') — open the phone URL manually.');
  }
}

// ---------------------------------------------------------------------------
// Shutdown
// ---------------------------------------------------------------------------
function shutdown(signal) {
  console.log('\n[' + signal + '] shutting down...');
  hub.close();
  const done = () => process.exit(0);
  httpServer.close(() => { if (httpsServer) httpsServer.close(done); else done(); });
  setTimeout(done, 1500).unref();
}
process.on('SIGINT', () => shutdown('SIGINT'));
process.on('SIGTERM', () => shutdown('SIGTERM'));

// A crash during a live demo must not take the server with it if it can be
// survived: log loudly, keep serving.
process.on('uncaughtException', (e) => {
  console.error('[uncaughtException]', e && e.stack ? e.stack : e);
});
process.on('unhandledRejection', (e) => {
  console.error('[unhandledRejection]', e && e.stack ? e.stack : e);
});

module.exports = { hub, httpServer, getLocalIPs };
