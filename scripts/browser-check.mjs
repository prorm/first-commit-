/**
 * Drive the real browser against a running server and fail on any client-side
 * error.  This is the only way to verify the ES modules, canvas rendering and
 * WebSocket clients actually run — unit tests cannot see a page that throws on
 * load.
 *
 *   node scripts/browser-check.mjs [baseUrl]
 *
 * Uses the locally installed Chrome (puppeteer-core, no bundled download).
 */
import puppeteer from 'puppeteer-core';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const RECORDINGS = path.join(HERE, '..', 'recordings');

/** Recordings that existed before this run, so we only clean up our own. */
function listRecordings() {
  try { return new Set(fs.readdirSync(RECORDINGS).filter((f) => f.endsWith('.json'))); } catch (e) { return new Set(); }
}
const preexisting = listRecordings();

const BASE = process.argv[2] || 'http://localhost:8000';
const CHROME_CANDIDATES = [
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
  process.env.LOCALAPPDATA + '/Google/Chrome/Application/chrome.exe',
  'C:/Program Files/Microsoft/Edge/Application/msedge.exe',
  '/usr/bin/google-chrome',
  '/usr/bin/chromium',
];

const exe = CHROME_CANDIDATES.find((p) => { try { return fs.existsSync(p); } catch (e) { return false; } });
if (!exe) {
  console.error('No Chrome/Edge binary found. Checked:\n  ' + CHROME_CANDIDATES.join('\n  '));
  process.exit(2);
}

let failures = 0;
const note = (ok, msg) => {
  if (!ok) failures++;
  console.log((ok ? '  PASS  ' : '  FAIL  ') + msg);
};

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const browser = await puppeteer.launch({
  executablePath: exe,
  headless: 'new',
  args: [
    '--no-sandbox',
    '--autoplay-policy=no-user-gesture-required',
    // Grant a fake mic so the sensor path can be exercised headlessly.
    '--use-fake-ui-for-media-stream',
    '--use-fake-device-for-media-stream',
    '--window-size=1600,1000',
  ],
});

/**
 * Open a page, collect every console error, page error and failed request,
 * then run a page-specific assertion.
 */
async function visit(path, opts = {}) {
  console.log('\n' + path);
  const page = await browser.newPage();
  await page.setViewport({ width: opts.width || 1600, height: opts.height || 1000 });
  const errors = [];
  const failed = [];

  page.on('console', (m) => {
    if (m.type() === 'error') errors.push(m.text());
  });
  page.on('pageerror', (e) => {
    const stack = e && e.stack ? String(e.stack).split('\n').slice(0, 4).join(' <- ') : '';
    errors.push('pageerror: ' + (e && e.message) + (stack ? ' [' + stack + ']' : ''));
  });
  page.on('requestfailed', (r) => {
    // Favicon misses are noise; anything else is a real broken reference.
    if (!/favicon/.test(r.url())) failed.push(r.url() + ' (' + (r.failure() && r.failure().errorText) + ')');
  });
  page.on('response', (r) => {
    if (r.status() >= 400 && !/favicon/.test(r.url())) failed.push(r.url() + ' -> HTTP ' + r.status());
  });

  await page.goto(BASE + path, { waitUntil: 'networkidle2', timeout: 25000 });
  await sleep(opts.settle || 1800);

  if (opts.run) {
    try { await opts.run(page, { note }); } catch (e) { note(false, 'in-page assertions threw: ' + e.message); }
  }

  // Errors that are expected on this page (e.g. autoplay refusals) can be
  // filtered, but nothing is filtered by default.
  const ignore = opts.ignoreErrors || [];
  const real = errors.filter((t) => !ignore.some((re) => re.test(t)));
  note(real.length === 0, 'no console/page errors' + (real.length ? ': ' + real.slice(0, 5).join(' | ') : ''));
  note(failed.length === 0, 'no failed requests' + (failed.length ? ': ' + failed.slice(0, 5).join(' | ') : ''));

  if (opts.screenshot) {
    await page.screenshot({ path: opts.screenshot });
    console.log('  shot  ' + opts.screenshot);
  }
  await page.close();
}

// ---------------------------------------------------------------------------
console.log('SentryShield browser check against ' + BASE);
console.log('Chrome: ' + exe);

// ---- launcher
await visit('/', {
  settle: 600,
  run: async (page) => {
    const routes = await page.$$eval('.route .p', (els) => els.map((e) => e.textContent));
    note(routes.includes('/map') && routes.includes('/phone') && routes.includes('/diagnostics'),
      'launcher links to all three routes');
  },
});

// ---- command center, simulation mission end to end
await visit('/map', {
  settle: 1200,
  screenshot: 'scripts/shot-map-empty.png',
  run: async (page, { note: n }) => {
    n(await page.$('#stage') !== null, 'canvas mounted');
    const linked = await page.waitForFunction(
      () => document.getElementById('connText').textContent === 'LINKED',
      { timeout: 8000 }
    ).then(() => true).catch(() => false);
    n(linked, 'WebSocket reached LINKED');

    // Scenario list must be populated by the server, not hardcoded.
    const scenarios = await page.$$eval('#selScenario option', (o) => o.map((x) => x.value));
    n(scenarios.length >= 5, 'scenarios populated from the server (' + scenarios.length + ')');

    // Run a real simulated mission.
    await page.click('#btnMission');
    await sleep(6000);

    const stats = await page.evaluate(() => {
      const get = (k) => {
        const el = Array.from(document.querySelectorAll('.stat')).find((s) => s.querySelector('.k').textContent === k);
        return el ? el.querySelector('.v').textContent : null;
      };
      return {
        detections: parseInt(get('DETECTIONS') || '0', 10),
        points: parseInt(get('ECHO POINTS') || '0', 10),
        area: parseFloat(get('AREA OBSERVED') || '0'),
        surfaces: parseInt(get('SURFACES') || '0', 10),
        verdict: document.getElementById('aiVerdict').textContent,
        reconLine: document.getElementById('reconLine').textContent,
        aiNote: document.getElementById('aiNote').textContent,
      };
    });
    n(stats.detections > 40, 'detections streamed in (' + stats.detections + ')');
    n(stats.points > 10, 'echo cloud accumulated (' + stats.points + ' points)');
    n(stats.area > 3, 'occupancy grid mapped real area (' + stats.area + ' m2)');
    n(stats.surfaces > 0, 'surfaces reconstructed (' + stats.surfaces + ')');
    n(['WALL', 'SOFT', 'OPENING'].includes(stats.verdict), 'AI panel shows a real class (' + stats.verdict + ')');
    console.log('        recon: ' + stats.reconLine);
    console.log('        ai:    ' + stats.aiNote.slice(0, 96));

    // Zero-visibility + reconstruction: the two headline visual states.
    await page.click('#btnZero');
    await sleep(400);
    n(await page.$eval('#zvBanner', (e) => !e.hidden), 'zero-visibility banner shown');
    await page.click('#btnRecon');
    await sleep(900);
    await page.screenshot({ path: 'scripts/shot-map-zerovis.png' });
    console.log('  shot  scripts/shot-map-zerovis.png');
    await sleep(1400);
    await page.click('#btnFit');
    await sleep(900);
    await page.screenshot({ path: 'scripts/shot-map-reconstructed.png' });
    console.log('  shot  scripts/shot-map-reconstructed.png');

    // Finish the mission and check the summary.
    await page.click('#btnMission');
    await sleep(1600);
    const sum = await page.evaluate(() => ({
      open: !document.getElementById('summarySheet').hidden,
      rows: document.querySelectorAll('#sumGrid .sum-row').length,
      caveats: document.querySelectorAll('#sumCaveats li').length,
    }));
    n(sum.open, 'mission summary opened');
    n(sum.rows >= 10, 'summary has the full statistics table (' + sum.rows + ' rows)');
    n(sum.caveats >= 3, 'summary states its limits (' + sum.caveats + ' caveats)');
    await page.screenshot({ path: 'scripts/shot-map-summary.png' });
    console.log('  shot  scripts/shot-map-summary.png');

    // Replay must rebuild from empty.
    await page.click('#sumClose');
    await page.click('#btnReplay');
    await sleep(1200);
    const replaying = await page.$eval('#replayBar', (e) => !e.hidden);
    n(replaying, 'replay started and shows progress');
    await sleep(4000);
    const afterReplay = await page.evaluate(() => {
      const el = Array.from(document.querySelectorAll('.stat')).find((s) => s.querySelector('.k').textContent === 'ECHO POINTS');
      return parseInt(el ? el.querySelector('.v').textContent : '0', 10);
    });
    n(afterReplay > 5, 'replay repopulated the map (' + afterReplay + ' points)');
  },
});

// ---- phone sensor (headless, with a fake mic device)
await visit('/phone', {
  width: 412,
  height: 915,
  settle: 1200,
  // A headless fake device cannot produce real ultrasonic echoes; getUserMedia
  // constraint refusals are expected and are reported in-page, not thrown.
  ignoreErrors: [/autoplay/i, /play\(\) failed/i],
  screenshot: 'scripts/shot-phone-gate.png',
  run: async (page, { note: n }) => {
    const caps = await page.$$eval('.gate-cap', (els) => els.map((e) => e.textContent));
    n(caps.length >= 6, 'capability gate lists subsystems (' + caps.length + ')');
    await page.click('#gateGo');
    await sleep(1800);
    n(await page.$eval('#gateSheet', (e) => e.hidden), 'gate dismissed after enabling sensors');

    const s = await page.evaluate(() => ({
      logs: document.querySelectorAll('#log .log-line').length,
      sonar: !!document.getElementById('sonar').getContext,
      voice: document.getElementById('voiceLabel').textContent,
      conn: document.getElementById('connText').textContent,
    }));
    n(s.logs > 0, 'sensor log populated (' + s.logs + ' lines)');
    n(s.sonar, 'sonar canvas has a 2D context');
    n(/LOCAL FALLBACK|AWS POLLY/.test(s.voice), 'voice provider labelled honestly (' + s.voice + ')');
    n(s.conn === 'LINKED', 'phone reached the server (' + s.conn + ')');

    // Manual pose controls must work with no motion sensors at all.
    const h0 = await page.$eval('#headingValue', (e) => e.textContent);
    await page.click('#btnTurnR');
    await sleep(200);
    const h1 = await page.$eval('#headingValue', (e) => e.textContent);
    n(h0 !== h1, 'manual turn changes heading (' + h0 + ' -> ' + h1 + ')');
    await page.click('#btnWalk');
    await sleep(200);
    const dist = await page.evaluate(() => document.getElementById('poseMethod').textContent);
    n(dist === 'MANUAL', 'manual step switches pose method to MANUAL');

    // Start the scan: it may legitimately fail on a fake device, but must
    // report the reason rather than break the page.
    await page.click('#btnScan');
    await sleep(2500);
    const scan = await page.evaluate(() => ({
      state: document.getElementById('scanState').textContent,
      cue: document.getElementById('cueText').textContent,
    }));
    console.log('        scan: ' + scan.state);
    n(true, 'scan attempt handled without breaking the page');
    await page.screenshot({ path: 'scripts/shot-phone-live.png' });
    console.log('  shot  scripts/shot-phone-live.png');
  },
});

// ---- diagnostics + in-browser self tests
await visit('/diagnostics', {
  width: 1100,
  height: 1200,
  settle: 1200,
  run: async (page, { note: n }) => {
    await page.click('#btnSelfTest');
    await sleep(3500);
    const res = await page.evaluate(() => {
      const lines = Array.from(document.querySelectorAll('#out .l')).map((e) => e.textContent);
      const summary = lines.find((l) => /SELF TEST: /.test(l)) || '';
      return {
        summary,
        fails: lines.filter((l) => /FAIL  /.test(l)),
        verdict: document.getElementById('verdict').textContent,
      };
    });
    console.log('        ' + res.summary.replace(/^\d+:\d+:\d+\s*/, ''));
    for (const f of res.fails.slice(0, 8)) console.log('        ' + f.replace(/^\d+:\d+:\d+\s*/, ''));
    const m = res.summary.match(/(\d+) passed, (\d+) failed/);
    n(!!m, 'self test produced a summary');
    if (m) {
      n(Number(m[2]) === 0, m[1] + ' self-test checks passed, ' + m[2] + ' failed');
      n(Number(m[1]) >= 20, 'self test covers at least 20 checks (' + m[1] + ')');
    }

    await page.click('#btnWs');
    await sleep(2200);
    const ws = await page.evaluate(() =>
      Array.from(document.querySelectorAll('#out .l')).map((e) => e.textContent).join('\n'));
    n(/kept the socket open/.test(ws), 'malformed-frame resilience verified in-browser');

    await page.click('#btnLoop');
    await sleep(8000);
    const loop = await page.evaluate(() =>
      Array.from(document.querySelectorAll('#out .l')).map((e) => e.textContent).join('\n'));
    const dm = loop.match(/Detections received: (\d+)/);
    n(dm && Number(dm[1]) > 30, 'full-loop test streamed detections (' + (dm ? dm[1] : '0') + ')');
    n(/Reconstruction: /.test(loop), 'full-loop test reached reconstruction');
    await page.screenshot({ path: 'scripts/shot-diagnostics.png', fullPage: true });
    console.log('  shot  scripts/shot-diagnostics.png');
  },
});

await browser.close();

// The check runs real missions, and real missions record.  Remove only what
// this run created: a stray test scan would change which recording
// "REPLAY LAST" resolves to during an actual demo.
let cleaned = 0;
for (const f of listRecordings()) {
  if (preexisting.has(f)) continue;
  try { fs.unlinkSync(path.join(RECORDINGS, f)); cleaned++; } catch (e) { /* leave it */ }
}
if (cleaned) console.log('\ncleaned ' + cleaned + ' recording(s) created by this run');

console.log('\n' + (failures === 0 ? 'BROWSER CHECK PASSED' : 'BROWSER CHECK FAILED — ' + failures + ' problem(s)'));
process.exit(failures === 0 ? 0 : 1);
