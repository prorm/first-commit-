/**
 * Capture the demo's finale: a full apartment scan, reconstructed.
 *
 * Runs the real command center in real Chrome for a realistic scan length, then
 * triggers zero-visibility and the reconstruction transition and screenshots
 * the result.  This is how the visual payoff gets verified without a person
 * sitting and watching it.
 *
 *   node scripts/capture-hero.mjs [seconds] [scenario]
 */
import puppeteer from 'puppeteer-core';
import fs from 'node:fs';

const SECONDS = Number(process.argv[2] || 32);
const SCENARIO = process.argv[3] || 'apartment';
const BASE = 'http://localhost:8000';

const CANDIDATES = [
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
  process.env.LOCALAPPDATA + '/Google/Chrome/Application/chrome.exe',
  '/usr/bin/google-chrome',
];
const exe = CANDIDATES.find((p) => { try { return fs.existsSync(p); } catch (e) { return false; } });
if (!exe) { console.error('No Chrome found'); process.exit(2); }

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const browser = await puppeteer.launch({
  executablePath: exe,
  headless: 'new',
  args: ['--no-sandbox', '--window-size=1920,1080'],
});
const page = await browser.newPage();
await page.setViewport({ width: 1920, height: 1080 });
page.on('pageerror', (e) => console.error('  pageerror:', e.message));

await page.goto(BASE + '/map/', { waitUntil: 'networkidle2' });
await page.waitForFunction(() => document.getElementById('connText').textContent === 'LINKED', { timeout: 10000 });

await page.select('#selScenario', SCENARIO);
await sleep(500);
await page.click('#btnMission');
console.log('scanning ' + SCENARIO + ' for ' + SECONDS + ' s...');

// Let the scan accumulate, reporting progress so a long run is not silent.
for (let s = 0; s < SECONDS; s += 8) {
  await sleep(Math.min(8000, (SECONDS - s) * 1000));
  const st = await page.evaluate(() => {
    const g = (k) => {
      const el = Array.from(document.querySelectorAll('.stat')).find((x) => x.querySelector('.k').textContent === k);
      return el ? el.querySelector('.v').textContent : '?';
    };
    return { d: g('DETECTIONS'), p: g('ECHO POINTS'), a: g('AREA OBSERVED'), s: g('SURFACES'), o: g('OPENINGS?') };
  });
  console.log('  ' + (s + 8) + 's  detections ' + st.d + ' · points ' + st.p + ' · area ' + st.a + ' m2 · surfaces ' + st.s + ' · openings ' + st.o);
}

await page.click('#btnFit');
await sleep(1200);
await page.screenshot({ path: 'scripts/hero-1-scanned.png' });
console.log('shot  scripts/hero-1-scanned.png  (raw echo map)');

await page.click('#btnZero');
await sleep(900);
await page.screenshot({ path: 'scripts/hero-2-zerovis.png' });
console.log('shot  scripts/hero-2-zerovis.png  (zero visibility)');

await page.click('#btnRecon');
await sleep(900);
await page.screenshot({ path: 'scripts/hero-3-reconstructing.png' });
console.log('shot  scripts/hero-3-reconstructing.png  (mid-transition)');

await sleep(2600);
await page.click('#btnFit');
await sleep(1400);
await page.screenshot({ path: 'scripts/hero-4-reconstructed.png' });
console.log('shot  scripts/hero-4-reconstructed.png  (final reconstruction)');

// With the ground-truth overlay on, so the fit can be judged against the room.
await page.click('#layTruth');
await sleep(900);
await page.screenshot({ path: 'scripts/hero-5-vs-truth.png' });
console.log('shot  scripts/hero-5-vs-truth.png  (reconstruction vs. ground truth)');

const final = await page.evaluate(() => ({
  recon: document.getElementById('reconLine').textContent,
  ai: document.getElementById('aiNote').textContent,
  verdict: document.getElementById('aiVerdict').textContent,
}));
console.log('\n' + final.recon);
console.log('AI: ' + final.verdict + ' — ' + final.ai);

await page.click('#btnMission');
await sleep(1800);
await page.screenshot({ path: 'scripts/hero-6-summary.png' });
console.log('shot  scripts/hero-6-summary.png  (mission summary)');

await browser.close();
