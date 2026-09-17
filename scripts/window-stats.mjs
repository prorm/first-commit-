/**
 * Compare the digital twin's 64-sample windows against the real training set's
 * shape statistics (printed by scripts/dataset_stats.py).  Same descriptors,
 * so any drift between simulated and trained-on echoes is visible.
 *
 *   node scripts/window-stats.mjs [nPerClass]
 */
import { synthesizePulse } from '../public/shared/echosynth.mjs';

const N = Number(process.argv[2] || 400);
const CLASSES = ['WALL', 'SOFT', 'OPENING'];
let seed = 7717;
const rng = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);

function feats(w) {
  const n = w.length;
  let pk = 0;
  for (let i = 1; i < n; i++) if (w[i] > w[pk]) pk = i;
  let mass = 0, width = 0, zeros = 0, lobe = 0, rough = 0, lmax = 0;
  for (let i = 0; i < n; i++) {
    mass += w[i];
    if (w[i] >= 0.5) width++;
    if (w[i] <= 1e-6) zeros++;
    if (i >= pk - 12 && i <= pk + 12) lobe += w[i];
  }
  for (let i = 1; i < n - 1; i++) {
    rough += Math.abs(w[i + 1] - 2 * w[i] + w[i - 1]);
    if (w[i] >= w[i - 1] && w[i] > w[i + 1] && w[i] > 0.25) lmax++;
  }
  return {
    peak: pk, mass, width, zeros: zeros / n,
    tailfrac: (mass - lobe) / Math.max(mass, 1e-9),
    rough: rough / (n - 2), lmax,
  };
}

console.log('  REAL (from training npz, for reference):');
console.log('           peakIdx    mass   w>0.5   zerofrac  tailfrac   rough   localmax');
console.log('    WALL       32.0   28.42   27.28     0.037     0.262  0.0038     1.01');
console.log('    SOFT       31.9   33.21   32.57     0.053     0.368  0.0049     1.33');
console.log('    OPENING    32.1   30.07   29.33     0.096     0.308  0.0051     1.23');
console.log('');
console.log('  SIMULATED (this digital twin):');
console.log('           peakIdx    mass   w>0.5   zerofrac  tailfrac   rough   localmax     snr');
for (let c = 0; c < 3; c++) {
  const acc = { peak: 0, mass: 0, width: 0, zeros: 0, tailfrac: 0, rough: 0, lmax: 0, snr: 0 };
  for (let k = 0; k < N; k++) {
    const r = 0.5 + rng() * (CLASSES[c] === 'OPENING' ? 2.2 : 2.9);
    const p = synthesizePulse(CLASSES[c], r, rng, {});
    const f = feats(p.win);
    for (const key of Object.keys(f)) acc[key] += f[key];
    acc.snr += p.snr_db;
  }
  const m = (k) => acc[k] / N;
  console.log(
    `    ${CLASSES[c].padEnd(8)} ${m('peak').toFixed(1).padStart(7)} ${m('mass').toFixed(2).padStart(7)} ` +
    `${m('width').toFixed(2).padStart(7)} ${m('zeros').toFixed(3).padStart(9)} ${m('tailfrac').toFixed(3).padStart(9)} ` +
    `${m('rough').toFixed(4).padStart(7)} ${m('lmax').toFixed(2).padStart(8)} ${m('snr').toFixed(1).padStart(7)}`
  );
}
