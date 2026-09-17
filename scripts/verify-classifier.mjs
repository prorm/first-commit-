/**
 * Cross-check the digital twin against the real classifier.
 *
 * Synthesises N pulses per class with the simulation physics, runs them
 * through the exported EchoNet weights, and prints the resulting confusion
 * matrix next to the one recorded at training time.  If simulation drifted
 * away from the training distribution, this is where it shows up.
 *
 *   node scripts/verify-classifier.mjs [nPerClass]
 */
import { createRequire } from 'node:module';
import { synthesizePulse } from '../public/shared/echosynth.mjs';

const require = createRequire(import.meta.url);
const EchoNet = require('../src/classifier/echonet_weights.js');

const N = Number(process.argv[2] || 300);
const CLASSES = ['WALL', 'SOFT', 'OPENING'];

// deterministic LCG so runs are comparable
let seed = 20260917;
const rng = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);

const st = EchoNet.selfTest();
console.log(`EchoNet self-test: ${st.ok ? 'PASS' : 'FAIL'} (maxAbsError ${st.maxAbsError.toExponential(2)})`);
console.log(`Shipped accuracy: ${(EchoNet.META.val_accuracy * 100).toFixed(1)} % (${(EchoNet.META.classes_in_use || ['WALL', 'SOFT']).join('/')}, chance 50 %)`);
console.log(`Basis: ${EchoNet.META.accuracy_basis}`);
console.log(`Confusion (rows = truth): ${JSON.stringify(EchoNet.META.confusion_matrix)}`);
console.log('');

const cm = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
let snrSum = 0;
for (let ci = 0; ci < 3; ci++) {
  for (let k = 0; k < N; k++) {
    const r = 0.45 + rng() * (CLASSES[ci] === 'OPENING' ? 2.6 : 3.1);
    const p = synthesizePulse(CLASSES[ci], r, rng, {});
    const out = EchoNet.forward(p.win);
    cm[ci][out.classIndex]++;
    snrSum += p.snr_db;
  }
}

console.log('Simulation-vs-EchoNet confusion (rows = simulated truth):');
console.log('            WALL   SOFT   OPEN   recall');
let correct = 0;
for (let i = 0; i < 3; i++) {
  const row = cm[i];
  const tot = row[0] + row[1] + row[2];
  correct += row[i];
  console.log(
    `  ${CLASSES[i].padEnd(8)}` +
    row.map((v) => String(v).padStart(6)).join(' ') +
    `   ${((row[i] / tot) * 100).toFixed(1)} %`
  );
}
console.log(`\noverall ${((correct / (3 * N)) * 100).toFixed(1)} %   mean SNR ${(snrSum / (3 * N)).toFixed(1)} dB`);
console.log('\nThese are the model\'s real outputs on physically synthesised echoes — not scripted labels.');
