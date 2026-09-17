/**
 * Reconstruction quality evaluation.
 *
 * Runs the digital twin headlessly through every scenario for a realistic scan
 * duration and reports what the map ends up with, plus how far the fitted
 * surfaces sit from the scenario's real geometry.  Ground truth is known here
 * (it is a simulation), so this gives an honest error figure for the
 * reconstruction layer rather than a vibe.
 *
 *   node scripts/eval-reconstruction.js [secondsPerScenario]
 */
const { SimulationEngine, SCENARIOS } = require('../server/simulation');
const { MapState } = require('../server/mapstate');

const SECONDS = Number(process.argv[2] || 30);
const RATE = 20;

/** Shortest distance from a point to a scenario surface segment. */
function distToSegment(px, py, s) {
  const ax = s.a[0], ay = s.a[1], bx = s.b[0], by = s.b[1];
  const dx = bx - ax, dy = by - ay;
  const len2 = dx * dx + dy * dy;
  if (len2 < 1e-9) return Math.hypot(px - ax, py - ay);
  let t = ((px - ax) * dx + (py - ay) * dy) / len2;
  t = t < 0 ? 0 : t > 1 ? 1 : t;
  return Math.hypot(ax + dx * t - px, ay + dy * t - py);
}

/** Distance from a fitted segment to the nearest real surface, sampled along it. */
function segmentError(seg, surfaces) {
  const solid = surfaces.filter((s) => s.kind !== 'OPENING');
  if (!solid.length) return null;
  let sum = 0;
  let worst = 0;
  const N = 12;
  for (let i = 0; i <= N; i++) {
    const t = i / N;
    const x = seg.a.x + (seg.b.x - seg.a.x) * t;
    const y = seg.a.y + (seg.b.y - seg.a.y) * t;
    let best = Infinity;
    for (const s of solid) best = Math.min(best, distToSegment(x, y, s));
    sum += best;
    if (best > worst) worst = best;
  }
  return { mean: sum / (N + 1), worst };
}

/** Real doorway centres in a scenario, for opening-recall bookkeeping. */
function trueOpenings(surfaces) {
  return surfaces.filter((s) => s.kind === 'OPENING').map((s) => ({
    x: (s.a[0] + s.b[0]) / 2,
    y: (s.a[1] + s.b[1]) / 2,
    width: Math.hypot(s.b[0] - s.a[0], s.b[1] - s.a[1]),
  }));
}

console.log('Reconstruction evaluation — ' + SECONDS + ' s per scenario at ' + RATE + ' Hz');
console.log('(simulation ground truth is known, so these errors are real measurements)\n');

const header = 'scenario     pulses  dets  points  surf  len(m)  corner  open  conf   segErr(mean/worst)  openFound';
console.log(header);
console.log('-'.repeat(header.length));

const totals = { segErr: [], conf: [], openFound: 0, openTrue: 0 };

for (const id of Object.keys(SCENARIOS)) {
  const map = new MapState();
  map.startMission();
  const sim = new SimulationEngine({ onDetection: (det) => { if (det) map.applyDetection(det); } });
  sim.setScenario(id, 12345);
  sim.setMode('simulation');

  const steps = SECONDS * RATE;
  const dt = 1 / RATE;
  for (let i = 0; i < steps; i++) sim.tick(dt);

  const recon = map.buildReconstruction(true);
  const surfaces = SCENARIOS[id].surfaces;
  const errs = recon.segments.map((s) => segmentError(s, surfaces)).filter(Boolean);
  const meanErr = errs.length ? errs.reduce((a, e) => a + e.mean, 0) / errs.length : NaN;
  const worstErr = errs.length ? Math.max.apply(null, errs.map((e) => e.worst)) : NaN;

  // An opening candidate counts as found if it lands within 0.8 m of a real
  // doorway centre — generous, because the sensor's beam is ~30 degrees wide.
  const tOpen = trueOpenings(surfaces);
  const candidates = (recon.openings || []).filter((o) => o.confidence > 0.2);
  let found = 0;
  for (const t of tOpen) {
    if (candidates.some((c) => Math.hypot(c.x - t.x, c.y - t.y) < 0.8)) found++;
  }

  const st = map.publicStats();
  console.log(
    id.padEnd(12)
    + String(sim.pulseCount).padStart(6)
    + String(st.detections).padStart(6)
    + String(st.cloudPoints).padStart(8)
    + String(recon.segments.length).padStart(6)
    + (recon.stats.totalLength || 0).toFixed(1).padStart(8)
    + String(recon.corners.length).padStart(8)
    + String(candidates.length).padStart(6)
    + (recon.confidence * 100).toFixed(0).padStart(6) + '%'
    + ((Number.isNaN(meanErr) ? '--' : meanErr.toFixed(2) + ' / ' + worstErr.toFixed(2)) + ' m').padStart(21)
    + (found + '/' + tOpen.length).padStart(11)
  );

  if (!Number.isNaN(meanErr)) totals.segErr.push(meanErr);
  totals.conf.push(recon.confidence);
  totals.openFound += found;
  totals.openTrue += tOpen.length;
  sim.stop();
}

const avgErr = totals.segErr.reduce((a, b) => a + b, 0) / Math.max(1, totals.segErr.length);
const avgConf = totals.conf.reduce((a, b) => a + b, 0) / Math.max(1, totals.conf.length);
console.log('\nmean surface error across scenarios: ' + (avgErr * 100).toFixed(1) + ' cm');
console.log('mean reconstruction confidence:      ' + (avgConf * 100).toFixed(0) + '%');
console.log('doorways proposed as candidates:     ' + totals.openFound + ' of ' + totals.openTrue);
console.log('\nSurface error is the distance from each fitted line to the nearest real wall,');
console.log('sampled along the fit. It is not a survey accuracy claim: the sensor has a ~30');
console.log('degree beam, so a "wall" here is an inferred boundary, not a measured plane.');
