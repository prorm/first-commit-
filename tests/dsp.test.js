/**
 * Live DSP pipeline tests.
 *
 * Synthesises raw *audio* (transmit leakage + delayed echoes + noise), pushes
 * it through the real pipeline, and checks the recovered range.  This exercises
 * the same code the phone runs, so the matched filter, gate, clutter
 * subtraction, CFAR and tracker can be verified without a phone or a room.
 */
const test = require('node:test');
const assert = require('node:assert');
const { createRequire } = require('node:module');

const {
  DetectionPipeline, AlphaBetaTracker, FFT_SIZE, SPEED_OF_SOUND, CHIRP_AMP,
} = require('../public/phone/dsp/pipeline.mjs');

const requireCjs = createRequire(__filename);
const EchoNet = requireCjs('../src/classifier/echonet_weights.js');

const FS = 48000;

/** Deterministic uniform RNG. */
function lcg(seed) {
  let s = (seed || 1) & 0x7fffffff;
  return () => ((s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
}

/**
 * Build one RX window: the chirp leaks directly into the mic, then echoes
 * arrive delayed by 2*range/c.  The window is arranged so the leakage sits
 * well inside it, as it does on a real device.
 */
function makeRx(pipeline, echoes, opts = {}) {
  const rng = opts.rng || lcg(99);
  const rx = new Float32Array(FFT_SIZE);
  const chirp = pipeline.chirp;
  const leakAt = opts.leakAt != null ? opts.leakAt : 200;
  const leakAmp = opts.leakAmp != null ? opts.leakAmp : 0.08;

  // Direct path (speaker -> mic through the chassis).
  for (let i = 0; i < chirp.length; i++) {
    const j = leakAt + i;
    if (j < FFT_SIZE) rx[j] += chirp[i] * (leakAmp / CHIRP_AMP);
  }
  // Echoes.
  for (const e of echoes) {
    const delay = Math.round((2 * e.range / SPEED_OF_SOUND) * FS);
    for (let i = 0; i < chirp.length; i++) {
      const j = leakAt + delay + i;
      if (j < FFT_SIZE) rx[j] += chirp[i] * (e.amp / CHIRP_AMP);
    }
  }
  // Broadband noise floor.
  const nAmp = opts.noise != null ? opts.noise : 3e-4;
  for (let i = 0; i < FFT_SIZE; i++) rx[i] += (rng() * 2 - 1) * nAmp;
  return rx;
}

/** Run enough pulses for the clutter baseline to settle, then measure. */
function settleAndMeasure(pipeline, echoes, opts = {}) {
  const rng = lcg(opts.seed || 4242);
  // Warm-up pulses contain only the static scene, so a moving target is not
  // averaged into the baseline (which is exactly how it behaves in the field).
  const staticEchoes = opts.staticEchoes || [];
  let t = 1000;
  for (let i = 0; i < 12; i++) {
    pipeline.process(makeRx(pipeline, staticEchoes, { rng, noise: opts.noise }), { t });
    t += 50;
  }
  let last = null;
  const n = opts.pulses || 8;
  for (let i = 0; i < n; i++) {
    last = pipeline.process(makeRx(pipeline, echoes, { rng, noise: opts.noise }), {
      t, classifier: opts.classifier,
    });
    t += 50;
  }
  return last;
}

test('reference chirp matches the specified signal parameters', () => {
  const p = new DetectionPipeline(FS);
  assert.equal(p.chirp.length, Math.round(0.015 * FS), 'chirp is 15 ms');
  // Hann window: both ends near zero, peak near 0.6 FS in the middle.
  assert.ok(Math.abs(p.chirp[0]) < 1e-6);
  assert.ok(Math.abs(p.chirp[p.chirp.length - 1]) < 1e-3);
  let peak = 0;
  for (const v of p.chirp) peak = Math.max(peak, Math.abs(v));
  assert.ok(peak > 0.55 && peak <= 0.6001, 'TX amplitude is 0.6 FS, got ' + peak);

  // Instantaneous frequency should sweep 17.5 -> 22 kHz. Measure by counting
  // zero crossings in the first and last third of the chirp.
  const third = Math.floor(p.chirp.length / 3);
  const rate = (a, b) => {
    let z = 0;
    for (let i = a + 1; i < b; i++) if ((p.chirp[i - 1] < 0) !== (p.chirp[i] < 0)) z++;
    return (z / 2) * (FS / (b - a));
  };
  const fEarly = rate(2, third);
  const fLate = rate(p.chirp.length - third, p.chirp.length - 2);
  assert.ok(fEarly > 17000 && fEarly < 19500, 'early frequency ~17.5-19 kHz, got ' + fEarly.toFixed(0));
  assert.ok(fLate > 20500 && fLate < 22500, 'late frequency ~21-22 kHz, got ' + fLate.toFixed(0));
});

test('recovers a known single-target range to within 5 cm', () => {
  for (const truth of [0.6, 1.0, 1.82, 2.5, 3.2]) {
    const p = new DetectionPipeline(FS);
    const res = settleAndMeasure(p, [{ range: truth, amp: 4e-3 }]);
    assert.ok(res.detection, 'should detect a target at ' + truth + ' m (' + (res.diagnostics.reason || '') + ')');
    const err = Math.abs(res.detection.range_m - truth);
    assert.ok(err < 0.05, 'range error at ' + truth + ' m was ' + err.toFixed(3) + ' m');
    assert.ok(res.detection.cfar_pass, 'a 4e-3 echo should clear CFAR at ' + truth + ' m');
    assert.ok(res.detection.snr_db > 10, 'SNR should be healthy, got ' + res.detection.snr_db.toFixed(1));
  }
});

test('direct-path gate keeps the transmit leakage out of the detections', () => {
  const p = new DetectionPipeline(FS);
  // Leakage only: no target at all.  A working gate means no detection near 0 m.
  const res = settleAndMeasure(p, [], { noise: 2e-4 });
  if (res.detection) {
    assert.ok(res.detection.range_m > 0.28, 'nothing should be reported inside the gate, got ' + res.detection.range_m);
  }
  assert.ok(p.gateSamples === Math.round(0.002 * FS), 'gate is +-2 ms');
  assert.ok(p.nLo >= p.gateSamples, 'search starts past the gate');
});

test('clutter subtraction suppresses a static echo and keeps a new one', () => {
  const p = new DetectionPipeline(FS);
  const staticEcho = { range: 1.2, amp: 5e-3 };
  const rng = lcg(7);
  let t = 0;
  // Learn a scene containing the static echo.
  for (let i = 0; i < 60; i++) {
    p.process(makeRx(p, [staticEcho], { rng }), { t });
    t += 50;
  }
  const baselineAt = p.baseline[Math.round((2 * 1.2 / SPEED_OF_SOUND) * FS)];
  assert.ok(baselineAt > 1e-4, 'the static echo should be learned into the baseline');

  // Now introduce a closer target; it should win despite being weaker.
  const res = p.process(makeRx(p, [staticEcho, { range: 0.75, amp: 2.5e-3 }], { rng }), { t });
  assert.ok(res.detection, 'new target should be detected');
  assert.ok(Math.abs(res.detection.range_m - 0.75) < 0.08,
    'should report the new target at 0.75 m, got ' + res.detection.range_m.toFixed(2));
});

test('tracker rejects a single-frame jump but accepts a persistent one', () => {
  const tr = new AlphaBetaTracker({ persistFrames: 3, maxJump: 0.8 });
  let t = 0;
  for (let i = 0; i < 6; i++) { tr.update(2.0, t); t += 50; }
  assert.ok(Math.abs(tr.range - 2.0) < 0.05);

  const spike = tr.update(0.5, (t += 50));
  assert.equal(spike.accepted, false, 'a one-frame 1.5 m jump must be rejected');
  assert.ok(Math.abs(tr.range - 2.0) < 0.1, 'range must not follow the spike');

  // The same new range three times running is a real target change.
  tr.update(0.5, (t += 50));
  const third = tr.update(0.5, (t += 50));
  assert.equal(third.accepted, true, 'a persistent jump should be adopted');
  assert.ok(Math.abs(tr.range - 0.5) < 0.1);
});

test('tracker signs velocity positive when closing', () => {
  const tr = new AlphaBetaTracker();
  let t = 0;
  let r = 3.0;
  let last = null;
  // Approach at 0.5 m/s, 20 Hz -> 2.5 cm per pulse.
  for (let i = 0; i < 40; i++) {
    last = tr.update(r, t);
    r -= 0.025;
    t += 50;
  }
  assert.ok(last.velocity > 0.2, 'closing velocity should be positive, got ' + last.velocity.toFixed(2));
  assert.ok(Math.abs(last.velocity - 0.5) < 0.25, 'should approximate 0.5 m/s, got ' + last.velocity.toFixed(2));

  const tr2 = new AlphaBetaTracker();
  t = 0; r = 1.0;
  for (let i = 0; i < 40; i++) { last = tr2.update(r, t); r += 0.025; t += 50; }
  assert.ok(last.velocity < -0.2, 'receding velocity should be negative, got ' + last.velocity.toFixed(2));
});

test('classifier window is 64 samples, peak-centred and peak-normalised', () => {
  const p = new DetectionPipeline(FS);
  const res = settleAndMeasure(p, [{ range: 1.5, amp: 5e-3 }], { classifier: EchoNet });
  assert.ok(res.detection);
  const win = res.diagnostics.window;
  assert.ok(win, 'a window should be extracted');
  assert.equal(win.length, 64);
  let mx = 0, mxAt = 0;
  for (let i = 0; i < 64; i++) if (win[i] > mx) { mx = win[i]; mxAt = i; }
  assert.ok(Math.abs(mx - 1) < 1e-6, 'window must be peak-normalised to 1, got ' + mx);
  assert.ok(Math.abs(mxAt - 32) <= 1, 'peak must sit at index 32, got ' + mxAt);
  for (let i = 0; i < 64; i++) assert.ok(win[i] >= 0, 'rectified profile must be non-negative');
});

test('EchoNet runs on live-pipeline windows and calls a hard wall WALL', () => {
  const p = new DetectionPipeline(FS);
  // A single strong specular return is the clearest WALL case there is; the
  // model's WALL recall is its strongest (91 %), so this is a fair check.
  const res = settleAndMeasure(p, [{ range: 1.4, amp: 8e-3 }], { classifier: EchoNet, pulses: 10 });
  assert.ok(res.detection);
  assert.ok(Array.isArray(res.detection.classProbs));
  const sum = res.detection.classProbs.reduce((a, b) => a + b, 0);
  assert.ok(Math.abs(sum - 1) < 1e-3, 'probabilities must sum to 1');
  assert.equal(res.detection.obstacleClass, 'WALL',
    'a clean single specular echo should classify as WALL, got ' + res.detection.obstacleClass);
  assert.ok(res.detection.classConfidence > 0.4);
});

test('calibration scales the reported range', () => {
  const p = new DetectionPipeline(FS);
  const base = settleAndMeasure(p, [{ range: 2.0, amp: 5e-3 }]);
  assert.ok(base.detection);
  const uncal = base.detection.range_m;

  const p2 = new DetectionPipeline(FS);
  p2.setCalibration(1.1, 0);
  const cal = settleAndMeasure(p2, [{ range: 2.0, amp: 5e-3 }]);
  assert.ok(cal.detection);
  assert.ok(cal.detection.range_m > uncal * 1.05, 'a 1.1 correction factor should raise the range');
  // Out-of-range factors are ignored rather than trusted.
  p2.setCalibration(9.0, 0);
  assert.equal(p2.rangeCorrection, 1.1, 'an implausible factor must be refused');
});

test('pipeline never throws on degenerate input', () => {
  const p = new DetectionPipeline(FS);
  const cases = [
    new Float32Array(FFT_SIZE),                                  // silence
    Float32Array.from({ length: FFT_SIZE }, () => 1),            // clipped DC
    Float32Array.from({ length: FFT_SIZE }, () => Math.random() * 2 - 1),  // full-scale noise
  ];
  for (const rx of cases) {
    for (let i = 0; i < 15; i++) {
      const res = p.process(rx, { t: i * 50, classifier: EchoNet });
      assert.ok(res && 'detection' in res, 'must always return a result object');
      if (res.detection) {
        assert.ok(Number.isFinite(res.detection.range_m));
        assert.ok(Number.isFinite(res.detection.vel_mps));
        assert.ok(res.detection.confidence >= 0 && res.detection.confidence <= 1);
      }
    }
  }
});

test('CFAR holds its false-alarm rate in an empty room', () => {
  // The margin in pipeline.mjs is calibrated against exactly this measurement.
  // Clutter subtraction rectifies at zero, so the textbook exponential-clutter
  // threshold is far too permissive; this test is what pins the real number.
  const p = new DetectionPipeline(FS);
  const rng = lcg(11);
  let t = 0;
  for (let i = 0; i < 25; i++) { p.process(makeRx(p, [], { rng }), { t }); t += 50; }

  let falseAlarms = 0;
  const trials = 120;
  for (let i = 0; i < trials; i++) {
    const res = p.process(makeRx(p, [], { rng }), { t });
    t += 50;
    if (res.detection) falseAlarms++;
  }
  const rate = falseAlarms / trials;
  assert.ok(rate < 0.05, 'empty-room false-alarm rate should be under 5 %, got ' + (rate * 100).toFixed(1) + '%');
});

test('tracks an approaching wall smoothly over a 3 m walk', () => {
  const p = new DetectionPipeline(FS);
  const rng = lcg(808);
  let t = 0;
  for (let i = 0; i < 25; i++) { p.process(makeRx(p, [], { rng }), { t }); t += 50; }

  // Walk 3.2 m -> 0.6 m at 0.5 m/s, 20 Hz.
  const truths = [];
  const measured = [];
  let r = 3.2;
  while (r > 0.6) {
    const res = p.process(makeRx(p, [{ range: r, amp: 3e-3 }], { rng }), { t, classifier: EchoNet });
    if (res.detection) { truths.push(r); measured.push(res.detection.range_m); }
    r -= 0.025;
    t += 50;
  }
  assert.ok(measured.length > 90, 'should hold the track across the approach, got ' + measured.length);

  // Accuracy
  let se = 0;
  for (let i = 0; i < measured.length; i++) se += Math.pow(measured[i] - truths[i], 2);
  const rmse = Math.sqrt(se / measured.length);
  assert.ok(rmse < 0.15, 'RMS range error over the walk should be under 15 cm, got ' + (rmse * 100).toFixed(1) + ' cm');

  // Monotonicity: the trace should descend, with few reversals.
  let reversals = 0;
  for (let i = 1; i < measured.length; i++) if (measured[i] > measured[i - 1] + 0.02) reversals++;
  assert.ok(reversals / measured.length < 0.15,
    'range trace should be near-monotonic while approaching, got ' + reversals + ' reversals in ' + measured.length);
});
