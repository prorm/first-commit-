/**
 * Range calibration.
 *
 * What it corrects: the fixed part of the measurement error — audio I/O
 * latency, the speaker-to-microphone offset in the chassis, and the assumed
 * speed of sound at the room's actual temperature.  Those show up as a
 * consistent scale-and-offset error, which one known distance can remove.
 *
 * What it does NOT correct, and the UI says so: beam-width ambiguity, which
 * surface in the beam produced the echo, per-material reflectivity, or the
 * classifier's accuracy.  A calibrated device is still a prototype sensor.
 *
 * Persisted in sessionStorage for the browser session, per the brief.
 */
const STORE_KEY = 'sentryshield.calibration.v1';

export class Calibrator {
  constructor(opts = {}) {
    this.onUpdate = opts.onUpdate || (() => {});
    this.targetSamples = opts.targetSamples || 20;
    this.knownDistanceM = 1.0;
    this.samples = [];
    this.active = false;
    this.result = this.load();
  }

  /** Begin collecting; `known` is the true distance to the chosen wall. */
  begin(knownDistanceM) {
    this.knownDistanceM = Number(knownDistanceM) || 1.0;
    this.samples = [];
    this.active = true;
    this.emit();
    return { active: true, knownDistanceM: this.knownDistanceM, target: this.targetSamples };
  }

  /** Feed a raw (uncalibrated) measured range during collection. */
  addSample(rangeM, confidence) {
    if (!this.active) return null;
    // Only confident, plausible samples count — calibrating against a noise
    // peak would bake an error in permanently.
    if (!(rangeM > 0.15) || rangeM > 6) return this.progress();
    if (confidence != null && confidence < 0.25) return this.progress();
    // Reject anything more than 60 % away from the declared distance: that is
    // a different surface, not a calibration sample.
    if (Math.abs(rangeM - this.knownDistanceM) > this.knownDistanceM * 0.6 + 0.3) return this.progress();

    this.samples.push(rangeM);
    if (this.samples.length >= this.targetSamples) return this.finish();
    return this.progress();
  }

  progress() {
    const p = {
      active: this.active,
      collected: this.samples.length,
      target: this.targetSamples,
      fraction: this.samples.length / this.targetSamples,
      knownDistanceM: this.knownDistanceM,
    };
    this.emit();
    return p;
  }

  /**
   * Compute the correction.  The median is used rather than the mean: one
   * stray echo off a door frame should not move the answer.
   */
  finish() {
    this.active = false;
    if (this.samples.length < 5) {
      this.emit();
      return { ok: false, error: 'not enough usable samples (' + this.samples.length + ')' };
    }
    const sorted = this.samples.slice().sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)];
    const mean = sorted.reduce((a, b) => a + b, 0) / sorted.length;
    const sd = Math.sqrt(sorted.reduce((a, b) => a + (b - mean) ** 2, 0) / sorted.length);

    const factor = this.knownDistanceM / median;
    const offsetM = 0;
    const plausible = factor > 0.6 && factor < 1.6;

    this.result = {
      ok: plausible,
      factor: plausible ? round3(factor) : 1,
      offsetM,
      knownDistanceM: this.knownDistanceM,
      measuredMedian: round3(median),
      spreadM: round3(sd),
      samples: sorted.length,
      at: Date.now(),
      // A wide spread means the wall was not the only thing in the beam.
      quality: sd < 0.03 ? 'good' : sd < 0.08 ? 'fair' : 'poor',
      warning: plausible ? (sd > 0.08 ? 'Sample spread was ' + (sd * 100).toFixed(0) + ' cm — the beam may be seeing more than one surface. Consider recalibrating closer to a flat wall.' : null)
        : 'Correction factor ' + factor.toFixed(2) + ' is outside the plausible range; calibration was not applied. Check the distance you entered.',
    };
    this.save();
    this.emit();
    return this.result;
  }

  cancel() {
    this.active = false;
    this.samples = [];
    this.emit();
  }

  clear() {
    this.result = null;
    try { sessionStorage.removeItem(STORE_KEY); } catch (e) { /* ignore */ }
    this.emit();
  }

  get factor() { return this.result && this.result.ok ? this.result.factor : 1; }
  get offsetM() { return this.result && this.result.ok ? this.result.offsetM : 0; }
  get isCalibrated() { return !!(this.result && this.result.ok); }

  save() {
    try { sessionStorage.setItem(STORE_KEY, JSON.stringify(this.result)); } catch (e) { /* private mode */ }
  }

  load() {
    try {
      const raw = sessionStorage.getItem(STORE_KEY);
      return raw ? JSON.parse(raw) : null;
    } catch (e) { return null; }
  }

  status() {
    return {
      calibrated: this.isCalibrated,
      label: this.isCalibrated ? 'ACTIVE' : 'REQUIRED',
      factor: this.factor,
      offsetM: this.offsetM,
      result: this.result,
      active: this.active,
      collected: this.samples.length,
      target: this.targetSamples,
      // Stated up front so nobody over-reads a green "ACTIVE" badge.
      scope: 'Corrects fixed latency, speaker-mic offset and speed-of-sound assumptions. It does not correct beam-width ambiguity, which surface returned the echo, or classifier accuracy.',
    };
  }

  emit() { this.onUpdate(this.status()); }
}

function round3(v) { return Math.round(v * 1000) / 1000; }
