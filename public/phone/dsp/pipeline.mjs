/**
 * Live detection pipeline — matched filter through to a Detection.
 *
 * The chain is the one specified in the project notes, in order, with nothing
 * skipped:
 *
 *   1. matched filter      FFT cross-correlation of the RX window against the
 *                          reference chirp, taken as an analytic signal so the
 *                          output is a smooth envelope
 *   2. range alignment     re-index the envelope so sample 0 is the direct
 *                          path, which is what makes step 3 possible at all
 *   3. direct-path gate    zero the +-2 ms around the transmit leakage
 *   4. clutter subtraction subtract a slow running baseline of static echoes,
 *                          then rectify (this is exactly the profile the
 *                          classifier was trained on)
 *   5. CFAR                smallest-of cell-averaging CFAR with a Kalman-
 *                          smoothed noise floor; strongest peak past the gate
 *   6. tracker             alpha-beta smoothing of range, differentiated for
 *                          velocity, with jump rejection
 *   7. classifier          64-sample peak-centred window -> EchoNet
 *
 * Everything is pre-allocated: at 20 Hz on a phone, per-pulse allocation is
 * what turns a smooth demo into a stuttering one.
 */
import { FFT } from './fft.mjs';
import { surfaceFromProbs } from '../../shared/surfaceclass.mjs';

export const SPEED_OF_SOUND = 343.2;
export const CHIRP_F0 = 17500;
export const CHIRP_F1 = 22000;
export const CHIRP_DURATION = 0.015;
export const CHIRP_AMP = 0.6;
export const FFT_SIZE = 4096;
export const CLASSIFIER_WIN = 64;

/** Detection search span, matching the classifier's training distribution. */
export const R_SEARCH_MIN = 0.30;
export const R_SEARCH_MAX = 3.75;
/** Direct-path gate half-width, seconds. */
export const GATE_HALF_S = 0.002;
/** Clutter baseline time constant, in pulses. */
export const BASELINE_PULSES = 40;

export class DetectionPipeline {
  constructor(sampleRate = 48000, opts = {}) {
    this.fs = sampleRate;
    this.fft = new FFT(FFT_SIZE);

    // Reference chirp and its conjugate spectrum (the matched filter kernel).
    this.chirp = null;
    this.kernelReal = new Float32Array(FFT_SIZE);
    this.kernelImag = new Float32Array(FFT_SIZE);

    // Scratch
    this.rxR = new Float32Array(FFT_SIZE);
    this.rxI = new Float32Array(FFT_SIZE);
    this.zR = new Float32Array(FFT_SIZE);
    this.zI = new Float32Array(FFT_SIZE);
    this.envelope = new Float32Array(FFT_SIZE);

    // Range-aligned buffers: index 0 is the direct path.
    this.profLen = Math.min(FFT_SIZE - 64, Math.ceil((2 * 4.2 / SPEED_OF_SOUND) * this.fs));
    this.aligned = new Float32Array(this.profLen);
    this.baseline = new Float32Array(this.profLen);
    this.profile = new Float32Array(this.profLen);
    this.baselineCount = 0;
    this.window64 = new Float32Array(CLASSIFIER_WIN);

    this.gateSamples = Math.round(GATE_HALF_S * this.fs);
    this.nLo = Math.max(this.gateSamples + 1, Math.round((2 * R_SEARCH_MIN / SPEED_OF_SOUND) * this.fs));
    this.nHi = Math.min(this.profLen - 2, Math.round((2 * R_SEARCH_MAX / SPEED_OF_SOUND) * this.fs));

    // CFAR
    // margin is an empirical multiplier on the textbook threshold, calibrated
    // in tests/dsp.test.js: 0 % false alarms in an empty room while still
    // detecting a 2.5e-4 FS echo 97 % of the time, with 0.4 cm mean error.
    this.cfar = { guard: 28, train: 32, pfa: 1e-4, margin: 10.0 };
    this.noiseFloor = 0;          // Kalman-smoothed noise amplitude
    this.noiseVar = 1e-6;

    this.tracker = new AlphaBetaTracker(opts.tracker);
    this.rangeCorrection = 1.0;   // set by calibration
    this.rangeOffsetM = 0.0;

    this.stats = { pulses: 0, detections: 0, cfarPass: 0, lastDirectVal: 0, lastSnrDb: 0 };
    this.buildReferenceChirp();
  }

  /** The exact TX waveform: Hann-windowed linear FM, 17.5 -> 22 kHz. */
  buildReferenceChirp() {
    const n = Math.round(CHIRP_DURATION * this.fs);
    const chirp = new Float32Array(n);
    const T = CHIRP_DURATION;
    for (let i = 0; i < n; i++) {
      const t = i / this.fs;
      const w = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (n - 1)));
      const phase = 2 * Math.PI * (CHIRP_F0 * t + (0.5 * (CHIRP_F1 - CHIRP_F0) / T) * t * t);
      chirp[i] = CHIRP_AMP * w * Math.sin(phase);
    }
    this.chirp = chirp;

    const tR = new Float32Array(FFT_SIZE);
    const tI = new Float32Array(FFT_SIZE);
    tR.set(chirp);
    this.fft.transform(tR, tI, false);
    let energy = 0;
    for (let i = 0; i < n; i++) energy += chirp[i] * chirp[i];
    // Normalise so a perfectly matched echo of amplitude A peaks at A.
    const norm = energy / CHIRP_AMP || 1;
    for (let k = 0; k < FFT_SIZE; k++) {
      this.kernelReal[k] = tR[k] / norm;
      this.kernelImag[k] = -tI[k] / norm;
    }
    return chirp;
  }

  setCalibration(factor, offsetM) {
    if (Number.isFinite(factor) && factor > 0.5 && factor < 2.0) this.rangeCorrection = factor;
    if (Number.isFinite(offsetM)) this.rangeOffsetM = offsetM;
  }

  resetClutter() {
    this.baseline.fill(0);
    this.baselineCount = 0;
    this.noiseFloor = 0;
  }

  reset() {
    this.resetClutter();
    this.tracker.reset();
    this.stats.pulses = 0;
    this.stats.detections = 0;
    this.stats.cfarPass = 0;
  }

  /**
   * Process one received window.
   *
   * @param {Float32Array} rx  FFT_SIZE samples ending after the echo arrives
   * @param {Object} [opts] { classifier, t }
   * @returns {Object} result with `detection` (or null) plus DSP diagnostics
   */
  process(rx, opts = {}) {
    this.stats.pulses++;

    // --- 1. matched filter, analytic ------------------------------------
    const N = FFT_SIZE;
    const half = N / 2;
    this.rxR.set(rx.subarray(0, N));
    this.rxI.fill(0);
    this.fft.transform(this.rxR, this.rxI, false);

    const { rxR, rxI, zR, zI, kernelReal: kr, kernelImag: ki } = this;
    zR[0] = rxR[0] * kr[0] - rxI[0] * ki[0];
    zI[0] = rxR[0] * ki[0] + rxI[0] * kr[0];
    // Doubling the positive frequencies and zeroing the negative ones makes
    // the IFFT return hilbert(correlation) directly — a smooth envelope,
    // without a separate Hilbert transform.
    for (let k = 1; k < half; k++) {
      zR[k] = 2 * (rxR[k] * kr[k] - rxI[k] * ki[k]);
      zI[k] = 2 * (rxR[k] * ki[k] + rxI[k] * kr[k]);
    }
    zR[half] = rxR[half] * kr[half] - rxI[half] * ki[half];
    zI[half] = rxR[half] * ki[half] + rxI[half] * kr[half];
    for (let k = half + 1; k < N; k++) { zR[k] = 0; zI[k] = 0; }
    this.fft.transform(zR, zI, true);

    let peakVal = 0;
    let directIdx = -1;
    for (let i = 0; i < N; i++) {
      const m = Math.sqrt(zR[i] * zR[i] + zI[i] * zI[i]);
      this.envelope[i] = m;
      if (m > peakVal) { peakVal = m; directIdx = i; }
    }
    this.stats.lastDirectVal = peakVal;

    // The transmit leakage is 20-40 dB above any echo, so the global maximum
    // is the direct path.  If it is not there, the pulse did not land in this
    // window and there is nothing honest to report.
    if (directIdx < 0 || peakVal < 1e-4 || directIdx + this.nLo >= N) {
      this.tracker.miss();
      return this.emptyResult('no direct path', directIdx, peakVal);
    }

    // --- 2. range alignment ---------------------------------------------
    const avail = Math.min(this.profLen, N - directIdx);
    this.aligned.fill(0);
    for (let i = 0; i < avail; i++) this.aligned[i] = this.envelope[directIdx + i];

    // --- 3 + 4. gate, clutter subtraction, rectify -----------------------
    // Sample 0 is the direct path, so the +-2 ms gate is simply i < gate.
    const gate = this.gateSamples;
    const alpha = 1 / BASELINE_PULSES;
    let sawBaseline = this.baselineCount >= 8;
    for (let i = 0; i < this.profLen; i++) {
      const live = i < gate ? 0 : this.aligned[i];
      // Running mean of *envelopes* (not of the complex signal): that keeps
      // the noise floor's positive bias, so subtracting it and rectifying
      // yields the zero-clipped profile EchoNet was trained on.
      if (this.baselineCount < BASELINE_PULSES) {
        this.baseline[i] += (live - this.baseline[i]) / (this.baselineCount + 1);
      } else {
        this.baseline[i] += (live - this.baseline[i]) * alpha;
      }
      const v = live - this.baseline[i];
      this.profile[i] = v > 0 ? v : 0;
    }
    if (this.baselineCount < BASELINE_PULSES * 2) this.baselineCount++;

    if (!sawBaseline) {
      // Still learning the static scene: report the warm-up honestly rather
      // than emitting detections dominated by un-subtracted clutter.
      this.tracker.miss();
      return this.emptyResult('clutter baseline warming up (' + this.baselineCount + '/8)', directIdx, peakVal);
    }

    // --- 5. CFAR ---------------------------------------------------------
    const det = this.cfarDetect(this.profile, this.nLo, Math.min(this.nHi, avail - 2));
    if (!det.idx) {
      this.tracker.miss();
      return this.emptyResult('no peak in search window', directIdx, peakVal, det);
    }
    // A peak that did not clear the CFAR threshold is noise until proven
    // otherwise.  It may *maintain* an established track (the peak is already
    // near the prediction), but it must never *acquire* one — letting noise
    // peaks start tracks is what makes a display jump around an empty room.
    if (!det.pass && this.tracker.range == null) {
      this.tracker.miss();
      return this.emptyResult('peak below CFAR threshold (' + det.snrDb.toFixed(1) + ' dB)', directIdx, peakVal, det);
    }

    const rawRange = (det.idx / this.fs) * SPEED_OF_SOUND * 0.5;
    const range = rawRange * this.rangeCorrection + this.rangeOffsetM;

    // --- 6. tracker ------------------------------------------------------
    const track = this.tracker.update(range, opts.t || performance.now());
    if (!track.accepted) {
      return this.emptyResult('peak rejected by tracker (jump ' + track.jump.toFixed(2) + ' m)', directIdx, peakVal, det);
    }

    // --- 7. classifier ---------------------------------------------------
    // EchoNet decides WALL vs SOFT and nothing else.  Its third head is not
    // used: an opening is the absence of a return, not a texture, so it is
    // recovered geometrically by reconstruct.mjs instead.  See
    // shared/surfaceclass.mjs for why.  The full three-way distribution still
    // rides along in classProbs so the diagnostics page can show the raw model.
    let classProbs = [0, 0, 0];
    let obstacleClass = null;
    let classConfidence = 0;
    let classMargin = 0;
    const win = this.extractWindow(det.idx);
    if (opts.classifier && win) {
      try {
        const out = opts.classifier.forward(win);
        classProbs = Array.from(out.probs);
        const surface = surfaceFromProbs(classProbs);
        obstacleClass = surface.className;
        classConfidence = surface.confidence;
        classMargin = surface.margin;
      } catch (e) {
        // A classifier failure must not cost us the range measurement.
        classProbs = [0, 0, 0];
      }
    }

    this.stats.detections++;
    if (det.pass) this.stats.cfarPass++;
    this.stats.lastSnrDb = det.snrDb;

    // Detection confidence blends how far above the noise the peak sits with
    // how consistently the tracker has been seeing it.  Both are measured.
    const snrTerm = clamp01((det.snrDb - 4) / 24);
    const confidence = clamp01(0.65 * snrTerm + 0.35 * track.support);

    return {
      detection: {
        t: opts.t || Date.now(),
        range_m: track.range,
        vel_mps: track.velocity,
        ttc_s: clampRange(track.range / Math.max(track.velocity, 0.05), 0, 10),
        confidence,
        snr_db: det.snrDb,
        cfar_pass: det.pass,
        obstacleClass,
        classConfidence,
        classProbs,
        classMargin,
      },
      diagnostics: {
        directIdx,
        directVal: peakVal,
        peakIdx: det.idx,
        rawRange,
        noiseFloor: this.noiseFloor,
        snrDb: det.snrDb,
        cfarPass: det.pass,
        trackSupport: track.support,
        window: win,
        reason: null,
      },
    };
  }

  emptyResult(reason, directIdx, directVal, det) {
    const peakIdx = det ? det.idx : -1;
    const rawRange = (peakIdx > 0) ? (peakIdx / this.fs) * SPEED_OF_SOUND * 0.5 : null;
    return {
      detection: null,
      diagnostics: {
        directIdx, directVal,
        peakIdx,
        rawRange,
        noiseFloor: this.noiseFloor,
        snrDb: det ? det.snrDb : 0,
        cfarPass: false,
        reason,
        window: (peakIdx > 0) ? this.extractWindow(peakIdx) : null,
      },
    };
  }

  /**
   * Smallest-of cell-averaging CFAR.
   *
   * Plain CA-CFAR self-masks extended targets — a person spans 60-90 samples,
   * so both training windows can sit on the target.  Taking the smaller of the
   * two sides only needs one clean side to give a valid noise estimate.
   */
  cfarDetect(profile, nLo, nHi) {
    const { guard, train, pfa } = this.cfar;
    if (nHi <= nLo + 2) return { idx: 0, pass: false, snrDb: 0, noise: this.noiseFloor };

    // Robust global noise power over the search window.
    //
    // This floor is not optional.  Clutter subtraction rectifies at zero, so
    // roughly half the noise-only cells are exactly 0; a run of them makes a
    // purely local noise estimate collapse toward zero and then *every* peak
    // clears the threshold.  One sigma-clipping pass gives a noise scale that
    // the target itself cannot inflate, and flooring the local estimate with
    // it keeps the CFAR adaptive without letting it divide by nothing.
    let sum = 0;
    let cnt = 0;
    for (let i = nLo; i <= nHi; i++) { sum += profile[i] * profile[i]; cnt++; }
    const mean1 = cnt ? sum / cnt : 0;
    let sum2 = 0;
    let cnt2 = 0;
    const clip = 4 * mean1;
    for (let i = nLo; i <= nHi; i++) {
      const p2 = profile[i] * profile[i];
      if (p2 < clip) { sum2 += p2; cnt2++; }
    }
    const globalNoise = Math.max(cnt2 > 20 ? sum2 / cnt2 : mean1, 1e-26);

    let bestIdx = 0;
    let bestVal = 0;
    let bestNoise = 1e-12;
    let bestPass = false;
    let fallbackIdx = 0;
    let fallbackVal = 0;
    let fallbackNoise = 1e-12;

    for (let i = nLo; i <= nHi; i++) {
      const v = profile[i];
      if (v <= 0) continue;
      // Local maximum test first: it is cheap and rejects most cells.
      if (!(v >= profile[i - 1] && v > profile[i + 1])) continue;

      const lead = meanPower(profile, i - guard - train, i - guard, this.gateSamples);
      const lag = meanPower(profile, i + guard + 1, i + guard + 1 + train, this.gateSamples);
      let noise;
      let nEff;
      if (lead.n >= 8 && lag.n >= 8) {
        noise = Math.min(lead.mean, lag.mean);
        nEff = lead.n + lag.n;
      } else if (lead.n >= 8) { noise = lead.mean; nEff = lead.n; }
      else if (lag.n >= 8) { noise = lag.mean; nEff = lag.n; }
      else continue;
      // The local estimate may only raise the bar, never lower it.  Near
      // clutter it correctly rises above the window average; over rectified
      // noise-only cells it would otherwise sag toward zero.
      if (!(noise > globalNoise)) noise = globalNoise;

      // Square-law CA-CFAR threshold factor, times an empirical margin.
      // The textbook factor assumes exponential clutter statistics; the
      // rectified profile is heavier-tailed than that, so the margin is
      // calibrated against measured false-alarm rate on noise-only pulses
      // (see tests/dsp.test.js "CFAR holds its false-alarm rate").
      const alpha = nEff * (Math.pow(pfa, -1 / nEff) - 1) * this.cfar.margin;
      const pass = v * v > alpha * noise;

      if (pass && v > bestVal) { bestVal = v; bestIdx = i; bestNoise = noise; bestPass = true; }
      if (!bestPass && v > fallbackVal) { fallbackVal = v; fallbackIdx = i; fallbackNoise = noise; }
    }

    const idx = bestPass ? bestIdx : fallbackIdx;
    const val = bestPass ? bestVal : fallbackVal;
    const noise = bestPass ? bestNoise : fallbackNoise;
    if (!idx) return { idx: 0, pass: false, snrDb: 0, noise: this.noiseFloor };

    // Kalman-ish smoothing of the noise floor: a single noisy estimate makes
    // the SNR readout jitter, which reads as instability even when the range
    // is steady.
    const measured = Math.sqrt(noise);
    if (this.noiseFloor === 0) this.noiseFloor = measured;
    else {
      const k = this.noiseVar / (this.noiseVar + 4e-7);
      this.noiseFloor += k * (measured - this.noiseFloor);
      this.noiseVar = (1 - k) * this.noiseVar + 2e-8;
    }
    const snrDb = 20 * Math.log10(Math.max(val, 1e-30) / Math.max(this.noiseFloor, 1e-30));

    return { idx, pass: bestPass, snrDb: Math.min(60, snrDb), noise: this.noiseFloor, val };
  }

  /** 64 samples of the rectified profile centred on the peak, peak-normalised. */
  extractWindow(peakIdx) {
    const half = CLASSIFIER_WIN / 2;
    if (peakIdx - half < 0 || peakIdx + half >= this.profLen) return null;
    let mx = 0;
    for (let i = 0; i < CLASSIFIER_WIN; i++) {
      const v = this.profile[peakIdx - half + i];
      this.window64[i] = v;
      if (v > mx) mx = v;
    }
    if (mx <= 0) return null;
    for (let i = 0; i < CLASSIFIER_WIN; i++) this.window64[i] /= mx;
    return this.window64;
  }

  /** Down-sampled profile for the waterfall / A-scope displays. */
  profileForDisplay(bins = 160, maxRangeM = 4.0) {
    const out = new Float32Array(bins);
    const hi = Math.min(this.profLen, Math.round((2 * maxRangeM / SPEED_OF_SOUND) * this.fs));
    const per = hi / bins;
    for (let b = 0; b < bins; b++) {
      const s = Math.floor(b * per);
      const e = Math.max(s + 1, Math.floor((b + 1) * per));
      let m = 0;
      for (let i = s; i < e && i < this.profLen; i++) if (this.profile[i] > m) m = this.profile[i];
      out[b] = m;
    }
    return out;
  }
}

function meanPower(profile, lo, hi, gate) {
  let sum = 0;
  let n = 0;
  const start = Math.max(lo, gate);
  const end = Math.min(hi, profile.length);
  for (let i = start; i < end; i++) {
    const v = profile[i];
    sum += v * v;
    n++;
  }
  return { mean: n ? sum / n : 0, n };
}

/**
 * Alpha-beta tracker.
 *
 * Smooths range, differentiates it for closing velocity, and refuses peak
 * jumps larger than `maxJump` unless the new range persists for
 * `persistFrames` consecutive pulses — that is what stops a single strong
 * clutter return from yanking the displayed range across the room.
 */
export class AlphaBetaTracker {
  constructor(opts = {}) {
    this.alpha = opts.alpha != null ? opts.alpha : 0.35;
    // beta ~ alpha^2/(2 - alpha) is the critically-damped choice: range
    // settles without the overshoot that makes a displayed range wobble.
    this.beta = opts.beta != null ? opts.beta : 0.074;
    this.maxJump = opts.maxJump != null ? opts.maxJump : 0.8;
    this.persistFrames = opts.persistFrames != null ? opts.persistFrames : 3;
    // After this many consecutive rejections/dropouts the track has lost lock.
    // Holding on longer strands the display on a stale range, which during a
    // demo looks exactly like a frozen app.
    this.maxMisses = opts.maxMisses != null ? opts.maxMisses : 5;
    this.reset();
  }

  reset() {
    this.range = null;
    this.velocity = 0;
    this.lastT = 0;
    this.hits = 0;
    this.misses = 0;
    this.candidate = null;
    this.candidateCount = 0;
  }

  /** No usable peak this pulse. */
  miss() {
    this.misses++;
    if (this.range != null && this.misses >= this.maxMisses) this.reset();
  }

  /** Adopt a measurement outright: new lock, no smoothing history. */
  acquire(measured, t, support) {
    this.range = measured;
    this.velocity = 0;
    this.lastT = t;
    this.hits = Math.max(1, this.hits);
    this.misses = 0;
    this.candidate = null;
    this.candidateCount = 0;
    return { accepted: true, range: measured, velocity: 0, support: support, jump: 0 };
  }

  update(measured, t) {
    if (this.range == null) return this.acquire(measured, t, 0.2);

    // Lock lost while we were rejecting: take the current measurement at face
    // value rather than dragging the old track toward it over half a second.
    if (this.misses >= this.maxMisses) {
      this.hits = 1;
      return this.acquire(measured, t, 0.2);
    }

    const dt = Math.max(0.005, Math.min(0.5, (t - this.lastT) / 1000));
    // `velocity` is closing speed, so range decreases as it increases.
    const predicted = this.range - this.velocity * dt;
    const jump = Math.abs(measured - predicted);

    if (jump > this.maxJump) {
      // Either a real new target or a clutter spike. Require the new range to
      // repeat before believing it; a spike will not.
      if (this.candidate != null && Math.abs(measured - this.candidate) < this.maxJump * 0.6) {
        this.candidateCount++;
        this.candidate = 0.5 * (this.candidate + measured);
      } else {
        this.candidate = measured;
        this.candidateCount = 1;
      }
      if (this.candidateCount >= this.persistFrames) {
        this.hits = this.persistFrames;
        return this.acquire(this.candidate, t, 0.3);
      }
      this.misses++;
      return { accepted: false, range: this.range, velocity: this.velocity, support: this.support(), jump };
    }

    this.candidate = null;
    this.candidateCount = 0;
    const residual = measured - predicted;
    this.range = predicted + this.alpha * residual;
    // Range falling => closing velocity rising, hence the sign.
    this.velocity -= (this.beta * residual) / dt;
    this.velocity = clampRange(this.velocity, -4, 4);
    this.lastT = t;
    this.hits++;
    this.misses = 0;
    return { accepted: true, range: this.range, velocity: this.velocity, support: this.support(), jump };
  }

  support() { return clamp01(this.hits / 8); }
}

function clamp01(v) { return v < 0 ? 0 : v > 1 ? 1 : v; }
function clampRange(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }
