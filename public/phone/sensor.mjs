/**
 * Acoustic sensor — audio in, Detection out.
 *
 * This is the ONLY module that touches raw audio.  Everything downstream
 * (phone UI, tones, haptics, the WebSocket uplink, the command center) consumes
 * the Detection objects this emits and nothing else.  That boundary is what
 * makes the sensor swappable: if Chrome's audio path fails on a given device,
 * this module can be replaced by a native bridge emitting the same Detections
 * and not one line of UI changes.
 *
 * Capture strategy, in order of preference:
 *   1. AudioWorklet — runs on the audio thread, no main-thread jank
 *   2. ScriptProcessor — deprecated but works everywhere, and is the path the
 *      original prototype proved on the target phone
 * Whichever succeeds writes into the same ring buffer; the DSP does not know
 * or care which one is in use.
 */
import { DetectionPipeline, FFT_SIZE, CHIRP_DURATION } from './dsp/pipeline.mjs';

const RING_SIZE = 16384;
/** Delay from transmit to processing: chirp length + max round trip + margin. */
const PROCESS_DELAY_MS = 28;

export class AcousticSensor {
  /**
   * @param {Object} opts
   * @param {(det:Object, diag:Object)=>void} opts.onDetection  every pulse (det may be null)
   * @param {(msg:string, level:string)=>void} [opts.onLog]
   */
  constructor(opts = {}) {
    this.onDetection = opts.onDetection || (() => {});
    this.onLog = opts.onLog || (() => {});
    this.classifier = opts.classifier || null;

    this.audioCtx = null;
    this.micStream = null;
    this.micSource = null;
    this.workletNode = null;
    this.scriptNode = null;
    this.sinkGain = null;
    this.chirpBuffer = null;
    this.pipeline = null;

    this.ring = new Float32Array(RING_SIZE);
    this.ringHead = 0;
    this.rxWindow = new Float32Array(FFT_SIZE);

    this.running = false;
    this.pulseTimer = null;
    this.rateHz = 20;
    this.captureMode = null;       // 'audioworklet' | 'scriptprocessor'
    this.sampleRate = 48000;
    this.channelCount = 1;
    this.lastError = null;
    this.pulseCount = 0;
    this.detectionCount = 0;
    this.lastPulseAt = 0;
    this.trackLoad = { sum: 0, n: 0, max: 0 };
  }

  // -------------------------------------------------------------------------
  // Capability detection — run before asking for anything
  // -------------------------------------------------------------------------
  static capabilities() {
    const secure = typeof isSecureContext !== 'undefined' ? isSecureContext : false;
    const hasMediaDevices = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
    const AC = typeof window !== 'undefined' ? (window.AudioContext || window.webkitAudioContext) : null;
    return {
      secureContext: secure,
      getUserMedia: hasMediaDevices,
      audioContext: !!AC,
      audioWorklet: !!(AC && AC.prototype && 'audioWorklet' in AC.prototype),
      scriptProcessor: !!(AC && AC.prototype && 'createScriptProcessor' in AC.prototype),
      // A microphone on plain HTTP is blocked by Chrome; simulation still works.
      liveAudioPossible: secure && hasMediaDevices && !!AC,
      reason: !secure
        ? 'not a secure context — mobile Chrome blocks the microphone on plain HTTP'
        : !hasMediaDevices ? 'navigator.mediaDevices unavailable'
          : !AC ? 'AudioContext unavailable' : null,
    };
  }

  /** Ask the browser what it will actually grant, without starting a scan. */
  static async probePermissions() {
    const out = { microphone: 'unknown', reason: null };
    try {
      if (navigator.permissions && navigator.permissions.query) {
        const st = await navigator.permissions.query({ name: 'microphone' });
        out.microphone = st.state;         // granted | denied | prompt
      }
    } catch (e) {
      out.reason = e.message;              // Firefox/Safari may not support this query
    }
    return out;
  }

  // -------------------------------------------------------------------------
  // Start / stop
  // -------------------------------------------------------------------------
  async start(opts = {}) {
    if (this.running) return { ok: true, already: true };
    const caps = AcousticSensor.capabilities();
    if (!caps.liveAudioPossible) {
      this.lastError = caps.reason;
      return { ok: false, error: caps.reason, capabilities: caps };
    }

    try {
      const AC = window.AudioContext || window.webkitAudioContext;
      // 48 kHz is requested explicitly; the chirp band reaches 22 kHz, which
      // 44.1 kHz cannot carry with any margin.
      this.audioCtx = new AC({ sampleRate: 48000, latencyHint: 'interactive' });
      // Autoplay policy: a context created outside a gesture starts suspended.
      if (this.audioCtx.state === 'suspended') await this.audioCtx.resume();
      this.sampleRate = this.audioCtx.sampleRate;
      if (Math.abs(this.sampleRate - 48000) > 1) {
        this.onLog('Device gave ' + this.sampleRate + ' Hz, not 48 kHz — ranges stay correct (the pipeline uses the real rate) but the top of the chirp band may be attenuated.', 'warn');
      }

      this.pipeline = new DetectionPipeline(this.sampleRate);
      this.buildChirpBuffer();

      // These three constraints MUST be nested inside `audio`. Chrome silently
      // ignores them at the top level, and with AGC or noise suppression on,
      // the echo is processed away before the DSP ever sees it.
      const constraints = {
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
          channelCount: opts.stereo ? 2 : 1,
          sampleRate: 48000,
        },
        video: false,
      };
      this.micStream = await navigator.mediaDevices.getUserMedia(constraints);

      const track = this.micStream.getAudioTracks()[0];
      const settings = track ? track.getSettings() : {};
      this.channelCount = settings.channelCount || 1;
      this.trackSettings = settings;
      // Report what was actually granted, not what was asked for.
      const applied = [
        'echoCancellation=' + settings.echoCancellation,
        'noiseSuppression=' + settings.noiseSuppression,
        'autoGainControl=' + settings.autoGainControl,
        'channels=' + this.channelCount,
        'rate=' + (settings.sampleRate || this.sampleRate),
      ].join(' ');
      this.onLog('Mic granted: ' + applied, 'success');
      if (settings.echoCancellation || settings.autoGainControl || settings.noiseSuppression) {
        this.onLog('Device refused to disable mic processing — echo amplitudes will be unreliable.', 'warn');
      }

      this.micSource = this.audioCtx.createMediaStreamSource(this.micStream);
      await this.attachCapture();

      this.running = true;
      this.pulseCount = 0;
      this.detectionCount = 0;
      this.startPulseLoop(opts.rateHz || this.rateHz);
      this.onLog('Acoustic scan active at ' + this.rateHz + ' Hz via ' + this.captureMode + '.', 'success');
      return { ok: true, captureMode: this.captureMode, sampleRate: this.sampleRate, channelCount: this.channelCount, settings };
    } catch (e) {
      this.lastError = e.message;
      const friendly = describeAudioError(e);
      this.onLog('Scan failed: ' + friendly, 'error');
      await this.stop();
      return { ok: false, error: friendly, raw: e.name + ': ' + e.message };
    }
  }

  /** AudioWorklet if available, ScriptProcessor otherwise — same ring buffer. */
  async attachCapture() {
    // The worklet is inlined as a blob so there is no extra file to fail to
    // load from a phone over a self-signed certificate.
    if (this.audioCtx.audioWorklet) {
      try {
        const src = `
          class CaptureProcessor extends AudioWorkletProcessor {
            constructor() {
              super();
              this.buf = new Float32Array(2048);
              this.n = 0;
            }
            process(inputs) {
              const ch = inputs[0] && inputs[0][0];
              if (ch) {
                for (let i = 0; i < ch.length; i++) {
                  this.buf[this.n++] = ch[i];
                  if (this.n === this.buf.length) {
                    this.port.postMessage(this.buf.slice(0));
                    this.n = 0;
                  }
                }
              }
              return true;
            }
          }
          registerProcessor('sentryshield-capture', CaptureProcessor);
        `;
        const url = URL.createObjectURL(new Blob([src], { type: 'application/javascript' }));
        await this.audioCtx.audioWorklet.addModule(url);
        URL.revokeObjectURL(url);
        this.workletNode = new AudioWorkletNode(this.audioCtx, 'sentryshield-capture');
        this.workletNode.port.onmessage = (ev) => this.writeRing(ev.data);
        this.micSource.connect(this.workletNode);
        // A worklet with no output still needs a graph path to stay scheduled.
        this.sinkGain = this.audioCtx.createGain();
        this.sinkGain.gain.value = 0;
        this.workletNode.connect(this.sinkGain);
        this.sinkGain.connect(this.audioCtx.destination);
        this.captureMode = 'audioworklet';
        return;
      } catch (e) {
        this.onLog('AudioWorklet unavailable (' + e.message + '); falling back to ScriptProcessor.', 'warn');
        this.workletNode = null;
      }
    }

    // Fallback: the path the original prototype proved on this phone.
    this.scriptNode = this.audioCtx.createScriptProcessor(2048, 1, 1);
    this.scriptNode.onaudioprocess = (e) => this.writeRing(e.inputBuffer.getChannelData(0));
    this.sinkGain = this.audioCtx.createGain();
    this.sinkGain.gain.value = 0;
    this.micSource.connect(this.scriptNode);
    this.scriptNode.connect(this.sinkGain);
    this.sinkGain.connect(this.audioCtx.destination);
    this.captureMode = 'scriptprocessor';
  }

  writeRing(chunk) {
    if (!chunk) return;
    const n = chunk.length;
    let h = this.ringHead;
    for (let i = 0; i < n; i++) {
      this.ring[h] = chunk[i];
      h = (h + 1) % RING_SIZE;
    }
    this.ringHead = h;
  }

  buildChirpBuffer() {
    const chirp = this.pipeline.chirp;
    this.chirpBuffer = this.audioCtx.createBuffer(1, chirp.length, this.sampleRate);
    this.chirpBuffer.getChannelData(0).set(chirp);
  }

  transmit() {
    if (!this.audioCtx || !this.chirpBuffer) return;
    try {
      const src = this.audioCtx.createBufferSource();
      src.buffer = this.chirpBuffer;
      src.connect(this.audioCtx.destination);
      src.start();
    } catch (e) { /* a dropped pulse is harmless; the next one is 50 ms away */ }
  }

  startPulseLoop(rateHz) {
    this.rateHz = Math.max(4, Math.min(25, rateHz));
    const period = Math.round(1000 / this.rateHz);
    this.stopPulseLoop();
    this.pulseTimer = setInterval(() => {
      this.transmit();
      // Process after the pulse has had time to fly out and back. Anything
      // beyond ~4 m is outside the search window anyway.
      setTimeout(() => this.processPulse(), PROCESS_DELAY_MS);
    }, period);
  }

  stopPulseLoop() {
    if (this.pulseTimer) { clearInterval(this.pulseTimer); this.pulseTimer = null; }
  }

  /** Copy the newest FFT_SIZE samples out of the ring and run the pipeline. */
  processPulse() {
    if (!this.running || !this.pipeline) return;
    const t0 = performance.now();
    let idx = (this.ringHead - FFT_SIZE + RING_SIZE) % RING_SIZE;
    for (let i = 0; i < FFT_SIZE; i++) {
      this.rxWindow[i] = this.ring[idx];
      idx = (idx + 1) % RING_SIZE;
    }

    this.pulseCount++;
    let result;
    try {
      result = this.pipeline.process(this.rxWindow, {
        t: Date.now(),
        classifier: this.classifier,
      });
    } catch (e) {
      // A DSP exception must not stop the scan.
      this.onLog('DSP error: ' + e.message, 'error');
      return;
    }
    this.lastPulseAt = Date.now();
    if (result.detection) this.detectionCount++;

    const ms = performance.now() - t0;
    this.trackLoad.sum += ms;
    this.trackLoad.n++;
    if (ms > this.trackLoad.max) this.trackLoad.max = ms;

    this.onDetection(result.detection, result.diagnostics);
  }

  setClassifier(c) { this.classifier = c; }
  setCalibration(factor, offsetM) { if (this.pipeline) this.pipeline.setCalibration(factor, offsetM); }
  resetClutter() { if (this.pipeline) this.pipeline.resetClutter(); }

  /** One chirp, no scan loop — used by the diagnostics page's speaker test. */
  async testChirp() {
    try {
      if (!this.audioCtx) {
        const AC = window.AudioContext || window.webkitAudioContext;
        this.audioCtx = new AC({ sampleRate: 48000 });
        this.sampleRate = this.audioCtx.sampleRate;
        this.pipeline = new DetectionPipeline(this.sampleRate);
        this.buildChirpBuffer();
      }
      if (this.audioCtx.state === 'suspended') await this.audioCtx.resume();
      if (!this.chirpBuffer) this.buildChirpBuffer();
      this.transmit();
      return { ok: true, state: this.audioCtx.state, sampleRate: this.sampleRate };
    } catch (e) {
      return { ok: false, error: e.message };
    }
  }

  async stop() {
    this.stopPulseLoop();
    this.running = false;
    try { if (this.micStream) this.micStream.getTracks().forEach((t) => t.stop()); } catch (e) { /* ignore */ }
    for (const node of [this.workletNode, this.scriptNode, this.micSource, this.sinkGain]) {
      try { if (node) node.disconnect(); } catch (e) { /* ignore */ }
    }
    if (this.workletNode && this.workletNode.port) this.workletNode.port.onmessage = null;
    if (this.scriptNode) this.scriptNode.onaudioprocess = null;
    this.workletNode = null;
    this.scriptNode = null;
    this.micSource = null;
    this.sinkGain = null;
    this.micStream = null;
    try { if (this.audioCtx && this.audioCtx.state !== 'closed') await this.audioCtx.close(); } catch (e) { /* ignore */ }
    this.audioCtx = null;
    this.chirpBuffer = null;
    this.ring.fill(0);
    this.ringHead = 0;
    return { ok: true };
  }

  status() {
    const load = this.trackLoad.n ? this.trackLoad.sum / this.trackLoad.n : 0;
    return {
      running: this.running,
      captureMode: this.captureMode,
      sampleRate: this.sampleRate,
      channelCount: this.channelCount,
      audioContextState: this.audioCtx ? this.audioCtx.state : 'closed',
      rateHz: this.rateHz,
      pulses: this.pulseCount,
      detections: this.detectionCount,
      detectionRate: this.pulseCount ? this.detectionCount / this.pulseCount : 0,
      dspMsAvg: Math.round(load * 100) / 100,
      dspMsMax: Math.round(this.trackLoad.max * 100) / 100,
      lastError: this.lastError,
      trackSettings: this.trackSettings || null,
      // Mono capture cannot resolve angle; bearing is the phone's boresight.
      bearingSource: this.channelCount >= 2 ? 'stereo-capable (TDOA not implemented)' : 'phone boresight',
      noiseFloor: this.pipeline ? this.pipeline.noiseFloor : 0,
      clutterReady: this.pipeline ? this.pipeline.baselineCount >= 8 : false,
    };
  }

  /** Down-sampled range profile for the A-scope display. */
  profile(bins, maxRangeM) {
    return this.pipeline ? this.pipeline.profileForDisplay(bins, maxRangeM) : null;
  }
}

/** Turn a getUserMedia exception into something an operator can act on. */
export function describeAudioError(e) {
  const name = e && e.name ? e.name : '';
  switch (name) {
    case 'NotAllowedError':
    case 'SecurityError':
      return 'Microphone permission denied. Allow it in the site settings (tap the lock icon in the address bar), then try again.';
    case 'NotFoundError':
    case 'DevicesNotFoundError':
      return 'No microphone found on this device.';
    case 'NotReadableError':
    case 'TrackStartError':
      return 'The microphone is in use by another app. Close it and retry.';
    case 'OverconstrainedError':
      return 'The device refused the requested audio format (48 kHz raw). Try simulation mode.';
    case 'AbortError':
      return 'Audio start was aborted by the browser.';
    default:
      return (name ? name + ': ' : '') + (e && e.message ? e.message : 'unknown audio error');
  }
}
