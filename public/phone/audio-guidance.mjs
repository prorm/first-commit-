/**
 * Local audio and haptic guidance.
 *
 * Two independent channels:
 *
 *   Proximity tone — a click whose repetition rate and pitch encode range.
 *   Continuous, immediate, and generated locally so it never waits on the
 *   network.  This is the channel someone would actually navigate by: rate
 *   rises from roughly 2 Hz at 3 m to 14 Hz inside half a metre, and the pitch
 *   rises with it, which is easier to judge than either cue alone.
 *
 *   Speech — sparse, debounced, and hedged.  Routed through VoiceProvider so
 *   Amazon Polly is used when credentials exist and the browser's own
 *   SpeechSynthesis when they do not.  The phone always talks.
 *
 * Panning: with a mono sensor there is no resolved bearing, so the stereo
 * position follows the *boresight offset* of the detection relative to the
 * operator's heading.  That is a real quantity, not a guess, and it makes
 * "on your left" audible as well as spoken.
 */

export class GuidanceEngine {
  constructor(opts = {}) {
    this.ctx = null;
    this.master = null;
    this.panner = null;
    this.enabled = true;
    this.speechEnabled = true;
    this.volume = opts.volume != null ? opts.volume : 0.35;
    this.voice = new VoiceProvider(opts.voice || {});
    this.lastClickAt = 0;
    this.lastHapticAt = 0;
    this.currentRate = 0;
    this.currentPitch = 0;
    this.tickTimer = null;
    this.latest = null;
    this.lastSpokenText = null;
    this.clicks = 0;
    this.supportsHaptics = typeof navigator !== 'undefined' && !!navigator.vibrate;
  }

  /** Must be called from a user gesture (autoplay policy). */
  async init(sharedCtx) {
    try {
      if (sharedCtx) this.ctx = sharedCtx;
      if (!this.ctx) {
        const AC = window.AudioContext || window.webkitAudioContext;
        this.ctx = new AC();
      }
      if (this.ctx.state === 'suspended') await this.ctx.resume();

      this.master = this.ctx.createGain();
      this.master.gain.value = this.volume;
      // StereoPanner is not universal; fall back to a plain connection.
      if (this.ctx.createStereoPanner) {
        this.panner = this.ctx.createStereoPanner();
        this.master.connect(this.panner);
        this.panner.connect(this.ctx.destination);
      } else {
        this.master.connect(this.ctx.destination);
      }
      this.startTicker();
      return { ok: true, state: this.ctx.state, panning: !!this.panner };
    } catch (e) {
      return { ok: false, error: e.message };
    }
  }

  /**
   * Feed the newest detection (or null when nothing is in range).
   * The tone loop reads this; it is deliberately not driven per-detection so
   * the cadence stays smooth at 20 Hz of input.
   */
  update(det, headingDeg) {
    this.latest = det ? {
      range: det.range_m,
      ttc: det.ttc_s,
      vel: det.vel_mps,
      confidence: det.confidence,
      cls: det.fusedClass || det.obstacleClass,
      offset: signedOffset(det.bearing_deg, headingDeg != null ? headingDeg : det.bearing_deg),
      at: Date.now(),
    } : null;
  }

  startTicker() {
    this.stopTicker();
    // A 40 ms scheduler is fine granularity for a cue that tops out at 14 Hz,
    // and far cheaper than one timer per click.
    this.tickTimer = setInterval(() => this.tick(), 40);
  }

  stopTicker() {
    if (this.tickTimer) { clearInterval(this.tickTimer); this.tickTimer = null; }
  }

  tick() {
    if (!this.enabled || !this.ctx || !this.latest) { this.currentRate = 0; return; }
    const age = Date.now() - this.latest.at;
    if (age > 700) { this.currentRate = 0; return; }   // stale: go quiet, do not invent

    const r = this.latest.range;
    const conf = this.latest.confidence;
    // Rate and pitch both rise as range falls. The mapping is deliberately
    // non-linear so the last metre is unmistakable.
    const prox = Math.max(0, Math.min(1, 1 - (r - 0.35) / 2.8));
    const rate = 2 + Math.pow(prox, 1.6) * 12;         // 2 -> 14 Hz
    const pitch = 420 + Math.pow(prox, 1.4) * 900;     // 420 -> 1320 Hz
    this.currentRate = rate;
    this.currentPitch = pitch;

    const now = performance.now();
    const interval = 1000 / rate;
    if (now - this.lastClickAt >= interval) {
      this.lastClickAt = now;
      // A low-confidence detection clicks quieter: the sound itself carries
      // how much the system trusts what it heard.
      this.click(pitch, 0.35 + 0.65 * conf, this.latest.offset, r);
    }

    // Haptics only for genuinely close obstacles; buzzing constantly is noise.
    if (r < 0.9 && conf > 0.3 && Date.now() - this.lastHapticAt > Math.max(180, r * 600)) {
      this.lastHapticAt = Date.now();
      this.vibrate(r < 0.5 ? [45, 30, 45] : [30]);
    }
  }

  /** One short shaped click — a raised-cosine burst, no clipping artefacts. */
  click(freq, gain, offsetDeg, range) {
    if (!this.ctx || !this.master) return;
    try {
      const t = this.ctx.currentTime;
      const osc = this.ctx.createOscillator();
      const env = this.ctx.createGain();
      osc.type = range < 0.6 ? 'square' : 'sine';      // urgency has a timbre
      osc.frequency.value = freq;
      env.gain.setValueAtTime(0, t);
      env.gain.linearRampToValueAtTime(gain, t + 0.004);
      env.gain.exponentialRampToValueAtTime(0.0008, t + 0.055);
      osc.connect(env);
      env.connect(this.master);
      if (this.panner) {
        // +-60 degrees of boresight offset maps to full pan.
        this.panner.pan.setTargetAtTime(clamp(offsetDeg / 60, -1, 1), t, 0.05);
      }
      osc.start(t);
      osc.stop(t + 0.07);
      this.clicks++;
    } catch (e) { /* a dropped click is inaudible */ }
  }

  vibrate(pattern) {
    try { if (navigator.vibrate) navigator.vibrate(pattern); } catch (e) { /* unsupported */ }
  }

  /** Speak guidance text. Returns which provider actually produced the audio. */
  async speak(text, level) {
    if (!this.speechEnabled || !text) return { provider: 'none' };
    this.lastSpokenText = text;
    // Urgent speech ducks the click track so it can be understood.
    const prevVol = this.master ? this.master.gain.value : null;
    if (this.master && level === 'critical') {
      try { this.master.gain.setTargetAtTime(this.volume * 0.25, this.ctx.currentTime, 0.05); } catch (e) { /* ignore */ }
    }
    const res = await this.voice.speak(text, { level });
    if (this.master && prevVol != null && level === 'critical') {
      setTimeout(() => {
        try { this.master.gain.setTargetAtTime(this.volume, this.ctx.currentTime, 0.3); } catch (e) { /* ignore */ }
      }, 900);
    }
    return res;
  }

  setVolume(v) {
    this.volume = clamp(v, 0, 1);
    if (this.master) this.master.gain.value = this.volume;
  }

  setEnabled(on) {
    this.enabled = !!on;
    if (!on) this.currentRate = 0;
  }

  status() {
    return {
      enabled: this.enabled,
      speechEnabled: this.speechEnabled,
      contextState: this.ctx ? this.ctx.state : 'none',
      panning: !!this.panner,
      rateHz: Math.round(this.currentRate * 10) / 10,
      pitchHz: Math.round(this.currentPitch),
      clicks: this.clicks,
      haptics: this.supportsHaptics,
      voice: this.voice.status(),
    };
  }

  destroy() {
    this.stopTicker();
    this.voice.cancel();
  }
}

/**
 * VoiceProvider — Amazon Polly when the server has credentials, the browser's
 * SpeechSynthesis when it does not.
 *
 * The phone never decides this by itself: the server owns the credentials and
 * reports which provider produced each cue, so the label the UI shows is the
 * truth about what was used rather than what was configured.
 */
export class VoiceProvider {
  constructor(opts = {}) {
    this.sendToServer = opts.sendToServer || null;   // (text, level) => void
    this.preferPolly = opts.preferPolly !== false;
    this.provider = 'local';
    this.lastProvider = null;
    this.lastReason = null;
    this.pollyAvailable = false;
    this.speaking = false;
    this.localVoice = null;
    this.audioEl = null;
    this.spoken = { polly: 0, local: 0, failed: 0 };
    this.synthSupported = typeof window !== 'undefined' && 'speechSynthesis' in window;
    if (this.synthSupported) this.pickLocalVoice();
  }

  pickLocalVoice() {
    const choose = () => {
      const voices = window.speechSynthesis.getVoices() || [];
      // Prefer a clear English voice; anything is better than nothing.
      this.localVoice = voices.find((v) => /en[-_](GB|US)/i.test(v.lang) && !/female/i.test(v.name))
        || voices.find((v) => /^en/i.test(v.lang))
        || voices[0] || null;
    };
    choose();
    // Chrome populates the voice list asynchronously.
    if (!this.localVoice) window.speechSynthesis.onvoiceschanged = choose;
  }

  setPollyAvailable(on, reason) {
    this.pollyAvailable = !!on;
    this.lastReason = reason || null;
    this.provider = this.pollyAvailable && this.preferPolly ? 'polly' : 'local';
  }

  /**
   * Ask for speech.  When Polly is available the request goes to the server,
   * which streams back MP3; otherwise we speak locally immediately.
   */
  async speak(text, opts = {}) {
    if (this.pollyAvailable && this.preferPolly && this.sendToServer) {
      this.sendToServer(text, opts.level);
      // The server answers with voice_audio; playPollyAudio() handles it. If
      // nothing arrives (dropped socket), the local fallback fires below.
      this.pendingPollyAt = Date.now();
      this.pendingText = text;
      clearTimeout(this.pollyTimeout);
      this.pollyTimeout = setTimeout(() => {
        if (this.pendingText === text) {
          this.pendingText = null;
          this.speakLocal(text);
          this.lastProvider = 'local';
          this.lastReason = 'Polly did not answer in time';
        }
      }, 1200);
      return { provider: 'polly-pending' };
    }
    this.speakLocal(text);
    return { provider: 'local' };
  }

  /** Play base64 MP3 returned by the server's Polly call. */
  playPollyAudio(base64, text) {
    clearTimeout(this.pollyTimeout);
    this.pendingText = null;
    try {
      if (!this.audioEl) {
        this.audioEl = new Audio();
        this.audioEl.preload = 'auto';
      }
      this.audioEl.src = 'data:audio/mpeg;base64,' + base64;
      const p = this.audioEl.play();
      if (p && p.catch) {
        p.catch(() => {
          // Autoplay refused the element; speak locally so the cue still lands.
          this.speakLocal(text);
          this.lastProvider = 'local';
          this.lastReason = 'browser blocked MP3 playback';
        });
      }
      this.spoken.polly++;
      this.lastProvider = 'polly';
      return true;
    } catch (e) {
      this.spoken.failed++;
      this.speakLocal(text);
      return false;
    }
  }

  speakLocal(text) {
    if (!this.synthSupported) { this.spoken.failed++; return false; }
    try {
      const u = new SpeechSynthesisUtterance(text);
      if (this.localVoice) u.voice = this.localVoice;
      u.rate = 1.08;
      u.pitch = 1.0;
      u.volume = 1.0;
      // Cancel any queued speech: a backlog of stale cues is worse than none.
      window.speechSynthesis.cancel();
      window.speechSynthesis.speak(u);
      this.spoken.local++;
      this.lastProvider = 'local';
      return true;
    } catch (e) {
      this.spoken.failed++;
      return false;
    }
  }

  cancel() {
    clearTimeout(this.pollyTimeout);
    try { if (this.synthSupported) window.speechSynthesis.cancel(); } catch (e) { /* ignore */ }
    try { if (this.audioEl) this.audioEl.pause(); } catch (e) { /* ignore */ }
  }

  status() {
    return {
      configured: this.provider,
      lastUsed: this.lastProvider,
      label: this.lastProvider === 'polly' ? 'AWS POLLY'
        : this.pollyAvailable && this.preferPolly ? 'AWS POLLY (ready)' : 'LOCAL FALLBACK',
      pollyAvailable: this.pollyAvailable,
      speechSynthesis: this.synthSupported,
      localVoice: this.localVoice ? this.localVoice.name : null,
      reason: this.lastReason,
      counts: Object.assign({}, this.spoken),
    };
  }
}

/** Signed bearing offset from the operator's heading, in (-180, 180]. */
export function signedOffset(bearingDeg, headingDeg) {
  return ((bearingDeg - headingDeg + 540) % 360) - 180;
}

function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }
