/**
 * Guidance policy — what the system says out loud, and when it keeps quiet.
 *
 * Shared by the server (so the command center can show the same cue log the
 * operator heard) and driven by the phone's own local copy for the tones.
 *
 * Two rules do most of the work:
 *   1. Nothing is spoken twice inside its cooldown, and a more urgent cue can
 *      pre-empt a less urgent one but not the reverse.
 *   2. A cue whose evidence is weak says so out loud ("possible opening"),
 *      because the classifier's OPENING recall is 43 % and speech that sounds
 *      certain would be the single most misleading thing in the demo.
 */

const PRIORITY = { critical: 3, warning: 2, info: 1, ambient: 0 };

/** Bearing relative to the operator's own heading, as a spoken direction. */
function relativeDirection(bearingDeg, headingDeg) {
  let d = ((bearingDeg - headingDeg + 540) % 360) - 180;
  const a = Math.abs(d);
  if (a <= 25) return { word: 'ahead', delta: d };
  if (a >= 155) return { word: 'behind you', delta: d };
  if (d > 0) return { word: a <= 70 ? 'on your right' : 'to your right', delta: d };
  return { word: a <= 70 ? 'on your left' : 'to your left', delta: d };
}

class GuidancePolicy {
  constructor(opts = {}) {
    this.cooldowns = {
      critical: 1800,
      warning: 3200,
      info: 5000,
      ambient: 9000,
    };
    this.minGapMs = opts.minGapMs || 1400;   // never two cues closer than this
    this.lastSpokenAt = 0;
    this.lastKey = null;
    this.lastLevel = 'ambient';
    this.history = [];
    this.perKey = new Map();
  }

  /**
   * Decide whether to speak about this detection.
   * @returns {null|{text:string, level:string, key:string, reason:string}}
   */
  evaluate(det, opts = {}) {
    const now = det && det.t ? det.t : Date.now();
    const cue = this.pick(det, opts);
    if (!cue) return null;

    const lastSame = this.perKey.get(cue.key) || 0;
    const cooldown = this.cooldowns[cue.level] || 4000;
    const sinceAny = now - this.lastSpokenAt;
    const sinceSame = now - lastSame;

    // A critical cue may interrupt a lower-priority one; nothing interrupts a
    // cue of equal or higher priority inside the minimum gap.
    const outranksLast = PRIORITY[cue.level] > PRIORITY[this.lastLevel];
    if (sinceAny < this.minGapMs && !(outranksLast && sinceAny > 500)) return null;
    if (sinceSame < cooldown) return null;

    this.lastSpokenAt = now;
    this.lastLevel = cue.level;
    this.lastKey = cue.key;
    this.perKey.set(cue.key, now);
    this.history.push({ t: now, text: cue.text, level: cue.level });
    if (this.history.length > 40) this.history.shift();
    return cue;
  }

  /** Choose the most useful thing to say about the current picture. */
  pick(det, opts = {}) {
    if (!det) {
      // Sustained absence of returns is itself information.
      if (opts.quietMs && opts.quietMs > 2500) {
        return { text: 'Clear path ahead.', level: 'ambient', key: 'clear', reason: 'no returns for ' + Math.round(opts.quietMs / 1000) + ' s' };
      }
      return null;
    }

    const r = det.range_m;
    const dir = relativeDirection(det.bearing_deg, det.phone ? det.phone.heading : det.bearing_deg);
    const cls = det.fusedClass || det.obstacleClass;
    const conf = det.fusedConfidence || det.classConfidence || 0;
    const stable = det.fusedStable !== false;

    // Imminent contact outranks everything, and never mentions a class — at
    // half a metre the class does not change what the operator should do.
    if (r < 0.55 && det.confidence > 0.25) {
      return { text: 'Stop. Obstacle ' + dir.word + '.', level: 'critical', key: 'stop', reason: 'range ' + r.toFixed(2) + ' m' };
    }
    if (det.ttc_s < 1.6 && det.vel_mps > 0.35 && r < 2.2) {
      return { text: 'Closing fast, obstacle ' + dir.word + '.', level: 'critical', key: 'closing', reason: 'ttc ' + det.ttc_s.toFixed(1) + ' s' };
    }

    // An opening is the one cue worth reporting even when weak — but it is
    // always hedged, and a low-confidence call says "possible".
    if (cls === 'OPENING' && det.confidence > 0.2) {
      const hedge = conf >= 0.55 && stable ? 'Opening detected ' : 'Possible opening ';
      return {
        text: hedge + dir.word + ', ' + r.toFixed(1) + ' metres.',
        level: 'info', key: 'opening-' + dir.word, reason: 'class OPENING at ' + (conf * 100).toFixed(0) + '%',
      };
    }

    if (r < 1.3 && det.confidence > 0.3) {
      const what = cls === 'SOFT' && conf > 0.5 ? 'Soft obstacle ' : 'Obstacle ';
      return { text: what + dir.word + ', ' + r.toFixed(1) + ' metres.', level: 'warning', key: 'near-' + dir.word, reason: 'range ' + r.toFixed(2) + ' m' };
    }
    if (r < 2.6 && det.confidence > 0.35) {
      return { text: 'Obstacle ' + dir.word + ', ' + r.toFixed(1) + ' metres.', level: 'info', key: 'mid-' + dir.word, reason: 'range ' + r.toFixed(2) + ' m' };
    }
    return null;
  }

  reset() {
    this.lastSpokenAt = 0;
    this.lastKey = null;
    this.lastLevel = 'ambient';
    this.history = [];
    this.perKey.clear();
  }
}

module.exports = { GuidancePolicy, relativeDirection, PRIORITY };
