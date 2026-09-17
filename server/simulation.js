/**
 * SentryShield digital twin.
 *
 * A deterministic virtual building, a virtual walker, and a real classifier.
 * Every pulse is raycast against actual 2-D geometry, turned into a synthetic
 * matched-filter envelope by shared/echosynth.mjs, and then classified by the
 * exported EchoNet weights — the same network the phone runs.  Detection
 * frames leave here on the identical schema live mode uses, so the command
 * center genuinely cannot tell the difference.
 *
 * Three modes use this file:
 *   simulation — virtual pose + virtual acoustics (works with no phone at all)
 *   hybrid     — real phone heading/motion + virtual acoustics (for bad venues)
 *   (live mode does not touch this file)
 */
const { createRequire } = require('node:module');
const { synthesizePulse, makeDeviceResponse, samplesToRange } = require('../public/shared/echosynth.mjs');
const { makePose, wrapDeg, clamp01, normalizeDetection } = require('../public/shared/protocol.mjs');

const requireCjs = createRequire(__filename);
let EchoNet = null;
let echonetError = null;
try {
  EchoNet = requireCjs('../src/classifier/echonet_weights.js');
  const st = EchoNet.selfTest();
  if (!st.ok) { echonetError = 'self-test failed (maxAbsError ' + st.maxAbsError + ')'; EchoNet = null; }
} catch (e) {
  echonetError = e.message;
}

// ---------------------------------------------------------------------------
// Scenarios — every wall is a real line segment with a real material
// ---------------------------------------------------------------------------
/**
 * A surface: { a:[x,y], b:[x,y], kind:'WALL'|'SOFT'|'OPENING' }
 * OPENING segments are the gaps: a doorway plane the beam passes through.
 * They still return an echo (jamb diffraction), just a very weak one.
 */
function seg(ax, ay, bx, by, kind) { return { a: [ax, ay], b: [bx, by], kind: kind || 'WALL' }; }

const SCENARIOS = {
  room: {
    label: 'ROOM — 6 x 4 m, single doorway, sofa',
    description: 'Closed room with one doorway on the north wall and a soft sofa against the east wall.',
    start: { x: 0, y: -1.2, heading: 0 },
    route: [[0, -1.2], [0, 0.6], [1.6, 1.0], [1.6, -0.6], [-1.8, -0.9], [-1.8, 0.8]],
    surfaces: [
      // north wall, broken by a 0.9 m doorway near x = +0.4
      seg(-3, 2, -0.05, 2), seg(-0.05, 2, 0.85, 2, 'OPENING'), seg(0.85, 2, 3, 2),
      seg(-3, -2, 3, -2),                 // south wall
      seg(-3, -2, -3, 2),                 // west wall
      seg(3, -2, 3, 2),                   // east wall
      seg(2.1, -1.3, 2.1, 0.2, 'SOFT'),   // sofa front face
      seg(-2.2, 1.2, -1.2, 1.2, 'SOFT'),  // curtain
    ],
  },
  corridor: {
    label: 'CORRIDOR — 1.6 m wide, 9 m long, two side doors',
    description: 'Long corridor with two doorways on the left and a closed end wall.',
    start: { x: 0, y: -4, heading: 0 },
    route: [[0, -4], [0, 4.2]],
    surfaces: [
      seg(-0.8, -4.5, -0.8, -1.0), seg(-0.8, -1.0, -0.8, -0.1, 'OPENING'), seg(-0.8, -0.1, -0.8, 1.8),
      seg(-0.8, 1.8, -0.8, 2.7, 'OPENING'), seg(-0.8, 2.7, -0.8, 4.5),
      seg(0.8, -4.5, 0.8, 4.5),
      seg(-0.8, 4.5, 0.8, 4.5),
      seg(-0.8, -4.5, 0.8, -4.5),
    ],
  },
  corner: {
    label: 'CORNER — right-angle turn into a second corridor',
    description: 'L-shaped corridor: walk north, turn east at the corner.',
    start: { x: 0, y: -3.4, heading: 0 },
    route: [[0, -3.4], [0, 1.2], [3.6, 1.2]],
    surfaces: [
      seg(-0.9, -4, -0.9, 2.1),
      seg(-0.9, 2.1, 4.5, 2.1),
      seg(0.9, -4, 0.9, 0.3),
      seg(0.9, 0.3, 4.5, 0.3),
      seg(4.5, 0.3, 4.5, 2.1),
      seg(-0.9, -4, 0.9, -4),
    ],
  },
  apartment: {
    label: 'APARTMENT — two rooms, connecting doorway, furniture',
    description: 'Larger multi-room scene: the richest reconstruction, best for the finale.',
    start: { x: -2, y: -2.4, heading: 0 },
    route: [[-2, -2.4], [-2, 1.4], [-0.2, 1.9], [2.0, 1.9], [2.4, -0.4], [0.4, -1.8], [-2, -2.0]],
    surfaces: [
      seg(-4, 3, 4, 3),
      seg(-4, -3, 4, -3),
      seg(-4, -3, -4, 3),
      seg(4, -3, 4, 3),
      // partition between the two rooms, with a doorway gap
      seg(0.6, -3, 0.6, 0.6), seg(0.6, 0.6, 0.6, 1.5, 'OPENING'), seg(0.6, 1.5, 0.6, 3),
      seg(-3.4, 2.2, -2.2, 2.2, 'SOFT'),   // bed
      seg(-3.9, -1.4, -3.9, -0.2, 'SOFT'), // wardrobe curtain
      seg(2.6, 2.4, 3.6, 2.4, 'SOFT'),     // armchair
      seg(1.4, -2.4, 2.6, -2.4),           // counter
    ],
  },
  openfield: {
    label: 'OPEN SPACE — sparse returns, honest empty map',
    description: 'Mostly open area with two distant walls: shows what a weak-evidence scan looks like.',
    start: { x: 0, y: 0, heading: 0 },
    route: [[0, 0], [0, 2.5], [2.5, 2.5], [0, 0]],
    surfaces: [seg(-6, 6, 6, 6), seg(6, -6, 6, 6)],
  },
};

const SCENARIO_LIST = Object.keys(SCENARIOS).map((k) => ({
  id: k, label: SCENARIOS[k].label, description: SCENARIOS[k].description,
}));

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------
/**
 * Cast a ray and return the nearest surface hit.
 * Bearing is compass degrees (0 = +y), matching the pose convention.
 */
function raycast(surfaces, x, y, bearingDeg, maxRange) {
  const r = (wrapDeg(bearingDeg) * Math.PI) / 180;
  const dx = Math.sin(r);
  const dy = Math.cos(r);
  let best = null;
  for (const s of surfaces) {
    const ex = s.b[0] - s.a[0];
    const ey = s.b[1] - s.a[1];
    const den = dx * ey - dy * ex;
    if (Math.abs(den) < 1e-9) continue;                 // parallel
    const t = ((s.a[0] - x) * ey - (s.a[1] - y) * ex) / den;   // along the ray
    const u = ((s.a[0] - x) * dy - (s.a[1] - y) * dx) / den;   // along the segment
    if (t <= 0.05 || t > maxRange || u < 0 || u > 1) continue;
    if (!best || t < best.range) {
      // Incidence angle matters: a near-grazing wall returns much less energy.
      const segLen = Math.hypot(ex, ey) || 1;
      const nx = -ey / segLen;
      const ny = ex / segLen;
      const cosInc = Math.abs(dx * nx + dy * ny);
      best = { range: t, kind: s.kind, cosInc, surface: s, u };
    }
  }
  return best;
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------
class SimulationEngine {
  /**
   * @param {Object} opts
   * @param {(det:Object)=>void} opts.onDetection  called per synthesised pulse
   * @param {string} [opts.scenario]
   * @param {number} [opts.seed]
   */
  constructor(opts = {}) {
    this.onDetection = opts.onDetection || (() => {});
    this.onStatus = opts.onStatus || (() => {});
    this.setScenario(opts.scenario || 'room', opts.seed);
    this.mode = 'simulation';
    this.timer = null;
    this.paused = false;
    this.rateHz = 20;
    this.walkSpeed = 0.55;              // [m/s]
    this.sweepRateDegPerS = 62;         // boresight sweep while walking
    this.sweepAmplitude = 55;           // +- degrees around the route heading
    this.manualSteer = null;            // degrees offset commanded by the UI
    this.externalPose = null;           // hybrid mode: pose from the real phone
    this.pulseCount = 0;
  }

  /** Deterministic LCG: the same seed replays the same scan, bit for bit. */
  makeRng(seed) {
    let s = (seed || 1) & 0x7fffffff;
    return () => ((s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
  }

  setScenario(id, seed) {
    const sc = SCENARIOS[id] || SCENARIOS.room;
    this.scenarioId = SCENARIOS[id] ? id : 'room';
    this.scenario = sc;
    this.seed = seed || 20260917;
    this.rng = this.makeRng(this.seed);
    this.device = makeDeviceResponse(this.rng);
    this.routeIdx = 0;
    this.routeT = 0;
    this.pos = { x: sc.start.x, y: sc.start.y };
    this.heading = sc.start.heading;
    this.sweepPhase = 0;
    this.pulseCount = 0;
    this.lastRange = null;
    this.lastRangeT = 0;
  }

  start(rateHz) {
    this.stop();
    if (rateHz) this.rateHz = rateHz;
    this.paused = false;
    const period = Math.max(25, Math.round(1000 / this.rateHz));
    this.timer = setInterval(() => {
      try { this.tick(period / 1000); } catch (e) { this.onStatus({ error: 'sim tick: ' + e.message }); }
    }, period);
    if (this.timer.unref) this.timer.unref();
  }

  stop() {
    if (this.timer) { clearInterval(this.timer); this.timer = null; }
  }

  setMode(mode) { this.mode = mode; }
  setPaused(p) { this.paused = !!p; }
  setSpeed(v) { this.walkSpeed = Math.max(0, Math.min(2.0, v)); }
  setSteer(deg) { this.manualSteer = deg == null ? null : Number(deg); }
  /** Hybrid mode: adopt the real phone's heading (and motion if it has any). */
  setExternalPose(pose) { this.externalPose = pose ? makePose(pose) : null; }

  /** Advance the virtual walker along its route. */
  advance(dt) {
    const route = this.scenario.route;
    if (!route || route.length < 2 || this.paused) return;
    let remaining = this.walkSpeed * dt;
    let guard = 0;
    while (remaining > 0 && guard++ < 8) {
      const a = route[this.routeIdx % route.length];
      const b = route[(this.routeIdx + 1) % route.length];
      const segLen = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1e-6;
      const step = remaining / segLen;
      if (this.routeT + step >= 1) {
        remaining -= (1 - this.routeT) * segLen;
        this.routeIdx = (this.routeIdx + 1) % route.length;
        this.routeT = 0;
      } else {
        this.routeT += step;
        remaining = 0;
      }
      const a2 = route[this.routeIdx % route.length];
      const b2 = route[(this.routeIdx + 1) % route.length];
      this.pos.x = a2[0] + (b2[0] - a2[0]) * this.routeT;
      this.pos.y = a2[1] + (b2[1] - a2[1]) * this.routeT;
      this.routeHeading = (Math.atan2(b2[0] - a2[0], b2[1] - a2[1]) * 180) / Math.PI;
    }
  }

  /**
   * The current sensor pose.  In hybrid mode the heading (and, if the phone
   * reports movement, the position) come from the real device — the acoustics
   * stay virtual.
   */
  currentPose() {
    if (this.mode === 'hybrid' && this.externalPose) {
      const ext = this.externalPose;
      return makePose({
        x: ext.method === 'dead-reckoning' ? ext.x : this.pos.x,
        y: ext.method === 'dead-reckoning' ? ext.y : this.pos.y,
        heading: ext.heading,
        confidence: Math.min(0.75, ext.confidence),
        method: 'hybrid',
      });
    }
    return makePose({
      x: this.pos.x, y: this.pos.y, heading: this.heading,
      confidence: 0.92, method: 'simulated',
    });
  }

  tick(dt) {
    this.advance(dt);

    // Boresight: a hand-held phone sweeps as the operator scans the space.
    if (this.mode === 'hybrid' && this.externalPose) {
      this.heading = this.externalPose.heading;
    } else if (this.manualSteer != null) {
      this.heading = wrapDeg((this.routeHeading || 0) + this.manualSteer);
    } else {
      this.sweepPhase += (this.sweepRateDegPerS * dt) / this.sweepAmplitude;
      const base = this.routeHeading != null ? this.routeHeading : this.scenario.start.heading;
      this.heading = wrapDeg(base + Math.sin(this.sweepPhase) * this.sweepAmplitude);
    }

    const pose = this.currentPose();
    const hit = raycast(this.scenario.surfaces, pose.x, pose.y, pose.heading, 4.0);
    this.pulseCount++;

    if (!hit) {
      // Nothing in range.  Emitting nothing is the honest outcome — the map
      // shows unscanned space rather than an invented return.
      this.onDetection(null, pose);
      return;
    }

    // Grazing incidence loses energy; fold it into the pointing loss so the
    // synthesised echo weakens exactly as a real oblique wall would.
    const pointing = Math.max(0.12, Math.pow(Math.max(hit.cosInc, 0.05), 0.7));
    const pulse = synthesizePulse(hit.kind, hit.range, this.rng, {
      device: this.device,
      pointing,
      phoneHeight: 1.05,
      floorR: 0.45,
      beyond: hit.kind === 'OPENING' ? this.beyondRange(pose, hit) : undefined,
    });

    // Real classifier, real output — including its mistakes.
    let probs = [0, 0, 0];
    let className = null;
    let classConfidence = 0;
    if (EchoNet) {
      const out = EchoNet.forward(pulse.win);
      probs = Array.from(out.probs);
      className = out.className;
      classConfidence = out.confidence;
    }

    // Closing velocity from consecutive ranges, as the live tracker computes it.
    const now = Date.now();
    let vel = 0;
    if (this.lastRange != null && now > this.lastRangeT) {
      const dtr = (now - this.lastRangeT) / 1000;
      if (dtr > 0.01 && dtr < 0.5) vel = (this.lastRange - pulse.measuredRange) / dtr;
    }
    this.lastRange = pulse.measuredRange;
    this.lastRangeT = now;

    const det = normalizeDetection({
      t: now,
      range_m: pulse.measuredRange,
      vel_mps: Math.max(-3, Math.min(3, vel)),
      confidence: clamp01(pulse.detectionConfidence),
      bearing_deg: pose.heading,
      beamwidth_deg: 30,
      snr_db: pulse.snr_db,
      cfar_pass: pulse.cfar_pass,
      obstacleClass: className,
      classConfidence,
      classProbs: probs,
      phone: pose,
      source: this.mode === 'hybrid' ? 'hybrid' : 'simulation',
    }, this.mode === 'hybrid' ? 'hybrid' : 'simulation');

    if (det) {
      // Ground truth is carried alongside for the diagnostics page only; the
      // map never reads it, so it cannot leak into the visualisation.
      det.debug = { truthClass: hit.kind, truthRange: Math.round(hit.range * 1000) / 1000, cosInc: Math.round(hit.cosInc * 100) / 100 };
      this.onDetection(det, pose);
    } else {
      this.onDetection(null, pose);
    }
  }

  /** How far the room extends past a doorway, for the OPENING far-wall term. */
  beyondRange(pose, hit) {
    const past = raycast(this.scenario.surfaces, pose.x, pose.y, pose.heading, 14.0);
    const beyond = raycast(
      this.scenario.surfaces,
      pose.x + Math.sin((pose.heading * Math.PI) / 180) * (hit.range + 0.2),
      pose.y + Math.cos((pose.heading * Math.PI) / 180) * (hit.range + 0.2),
      pose.heading, 14.0
    );
    const r = beyond ? hit.range + 0.2 + beyond.range : 5.5;
    return Math.max(3.9, r);
  }

  info() {
    return {
      scenario: this.scenarioId,
      label: this.scenario.label,
      description: this.scenario.description,
      surfaces: this.scenario.surfaces,
      route: this.scenario.route,
      mode: this.mode,
      running: !!this.timer,
      paused: this.paused,
      rateHz: this.rateHz,
      walkSpeed: this.walkSpeed,
      pulses: this.pulseCount,
      seed: this.seed,
      classifier: EchoNet ? 'loaded' : 'unavailable',
      classifierError: echonetError,
    };
  }
}

module.exports = {
  SimulationEngine,
  SCENARIOS,
  SCENARIO_LIST,
  raycast,
  EchoNet,
  echonetError,
  samplesToRange,
};
