/**
 * Approximate pose estimation.
 *
 * This is NOT SLAM and the UI says so.  A phone gives us a compass heading and
 * an accelerometer; from those we get a heading we half-trust and a position we
 * dead-reckon from counted steps.  Error accumulates, so `confidence` decays
 * with distance travelled since the last reset, and the map renders the phone
 * marker accordingly.
 *
 * Heading sources, best first:
 *   1. webkitCompassHeading — iOS/Chrome true compass degrees
 *   2. DeviceOrientationEvent.alpha with absolute=true — magnetometer-referenced
 *   3. alpha without absolute — arbitrary origin, so it is treated as relative
 *      and zeroed on calibration
 *   4. manual — the operator turns the heading dial themselves
 *
 * Position sources:
 *   1. step detection from devicemotion acceleration peaks
 *   2. manual walk control
 *   3. static
 */
export const POSE_METHODS = ['dead-reckoning', 'manual', 'static', 'simulated'];

export class PoseEstimator {
  constructor(opts = {}) {
    this.onUpdate = opts.onUpdate || (() => {});
    this.strideM = opts.strideM || 0.68;        // typical adult stride
    this.x = 0;
    this.y = 0;
    this.heading = 0;
    this.smoothedHeading = null;
    this.spikeFrames = 0;
    this.headingOffset = 0;
    this.headingSource = 'none';
    this.headingAbsolute = false;
    this.hasAbsoluteOrientation = false;
    this.poseMode = opts.poseMode || 'walk'; // default to dead-reckoning walking mode
    this.rawAlpha = null;
    this.pitch = 0;
    this.roll = 0;
    this.steps = 0;
    this.distance = 0;
    this.method = 'dead-reckoning';
    this.lastStepAt = 0;
    this.available = { orientation: false, motion: false, compass: false };
    this.permission = { orientation: 'unknown', motion: 'unknown' };

    // Step detector state: high-passed acceleration magnitude with hysteresis.
    this.accMag = 9.81;
    this.accSlow = 9.81;
    this.armed = false;
    this.peak = 0;
    this.stepThreshold = 1.6;                   // m/s^2 above slow average, responsive for natural walking
    this.minStepIntervalMs = 280;

    this.boundOrientation = (e) => this.onOrientation(e);
    this.boundMotion = (e) => this.onMotion(e);
    this.listening = false;
  }

  static supported() {
    return {
      orientation: typeof window !== 'undefined' && 'DeviceOrientationEvent' in window,
      motion: typeof window !== 'undefined' && 'DeviceMotionEvent' in window,
      // iOS 13+ gates both behind an explicit user-gesture permission call.
      needsPermission: typeof DeviceOrientationEvent !== 'undefined'
        && typeof DeviceOrientationEvent.requestPermission === 'function',
    };
  }

  /**
   * Ask for sensor access.  Must be called from a user gesture on iOS; on
   * Android Chrome it resolves immediately.  Never throws.
   */
  async requestPermission() {
    const out = { orientation: 'granted', motion: 'granted', errors: [] };
    try {
      if (typeof DeviceOrientationEvent !== 'undefined'
        && typeof DeviceOrientationEvent.requestPermission === 'function') {
        out.orientation = await DeviceOrientationEvent.requestPermission();
      }
    } catch (e) {
      out.orientation = 'denied';
      out.errors.push('orientation: ' + e.message);
    }
    try {
      if (typeof DeviceMotionEvent !== 'undefined'
        && typeof DeviceMotionEvent.requestPermission === 'function') {
        out.motion = await DeviceMotionEvent.requestPermission();
      }
    } catch (e) {
      out.motion = 'denied';
      out.errors.push('motion: ' + e.message);
    }
    this.permission = { orientation: out.orientation, motion: out.motion };
    return out;
  }

  start() {
    if (this.listening || typeof window === 'undefined') return;
    // deviceorientationabsolute is magnetometer-referenced where supported,
    // which is the difference between a real compass heading and a relative one.
    if ('ondeviceorientationabsolute' in window) {
      window.addEventListener('deviceorientationabsolute', this.boundOrientation, true);
    }
    window.addEventListener('deviceorientation', this.boundOrientation, true);
    window.addEventListener('devicemotion', this.boundMotion, true);
    this.listening = true;
  }

  stop() {
    if (!this.listening || typeof window === 'undefined') return;
    window.removeEventListener('deviceorientationabsolute', this.boundOrientation, true);
    window.removeEventListener('deviceorientation', this.boundOrientation, true);
    window.removeEventListener('devicemotion', this.boundMotion, true);
    this.listening = false;
  }

  onOrientation(e) {
    if (e == null) return;
    this.available.orientation = true;

    // Guard against dual-listener conflict on Android Chrome:
    // If deviceorientationabsolute is available, ignore relative deviceorientation events
    const isAbsolute = !!e.absolute || e.type === 'deviceorientationabsolute';
    if (isAbsolute) {
      this.hasAbsoluteOrientation = true;
    } else if (this.hasAbsoluteOrientation) {
      return; // Stick to the absolute stream; don't oscillate reference frames!
    }

    let rawH = null;
    if (typeof e.webkitCompassHeading === 'number' && !Number.isNaN(e.webkitCompassHeading)) {
      rawH = wrap(e.webkitCompassHeading);
      this.headingSource = 'compass';
      this.headingAbsolute = true;
      this.available.compass = true;
    } else if (typeof e.alpha === 'number' && !Number.isNaN(e.alpha)) {
      this.rawAlpha = e.alpha;
      // alpha counts anticlockwise from the device's reference; compass
      // heading is clockwise, hence 360 - alpha.
      rawH = wrap(360 - e.alpha + this.headingOffset);
      this.headingAbsolute = isAbsolute;
      this.headingSource = isAbsolute ? 'orientation-absolute' : 'orientation-relative';
    }

    // Circular low-pass EMA filter to eliminate sensor jitter
    if (rawH != null) {
      if (this.smoothedHeading == null) {
        this.smoothedHeading = rawH;
      } else {
        const diff = shortestAngleDiff(rawH, this.smoothedHeading);
        // Slew limiter: reject single-frame wild jumps (>85 deg) unless sustained
        if (Math.abs(diff) > 85 && this.spikeFrames < 3) {
          this.spikeFrames++;
        } else {
          this.spikeFrames = 0;
          this.smoothedHeading = wrap(this.smoothedHeading + diff * 0.25);
        }
      }
      this.heading = Math.round(this.smoothedHeading * 10) / 10;
    }

    if (typeof e.beta === 'number') this.pitch = e.beta;
    if (typeof e.gamma === 'number') this.roll = e.gamma;
    this.method = this.poseMode === 'walk' ? 'dead-reckoning' : 'static';
    this.emit();
  }

  onMotion(e) {
    // Only pause step detection if user explicitly locked to stationary rotation mode
    if (this.poseMode === 'rotation') return;

    const a = e && (e.accelerationIncludingGravity || e.acceleration);
    if (!a || a.x == null) return;
    this.available.motion = true;
    const mag = Math.sqrt((a.x || 0) ** 2 + (a.y || 0) ** 2 + (a.z || 0) ** 2);
    // Slow average approximates gravity; the residual is motion.
    this.accSlow += (mag - this.accSlow) * 0.02;
    this.accMag = mag;
    const residual = mag - this.accSlow;
    const now = Date.now();

    // Peak detection with hysteresis and a refractory period: one step per
    // acceleration peak, and no double counting on the rebound.
    if (!this.armed && residual > this.stepThreshold) {
      this.armed = true;
      this.peak = residual;
    } else if (this.armed) {
      if (residual > this.peak) this.peak = residual;
      if (residual < this.stepThreshold * 0.35) {
        this.armed = false;
        if (now - this.lastStepAt > this.minStepIntervalMs) {
          this.lastStepAt = now;
          this.registerStep();
        }
      }
    }
  }

  /** One step forward along the current heading. */
  registerStep() {
    this.steps++;
    // Stride grows slightly with cadence; a fast walker covers more per step.
    const cadence = this.lastStepInterval ? 1000 / this.lastStepInterval : 2;
    const stride = this.strideM * (0.85 + 0.15 * Math.min(2, cadence));
    this.advance(stride);
    this.method = 'dead-reckoning';
    this.emit();
  }

  advance(metres) {
    const r = (this.heading * Math.PI) / 180;
    this.x += metres * Math.sin(r);
    this.y += metres * Math.cos(r);
    this.distance += Math.abs(metres);
  }

  /** Manual controls, for when the sensors are unavailable or untrusted. */
  manualWalk(metres) {
    this.advance(metres);
    this.method = 'manual';
    this.emit();
  }

  manualTurn(deg) {
    if (this.headingSource === 'none' || this.headingSource === 'manual') {
      this.heading = wrap(this.heading + deg);
      this.smoothedHeading = this.heading;
      this.headingSource = 'manual';
    } else {
      // A live sensor stays in charge; the operator is trimming its origin.
      this.headingOffset = wrap(this.headingOffset + deg);
      if (this.rawAlpha != null) {
        const raw = wrap(360 - this.rawAlpha + this.headingOffset);
        this.smoothedHeading = raw;
        this.heading = raw;
      }
    }
    this.emit();
  }

  setHeading(deg) {
    this.heading = wrap(deg);
    this.smoothedHeading = wrap(deg);
    this.headingSource = 'manual';
    this.emit();
  }

  setPoseMode(mode) {
    this.poseMode = mode === 'walk' ? 'walk' : 'rotation';
    this.method = this.poseMode === 'walk' ? 'dead-reckoning' : 'static';
    this.emit();
    return this.poseMode;
  }

  /** Declare "this is where I am, facing this way" — resets accumulated drift. */
  zero(headingDeg) {
    this.x = 0;
    this.y = 0;
    this.steps = 0;
    this.distance = 0;
    if (headingDeg != null) {
      if (this.rawAlpha != null) this.headingOffset = wrap(headingDeg - (360 - this.rawAlpha));
      this.heading = wrap(headingDeg);
      this.smoothedHeading = wrap(headingDeg);
    } else if (this.smoothedHeading != null) {
      this.heading = Math.round(this.smoothedHeading * 10) / 10;
    }
    this.emit();
  }

  /**
   * Pose confidence.
   *
   * Starts high and decays with dead-reckoned distance: step-counted position
   * error grows roughly linearly, so after ~20 m of walking the position is
   * worth very little and the map should show that.  A heading without a
   * magnetometer reference is capped lower still.
   */
  confidence() {
    if (this.method === 'static') return 0.55;
    const driftTerm = Math.max(0.12, 1 - this.distance / 22);
    const headingTerm = this.headingAbsolute ? 1.0
      : this.headingSource === 'manual' ? 0.75
        : this.headingSource === 'none' ? 0.4 : 0.7;
    return Math.max(0.05, Math.min(0.95, driftTerm * headingTerm));
  }

  pose() {
    return {
      x: this.x,
      y: this.y,
      heading: this.heading,
      confidence: this.confidence(),
      method: this.method,
    };
  }

  status() {
    return {
      pose: this.pose(),
      headingSource: this.headingSource,
      headingAbsolute: this.headingAbsolute,
      pitch: Math.round(this.pitch),
      roll: Math.round(this.roll),
      steps: this.steps,
      distance: Math.round(this.distance * 100) / 100,
      available: Object.assign({}, this.available),
      permission: Object.assign({}, this.permission),
      listening: this.listening,
      accMag: Math.round(this.accMag * 100) / 100,
      // Said plainly, because the map's honesty depends on it.
      note: this.headingAbsolute
        ? 'Heading is magnetometer-referenced. Position is dead-reckoned from step counting and drifts.'
        : 'Heading has no magnetic reference (relative only). Position is estimated, not surveyed.',
    };
  }

  emit() { this.onUpdate(this.pose(), this); }
}

function wrap(d) { const x = d % 360; return x < 0 ? x + 360 : x; }

function shortestAngleDiff(target, source) {
  let diff = (target - source) % 360;
  if (diff < -180) diff += 360;
  if (diff > 180) diff -= 360;
  return diff;
}
