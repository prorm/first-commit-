/**
 * Approximate pose estimation.
 *
 * This is NOT SLAM and the UI says so.  A phone gives us an orientation triple
 * and an accelerometer; from those we get a heading we half-trust and a
 * position we dead-reckon from counted steps.  Error accumulates, so
 * `confidence` decays with distance travelled since the last reset, and the map
 * renders the phone marker accordingly.
 *
 * Two things here exist because the naive version of each produced a map that
 * looked like a drunk walk, and both are worth stating plainly:
 *
 * 1. HEADING IS TILT-COMPENSATED.  `alpha` alone is only a compass bearing
 *    when the phone lies flat.  Aim a phone at a wall — which is the entire
 *    point of this device — and beta approaches 90 deg, where alpha and gamma
 *    become degenerate and a 2 deg physical wobble swings alpha by tens of
 *    degrees.  So we build the full device->world rotation from (alpha, beta,
 *    gamma), take the axis the phone is actually pointing along, project it
 *    onto the horizontal plane, and read the bearing off that.  Near-vertical
 *    aim has no bearing at all and we hold the last one rather than invent one.
 *
 * 2. STEPS REQUIRE A GAIT, NOT A BUMP.  Sweeping a phone through an arc by
 *    hand produces acceleration peaks indistinguishable from footfalls by
 *    threshold alone.  Scanning a room from a chair therefore used to
 *    dead-reckon tens of metres across a room the operator never left, and
 *    every echo was pinned at a fictional place.  A peak is now only a step if
 *    the gyro says the phone is not being swept AND the peak lands in a
 *    sustained, evenly-spaced run.  Default mode is stationary: you opt in to
 *    walking, not out of it.
 *
 * Heading sources, best first:
 *   1. webkitCompassHeading — iOS/Chrome true compass degrees
 *   2. DeviceOrientationEvent with absolute=true — magnetometer-referenced
 *   3. orientation without absolute — arbitrary origin, so it is treated as
 *      relative and zeroed on calibration
 *   4. manual — the operator turns the heading dial themselves
 *
 * Position sources:
 *   1. step detection from devicemotion acceleration peaks (walk mode only)
 *   2. manual walk control
 *   3. static
 */
export const POSE_METHODS = ['dead-reckoning', 'manual', 'static', 'simulated'];

const D2R = Math.PI / 180;
const R2D = 180 / Math.PI;

/**
 * The two candidate boresights, in world coordinates (east, north, up).
 *
 * W3C device orientation is R = Rz(alpha)*Rx(beta)*Ry(gamma), mapping the
 * device frame (+x right of screen, +y top of screen, +z out of screen) into
 * the Earth frame (+x east, +y north, +z up).  We need two of its columns:
 *
 *   `flat` — the top edge of the screen, i.e. where a compass app points when
 *            the phone is held level in front of you.
 *   `aim`  — straight out the back of the phone, i.e. where the phone points
 *            when you hold it upright and aim it at a wall like a torch.
 *
 * `screenDeg` is screen.orientation.angle, so a landscape-rotated UI still
 * calls the same physical edge "up".
 */
export function orientationAxes(aDeg, bDeg, gDeg, screenDeg = 0) {
  const a = aDeg * D2R;
  const b = bDeg * D2R;
  const g = gDeg * D2R;
  const cA = Math.cos(a); const sA = Math.sin(a);
  const cB = Math.cos(b); const sB = Math.sin(b);
  const cG = Math.cos(g); const sG = Math.sin(g);

  // Columns of R: the world-frame images of the device's own x, y, z axes.
  const xE = cA * cG - sA * sB * sG;
  const xN = sA * cG + cA * sB * sG;
  const xU = -cB * sG;

  const yE = -sA * cB;
  const yN = cA * cB;
  const yU = sB;

  const zE = cA * sG + sA * sB * cG;
  const zN = sA * sG - cA * sB * cG;
  const zU = cB * cG;

  const s = screenDeg * D2R;
  const ss = Math.sin(s); const cs = Math.cos(s);

  return {
    flat: { e: -xE * ss + yE * cs, n: -xN * ss + yN * cs, u: -xU * ss + yU * cs },
    aim: { e: -zE, n: -zN, u: -zU },
  };
}

/**
 * Compass bearing of a world-frame axis, or null when the axis is too close to
 * vertical for its horizontal projection to mean anything.
 */
export function axisHeading(axis, minHorizontal = 0.17) {
  const h = Math.hypot(axis.e, axis.n);
  if (!(h > minHorizontal)) return null;
  return wrap(Math.atan2(axis.e, axis.n) * R2D);
}

export class PoseEstimator {
  constructor(opts = {}) {
    this.onUpdate = opts.onUpdate || (() => {});
    this.strideM = opts.strideM || 0.68;        // typical adult stride
    this.x = 0;
    this.y = 0;
    this.heading = 0;
    this.smoothedHeading = null;
    this.rawHeading = null;                     // untrimmed, straight off the sensor
    this.headingRate = 0;                       // deg/s, smoothed; how fast you are sweeping
    this.lastHeadingAt = 0;
    this.spikeFrames = 0;
    this.headingOffset = 0;
    this.headingSource = 'none';
    this.headingAbsolute = false;
    this.hasAbsoluteOrientation = false;
    // Stationary by default.  Scanning a room from one spot is the common case
    // and the one that phantom steps ruin; walking is the deliberate choice.
    this.poseMode = opts.poseMode === 'walk' ? 'walk' : 'rotation';
    this.aimAxis = null;                        // 'flat' or 'aim', chosen per sample
    this.pitch = 0;
    this.roll = 0;
    this.steps = 0;                             // steps that actually moved us
    this.stepCandidates = 0;                    // acceleration peaks that looked step-shaped
    this.distance = 0;
    this.method = 'static';
    this.lastStepAt = 0;
    this.lastStepInterval = 0;
    this.gaitRun = 0;
    this.available = { orientation: false, motion: false, compass: false, gyro: false };
    this.permission = { orientation: 'unknown', motion: 'unknown' };

    // Step detector state: high-passed acceleration magnitude with hysteresis.
    this.accMag = 9.81;
    this.accSlow = 9.81;
    this.rotRate = 0;                           // deg/s, smoothed gyro magnitude
    this.armed = false;
    this.peak = 0;
    this.stepThreshold = opts.stepThreshold || 0.8;  // m/s^2 above slow average
    this.minStepIntervalMs = 280;
    this.maxStepIntervalMs = 1100;              // slower than this is not a gait
    this.maxStepRotDps = opts.maxStepRotDps || 45;   // above this you are waving, not walking
    // Evenly-spaced peaks before we believe it. Do not lower this to make walking
    // register sooner: at 1 every bump is a step, and a hand-held phone snakes the
    // map across a room nobody walked (tests/pose.test.js pins it at 3).
    this.gaitConfirm = opts.gaitConfirm || 3;

    // Heading filter.
    this.headingDeadbandDeg = 0.4;              // below this it is sensor shimmer
    this.headingGainMin = 0.15;
    this.headingGainMax = 0.5;

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

  /** screen.orientation.angle, or 0 anywhere that does not expose it. */
  screenAngle() {
    if (typeof screen === 'undefined' || !screen) return 0;
    if (screen.orientation && typeof screen.orientation.angle === 'number') {
      return screen.orientation.angle;
    }
    return typeof window !== 'undefined' && typeof window.orientation === 'number'
      ? window.orientation : 0;
  }

  /**
   * Pick the boresight to read the bearing off.
   *
   * Whichever candidate axis is closer to horizontal is the one carrying real
   * bearing information; the other is pointing at the floor or the ceiling and
   * its projection is noise.  Hysteresis stops the two swapping back and forth
   * at the crossover, which would read as a heading that jumps by 90 deg
   * whenever you tilt through 45.
   */
  _chooseAxis(axes) {
    const flatV = Math.abs(axes.flat.u);
    const aimV = Math.abs(axes.aim.u);
    if (this.aimAxis == null) {
      this.aimAxis = aimV < flatV ? 'aim' : 'flat';
    } else if (this.aimAxis === 'flat' && aimV < flatV - 0.12) {
      this.aimAxis = 'aim';
    } else if (this.aimAxis === 'aim' && flatV < aimV - 0.12) {
      this.aimAxis = 'flat';
    }
    return axes[this.aimAxis];
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

    let sensorH = null;
    if (typeof e.webkitCompassHeading === 'number' && !Number.isNaN(e.webkitCompassHeading)) {
      sensorH = wrap(e.webkitCompassHeading);
      this.headingSource = 'compass';
      this.headingAbsolute = true;
      this.available.compass = true;
    } else if (typeof e.alpha === 'number' && !Number.isNaN(e.alpha)) {
      const axes = orientationAxes(e.alpha, num(e.beta), num(e.gamma), this.screenAngle());
      sensorH = axisHeading(this._chooseAxis(axes));
      this.headingAbsolute = isAbsolute;
      this.headingSource = isAbsolute ? 'orientation-absolute' : 'orientation-relative';
    }

    if (typeof e.beta === 'number') this.pitch = e.beta;
    if (typeof e.gamma === 'number') this.roll = e.gamma;

    // sensorH is null when the phone is aimed at the floor or the ceiling and
    // has no horizontal bearing at all.  Hold the last good heading; anything
    // else would spin the map while the operator looks down.
    if (sensorH != null) {
      this.rawHeading = sensorH;
      this._applyHeading(wrap(sensorH + this.headingOffset));
    }

    this.method = this.poseMode === 'walk' ? 'dead-reckoning' : 'static';
    this.emit();
  }

  /**
   * Circular low-pass with a deadband and a rate-adaptive gain: heavy smoothing
   * while you hold still (where the residual is all sensor noise), light
   * smoothing through a deliberate turn (where lag would smear every echo
   * across the arc).
   */
  _applyHeading(rawH) {
    if (this.smoothedHeading == null) {
      this.smoothedHeading = rawH;
      this.heading = Math.round(rawH * 10) / 10;
      return;
    }
    const diff = shortestAngleDiff(rawH, this.smoothedHeading);

    // Slew limiter: reject single-frame wild jumps (>85 deg) unless sustained
    if (Math.abs(diff) > 85 && this.spikeFrames < 3) {
      this.spikeFrames++;
      return;
    }
    this.spikeFrames = 0;
    if (Math.abs(diff) < this.headingDeadbandDeg) return;

    const k = Math.min(this.headingGainMax,
      this.headingGainMin + 0.35 * Math.min(1, Math.abs(diff) / 12));
    const applied = diff * k;
    this.smoothedHeading = wrap(this.smoothedHeading + applied);
    this.heading = Math.round(this.smoothedHeading * 10) / 10;

    const now = nowMs();
    if (this.lastHeadingAt) {
      const dt = (now - this.lastHeadingAt) / 1000;
      if (dt > 0.002 && dt < 1) {
        const inst = Math.abs(applied) / dt;
        this.headingRate += (inst - this.headingRate) * 0.2;
      }
    }
    this.lastHeadingAt = now;
  }

  onMotion(e) {
    const rr = e && e.rotationRate;
    if (rr && (rr.alpha != null || rr.beta != null || rr.gamma != null)) {
      this.available.gyro = true;
      const m = Math.hypot(num(rr.alpha), num(rr.beta), num(rr.gamma));
      this.rotRate += (m - this.rotRate) * 0.25;
    }

    const a = e && (e.accelerationIncludingGravity || e.acceleration);
    if (!a || a.x == null) return;
    this.available.motion = true;
    const mag = Math.sqrt((a.x || 0) ** 2 + (a.y || 0) ** 2 + (a.z || 0) ** 2);
    // Slow average approximates gravity; the residual is motion.  Tracked in
    // every mode so switching to walk does not start from a stale baseline.
    this.accSlow += (mag - this.accSlow) * 0.02;
    this.accMag = mag;
    if (this.poseMode !== 'walk') return;

    const residual = mag - this.accSlow;
    const now = Date.now();

    // Peak detection with hysteresis and a refractory period: one candidate per
    // acceleration peak, and no double counting on the rebound.
    if (!this.armed && residual > this.stepThreshold) {
      this.armed = true;
      this.peak = residual;
    } else if (this.armed) {
      if (residual > this.peak) this.peak = residual;
      if (residual < this.stepThreshold * 0.35) {
        this.armed = false;
        if (now - this.lastStepAt > this.minStepIntervalMs) this._candidateStep(now);
      }
    }
  }

  /**
   * A step-shaped acceleration peak.  Whether it moves us is a separate
   * question, and the one that matters: see the gait note at the top.
   */
  _candidateStep(now) {
    const interval = this.lastStepAt ? now - this.lastStepAt : 0;
    this.lastStepAt = now;
    this.stepCandidates++;

    // The gyro is the discriminator.  A hand sweeping the phone across a room
    // turns at 60-200 deg/s; a walker holding a phone steady turns far slower,
    // and their footfalls arrive anyway.
    if (this.available.gyro && this.rotRate > this.maxStepRotDps) {
      this.gaitRun = 0;
      this.lastStepInterval = 0;
      return;
    }

    const inWindow = interval >= this.minStepIntervalMs && interval <= this.maxStepIntervalMs;
    const steady = this.lastStepInterval > 0
      && interval > this.lastStepInterval * 0.6
      && interval < this.lastStepInterval * 1.7;
    if (inWindow && (this.lastStepInterval === 0 || steady)) this.gaitRun++;
    else this.gaitRun = inWindow ? 1 : 0;
    this.lastStepInterval = inWindow ? interval : 0;

    if (this.gaitRun < this.gaitConfirm) return;

    // On the frame the gait is confirmed, credit the peaks that established it:
    // those were real steps, we just were not sure yet.
    const credit = this.gaitRun === this.gaitConfirm ? this.gaitConfirm : 1;
    for (let i = 0; i < credit; i++) this.registerStep(interval, i === credit - 1);
  }

  /** One step forward along the current heading. */
  registerStep(intervalMs, emit = true) {
    this.steps++;
    // Stride grows slightly with cadence; a fast walker covers more per step.
    const cadence = intervalMs > 0 ? 1000 / intervalMs : 2;
    const stride = this.strideM * (0.85 + 0.15 * Math.min(2, cadence));
    this.advance(stride);
    this.method = 'dead-reckoning';
    if (emit) this.emit();
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
      if (this.rawHeading != null) {
        const raw = wrap(this.rawHeading + this.headingOffset);
        this.smoothedHeading = raw;
        this.heading = Math.round(raw * 10) / 10;
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
    // Entering walk mode must not inherit a half-finished gait from whatever
    // the phone was doing while locked.
    this.gaitRun = 0;
    this.lastStepInterval = 0;
    this.armed = false;
    this.emit();
    return this.poseMode;
  }

  /** Declare "this is where I am, facing this way" — resets accumulated drift. */
  zero(headingDeg) {
    this.x = 0;
    this.y = 0;
    this.steps = 0;
    this.stepCandidates = 0;
    this.distance = 0;
    this.gaitRun = 0;
    this.lastStepInterval = 0;
    if (headingDeg != null) {
      if (this.rawHeading != null) this.headingOffset = wrap(headingDeg - this.rawHeading);
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
   * Dead-reckoned position error grows roughly linearly with distance walked,
   * so after ~20 m the position is worth very little and the map should show
   * that.  A stationary scan has no such term — the operator asserted the
   * position and did not move — but both are still limited by how well the
   * heading is known, and by whether the phone was being swept at the moment
   * the bearing was taken.
   */
  confidence() {
    const headingTerm = this.headingAbsolute ? 1.0
      : this.headingSource === 'manual' ? 0.75
        : this.headingSource === 'none' ? 0.4 : 0.7;
    // A bearing read mid-sweep is smeared across the arc the phone covered.
    const slewTerm = Math.max(0.45, 1 - this.headingRate / 220);
    const driftTerm = (this.poseMode === 'walk' || this.distance > 0)
      ? Math.max(0.12, 1 - this.distance / 22)
      : 0.92;
    return Math.max(0.05, Math.min(0.95, driftTerm * headingTerm * slewTerm));
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
      poseMode: this.poseMode,
      headingSource: this.headingSource,
      headingAbsolute: this.headingAbsolute,
      headingRate: Math.round(this.headingRate),
      aimAxis: this.aimAxis,
      pitch: Math.round(this.pitch),
      roll: Math.round(this.roll),
      steps: this.steps,
      stepCandidates: this.stepCandidates,
      rotRate: Math.round(this.rotRate),
      distance: Math.round(this.distance * 100) / 100,
      available: Object.assign({}, this.available),
      permission: Object.assign({}, this.permission),
      listening: this.listening,
      accMag: Math.round(this.accMag * 100) / 100,
      // Said plainly, because the map's honesty depends on it.
      note: this.poseMode !== 'walk'
        ? 'Stationary scan: position is locked at the origin and only the heading moves.'
        : this.headingAbsolute
          ? 'Heading is magnetometer-referenced. Position is dead-reckoned from step counting and drifts.'
          : 'Heading has no magnetic reference (relative only). Position is estimated, not surveyed.',
    };
  }

  emit() { this.onUpdate(this.pose(), this); }
}

function num(v, d = 0) { return typeof v === 'number' && Number.isFinite(v) ? v : d; }

function nowMs() {
  return typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now();
}

function wrap(d) { const x = d % 360; return x < 0 ? x + 360 : x; }

function shortestAngleDiff(target, source) {
  let diff = (target - source) % 360;
  if (diff < -180) diff += 360;
  if (diff > 180) diff -= 360;
  return diff;
}
