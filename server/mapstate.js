/**
 * Authoritative fused map state.
 *
 * The server owns one MapState per mission.  It is the thing that survives a
 * phone reconnect or a command-center refresh: both re-sync from a snapshot
 * rather than starting over, which is what keeps a demo alive through flaky
 * venue WiFi.
 *
 * The heavy lifting lives in public/shared/*.mjs and is shared verbatim with
 * the browser, so the command center's local rendering and the server's
 * authoritative copy can never drift apart.
 */
const { OccupancyGrid, PointCloud, Trajectory, ClassFuser } = require('../public/shared/spatial.mjs');
const { reconstruct } = require('../public/shared/reconstruct.mjs');
const { makePose, CLASSES } = require('../public/shared/protocol.mjs');

class MapState {
  constructor(opts = {}) {
    this.grid = new OccupancyGrid({ cell: opts.cell || 0.1, halfExtent: opts.halfExtent || 12 });
    this.cloud = new PointCloud();
    this.trajectory = new Trajectory();
    this.fuser = new ClassFuser();
    this.reset(true);
  }

  reset(hard = false) {
    this.grid.reset();
    this.cloud.reset();
    this.trajectory.reset();
    this.fuser.reset();
    this.pose = makePose({ confidence: 0.8, method: 'static' });
    this.recent = [];                 // rolling detection tail for the AI panel
    this.stats = {
      detections: 0, cfarPass: 0, missionStart: null, missionEnd: null,
      classCounts: { WALL: 0, SOFT: 0, OPENING: 0 },
      confidenceSum: 0, classConfidenceSum: 0,
      minRange: Infinity, maxRange: 0,
    };
    this.reconstruction = null;
    this.reconstructionAt = 0;
    this.lastDetection = null;
    if (hard) this.missionActive = false;
  }

  startMission() {
    this.reset();
    this.missionActive = true;
    this.stats.missionStart = Date.now();
    this.trajectory.push(this.pose, Date.now());
  }

  endMission() {
    this.missionActive = false;
    this.stats.missionEnd = Date.now();
    return this.summary();
  }

  /** Pose-only update (phone turning or stepping between pulses). */
  applyPose(poseLike) {
    this.pose = makePose(poseLike);
    this.trajectory.push(this.pose, Date.now());
    return this.pose;
  }

  /**
   * Integrate one detection.  Returns the detection with server-side temporal
   * fusion applied, which is what gets broadcast to the command center.
   */
  applyDetection(det) {
    this.pose = det.phone;
    this.trajectory.push(this.pose, det.t);

    const fused = this.fuser.fuse(det);
    const out = Object.assign({}, det, {
      fusedClass: fused.className,
      fusedConfidence: fused.confidence,
      fusedProbs: fused.probs,
      fusedHistory: fused.history,
      fusedSupport: fused.support,
      fusedStable: fused.stable,
    });

    // Grid weight folds detection confidence with pose confidence: a detection
    // taken from a badly-known position should not carve crisp geometry.
    const weight = det.confidence * (0.35 + 0.65 * det.phone.confidence);
    this.grid.integrate(det.phone, det.bearing_deg, det.range_m, {
      weight,
      className: out.fusedClass,
      beamwidth_deg: det.beamwidth_deg,
    });

    if (out.fusedClass !== 'OPENING') this.cloud.add(out);
    else this.cloud.add(out);           // kept, but reconstruct() treats it as absence evidence

    const s = this.stats;
    s.detections++;
    if (det.cfar_pass) s.cfarPass++;
    if (out.fusedClass && s.classCounts[out.fusedClass] != null) s.classCounts[out.fusedClass]++;
    s.confidenceSum += det.confidence;
    s.classConfidenceSum += out.fusedConfidence || 0;
    if (det.range_m < s.minRange) s.minRange = det.range_m;
    if (det.range_m > s.maxRange) s.maxRange = det.range_m;

    this.lastDetection = out;
    this.recent.push(out);
    if (this.recent.length > 40) this.recent.shift();
    return out;
  }

  /** Run surface reconstruction (cached briefly — it is the expensive step). */
  buildReconstruction(force = false) {
    const now = Date.now();
    if (!force && this.reconstruction && now - this.reconstructionAt < 700) return this.reconstruction;
    this.reconstruction = reconstruct(this.cloud.points);
    this.reconstructionAt = now;
    return this.reconstruction;
  }

  /** Everything the command center needs to draw the world from scratch. */
  snapshot(opts = {}) {
    const recon = opts.reconstruct ? this.buildReconstruction() : this.reconstruction;
    return {
      pose: this.pose,
      grid: this.grid.serialize(),
      cloud: this.cloud.serialize(),
      trajectory: this.trajectory.serialize(),
      reconstruction: recon,
      stats: this.publicStats(),
      recent: this.recent.slice(-12),
      missionActive: this.missionActive,
      lastDetection: this.lastDetection,
    };
  }

  publicStats() {
    const s = this.stats;
    const cloudStats = this.cloud.stats();
    const area = this.grid.areaStats();
    return {
      detections: s.detections,
      cfarPass: s.cfarPass,
      cfarRate: s.detections ? s.cfarPass / s.detections : 0,
      classCounts: Object.assign({}, s.classCounts),
      avgConfidence: s.detections ? s.confidenceSum / s.detections : 0,
      avgClassConfidence: s.detections ? s.classConfidenceSum / s.detections : 0,
      minRange: Number.isFinite(s.minRange) ? s.minRange : null,
      maxRange: s.maxRange || null,
      distanceScanned: this.trajectory.distance,
      coverage: this.grid.coverage(),
      freeArea: area.freeArea,
      occupiedArea: area.occupiedArea,
      cloudPoints: cloudStats.total,
      cloudByClass: { WALL: cloudStats.WALL, SOFT: cloudStats.SOFT, OPENING: cloudStats.OPENING },
      missionStart: s.missionStart,
      missionEnd: s.missionEnd,
      elapsedMs: s.missionStart ? (s.missionEnd || Date.now()) - s.missionStart : 0,
    };
  }

  /** End-of-mission report.  Every number here is measured, none are decorative. */
  summary() {
    const recon = this.buildReconstruction(true);
    const stats = this.publicStats();
    const openings = (recon.openings || []).filter((o) => o.confidence > 0.25);
    return {
      stats,
      reconstruction: {
        segments: recon.segments.length,
        totalWallLength: recon.stats.totalLength || 0,
        corners: recon.corners.length,
        openings: openings.length,
        openingDetail: openings.slice(0, 6),
        corridors: recon.corridors.length,
        confidence: recon.confidence,
        inlierRatio: recon.stats.inlierRatio || 0,
      },
      // Spelled out so the summary screen never has to invent a caveat.
      caveats: [
        'Ranges are estimated from a single-element near-ultrasonic sensor; bearing is the phone boresight, not a resolved angle.',
        'Surface labels (WALL / SOFT) come from EchoNet and are EXPERIMENTAL: 39 % accuracy on real recordings held out by session, against a 33 % chance floor. Range, velocity and time-to-contact do not depend on them.',
        'Openings are geometric candidates - a door-width gap in an otherwise continuous run of reconstructed wall - not classifier output, and not confirmed doorways.',
        'Phone position is dead-reckoned or simulated, not surveyed.',
      ],
    };
  }
}

module.exports = { MapState, CLASSES };
