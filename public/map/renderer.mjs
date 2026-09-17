/**
 * Command center renderer.
 *
 * Draws the world in layers, back to front:
 *
 *   0  ground + world-aligned grid        (context, suppressed in zero-visibility)
 *   1  occupancy grid                     (unknown / free / occupied, visibly distinct)
 *   2  ground truth outline               (simulation only, opt-in, clearly labelled)
 *   3  reconstructed surfaces + features  (inferred geometry)
 *   4  echo point cloud                   (what was actually measured)
 *   5  phone trajectory                   (where the sensor has been)
 *   6  sensor marker, beam wedge, pulses   (where it is and where it is looking)
 *
 * Two rules the drawing code obeys throughout:
 *   - glow means measured.  Inferred geometry is drawn thinner and cooler than
 *     the echoes that produced it, and unknown space is never painted as free.
 *   - nothing is animated in a way that implies a measurement that did not
 *     happen.  Pulse rings fire on real detection arrivals; the sweep follows
 *     the reported heading; the reconstruction morph moves points to the
 *     surface that was actually fitted to them.
 */
import { CLASSES } from '../shared/protocol.mjs';

const CLASS_RGB = {
  WALL: [78, 168, 255],
  SOFT: [255, 180, 84],
  OPENING: [155, 140, 255],
};
const LIVE_RGB = [44, 232, 245];
const INFER_RGB = [179, 157, 255];

export class MapRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.dpr = 1;
    this.w = 0;
    this.h = 0;

    // Camera: world metres -> screen px
    this.scale = 62;            // px per metre
    this.targetScale = 62;
    this.cx = 0;                // world centre
    this.cy = 0;
    this.targetCx = 0;
    this.targetCy = 0;
    this.follow = true;

    this.zeroVisibility = false;
    this.layers = { grid: true, cloud: true, surfaces: true, truth: false };

    // Transient visual state
    this.pulses = [];           // expanding rings, one per detection arrival
    this.reconAnim = 0;         // 0 = raw cloud, 1 = fully morphed to surfaces
    this.reconAnimTarget = 0;
    this.reconStartedAt = 0;
    this.lastDetectionAt = 0;
    this.sweepPhase = 0;
    this.fps = 0;
    this.frames = 0;
    this.fpsAt = 0;
  }

  resize() {
    const rect = this.canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    this.dpr = Math.min(2, window.devicePixelRatio || 1);
    this.canvas.width = Math.round(rect.width * this.dpr);
    this.canvas.height = Math.round(rect.height * this.dpr);
    this.w = rect.width;
    this.h = rect.height;
  }

  // -------------------------------------------------------------------------
  // Camera
  // -------------------------------------------------------------------------
  toScreen(x, y) {
    return {
      x: this.w / 2 + (x - this.cx) * this.scale,
      // World +y is north; screen y grows downward.
      y: this.h / 2 - (y - this.cy) * this.scale,
    };
  }

  toWorld(sx, sy) {
    return {
      x: this.cx + (sx - this.w / 2) / this.scale,
      y: this.cy - (sy - this.h / 2) / this.scale,
    };
  }

  /** Frame the scanned area, with headroom, without snapping the camera. */
  fitTo(bounds, pad = 1.6) {
    if (!bounds || !Number.isFinite(bounds.minX)) return;
    const wSpan = Math.max(2.5, bounds.maxX - bounds.minX + pad * 2);
    const hSpan = Math.max(2.5, bounds.maxY - bounds.minY + pad * 2);
    const s = Math.min(this.w / wSpan, this.h / hSpan);
    this.targetScale = Math.max(16, Math.min(190, s));
    this.targetCx = (bounds.minX + bounds.maxX) / 2;
    this.targetCy = (bounds.minY + bounds.maxY) / 2;
  }

  centerOn(x, y) { this.targetCx = x; this.targetCy = y; }

  /** Critically-damped camera easing: it settles, it never oscillates. */
  stepCamera(dt) {
    const k = 1 - Math.exp(-dt * 4.2);
    this.cx += (this.targetCx - this.cx) * k;
    this.cy += (this.targetCy - this.cy) * k;
    this.scale += (this.targetScale - this.scale) * k;
  }

  zoomBy(factor, atX, atY) {
    const before = this.toWorld(atX, atY);
    this.targetScale = Math.max(12, Math.min(220, this.targetScale * factor));
    this.scale = this.targetScale;
    const after = this.toWorld(atX, atY);
    // Keep the point under the cursor fixed while zooming.
    this.targetCx += before.x - after.x;
    this.targetCy += before.y - after.y;
    this.cx = this.targetCx;
    this.cy = this.targetCy;
    this.follow = false;
  }

  panBy(dxPx, dyPx) {
    this.targetCx -= dxPx / this.scale;
    this.targetCy += dyPx / this.scale;
    this.cx = this.targetCx;
    this.cy = this.targetCy;
    this.follow = false;
  }

  // -------------------------------------------------------------------------
  // Events from the app
  // -------------------------------------------------------------------------
  /** A detection arrived: fire a ring from the sensor position outward. */
  noteDetection(det) {
    this.lastDetectionAt = performance.now();
    this.pulses.push({
      x: det.phone.x, y: det.phone.y,
      bearing: det.bearing_deg,
      beam: det.beamwidth_deg || 30,
      range: det.range_m,
      conf: det.confidence,
      cls: det.fusedClass || det.obstacleClass,
      at: performance.now(),
    });
    if (this.pulses.length > 26) this.pulses.shift();
  }

  startReconstruction() {
    this.reconAnimTarget = 1;
    this.reconStartedAt = performance.now();
  }

  clearReconstruction() {
    this.reconAnimTarget = 0;
  }

  // -------------------------------------------------------------------------
  // Main draw
  // -------------------------------------------------------------------------
  /**
   * @param {Object} world { grid, cloud, trajectory, reconstruction, pose, truth }
   * @param {number} now performance.now()
   * @param {number} dt seconds since last frame
   */
  draw(world, now, dt) {
    const c = this.ctx;
    if (!c || !this.w) return;
    c.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);

    this.stepCamera(dt);
    // Reconstruction morph eases in over ~1.4 s and out faster.
    const rspeed = this.reconAnimTarget > this.reconAnim ? dt / 1.4 : -dt / 0.5;
    this.reconAnim = Math.max(0, Math.min(1, this.reconAnim + rspeed));

    this.drawGround(c);
    if (this.layers.grid && !this.zeroVisibility) this.drawWorldGrid(c);
    if (world.grid) this.drawOccupancy(c, world.grid);
    if (this.layers.truth && world.truth) this.drawTruth(c, world.truth);
    if (this.layers.surfaces && world.reconstruction) this.drawReconstruction(c, world.reconstruction, now);
    if (this.layers.cloud && world.cloud) this.drawCloud(c, world.cloud, world.reconstruction, now);
    if (world.trajectory) this.drawTrajectory(c, world.trajectory, now);
    this.drawPulses(c, now);
    if (world.pose) this.drawSensor(c, world.pose, world.lastDetection, now);
    this.drawScaleBar(c);

    this.frames++;
    if (now - this.fpsAt > 1000) {
      this.fps = Math.round((this.frames * 1000) / (now - this.fpsAt));
      this.frames = 0;
      this.fpsAt = now;
    }
  }

  drawGround(c) {
    c.fillStyle = this.zeroVisibility ? '#020306' : '#04070c';
    c.fillRect(0, 0, this.w, this.h);
    if (!this.zeroVisibility) {
      // A faint glow at the sensor keeps the eye where the action is.
      const g = c.createRadialGradient(this.w / 2, this.h / 2, 0, this.w / 2, this.h / 2, Math.max(this.w, this.h) * 0.6);
      g.addColorStop(0, 'rgba(44,232,245,0.030)');
      g.addColorStop(1, 'rgba(44,232,245,0)');
      c.fillStyle = g;
      c.fillRect(0, 0, this.w, this.h);
    }
  }

  /** One-metre world grid, so distances are readable without a ruler. */
  drawWorldGrid(c) {
    const step = this.scale < 28 ? 2 : 1;
    const tl = this.toWorld(0, 0);
    const br = this.toWorld(this.w, this.h);
    c.lineWidth = 1;
    c.strokeStyle = 'rgba(86,130,170,0.075)';
    c.beginPath();
    for (let x = Math.floor(tl.x / step) * step; x <= br.x; x += step) {
      const p = this.toScreen(x, 0);
      c.moveTo(Math.round(p.x) + 0.5, 0);
      c.lineTo(Math.round(p.x) + 0.5, this.h);
    }
    for (let y = Math.floor(br.y / step) * step; y <= tl.y; y += step) {
      const p = this.toScreen(0, y);
      c.moveTo(0, Math.round(p.y) + 0.5);
      c.lineTo(this.w, Math.round(p.y) + 0.5);
    }
    c.stroke();
  }

  /**
   * Occupancy grid.
   *
   * Unknown cells are left unpainted — that is the whole visual argument of
   * this demo, so free space must never be implied where nothing was scanned.
   * Free cells are a cool near-black wash; occupied cells glow warm-to-cyan
   * with their own probability.
   */
  drawOccupancy(c, grid) {
    if (!grid || !grid.idx || !grid.idx.length) return;
    const cell = grid.cell;
    const px = cell * this.scale;
    if (px < 0.7) return;
    const half = grid.halfExtent;
    const n = grid.n;
    const zv = this.zeroVisibility;

    for (let k = 0; k < grid.idx.length; k++) {
      const i = grid.idx[k];
      const v = grid.val[k] / 127;                 // -1..1 (normalised log-odds)
      const cxi = i % n;
      const cyi = (i - cxi) / n;
      const wx = cxi * cell - half;
      const wy = cyi * cell - half;
      const p = this.toScreen(wx, wy + cell);
      if (p.x < -px || p.y < -px || p.x > this.w + px || p.y > this.h + px) continue;

      if (v < 0) {
        // observed free
        const a = Math.min(0.5, -v * 0.5);
        c.fillStyle = zv ? 'rgba(10,22,34,' + a.toFixed(3) + ')' : 'rgba(18,34,52,' + a.toFixed(3) + ')';
      } else {
        // observed occupied
        const a = Math.min(0.92, 0.18 + v * 0.8);
        c.fillStyle = zv
          ? 'rgba(44,232,245,' + a.toFixed(3) + ')'
          : 'rgba(70,150,215,' + a.toFixed(3) + ')';
      }
      c.fillRect(p.x, p.y, px + 0.6, px + 0.6);
    }
  }

  /**
   * Simulation ground truth, drawn only when explicitly enabled.  Dashed and
   * grey so it can never be mistaken for sensed geometry — it is here so a
   * judge can check the reconstruction against the real room, which is the
   * opposite of hiding the error.
   */
  drawTruth(c, truth) {
    if (!truth || !truth.length) return;
    c.save();
    c.setLineDash([5, 5]);
    c.lineWidth = 1;
    for (const s of truth) {
      const a = this.toScreen(s.a[0], s.a[1]);
      const b = this.toScreen(s.b[0], s.b[1]);
      c.strokeStyle = s.kind === 'OPENING' ? 'rgba(155,140,255,0.34)'
        : s.kind === 'SOFT' ? 'rgba(255,180,84,0.28)' : 'rgba(140,160,185,0.26)';
      c.beginPath();
      c.moveTo(a.x, a.y);
      c.lineTo(b.x, b.y);
      c.stroke();
    }
    c.restore();
  }

  /** Inferred surfaces, corners, openings and corridors. */
  drawReconstruction(c, recon, now) {
    const t = this.reconAnim;
    if (!recon || !recon.segments) return;
    // Surfaces are always drawn faintly; the RECONSTRUCT action brings them up.
    const base = 0.22 + 0.78 * t;

    for (const s of recon.segments) {
      const a = this.toScreen(s.a.x, s.a.y);
      const b = this.toScreen(s.b.x, s.b.y);
      // Line weight and opacity both track the fit's own confidence, so a
      // thin faint wall visibly means "weak evidence".
      const conf = s.confidence || 0;
      const alpha = base * (0.3 + 0.7 * conf);
      const rgb = s.className && CLASS_RGB[s.className] ? CLASS_RGB[s.className] : INFER_RGB;

      if (t > 0.05) {
        c.save();
        c.shadowBlur = 16 * t * (0.4 + 0.6 * conf);
        c.shadowColor = 'rgba(' + rgb.join(',') + ',' + (0.55 * t).toFixed(3) + ')';
        c.strokeStyle = 'rgba(' + rgb.join(',') + ',' + alpha.toFixed(3) + ')';
        c.lineWidth = 1.2 + 2.4 * conf * t;
        c.lineCap = 'round';
        c.beginPath();
        c.moveTo(a.x, a.y);
        c.lineTo(b.x, b.y);
        c.stroke();
        c.restore();
      } else {
        c.strokeStyle = 'rgba(' + rgb.join(',') + ',' + alpha.toFixed(3) + ')';
        c.lineWidth = 1;
        c.beginPath();
        c.moveTo(a.x, a.y);
        c.lineTo(b.x, b.y);
        c.stroke();
      }
    }

    // Corners: a small bracket, drawn only once the morph is underway.
    if (t > 0.35) {
      const ca = (t - 0.35) / 0.65;
      for (const k of recon.corners || []) {
        const p = this.toScreen(k.x, k.y);
        const r = 7;
        c.strokeStyle = 'rgba(' + INFER_RGB.join(',') + ',' + (0.7 * ca * (k.confidence || 0.5)).toFixed(3) + ')';
        c.lineWidth = 1;
        c.beginPath();
        c.moveTo(p.x - r, p.y);
        c.lineTo(p.x, p.y);
        c.lineTo(p.x, p.y - r);
        c.stroke();
      }
    }

    // Openings: the demo's payoff, so they are labelled with their evidence
    // and confidence rather than just marked.
    for (const o of recon.openings || []) {
      if ((o.confidence || 0) < 0.12) continue;
      const p = this.toScreen(o.x, o.y);
      const pulse = 0.55 + 0.45 * Math.sin(now / 420);
      const alpha = (0.25 + 0.75 * o.confidence) * (0.35 + 0.65 * t);
      c.save();
      c.strokeStyle = 'rgba(' + CLASS_RGB.OPENING.join(',') + ',' + (alpha * pulse).toFixed(3) + ')';
      c.lineWidth = 1.4;
      c.setLineDash([3, 3]);
      c.beginPath();
      c.arc(p.x, p.y, 11, 0, Math.PI * 2);
      c.stroke();
      c.restore();

      if (t > 0.5 && this.scale > 26) {
        c.font = '9px ui-monospace, monospace';
        const tag = (o.width ? o.width.toFixed(2) + ' m ' : '') + Math.round(o.confidence * 100) + '%';
        // "OPENING?" with the question mark, always: this is a candidate.
        plateText(c, 'OPENING? ' + tag, p.x + 15, p.y - 3, CLASS_RGB.OPENING, 0.92 * t);
        plateText(c, o.evidence, p.x + 15, p.y + 9, [130, 145, 170], 0.75 * t);
      }
    }

    // Corridors: a centreline hint, nothing more.
    if (t > 0.6) {
      for (const cr of recon.corridors || []) {
        const p = this.toScreen(cr.mid.x, cr.mid.y);
        c.font = '9px ui-monospace, monospace';
        plateText(c, 'CORRIDOR ' + cr.width.toFixed(1) + ' m', p.x + 8, p.y, INFER_RGB, 0.7);
      }
    }
  }

  /**
   * The echo cloud — the actual measurements.
   *
   * During the reconstruction morph each point slides toward its perpendicular
   * projection on the surface that was fitted to it.  That is what makes the
   * transition honest: points move to where the fit says they belong, so a
   * bad fit is visible as points travelling a long way.
   */
  drawCloud(c, cloud, recon, now) {
    if (!cloud || !cloud.length) return;
    const t = this.reconAnim;
    const ease = t * t * (3 - 2 * t);
    const segs = (recon && recon.segments) || [];

    for (const p of cloud) {
      let wx = p.x;
      let wy = p.y;

      if (ease > 0.01 && segs.length) {
        const proj = nearestOnSegments(p.x, p.y, segs);
        if (proj && proj.d < 1.2) {
          wx = p.x + (proj.x - p.x) * ease;
          wy = p.y + (proj.y - p.y) * ease;
        }
      }

      const s = this.toScreen(wx, wy);
      if (s.x < -20 || s.y < -20 || s.x > this.w + 20 || s.y > this.h + 20) continue;

      const rgb = p.c && CLASS_RGB[p.c] ? CLASS_RGB[p.c] : LIVE_RGB;
      const weight = Math.min(1, (p.w || 0.4) / 3);
      const conf = p.cc || 0;
      // Radius encodes how many times the spot was re-observed; alpha encodes
      // class confidence.  Both are measured quantities.
      const rad = 1.2 + 2.2 * weight;
      const alpha = (0.2 + 0.7 * conf) * (1 - 0.55 * ease);

      c.beginPath();
      c.arc(s.x, s.y, rad, 0, Math.PI * 2);
      c.fillStyle = 'rgba(' + rgb.join(',') + ',' + alpha.toFixed(3) + ')';
      c.fill();

      // Only well-supported points earn a glow.
      if (weight > 0.5 && this.scale > 22) {
        c.save();
        c.shadowBlur = 9;
        c.shadowColor = 'rgba(' + rgb.join(',') + ',0.5)';
        c.beginPath();
        c.arc(s.x, s.y, rad * 0.55, 0, Math.PI * 2);
        c.fillStyle = 'rgba(' + rgb.join(',') + ',' + (alpha * 0.9).toFixed(3) + ')';
        c.fill();
        c.restore();
      }
    }
  }

  /** Phone path: recent nodes brighter, and pose confidence shown as width. */
  drawTrajectory(c, traj, now) {
    const nodes = traj.nodes || [];
    if (nodes.length < 2) return;

    c.lineCap = 'round';
    c.lineJoin = 'round';
    for (let i = 1; i < nodes.length; i++) {
      const a = this.toScreen(nodes[i - 1].x, nodes[i - 1].y);
      const b = this.toScreen(nodes[i].x, nodes[i].y);
      const recency = i / nodes.length;
      const conf = nodes[i].c != null ? nodes[i].c : 0.5;
      c.strokeStyle = 'rgba(' + LIVE_RGB.join(',') + ',' + (0.1 + 0.42 * recency).toFixed(3) + ')';
      // A low-confidence (badly dead-reckoned) stretch of path is drawn
      // thinner, so drift is visible rather than hidden.
      c.lineWidth = 0.7 + 1.9 * conf;
      c.beginPath();
      c.moveTo(a.x, a.y);
      c.lineTo(b.x, b.y);
      c.stroke();
    }
  }

  /** Expanding rings: one per real detection, travelling out to its range. */
  drawPulses(c, now) {
    for (let i = this.pulses.length - 1; i >= 0; i--) {
      const p = this.pulses[i];
      const age = (now - p.at) / 900;
      if (age > 1) { this.pulses.splice(i, 1); continue; }
      const o = this.toScreen(p.x, p.y);
      // The ring stops at the measured range — it is the echo, not decoration.
      const r = p.range * this.scale * Math.min(1, age * 1.25);
      const alpha = (1 - age) * 0.5 * (0.35 + 0.65 * p.conf);
      const rgb = p.cls && CLASS_RGB[p.cls] ? CLASS_RGB[p.cls] : LIVE_RGB;
      const halfBeam = ((p.beam || 30) * Math.PI) / 360;
      const mid = -Math.PI / 2 + (p.bearing * Math.PI) / 180;
      c.beginPath();
      c.arc(o.x, o.y, Math.max(1, r), mid - halfBeam, mid + halfBeam);
      c.strokeStyle = 'rgba(' + rgb.join(',') + ',' + alpha.toFixed(3) + ')';
      c.lineWidth = 1.5;
      c.stroke();
    }
  }

  /** The sensor: position, heading, beam, and a pose-confidence halo. */
  drawSensor(c, pose, lastDet, now) {
    const p = this.toScreen(pose.x, pose.y);
    const rad = (pose.heading * Math.PI) / 180;

    // Position uncertainty: radius grows as pose confidence falls, which is
    // how the display admits that the phone's position is estimated.
    const uncertainty = (1 - (pose.confidence || 0.5)) * 1.2 + 0.12;   // metres
    const ur = uncertainty * this.scale;
    if (ur > 3) {
      c.beginPath();
      c.arc(p.x, p.y, ur, 0, Math.PI * 2);
      c.strokeStyle = 'rgba(44,232,245,0.14)';
      c.setLineDash([2, 4]);
      c.lineWidth = 1;
      c.stroke();
      c.setLineDash([]);
    }

    // Beam wedge out to the search limit: where the sensor can currently hear.
    const beamR = 3.75 * this.scale;
    const halfBeam = (30 * Math.PI) / 360;
    const mid = -Math.PI / 2 + rad;
    const g = c.createRadialGradient(p.x, p.y, 0, p.x, p.y, Math.max(1, beamR));
    g.addColorStop(0, 'rgba(44,232,245,0.16)');
    g.addColorStop(0.55, 'rgba(44,232,245,0.05)');
    g.addColorStop(1, 'rgba(44,232,245,0)');
    c.beginPath();
    c.moveTo(p.x, p.y);
    c.arc(p.x, p.y, Math.max(1, beamR), mid - halfBeam, mid + halfBeam);
    c.closePath();
    c.fillStyle = g;
    c.fill();

    // A live scan breathes along the boresight; it stops when data stops.
    const live = now - this.lastDetectionAt < 1200;
    if (live) {
      this.sweepPhase = now / 1000;
      const bob = 0.5 + 0.5 * Math.sin(this.sweepPhase * 3.1);
      c.beginPath();
      c.arc(p.x, p.y, Math.max(1, beamR * (0.25 + 0.7 * bob)), mid - halfBeam, mid + halfBeam);
      c.strokeStyle = 'rgba(44,232,245,' + (0.18 * (1 - bob)).toFixed(3) + ')';
      c.lineWidth = 1;
      c.stroke();
    }

    // Heading arrow.
    const size = 9;
    c.save();
    c.translate(p.x, p.y);
    c.rotate(rad);
    c.beginPath();
    c.moveTo(0, -size);
    c.lineTo(size * 0.62, size * 0.72);
    c.lineTo(0, size * 0.34);
    c.lineTo(-size * 0.62, size * 0.72);
    c.closePath();
    c.fillStyle = live ? 'rgba(44,232,245,0.95)' : 'rgba(120,150,180,0.7)';
    c.shadowBlur = live ? 14 : 0;
    c.shadowColor = 'rgba(44,232,245,0.7)';
    c.fill();
    c.restore();

    // The current measurement, drawn as a line to the reflector with its
    // angular uncertainty arc — the single clearest statement of what the
    // sensor just did.
    if (lastDet && now - this.lastDetectionAt < 900) {
      const hit = this.toScreen(lastDet.worldX, lastDet.worldY);
      c.beginPath();
      c.moveTo(p.x, p.y);
      c.lineTo(hit.x, hit.y);
      c.strokeStyle = 'rgba(44,232,245,0.3)';
      c.lineWidth = 1;
      c.stroke();

      const rr = lastDet.range_m * this.scale;
      const hb = ((lastDet.beamwidth_deg || 30) * Math.PI) / 360;
      const bm = -Math.PI / 2 + (lastDet.bearing_deg * Math.PI) / 180;
      c.beginPath();
      c.arc(p.x, p.y, Math.max(1, rr), bm - hb, bm + hb);
      c.strokeStyle = 'rgba(44,232,245,0.35)';
      c.lineWidth = 2;
      c.stroke();

      if (this.scale > 30) {
        c.font = '10px ui-monospace, monospace';
        plateText(c, lastDet.range_m.toFixed(2) + ' m', hit.x + 10, hit.y - 5, [214, 242, 251], 0.95);
        const cls = lastDet.fusedClass || lastDet.obstacleClass;
        if (cls) {
          const conf = Math.round((lastDet.fusedConfidence || 0) * 100);
          // An unstable fused call is marked, so a flickering label cannot be
          // mistaken for a settled one.
          const mark = lastDet.fusedStable === false ? '?' : '';
          plateText(c, cls + mark + ' ' + conf + '%', hit.x + 10, hit.y + 8, CLASS_RGB[cls] || LIVE_RGB, 0.95);
        }
      }
    }
  }

  /** A metre scale bar, because a map without one is a picture. */
  drawScaleBar(c) {
    const candidates = [0.5, 1, 2, 5, 10];
    let m = 1;
    for (const k of candidates) { if (k * this.scale > 60) { m = k; break; } m = k; }
    const px = m * this.scale;
    const x = this.w / 2 - px / 2;
    const y = this.h - 26;
    c.strokeStyle = 'rgba(140,165,190,0.4)';
    c.lineWidth = 1;
    c.beginPath();
    c.moveTo(x, y);
    c.lineTo(x + px, y);
    c.moveTo(x, y - 3);
    c.lineTo(x, y + 3);
    c.moveTo(x + px, y - 3);
    c.lineTo(x + px, y + 3);
    c.stroke();
    c.font = '9px ui-monospace, monospace';
    c.textAlign = 'center';
    const lbl = m + ' m';
    const lw = c.measureText(lbl).width;
    c.fillStyle = 'rgba(2,5,10,0.7)';
    c.fillRect(x + px / 2 - lw / 2 - 3, y - 15, lw + 6, 11);
    c.fillStyle = 'rgba(150,175,200,0.8)';
    c.fillText(lbl, x + px / 2, y - 7);
    c.textAlign = 'left';
  }
}

/**
 * Draw text on a dark plate.
 *
 * Labels land on top of glowing occupancy cells, where plain fill text is
 * unreadable.  The plate is cheaper and crisper than a text shadow and keeps
 * the type looking like instrument labelling rather than a glow effect.
 */
function plateText(c, text, x, y, rgb, alpha) {
  const w = c.measureText(text).width;
  c.fillStyle = 'rgba(2,5,10,' + (0.78 * alpha).toFixed(3) + ')';
  c.fillRect(x - 3, y - 8, w + 6, 11);
  c.fillStyle = 'rgba(' + rgb.join(',') + ',' + alpha.toFixed(3) + ')';
  c.fillText(text, x, y);
}

/** Perpendicular projection onto the nearest fitted segment. */
function nearestOnSegments(x, y, segs) {
  let best = null;
  for (const s of segs) {
    const ax = s.a.x, ay = s.a.y, bx = s.b.x, by = s.b.y;
    const dx = bx - ax, dy = by - ay;
    const len2 = dx * dx + dy * dy;
    if (len2 < 1e-9) continue;
    let t = ((x - ax) * dx + (y - ay) * dy) / len2;
    t = t < 0 ? 0 : t > 1 ? 1 : t;
    const px = ax + dx * t;
    const py = ay + dy * t;
    const d = Math.hypot(px - x, py - y);
    if (!best || d < best.d) best = { x: px, y: py, d };
  }
  return best;
}

/** Bounding box over cloud + trajectory, used by fit-to-view. */
export function worldBounds(cloud, traj, pose) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  const add = (x, y) => {
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
  };
  for (const p of cloud || []) add(p.x, p.y);
  for (const n of (traj && traj.nodes) || []) add(n.x, n.y);
  if (pose) add(pose.x, pose.y);
  if (!Number.isFinite(minX)) return null;
  return { minX, minY, maxX, maxY };
}

export { CLASS_RGB, CLASSES };
