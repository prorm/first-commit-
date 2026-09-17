/**
 * Live wall boundaries — the tactical blueprint layer.
 *
 * `reconstruct.mjs` fits surfaces to an *accumulated* cloud: it needs a cluster
 * of history before it will commit to a line, which is correct for a survey and
 * useless while walking. Standing 2.5 m from a flat wall with a confident echo,
 * the operator should see a wall, not a dot.
 *
 * So this runs alongside it, from a different piece of physics. A single echo
 * does not locate a point, it locates a *wavefront tangent*: the reflector lies
 * somewhere on an arc of radius `range` across the beam, and for a flat surface
 * the chord of that arc is the surface. One confident detection is therefore
 * already enough to draw a short bar, pinned in world coordinates — which is
 * the whole point. Walk forward, turn around, and the bars stay where the walls
 * are rather than sweeping with the sensor.
 *
 * Successive chirps on the same wall merge into one longer bar instead of
 * stacking hundreds of overlapping lines, and a bar that a later pulse sees
 * straight through loses strength and disappears. Both behaviours are what stop
 * this from being a smear.
 *
 * What this is NOT: a replacement for reconstruct.mjs. These bars are raw
 * per-echo evidence with no line fit, no residual and no inlier test. Pressing
 * RECONSTRUCT still runs the real fit, and the renderer fades these out as it
 * takes over, so the two never argue on screen.
 *
 * OPENING never appears here. An opening is the absence of a return and cannot
 * produce a tangent; it stays where it belongs, as a geometric gap between
 * collinear boundaries in reconstruct.mjs. See shared/surfaceclass.mjs.
 */

const DEF = {
  minConfidence: 0.65,   // below this a single echo is not evidence of a surface
  maxRange: 4.0,         // beyond the search window there is nothing honest to draw
  minHalfLen: 0.10,      // m, a bar is never a point
  maxHalfLen: 0.55,      // m, beam geometry stops being a chord well before this
  mergeAngleDeg: 14,     // collinear enough to be the same wall
  mergeDistM: 0.15,      // perpendicular separation that still counts as one wall
  mergeGapM: 0.45,       // end-to-end gap a merge may bridge along the wall
  contradictM: 0.22,     // a pulse passing this far beyond a bar contradicts it
  contradictLoss: 0.34,  // strength removed per contradicting pulse
  hitGain: 0.22,         // strength added per confirming pulse
  minStrength: 0.12,     // below this the bar is dropped
  maxBars: 420,          // hard cap; the weakest go first
};

const D2R = Math.PI / 180;

/** Angular difference between two undirected line directions, in degrees. */
function lineAngleDelta(a, b) {
  const d = Math.abs(a - b) % 180;
  return d > 90 ? 180 - d : d;
}

export class BoundaryMap {
  constructor(opts = {}) {
    this.cfg = Object.assign({}, DEF, opts);
    this.bars = [];
    this.seq = 0;
  }

  reset() {
    this.bars = [];
    this.seq = 0;
  }

  /**
   * Fold one detection in.
   *
   * @param {Object} det a normalised Detection (needs phone, bearing_deg,
   *   range_m, confidence; obstacleClass/fusedClass optional)
   * @returns {Object|null} the bar this detection landed on, or null if the
   *   detection was not strong enough to draw
   */
  add(det) {
    if (!det || !det.phone) return null;
    const cfg = this.cfg;
    const r = det.range_m;
    if (!(r > 0) || r > cfg.maxRange) return null;

    // Every pulse gets to contradict, even one too weak to draw with: seeing
    // through a wall is information regardless of what the classifier said.
    this._contradict(det);

    if (!(det.confidence >= cfg.minConfidence)) return null;
    if (det.cfar_pass === false) return null;
    const cls = det.fusedClass || det.obstacleClass;
    if (cls !== 'WALL' && cls !== 'SOFT') return null;

    const bearing = det.bearing_deg;
    const rad = bearing * D2R;
    const cx = det.phone.x + r * Math.sin(rad);
    const cy = det.phone.y + r * Math.cos(rad);

    // The bar is the chord the beam cuts across the range arc, so its length
    // grows with range exactly as the beam does. That is why a far wall draws
    // as a long segment and a near one as a short one: it is the real
    // cross-range ambiguity, not a styling choice.
    const beam = Math.max(4, Math.min(120, det.beamwidth_deg || 30));
    const half = Math.max(
      cfg.minHalfLen,
      Math.min(cfg.maxHalfLen, r * Math.tan((beam / 2) * D2R))
    );

    // Tangent to the wavefront: perpendicular to the look direction.
    const dirDeg = (bearing + 90 + 360) % 180;
    const dr = dirDeg * D2R;
    const ux = Math.sin(dr);
    const uy = Math.cos(dr);

    const poseConf = det.phone.confidence == null ? 1 : det.phone.confidence;
    const w = det.confidence * (0.4 + 0.6 * poseConf);

    const existing = this._findMergeTarget(cx, cy, dirDeg, half, cls);
    if (existing) return this._merge(existing, cx, cy, dirDeg, half, ux, uy, w, det.t);

    const bar = {
      id: ++this.seq,
      cx,
      cy,
      dirDeg,
      half,
      className: cls,
      strength: Math.min(1, 0.45 + 0.55 * w),
      hits: 1,
      weight: w,
      t: det.t || Date.now(),
    };
    this.bars.push(bar);
    this._prune();
    return bar;
  }

  /**
   * A pulse that measured *past* a bar, along a bearing that looks through it,
   * is evidence the bar is not there. Free space wins over a stale reflection:
   * this is what keeps a turn, or a mis-tracked echo, from leaving debris.
   */
  _contradict(det) {
    const cfg = this.cfg;
    const rad = det.bearing_deg * D2R;
    const sx = det.phone.x;
    const sy = det.phone.y;
    const dx = Math.sin(rad);
    const dy = Math.cos(rad);
    const reach = det.range_m - cfg.contradictM;
    if (!(reach > 0.2)) return;

    for (let i = this.bars.length - 1; i >= 0; i--) {
      const b = this.bars[i];
      const ux = Math.sin(b.dirDeg * D2R);
      const uy = Math.cos(b.dirDeg * D2R);
      // Ray (s + t*d) against the bar's finite segment.
      const den = dx * uy - dy * ux;
      if (Math.abs(den) < 1e-6) continue;          // parallel: never crossed
      const ox = b.cx - sx;
      const oy = b.cy - sy;
      const t = (ox * uy - oy * ux) / den;         // distance along the ray
      const s = (ox * dy - oy * dx) / den;         // offset along the bar
      if (t <= 0.15 || t >= reach) continue;       // not crossed before the echo
      if (Math.abs(s) > b.half) continue;          // missed the bar's extent

      b.strength -= cfg.contradictLoss;
      if (b.strength < cfg.minStrength) this.bars.splice(i, 1);
    }
  }

  _findMergeTarget(cx, cy, dirDeg, half, cls) {
    const cfg = this.cfg;
    let best = null;
    let bestPerp = Infinity;
    for (const b of this.bars) {
      if (b.className !== cls) continue;
      if (lineAngleDelta(b.dirDeg, dirDeg) > cfg.mergeAngleDeg) continue;
      const bx = Math.sin(b.dirDeg * D2R);
      const by = Math.cos(b.dirDeg * D2R);
      const ox = cx - b.cx;
      const oy = cy - b.cy;
      const along = ox * bx + oy * by;
      const perp = Math.abs(ox * -by + oy * bx);
      if (perp > cfg.mergeDistM) continue;
      // Overlapping, or close enough end-to-end to be the same wall continuing.
      if (Math.abs(along) > b.half + half + cfg.mergeGapM) continue;
      if (perp < bestPerp) { bestPerp = perp; best = b; }
    }
    return best;
  }

  /**
   * Absorb a new observation into an existing bar: nudge its direction and
   * offset toward the new evidence, then re-span it over both extents so the
   * wall grows along its own length instead of drifting sideways.
   */
  _merge(b, cx, cy, dirDeg, half, ux, uy, w, t) {
    const total = b.weight + w;
    const k = w / total;

    // Directions are undirected; fold across the 180 boundary before averaging.
    let d = dirDeg;
    if (d - b.dirDeg > 90) d -= 180;
    else if (b.dirDeg - d > 90) d += 180;
    b.dirDeg = (((b.dirDeg + (d - b.dirDeg) * k) % 180) + 180) % 180;

    const nx = Math.sin(b.dirDeg * D2R);
    const ny = Math.cos(b.dirDeg * D2R);

    // The centre moves toward the new observation, then the bar is re-spanned
    // over both segments' endpoints, so a confirming echo tightens and extends
    // the line rather than sliding it along its own length.
    const mx = b.cx + (cx - b.cx) * k;
    const my = b.cy + (cy - b.cy) * k;
    const pts = [
      [b.cx - nx * b.half, b.cy - ny * b.half],
      [b.cx + nx * b.half, b.cy + ny * b.half],
      [cx - ux * half, cy - uy * half],
      [cx + ux * half, cy + uy * half],
    ];
    let lo = Infinity;
    let hi = -Infinity;
    for (const p of pts) {
      const s = (p[0] - mx) * nx + (p[1] - my) * ny;
      if (s < lo) lo = s;
      if (s > hi) hi = s;
    }
    b.cx = mx + (nx * (lo + hi)) / 2;
    b.cy = my + (ny * (lo + hi)) / 2;
    b.half = Math.max(this.cfg.minHalfLen, (hi - lo) / 2);

    b.weight = total;
    b.hits++;
    b.strength = Math.min(1, b.strength + this.cfg.hitGain * w);
    b.t = t || b.t;
    return b;
  }

  _prune() {
    const cap = this.cfg.maxBars;
    if (this.bars.length <= cap) return;
    // Drop the weakest first, and among equals the oldest.
    this.bars.sort((a, b) => (b.strength - a.strength) || (b.t - a.t));
    this.bars.length = cap;
  }

  /**
   * Bars ready to draw, as world-space endpoints.
   * @returns {Array<{a:Object, b:Object, className:string, strength:number}>}
   */
  segments() {
    const out = [];
    for (const b of this.bars) {
      const ux = Math.sin(b.dirDeg * D2R);
      const uy = Math.cos(b.dirDeg * D2R);
      out.push({
        id: b.id,
        a: { x: b.cx - ux * b.half, y: b.cy - uy * b.half },
        b: { x: b.cx + ux * b.half, y: b.cy + uy * b.half },
        className: b.className,
        strength: b.strength,
        hits: b.hits,
        t: b.t,
      });
    }
    return out;
  }

  get length() { return this.bars.length; }
}
