/**
 * SentryShield spatial engine — accumulated point cloud, probabilistic
 * occupancy grid and temporal class fusion.
 *
 * Runs identically on the server (authoritative map state, mission statistics,
 * recordings) and in the command center (smooth local rendering between
 * snapshots).  Pure data in, pure data out: no DOM, no Node APIs, no globals.
 *
 * World frame: metres, +x east, +y north, origin = mission start pose.
 */

import { CLASSES, clamp01, wrapDeg, angleDelta, polarToWorld } from './protocol.mjs';

export const DEFAULTS = {
  cell: 0.10,            // [m]  occupancy grid resolution
  halfExtent: 12.0,      // [m]  grid spans +-halfExtent around the origin
  mergeRadius: 0.16,     // [m]  point-cloud consolidation radius
  maxPoints: 4000,       // hard cap; oldest weak points are dropped first
  beamRays: 5,           // rays traced across the beam cone per detection
  lFree: -0.42,          // log-odds added to cells the beam passed through
  lOcc: 1.05,            // log-odds added to the cell holding the reflector
  lClamp: 5.5,           // saturation, so a stale cell can still be revised
  fusionWindow: 2200,    // [ms] class-history horizon for a cloud point
  fusionRadius: 0.45,    // [m]  detections within this radius fuse together
};

// ---------------------------------------------------------------------------
// Occupancy grid
// ---------------------------------------------------------------------------
/**
 * Log-odds occupancy grid.  Unknown is exactly 0 and stays visually distinct
 * from observed-free (negative) and observed-occupied (positive) — that
 * distinction is the whole point of the "environment emerging from darkness"
 * visual, so it is never smoothed away.
 */
export class OccupancyGrid {
  constructor(opts = {}) {
    this.cell = opts.cell || DEFAULTS.cell;
    this.halfExtent = opts.halfExtent || DEFAULTS.halfExtent;
    this.n = Math.max(8, Math.round((2 * this.halfExtent) / this.cell));
    this.data = new Float32Array(this.n * this.n);
    this.touched = 0;           // count of cells ever updated -> map coverage
    this.lClamp = opts.lClamp || DEFAULTS.lClamp;
  }

  /** World metres -> integer cell coords. */
  toCell(x, y) {
    return {
      cx: Math.floor((x + this.halfExtent) / this.cell),
      cy: Math.floor((y + this.halfExtent) / this.cell),
    };
  }

  /** Cell coords -> world metres at the cell centre. */
  toWorld(cx, cy) {
    return {
      x: (cx + 0.5) * this.cell - this.halfExtent,
      y: (cy + 0.5) * this.cell - this.halfExtent,
    };
  }

  inBounds(cx, cy) { return cx >= 0 && cy >= 0 && cx < this.n && cy < this.n; }

  get(cx, cy) { return this.inBounds(cx, cy) ? this.data[cy * this.n + cx] : 0; }

  /** Accumulate log-odds into one cell, saturating at +-lClamp. */
  bump(cx, cy, delta) {
    if (!this.inBounds(cx, cy) || delta === 0) return;
    const i = cy * this.n + cx;
    if (this.data[i] === 0) this.touched++;
    let v = this.data[i] + delta;
    if (v > this.lClamp) v = this.lClamp;
    else if (v < -this.lClamp) v = -this.lClamp;
    // A cell nudged exactly back to 0 would read as "never observed"; keep it
    // barely observed so coverage statistics stay monotonic.
    this.data[i] = v === 0 ? (delta > 0 ? 1e-3 : -1e-3) : v;
  }

  /** Occupancy probability 0..1 (0.5 = unknown). */
  prob(cx, cy) {
    const l = this.get(cx, cy);
    return 1 - 1 / (1 + Math.exp(l));
  }

  /**
   * Integrate one range measurement: cells along the beam become free, the
   * cell at the reflector becomes occupied.  Weight scales both, so a
   * low-confidence detection leaves a faint mark instead of a hard wall.
   *
   * An OPENING detection deliberately does NOT mark occupied — the acoustic
   * evidence is a *lack* of a specular boundary, so we carve free space a
   * little past the range instead.  That is what makes doorways appear as
   * gaps in the reconstructed geometry.
   */
  integrate(pose, bearingDeg, rangeM, opts = {}) {
    const weight = clamp01(opts.weight == null ? 1 : opts.weight);
    if (weight <= 0.02 || !(rangeM > 0)) return;
    const isOpening = opts.className === 'OPENING';
    const beam = opts.beamwidth_deg == null ? 30 : opts.beamwidth_deg;
    const rays = Math.max(1, opts.rays || DEFAULTS.beamRays);

    for (let k = 0; k < rays; k++) {
      // Rays fan across the cone; edge rays carry less weight than boresight.
      const frac = rays === 1 ? 0 : (k / (rays - 1)) * 2 - 1;      // -1..+1
      const ang = bearingDeg + frac * beam * 0.5;
      const rayW = weight * (1 - 0.55 * Math.abs(frac));
      const freeTo = isOpening ? rangeM * 1.18 : rangeM - this.cell;
      this.castFree(pose.x, pose.y, ang, Math.max(0, freeTo), rayW);
      if (!isOpening) {
        const hit = polarToWorld(pose, ang, rangeM);
        const c = this.toCell(hit.x, hit.y);
        this.bump(c.cx, c.cy, DEFAULTS.lOcc * rayW);
      }
    }
  }

  /** March a ray marking cells free, stepping half a cell at a time. */
  castFree(x0, y0, bearingDeg, dist, weight) {
    if (dist <= 0 || weight <= 0) return;
    const r = (wrapDeg(bearingDeg) * Math.PI) / 180;
    const dx = Math.sin(r);
    const dy = Math.cos(r);
    const step = this.cell * 0.5;
    let last = -1;
    for (let d = step; d < dist; d += step) {
      const c = this.toCell(x0 + dx * d, y0 + dy * d);
      const key = c.cy * this.n + c.cx;
      if (key === last) continue;                 // do not double-count a cell
      last = key;
      this.bump(c.cx, c.cy, DEFAULTS.lFree * weight);
    }
  }

  /** Fraction of the grid that has ever been observed. */
  coverage() { return this.touched / (this.n * this.n); }

  /** Observed area in square metres, split free / occupied. */
  areaStats() {
    let free = 0, occ = 0;
    for (let i = 0; i < this.data.length; i++) {
      const v = this.data[i];
      if (v > 0.35) occ++;
      else if (v < -0.35) free++;
    }
    const a = this.cell * this.cell;
    return { freeArea: free * a, occupiedArea: occ * a, freeCells: free, occupiedCells: occ };
  }

  /** Sparse wire format: only non-zero cells, quantised to one byte each. */
  serialize() {
    const idx = [];
    const val = [];
    for (let i = 0; i < this.data.length; i++) {
      if (this.data[i] !== 0) {
        idx.push(i);
        val.push(Math.round((this.data[i] / this.lClamp) * 127));
      }
    }
    return { n: this.n, cell: this.cell, halfExtent: this.halfExtent, idx, val };
  }

  static deserialize(s) {
    const g = new OccupancyGrid({ cell: s.cell, halfExtent: s.halfExtent });
    if (g.n !== s.n) { g.n = s.n; g.data = new Float32Array(s.n * s.n); }
    for (let k = 0; k < s.idx.length; k++) {
      g.data[s.idx[k]] = (s.val[k] / 127) * g.lClamp;
      g.touched++;
    }
    return g;
  }

  reset() { this.data.fill(0); this.touched = 0; }
}

// ---------------------------------------------------------------------------
// Temporal class fusion
// ---------------------------------------------------------------------------
/**
 * Confidence-weighted class fusion over a short temporal window.
 *
 * Why this exists: EchoNet's per-echo accuracy is 69.5 % on synthetic
 * validation (WALL recall 91 %, SOFT 74 %, OPENING 43 %).  A single echo is
 * therefore not a decision.  Fusing consecutive looks at the *same place*
 * suppresses flicker without inventing certainty: the fused confidence is the
 * mean posterior, so a class that keeps winning narrowly stays reported as
 * low-confidence rather than being rounded up to a clean answer.
 */
export class ClassFuser {
  constructor(opts = {}) {
    this.window = opts.fusionWindow || DEFAULTS.fusionWindow;
    this.radius = opts.fusionRadius || DEFAULTS.fusionRadius;
    this.tracks = [];    // { x, y, samples:[{t, probs, conf}] }
  }

  /**
   * @returns {{className:string|null, confidence:number, probs:number[],
   *            history:number[], support:number, stable:boolean}}
   */
  fuse(det) {
    const now = det.t || Date.now();
    const probs = det.classProbs || [0, 0, 0];
    let track = null;
    let bestD = Infinity;
    for (const tr of this.tracks) {
      const d = Math.hypot(tr.x - det.worldX, tr.y - det.worldY);
      if (d < this.radius && d < bestD) { bestD = d; track = tr; }
    }
    if (!track) {
      track = { x: det.worldX, y: det.worldY, samples: [] };
      this.tracks.push(track);
    }
    // Track centroid drifts toward the newest observation.
    track.x += (det.worldX - track.x) * 0.25;
    track.y += (det.worldY - track.y) * 0.25;
    track.samples.push({ t: now, probs: probs.slice(), conf: det.confidence });
    track.samples = track.samples.filter((s) => now - s.t <= this.window);

    // Drop tracks nobody has revisited, so the fuser stays bounded.
    if (this.tracks.length > 64) {
      this.tracks = this.tracks
        .filter((tr) => tr.samples.length && now - tr.samples[tr.samples.length - 1].t < this.window * 3)
        .slice(-64);
    }

    const acc = [0, 0, 0];
    let wsum = 0;
    for (const s of track.samples) {
      // Recency weight decays linearly across the window; detection
      // confidence scales how much that look is trusted at all.
      const age = (now - s.t) / this.window;
      const w = Math.max(0.05, (1 - 0.6 * age)) * Math.max(0.05, s.conf);
      for (let i = 0; i < 3; i++) acc[i] += s.probs[i] * w;
      wsum += w;
    }
    if (wsum > 0) for (let i = 0; i < 3; i++) acc[i] /= wsum;

    let best = 0;
    for (let i = 1; i < 3; i++) if (acc[i] > acc[best]) best = i;
    const history = track.samples.map((s) => {
      let b = 0;
      for (let i = 1; i < 3; i++) if (s.probs[i] > s.probs[b]) b = i;
      return b === best ? s.probs[best] : -s.probs[b];   // sign marks disagreement
    });
    const agree = track.samples.filter((s) => argmax(s.probs) === best).length;
    return {
      className: acc[best] > 0 ? CLASSES[best] : null,
      confidence: clamp01(acc[best]),
      probs: acc,
      history: history.slice(-12),
      support: track.samples.length,
      // "Stable" needs both repetition and agreement — never confidence alone.
      stable: track.samples.length >= 3 && agree / track.samples.length >= 0.6,
    };
  }

  reset() { this.tracks = []; }
}

function argmax(a) { let b = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[b]) b = i; return b; }

// ---------------------------------------------------------------------------
// Point cloud
// ---------------------------------------------------------------------------
/**
 * Accumulated detections in world coordinates, consolidated on a spatial hash.
 * Re-observing the same spot reinforces one point (weight up, position
 * averaged) instead of stacking duplicates — repeated looks make the map
 * firmer, which is exactly the behaviour the demo narrates.
 */
export class PointCloud {
  constructor(opts = {}) {
    this.mergeRadius = opts.mergeRadius || DEFAULTS.mergeRadius;
    this.maxPoints = opts.maxPoints || DEFAULTS.maxPoints;
    this.points = [];
    this.buckets = new Map();
    this.seq = 0;
  }

  key(x, y) {
    const s = this.mergeRadius;
    return Math.floor(x / s) + ':' + Math.floor(y / s);
  }

  add(det) {
    const k = this.key(det.worldX, det.worldY);
    // Search the 3x3 bucket neighbourhood so points near a bucket seam merge.
    const bx = Math.floor(det.worldX / this.mergeRadius);
    const by = Math.floor(det.worldY / this.mergeRadius);
    let hit = null;
    let bestD = this.mergeRadius;
    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        const list = this.buckets.get((bx + dx) + ':' + (by + dy));
        if (!list) continue;
        for (const p of list) {
          const d = Math.hypot(p.x - det.worldX, p.y - det.worldY);
          if (d < bestD) { bestD = d; hit = p; }
        }
      }
    }

    const cls = det.fusedClass || det.obstacleClass;
    const conf = det.fusedConfidence || det.classConfidence || 0;

    if (hit) {
      const w = hit.weight;
      const nw = Math.min(w + det.confidence, 12);
      hit.x = (hit.x * w + det.worldX * det.confidence) / (w + det.confidence);
      hit.y = (hit.y * w + det.worldY * det.confidence) / (w + det.confidence);
      hit.weight = nw;
      hit.hits++;
      hit.t = det.t;
      hit.range_m = det.range_m;
      hit.snr_db = det.snr_db;
      if (cls) {
        hit.classVotes[cls] = (hit.classVotes[cls] || 0) + conf;
        hit.className = topVote(hit.classVotes);
        hit.classConfidence = voteConfidence(hit.classVotes);
      }
      return hit;
    }

    const p = {
      id: ++this.seq,
      x: det.worldX,
      y: det.worldY,
      t: det.t,
      weight: det.confidence,
      hits: 1,
      range_m: det.range_m,
      snr_db: det.snr_db,
      className: cls,
      classConfidence: conf,
      classVotes: cls ? { [cls]: conf } : {},
      source: det.source,
    };
    this.points.push(p);
    let list = this.buckets.get(k);
    if (!list) { list = []; this.buckets.set(k, list); }
    list.push(p);
    if (this.points.length > this.maxPoints) this.prune();
    return p;
  }

  /** Drop the weakest quarter of the cloud, then rebuild the hash. */
  prune() {
    this.points.sort((a, b) => (b.weight + b.hits * 0.3) - (a.weight + a.hits * 0.3));
    this.points.length = Math.floor(this.maxPoints * 0.75);
    this.buckets.clear();
    for (const p of this.points) {
      const k = this.key(p.x, p.y);
      let list = this.buckets.get(k);
      if (!list) { list = []; this.buckets.set(k, list); }
      list.push(p);
    }
  }

  byClass(name) { return this.points.filter((p) => p.className === name); }

  stats() {
    const out = { total: this.points.length, WALL: 0, SOFT: 0, OPENING: 0, unclassified: 0, avgConfidence: 0 };
    let cs = 0;
    for (const p of this.points) {
      if (p.className && out[p.className] != null) out[p.className]++;
      else out.unclassified++;
      cs += p.classConfidence || 0;
    }
    out.avgConfidence = this.points.length ? cs / this.points.length : 0;
    return out;
  }

  serialize() {
    return this.points.map((p) => ({
      id: p.id, x: r3(p.x), y: r3(p.y), t: p.t, w: r3(p.weight), h: p.hits,
      c: p.className, cc: r3(p.classConfidence), s: r3(p.snr_db),
    }));
  }

  loadSerialized(arr) {
    this.points = [];
    this.buckets.clear();
    for (const q of arr || []) {
      const p = {
        id: q.id, x: q.x, y: q.y, t: q.t, weight: q.w, hits: q.h,
        range_m: 0, snr_db: q.s || 0, className: q.c, classConfidence: q.cc || 0,
        classVotes: q.c ? { [q.c]: q.cc || 0 } : {}, source: 'replay',
      };
      this.points.push(p);
      const k = this.key(p.x, p.y);
      let list = this.buckets.get(k);
      if (!list) { list = []; this.buckets.set(k, list); }
      list.push(p);
      if (q.id > this.seq) this.seq = q.id;
    }
  }

  reset() { this.points = []; this.buckets.clear(); this.seq = 0; }
}

function topVote(votes) {
  let best = null, bv = -1;
  for (const k of Object.keys(votes)) if (votes[k] > bv) { bv = votes[k]; best = k; }
  return best;
}

function voteConfidence(votes) {
  let sum = 0, top = 0;
  for (const k of Object.keys(votes)) { sum += votes[k]; if (votes[k] > top) top = votes[k]; }
  return sum > 0 ? top / sum : 0;
}

function r3(v) { return Math.round(v * 1000) / 1000; }

// ---------------------------------------------------------------------------
// Trajectory
// ---------------------------------------------------------------------------
/** Sparse phone path: a node is kept only once the pose has actually moved. */
export class Trajectory {
  constructor(minStep = 0.06) {
    this.minStep = minStep;
    this.nodes = [];
    this.distance = 0;
  }

  push(pose, t) {
    const last = this.nodes[this.nodes.length - 1];
    if (last) {
      const d = Math.hypot(pose.x - last.x, pose.y - last.y);
      if (d < this.minStep && Math.abs(angleDelta(last.heading, pose.heading)) < 8) return false;
      this.distance += d;
    }
    this.nodes.push({ x: pose.x, y: pose.y, heading: pose.heading, c: pose.confidence, t: t || Date.now() });
    if (this.nodes.length > 3000) this.nodes.splice(0, 500);
    return true;
  }

  serialize() {
    return { distance: r3(this.distance), nodes: this.nodes.map((n) => ({ x: r3(n.x), y: r3(n.y), h: Math.round(n.heading), c: r3(n.c), t: n.t })) };
  }

  loadSerialized(s) {
    this.distance = (s && s.distance) || 0;
    this.nodes = ((s && s.nodes) || []).map((n) => ({ x: n.x, y: n.y, heading: n.h, c: n.c, t: n.t }));
  }

  reset() { this.nodes = []; this.distance = 0; }
}
