/**
 * SentryShield surface reconstruction.
 *
 * Turns an accumulated acoustic point cloud into human-readable geometry:
 * wall segments, corners, openings and corridors.
 *
 * HONESTY NOTE, and it matters for how this is presented:
 * a single-element near-ultrasonic sensor has an angular beamwidth of tens of
 * degrees.  Every point in the cloud is therefore "somewhere in a cone", not a
 * surveyed coordinate.  These segments are *inferred* boundaries fitted to a
 * noisy cloud, and every returned feature carries its own confidence and
 * inlier count so the UI can show how thin the evidence is.  Nothing here
 * claims geometric survey accuracy.
 *
 * Method: grid-neighbour clustering (single link, tolerant of the cloud's
 * uneven density) -> principal-axis line fit -> recursive split where the
 * residual says one cluster spans a corner -> gap analysis along each fitted
 * axis to expose doorways.
 */

const DEF = {
  clusterRadius: 0.38,     // [m] single-link distance for the same surface
  minClusterPoints: 4,
  minSegmentLength: 0.35,  // [m] shorter fits are noise, not a wall
  splitResidual: 0.13,     // [m] max perpendicular error before splitting
  maxAcceptRms: 0.22,      // [m] a fit still this scattered is not a surface
  maxSupportGap: 1.6,      // [m] unsupported stretch inside a fit -> two walls
  maxSplitDepth: 6,
  cornerAngleDeg: 55,      // segments meeting sharper than this = corner
  cornerReach: 1.1,        // [m] how far a fitted line may be extended to meet another
  gapMin: 0.55,            // [m] along-axis gap that could be a doorway
  gapMax: 1.60,            // [m] wider than a door -> just unscanned space
  corridorWidthMax: 2.6,   // [m] parallel walls closer than this = corridor
};

/**
 * @param {Array} points  cloud points: { x, y, weight, hits, className, classConfidence }
 * @param {Object} [opts]
 * @returns {{segments:Array, corners:Array, openings:Array, corridors:Array, confidence:number, stats:Object}}
 */
export function reconstruct(points, opts = {}) {
  const cfg = Object.assign({}, DEF, opts);

  // Boundary evidence = anything that produced a specular-ish return.  OPENING
  // points are deliberately excluded: they are evidence of absence, and are
  // handled separately as opening candidates.
  const structural = (points || []).filter(
    (p) => p.className !== 'OPENING' && (p.hits >= 1) && (p.weight || 0) > 0.05
  );
  const openingPts = (points || []).filter((p) => p.className === 'OPENING');

  if (structural.length < cfg.minClusterPoints) {
    return { segments: [], corners: [], openings: openingCandidatesFromPoints(openingPts, []), corridors: [], confidence: 0, stats: { clusters: 0, structuralPoints: structural.length, inliers: 0 } };
  }

  const clusters = clusterPoints(structural, cfg.clusterRadius, cfg.minClusterPoints);

  const segments = [];
  for (const cl of clusters) fitRecursive(cl, cfg, segments, 0);

  const kept = segments.filter((s) => s.length >= cfg.minSegmentLength);
  kept.sort((a, b) => b.length - a.length);
  kept.forEach((s, i) => { s.id = i + 1; });

  const corners = findCorners(kept, cfg);
  const openings = dedupeOpenings(
    findGapOpenings(kept, cfg)
      .concat(findGapsBetweenSegments(kept, cfg))
      .concat(openingCandidatesFromPoints(openingPts, kept))
  );
  const corridors = findCorridors(kept, cfg);

  let inliers = 0;
  for (const s of kept) inliers += s.support;
  const confidence = reconstructionConfidence(kept, structural.length, inliers);

  return {
    segments: kept,
    corners,
    openings,
    corridors,
    confidence,
    stats: {
      clusters: clusters.length,
      structuralPoints: structural.length,
      inliers,
      inlierRatio: structural.length ? inliers / structural.length : 0,
      openingPoints: openingPts.length,
      totalLength: kept.reduce((a, s) => a + s.length, 0),
    },
  };
}

// ---------------------------------------------------------------------------
// Clustering — single-link over a spatial hash (O(n) for our densities)
// ---------------------------------------------------------------------------
export function clusterPoints(points, radius, minPoints) {
  const buckets = new Map();
  const bk = (x, y) => Math.floor(x / radius) + ':' + Math.floor(y / radius);
  points.forEach((p, i) => {
    const k = bk(p.x, p.y);
    let l = buckets.get(k);
    if (!l) { l = []; buckets.set(k, l); }
    l.push(i);
  });

  const seen = new Uint8Array(points.length);
  const out = [];
  for (let i = 0; i < points.length; i++) {
    if (seen[i]) continue;
    const queue = [i];
    seen[i] = 1;
    const members = [];
    while (queue.length) {
      const j = queue.pop();
      const pj = points[j];
      members.push(pj);
      const bx = Math.floor(pj.x / radius);
      const by = Math.floor(pj.y / radius);
      for (let dx = -1; dx <= 1; dx++) {
        for (let dy = -1; dy <= 1; dy++) {
          const l = buckets.get((bx + dx) + ':' + (by + dy));
          if (!l) continue;
          for (const m of l) {
            if (seen[m]) continue;
            if (Math.hypot(points[m].x - pj.x, points[m].y - pj.y) <= radius) {
              seen[m] = 1;
              queue.push(m);
            }
          }
        }
      }
    }
    if (members.length >= minPoints) out.push(members);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Line fitting
// ---------------------------------------------------------------------------
/**
 * Total-least-squares line through weighted points, via the principal axis of
 * the weighted covariance.  Returns the fit plus its perpendicular residual,
 * which is what tells us whether the cluster is one wall or a corner.
 */
export function fitLine(points) {
  let W = 0, mx = 0, my = 0;
  for (const p of points) {
    const w = Math.max(0.05, p.weight || 1);
    W += w; mx += p.x * w; my += p.y * w;
  }
  mx /= W; my /= W;

  let sxx = 0, syy = 0, sxy = 0;
  for (const p of points) {
    const w = Math.max(0.05, p.weight || 1);
    const dx = p.x - mx, dy = p.y - my;
    sxx += w * dx * dx; syy += w * dy * dy; sxy += w * dx * dy;
  }
  sxx /= W; syy /= W; sxy /= W;

  // Principal axis of the 2x2 covariance.
  const theta = 0.5 * Math.atan2(2 * sxy, sxx - syy);
  const ux = Math.cos(theta), uy = Math.sin(theta);

  let tMin = Infinity, tMax = -Infinity, res = 0, resMax = 0;
  for (const p of points) {
    const dx = p.x - mx, dy = p.y - my;
    const t = dx * ux + dy * uy;
    const n = Math.abs(-dx * uy + dy * ux);
    if (t < tMin) tMin = t;
    if (t > tMax) tMax = t;
    res += n * n;
    if (n > resMax) resMax = n;
  }
  const rms = Math.sqrt(res / points.length);

  return {
    cx: mx, cy: my, ux, uy,
    tMin, tMax,
    length: tMax - tMin,
    rms, resMax,
    a: { x: mx + ux * tMin, y: my + uy * tMin },
    b: { x: mx + ux * tMax, y: my + uy * tMax },
    angleDeg: (Math.atan2(ux, uy) * 180) / Math.PI,   // compass-style for display
    points,
  };
}

/**
 * Fit a cluster to one or more surfaces.
 *
 * Two failure modes have to be handled, and both come from single-link
 * clustering being deliberately permissive:
 *
 *   curved  — the cluster spans a corner, so the principal-axis fit cuts the
 *             corner off.  Split at the point of maximum perpendicular
 *             deviation, which is where the corner sits.
 *   chained — a run of echo arcs at different bearings links across a room.
 *             The chain can be almost straight, so the residual looks fine,
 *             but the fit spans open space its own points never occupied.
 *             The signature is a long *unsupported stretch along the axis*: a
 *             real wall does not have a two-metre hole in the middle of its
 *             own evidence.  Split there.
 */
function fitRecursive(cluster, cfg, out, depth) {
  const fit = fitLine(cluster);
  const canSplit = depth < cfg.maxSplitDepth && cluster.length >= cfg.minClusterPoints * 2;

  // --- chained: split at the largest along-axis gap in the support ---------
  if (canSplit) {
    const along = cluster
      .map((p) => ({ p, t: (p.x - fit.cx) * fit.ux + (p.y - fit.cy) * fit.uy }))
      .sort((a, b) => a.t - b.t);
    let gapAt = -1;
    let gapSize = 0;
    for (let i = 1; i < along.length; i++) {
      const d = along[i].t - along[i - 1].t;
      if (d > gapSize) { gapSize = d; gapAt = i; }
    }
    // Anything wider than a doorway is not a gap in a wall, it is two walls.
    if (gapSize > cfg.maxSupportGap && gapAt > 0) {
      const left = along.slice(0, gapAt).map((e) => e.p);
      const right = along.slice(gapAt).map((e) => e.p);
      if (left.length >= cfg.minClusterPoints && right.length >= cfg.minClusterPoints) {
        fitRecursive(left, cfg, out, depth + 1);
        fitRecursive(right, cfg, out, depth + 1);
        return;
      }
      // One side is too thin to be a surface on its own: keep the dense side
      // and discard the stray points rather than fitting across the gap.
      const dense = left.length >= right.length ? left : right;
      if (dense.length >= cfg.minClusterPoints) {
        fitRecursive(dense, cfg, out, depth + 1);
        return;
      }
      return;
    }
  }

  // --- curved: split at the corner ----------------------------------------
  const tooCurved = fit.rms > cfg.splitResidual;
  if (tooCurved && canSplit) {
    let splitT = 0;
    let worst = -1;
    for (const p of cluster) {
      const dx = p.x - fit.cx, dy = p.y - fit.cy;
      const t = dx * fit.ux + dy * fit.uy;
      const n = Math.abs(-dx * fit.uy + dy * fit.ux);
      if (n > worst) { worst = n; splitT = t; }
    }
    const left = [], right = [];
    for (const p of cluster) {
      const dx = p.x - fit.cx, dy = p.y - fit.cy;
      const t = dx * fit.ux + dy * fit.uy;
      (t <= splitT ? left : right).push(p);
    }
    if (left.length >= cfg.minClusterPoints && right.length >= cfg.minClusterPoints) {
      fitRecursive(left, cfg, out, depth + 1);
      fitRecursive(right, cfg, out, depth + 1);
      return;
    }
  }

  // A fit still this scattered after splitting is not a surface.  The points
  // stay in the cloud and simply go unexplained, which is the honest outcome.
  if (fit.rms > cfg.maxAcceptRms) return;

  out.push(toSegment(fit, cfg));
}

function toSegment(fit, cfg) {
  let wsum = 0, csum = 0, hits = 0;
  const cls = {};
  for (const p of fit.points) {
    const w = Math.max(0.05, p.weight || 1);
    wsum += w;
    csum += (p.classConfidence || 0) * w;
    hits += p.hits || 1;
    if (p.className) cls[p.className] = (cls[p.className] || 0) + w;
  }
  const dominant = Object.keys(cls).sort((a, b) => cls[b] - cls[a])[0] || null;
  // Straightness dominates: a tight fit over many looks is a boundary we can
  // stand behind; a loose fit over three points is a guess, and reads as one.
  const straightness = Math.max(0, 1 - fit.rms / (cfg.splitResidual * 2));
  const density = Math.min(1, fit.points.length / 14);
  const repeat = Math.min(1, hits / (fit.points.length * 3));
  const confidence = clamp01(0.5 * straightness + 0.3 * density + 0.2 * repeat);

  return {
    id: 0,
    a: { x: round3(fit.a.x), y: round3(fit.a.y) },
    b: { x: round3(fit.b.x), y: round3(fit.b.y) },
    cx: round3(fit.cx), cy: round3(fit.cy),
    ux: round3(fit.ux), uy: round3(fit.uy),
    length: round3(fit.length),
    rms: round3(fit.rms),
    support: fit.points.length,
    hits,
    className: dominant,
    classConfidence: round3(wsum ? csum / wsum : 0),
    confidence: round3(confidence),
    angleDeg: round3(fit.angleDeg),
    tMin: fit.tMin, tMax: fit.tMax,
    // Along-axis coordinates of the member points; the gap scan reuses these.
    ts: fit.points.map((p) => round3((p.x - fit.cx) * fit.ux + (p.y - fit.cy) * fit.uy)).sort((x, y) => x - y),
  };
}

// ---------------------------------------------------------------------------
// Features
// ---------------------------------------------------------------------------
/**
 * Corners, found by intersecting the fitted lines rather than by matching
 * endpoints.
 *
 * Endpoint matching alone almost never fires on this sensor: the echo cloud
 * thins out towards a corner (the wall goes oblique to the beam and stops
 * returning), so both fitted segments end well short of the actual corner.
 * Extending the two lines to their intersection recovers it, provided the
 * intersection lies just beyond both segments rather than far off in space.
 */
export function findCorners(segments, cfg = DEF) {
  const corners = [];
  for (let i = 0; i < segments.length; i++) {
    for (let j = i + 1; j < segments.length; j++) {
      const s1 = segments[i], s2 = segments[j];
      const ang = Math.abs(lineAngleBetween(s1, s2));
      if (ang < cfg.cornerAngleDeg) continue;

      const hit = lineIntersection(s1, s2);
      if (!hit) continue;
      // How far past each segment's end the intersection lies.
      const over1 = overshoot(s1, hit);
      const over2 = overshoot(s2, hit);
      const reach = cfg.cornerReach;
      if (over1 > reach || over2 > reach) continue;

      // Closeness of the nearest endpoint pair still matters as evidence.
      let bestD = Infinity;
      for (const [p, q] of [[s1.a, s2.a], [s1.a, s2.b], [s1.b, s2.a], [s1.b, s2.b]]) {
        bestD = Math.min(bestD, Math.hypot(p.x - q.x, p.y - q.y));
      }
      const slack = Math.max(over1, over2);
      corners.push({
        x: round3(hit.x), y: round3(hit.y),
        angleDeg: round3(ang),
        segments: [s1.id, s2.id],
        gapM: round3(bestD),
        confidence: round3(clamp01(Math.min(s1.confidence, s2.confidence) * (1 - slack / (reach * 1.35)))),
      });
    }
  }
  return corners.sort((a, b) => b.confidence - a.confidence);
}

/** Intersection of two fitted lines (not segments); null if near-parallel. */
function lineIntersection(s1, s2) {
  const den = s1.ux * s2.uy - s1.uy * s2.ux;
  if (Math.abs(den) < 1e-6) return null;
  const t = ((s2.cx - s1.cx) * s2.uy - (s2.cy - s1.cy) * s2.ux) / den;
  return { x: s1.cx + s1.ux * t, y: s1.cy + s1.uy * t };
}

/** How far a point lies beyond a segment's own extent, along its axis. */
function overshoot(s, p) {
  const t = (p.x - s.cx) * s.ux + (p.y - s.cy) * s.uy;
  if (t < s.tMin) return s.tMin - t;
  if (t > s.tMax) return t - s.tMax;
  return 0;
}

/**
 * A gap in an otherwise continuous wall, door-width wide, is an opening
 * candidate.  This is geometric inference (a hole in the boundary), reported
 * separately from EchoNet's acoustic OPENING class — two independent kinds of
 * evidence, and the UI says which is which.
 */
export function findGapOpenings(segments, cfg = DEF) {
  const out = [];
  for (const s of segments) {
    const ts = s.ts;
    for (let i = 1; i < ts.length; i++) {
      const gap = ts[i] - ts[i - 1];
      if (gap >= cfg.gapMin && gap <= cfg.gapMax) {
        const tMid = (ts[i] + ts[i - 1]) / 2;
        out.push({
          x: round3(s.cx + s.ux * tMid),
          y: round3(s.cy + s.uy * tMid),
          width: round3(gap),
          evidence: 'geometric-gap',
          segment: s.id,
          confidence: round3(clamp01(s.confidence * 0.7)),
        });
      }
    }
  }
  return out;
}

/**
 * A doorway between two segments.
 *
 * This is the case that actually happens: a door-width hole in a wall is wider
 * than the clustering radius, so the wall arrives here as *two* segments rather
 * than one segment with a hole.  Looking only inside segments (above) would
 * therefore miss every real doorway.  Two nearly-collinear, nearly-aligned
 * segments whose facing ends are a door's width apart is the strongest purely
 * geometric evidence of an opening this sensor can produce.
 */
export function findGapsBetweenSegments(segments, cfg = DEF) {
  const out = [];
  for (let i = 0; i < segments.length; i++) {
    for (let j = i + 1; j < segments.length; j++) {
      const s1 = segments[i];
      const s2 = segments[j];
      // Collinear: same direction, and s2's centre lies close to s1's line.
      if (Math.abs(lineAngleBetween(s1, s2)) > 14) continue;
      const perp = Math.abs((s2.cx - s1.cx) * -s1.uy + (s2.cy - s1.cy) * s1.ux);
      if (perp > 0.28) continue;

      // Closest pair of endpoints is the gap.
      const pairs = [[s1.a, s2.a], [s1.a, s2.b], [s1.b, s2.a], [s1.b, s2.b]];
      let best = null;
      for (const [p, q] of pairs) {
        const d = Math.hypot(p.x - q.x, p.y - q.y);
        if (!best || d < best.d) best = { d, p, q };
      }
      if (!best || best.d < cfg.gapMin || best.d > cfg.gapMax) continue;

      // Both sides must be substantial: a stub either side of a "gap" is more
      // likely a clustering artefact than a door.
      if (s1.length < 0.3 || s2.length < 0.3) continue;

      out.push({
        x: round3((best.p.x + best.q.x) / 2),
        y: round3((best.p.y + best.q.y) / 2),
        width: round3(best.d),
        evidence: 'geometric-gap',
        segment: s1.id,
        between: [s1.id, s2.id],
        confidence: round3(clamp01(Math.min(s1.confidence, s2.confidence) * (1 - perp / 0.4) * 0.85)),
      });
    }
  }
  return out;
}

/**
 * Openings the classifier itself proposed.
 *
 * Nothing in the live path produces these any more — the pipeline no longer
 * emits an acoustic OPENING class at all, because an opening is the absence of
 * a return rather than a texture, and the peak CFAR locks onto through a
 * doorway belongs to the far wall behind it.  See shared/surfaceclass.mjs.
 * Openings now come from findGapOpenings() instead.
 *
 * This is kept so recordings captured before that change still replay, and the
 * confidence cap stays: a lone acoustic OPENING echo was always weak evidence
 * and must never render like a confirmed doorway.
 */
export function openingCandidatesFromPoints(openingPts, segments) {
  const clusters = clusterPoints(openingPts, 0.5, 2);
  return clusters.map((cl) => {
    let x = 0, y = 0, c = 0, hits = 0;
    for (const p of cl) { x += p.x; y += p.y; c += p.classConfidence || 0; hits += p.hits || 1; }
    const n = cl.length;
    const agreement = Math.min(1, hits / (n * 3));
    return {
      x: round3(x / n), y: round3(y / n),
      width: null,
      evidence: 'acoustic-class',
      support: n,
      // 0.6 cap: the model's own OPENING recall does not justify more.
      confidence: round3(clamp01((c / n) * 0.6 * (0.5 + 0.5 * agreement))),
    };
  });
}

function dedupeOpenings(list) {
  const out = [];
  for (const o of list) {
    const near = out.find((q) => Math.hypot(q.x - o.x, q.y - o.y) < 0.6);
    if (near) {
      // Two independent kinds of evidence at one spot is worth more than either.
      if (near.evidence !== o.evidence) {
        near.confidence = round3(clamp01(near.confidence + o.confidence * 0.5));
        near.evidence = 'geometric+acoustic';
        if (o.width && !near.width) near.width = o.width;
      } else if (o.confidence > near.confidence) {
        Object.assign(near, o);
      }
    } else out.push(Object.assign({}, o));
  }
  return out.sort((a, b) => b.confidence - a.confidence);
}

/** Two long roughly-parallel walls a walkable distance apart = corridor. */
export function findCorridors(segments, cfg = DEF) {
  const out = [];
  const longs = segments.filter((s) => s.length > 1.0);
  for (let i = 0; i < longs.length; i++) {
    for (let j = i + 1; j < longs.length; j++) {
      const s1 = longs[i], s2 = longs[j];
      const ang = Math.abs(lineAngleBetween(s1, s2));
      if (ang > 20 && ang < 160) continue;
      const d = Math.abs((s2.cx - s1.cx) * -s1.uy + (s2.cy - s1.cy) * s1.ux);
      if (d < 0.7 || d > cfg.corridorWidthMax) continue;
      const overlap = axisOverlap(s1, s2);
      if (overlap < 0.8) continue;
      out.push({
        width: round3(d),
        length: round3(overlap),
        axisDeg: round3(s1.angleDeg),
        mid: { x: round3((s1.cx + s2.cx) / 2), y: round3((s1.cy + s2.cy) / 2) },
        segments: [s1.id, s2.id],
        confidence: round3(Math.min(s1.confidence, s2.confidence)),
      });
    }
  }
  return out;
}

function axisOverlap(s1, s2) {
  // Project s2's endpoints onto s1's axis and measure the shared span.
  const p = (x, y) => (x - s1.cx) * s1.ux + (y - s1.cy) * s1.uy;
  const t1 = [s1.tMin, s1.tMax];
  const t2 = [p(s2.a.x, s2.a.y), p(s2.b.x, s2.b.y)].sort((a, b) => a - b);
  return Math.max(0, Math.min(t1[1], t2[1]) - Math.max(t1[0], t2[0]));
}

function lineAngleBetween(s1, s2) {
  const dot = s1.ux * s2.ux + s1.uy * s2.uy;
  return (Math.acos(Math.max(-1, Math.min(1, Math.abs(dot)))) * 180) / Math.PI;
}

/**
 * Overall reconstruction confidence, reported next to the rendered geometry.
 * Three things have to be true at once: most of the cloud was explained by
 * some surface, there is enough total boundary to be a room rather than a
 * fragment, and the individual fits are decent.
 */
export function reconstructionConfidence(segments, nStructural, inliers) {
  if (!segments.length || !nStructural) return 0;
  const explained = inliers / nStructural;
  const total = segments.reduce((a, s) => a + s.length, 0);
  const extent = Math.min(1, total / 8);
  const meanFit = segments.reduce((a, s) => a + s.confidence, 0) / segments.length;
  return round3(clamp01(0.4 * explained + 0.25 * extent + 0.35 * meanFit));
}

function clamp01(v) { return v < 0 ? 0 : v > 1 ? 1 : v; }
function round3(v) { return Math.round(v * 1000) / 1000; }
