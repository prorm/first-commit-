/**
 * Spatial engine, protocol and simulation tests.
 *
 * These cover the layers between the sensor and the screen: coordinate
 * conversion, occupancy updates, point-cloud consolidation, temporal fusion,
 * surface reconstruction, wire validation and deterministic replay.
 */
const test = require('node:test');
const assert = require('node:assert');

const {
  normalizeDetection, validateMessage, polarToWorld, makePose, envelope,
  angleDelta, wrapDeg, timeToContact, PROTOCOL_VERSION, MESSAGE_TYPES,
} = require('../public/shared/protocol.mjs');
const { OccupancyGrid, PointCloud, Trajectory, ClassFuser } = require('../public/shared/spatial.mjs');
const { surfaceFromProbs, SURFACE_CLASSES } = require('../public/shared/surfaceclass.mjs');
const { BoundaryMap } = require('../public/shared/boundary.mjs');
const { reconstruct, fitLine, clusterPoints, findCorners } = require('../public/shared/reconstruct.mjs');
const { SimulationEngine, SCENARIOS, raycast } = require('../server/simulation');
const { MapState } = require('../server/mapstate');
const { GuidancePolicy, relativeDirection } = require('../server/guidance');
const { Recorder, ReplayPlayer } = require('../server/recorder');
const { AwsAdapter } = require('../server/aws/adapter');

// ---------------------------------------------------------------------------
// Coordinate conversion
// ---------------------------------------------------------------------------
test('polar to world uses compass bearings with +y north', () => {
  const at = { x: 0, y: 0, heading: 0 };
  const n = polarToWorld(at, 0, 1);
  assert.ok(Math.abs(n.x) < 1e-9 && Math.abs(n.y - 1) < 1e-9, 'bearing 0 is north');
  const e = polarToWorld(at, 90, 1);
  assert.ok(Math.abs(e.x - 1) < 1e-9 && Math.abs(e.y) < 1e-9, 'bearing 90 is east');
  const s = polarToWorld(at, 180, 1);
  assert.ok(Math.abs(s.x) < 1e-9 && Math.abs(s.y + 1) < 1e-9, 'bearing 180 is south');
  const w = polarToWorld(at, 270, 1);
  assert.ok(Math.abs(w.x + 1) < 1e-9 && Math.abs(w.y) < 1e-9, 'bearing 270 is west');
});

test('polar to world is relative to the phone position', () => {
  const p = polarToWorld({ x: 3, y: -2, heading: 45 }, 45, Math.SQRT2);
  assert.ok(Math.abs(p.x - 4) < 1e-9, 'x should be 3 + 1');
  assert.ok(Math.abs(p.y - -1) < 1e-9, 'y should be -2 + 1');
});

test('angle helpers wrap correctly', () => {
  assert.equal(wrapDeg(-90), 270);
  assert.equal(wrapDeg(450), 90);
  assert.equal(angleDelta(350, 10), 20, 'crossing north is a +20 deg turn');
  assert.equal(angleDelta(10, 350), -20);
  assert.equal(Math.abs(angleDelta(0, 180)), 180, 'an exact half turn is +-180');
});

test('time to contact is clamped and safe at zero velocity', () => {
  assert.equal(timeToContact(2, 0), 10, 'standing still clamps to the 10 s cap');
  assert.ok(Math.abs(timeToContact(1, 1) - 1) < 1e-9);
  assert.equal(timeToContact(0.1, 5), 0.02);
  assert.ok(Number.isFinite(timeToContact(2, -3)), 'receding must not produce NaN or Infinity');
});

// ---------------------------------------------------------------------------
// Protocol
// ---------------------------------------------------------------------------
test('every documented message type has a direction', () => {
  for (const [name, spec] of Object.entries(MESSAGE_TYPES)) {
    assert.ok(spec.from, name + ' must declare a direction');
    assert.ok(spec.doc && spec.doc.length > 8, name + ' must be documented');
  }
  assert.equal(PROTOCOL_VERSION, 1);
});

test('detection normalisation repairs bad input instead of throwing', () => {
  assert.equal(normalizeDetection(null), null);
  assert.equal(normalizeDetection({}), null, 'no range means no detection');
  assert.equal(normalizeDetection({ range_m: -1 }), null);
  assert.equal(normalizeDetection({ range_m: 1e9 }), null);

  const d = normalizeDetection({
    range_m: 2, vel_mps: 1e6, confidence: -4, classProbs: [5, 5, 0],
    bearing_deg: -90, beamwidth_deg: 9999, snr_db: 1e6,
  });
  assert.equal(d.vel_mps, 5, 'velocity clamped');
  assert.equal(d.confidence, 0, 'confidence clamped');
  assert.equal(d.bearing_deg, 270, 'bearing wrapped');
  assert.equal(d.beamwidth_deg, 120, 'beamwidth clamped');
  assert.equal(d.snr_db, 60, 'snr clamped');
  assert.ok(Math.abs(d.classProbs[0] - 0.5) < 1e-9, 'probabilities renormalised');
  assert.ok(Number.isFinite(d.worldX) && Number.isFinite(d.worldY));
  assert.equal(d.phone.method, 'static', 'a missing pose becomes an explicit static pose');
});

test('detection bearing defaults to the phone heading', () => {
  const d = normalizeDetection({ range_m: 1, phone: { x: 0, y: 0, heading: 135 } });
  assert.equal(d.bearing_deg, 135, 'a mono sensor points where the phone points');
});

test('message validation accepts the good and rejects the bad', () => {
  assert.equal(validateMessage('not json').ok, false);
  assert.equal(validateMessage('[]').ok, false);
  assert.equal(validateMessage({ }).ok, false, 'missing type');
  assert.equal(validateMessage({ type: 'made_up' }).ok, false);
  assert.equal(validateMessage({ type: 'hello', v: 7 }).ok, false, 'version mismatch');
  assert.equal(validateMessage({ type: 'hello', role: 'toaster' }).ok, false);
  assert.equal(validateMessage({ type: 'set_mode', mode: 'telepathy' }).ok, false);

  assert.equal(validateMessage({ type: 'hello', role: 'phone' }).ok, true);
  assert.equal(validateMessage({ type: 'set_mode', mode: 'hybrid' }).ok, true);
  const env = envelope('heartbeat', { clientTime: 1 });
  const v = validateMessage(JSON.stringify(env));
  assert.equal(v.ok, true);
  assert.equal(v.msg.clientTime, 1);
});

test('pose normalisation fills every field', () => {
  const p = makePose({ x: 1, heading: -45 });
  assert.equal(p.y, 0);
  assert.equal(p.heading, 315);
  assert.ok(p.confidence >= 0 && p.confidence <= 1);
  assert.equal(typeof p.method, 'string');
});

// ---------------------------------------------------------------------------
// Occupancy grid
// ---------------------------------------------------------------------------
test('occupancy grid keeps unknown, free and occupied distinct', () => {
  const g = new OccupancyGrid({ cell: 0.1, halfExtent: 8 });
  const far = g.toCell(5, 5);
  assert.equal(g.get(far.cx, far.cy), 0, 'untouched cells are exactly zero');
  assert.ok(Math.abs(g.prob(far.cx, far.cy) - 0.5) < 1e-9, 'unknown means p = 0.5');

  g.integrate({ x: 0, y: 0, heading: 0 }, 0, 2, { weight: 1, className: 'WALL' });
  const hit = g.toCell(0, 2);
  const mid = g.toCell(0, 1);
  assert.ok(g.get(hit.cx, hit.cy) > 0, 'reflector cell occupied');
  assert.ok(g.prob(hit.cx, hit.cy) > 0.6);
  assert.ok(g.get(mid.cx, mid.cy) < 0, 'traversed cell free');
  assert.ok(g.prob(mid.cx, mid.cy) < 0.4);
  assert.equal(g.get(far.cx, far.cy), 0, 'a beam elsewhere leaves distant cells unknown');
});

test('repeated observations reinforce and saturate', () => {
  const g = new OccupancyGrid({ cell: 0.1, halfExtent: 8 });
  const cell = g.toCell(0, 1.5);
  let prev = 0;
  for (let i = 0; i < 6; i++) {
    g.integrate({ x: 0, y: 0, heading: 0 }, 0, 1.5, { weight: 1, className: 'WALL' });
    const v = g.get(cell.cx, cell.cy);
    assert.ok(v >= prev, 'occupancy should not decrease with repeat looks');
    prev = v;
  }
  for (let i = 0; i < 80; i++) g.integrate({ x: 0, y: 0, heading: 0 }, 0, 1.5, { weight: 1, className: 'WALL' });
  assert.ok(g.get(cell.cx, cell.cy) <= g.lClamp + 1e-6, 'log-odds saturate rather than run away');
});

test('an OPENING detection carves free space instead of a wall', () => {
  const g = new OccupancyGrid({ cell: 0.1, halfExtent: 8 });
  g.integrate({ x: 0, y: 0, heading: 0 }, 0, 2, { weight: 1, className: 'OPENING' });
  const at = g.toCell(0, 2);
  assert.ok(g.get(at.cx, at.cy) <= 0,
    'a doorway is evidence of absence: it must not paint an occupied cell');
  const beyond = g.toCell(0, 2.2);
  assert.ok(g.get(beyond.cx, beyond.cy) < 0, 'space just past the doorway reads free');
});

test('weight scales how hard a detection marks the map', () => {
  const strong = new OccupancyGrid({ cell: 0.1, halfExtent: 8 });
  const weak = new OccupancyGrid({ cell: 0.1, halfExtent: 8 });
  strong.integrate({ x: 0, y: 0, heading: 0 }, 0, 2, { weight: 1, className: 'WALL' });
  weak.integrate({ x: 0, y: 0, heading: 0 }, 0, 2, { weight: 0.2, className: 'WALL' });
  const c = strong.toCell(0, 2);
  assert.ok(strong.get(c.cx, c.cy) > weak.get(c.cx, c.cy) * 2,
    'a low-confidence detection leaves a fainter mark');
});

test('occupancy grid round-trips through the wire format', () => {
  const g = new OccupancyGrid({ cell: 0.1, halfExtent: 6 });
  for (let b = 0; b < 360; b += 20) {
    g.integrate({ x: 0, y: 0, heading: b }, b, 1.5, { weight: 0.8, className: 'WALL' });
  }
  const ser = g.serialize();
  assert.ok(ser.idx.length > 100, 'serialisation should carry the touched cells');
  assert.equal(ser.idx.length, ser.val.length);
  const back = OccupancyGrid.deserialize(ser);
  let maxDiff = 0;
  for (let i = 0; i < g.data.length; i++) maxDiff = Math.max(maxDiff, Math.abs(g.data[i] - back.data[i]));
  assert.ok(maxDiff < 0.1, 'byte quantisation error stays under 0.1 log-odds, got ' + maxDiff);
  assert.ok(back.areaStats().occupiedArea > 0);
});

// ---------------------------------------------------------------------------
// Point cloud + trajectory
// ---------------------------------------------------------------------------
test('point cloud consolidates nearby observations', () => {
  const c = new PointCloud();
  const mk = (x, y) => normalizeDetection({
    range_m: 1, bearing_deg: 0, confidence: 0.7,
    obstacleClass: 'WALL', classConfidence: 0.8, classProbs: [0.8, 0.1, 0.1],
    phone: { x, y: y - 1, heading: 0 },
  });
  for (let i = 0; i < 6; i++) c.add(mk(0.02 * i, 0));
  assert.equal(c.points.length, 1, 'six looks at one spot is one point');
  assert.equal(c.points[0].hits, 6);
  assert.ok(c.points[0].weight > 0.7, 'weight accumulates with hits');

  c.add(mk(3, 0));
  assert.equal(c.points.length, 2, 'a distant observation is a separate point');
});

test('point cloud merges across spatial-hash bucket seams', () => {
  const c = new PointCloud();
  const r = c.mergeRadius;
  // Straddle a bucket boundary: naive single-bucket lookup would miss this.
  const a = normalizeDetection({ range_m: 1, bearing_deg: 90, confidence: 0.7, phone: { x: r - 0.01, y: 0, heading: 90 } });
  const b = normalizeDetection({ range_m: 1, bearing_deg: 90, confidence: 0.7, phone: { x: r + 0.01, y: 0, heading: 90 } });
  c.add(a);
  c.add(b);
  assert.equal(c.points.length, 1, 'points 2 cm apart must merge even across a bucket seam');
});

test('point cloud votes on class rather than taking the last answer', () => {
  const c = new PointCloud();
  const mk = (cls, conf) => normalizeDetection({
    range_m: 1, bearing_deg: 0, confidence: 0.8,
    obstacleClass: cls, classConfidence: conf,
    classProbs: cls === 'WALL' ? [conf, 0, 0] : [0, conf, 0],
    phone: { x: 0, y: -1, heading: 0 },
  });
  c.add(mk('WALL', 0.9));
  c.add(mk('WALL', 0.9));
  c.add(mk('SOFT', 0.6));
  assert.equal(c.points[0].className, 'WALL', 'two confident WALLs outvote one weak SOFT');
  assert.ok(c.points[0].classConfidence < 1, 'disagreement must reduce confidence');
});

test('point cloud stays bounded under sustained input', () => {
  const c = new PointCloud({ maxPoints: 200 });
  for (let i = 0; i < 2000; i++) {
    c.add(normalizeDetection({
      range_m: 1 + (i % 30) * 0.1, bearing_deg: (i * 7) % 360, confidence: 0.5,
      phone: { x: (i % 50) * 0.3, y: (i % 37) * 0.3, heading: 0 },
    }));
  }
  assert.ok(c.points.length <= 200, 'cloud must not grow without bound, got ' + c.points.length);
});

test('trajectory records movement but not jitter', () => {
  const t = new Trajectory(0.06);
  t.push(makePose({ x: 0, y: 0, heading: 0 }), 1);
  t.push(makePose({ x: 0.01, y: 0, heading: 0 }), 2);
  assert.equal(t.nodes.length, 1, '1 cm of jitter is not a step');
  t.push(makePose({ x: 0.5, y: 0, heading: 0 }), 3);
  assert.equal(t.nodes.length, 2);
  assert.ok(Math.abs(t.distance - 0.5) < 1e-9);
  // A pure rotation is worth recording even without translation.
  t.push(makePose({ x: 0.5, y: 0, heading: 40 }), 4);
  assert.equal(t.nodes.length, 3, 'a large turn in place is a trajectory event');
});

// ---------------------------------------------------------------------------
// Temporal fusion
// ---------------------------------------------------------------------------
test('fusion smooths a noisy but consistent sequence', () => {
  const f = new ClassFuser();
  const seq = [[0.55, 0.25, 0.20], [0.62, 0.20, 0.18], [0.71, 0.17, 0.12], [0.80, 0.12, 0.08]];
  let r = null;
  const t0 = Date.now();
  seq.forEach((probs, i) => {
    r = f.fuse(normalizeDetection({
      t: t0 + i * 50, range_m: 2, bearing_deg: 0, confidence: 0.8,
      classProbs: probs, phone: { x: 0, y: 0, heading: 0 },
    }));
  });
  assert.equal(r.className, 'WALL');
  assert.equal(r.support, 4);
  assert.ok(r.stable, 'four agreeing looks is stable');
  assert.ok(r.confidence > 0.55 && r.confidence < 0.8,
    'fused confidence sits inside the observed range, not above it: ' + r.confidence.toFixed(3));
});

test('fusion refuses to call an alternating sequence stable', () => {
  const f = new ClassFuser();
  const alt = [[0.6, 0.3, 0.1], [0.2, 0.7, 0.1], [0.6, 0.3, 0.1], [0.2, 0.7, 0.1], [0.55, 0.35, 0.1]];
  let r = null;
  const t0 = Date.now();
  alt.forEach((probs, i) => {
    r = f.fuse(normalizeDetection({
      t: t0 + i * 50, range_m: 2, bearing_deg: 0, confidence: 0.8,
      classProbs: probs, phone: { x: 0, y: 0, heading: 0 },
    }));
  });
  assert.equal(r.stable, false, 'disagreement must be reported, not averaged away');
});

test('fusion separates spatially distinct targets', () => {
  const f = new ClassFuser();
  const t0 = Date.now();
  const near = f.fuse(normalizeDetection({
    t: t0, range_m: 1, bearing_deg: 0, confidence: 0.8,
    classProbs: [0.9, 0.05, 0.05], phone: { x: 0, y: 0, heading: 0 },
  }));
  const far = f.fuse(normalizeDetection({
    t: t0 + 50, range_m: 3, bearing_deg: 0, confidence: 0.8,
    classProbs: [0.05, 0.9, 0.05], phone: { x: 0, y: 0, heading: 0 },
  }));
  assert.equal(near.className, 'WALL');
  assert.equal(far.className, 'SOFT', 'a target 2 m away is a different track');
  assert.equal(far.support, 1);
});

test('fusion stays bounded over a long scan', () => {
  const f = new ClassFuser();
  const t0 = Date.now();
  for (let i = 0; i < 1200; i++) {
    f.fuse(normalizeDetection({
      t: t0 + i * 50, range_m: 1 + (i % 40) * 0.08, bearing_deg: (i * 11) % 360,
      confidence: 0.7, classProbs: [0.5, 0.3, 0.2],
      phone: { x: (i % 60) * 0.2, y: 0, heading: 0 },
    }));
  }
  assert.ok(f.tracks.length <= 64, 'fuser track list must stay bounded, got ' + f.tracks.length);
});

// ---------------------------------------------------------------------------
// Reconstruction
// ---------------------------------------------------------------------------
function wallPoints(x0, y0, x1, y1, n, jitter = 0) {
  const out = [];
  for (let i = 0; i <= n; i++) {
    const t = i / n;
    out.push({
      x: x0 + (x1 - x0) * t + (Math.sin(i * 12.9898) * jitter),
      y: y0 + (y1 - y0) * t + (Math.cos(i * 78.233) * jitter),
      weight: 1.5, hits: 3, className: 'WALL', classConfidence: 0.9,
    });
  }
  return out;
}

test('line fitting recovers direction and extent', () => {
  const fit = fitLine(wallPoints(0, 0, 3, 0, 20));
  assert.ok(Math.abs(fit.length - 3) < 0.02, 'length ' + fit.length);
  assert.ok(fit.rms < 1e-6, 'a straight line has no residual');
  assert.ok(Math.abs(Math.abs(fit.ux) - 1) < 1e-6, 'axis should be along x');
});

test('clustering separates surfaces that are far apart', () => {
  const pts = wallPoints(0, 0, 2, 0, 16).concat(wallPoints(0, 4, 2, 4, 16));
  const clusters = clusterPoints(pts, 0.38, 4);
  assert.equal(clusters.length, 2, 'two walls 4 m apart are two clusters');
});

// ---------------------------------------------------------------------------
// Live boundaries
// ---------------------------------------------------------------------------

function echo(x, y, bearing, range, opts = {}) {
  return normalizeDetection(Object.assign({
    t: opts.t || 1000, range_m: range, bearing_deg: bearing, confidence: 0.9,
    cfar_pass: true, obstacleClass: 'WALL', classConfidence: 0.9,
    classProbs: [0.9, 0.05, 0.05], fusedClass: 'WALL',
    phone: { x, y, heading: bearing, confidence: 1 },
  }, opts));
}

test('a single confident echo draws a wall tangent, not a point', () => {
  const bm = new BoundaryMap();
  assert.ok(bm.add(echo(0, 0, 0, 2.5)), 'one confident echo is enough');
  const [s] = bm.segments();
  // Facing north at a wall 2.5 m out: the bar sits at y = 2.5 and runs east-west.
  assert.ok(Math.abs(s.a.y - 2.5) < 1e-6 && Math.abs(s.b.y - 2.5) < 1e-6,
    'the bar lies on the range arc at y = 2.5');
  assert.ok(Math.abs(s.a.x + s.b.x) < 1e-6, 'centred on the boresight');
  const len = Math.hypot(s.b.x - s.a.x, s.b.y - s.a.y);
  assert.ok(len > 0.3, 'the bar spans the beam, it is not a dot: ' + len.toFixed(2));
});

test('boundaries are pinned in world space as the operator walks', () => {
  const bm = new BoundaryMap();
  // Walk 0 -> 1.95 m north at a wall at y = 2.5, 40 pulses.
  for (let i = 0; i < 40; i++) {
    const y = i * 0.05;
    bm.add(echo(0, y, 0, 2.5 - y, { t: 1000 + i * 50 }));
  }
  assert.equal(bm.length, 1, 'forty looks at one wall stay one wall, not forty bars');
  const [s] = bm.segments();
  assert.ok(Math.abs(s.a.y - 2.5) < 0.02 && Math.abs(s.b.y - 2.5) < 0.02,
    'the wall stays at y = 2.5 while the sensor moves: ' + s.a.y.toFixed(3));
  assert.ok(s.hits === 40, 'every look counted: ' + s.hits);
});

test('turning 180 degrees leaves the wall behind you where it was', () => {
  const bm = new BoundaryMap();
  for (let i = 0; i < 6; i++) bm.add(echo(0, 0, 0, 2.0, { t: 1000 + i * 50 }));
  const before = bm.segments()[0];
  // Now face south at the opposite wall.
  for (let i = 0; i < 6; i++) bm.add(echo(0, 0, 180, 2.0, { t: 2000 + i * 50 }));
  const segs = bm.segments();
  assert.equal(segs.length, 2, 'two walls, one each side');
  const kept = segs.find((s) => s.id === before.id);
  assert.ok(kept, 'the wall behind the operator survives the turn');
  assert.ok(Math.abs(kept.a.y - 2.0) < 1e-6, 'and has not moved');
  assert.ok(segs.some((s) => Math.abs(s.a.y + 2.0) < 1e-6), 'the new wall is at y = -2');
});

test('a bar a later pulse measures straight through is withdrawn', () => {
  const bm = new BoundaryMap();
  for (let i = 0; i < 4; i++) bm.add(echo(0, 0, 0, 1.0, { t: 1000 + i * 50 }));
  assert.equal(bm.length, 1, 'a wall at 1 m');
  // Same bearing, but the echo now comes from 3 m: the 1 m bar cannot be there.
  for (let i = 0; i < 8; i++) bm.add(echo(0, 0, 0, 3.0, { t: 2000 + i * 50 }));
  const ys = bm.segments().map((s) => s.a.y);
  assert.ok(!ys.some((y) => Math.abs(y - 1.0) < 0.2),
    'free space withdrew the contradicted bar: ' + JSON.stringify(ys));
  assert.ok(ys.some((y) => Math.abs(y - 3.0) < 0.2), 'and the real wall is drawn');
});

test('weak, non-CFAR and unclassified echoes never draw a boundary', () => {
  const bm = new BoundaryMap();
  assert.equal(bm.add(echo(0, 0, 0, 2, { confidence: 0.4 })), null, 'low confidence');
  assert.equal(bm.add(echo(0, 0, 0, 2, { cfar_pass: false })), null, 'below CFAR');
  assert.equal(bm.length, 0, 'nothing drawn from weak evidence');
});

test('OPENING can never become a live boundary', () => {
  // An opening is the absence of a return; it cannot produce a tangent. Even a
  // legacy replay frame carrying the class must not draw a wall.
  const bm = new BoundaryMap();
  const legacy = normalizeDetection({
    t: 1000, range_m: 2, bearing_deg: 0, confidence: 0.95, cfar_pass: true,
    obstacleClass: 'OPENING', fusedClass: 'OPENING', classConfidence: 0.95,
    classProbs: [0.02, 0.03, 0.95], phone: { x: 0, y: 0, heading: 0, confidence: 1 },
  });
  assert.equal(bm.add(legacy), null, 'OPENING draws nothing');
  assert.equal(bm.length, 0);
});

test('the boundary layer stays bounded under a long scan', () => {
  const bm = new BoundaryMap();
  let t = 1000;
  for (let lap = 0; lap < 40; lap++) {
    for (let d = 0; d < 360; d += 5) {
      bm.add(echo(0, 0, d, 2 + (lap % 3) * 0.5, { t: (t += 50) }));
    }
  }
  assert.ok(bm.length <= bm.cfg.maxBars, 'hard cap holds: ' + bm.length);
  assert.ok(bm.segments().every((s) => Number.isFinite(s.a.x) && Number.isFinite(s.b.y)),
    'no NaN geometry survives merging');
});

test('reconstruction fits a straight wall and reports low residual', () => {
  const r = reconstruct(wallPoints(-1.5, 2, 1.5, 2, 28, 0.01));
  assert.equal(r.segments.length, 1);
  assert.ok(r.segments[0].length > 2.8, 'should span the wall');
  assert.ok(r.segments[0].rms < 0.05, 'rms ' + r.segments[0].rms);
  assert.ok(r.segments[0].confidence > 0.6);
  assert.ok(r.confidence > 0.4);
});

test('reconstruction splits an L into two surfaces and finds the corner', () => {
  const pts = wallPoints(0, 0, 0, 2, 18, 0.01).concat(wallPoints(0, 2, 2, 2, 18, 0.01));
  const r = reconstruct(pts);
  assert.ok(r.segments.length >= 2, 'an L is not one wall, got ' + r.segments.length);
  assert.ok(r.corners.length >= 1, 'the corner should be reported');
  const k = r.corners[0];
  assert.ok(Math.hypot(k.x - 0, k.y - 2) < 0.4,
    'corner should land near (0, 2), got (' + k.x + ', ' + k.y + ')');
  assert.ok(k.angleDeg > 70 && k.angleDeg < 110, 'right angle, got ' + k.angleDeg);
});

test('reconstruction finds corners the echo cloud never reached', () => {
  // Both walls stop 0.5 m short of the corner, as a real beam would.
  const pts = wallPoints(0, 0, 0, 1.5, 14, 0.01).concat(wallPoints(0.5, 2, 2, 2, 14, 0.01));
  const r = reconstruct(pts);
  assert.ok(r.corners.length >= 1,
    'extending the fits should still find the corner despite the gap in evidence');
});

test('reconstruction proposes a doorway between two collinear walls', () => {
  const pts = wallPoints(-2, 2, -0.45, 2, 16, 0.01).concat(wallPoints(0.45, 2, 2, 2, 16, 0.01));
  const r = reconstruct(pts);
  const gaps = r.openings.filter((o) => o.evidence === 'geometric-gap');
  assert.ok(gaps.length >= 1, 'a 0.9 m hole in a wall is an opening candidate');
  assert.ok(Math.abs(gaps[0].x) < 0.35, 'candidate should sit in the gap, got x = ' + gaps[0].x);
  assert.ok(gaps[0].width > 0.6 && gaps[0].width < 1.3, 'width ' + gaps[0].width);
});

test('reconstruction ignores gaps too wide to be a doorway', () => {
  const pts = wallPoints(-4, 2, -2, 2, 16, 0.01).concat(wallPoints(2, 2, 4, 2, 16, 0.01));
  const r = reconstruct(pts);
  const gaps = r.openings.filter((o) => o.evidence === 'geometric-gap');
  assert.equal(gaps.length, 0, 'a 4 m gap is unscanned space, not a door');
});

test('OPENING-classified echoes are damped, never confident', () => {
  const pts = [];
  for (let i = 0; i < 6; i++) {
    pts.push({ x: 0.05 * i, y: 2, weight: 2, hits: 6, className: 'OPENING', classConfidence: 1.0 });
  }
  const r = reconstruct(pts);
  const acoustic = r.openings.filter((o) => /acoustic/.test(o.evidence));
  assert.ok(acoustic.length >= 1);
  assert.ok(acoustic[0].confidence <= 0.6,
    'the model\'s 43 % OPENING recall must cap displayed confidence, got ' + acoustic[0].confidence);
  assert.equal(r.segments.length, 0, 'OPENING points are absence evidence, not boundary evidence');
});

test('reconstruction finds a corridor between parallel walls', () => {
  const pts = wallPoints(-0.8, -2, -0.8, 2, 30, 0.01).concat(wallPoints(0.8, -2, 0.8, 2, 30, 0.01));
  const r = reconstruct(pts);
  assert.ok(r.corridors.length >= 1, 'two parallel walls 1.6 m apart is a corridor');
  assert.ok(Math.abs(r.corridors[0].width - 1.6) < 0.2, 'width ' + r.corridors[0].width);
});

test('reconstruction returns empty, not broken, with no evidence', () => {
  for (const input of [null, [], [{ x: 0, y: 0, weight: 1, hits: 1 }]]) {
    const r = reconstruct(input);
    assert.ok(Array.isArray(r.segments) && r.segments.length === 0);
    assert.equal(r.confidence, 0);
    assert.ok(Array.isArray(r.openings));
  }
});

// ---------------------------------------------------------------------------
// Simulation
// ---------------------------------------------------------------------------
test('raycasting hits the nearest surface with the right kind', () => {
  const surfaces = SCENARIOS.room.surfaces;
  // From the room centre, looking south, the south wall is 2 m away.
  const hit = raycast(surfaces, 0, 0, 180, 6);
  assert.ok(hit, 'should hit the south wall');
  assert.ok(Math.abs(hit.range - 2) < 0.02, 'range ' + hit.range);
  assert.equal(hit.kind, 'WALL');
  assert.ok(hit.cosInc > 0.9, 'normal incidence on a wall straight ahead');

  // Looking north through the doorway gap near x = +0.4.
  const door = raycast(surfaces, 0.4, 0, 0, 6);
  assert.ok(door, 'should hit the doorway plane');
  assert.equal(door.kind, 'OPENING', 'the doorway must report as an OPENING surface');
});

test('raycast returns null when nothing is in range', () => {
  const hit = raycast(SCENARIOS.openfield.surfaces, 0, 0, 180, 4);
  assert.equal(hit, null, 'an open direction should return nothing, not a fabricated echo');
});

test('every scenario is geometrically sane', () => {
  for (const [id, sc] of Object.entries(SCENARIOS)) {
    assert.ok(sc.surfaces.length > 0, id + ' needs surfaces');
    assert.ok(sc.route.length >= 2, id + ' needs a route');
    assert.ok(sc.label && sc.description, id + ' needs a label and description');
    for (const s of sc.surfaces) {
      const len = Math.hypot(s.b[0] - s.a[0], s.b[1] - s.a[1]);
      assert.ok(len > 0.05, id + ' has a degenerate surface');
      assert.ok(['WALL', 'SOFT', 'OPENING'].includes(s.kind), id + ' has an unknown surface kind');
    }
    // The walker must start inside, not embedded in a wall.
    const clearance = Math.min.apply(null, sc.surfaces.map((s) => {
      const dx = s.b[0] - s.a[0], dy = s.b[1] - s.a[1];
      const l2 = dx * dx + dy * dy;
      let t = ((sc.start.x - s.a[0]) * dx + (sc.start.y - s.a[1]) * dy) / l2;
      t = Math.max(0, Math.min(1, t));
      return Math.hypot(s.a[0] + dx * t - sc.start.x, s.a[1] + dy * t - sc.start.y);
    }));
    assert.ok(clearance > 0.3, id + ' starts only ' + clearance.toFixed(2) + ' m from a surface');
  }
});

test('simulation emits schema-valid detections with real classifier output', () => {
  const dets = [];
  const sim = new SimulationEngine({ onDetection: (d) => { if (d) dets.push(d); } });
  sim.setScenario('room', 777);
  for (let i = 0; i < 200; i++) sim.tick(0.05);
  sim.stop();

  assert.ok(dets.length > 100, 'expected a detection stream, got ' + dets.length);
  for (const d of dets.slice(0, 30)) {
    const v = validateMessage({ type: 'detection', detection: d });
    assert.ok(v.ok, 'simulated detections must pass the same validation as live ones');
    assert.equal(d.source, 'simulation');
    assert.ok(d.range_m > 0.2 && d.range_m < 6);
    assert.ok(Math.abs(d.classProbs.reduce((a, b) => a + b, 0) - 1) < 1e-3);
  }
  // Class output must vary — a constant answer would mean the model is not running.
  const classes = new Set(dets.map((d) => d.obstacleClass));
  assert.ok(classes.size >= 2, 'classifier should produce more than one class across a room');
  const confs = new Set(dets.map((d) => Math.round(d.classConfidence * 50)));
  assert.ok(confs.size > 4, 'class confidence must vary, not be scripted');
});

test('simulation is deterministic for a given seed', () => {
  const run = () => {
    const out = [];
    const sim = new SimulationEngine({ onDetection: (d) => { if (d) out.push(d.range_m.toFixed(4) + ':' + d.obstacleClass); } });
    sim.setScenario('corridor', 4242);
    for (let i = 0; i < 80; i++) sim.tick(0.05);
    sim.stop();
    return out;
  };
  const a = run();
  const b = run();
  assert.ok(a.length > 30);
  assert.deepEqual(a, b, 'the same seed must replay the same scan');
});

test('hybrid mode adopts the phone heading but keeps virtual acoustics', () => {
  const dets = [];
  const sim = new SimulationEngine({ onDetection: (d) => { if (d) dets.push(d); } });
  sim.setScenario('room', 99);
  sim.setMode('hybrid');
  sim.setExternalPose({ x: 0, y: 0, heading: 180, confidence: 0.8, method: 'dead-reckoning' });
  for (let i = 0; i < 40; i++) sim.tick(0.05);
  sim.stop();
  assert.ok(dets.length > 10);
  for (const d of dets) {
    assert.equal(d.source, 'hybrid');
    assert.ok(Math.abs(d.bearing_deg - 180) < 1, 'bearing must follow the real phone heading');
    assert.equal(d.phone.method, 'hybrid');
    assert.ok(d.phone.confidence <= 0.75, 'hybrid pose confidence is capped and honest');
  }
});

test('simulated ranges agree with the scenario geometry', () => {
  const rows = [];
  const sim = new SimulationEngine({
    onDetection: (d) => { if (d && d.debug) rows.push({ measured: d.range_m, truth: d.debug.truthRange }); },
  });
  sim.setScenario('room', 31337);
  for (let i = 0; i < 300; i++) sim.tick(0.05);
  sim.stop();
  assert.ok(rows.length > 100);
  // Ranging error against known geometry. The classifier crop lands on the
  // strongest scatterer, which for a wall includes the floor/wall dihedral, so
  // a modest positive bias is expected rather than a bug.
  const errs = rows.map((r) => Math.abs(r.measured - r.truth));
  errs.sort((a, b) => a - b);
  const median = errs[Math.floor(errs.length / 2)];
  assert.ok(median < 0.25, 'median simulated ranging error should be under 25 cm, got ' + median.toFixed(3));
});

// ---------------------------------------------------------------------------
// Map state
// ---------------------------------------------------------------------------
test('map state accumulates a mission and reports honest statistics', () => {
  const map = new MapState();
  map.startMission();
  const sim = new SimulationEngine({ onDetection: (d) => { if (d) map.applyDetection(d); } });
  sim.setScenario('apartment', 5150);
  for (let i = 0; i < 400; i++) sim.tick(0.05);
  sim.stop();

  const summary = map.endMission();
  const s = summary.stats;
  assert.ok(s.detections > 200, 'detections ' + s.detections);
  assert.ok(s.distanceScanned > 2, 'distance ' + s.distanceScanned);
  assert.ok(s.cloudPoints > 20);
  assert.ok(s.freeArea > 2, 'should have observed free space');
  assert.ok(s.avgClassConfidence > 0 && s.avgClassConfidence <= 1);
  assert.ok(summary.reconstruction.segments >= 2, 'segments ' + summary.reconstruction.segments);
  assert.ok(summary.reconstruction.totalWallLength > 3);
  assert.ok(summary.caveats.length >= 3, 'the summary must always carry its limits');
  const caveatText = summary.caveats.join(' ');
  assert.ok(/dead-reckoned|not surveyed/i.test(caveatText), 'position limit must be stated');
  assert.ok(/EXPERIMENTAL/.test(caveatText), 'the class label must be marked experimental');
  assert.ok(/openings are geometric/i.test(caveatText),
    'the summary must say openings are geometry, not classifier output');
  assert.equal(summary.stats.classCounts.OPENING, 0,
    'no live detection may carry the OPENING class');

  const snap = map.snapshot({ reconstruct: true });
  assert.ok(snap.grid.idx.length > 50);
  assert.ok(snap.cloud.length > 20);
  assert.ok(snap.trajectory.nodes.length > 5);
  assert.ok(snap.reconstruction.segments.length >= 2);
});

test('map state survives being reset mid-mission', () => {
  const map = new MapState();
  map.startMission();
  for (let i = 0; i < 20; i++) {
    map.applyDetection(normalizeDetection({
      range_m: 1.5, bearing_deg: i * 5, confidence: 0.7,
      classProbs: [0.8, 0.1, 0.1], phone: { x: 0, y: 0, heading: i * 5 },
    }));
  }
  assert.ok(map.stats.detections === 20);
  map.reset();
  assert.equal(map.stats.detections, 0);
  assert.equal(map.cloud.points.length, 0);
  assert.equal(map.grid.coverage(), 0);
  // And it must still work afterwards.
  map.applyDetection(normalizeDetection({
    range_m: 2, bearing_deg: 0, confidence: 0.7,
    classProbs: [0.8, 0.1, 0.1], phone: { x: 0, y: 0, heading: 0 },
  }));
  assert.equal(map.stats.detections, 1);
});

// ---------------------------------------------------------------------------
// Guidance
// ---------------------------------------------------------------------------
test('relative direction words match the geometry', () => {
  assert.equal(relativeDirection(0, 0).word, 'ahead');
  assert.equal(relativeDirection(20, 0).word, 'ahead');
  assert.equal(relativeDirection(60, 0).word, 'on your right');
  assert.equal(relativeDirection(300, 0).word, 'on your left');
  assert.equal(relativeDirection(180, 0).word, 'behind you');
  // Heading-relative, not absolute.
  assert.equal(relativeDirection(90, 90).word, 'ahead');
  assert.equal(relativeDirection(90, 30).word, 'on your right');
});

test('guidance debounces and never spams', () => {
  const g = new GuidancePolicy();
  const det = (range, t) => normalizeDetection({
    t, range_m: range, vel_mps: 0.1, confidence: 0.8, bearing_deg: 0,
    classProbs: [0.8, 0.1, 0.1], phone: { x: 0, y: 0, heading: 0 },
  });
  const t0 = 1000000;
  const first = g.evaluate(det(1.0, t0));
  assert.ok(first, 'a close obstacle should produce a cue');
  // 20 Hz of identical detections must not produce 20 cues.
  let extra = 0;
  for (let i = 1; i < 40; i++) if (g.evaluate(det(1.0, t0 + i * 50))) extra++;
  assert.ok(extra <= 1, 'two seconds of the same obstacle gave ' + extra + ' extra cues');
});

test('a critical cue pre-empts a lower-priority one', () => {
  const g = new GuidancePolicy();
  const t0 = 2000000;
  const mid = g.evaluate(normalizeDetection({
    t: t0, range_m: 2.0, confidence: 0.8, bearing_deg: 0,
    classProbs: [0.8, 0.1, 0.1], phone: { x: 0, y: 0, heading: 0 },
  }));
  assert.ok(mid);
  const critical = g.evaluate(normalizeDetection({
    t: t0 + 700, range_m: 0.4, confidence: 0.8, bearing_deg: 0,
    classProbs: [0.8, 0.1, 0.1], phone: { x: 0, y: 0, heading: 0 },
  }));
  assert.ok(critical, 'imminent contact must interrupt');
  assert.equal(critical.level, 'critical');
  assert.ok(/stop/i.test(critical.text));
});

test('an acoustic OPENING call is never spoken as a way through', () => {
  // Through a doorway, CFAR locks onto the far wall of the next room, so an
  // acoustic OPENING is a distant wall wearing the wrong label. Even a
  // confident, stable one must not produce speech that invites the operator
  // to walk forward.
  const g = new GuidancePolicy();
  const d = normalizeDetection({
    t: 4000000, range_m: 2.7, confidence: 0.7, bearing_deg: 0,
    obstacleClass: 'OPENING', classConfidence: 0.95, classProbs: [0.02, 0.03, 0.95],
    fusedClass: 'OPENING', fusedConfidence: 0.95, phone: { x: 0, y: 0, heading: 0 },
  });
  d.fusedStable = true;
  const cue = g.evaluate(d);
  if (cue) {
    assert.ok(!/opening/i.test(cue.text), 'no opening cue may be spoken: ' + cue.text);
    assert.ok(!/clear|through|doorway/i.test(cue.text), 'must not invite forward motion: ' + cue.text);
  }
});

test('the classifier can soften a cue but never cause one', () => {
  // Every cue is triggered by range, velocity or TTC — all measured. The
  // class only changes the noun, so a weak classifier cannot invent speech.
  const g = new GuidancePolicy();
  const far = g.evaluate(normalizeDetection({
    t: 3000000, range_m: 3.4, confidence: 0.9, bearing_deg: 300,
    obstacleClass: 'SOFT', classConfidence: 0.99, classProbs: [0.005, 0.99, 0.005],
    fusedClass: 'SOFT', fusedConfidence: 0.99, phone: { x: 0, y: 0, heading: 0 },
  }));
  assert.equal(far, null, 'a confident class at 3.4 m must not create a cue on its own');

  const g2 = new GuidancePolicy();
  const near = g2.evaluate(Object.assign(normalizeDetection({
    t: 3000000, range_m: 1.1, confidence: 0.9, bearing_deg: 300,
    obstacleClass: 'SOFT', classConfidence: 0.9, classProbs: [0.05, 0.9, 0.05],
    fusedClass: 'SOFT', fusedConfidence: 0.9, phone: { x: 0, y: 0, heading: 0 },
  }), { fusedStable: true }));
  assert.ok(near, 'range alone must still produce the cue');
  assert.ok(/soft obstacle/i.test(near.text), 'the class names the obstacle: ' + near.text);
  assert.ok(/left/i.test(near.text), 'direction should be spoken: ' + near.text);
});

test('no live path can emit obstacleClass OPENING', () => {
  // The two-class rule has to hold at every re-entry point: the pipeline,
  // normalizeDetection's fallback derivation, and the temporal fuser.
  const openingHeavy = [0.02, 0.03, 0.95];
  assert.equal(surfaceFromProbs(openingHeavy).className, 'SOFT',
    'the discarded head must not win');

  const derived = normalizeDetection({
    range_m: 2, confidence: 0.6, classProbs: openingHeavy,
    phone: { x: 0, y: 0, heading: 0 },
  });
  assert.notEqual(derived.obstacleClass, 'OPENING',
    'normalizeDetection must not derive OPENING from probabilities');

  const fuser = new ClassFuser();
  let fused = null;
  for (let i = 0; i < 6; i++) {
    fused = fuser.fuse(normalizeDetection({
      t: 5000000 + i * 50, range_m: 2, confidence: 0.9, bearing_deg: 0,
      classProbs: openingHeavy, phone: { x: 0, y: 0, heading: 0 },
    }));
  }
  assert.notEqual(fused.className, 'OPENING',
    'the fuser must not conjure a class the pipeline refused to emit');
});

test('a coin-flip between WALL and SOFT is reported as no call', () => {
  const tied = surfaceFromProbs([0.45, 0.45, 0.10]);
  assert.equal(tied.className, null, 'an unseparated pair is not a decision');
  assert.equal(tied.confidence, 0);
  const clear = surfaceFromProbs([0.7, 0.2, 0.1]);
  assert.equal(clear.className, 'WALL');
  assert.ok(clear.confidence > 0.7,
    'confidence renormalises over the retained heads, not the discarded one');
});

test('sustained silence is reported as a clear path', () => {
  const g = new GuidancePolicy();
  assert.equal(g.evaluate(null, { quietMs: 500 }), null, 'a brief gap says nothing');
  const cue = g.evaluate(null, { quietMs: 4000 });
  assert.ok(cue);
  assert.ok(/clear/i.test(cue.text));
});

// ---------------------------------------------------------------------------
// Record / replay
// ---------------------------------------------------------------------------
test('a recording replays the frames it captured', async (t) => {
  const rec = new Recorder();
  // Clean up after ourselves: a stray test recording left on disk would change
  // which scan "REPLAY LAST" picks during an actual demo.
  let savedId = null;
  t.after(() => {
    if (!savedId) return;
    try { require('node:fs').unlinkSync(require('node:path').join(__dirname, '..', 'recordings', savedId + '.json')); } catch (e) { /* already gone */ }
  });
  rec.start({ mode: 'simulation', scenario: 'room', note: 'unit test' });
  const made = [];
  for (let i = 0; i < 30; i++) {
    const d = normalizeDetection({
      t: Date.now(), range_m: 1 + i * 0.05, bearing_deg: i * 3, confidence: 0.8,
      classProbs: [0.7, 0.2, 0.1], phone: { x: 0, y: i * 0.02, heading: i * 3 },
    });
    made.push(d);
    rec.push('detection', d);
  }
  const saved = rec.stop();
  savedId = saved.id;
  assert.ok(saved.id);
  assert.equal(saved.frameCount, 30);

  const loaded = rec.load(saved.id);
  assert.ok(loaded, 'recording should persist to disk');
  assert.equal(loaded.frames.length, 30);

  const seen = [];
  await new Promise((resolve) => {
    const player = new ReplayPlayer(loaded, {
      speed: 40,
      onFrame: (f) => { if (f.k === 'detection') seen.push(f.p); },
      onDone: resolve,
    });
    player.start();
  });
  assert.equal(seen.length, 30, 'replay must emit every recorded frame');
  assert.ok(Math.abs(seen[0].range_m - made[0].range_m) < 1e-9, 'frames must survive the round trip');

  // Replaying through the map must rebuild real state.
  const map = new MapState();
  map.startMission();
  for (const d of seen) map.applyDetection(Object.assign({}, d, { source: 'replay' }));
  assert.equal(map.stats.detections, 30);
  assert.ok(map.cloud.points.length > 0);
});

test('the recorder survives unreadable or missing recordings', () => {
  const rec = new Recorder();
  assert.equal(rec.load('does-not-exist'), null);
  assert.equal(rec.load('../../etc/passwd'), null, 'path traversal must not resolve');
  assert.ok(Array.isArray(rec.list()));
});

// ---------------------------------------------------------------------------
// AWS adapter
// ---------------------------------------------------------------------------
test('AWS adapter falls back to local voice with no credentials', async () => {
  const aws = new AwsAdapter({});
  const st = aws.status();
  assert.equal(st.credentialsDetected, false);
  assert.equal(st.voice.provider, 'local');
  assert.equal(st.voice.label, 'LOCAL FALLBACK');
  assert.ok(st.voice.reason, 'the reason must be stated, not hidden');
  assert.equal(st.bedrock.enabled, false);

  const spoken = await aws.speak('Obstacle ahead.');
  assert.equal(spoken.provider, 'local', 'speech must still be delivered locally');

  const sum = await aws.summarizeScan({ stats: {}, reconstruction: {} });
  assert.equal(sum.available, false);
  assert.ok(sum.reason);
});

test('AWS adapter reports Polly ready when a region and keys are present', () => {
  // Placeholder values only: this asserts capability detection, and makes no
  // network call.
  const aws = new AwsAdapter({
    AWS_REGION: 'eu-west-1',
    AWS_ACCESS_KEY_ID: 'PLACEHOLDER',
    AWS_SECRET_ACCESS_KEY: 'PLACEHOLDER',
  });
  const st = aws.status();
  assert.equal(st.credentialsDetected, true);
  assert.equal(st.credentialSource, 'static-env-keys');
  assert.equal(st.voice.provider, 'polly');
  assert.equal(st.voice.label, 'AWS POLLY');
  // Bedrock stays off until explicitly enabled with a model id.
  assert.equal(st.bedrock.enabled, false);
  assert.ok(/ENABLE_BEDROCK|BEDROCK_MODEL_ID/.test(st.bedrock.reason));
});

test('an EC2 instance role is recognised only when declared', () => {
  // A role leaves no env var behind, so without the flag Polly must stay off
  // rather than guess; with it, the SDK's default chain does the lookup.
  const base = { AWS_REGION: 'us-east-1' };
  assert.equal(new AwsAdapter(base).status().credentialsDetected, false);
  const role = new AwsAdapter(Object.assign({ AWS_USE_INSTANCE_ROLE: 'true' }, base)).status();
  assert.equal(role.credentialsDetected, true);
  assert.equal(role.credentialSource, 'profile-or-role');
});

test('S3 archive is off by default, says why, and archives with a fake client', async () => {
  const off = new AwsAdapter({});
  assert.equal(off.status().s3.enabled, false);
  assert.ok(/S3_BUCKET/.test(off.status().s3.reason));
  assert.equal((await off.archiveRecording('scan-1', '{}')).ok, false);

  const env = { AWS_REGION: 'us-east-1', AWS_USE_INSTANCE_ROLE: 'true', S3_BUCKET: 'demo-bucket', S3_PREFIX: 'scans' };
  const aws = new AwsAdapter(env);
  assert.equal(aws.status().s3.enabled, true);
  assert.equal(aws.status().s3.prefix, 'scans/', 'prefix is normalised to end in a slash');

  const sent = [];
  aws.archive.client = { send: async (cmd) => { sent.push(cmd.input); } };
  const res = await aws.archiveRecording('scan-2026-09-19', '{"frames":[]}');
  assert.equal(res.ok, true);
  assert.equal(sent.length, 1);
  assert.equal(sent[0].Bucket, 'demo-bucket');
  assert.equal(sent[0].Key, 'scans/scan-2026-09-19.json');
  assert.equal(sent[0].ContentType, 'application/json');

  // Path tricks in an id cannot escape the prefix.
  await aws.archiveRecording('../../etc/passwd', '{}');
  assert.ok(sent[1].Key.startsWith('scans/') && !sent[1].Key.includes('/../') && !sent[1].Key.slice(6).includes('/'));

  // A failing upload is reported and counted, never thrown.
  aws.archive.client = { send: async () => { const e = new Error('denied'); e.name = 'AccessDenied'; throw e; } };
  const bad = await aws.archiveRecording('scan-3', '{}');
  assert.equal(bad.ok, false);
  assert.ok(/AccessDenied/.test(bad.error));
  assert.equal(aws.stats.s3Uploads, 2);
  assert.equal(aws.stats.s3Failures, 1);
});

test('a finished recording is handed to the archive hook without blocking the save', async () => {
  const fs = require('node:fs');
  const path = require('node:path');
  const { RECORDINGS_DIR } = require('../server/recorder');
  const calls = [];
  const rec = new Recorder({ onSaved: (id, json) => { calls.push({ id, json }); throw new Error('hook blew up'); } });
  rec.start({ mode: 'simulation', note: 'archive-hook-test' });
  rec.push('pose', { x: 0, y: 0 });
  const saved = rec.stop();
  try {
    assert.ok(!saved.persistError, 'a throwing hook must not turn a saved scan into an error');
    assert.equal(calls.length, 1);
    assert.equal(calls[0].id, saved.id);
    assert.equal(JSON.parse(calls[0].json).frames.length, 1);
    assert.ok(fs.existsSync(path.join(RECORDINGS_DIR, saved.id + '.json')), 'local copy still written first');
  } finally {
    try { fs.unlinkSync(path.join(RECORDINGS_DIR, saved.id + '.json')); } catch (e) { /* already gone */ }
  }
});

test('Bedrock prompt carries the measured numbers and the model limits', () => {
  const { BedrockSummarizer } = require('../server/aws/bedrock');
  const b = new BedrockSummarizer({ region: 'eu-west-1', modelId: 'anthropic.test', enabled: false });
  const prompt = b.buildPrompt({
    stats: {
      distanceScanned: 6.2, detections: 180, cfarRate: 0.97, minRange: 0.7, maxRange: 3.4,
      classCounts: { WALL: 140, SOFT: 20, OPENING: 20 }, avgClassConfidence: 0.78,
    },
    reconstruction: {
      segments: 5, totalWallLength: 14.2, corners: 2, corridors: 1, confidence: 0.81, openings: 2,
      openingDetail: [{ x: 1.2, y: 2.0, evidence: 'geometric-gap', confidence: 0.5 }],
    },
  });
  assert.ok(prompt.includes('6.2 m'), 'measured distance must be in the prompt');
  assert.ok(prompt.includes('180'), 'detection count must be in the prompt');
  assert.ok(/EXPERIMENTAL/.test(prompt) && /39%/.test(prompt) && /33% chance/.test(prompt),
    'the model must be told the classifier is weak, and by how much');
  assert.ok(/does not detect openings/i.test(prompt),
    'the model must be told openings are geometric, not classified');
  assert.ok(/absence of a return/i.test(prompt),
    'the model must be told why an acoustic opening call is meaningless');
  assert.ok(/not navigation-certified|no safety guarantees|Do not make safety guarantees/i.test(prompt));
  assert.ok(/hedged language/i.test(prompt), 'the prompt must require hedging');
});

test('Bedrock speaks the Nova schema and reads the Nova response', () => {
  const { BedrockSummarizer } = require('../server/aws/bedrock');
  const nova = new BedrockSummarizer({ region: 'us-east-1', modelId: 'amazon.nova-lite-v1:0', enabled: false });
  assert.ok(nova.isNovaModel() && !nova.isAnthropicModel());
  assert.ok(new BedrockSummarizer({ modelId: 'us.amazon.nova-lite-v1:0' }).isNovaModel(),
    'an inference-profile id must be recognised too');
  assert.equal(nova.extractText({ output: { message: { role: 'assistant', content: [{ text: ' A room. ' }] } } }), 'A room.');
  // Titan and Claude response shapes must keep working.
  assert.equal(nova.extractText({ results: [{ outputText: 'titan' }] }), 'titan');
  assert.equal(nova.extractText({ content: [{ type: 'text', text: 'claude' }] }), 'claude');
});

test('a recording listing always reports an id that load() can resolve', () => {
  // The demo's fallback depends on this: a hand-renamed recording must not
  // list fine and then fail to open.
  const rec = new Recorder();
  for (const info of rec.list()) {
    const loaded = rec.load(info.id);
    assert.ok(loaded, 'list() reported id "' + info.id + '" but load() returned null');
    assert.ok(Array.isArray(loaded.frames), 'recording ' + info.id + ' has no frames');
  }
});

test('the shipped reference scan replays into a full map', async (t) => {
  // Ships with the repo so REPLAY LAST works on a fresh clone with no phone.
  const rec = new Recorder().load('reference-apartment-scan');
  if (!rec) {
    t.diagnostic('reference-apartment-scan.json is absent; skipping');
    return;
  }
  const map = new MapState();
  map.startMission();
  let n = 0;
  await new Promise((resolve) => {
    const player = new ReplayPlayer(rec, {
      speed: 400,
      onFrame: (f) => {
        if (f.k === 'detection') { map.applyDetection(Object.assign({}, f.p, { source: 'replay' })); n++; }
      },
      onDone: resolve,
    });
    player.start();
  });
  const recon = map.buildReconstruction(true);
  assert.ok(n > 300, 'should replay the whole scan, got ' + n + ' detections');
  assert.ok(recon.segments.length >= 3, 'should rebuild surfaces, got ' + recon.segments.length);
  assert.ok(map.publicStats().cloudPoints > 40);
});
