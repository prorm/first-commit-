/**
 * Pose estimator tests.
 *
 * Two regressions live here, both of which produced the same visible symptom —
 * a room map that snaked across tens of metres and whose walls never lined up:
 *
 *   1. Heading read straight off `alpha`, which is degenerate the moment the
 *      phone is aimed rather than laid flat.
 *   2. Steps counted from any acceleration peak, so sweeping the phone by hand
 *      dead-reckoned a walk the operator never took.
 *
 * pose.mjs touches no DOM at module scope, so it runs here unchanged.
 */
const test = require('node:test');
const assert = require('node:assert');

const { PoseEstimator, orientationAxes, axisHeading } = require('../public/phone/pose.mjs');

/** Fire one orientation sample straight at the estimator. */
function orient(p, alpha, beta, gamma, absolute = true) {
  p.onOrientation({ type: 'deviceorientationabsolute', absolute, alpha, beta, gamma });
}

/**
 * One acceleration peak, shaped like a footfall: a sample well above the slow
 * average to arm the detector and one at the average to release it.
 */
function peak(p, rotDps) {
  const rr = { alpha: rotDps, beta: 0, gamma: 0 };
  p.onMotion({ accelerationIncludingGravity: { x: 0, y: 0, z: p.accSlow + 3 }, rotationRate: rr });
  p.onMotion({ accelerationIncludingGravity: { x: 0, y: 0, z: p.accSlow }, rotationRate: rr });
}

/** Run `fn` with Date.now() under our control, in milliseconds. */
function withClock(fn) {
  const real = Date.now;
  const clock = { t: 1700000000000 };
  Date.now = () => clock.t;
  try { fn(clock); } finally { Date.now = real; }
}

// ---------------------------------------------------------------------------
// Heading
// ---------------------------------------------------------------------------
test('a flat phone still reads the compass the way it always did', () => {
  // Held level with the top edge forward, the tilt-compensated bearing must
  // reduce to the old 360 - alpha, or every previously-correct scan breaks.
  for (const alpha of [0, 30, 90, 217, 359]) {
    const axes = orientationAxes(alpha, 0, 0, 0);
    assert.ok(Math.abs(axes.flat.u) < 1e-9, 'a level phone has no vertical component on its top edge');
    const h = axisHeading(axes.flat);
    const expected = (360 - alpha) % 360;
    assert.ok(Math.abs(((h - expected + 540) % 360) - 180) < 1e-6,
      'flat heading ' + h + ' should be ' + expected);
  }
});

test('an aimed phone reads the direction it points, not alpha', () => {
  // Upright, back of the phone facing north: beta = 90.
  const axes = orientationAxes(0, 90, 0, 0);
  assert.ok(Math.abs(axes.aim.u) < 1e-9, 'a horizontally-aimed phone has no vertical boresight');
  assert.ok(Math.abs(axisHeading(axes.aim)) < 1e-6, 'aim heading is due north');
  assert.ok(Math.abs(Math.abs(axes.flat.u) - 1) < 1e-9, 'its top edge points at the ceiling');
});

test('gimbal lock does not move a phone that has not moved', () => {
  // At beta = 90 the alpha and gamma rotations are the same rotation, so the
  // browser may report any (alpha, gamma) pair with a constant sum for one
  // physical orientation. Reading alpha alone turns that ambiguity into tens of
  // degrees of phantom bearing swing — this is the wobble.
  const ref = axisHeading(orientationAxes(0, 90, 0, 0).aim);
  for (const a of [10, 40, 75, 120]) {
    const h = axisHeading(orientationAxes(a, 90, -a, 0).aim);
    assert.ok(Math.abs(((h - ref + 540) % 360) - 180) < 1e-6,
      'alpha=' + a + ' gamma=' + -a + ' is the same orientation, got ' + h);
    // What the old code would have read, for the record.
    assert.ok(Math.abs((360 - a) % 360 - ref) > 5, 'alpha alone would have swung ' + a + ' deg');
  }
});

test('the boresight is always the axis with real bearing information', () => {
  // The two candidates are orthogonal, so picking the more horizontal of them
  // guarantees a well-conditioned projection at every attitude.
  const p = new PoseEstimator();
  for (let beta = -180; beta <= 180; beta += 7) {
    for (let gamma = -90; gamma <= 90; gamma += 9) {
      const axis = p._chooseAxis(orientationAxes(23, beta, gamma, 0));
      assert.ok(Math.hypot(axis.e, axis.n) > 0.55,
        'beta ' + beta + ' gamma ' + gamma + ' left a near-vertical boresight');
    }
  }
});

test('a straight-down boresight yields no bearing rather than a made-up one', () => {
  assert.equal(axisHeading({ e: 0, n: 0, u: -1 }), null);
  assert.equal(axisHeading({ e: 0.05, n: 0.05, u: -0.99 }), null);
});

test('heading holds steady through sensor shimmer and still follows a real turn', () => {
  const p = new PoseEstimator();
  orient(p, 0, 0, 0);
  const start = p.heading;
  // Sub-degree jitter around a fixed attitude must not move the reading.
  for (let i = 0; i < 40; i++) orient(p, (i % 2 ? 0.2 : -0.2), 0, 0);
  assert.ok(Math.abs(((p.heading - start + 540) % 360) - 180) < 0.5, 'shimmer moved the heading');

  // A deliberate 90 deg turn must arrive, and quickly.
  for (let i = 0; i < 60; i++) orient(p, 270, 0, 0);
  assert.ok(Math.abs(((p.heading - 90 + 540) % 360) - 180) < 1.5,
    'turn did not converge, heading ' + p.heading);
});

// ---------------------------------------------------------------------------
// Steps
// ---------------------------------------------------------------------------
test('scanning a room from a chair does not walk across it', () => {
  // The reported bug, in one test: sit still, sweep the phone, and the map
  // used to dead-reckon ~20 m of travel. Stationary is now the default mode.
  const p = new PoseEstimator();
  assert.equal(p.poseMode, 'rotation', 'stationary is the default');
  withClock((clock) => {
    for (let i = 0; i < 40; i++) {
      clock.t += 320;
      orient(p, (i * 9) % 360, 60, 0);
      peak(p, 110);
    }
  });
  assert.equal(p.x, 0);
  assert.equal(p.y, 0);
  assert.equal(p.distance, 0);
});

test('a hand sweep is not a gait even with walking switched on', () => {
  const p = new PoseEstimator();
  p.setPoseMode('walk');
  withClock((clock) => {
    for (let i = 0; i < 40; i++) {
      clock.t += 420;
      peak(p, 110);          // gyro says the phone is being swept, not carried
    }
  });
  assert.ok(p.stepCandidates > 10, 'the peaks were seen');
  assert.equal(p.steps, 0, 'but none of them moved us');
  assert.equal(p.distance, 0);
});

test('an even gait with a steady phone does move', () => {
  const p = new PoseEstimator();
  p.setPoseMode('walk');
  orient(p, 0, 0, 0);        // facing north
  withClock((clock) => {
    for (let i = 0; i < 6; i++) {
      clock.t += 500;
      peak(p, 8);            // phone held steady in the hand
    }
  });
  assert.equal(p.steps, 5, 'four confirming peaks, three of them credited at once');
  assert.ok(p.y > 3 && p.y < 4.5, 'travelled north, got ' + p.y);
  assert.ok(Math.abs(p.x) < 1e-9, 'no sideways drift');
});

test('irregularly spaced peaks never establish a gait', () => {
  const p = new PoseEstimator();
  p.setPoseMode('walk');
  withClock((clock) => {
    const gaps = [300, 900, 320, 1000, 290, 950, 310, 1050];
    for (const g of gaps.concat(gaps, gaps)) {
      clock.t += g;
      peak(p, 5);
    }
  });
  assert.equal(p.steps, 0, 'a limp that alternates 300 and 950 ms is not a walk');
});

// ---------------------------------------------------------------------------
// Confidence
// ---------------------------------------------------------------------------
test('a locked scan is confident about position and honest about sweep rate', () => {
  const p = new PoseEstimator();
  orient(p, 0, 0, 0);
  const still = p.confidence();
  assert.ok(still > 0.8, 'a position the operator asserted is not a guess, got ' + still);

  p.headingRate = 200;
  assert.ok(p.confidence() < still * 0.75, 'a bearing read mid-sweep is worth less');
});

test('dead-reckoned confidence still decays with distance walked', () => {
  const p = new PoseEstimator({ poseMode: 'walk' });
  orient(p, 0, 0, 0);
  const near = p.confidence();
  p.manualWalk(20);
  assert.ok(p.confidence() < near * 0.5, 'twenty metres of step counting is not free');
});
