# PITCHBLACK — project memory

## What this is
Semantic acoustic navigation PWA. Phone speaker emits inaudible 17.5–22kHz
FMCW chirps, mic captures echoes, on-device DSP computes obstacle range +
closing velocity 20x/sec, converts to audio/haptic feedback for zero-visibility
navigation. Stretch: tiny on-device classifier (WALL/SOFT/OPENING).

## Non-negotiable architecture rule
Sensor and UI are split at a strict message boundary. Every downstream
consumer (audio, haptics, phone UI, WS uplink) reads ONLY this type — never
touch raw audio buffers outside the sensor module:

\`\`\`ts
type Detection = {
  t: number;
  range_m: number;
  vel_mps: number;
  ttc_s: number;
  confidence: number;
  bearing_deg?: number;
  profile?: Float32Array;
  obstacleClass?: 'WALL' | 'SOFT' | 'OPENING';
  classConfidence?: number;
};
\`\`\`
Reason: if Chrome audio fails, we swap ONLY the sensor module for a native
Android bridge emitting the same Detection over local WebSocket. Zero UI rework.

`obstacleClass` is WALL or SOFT (or null) on every live detection. 'OPENING'
stays in the wire vocabulary only so pre-2026-09-18 recordings still replay —
nothing derives it. An opening is the absence of a return, not a texture: CFAR
takes the strongest peak past the gate, so aimed through a doorway the detector
locks onto the far wall of the next room. Openings come from `findGapOpenings()`
in reconstruct.mjs instead. The two-class rule lives in
`public/shared/surfaceclass.mjs` and is enforced at all three re-entry points
(pipeline, `normalizeDetection()`, `ClassFuser`) — adding a fourth consumer that
takes a raw three-way argmax reintroduces the bug.

## Signal parameters — use these, do not re-derive
- Linear FM chirp, Hann-windowed, 17,500→22,000 Hz, bandwidth 4500Hz
- Chirp duration 15ms, pulse interval 50ms (20Hz), sample rate 48000Hz (NOT 44.1k)
- Mono TX; stereo RX if channelCount:2 granted (unlocks TDOA bearing)
- TX amplitude 0.6 FS, ramped
- getUserMedia constraints MUST be nested inside the `audio` object:
  echoCancellation:false, noiseSuppression:false, autoGainControl:false —
  Chrome silently ignores these if not nested correctly.

## Processing chain (in order, don't skip steps)
1. Matched filter: FFT cross-correlate RX window vs reference chirp
2. Direct-path gate: zero out ±2ms around calibrated system delay
3. Clutter subtraction: subtract averaged 40-pulse static baseline from live profile
4. CFAR detection: cell-averaging CFAR + noise-floor Kalman, strongest peak beyond gate
5. Alpha-beta tracker: smooth range, differentiate velocity. Reject peak jumps >0.8m
   between consecutive pulses unless persists 3 frames.
6. TTC = range / max(vel, 0.05), clamped [0,10]

## File layout (as built — SentryShield)
- `server/` — hub.js (the message boundary), simulation.js (digital twin),
  mapstate.js, recorder.js, guidance.js, static.js, index.js, aws/
- `public/shared/*.mjs` — protocol, spatial engine, reconstruction, echo
  synthesis, ws client. **Imported verbatim by both the browser and Node**
  (Node 22 `require(esm)`), so client and server cannot drift.
- `public/phone/` — sensor.mjs (the only audio-aware module), dsp/pipeline.mjs,
  dsp/fft.mjs, pose.mjs, audio-guidance.mjs, calibration.mjs, phone.mjs
- `public/map/` — map.mjs, renderer.mjs (the command center)
- `public/diagnostics/` — capability checks + 26 in-browser self-tests
- `src/classifier/` — EchoNet weights + the generator (unchanged)
- `tests/` — 69 tests: dsp, spatial/protocol/sim, e2e over a real WebSocket
- `scripts/` — browser-check (real Chrome), verify-classifier, eval-reconstruction,
  window-stats, check-syntax
- Docs: README.md, DEMO.md, ARCHITECTURE.md, STATUS.md, PROTOCOL.md

Routes: `/map` `/phone` `/diagnostics` `/legacy/gate-test` (original prototype).
No build step. `npm start` prints URLs + a QR code.

## Calibrated constants — do not re-tune without re-running the measurement
- CFAR: guard 28, train 32, Pfa 1e-4, **margin x10**. The margin is empirical:
  clutter subtraction rectifies at zero, so the rectified profile is
  heavier-tailed than the textbook exponential assumption. Pinned by
  `tests/dsp.test.js` ("CFAR holds its false-alarm rate"): 0% empty-room false
  alarms, 97% detection of a 2.5e-4 FS echo.
- The CFAR noise estimate MUST be floored by a sigma-clipped window average.
  Without it, runs of rectified zeros collapse the local estimate and the
  measured false-alarm rate is 100%.
- Tracker: alpha 0.35, beta 0.074 (critically damped), maxJump 0.8 m,
  persist 3 frames, maxMisses 5. A non-CFAR-passing peak may maintain a track
  but must never acquire one.
- Simulation noise: matched-filter output noise must be **band-limited**, not
  white — white noise gives a jagged envelope no real receiver produces, and the
  classifier keys on roughness.

## Pose — the two rules that make the map stop wobbling
Both were the *same* visible bug: a room scan that snaked tens of metres across
a room nobody walked, with walls that never lined up. Pinned by `tests/pose.test.js`.

- **Heading is tilt-compensated, never raw `alpha`.** `360 - alpha` is only a
  bearing when the phone lies flat. Aimed at a wall, beta approaches 90 deg,
  where alpha and gamma are the *same* rotation — the browser can report any
  (alpha, gamma) pair with a constant sum for one physical orientation, so alpha
  alone swings by tens of degrees while the phone does not move. `orientationAxes()`
  builds the full W3C rotation R = Rz(a)·Rx(b)·Ry(g) and returns two boresights:
  the screen's top edge (flat/compass hold) and out the back of the phone (aimed
  hold). Whichever is more horizontal wins, with 0.12 hysteresis. They are
  orthogonal, so the winner always has ≥0.7 of horizontal projection — the
  bearing is never ill-conditioned.
- **A step needs a gait, not a bump.** Sweeping a phone by hand makes
  acceleration peaks identical to footfalls under a threshold test. The gyro is
  the discriminator: a sweep turns at 60–200 deg/s, a walker's hand does not.
  Peaks above `maxStepRotDps` 45 are rejected outright; the rest must land in a
  run of `gaitConfirm` 3 evenly-spaced intervals (280–1100 ms, each within
  0.6–1.7x of the last) before any translation is committed, and the three
  confirming peaks are credited together at that moment. **Default pose mode is
  `rotation` (position locked at 0,0)** — walking is opt-in, because scanning
  from one spot is the common case and the one phantom steps destroy.

Sweep rate is also folded into the measurement rather than ignored: `phone.mjs`
widens `beamwidth_deg` by the arc crossed during the pulse and scales confidence
by `1 - rate/250`, so a fast sweep draws a longer, weaker boundary bar and past
~150 deg/s falls under the boundary layer's 0.65 gate entirely.

## Classifier reality check — do not re-litigate
EchoNet is EXPERIMENTAL and weak. 69.5% on synthetic validation, but **39% on
the real recordings held out by whole session, against a 33% chance floor**; a
synthetic-only model scores *below* chance on a two-class split. This is the
dataset's ceiling, not a hyperparameter problem — a 300-tree random forest gets
39.4% over every feature set tried. `scripts/diagnose-dataset.py` reports the
three causes: the 5,250 pulses are 30 contiguous bursts of ~175 near-identical
pulses (real sample size 30); `range_m` alone predicts the class at 37.8%
because each class was recorded at its own standoff; and OPENING-labelled pulses
carry the *highest* target strength of any class, which is physically backwards.

Do not tune hyperparameters to chase this number. The fixes that would actually
move it are, in order: sub-band spectral tilt (two matched filters across the
chirp band — soft absorbers roll off high frequencies, and this survives peak
normalisation); range-compensated amplitude as a model input; a window that
reaches past the ±11 cm the current 64-sample peak-centred one covers, into the
reverberant tail; and re-collection with range decorrelated from class (many
short bursts across many scenes, not few scenes × many pulses).

`npm run train-real` splits by session, skips duplicate sessions, and refuses to
export weights that regress on holdout. Never evaluate by shuffling pulses —
neighbouring pulses in a burst are near-copies and a random split reads ~62%
where the honest number is ~37%.

## Build status
All of the above is built and tested; see STATUS.md for measured numbers and
the list of things that still require the physical OnePlus 12R. The one thing
never verified on hardware is live acoustic performance — simulation and hybrid
modes exist precisely so the demo does not depend on it.

## Out of scope — do not build
Multi-floor mapping, SLAM, victim detection, BLE/mesh, iOS, accounts,
any certification claims. This is a research prototype.

## Definition of "core loop done"
Blindfolded person stops within 30–90cm of a wall using audio only, first try.
Range trace over 3m approach is monotonic, <15cm RMS jitter.

The trace half is met in test: `tests/dsp.test.js` measures <15 cm RMSE and a
near-monotonic descent over a 3 m approach at 0.5 m/s, on synthetic audio
through the real pipeline. The blindfold half needs the phone and a person.