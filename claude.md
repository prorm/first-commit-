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