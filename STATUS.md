# SentryShield — Status

Last verified: 2026-09-17. Every figure below was produced by a command in this
repo, named next to it. Nothing here is estimated.

```
npm test                            69 tests, 69 pass
node scripts/browser-check.mjs      all pages, real Chrome, 0 console errors
node scripts/verify-classifier.mjs  digital twin vs. the real classifier
node scripts/eval-reconstruction.js reconstruction error vs. ground truth
node scripts/window-stats.mjs       simulated vs. real training windows
```

---

## COMPLETED

### DSP pipeline — verified on synthetic audio through the real code
`public/phone/dsp/pipeline.mjs` · `tests/dsp.test.js` (12 tests)

- Matched filter, analytic envelope, range alignment, ±2 ms direct-path gate,
  40-pulse clutter subtraction with rectification, smallest-of CA-CFAR with a
  Kalman-smoothed noise floor, alpha-beta tracker with jump rejection,
  64-sample classifier windowing.
- **0.4 cm mean error** on a static target at 0.6–3.2 m.
- **<15 cm RMSE, near-monotonic** over a 3 m approach at 0.5 m/s — this meets
  the project's own "core loop done" bar for trace quality.
- **0% false alarms** in an empty room, while detecting a 2.5e-4 FS echo 97% of
  the time.
- Chirp verified to be 15 ms of Hann-windowed LFM sweeping 17.5→22 kHz at
  0.6 FS by measuring the generated waveform.
- Never throws on silence, clipped DC, or full-scale noise.

### Protocol and message boundary
`public/shared/protocol.mjs` · `PROTOCOL.md` · 9 tests

- Schema v1, one definition shared verbatim by server and both clients.
- Malformed frames are repaired or rejected with an `error`; the socket survives.
  Verified in Node **and** in-browser.
- Version mismatch, unknown type, unknown role/mode, and unusable range are all
  rejected.

### Spatial engine
`public/shared/spatial.mjs` · 14 tests

- Log-odds occupancy grid, 0.1 m cells, 5-ray beam cone, saturating.
  **Unknown / free / occupied stay three distinct states.**
- `OPENING` detections carve free space instead of painting a wall.
- Point cloud consolidation on a spatial hash, including across bucket seams,
  with confidence-weighted class voting and a hard cap.
- Temporal class fusion; reports instability rather than averaging it away.
- Sparse grid wire format, round-trip error <0.1 log-odds.

### Surface reconstruction
`public/shared/reconstruct.mjs` · 10 tests

- Clustering → TLS line fitting → recursive splitting on **both** curvature
  (corners) and unsupported along-axis stretches (cluster chaining) → corner
  detection by line intersection → opening detection (within and between
  segments) → corridor detection. Fits still scattered after splitting are
  discarded rather than drawn.
- **Measured against known geometry, 30 s per scenario:**

| scenario | surfaces | boundary | corners | confidence | error (mean/worst) | doorways |
|---|---|---|---|---|---|---|
| room | 4 | 5.6 m | 2 | 56% | 0.07 / 0.39 m | 1/1 |
| corridor | 3 | 7.5 m | 1 | 72% | 0.03 / 0.33 m | 1/2 |
| corner | 3 | 13.0 m | 1 | 87% | 0.02 / 0.14 m | – |
| apartment | 6 | 13.1 m | 3 | 75% | 0.03 / 0.25 m | 1/1 |
| openfield | 2 | 3.1 m | 0 | 68% | 0.00 / 0.00 m | – |

  Mean surface error **3.0 cm** (worst 0.39 m); mean confidence **72%**;
  3 of 4 doorways proposed as candidates. `openfield` stays honestly
  near-empty. Confidence is modest because only swept boundary is drawn —
  phantom walls from cluster chaining were removed rather than counted.

### Digital twin (simulation + hybrid)
`server/simulation.js` · `public/shared/echosynth.mjs` · 7 tests

- Five scenarios, real 2-D raycasting, physics-based envelope synthesis, and the
  **real EchoNet** doing the classification — not scripted labels.
- Fidelity vs. the actual training windows (`window-stats.mjs`): mass, −3 dB
  width, zero fraction, roughness and local-maxima count all match closely.
- Classifier behaviour on the twin vs. at training time
  (`verify-classifier.mjs`): WALL 96.9% vs 91.5%, SOFT 72.6% vs 74.3%,
  OPENING 47.4% vs 42.7%, overall 72.3% vs 69.5% — same class-difficulty
  ordering, within a few points.
- **Deterministic**: same seed, same scan, bit for bit.
- Works with no phone, no microphone, no network, no credentials.

### Server, sessions and network robustness
`server/hub.js` · `server/index.js` · 4 end-to-end tests

- HTTP + HTTPS listeners sharing one session hub, LAN discovery preferring Wi-Fi
  adapters, terminal QR code, `/api/{status,health,scenarios,snapshot,recordings,aws}`.
- Reconnection with jittered backoff; `hello` replayed automatically.
- Bounded outbound queue that drops stale detections before control messages.
- Bidirectional heartbeat, median-of-8 latency, 14 s peer timeout.
- **Map state survives a phone disconnect or a page refresh mid-mission** and
  rebuilds completely from a snapshot.
- Trailing-slash redirects (found by the browser check: `/map` without the slash
  silently loaded no CSS or JS).

### Record and replay
`server/recorder.js` · 2 tests + browser check

- Missions record automatically; frames replay through the *same* pipeline, so
  the grid, fits and statistics are recomputed from an empty canvas.
- Path traversal and unreadable recordings handled.
- Verified end to end in Chrome: replay repopulated the map with 43 points.

### Phone sensor application
`public/phone/`

- AudioWorklet capture with automatic ScriptProcessor fallback (the path the
  original prototype proved on the target phone).
- `getUserMedia` constraints correctly nested inside `audio`, and the *granted*
  track settings are read back and warned about if the device refused.
- Capability detection before anything is requested; a capability table shown
  on the first-run gate.
- Pose estimation: compass → absolute orientation → relative → manual, with
  step-counted dead reckoning and a confidence that decays with drift.
- Guidance: distance-dependent click cadence (2→14 Hz) and pitch
  (420→1320 Hz), stereo panning by boresight offset, haptics inside 0.9 m,
  debounced hedged speech.
- Calibration against a known wall with median estimation, quality grading, and
  an explicit statement of what it does **not** fix.
- Sonar display drawing the ~30° beam and each return's arc of angular
  uncertainty, not a bare dot.

### Command center
`public/map/`

- Fullscreen canvas, layered rendering, eased camera, pan/zoom, keyboard
  shortcuts.
- Occupancy grid, echo cloud, reconstructed surfaces, corners, opening
  candidates, corridors, trajectory, sensor marker with beam wedge and pose
  uncertainty, pulse rings fired by real detection arrivals.
- Zero-visibility mode; ghost reconstruction transition that moves each point to
  the surface actually fitted to it.
- AI panel with per-echo probabilities, fused distribution, temporal history
  (disagreeing looks in amber) and a plain-language note about stability.
- Mission summary with measured statistics and explicit caveats.
- Optional ground-truth overlay (simulation only, dashed and grey) so the
  reconstruction can be checked against the real room.

### Diagnostics
`public/diagnostics/`

- Per-subsystem status with remedies, live readings, an A-scope of the range
  profile, and 26 in-browser self-tests of DSP, geometry, occupancy, fusion,
  reconstruction and protocol.
- `TEST CHIRP`, `TEST MICROPHONE`, `TEST ORIENTATION`, `TEST WEBSOCKET`,
  `RUN SELF TEST`, `TEST FULL LOOP`.
- Ends with a single verdict naming the mode that will work on that device.

### AWS integration
`server/aws/`

- Clean adapter; the app never imports an SDK directly and never fails when AWS
  is absent.
- **Amazon Polly** for spoken guidance, with per-session caching and an honest
  `AWS POLLY` / `LOCAL FALLBACK` label reflecting what actually produced the
  last cue.
- **Amazon Bedrock** (optional, off by default) for one post-scan paragraph,
  deliberately outside the real-time loop, with a prompt that states the
  sensor's limits and requires hedging.
- `.env.example` with placeholders only. No credentials in the repo.
- 4 tests, including that the Bedrock prompt carries the measured numbers and
  the classifier's recall figures.

### Documentation
`DEMO.md` (setup, modes, calibration, AWS, a timed script, a failure table),
`ARCHITECTURE.md` (design, DSP, ML, spatial, twin, limits), `PROTOCOL.md`
(schema v1), `STATUS.md` (this file).

---

## PARTIAL

**Live acoustic performance on the OnePlus 12R — unverified on hardware.**
The DSP is verified against synthetic audio through the real code path, and the
capture code is written for mobile Chrome, but no run on the physical phone has
happened. See *Requires physical testing* below.

**Pose accuracy.** Step detection is implemented and its thresholds are
reasonable, but stride length is not personalised and no drift measurement
against ground truth has been made. Confidence decays with distance as a
stand-in for a real error model.

**Hybrid mode position.** Heading follows the real phone. Position follows the
phone only when it reports dead-reckoned movement; otherwise the virtual
walker's position is used. That is intentional but means hybrid pose is a blend,
and it is labelled `hybrid` with confidence capped at 0.75.

**Occupancy from a wide beam.** The 5-ray cone is a reasonable approximation but
a single detection genuinely cannot say *where* along the arc the reflector was.
Walls form from repeated observations at different angles; a single pass leaves
arcs rather than walls. This is visible in the demo and is honest, not a bug —
but it is not a proper inverse sensor model.

---

## STRETCH / NOT BUILT

- **TDOA bearing from stereo capture.** Stereo availability is detected and
  reported; the estimator is not implemented. Every detection carries
  `beamwidth_deg: 30` and the UI draws that uncertainty instead.
- **Real-echo validation of the classifier.** EchoNet has only ever been
  validated on synthetic data.
- **Loop closure / pose graph.** Dead-reckoning drift is displayed, never
  corrected.
- **Multi-phone fusion.** The hub supports multiple sockets but only one
  phone's pose drives the map.
- **Bedrock in the real-time loop.** Deliberately excluded.
- **Out of scope by design:** multi-floor mapping, SLAM, victim detection,
  BLE/mesh, iOS-specific work, accounts, any certification claim.

---

## KNOWN LIMITATIONS

These are properties of the approach, not defects to be fixed later.

1. **Range only, along a wide beam.** One measurement per pulse over ~30°.
   Every point is "somewhere in a cone".
2. **Bearing is the phone's boresight**, not a resolved angle. A mono
   microphone cannot resolve direction.
3. **Position is dead-reckoned or simulated**, never surveyed. Error grows with
   distance walked.
4. **The classifier is 69.5% accurate** on synthetic validation:
   WALL 91%, SOFT 74%, **OPENING 43%**. Its training set is synthetic.
   The system responds by fusing over time, showing disagreement, capping
   OPENING confidence at 0.6, labelling openings `OPENING?` with their evidence
   type, and hedging speech. It does not respond by rounding up.
5. **Reconstruction is inference.** The 3.0 cm figure is fit-to-truth error in
   simulation, not survey accuracy, and it only covers boundary the beam
   actually swept.
6. **Hardware-limited band.** Phone speakers roll off above ~15 kHz; usable
   range is roughly 0.35–3.5 m and varies by device and room. The 0.343 m floor
   is set by the direct-path gate.
7. **Absorptive rooms defeat it.** Soft furnishings at 20 kHz can leave nothing
   to detect. Simulation and hybrid exist for exactly this.
8. **Chirps are near-ultrasonic, not silent.** Some people and many animals
   hear them. The phone says so before the first transmission.
9. **Not a navigation or safety device.** No smoke-penetration,
   collision-avoidance, or rescue-readiness claim is made anywhere in the
   system, and none should be made when presenting it.

---

## REQUIRES PHYSICAL TESTING ON THE PHONE

Run `/diagnostics` on the OnePlus 12R first; it checks every item below and
tells you what to do about each.

1. **Speaker output at 17.5–22 kHz** — `TEST CHIRP`. Phone speakers roll off;
   if nothing comes back, this is the likely cause.
2. **Microphone permission and raw capture** — `TEST MICROPHONE`. Confirms
   echo cancellation, noise suppression and AGC were actually disabled. If the
   device refuses, echo amplitudes are unreliable.
3. **Real sample rate** — whether Chrome grants 48 kHz. The pipeline adapts to
   whatever it gets, but a lower rate attenuates the top of the band.
4. **AudioWorklet vs. ScriptProcessor** — which path is in use, and the measured
   DSP time per pulse.
5. **Echo detection rate in a real room** — the six-second listen in
   `TEST MICROPHONE` reports detections per pulse.
6. **Calibration factor** for this specific handset (audio I/O latency and
   speaker-mic offset).
7. **Compass availability** — `TEST ORIENTATION`, and specifically whether
   `absolute` is true. Without it, heading has no magnetic reference.
8. **Step detection** — whether the 1.6 m/s² threshold suits a normal walking
   gait with this phone in hand.
9. **Range accuracy and jitter** against a tape measure at 0.5, 1, 2 and 3 m.
10. **Sustained thermal/battery behaviour** over a few minutes of 20 Hz
    scanning.
11. **HTTPS certificate acceptance** in mobile Chrome, and that the WebSocket
    upgrades over `wss://`.

**If any of these fail, the demo still runs.** Simulation mode covers every
visual and every claim except live audio itself, and hybrid mode lets the phone
be physically moved while the acoustics stay virtual.
