# SentryShield — Architecture

Written for an engineer who wants to know how this actually works and where the
seams are.

---

## 1. The one rule

Sensor and UI are split at a strict message boundary. Everything downstream of
the sensor module — phone display, tones, haptics, the WebSocket uplink, the
command center, recordings — reads **only** `Detection` objects. Nothing outside
`public/phone/sensor.mjs` and `public/phone/dsp/` ever touches an audio buffer.

That boundary is the reason the system is demoable. If Chrome's audio path fails
on a device, the sensor module can be replaced by a native Android bridge
emitting the same `detection` frame over the same socket, with zero UI rework.
It is also why **simulation, hybrid, live and replay are interchangeable**: all
four produce byte-identical frames, so the map genuinely cannot tell them apart.

---

## 2. System shape

```
PHONE  (public/phone/)                    browser, no build step
├── sensor.mjs          AudioWorklet (→ ScriptProcessor fallback) capture,
│                       chirp transmit, ring buffer.  ONLY audio-aware module.
├── dsp/pipeline.mjs    matched filter → gate → clutter subtract → CFAR →
│                       alpha-beta tracker → 64-sample window → EchoNet
├── dsp/fft.mjs         radix-2 FFT (carried over from the working prototype)
├── pose.mjs            orientation + step-counted dead reckoning, with a
│                       confidence that decays as drift accumulates
├── audio-guidance.mjs  proximity tone (rate + pitch + pan), haptics,
│                       VoiceProvider (Polly | SpeechSynthesis)
├── calibration.mjs     known-wall range correction, session-persisted
└── phone.mjs           UI, sonar rendering, uplink

        │  WebSocket /ws   —  `detection` frames, ~20 Hz, ~400 bytes each
        ▼

NODE SERVER  (server/)
├── index.js            HTTP + HTTPS listeners, LAN discovery, QR, /api/*
├── static.js           dependency-free static serving, route redirects
├── hub.js              THE message boundary: sessions, heartbeat, relay,
│                       mode switching, mission lifecycle
├── mapstate.js         authoritative fused state (grid + cloud + trajectory)
├── simulation.js       digital twin: raycast + echo synthesis + real EchoNet
├── recorder.js         record / replay through the same pipeline
├── guidance.js         what to say, and when to stay quiet
└── aws/                adapter.js · polly.js · bedrock.js

        │  `detection` + periodic `state_snapshot`
        ▼

COMMAND CENTER  (public/map/)
├── map.mjs             local world model, HUD, AI panel, controls
└── renderer.mjs        layered canvas: occupancy → surfaces → cloud →
                        trajectory → sensor → pulses

SHARED  (public/shared/)   imported unchanged by BOTH sides
├── protocol.mjs        schema v1, validation, normalisation, polar→world
├── spatial.mjs         OccupancyGrid, PointCloud, Trajectory, ClassFuser
├── reconstruct.mjs     clustering, line fitting, corners, openings, corridors
├── echosynth.mjs       physics-based echo synthesis for the digital twin
└── wsclient.mjs        reconnecting client with backoff and queueing
```

The shared modules are `.mjs` and imported *verbatim* by the browser and by Node
(Node 22's `require(esm)`). The command center's smooth local rendering and the
server's authoritative state are therefore literally the same computation — they
cannot drift apart.

---

## 3. Data flow

```
speaker ──▶ 15 ms Hann-windowed LFM chirp, 17.5→22 kHz, 0.6 FS, every 50 ms
                                │
                          environment
                                │
microphone ──▶ 48 kHz raw capture (EC/NS/AGC all disabled)
                                │
        ┌───────────────────────┴───────────────────────┐
        │  1  matched filter   FFT cross-correlation vs the reference chirp,
        │                      taken as an analytic signal → smooth envelope
        │  2  range alignment  re-index so sample 0 is the direct path
        │  3  direct-path gate zero ±2 ms around the transmit leakage
        │  4  clutter subtract live − running 40-pulse envelope mean, rectified
        │  5  CFAR             smallest-of CA-CFAR + Kalman noise floor,
        │                      search 0.34–3.75 m
        │  6  tracker          alpha-beta smoothing, jump rejection >0.8 m
        │  7  classifier       64-sample peak-centred window → EchoNet
        └───────────────────────┬───────────────────────┘
                                │
                          Detection  ─── the ONLY thing that leaves
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
  local tones/haptics    WebSocket uplink         phone sonar display
                                │
                                ▼
                    temporal class fusion (server)
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
   occupancy grid         point cloud             trajectory
        └───────────────────────┼───────────────────────┘
                                ▼
                     surface reconstruction
                                ▼
                       command center render
```

---

## 4. DSP, step by step

**Signal parameters** (fixed, not re-derived): linear FM chirp, Hann-windowed,
17,500→22,000 Hz, 4500 Hz bandwidth, 15 ms duration, 50 ms pulse interval
(20 Hz), 48 kHz sample rate, TX amplitude 0.6 FS.

**1 — Matched filter.** FFT cross-correlation of the RX window against the
reference chirp. Doubling the positive frequencies and zeroing the negative ones
before the inverse transform returns `hilbert(correlation)` directly, giving a
smooth envelope without a separate Hilbert stage. Compression gain is ~16×; the
compressed pulse is ~28 samples wide at −3 dB (Hann² spectral weighting).

**2 — Range alignment.** The transmit leakage is 20–40 dB above any echo, so the
global envelope maximum *is* the direct path. Re-indexing the envelope so that
peak sits at sample 0 makes every subsequent stage range-referenced, which is
what lets a running clutter baseline be accumulated at all.

**3 — Direct-path gate.** Zero ±2 ms (96 samples) around sample 0. This also
sets the true minimum range: 0.343 m, further out than the nominal 0.30 m
search floor.

**4 — Clutter subtraction.** A running mean of the last ~40 *envelopes* is
subtracted and the result rectified at zero. Averaging envelopes (not complex
signals) retains the noise floor's positive bias, so subtracting it and clipping
leaves a zero-heavy residual — **exactly the profile EchoNet was trained on.**
Static returns fade into the baseline over ~2 s; a moving target does not.

**5 — CFAR.** Smallest-of cell-averaging CFAR (guard 28, train 32, Pfa 1e-4).
Plain CA-CFAR self-masks extended targets — a person spans 60–90 samples, so
both training windows can sit on the target — and taking the smaller side only
needs one clean side.

Two details that mattered:

- *The noise estimate is floored by a sigma-clipped window average.* Because
  step 4 rectifies at zero, roughly half the noise-only cells are exactly 0; a
  purely local estimate collapses toward zero over such a run and then **every**
  peak clears the threshold. Measured false-alarm rate before the floor: 100%.
- *The threshold carries a calibrated margin (×10).* The textbook factor assumes
  exponential clutter; a rectified profile is heavier-tailed. The margin was
  fitted against measured false-alarm rate, and the number is pinned by
  `tests/dsp.test.js`: 0% false alarms in an empty room, while still detecting a
  2.5e-4 FS echo 97% of the time.

A peak that fails CFAR may *maintain* an established track but may never
*acquire* one. Letting noise peaks start tracks is what makes a display jump
around an empty room.

**6 — Alpha-beta tracker.** α = 0.35, β = 0.074 (critically damped, so range
settles without overshoot). Jumps beyond 0.8 m are rejected unless the new range
repeats for 3 consecutive pulses. After 5 consecutive rejections the track is
considered lost and the next measurement is adopted outright — holding on longer
strands the display on a stale range, which looks exactly like a frozen app.

**7 — Classifier input.** 64 samples of the rectified profile centred on the
CFAR peak, peak-normalised to [0, 1] — the same construction as the training
set, verified by test.

**Measured performance** (`npm test`, synthetic audio through the real
pipeline): 0.4 cm mean error on a static target; **<15 cm RMSE and near-monotonic
over a 3 m approach at 0.5 m/s**; 0% empty-room false alarms.

---

## 5. ML pipeline

**EchoNet** — `src/classifier/echonet_weights.js`, 2,339 parameters, pure JS,
zero dependencies, runs in the browser and in Node.

```
Float32Array(64)  peak-normalised envelope window, peak at index 32
  → Conv1d(1→16, k5, p2) + BN + ReLU + MaxPool2
  → Conv1d(16→32, k3, p1) + BN + ReLU + MaxPool2
  → GlobalAvgPool → Linear(32→16) + ReLU → Linear(16→3)
  → softmax over {WALL, SOFT, OPENING}
```

BatchNorm is folded into the convolutions for inference. The file ships a
self-test vector captured from PyTorch at export time; the diagnostics page
runs it and reports max absolute error (currently 7.2e-7).

**Training data** came from `src/classifier/generate_echonet_dataset.py`, a
physics simulator: ISO 9613-1 air absorption, r⁻² spreading, per-device speaker
and MEMS-mic transfer functions, room reverberation, floor and dihedral
multipath, Doppler, 16-bit quantisation, and class-specific scattering physics.

**The honest numbers** (synthetic validation, 3,000 samples):

| | recall | confusion |
|---|---|---|
| WALL | **91.5%** | 915 / 34 / 51 |
| SOFT | **74.3%** | 62 / 743 / 195 |
| OPENING | **42.7%** | 165 / 408 / 427 |

Overall accuracy **69.5%** — on synthetic data, which is the whole problem.
None of it survives real echoes. Measured against the recordings in
`recordings/`, held out by whole session, the three-class model scores **39%
against a 33% chance floor**, and a purely synthetic-trained model scores
**48.2% on a two-class WALL/SOFT split, below its 50% chance floor.**

OPENING is the weak class, and the synthetic numbers mislabel *why*. It is not
that diffraction is subtle. It is that **an opening produces no echo to
classify at all.** CFAR takes the strongest peak past the direct-path gate, so
aimed through a doorway the detector locks onto the far wall of the next room,
and the window handed to the network is a wall echo. The real recordings say
this outright: pulses labelled OPENING carry the *highest*
spreading-compensated target strength of any class (23.7 dB against WALL's
18.2 dB), which is physically backwards for a hole in a wall.

### Live boundaries vs. fitted surfaces

Two layers draw walls, from two different claims, and the distinction is the
point rather than a redundancy.

`shared/boundary.mjs` (**live boundaries**) works per echo. A single return does
not locate a point, it locates a *wavefront tangent*: the reflector lies
somewhere on an arc of radius `range` across the beam, and for a flat surface
the chord of that arc is the surface. So one confident echo already draws a
short bar, pinned in world coordinates — walk forward or turn around and it
stays where the wall is. Successive chirps on the same wall merge (within 15 cm
and 14 degrees) into one longer bar instead of stacking, and a bar a later pulse
measures straight through loses strength and is withdrawn. In the apartment
scenario 753 detections collapse to 43 bars spanning 9 x 7 m.

`shared/reconstruct.mjs` (**fitted surfaces**) works on the accumulated cloud
with a real line fit, residual and inlier test. It is a stronger claim and it
needs history, which is why `SURFACES` sits at 0 while `BOUNDARIES` is already
climbing.

The renderer fades the live bars out as the RECONSTRUCT morph comes up, so the
weaker claim yields to the stronger one instead of arguing with it on screen.
Neither layer touches the wire protocol: boundaries are computed client-side
from detections the map already receives.

**How the architecture handles that**, rather than hiding it:

0. **The OPENING head is not used.** EchoNet decides WALL vs SOFT only
   (`shared/surfaceclass.mjs`), and returns no call when the two cannot be
   separated. Openings are recovered geometrically by `findGapOpenings()` — a
   door-width gap in an otherwise continuous run of reconstructed wall — which
   is evidence this sensor can actually produce. The rule is enforced at all
   three re-entry points: the pipeline, `normalizeDetection()`'s fallback
   derivation, and `ClassFuser`.

1. **Temporal fusion** (`spatial.mjs → ClassFuser`). Confidence-weighted,
   recency-weighted aggregation over a 2.2 s window for detections within 0.45 m
   of each other. The fused confidence is the mean posterior, so a class that
   keeps winning narrowly stays reported as low-confidence. `stable` requires
   both ≥3 looks **and** ≥60% agreement — never confidence alone.
2. **Disagreement is displayed.** The AI panel's history bars turn amber for
   looks that disagreed with the final decision, and the note reads
   "Unstable: N echoes at this spot, predictions disagree."
3. **OPENING confidence is capped at 0.6** in reconstruction, and openings are
   always drawn as `OPENING?` with their evidence type.
4. **Two independent kinds of opening evidence** are tracked separately —
   acoustic (the classifier said so) and geometric (a door-width hole in an
   otherwise continuous wall). Agreement between them raises confidence; either
   alone stays a candidate.
5. **Spoken guidance hedges.** Below 0.55 fused confidence the phrase is
   "Possible opening on your left", not "Opening detected".

---

## 6. Spatial reconstruction

**Occupancy grid** — 0.1 m cells, log-odds, ±12 m. Each detection traces 5 rays
across the ~30° beam cone (edge rays weighted lower): traversed cells accumulate
`−0.42 × weight`, the reflector cell `+1.05 × weight`, saturating at ±5.5. Cell
weight folds detection confidence with *pose* confidence, so a detection taken
from a badly dead-reckoned position cannot carve crisp geometry.

**Unknown is exactly 0 and is never painted.** Observed-free, observed-occupied
and never-observed stay three visually distinct states. That distinction is the
entire visual argument of the demo.

An `OPENING` detection deliberately marks **no** occupied cell and instead
carves free space ~18% past the measured range. The evidence is a *lack* of
boundary, so doorways appear as gaps in the geometry.

**Point cloud** — world points consolidated on a 0.16 m spatial hash, searching
the 3×3 bucket neighbourhood so points near a seam still merge. Re-observing a
spot raises one point's weight and averages its position rather than stacking
duplicates; class is decided by confidence-weighted voting. Capped at 4,000
points, pruned weakest-first.

**Surface reconstruction** (`reconstruct.mjs`):

1. Single-link clustering at 0.38 m over a spatial hash.
2. Total-least-squares line fit via the principal axis of the weighted
   covariance.
3. Recursive split, on two independent signatures:
   - *curved* — residual above 0.13 m means the cluster spans a corner; split
     at the point of maximum perpendicular deviation, which is where the
     corner is.
   - *chained* — single-link clustering is permissive by design, so a run of
     echo arcs at different bearings can link right across a room. The chain
     can be almost straight, so the residual looks fine while the fit spans
     open space its own points never occupied. The signature is a long
     **unsupported stretch along the fitted axis**: a real wall does not have a
     two-metre hole in the middle of its own evidence. Splitting there cut mean
     surface error from 8.7 cm to 3.0 cm and removed long diagonal phantom
     walls from the display.
   A fit still scattered beyond 0.22 m after splitting is discarded entirely;
   its points stay in the cloud and simply go unexplained.
4. **Corners** by intersecting the fitted *lines*, not by matching endpoints.
   Endpoint matching almost never fires: the cloud thins towards a corner
   because the wall goes oblique to the beam and stops returning, so both fits
   end short. Lines may be extended up to 1.1 m to meet.
5. **Openings** two ways — a door-width gap *within* one fit's along-axis point
   spread, and a door-width gap *between* two nearly-collinear fits. The second
   is the case that actually happens, since a doorway is wider than the
   clustering radius and splits the wall into two segments.
6. **Corridors** — two long, roughly parallel fits 0.7–2.6 m apart with
   substantial overlap.

Every feature carries its own confidence and inlier count. Overall
reconstruction confidence requires three things at once: most of the cloud
explained by some surface, enough total boundary to be a room rather than a
fragment, and decent individual fits.

**Measured against known geometry** (`node scripts/eval-reconstruction.js`,
30 s per scenario — ground truth is known because it is a simulation):

| scenario | surfaces | boundary | corners | confidence | surface error (mean/worst) | doorways |
|---|---|---|---|---|---|---|
| room | 4 | 5.6 m | 2 | 56% | 0.07 / 0.39 m | 1/1 |
| corridor | 3 | 7.5 m | 1 | 72% | 0.03 / 0.33 m | 1/2 |
| corner | 3 | 13.0 m | 1 | 87% | 0.02 / 0.14 m | – |
| apartment | 6 | 13.1 m | 3 | 75% | 0.03 / 0.25 m | 1/1 |
| openfield | 2 | 3.1 m | 0 | 68% | 0.00 / 0.00 m | – |

Mean surface error **3.0 cm**, worst case 0.39 m, mean confidence **72%**,
3 of 4 doorways proposed. `openfield` produces almost nothing — the correct
outcome for a mostly-open space, and the map stays honestly empty.

The confidence figures are deliberately modest because the total reconstructed
boundary is short: only surfaces the beam actually swept are drawn. An earlier
version reported 86% confidence and 8.7 cm error over ~16 m of boundary per
scenario, but a chunk of that boundary was an artefact — see the chaining note
in step 3 below.

**This is not a survey accuracy claim.** The beam is tens of degrees wide, so
every point is "somewhere in a cone" and every segment is an *inferred*
boundary fitted to a noisy cloud.

---

## 7. The digital twin

`server/simulation.js` + `public/shared/echosynth.mjs`.

Simulation mode does **not** fabricate classifier output. Per pulse:

1. **Raycast** the phone's boresight against real 2-D scenario geometry →
   nearest surface, its range, its material, and the incidence angle (grazing
   incidence loses energy, folded into the pointing loss).
2. **Synthesise a matched-filter envelope** with the training set's physics:
   - complex point-spread function `χ(τ) = ∫ shape(f)·e^{j2π(f−f_mid)τ} df`,
     where `shape` is Hann² × the device response × the surface's transfer
     function. Spectral tilt is what broadens and skews the lobe, so it is
     modelled rather than approximated.
   - per-class physics: WALL = strong single specular lobe plus dihedral and
     floor-bounce satellites; SOFT = weak, low-passed `(f/F0)^-p`, dispersed
     over 15–30 samples with a glint cluster up to 0.25 m deep; OPENING = no
     specular return at all, only knife-edge diffraction (`~1/√f`, −π/4 phase)
     plus the far room beyond the doorway.
   - **band-limited** receiver noise. The matched filter only passes
     17.5–22 kHz, so its output noise wanders on the scale of the compressed
     pulse, not sample to sample. White noise produces a visibly jagged envelope
     no real receiver ever sees — and the classifier notices, because roughness
     is one of the few cues it has.
   - clutter subtraction against a Rician-mean baseline, rectified at zero.
3. **Classify with the real EchoNet.**

Fidelity check (`node scripts/window-stats.mjs`) against the real training
windows:

| descriptor | real | simulated |
|---|---|---|
| mass (WALL/SOFT/OPENING) | 28.4 / 33.2 / 30.1 | 28.7 / 34.5 / 28.9 |
| width >0.5 | 27.3 / 32.6 / 29.3 | 27.6 / 34.4 / 27.9 |
| zero fraction | .037 / .053 / .096 | .022 / .032 / .114 |
| roughness | .0038 / .0049 / .0051 | .0034 / .0044 / .0055 |
| local maxima | 1.01 / 1.33 / 1.23 | 1.00 / 1.26 / 1.32 |

And the classifier's behaviour on them (`node scripts/verify-classifier.mjs`):

| | trained | on the twin |
|---|---|---|
| WALL recall | 91.5% | 96.9% |
| SOFT recall | 74.3% | 72.6% |
| OPENING recall | 42.7% | 47.4% |
| overall | 69.5% | 72.3% |

An independent reimplementation landing within a few points, and reproducing the
class-difficulty ordering, is the claim being made — not that they are identical.
Both columns are synthetic-on-synthetic, and neither transfers: see the real-echo
figures above. The twin now routes its classifier output through the same
WALL/SOFT rule the phone uses, so it cannot report a class the live sensor would
never emit.

Simulation is **deterministic**: same seed, same scan, bit for bit (tested).

---

## 8. Pose

Not SLAM, and the UI says so.

**Heading**, best source first: `webkitCompassHeading` → `alpha` with
`absolute=true` (magnetometer-referenced) → `alpha` without (relative, zeroed on
calibration) → manual dial.

**Position**: step detection from accelerometer peaks with hysteresis and a
260 ms refractory period, advanced along the current heading by a cadence-scaled
stride. Or the manual step button. Or simulated.

**Confidence** decays with dead-reckoned distance (`1 − d/22`, floored at 0.12)
and is capped lower without a magnetic reference. The map draws it: a dashed
uncertainty circle around the sensor that grows as confidence falls, and
trajectory segments drawn thinner where pose confidence was low, so drift is
visible rather than hidden.

**Bearing is the phone's boresight, not a resolved angle.** A mono microphone
cannot resolve direction. Stereo capture is detected and reported, but TDOA
bearing is *not* implemented — every detection carries `beamwidth_deg: 30` and
the UI draws that arc of angular uncertainty rather than a bare dot.

---

## 9. AWS integration

```
GuidanceEngine (phone)
  └── VoiceProvider
        ├── asks the server (Polly path)          AwsAdapter → PollyVoice
        └── SpeechSynthesis (local fallback)      always available

MapState.summary()  ── once, post-mission ──▶     AwsAdapter → BedrockSummarizer
```

`server/aws/adapter.js` is the only thing that knows AWS exists. Contract:

- The app never imports an AWS SDK directly, and never fails because AWS is
  absent.
- Every call returns a `provider` field, so the UI states which service
  **actually** produced the last cue rather than what is configured.
- Credentials come from the SDK's default chain (env, profile, role). None are
  ever created, guessed or defaulted in code.
- A credential or permission failure disables Polly for the session rather than
  stalling every future cue behind a doomed network call; a timeout does not.
- Successful syntheses are cached by text — guidance phrases repeat constantly,
  so a full demo is a handful of API calls.
- The SDKs are `optionalDependencies`: if they are not installed, the adapter
  reports itself unavailable and everything else runs.

Bedrock is deliberately narrow: **never in the real-time loop**, once per
mission, on the numeric summary only. The prompt lists the measured figures,
states the sensor's limits and the classifier's recall numbers, and requires
hedged language. Response shapes for Anthropic and Titan are both handled.

---

## 10. Network robustness

- **Reconnection** with exponential backoff plus jitter, capped at 8 s. The
  `hello` handshake is replayed on every reconnect, so the server re-learns the
  role automatically.
- **Bounded outbound queue** (200 frames) while offline. Under pressure stale
  `detection`/`pose` frames are dropped in preference to control messages — a
  stale detection is worthless, a queued `mission_start` is not.
- **Heartbeat** every 3–4 s in both directions; a peer silent for 14 s is
  terminated rather than assumed alive. Latency is the median of the last 8
  round trips.
- **State survives disconnection.** A phone dropping does not clear the map. On
  reconnect the client requests a `state_snapshot` and rebuilds the grid, cloud,
  trajectory and reconstruction completely — a page refresh mid-mission loses
  nothing.
- **A malformed frame produces an `error` message, never a closed socket.**
  Verified both in `npm test` and in-browser on the diagnostics page.
- Detections are ~400 bytes of JSON at 20 Hz (~8 kB/s). Raw audio never crosses
  the network.

---

## 11. Performance

- All DSP buffers are pre-allocated; the per-pulse path allocates nothing. At
  20 Hz the measured DSP cost is reported live on the diagnostics page.
- Capture prefers AudioWorklet (audio thread, no main-thread jank) and falls
  back automatically to ScriptProcessor — the path the original prototype proved
  on the target phone. The worklet is inlined as a blob, so there is no extra
  file to fail to load over a self-signed certificate.
- The worklet batches 2,048 samples per `postMessage` (~23/s) rather than
  posting every 128-sample render quantum.
- The occupancy grid is serialised for the renderer at most every 220 ms, not
  per frame: a full scan is tens of thousands of cells.
- Reconstruction runs on a ~450 ms cadence client-side and a 700 ms cache
  server-side, not per detection.
- The grid crosses the wire sparsely — only non-zero cells, one quantised byte
  each.

---

## 12. Failure behaviour

None of these produce a blank screen. Each degrades one capability and reports
why, in place:

| Failure | Behaviour |
|---|---|
| Microphone denied | Phone shows the exact cause and the remedy; simulation and hybrid still work |
| Not a secure context | Detected before anything is requested; the page says the mic needs HTTPS |
| AudioWorklet unavailable | Automatic ScriptProcessor fallback, logged |
| Device forces ≠48 kHz | Pipeline uses the real rate; ranges stay correct, warning shown |
| Mic processing can't be disabled | Detected from the granted track settings and warned about |
| Orientation unavailable | Heading falls back to manual; the map labels the pose source |
| Motion unavailable | Dead reckoning off, manual step control offered |
| Classifier missing | Ranges and mapping continue; classification is reported unavailable |
| Classifier throws | Caught per pulse; the range measurement survives |
| AWS absent/failing | Local speech, Bedrock section hidden, both labelled |
| WebSocket down | Reconnects; map keeps its state and re-syncs |
| Malformed packet | Repaired or rejected with an `error`; the session lives |
| DSP exception | Caught per pulse; the scan continues |
| Server exception | Logged by top-level handlers; the process keeps serving |

---

## 13. Limitations

Stated plainly, because they are the difference between a prototype and a claim.

- **Range, not position.** One measurement per pulse, along a ~30° beam. Every
  point is "somewhere in a cone".
- **No resolved bearing.** Bearing is the phone's boresight. TDOA is not
  implemented.
- **Pose is dead-reckoned or simulated**, never surveyed. Error accumulates.
- **Classifier is experimental and weak.** 69.5% on synthetic validation, but
  **39% on real echoes held out by session, against a 33% chance floor**; a
  synthetic-only model scores *below* chance on a two-class split. It is scoped
  to WALL vs SOFT, may return no call, and cannot cause a spoken cue. Openings
  are geometric, not classified. Nothing in the range/velocity/TTC path or the
  reconstruction depends on it.
- **Reconstruction is inference**, fitted to a noisy cloud. The 3.0 cm figure is
  fit-to-truth error *in simulation*, over boundary the beam actually swept.
- **Band-limited by the hardware.** Phone speakers roll off above ~15 kHz, so
  usable range is roughly 0.35–3.5 m and varies a lot by device and room.
- **Absorptive rooms defeat it.** Soft furnishings at 20 kHz can leave nothing
  to detect.
- **Not a navigation or safety device.** No smoke-penetration, collision-
  avoidance or rescue-readiness claims are made or implied anywhere in the
  system.

---

## 14. What is reused vs. new

**Reused as-is** — the radix-2 FFT and the matched-filter core from the working
PITCHBLACK gate-test prototype (it already produced correct ranges on the target
phone), the ScriptProcessor capture path, the exported EchoNet weights, and the
physics conventions of `generate_echonet_dataset.py`. The original prototype is
still served intact at `/legacy/gate-test`.

**New** — range alignment, clutter subtraction, CFAR, the tracker, the
classifier windowing, pose estimation, guidance, calibration, the protocol, the
spatial engine, reconstruction, the digital twin, record/replay, the AWS
adapter, both UIs, diagnostics, and the tests.
