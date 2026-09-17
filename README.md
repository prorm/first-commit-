# SentryShield

**Acoustic perception from a phone.** The speaker emits near-ultrasonic chirps,
the microphone listens for echoes, on-device DSP turns each echo into a range, a
2,339-parameter neural network classifies the surface, and a laptop fuses the
stream into a live map of the space — built from sound rather than light.

Research prototype for an AWS-sponsored hackathon. Not a navigation or safety
device.

```bash
npm install
npm start
```

The server prints the command-center URL, the phone URL, and a QR code to scan.
**No build step, no bundler, and no phone required** — simulation mode runs the
whole demo with no hardware, no network and no credentials.

| | |
|---|---|
| **[DEMO.md](DEMO.md)** | Setup, connecting the phone, the three modes, calibration, AWS, a timed 2–3 minute script, and what to do when something breaks. |
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | How it works: DSP chain, ML pipeline, spatial reconstruction, the digital twin, AWS integration, limitations. |
| **[STATUS.md](STATUS.md)** | What is complete, partial, and not built — with measured numbers and the list of things that still need the physical phone. |
| **[PROTOCOL.md](PROTOCOL.md)** | Wire schema v1: the single message boundary between sensor and UI. |

## Routes

- `/map` — command center (open on the laptop)
- `/phone` — the sensor itself (open on the phone)
- `/diagnostics` — capability checks, per-subsystem tests, 26 in-browser self-tests
- `/legacy/gate-test` — the original single-page prototype, kept intact

## Commands

```bash
npm start      # run the demo
npm test       # 69 automated tests
npm run lint   # syntax-check every source file, including browser-only modules
npm run browser  # drive real Chrome against every page, fail on any console error
npm run eval   # classifier, reconstruction and echo-fidelity measurements
npm run verify # lint + test + browser check
```

## Measured, not claimed

| | |
|---|---|
| Range accuracy (synthetic audio through the real pipeline) | 0.4 cm mean error; <15 cm RMSE over a 3 m approach |
| Empty-room false alarms | 0% |
| Surface reconstruction vs. known geometry | 3.0 cm mean error (worst 0.39 m), 72% mean confidence |
| Echo classifier (synthetic validation) | 69.5% overall — WALL 91%, SOFT 74%, **OPENING 43%** |

The classifier's weak class is surfaced everywhere it matters: openings are
labelled `OPENING?` with their evidence type, their confidence is capped, speech
hedges them as "possible", and the AI panel shows when consecutive predictions
disagree. See [STATUS.md](STATUS.md#known-limitations) for the full list.
