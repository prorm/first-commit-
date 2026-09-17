# SentryShield — Demo Guide

A phone emits near-ultrasonic chirps and listens for the echoes. On-device DSP
turns each echo into a range, a 2,339-parameter neural network classifies the
surface, and a laptop fuses the stream into a live map of the space — built from
sound rather than light.

**This guide assumes nothing works.** Every section names its fallback, because
the demo has to survive a venue with bad acoustics, bad WiFi, no AWS account,
and no time.

---

## 1. Install and start

```bash
npm install
npm start
```

That's the whole setup. No build step, no bundler, no framework. The server
prints everything you need:

```
================================================================
  SENTRYSHIELD READY        acoustic perception prototype
================================================================

  COMMAND CENTER   http://localhost:8000/map
  PHONE SENSOR     https://192.168.29.17:8443/phone
  DIAGNOSTICS      https://192.168.29.17:8443/diagnostics

  MODE             SIMULATION   (no phone needed for simulation)
  CLASSIFIER       EchoNet loaded  (2339 params, WALL/SOFT only, 59% on real echoes vs 50% chance - experimental)
  VOICE            LOCAL FALLBACK   (AWS_REGION not set)
  BEDROCK          disabled  (no AWS credentials in environment)
  SCENARIOS        room, corridor, corner, apartment, openfield

  SCAN TO CONNECT PHONE
  [QR code]
```

The LAN address is detected automatically and Wi-Fi adapters are preferred, so
you never edit an IP by hand. If port 8000 is busy the server says so and tells
you how to change it (`PORT=8001 npm start`).

**Requires:** Node 18+ (developed on Node 22). No Python needed to run the demo —
Python is only for regenerating the classifier.

---

## 2. Connect the phone

Scan the QR code, or open the phone URL directly.

The phone URL uses **HTTPS** when `key.pem` and `cert.pem` exist in the project
root, because mobile Chrome blocks `getUserMedia` outside a secure context. On
first visit you will see a certificate warning for the self-signed cert:

> **Tap "Advanced" → "Proceed to 192.168.x.x (unsafe)"**

If you don't have certificates, generate them once:

```bash
# OpenSSL (Git Bash on Windows includes it)
openssl req -x509 -newkey rsa:2048 -nodes -keyout key.pem -out cert.pem -days 365 \
  -subj "/CN=sentryshield" \
  -addext "subjectAltName=IP:192.168.29.17,DNS:localhost"
```

Put your own LAN IP in `subjectAltName`. Then restart the server.

**No HTTPS?** The server still starts and prints an HTTP phone URL. Live audio
will not work (Chrome blocks the mic), but simulation and hybrid modes do, and
the command center is unaffected.

On the phone, tap **ENABLE SENSORS**. This one gesture is required by browser
policy to unlock the AudioContext and (on iOS) the orientation sensors. The
screen lists what the browser actually granted before you tap.

---

## 3. Check the device first: `/diagnostics`

Open `/diagnostics` on the phone you intend to demo with, and press
**RUN SELF TEST**. It runs 26 in-browser checks of the DSP, coordinate
conversion, occupancy updates, temporal fusion, reconstruction and wire
protocol — no microphone or network needed. If those pass, the build is sound
and anything else that fails is device-specific.

Then run the device tests:

| Button | What it proves | If it fails |
|---|---|---|
| **TEST CHIRP** | The speaker reaches 17.5–22 kHz | Raise the volume. Some speakers roll off; use simulation. |
| **TEST MICROPHONE** | Raw capture, mic processing disabled, echoes detected | Check the permission. If mic processing stays on, amplitudes are unreliable — use simulation. |
| **TEST ORIENTATION** | Compass heading and step counting | Without a magnetometer, heading is relative — use the manual pose controls. |
| **TEST WEBSOCKET** | Phone can reach the laptop, and a malformed frame doesn't kill the session | Check both devices are on the same network and no AP isolation is enabled. |
| **TEST FULL LOOP** | Server, digital twin, detections, fusion and reconstruction end to end | This one needs no phone hardware at all. If it passes, you have a demo. |

The page ends with a single verdict naming the mode that will work on this
device.

---

## 4. The three modes

Selected from the dropdown on the command center, or cycled with **MODE** on
the phone. All three emit the **identical** detection schema, so the map cannot
tell them apart.

### LIVE
Real chirps, real echoes, real DSP on the phone. Best when the room has hard
flat surfaces at 0.5–3 m and the venue is not too loud above 17 kHz.

Starting the scan on the phone switches the system to live mode automatically.

### SIMULATION *(the default, and the safety net)*
A deterministic virtual building. Each pulse is **raycast against real 2-D
geometry**, turned into a synthetic matched-filter envelope using the same
physics that generated the training set, and then classified by **the real
EchoNet** — so a simulated wall is labelled by the same network that labels a
live echo, mistakes included.

Five scenarios: `room`, `corridor`, `corner`, `apartment`, `openfield`.
For the finale use **apartment** (two rooms, a connecting doorway, furniture) —
it gives the richest reconstruction. `openfield` is deliberately sparse: it
shows what a weak-evidence scan honestly looks like.

Needs no phone, no microphone, no network, no credentials.

### HYBRID
Real phone orientation and motion, virtual acoustics. Lets you physically turn
and walk with the phone during the presentation even if the venue's acoustics
are hopeless. The sweep on both screens follows your actual heading.

**Say out loud that this is a demonstration mode.** The map labels the pose
source, and hybrid pose confidence is capped at 0.75 so it reads as estimated.

---

## 5. Calibration

On the phone, with the scan running, tap **CALIBRATE**:

1. Stand facing a flat wall with nothing else in front of you.
2. Hold the phone upright, speaker toward the wall.
3. Pick or type the real distance (0.5 / 1.0 / 1.5 / 2.0 m).
4. **COLLECT SAMPLES** — hold still for ~20 pulses (about one second).

It takes the median of the accepted samples and computes a correction factor,
persisted for the browser session. The badge reads **CAL: ACTIVE**.

Calibration removes fixed audio I/O latency, the speaker-to-mic offset in the
chassis, and speed-of-sound error from room temperature. It does **not** fix
beam-width ambiguity, which surface in the beam returned the echo, or classifier
accuracy — the UI says so, and so should you. A sample spread wider than 8 cm is
flagged as "poor": the beam was probably seeing more than one surface.

---

## 6. Record and replay — the reliability feature

Missions record automatically. To replay:

- **REPLAY LAST** on the command center, or **REPLAY THIS SCAN** on the summary.

A replay pushes the recorded frames back through the *same* pipeline live data
uses, so the occupancy grid, surface fits and statistics are **recomputed** from
an empty canvas rather than replayed as pictures. It needs no phone and no
microphone.

**Do a good run before the judges arrive and keep it.** Recordings are JSON in
`recordings/`. If the live demo collapses, replay is a complete substitute that
tells the whole story.

---

## 7. AWS

The core system never depends on AWS. With credentials present two things
change, and both are labelled on screen.

**Amazon Polly** — spoken guidance. The server synthesises the cue and streams
MP3 to the phone. The label reads `VOICE: AWS POLLY`. Without credentials the
phone speaks with the browser's own SpeechSynthesis and the label reads
`VOICE: LOCAL FALLBACK`. The label always reflects what actually produced the
last cue, not what is configured.

**Amazon Bedrock** *(optional, off by default)* — one paragraph of post-scan
interpretation appended to the mission summary. It runs **once, after** the
mission, on the numeric summary only — never in the real-time DSP loop. The
prompt is constrained to the measured numbers and is explicitly told the
classifier's recall figures so the prose stays hedged.

```bash
cp .env.example .env
# then fill in:
#   AWS_REGION=eu-west-1
#   AWS_ACCESS_KEY_ID=...
#   AWS_SECRET_ACCESS_KEY=...
#   ENABLE_BEDROCK=true
#   BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20240620-v1:0
npm start
```

IAM needs `polly:SynthesizeSpeech` and, for Bedrock, `bedrock:InvokeModel` plus
model access granted in the console. Check what the server detected at any time
with `curl localhost:8000/api/aws`.

---

## 8. The 2–3 minute presentation script

Setup before you start: server running, `/map` fullscreen on the laptop,
`/phone` open on the phone, mode **SIMULATION**, scenario **apartment**.

> **[0:00] The problem.** *(map on screen, empty and dark)*
>
> "When you can't see — smoke, darkness, dust — cameras and lidar stop helping.
> But sound still travels. This phone is emitting chirps at 17 to 22 kilohertz,
> just above hearing, and listening for the echoes. Everything you're about to
> see is built from that."

> **[0:20] Start the mission.** *(press START MISSION, or `M`)*
>
> "Environment unknown. The map starts black — and black means *unscanned*, not
> empty. That distinction is the whole point."
>
> Detections begin arriving at 20 Hz. Arcs appear where echoes came back.

> **[0:40] Zero visibility.** *(press ZERO VISIBILITY, or `Z`)*
>
> "Optical visibility zero. Everything still on the screen was heard, not seen."
>
> The background context drops away; only the acoustic reconstruction remains.

> **[1:00] Point at the AI panel.** *(bottom left)*
>
> "That's the real classifier — 2,339 parameters, running on the phone for every
> single echo. Wall 0.93, soft 0.02. The third bar is greyed out and marked
> unused, and that's deliberate — I'll come back to it. The bars below are the
> temporal history: it fuses consecutive looks at the same spot, and when they
> disagree it says so instead of averaging the doubt away."
>
> **Be ready to say this, because it is the most credible thing you can say:**
> "We collected 5,000 real echo pulses across four phones and the classifier
> came out at 39% — chance is 33%. So we went and found out why instead of
> tuning it. Three things. The 5,000 pulses were really 30 bursts of 175
> near-identical pulses, so our sample size was 30, not 5,000; our evaluation
> had been scoring the model on its own training pulses. Each class got
> collected at its own standoff, so range alone predicted the label at 38% —
> the network was learning where the operator stood. And the OPENING class was
> physically impossible: those pulses had the *highest* target strength of any
> class, when a hole in a wall should return the least."
>
> **Then the insight, which is the part worth remembering:**
> "An opening is not a sound texture. It's the absence of a return. CFAR takes
> the strongest peak past the gate, so pointed through a doorway the sensor
> locks onto the far wall of the next room — we were training the network to
> call a distant wall a hole. So we cut the head. EchoNet now does hard versus
> soft, and openings come from geometry: a door-width gap in a continuous run
> of reconstructed wall. That's evidence this sensor can actually produce."
>
> **If asked what it cost:** "Nothing we were relying on. Range, closing
> velocity and time-to-contact are measured, not classified, and the
> reconstruction got *better* — the apartment scan went from 3 surfaces and
> 7.5 m of wall to 6 surfaces, 11.2 m and 2 corners, because points we'd been
> throwing away as acoustic openings are real boundary evidence."

> **[1:30] The walls form.** *(the walker has covered ground by now)*
>
> "Repeated observations reinforce the same cells. The room is emerging from the
> echoes."
>
> Point out the opening candidate marker if one has appeared.

> **[1:50] Reconstruct.** *(press RECONSTRUCT, or `R`)*
>
> "Now fuse it into geometry."
>
> The point cloud animates into fitted surfaces. Each point slides to the wall
> it was actually assigned to. The title card reports surfaces, corners,
> opening candidates and confidence.
>
> "Six surfaces, three corners, one possible doorway, 77% reconstruction
> confidence. Against the known geometry the fitted walls sit within about
> three centimetres — but that's inferred boundary, not a surveyed plane,
> because the beam is 30 degrees wide, and we only draw what the beam actually
> swept."

> **[2:15] Finish.** *(press `M` again)*
>
> The summary shows distance scanned, detections, class counts, area observed,
> reconstruction figures — and, at the bottom, the limits of the scan, listed
> explicitly.
>
> "Every number there is measured. The caveats are in the product, not just the
> slide."

> **[2:35] Replay.** *(REPLAY THIS SCAN)*
>
> "And it's all recorded. This replays through the same pipeline from an empty
> canvas — no phone, no microphone, no network. Which is also our fallback if
> the room fights us."

> **[2:50] Close on AWS and the swap.**
>
> "Polly handles the spoken guidance when credentials are present, browser
> speech when they're not — the label always tells you which. And the phone and
> the map only ever exchange one message type, so the sensor is swappable:
> if browser audio fails on a device, a native bridge emitting the same
> `detection` frame drops in with zero UI changes."

**Keyboard shortcuts on the map:** `M` mission · `Z` zero visibility ·
`R` reconstruct · `P` replay · `F` fit view · `C` re-centre on the sensor.
Mouse wheel zooms, drag pans, double-click resumes following.

---

## 9. If something breaks mid-demo

| Symptom | What to do |
|---|---|
| Phone won't connect | Check both devices on the same WiFi, no AP isolation. The map keeps working — switch to SIMULATION and carry on. |
| WiFi drops mid-scan | Nothing to do. The client reconnects with backoff and re-syncs from a server snapshot; the map keeps what it had drawn. |
| No echoes in live mode | The venue is probably absorbing above 17 kHz. Switch to SIMULATION or HYBRID. Say why — it's a real limitation of the band. |
| Mic permission denied | The phone shows the exact reason and what to tap. Use SIMULATION. |
| Ranges look wrong | Calibrate against a known wall. Check the diagnostics page for mic processing left on. |
| Map looks empty | Black means unscanned. Press `F` to fit the view; check the mode; confirm detections are arriving in the stat panel. |
| Reconstruction finds nothing | It needs a few hundred echoes. Keep scanning, then press `R` again. It says "insufficient echoes" rather than inventing walls. |
| Server crashes | Restart it. Map state is lost but recordings are on disk — replay the last good scan. |
| Everything is broken | `/diagnostics` → **TEST FULL LOOP**. It exercises the whole chain with no hardware. If that passes, run the demo in simulation. |

---

## 10. Commands

```bash
npm start                              # run the demo
npm test                               # 69 automated tests
node scripts/browser-check.mjs         # drive real Chrome against every page
node scripts/verify-classifier.mjs     # digital twin vs. the real classifier
node scripts/eval-reconstruction.js    # reconstruction error vs. ground truth
node scripts/capture-hero.mjs 40       # screenshot the finale (needs the server up)
node scripts/window-stats.mjs          # simulated vs. real training windows
curl localhost:8000/api/status         # full server state as JSON
curl localhost:8000/api/aws            # what the AWS adapter detected
```

Further reading: [ARCHITECTURE.md](ARCHITECTURE.md) for how it works,
[STATUS.md](STATUS.md) for what is complete and what isn't,
[PROTOCOL.md](PROTOCOL.md) for the wire schema.
