# SentryShield — Wire Protocol v1

One message boundary for the whole system. Defined once in
`public/shared/protocol.mjs` and imported unchanged by the Node server and both
browser clients, so there is no second copy to drift.

Transport: JSON text frames over WebSocket at `/ws` (`ws://` for HTTP pages,
`wss://` for HTTPS).

---

## Envelope

Every message:

```json
{ "v": 1, "type": "<message type>", "t": 1758067200000, "...": "payload" }
```

| field | meaning |
|---|---|
| `v` | protocol version. A mismatch is rejected with an `error`. Omitted is accepted. |
| `type` | one of the types below. Unknown types are rejected. |
| `t` | sender's `Date.now()` |

**Validation never throws and never closes the socket.** A malformed frame
produces an `error` message back to the sender; the session continues. This is
tested in `tests/e2e-pipeline.test.js` and in-browser via
`/diagnostics → TEST WEBSOCKET`.

---

## `Detection` — the core type

The only thing that crosses the sensor boundary. Live, simulation, hybrid and
replay all produce byte-identical frames.

```json
{
  "v": 1,
  "type": "detection",
  "t": 1758067200123,
  "detection": {
    "t": 1758067200123,
    "range_m": 1.82,
    "vel_mps": 0.34,
    "ttc_s": 5.35,
    "confidence": 0.81,
    "bearing_deg": 94.0,
    "beamwidth_deg": 30,
    "snr_db": 22.4,
    "cfar_pass": true,
    "obstacleClass": "WALL",
    "classConfidence": 0.94,
    "classProbs": [0.94, 0.04, 0.02],
    "fusedClass": "WALL",
    "fusedConfidence": 0.91,
    "fusedProbs": [0.91, 0.06, 0.03],
    "fusedHistory": [0.81, 0.88, 0.91, 0.94],
    "fusedSupport": 4,
    "fusedStable": true,
    "worldX": 1.815,
    "worldY": -0.127,
    "phone": {
      "x": 0.0, "y": 0.0, "heading": 94.0,
      "confidence": 0.72, "method": "dead-reckoning"
    },
    "source": "live"
  }
}
```

| field | type | range | meaning |
|---|---|---|---|
| `t` | number | — | capture time, ms since epoch |
| `range_m` | number | 0 < r ≤ 20 | estimated range to the reflector. **Required**; a frame without a usable one is rejected. |
| `vel_mps` | number | −5…5 | closing velocity, **positive = approaching** |
| `ttc_s` | number | 0…10 | `range / max(vel, 0.05)`, clamped |
| `confidence` | number | 0…1 | *detection* confidence (SNR above the noise floor × track support). Not class confidence. |
| `bearing_deg` | number | 0…360 | absolute compass bearing of the beam boresight. 0 = +y (north), clockwise. Defaults to `phone.heading`. |
| `beamwidth_deg` | number | 4…120 | angular uncertainty of that bearing. Always ~30 for this sensor: it is **not** a resolved angle. |
| `snr_db` | number | −20…60 | CFAR signal-to-noise estimate |
| `cfar_pass` | boolean | — | did the peak clear the CFAR threshold |
| `obstacleClass` | string\|null | `WALL`\|`SOFT`\|`OPENING` | EchoNet argmax for **this single echo** |
| `classConfidence` | number | 0…1 | EchoNet max softmax probability for this echo |
| `classProbs` | number[3] | sums to 1 | full `[WALL, SOFT, OPENING]` distribution |
| `fusedClass` | string\|null | — | decision after temporal fusion (server-added) |
| `fusedConfidence` | number | 0…1 | mean posterior across the fused window |
| `fusedProbs` | number[3] | sums to 1 | fused distribution |
| `fusedHistory` | number[] | — | recent per-look confidences; a **negative** value marks a look that disagreed with the final decision |
| `fusedSupport` | number | — | how many looks are in the window |
| `fusedStable` | boolean | — | ≥3 looks **and** ≥60% agreement. Never confidence alone. |
| `worldX`, `worldY` | number | metres | reflector in the world frame; computed if absent |
| `phone` | Pose | — | the pose used to project this detection |
| `source` | string | `live`\|`simulation`\|`hybrid`\|`replay` | what produced it |

### `Pose`

| field | type | meaning |
|---|---|---|
| `x`, `y` | number | metres east / north of the mission origin |
| `heading` | number | compass degrees, 0 = +y, clockwise |
| `confidence` | number | 0…1, decays with dead-reckoned distance |
| `method` | string | `dead-reckoning` \| `manual` \| `simulated` \| `hybrid` \| `static` |

### Coordinate convention

```
worldX = phone.x + range_m · sin(bearing_deg)
worldY = phone.y + range_m · cos(bearing_deg)
```

+x east, +y north, origin at the mission start pose. Bearing 0 = north,
90 = east, 180 = south, 270 = west.

### Normalisation

`normalizeDetection()` **repairs** rather than rejects: out-of-range numbers are
clamped, bearings wrapped, probability vectors renormalised, a missing pose
becomes an explicit `static` pose, and `worldX`/`worldY` are computed. It returns
`null` only when there is no usable `range_m`. A demo must not die on one bad
frame.

---

## Client → server

| type | sender | payload | purpose |
|---|---|---|---|
| `hello` | any | `role`, `device` | handshake. `role` ∈ `phone`\|`map`\|`diagnostics`. Replayed on every reconnect. |
| `detection` | phone | `detection` | one acoustic detection |
| `pose` | phone | `pose` | pose-only update between detections |
| `calibration` | phone | `factor`, `offsetM`, `samples`, `knownDistanceM` | measured range correction |
| `heartbeat` | any | `clientTime` | liveness + latency |
| `scan_start` / `scan_stop` | any | `rateHz` | sensor started / stopped |
| `mission_start` | any | `scenario`, `rateHz`, `record`, `seed` | clear state, start stats, begin recording |
| `mission_complete` | any | — | end mission; server replies `mission_summary` |
| `set_mode` | any | `mode` ∈ `live`\|`simulation`\|`hybrid` | switch data source |
| `sim_control` | any | `scenario`, `speed`, `steer`, `paused`, `action` | drive the digital twin |
| `record_start` / `record_stop` | any | `note` | manual recording control |
| `replay_start` | any | `id`, `speed` | replay a recording (newest if `id` omitted) |
| `replay_stop` | any | — | abort |
| `request_state` | any | — | ask for a full snapshot (used after reconnect) |
| `request_summary` | any | `summary` | ask the AWS adapter for a Bedrock interpretation |
| `speak` | any | `text`, `level` | ask the server to synthesise speech |
| `reconstruct` | any | — | force a reconstruction pass and broadcast it |

## Server → client

| type | payload | purpose |
|---|---|---|
| `welcome` | `sessionId`, `protocolVersion`, `mode`, `capabilities`, `scenarios`, `simulation`, `aws`, `recordings` | sent on connect |
| `state_snapshot` | `pose`, `grid`, `cloud`, `trajectory`, `reconstruction`, `stats`, `recent`, `missionActive` | complete fused state |
| `detection` | `detection`, `replay?` | relayed with fusion applied |
| `pose` | `pose` | relayed pose update |
| `guidance` | `text`, `level`, `key`, `reason`, `voice` | what to say; `voice.label` is `AWS POLLY` or `LOCAL FALLBACK` |
| `voice_audio` | `audio` (base64 MP3), `format`, `text` | Polly output, phone only |
| `mission_summary` | `summary`, `recording`, `recordings`, `bedrockAvailable` | end-of-mission statistics + caveats |
| `scan_summary` | `available`, `text`, `modelId`, `reason` | optional Bedrock paragraph |
| `recording_list` | `recordings`, `saved` | available recordings |
| `replay_start` | `id`, `frames`, `mode`, `scenario`, `speed`, `durationMs` | replay beginning |
| `replay_status` | `state` ∈ `playing`\|`finished`\|`stopped`, `index`, `total`, `fraction` | progress |
| `status` | `mode`, `scanning`, `peers`, `clients`, `simulation`, `recording`, `replaying`, `calibration`, `aws`, `capabilities`, `stats` | broadcast on change |
| `error` | `error`, `type?`, `fatal` | non-fatal problem. `fatal` is always `false`. |
| `heartbeat` | `serverTime`, `peers` | liveness |

`detection`, `pose` and `state_snapshot` go to the `map` and `diagnostics`
roles; the `phone` produces them. `voice_audio` goes only to the phone.

---

## Grid wire format

The occupancy grid crosses sparsely: only non-zero cells, one quantised byte
each.

```json
{ "n": 240, "cell": 0.1, "halfExtent": 12,
  "idx": [28801, 28802, 29041],
  "val": [-61, 92, 118] }
```

`idx` is the flat cell index (`cy * n + cx`); `val` is the log-odds normalised to
±127. Reconstruct with `OccupancyGrid.deserialize()`. Quantisation error stays
under 0.1 log-odds (tested). A cell absent from `idx` is **unknown** — which is
not the same as free, and is never drawn as free.

---

## Versioning

`PROTOCOL_VERSION = 1`. A frame carrying a different `v` is rejected with an
`error`, so a stale phone cannot silently corrupt a newer map. Adding an
optional field is backward compatible because `normalizeDetection()` fills in
every field it does not receive; removing or repurposing one requires a version
bump.
