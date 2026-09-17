/**
 * SentryShield wire protocol — schema v1.
 *
 * ONE message boundary for the whole system. The phone (sensor) and the
 * command center (consumer) only ever exchange these envelopes; nothing
 * downstream of the sensor module ever sees raw audio. Live mode, simulation
 * mode and replay all emit byte-identical `detection` frames, which is why the
 * map cannot tell them apart.
 *
 * Shared verbatim by the Node server (ESM import) and both browser clients.
 */

import { surfaceFromProbs } from './surfaceclass.mjs';

export const PROTOCOL_VERSION = 1;

export const ROLES = ['phone', 'map', 'diagnostics'];
export const MODES = ['live', 'simulation', 'hybrid', 'replay'];
/**
 * The class vocabulary accepted on the wire. OPENING stays in the list so
 * recordings captured before openings moved to the geometry layer still
 * replay, but nothing in the live path derives it any more — see
 * shared/surfaceclass.mjs.
 */
export const CLASSES = ['WALL', 'SOFT', 'OPENING'];

/** Every message type on the wire, with its direction and purpose. */
export const MESSAGE_TYPES = {
  // ---- client -> server
  hello:            { from: 'client', doc: 'role/mode handshake; server answers with `welcome`' },
  detection:        { from: 'phone',  doc: 'one acoustic detection in polar + world coords' },
  pose:             { from: 'phone',  doc: 'pose-only update (heading/motion) between detections' },
  calibration:      { from: 'phone',  doc: 'range correction factor measured against a known wall' },
  heartbeat:        { from: 'both',   doc: 'liveness + round-trip latency probe' },
  scan_start:       { from: 'client', doc: 'sensor started emitting chirps' },
  scan_stop:        { from: 'client', doc: 'sensor stopped' },
  mission_start:    { from: 'client', doc: 'begin a mission: clears map state, starts stats' },
  mission_complete: { from: 'both',   doc: 'end a mission; server answers with the summary' },
  set_mode:         { from: 'client', doc: 'switch live / simulation / hybrid' },
  sim_control:      { from: 'client', doc: 'drive the digital twin (scenario, speed, steer, pause)' },
  record_start:     { from: 'client', doc: 'begin capturing frames to a recording' },
  record_stop:      { from: 'client', doc: 'close the recording and persist it' },
  replay_start:     { from: 'client', doc: 'replay a stored recording through the live pipeline' },
  replay_stop:      { from: 'client', doc: 'abort a replay' },
  request_state:    { from: 'client', doc: 'ask for a full map snapshot (used after reconnect)' },
  request_summary:  { from: 'client', doc: 'ask the AWS adapter for a post-scan interpretation' },
  speak:            { from: 'client', doc: 'ask the server to synthesise guidance speech' },
  reconstruct:      { from: 'client', doc: 'force a surface-reconstruction pass and broadcast it' },
  save_training_dataset: { from: 'client', doc: 'stream collected real acoustic envelope windows for neural net training' },

  // ---- server -> client
  welcome:          { from: 'server', doc: 'session id, protocol version, server capabilities' },
  state_snapshot:   { from: 'server', doc: 'complete fused map state for (re)synchronisation' },
  guidance:         { from: 'server', doc: 'guidance text + voice provider actually used' },
  voice_audio:      { from: 'server', doc: 'base64 MP3 from Amazon Polly' },
  mission_summary:  { from: 'server', doc: 'end-of-mission statistics' },
  scan_summary:     { from: 'server', doc: 'optional Amazon Bedrock natural-language interpretation' },
  recording_list:   { from: 'server', doc: 'available recordings' },
  replay_status:    { from: 'server', doc: 'replay progress / finished' },
  training_dataset_saved: { from: 'server', doc: 'acknowledgement with saved dataset filename and sample count' },
  status:           { from: 'server', doc: 'session + peer + adapter status broadcast' },
  error:            { from: 'server', doc: 'non-fatal problem report; never closes the socket' },
};

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------
/**
 * @typedef {Object} Pose
 * @property {number} x           metres east of mission origin
 * @property {number} y           metres north of mission origin
 * @property {number} heading     compass degrees, 0 = +y
 * @property {number} confidence  0..1, decays with dead-reckoned distance
 * @property {string} method      'dead-reckoning' | 'simulated' | 'manual' | 'static'
 */

/**
 * @typedef {Object} Detection
 * @property {number}  t             capture timestamp, ms since epoch
 * @property {number}  range_m       estimated range to the reflector, metres
 * @property {number}  vel_mps       closing velocity, positive = approaching
 * @property {number}  ttc_s         range / closing velocity, clamped [0, 10]
 * @property {number}  confidence    detection (not class) confidence, 0..1
 * @property {number}  bearing_deg   absolute compass bearing of the beam boresight
 * @property {number}  beamwidth_deg angular uncertainty of that bearing
 * @property {number}  snr_db        CFAR signal-to-noise estimate
 * @property {boolean} cfar_pass     did the peak clear the CFAR threshold
 * @property {string|null} obstacleClass  EchoNet argmax: WALL | SOFT | OPENING
 * @property {number}  classConfidence   EchoNet max softmax probability
 * @property {number[]} classProbs   full [WALL, SOFT, OPENING] distribution
 * @property {string|null} fusedClass     temporally fused decision
 * @property {number}  fusedConfidence
 * @property {number}  worldX        reflector position in world frame, metres
 * @property {number}  worldY
 * @property {Pose}    phone         pose used to project this detection
 * @property {string}  source        live | simulation | hybrid | replay
 */

export function makePose(p = {}) {
  const src = p || {};
  return {
    x: num(src.x, 0),
    y: num(src.y, 0),
    heading: wrapDeg(num(src.heading, 0)),
    confidence: clamp01(num(src.confidence, 0.5)),
    method: typeof src.method === 'string' ? src.method : 'static',
  };
}

/**
 * Normalise anything detection-shaped into a full, finite, in-range Detection.
 * Malformed packets are repaired rather than thrown away — a demo must never
 * die on one bad frame. Returns null only if there is no usable range.
 */
export function normalizeDetection(raw, fallbackSource = 'live') {
  if (!raw || typeof raw !== 'object') return null;
  const range = num(raw.range_m, NaN);
  if (!Number.isFinite(range) || range <= 0 || range > 20) return null;

  const pose = makePose(raw.phone || raw.pose);
  const bearing = wrapDeg(num(raw.bearing_deg, pose.heading));
  const probs = normalizeProbs(raw.classProbs);
  // An explicit obstacleClass is honoured as sent (so archived OPENING frames
  // replay unchanged); an absent one is derived under the two-class rule.
  const cls = CLASSES.includes(raw.obstacleClass) ? raw.obstacleClass : classFromProbs(probs);
  const vel = clampRange(num(raw.vel_mps, 0), -5, 5);
  const world = polarToWorld(pose, bearing, range);

  return {
    t: num(raw.t, Date.now()),
    range_m: range,
    vel_mps: vel,
    ttc_s: Number.isFinite(raw.ttc_s) ? clampRange(raw.ttc_s, 0, 10) : timeToContact(range, vel),
    confidence: clamp01(num(raw.confidence, 0.5)),
    bearing_deg: bearing,
    beamwidth_deg: clampRange(num(raw.beamwidth_deg, 30), 4, 120),
    snr_db: clampRange(num(raw.snr_db, 0), -20, 60),
    cfar_pass: !!raw.cfar_pass,
    obstacleClass: cls,
    classConfidence: clamp01(num(raw.classConfidence, probs ? Math.max.apply(null, probs) : 0)),
    classProbs: probs,
    fusedClass: CLASSES.includes(raw.fusedClass) ? raw.fusedClass : cls,
    fusedConfidence: clamp01(num(raw.fusedConfidence, num(raw.classConfidence, 0))),
    worldX: Number.isFinite(raw.worldX) ? raw.worldX : world.x,
    worldY: Number.isFinite(raw.worldY) ? raw.worldY : world.y,
    phone: pose,
    source: MODES.includes(raw.source) ? raw.source : fallbackSource,
  };
}

/** Project a polar measurement taken at `pose` into the world frame. */
export function polarToWorld(pose, bearingDeg, rangeM) {
  const r = (wrapDeg(bearingDeg) * Math.PI) / 180;
  return { x: pose.x + rangeM * Math.sin(r), y: pose.y + rangeM * Math.cos(r) };
}

export function timeToContact(rangeM, velMps) {
  return clampRange(rangeM / Math.max(velMps, 0.05), 0, 10);
}

// ---------------------------------------------------------------------------
// Envelope validation
// ---------------------------------------------------------------------------
/**
 * Validate an inbound envelope. Returns { ok, type, msg } or { ok:false, error }.
 * Never throws: a malformed frame is reported over the `error` message type
 * instead of tearing down the session.
 */
export function validateMessage(input) {
  let msg = input;
  if (typeof input === 'string') {
    try { msg = JSON.parse(input); } catch (e) { return { ok: false, error: 'malformed JSON' }; }
  }
  if (!msg || typeof msg !== 'object' || Array.isArray(msg)) return { ok: false, error: 'not an object' };
  if (typeof msg.type !== 'string') return { ok: false, error: 'missing type' };
  if (!Object.prototype.hasOwnProperty.call(MESSAGE_TYPES, msg.type)) {
    return { ok: false, error: 'unknown type "' + msg.type + '"' };
  }
  if (msg.v != null && msg.v !== PROTOCOL_VERSION) {
    return { ok: false, error: 'protocol version ' + msg.v + ' != ' + PROTOCOL_VERSION };
  }
  if (msg.type === 'hello' && msg.role && !ROLES.includes(msg.role)) {
    return { ok: false, error: 'unknown role "' + msg.role + '"' };
  }
  if (msg.type === 'set_mode' && !MODES.includes(msg.mode)) {
    return { ok: false, error: 'unknown mode "' + msg.mode + '"' };
  }
  if (msg.type === 'detection') {
    const det = normalizeDetection(msg.detection || msg, 'live');
    if (!det) return { ok: false, error: 'detection has no usable range_m' };
    return { ok: true, type: msg.type, msg: Object.assign({}, msg, { detection: det }) };
  }
  return { ok: true, type: msg.type, msg };
}

export function envelope(type, payload) {
  return Object.assign({ v: PROTOCOL_VERSION, type: type, t: Date.now() }, payload || {});
}

// ---------------------------------------------------------------------------
// small numeric helpers (shared so client and server round identically)
// ---------------------------------------------------------------------------
export function num(v, d) { return typeof v === 'number' && Number.isFinite(v) ? v : d; }
export function clamp01(v) { return v < 0 ? 0 : v > 1 ? 1 : v; }
export function clampRange(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }
export function wrapDeg(d) { const x = d % 360; return x < 0 ? x + 360 : x; }
/** Signed smallest angle from a to b, in degrees, in [-180, 180). An exact
 *  half turn returns -180; the sign is arbitrary there by construction. */
export function angleDelta(a, b) { return (wrapDeg(b) - wrapDeg(a) + 540) % 360 - 180; }

function normalizeProbs(p) {
  if (!p || (!Array.isArray(p) && !(p instanceof Float32Array))) return [0, 0, 0];
  const out = [0, 0, 0];
  let s = 0;
  for (let i = 0; i < 3; i++) { out[i] = Math.max(0, num(p[i], 0)); s += out[i]; }
  if (s > 0) for (let i = 0; i < 3; i++) out[i] /= s;
  return out;
}

/**
 * Derive a class when the sender did not name one.
 *
 * This deliberately does NOT take the three-way argmax. Doing so let the
 * discarded OPENING head back in through the side door: a sender that had
 * already declined to call a class would have one invented for it here.
 */
function classFromProbs(p) {
  return surfaceFromProbs(p).className;
}
