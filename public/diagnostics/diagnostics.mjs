/**
 * Diagnostics and self-tests.
 *
 * Written for a situation this project actually has: the phone that will be
 * demoed cannot be tested by the person writing the code.  So every subsystem
 * is probed independently, every result says what it means and what to do
 * about it, and the page ends with a single verdict naming the mode that will
 * work on this device.
 */
import { AcousticSensor } from '../phone/sensor.mjs';
import { PoseEstimator } from '../phone/pose.mjs';
import { GuidanceEngine } from '../phone/audio-guidance.mjs';
import { DetectionPipeline, SPEED_OF_SOUND, FFT_SIZE } from '../phone/dsp/pipeline.mjs';
import { WsClient } from '../shared/wsclient.mjs';
import { validateMessage, normalizeDetection, polarToWorld } from '../shared/protocol.mjs';
import { OccupancyGrid, PointCloud, ClassFuser } from '../shared/spatial.mjs';
import { reconstruct } from '../shared/reconstruct.mjs';

const $ = (id) => document.getElementById(id);
const EchoNet = (typeof globalThis !== 'undefined' && globalThis.EchoNet) || null;

const state = {
  serverStatus: null,
  micResult: null,
  orientResult: null,
  wsResult: null,
  chirpResult: null,
  loopResult: null,
  live: {},
};

let sensor = null;
let pose = null;
let guidance = null;
let net = null;

// ---------------------------------------------------------------------------
// Output log
// ---------------------------------------------------------------------------
function out(msg, kind = '') {
  const el = document.createElement('div');
  el.className = 'l ' + kind;
  el.innerHTML = '<span class="t"></span><span class="m"></span>';
  el.querySelector('.t').textContent = new Date().toLocaleTimeString([], { hour12: false });
  el.querySelector('.m').textContent = msg;
  $('out').prepend(el);
  while ($('out').childElementCount > 300) $('out').lastElementChild.remove();
}

function row(k, v, cls, note) {
  return '<div class="row' + (note ? ' stack' : '') + '"><span class="k">' + k
    + '</span><span class="v ' + (cls || '') + '">' + v + '</span>'
    + (note ? '<span class="note">' + note + '</span>' : '') + '</div>';
}

// ---------------------------------------------------------------------------
// Static reports
// ---------------------------------------------------------------------------
function renderCore() {
  const caps = AcousticSensor.capabilities();
  const ps = PoseEstimator.supported();
  const s = sensor ? sensor.status() : null;
  const m = state.micResult;
  const o = state.orientResult;

  const micVal = m ? (m.ok ? 'AVAILABLE' : 'BLOCKED') : (caps.getUserMedia ? 'NOT TESTED' : 'UNAVAILABLE');
  const micCls = m ? (m.ok ? 'ok' : 'err') : 'idle';
  const ctxVal = s && s.audioContextState ? s.audioContextState.toUpperCase()
    : (guidance && guidance.ctx ? guidance.ctx.state.toUpperCase() : 'NOT OPENED');
  const ctxCls = ctxVal === 'RUNNING' ? 'ok' : ctxVal === 'SUSPENDED' ? 'warn' : 'idle';

  $('rowsCore').innerHTML = [
    row('Microphone', micVal, micCls, m && !m.ok ? m.error : null),
    row('AudioContext', ctxVal, ctxCls,
      ctxVal === 'SUSPENDED' ? 'Suspended contexts resume on the next tap — browsers require a gesture.' : null),
    row('Speaker / chirp', state.chirpResult ? (state.chirpResult.ok ? 'EMITTED' : 'FAILED') : 'NOT TESTED',
      state.chirpResult ? (state.chirpResult.ok ? 'ok' : 'err') : 'idle',
      state.chirpResult && state.chirpResult.ok ? 'A 15 ms 17.5-22 kHz chirp was played. Most adults will not hear it; some will hear a faint tick.' : null),
    row('DeviceOrientation', o ? (o.events > 0 ? 'AVAILABLE' : 'NO EVENTS') : (ps.orientation ? 'NOT TESTED' : 'UNAVAILABLE'),
      o ? (o.events > 0 ? 'ok' : 'warn') : 'idle',
      o && o.events > 0 ? ('Heading source: ' + o.source + (o.absolute ? ' (magnetometer-referenced)' : ' (relative — no magnetic north)')) : null),
    row('DeviceMotion', o ? (o.motionEvents > 0 ? 'AVAILABLE' : 'NO EVENTS') : (ps.motion ? 'NOT TESTED' : 'UNAVAILABLE'),
      o ? (o.motionEvents > 0 ? 'ok' : 'warn') : 'idle',
      o && o.motionEvents === 0 ? 'Without motion events the position cannot be dead-reckoned; use the manual step control or simulation.' : null),
    row('Haptics', (typeof navigator !== 'undefined' && navigator.vibrate) ? 'AVAILABLE' : 'UNAVAILABLE',
      (typeof navigator !== 'undefined' && navigator.vibrate) ? 'ok' : 'warn'),
    row('Capture path', s && s.captureMode ? s.captureMode.toUpperCase() : (caps.audioWorklet ? 'AUDIOWORKLET (expected)' : 'SCRIPTPROCESSOR (fallback)'),
      'info'),
  ].join('');
}

function renderPlatform() {
  const caps = AcousticSensor.capabilities();
  const n = navigator;
  const s = sensor ? sensor.status() : null;
  const rate = s && s.sampleRate ? s.sampleRate : null;
  $('rowsPlatform').innerHTML = [
    row('Secure context', caps.secureContext ? 'YES (HTTPS)' : 'NO (plain HTTP)',
      caps.secureContext ? 'ok' : 'err',
      caps.secureContext ? null : 'Mobile Chrome blocks the microphone outside a secure context. Open the HTTPS URL the server printed, or use simulation mode.'),
    row('Sample rate', rate ? rate + ' Hz' : '48000 Hz requested', rate && Math.abs(rate - 48000) < 1 ? 'ok' : rate ? 'warn' : 'idle',
      rate && Math.abs(rate - 48000) > 1 ? 'The device forced a different rate. Ranges stay correct (the DSP uses the real rate) but the top of the 22 kHz band may be attenuated.' : null),
    row('Channels granted', s && s.channelCount ? String(s.channelCount) : '--', 'info',
      s && s.channelCount >= 2 ? 'Stereo capture is available. TDOA bearing is not implemented in this prototype; bearing is the phone boresight.' : null),
    row('Browser', shortUA(n.userAgent), 'info'),
    row('CPU cores', String(n.hardwareConcurrency || '?'), 'info'),
    row('Screen', typeof screen !== 'undefined' ? screen.width + '×' + screen.height + ' @' + (window.devicePixelRatio || 1) + 'x' : '--', 'info'),
    row('WebSocket', typeof WebSocket !== 'undefined' ? 'SUPPORTED' : 'MISSING', typeof WebSocket !== 'undefined' ? 'ok' : 'err'),
    row('SpeechSynthesis', ('speechSynthesis' in window) ? 'AVAILABLE' : 'UNAVAILABLE',
      ('speechSynthesis' in window) ? 'ok' : 'warn',
      ('speechSynthesis' in window) ? null : 'Without it, spoken guidance needs AWS Polly.'),
  ].join('');
}

function renderClassifier() {
  if (!EchoNet) {
    $('rowsClassifier').innerHTML = row('EchoNet', 'NOT LOADED', 'err',
      'Ranges and the map still work; classification does not. Check that /vendor/echonet_weights.js is served.');
    return;
  }
  const st = EchoNet.selfTest();
  const meta = EchoNet.META;
  const cm = meta.confusion_matrix;
  const recall = cm.map((r, i) => r[i] / r.reduce((a, b) => a + b, 0));
  $('rowsClassifier').innerHTML = [
    row('Weights', 'LOADED', 'ok'),
    row('Self-test', st.ok ? 'PASS' : 'FAIL', st.ok ? 'ok' : 'err',
      'Max abs error vs the PyTorch reference: ' + st.maxAbsError.toExponential(2)),
    row('Parameters', String(meta.n_params), 'info'),
    row('Input', meta.window + ' samples, peak at ' + meta.peak_index, 'info'),
    row('Real-echo accuracy', (meta.val_accuracy * 100).toFixed(0) + '% vs 50% chance', 'warn',
      meta.accuracy_basis || 'See META.accuracy_basis.'),
    row('Heads in use', (meta.classes_in_use || ['WALL', 'SOFT']).join(', '), 'info',
      meta.unused_head || 'Openings come from wall-gap geometry in reconstruct.mjs, not from the classifier.'),
    row('WALL recall (real)', (recall[0] * 100).toFixed(0) + '%', 'warn'),
    row('SOFT recall (real)', (recall[1] * 100).toFixed(0) + '%', 'warn'),
    row('Trained for', meta.f0_hz / 1000 + '-' + meta.f1_hz / 1000 + ' kHz @ ' + meta.fs_hz / 1000 + ' kHz', 'info'),
  ].join('');
}

function renderServer() {
  const st = state.serverStatus;
  if (!st) {
    $('rowsServer').innerHTML = row('Server', 'NOT REACHED', 'warn', 'Fetching /api/status…');
    return;
  }
  const aws = st.aws || {};
  const voice = aws.voice || {};
  const bed = aws.bedrock || {};
  $('rowsServer').innerHTML = [
    row('Server', 'REACHABLE', 'ok'),
    row('Mode', String(st.mode).toUpperCase(), 'info'),
    row('Peers', 'phone ' + (st.peers.phone || 0) + ' · map ' + (st.peers.map || 0), 'info'),
    row('Server classifier', st.capabilities.classifier === 'loaded' ? 'LOADED' : 'ERROR',
      st.capabilities.classifier === 'loaded' ? 'ok' : 'err'),
    row('Simulation', st.simulation && st.simulation.running ? 'RUNNING' : 'IDLE', 'info',
      st.simulation ? 'Scenario: ' + st.simulation.scenario + ' · ' + st.simulation.pulses + ' pulses emitted' : null),
    row('AWS region', aws.region || 'NOT SET', aws.region ? 'ok' : 'warn'),
    row('AWS credentials', aws.credentialsDetected ? 'DETECTED (' + aws.credentialSource + ')' : 'NOT PRESENT',
      aws.credentialsDetected ? 'ok' : 'warn'),
    row('Voice provider', voice.label || 'LOCAL FALLBACK', voice.provider === 'polly' ? 'ok' : 'warn', voice.reason),
    row('Amazon Bedrock', bed.enabled ? 'ENABLED' : 'DISABLED', bed.enabled ? 'ok' : 'idle', bed.reason),
    row('Frames in / out', (st.stats.framesIn || 0) + ' / ' + (st.stats.framesOut || 0), 'info',
      st.stats.badFrames ? st.stats.badFrames + ' malformed frames were rejected without dropping a session.' : null),
  ].join('');
}

function renderLive() {
  const s = sensor ? sensor.status() : null;
  const p = pose ? pose.status() : null;
  const g = guidance ? guidance.status() : null;
  const rows = [];
  if (s) {
    rows.push(row('Scan', s.running ? 'ACTIVE ' + s.rateHz + ' Hz' : 'IDLE', s.running ? 'ok' : 'idle'));
    rows.push(row('Pulses / detections', s.pulses + ' / ' + s.detections, 'info',
      s.pulses ? 'Detection rate ' + (s.detectionRate * 100).toFixed(0) + '%' : null));
    rows.push(row('DSP time', s.dspMsAvg + ' ms avg · ' + s.dspMsMax + ' ms max',
      s.dspMsAvg < 12 ? 'ok' : s.dspMsAvg < 25 ? 'warn' : 'err',
      s.dspMsAvg > 25 ? 'The DSP is taking long enough to affect the pulse cadence on this device.' : null));
    rows.push(row('Clutter baseline', s.clutterReady ? 'READY' : 'WARMING UP', s.clutterReady ? 'ok' : 'warn'));
    rows.push(row('Noise floor', s.noiseFloor ? s.noiseFloor.toExponential(2) : '--', 'info'));
    rows.push(row('Bearing source', s.bearingSource, 'info'));
  }
  if (p) {
    rows.push(row('Heading', Math.round(p.pose.heading) + '° (' + p.headingSource + ')',
      p.headingAbsolute ? 'ok' : 'warn'));
    rows.push(row('Pose', p.pose.method + ' · ' + (p.pose.confidence * 100).toFixed(0) + '% confidence', 'info', p.note));
    rows.push(row('Steps / distance', p.steps + ' / ' + p.distance.toFixed(2) + ' m', 'info'));
    rows.push(row('Tilt', 'pitch ' + p.pitch + '° · roll ' + p.roll + '°', 'info'));
  }
  if (g) {
    rows.push(row('Guidance tone', g.enabled ? g.rateHz + ' Hz @ ' + g.pitchHz + ' Hz' : 'OFF', g.enabled ? 'ok' : 'idle'));
    rows.push(row('Voice last used', g.voice.label, g.voice.lastUsed === 'polly' ? 'ok' : 'warn', g.voice.reason));
  }
  if (!rows.length) rows.push(row('Live readings', 'RUN A TEST TO POPULATE', 'idle'));
  $('rowsLive').innerHTML = rows.join('');
}

function shortUA(ua) {
  if (!ua) return 'unknown';
  const m = ua.match(/(Chrome|CriOS|Firefox|Safari|Edg)\/([\d.]+)/);
  const plat = /Android/i.test(ua) ? 'Android' : /iPhone|iPad/i.test(ua) ? 'iOS' : /Windows/i.test(ua) ? 'Windows' : /Mac/i.test(ua) ? 'macOS' : /Linux/i.test(ua) ? 'Linux' : '';
  const model = ua.match(/;\s*([A-Z]{2}\d{4}|[A-Za-z0-9_-]+)\s+Build/);
  return (plat + ' · ' + (m ? m[1] + ' ' + m[2].split('.')[0] : 'unknown') + (model ? ' · ' + model[1] : '')).trim();
}

function renderAll() {
  renderCore();
  renderPlatform();
  renderClassifier();
  renderServer();
  renderLive();
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
$('btnChirp').addEventListener('click', async () => {
  out('TEST CHIRP', 'head');
  if (!sensor) sensor = makeSensor();
  const res = await sensor.testChirp();
  state.chirpResult = res;
  if (res.ok) out('Chirp emitted (AudioContext ' + res.state + ' @ ' + res.sampleRate + ' Hz). If you heard a faint tick, the speaker reaches the band.', 'pass');
  else out('Chirp failed: ' + res.error, 'fail');
  renderAll();
});

$('btnMic').addEventListener('click', async () => {
  out('TEST MICROPHONE', 'head');
  if (!sensor) sensor = makeSensor();
  const res = await sensor.start({ rateHz: 20 });
  state.micResult = res;
  if (!res.ok) {
    out('Microphone unavailable: ' + res.error, 'fail');
    out('Simulation mode does not need the microphone — the map behaves identically.', 'warn');
    renderAll();
    return;
  }
  out('Microphone granted via ' + res.captureMode + ' at ' + res.sampleRate + ' Hz, ' + res.channelCount + ' channel(s).', 'pass');
  const s = res.settings || {};
  if (s.echoCancellation || s.noiseSuppression || s.autoGainControl) {
    out('WARNING: the device kept mic processing on (ec=' + s.echoCancellation + ' ns=' + s.noiseSuppression + ' agc=' + s.autoGainControl + '). Echo amplitudes will be unreliable.', 'warn');
  } else {
    out('Raw capture confirmed: echo cancellation, noise suppression and AGC are all off.', 'pass');
  }
  out('Listening for 6 s — point the phone at a wall about 1-2 m away.', '');
  setTimeout(async () => {
    const st = sensor.status();
    out('Result: ' + st.detections + ' detections from ' + st.pulses + ' pulses ('
      + (st.detectionRate * 100).toFixed(0) + '%), DSP ' + st.dspMsAvg + ' ms avg.',
      st.detections > 5 ? 'pass' : 'warn');
    if (st.detections === 0) {
      out('No echoes detected. Usual causes: the phone speaker rolls off before 17.5 kHz, the volume is low, or the room is anechoic at these frequencies. Simulation mode is the fallback.', 'warn');
    }
    await sensor.stop();
    renderAll();
  }, 6000);
  renderAll();
});

$('btnOrient').addEventListener('click', async () => {
  out('TEST ORIENTATION', 'head');
  if (!pose) pose = new PoseEstimator({ onUpdate: () => {} });
  const perm = await pose.requestPermission();
  out('Permission — orientation: ' + perm.orientation + ', motion: ' + perm.motion,
    perm.orientation === 'granted' ? 'pass' : 'warn');
  pose.start();

  let oEvents = 0;
  let mEvents = 0;
  const onO = () => { oEvents++; };
  const onM = () => { mEvents++; };
  window.addEventListener('deviceorientation', onO, true);
  window.addEventListener('devicemotion', onM, true);
  out('Rotate and gently shake the phone for 5 s…', '');

  setTimeout(() => {
    window.removeEventListener('deviceorientation', onO, true);
    window.removeEventListener('devicemotion', onM, true);
    const st = pose.status();
    state.orientResult = {
      events: oEvents, motionEvents: mEvents,
      source: st.headingSource, absolute: st.headingAbsolute,
    };
    out('Orientation events: ' + oEvents + ' · motion events: ' + mEvents, oEvents > 0 ? 'pass' : 'fail');
    out('Heading ' + Math.round(st.pose.heading) + '° from ' + st.headingSource
      + (st.headingAbsolute ? ' (magnetometer-referenced)' : ' (relative only)'),
      st.headingAbsolute ? 'pass' : 'warn');
    out('Tilt-compensated off the ' + (st.aimAxis === 'aim' ? 'back of the phone (aimed)' : 'top edge (held flat)')
      + ' · pitch ' + st.pitch + '°, roll ' + st.roll + '°', st.aimAxis ? 'pass' : 'warn');
    out('Gyro: ' + (st.available.gyro ? st.rotRate + '°/s (step rejection active)'
      : 'unavailable — hand sweeps cannot be told from footfalls'),
      st.available.gyro ? 'pass' : 'warn');
    if (mEvents > 0) {
      out('Step-shaped peaks seen: ' + st.stepCandidates + ' · accepted as walking: ' + st.steps
        + (st.stepCandidates > 0 && st.steps === 0
          ? ' (correct: shaking in place is not a gait)' : ''), 'pass');
    }
    else out('No motion events: dead reckoning is unavailable. Use the manual step control or simulation.', 'warn');
    renderAll();
  }, 5000);
});

$('btnWs').addEventListener('click', () => {
  out('TEST WEBSOCKET', 'head');
  const started = Date.now();
  const probe = new WsClient({
    role: 'diagnostics',
    onMessage: (msg) => {
      if (msg.type === 'welcome') {
        const ms = Date.now() - started;
        out('Connected in ' + ms + ' ms — session ' + msg.sessionId + ', protocol v' + msg.protocolVersion + '.', 'pass');
        out('Server capabilities: classifier=' + msg.capabilities.classifier
          + ', polly=' + msg.capabilities.polly + ', bedrock=' + msg.capabilities.bedrock
          + ', scenarios=' + (msg.scenarios || []).length, '');
        state.wsResult = { ok: true, ms, sessionId: msg.sessionId };
        // Prove the error path: a malformed frame must not close the socket.
        probe.ws.send('{ not json');
      }
      if (msg.type === 'error') {
        out('Server rejected a deliberately malformed frame ("' + msg.error + '") and kept the socket open — as designed.', 'pass');
        setTimeout(() => {
          out('Socket state after the bad frame: ' + (probe.isOpen() ? 'OPEN' : 'CLOSED'), probe.isOpen() ? 'pass' : 'fail');
          probe.close();
          renderAll();
        }, 250);
      }
    },
    onState: (s) => out('WebSocket state: ' + s, s === 'connected' ? 'pass' : ''),
  });
  probe.connect();
  setTimeout(() => {
    if (!state.wsResult) {
      out('No welcome frame within 4 s — the server may not be reachable from this device.', 'fail');
      state.wsResult = { ok: false };
      probe.close();
      renderAll();
    }
  }, 4000);
});

/**
 * Pure self-tests: the DSP, geometry, occupancy, fusion, reconstruction and
 * protocol paths, all verified in-browser with synthetic data.  No microphone,
 * no network, no server — so this runs anywhere and tells you whether the
 * build itself is sound.
 */
$('btnSelfTest').addEventListener('click', () => {
  out('RUN SELF TEST', 'head');
  let pass = 0;
  let fail = 0;
  const check = (name, cond, detail) => {
    if (cond) { pass++; out('PASS  ' + name + (detail ? ' — ' + detail : ''), 'pass'); }
    else { fail++; out('FAIL  ' + name + (detail ? ' — ' + detail : ''), 'fail'); }
  };

  // --- classifier
  if (EchoNet) {
    const st = EchoNet.selfTest();
    check('EchoNet matches its PyTorch reference', st.ok, 'max abs error ' + st.maxAbsError.toExponential(2));
    const flat = new Float32Array(64).fill(0.5);
    const o = EchoNet.forward(flat);
    const sum = o.probs.reduce((a, b) => a + b, 0);
    check('EchoNet output is a probability distribution', Math.abs(sum - 1) < 1e-4, 'sum = ' + sum.toFixed(6));
  } else {
    check('EchoNet loaded', false, 'globalThis.EchoNet is missing');
  }

  // --- range conversion
  const p = new DetectionPipeline(48000);
  const idx = Math.round((2 * 1.82 / SPEED_OF_SOUND) * 48000);
  const back = (idx / 48000) * SPEED_OF_SOUND * 0.5;
  check('range <-> sample conversion round-trips', Math.abs(back - 1.82) < 0.005,
    '1.82 m -> ' + idx + ' samples -> ' + back.toFixed(4) + ' m');
  check('direct-path gate is +-2 ms', p.gateSamples === 96, p.gateSamples + ' samples');
  // The gate is 2 ms = 96 samples = 0.343 m, which is further out than the
  // 0.30 m search floor, so the gate is what sets the true minimum range.
  const rLo = (p.nLo / 48000) * SPEED_OF_SOUND / 2;
  const rHi = (p.nHi / 48000) * SPEED_OF_SOUND / 2;
  check('CFAR search starts past the direct-path gate and reaches 3.75 m',
    p.nLo > p.gateSamples && rLo >= 0.30 && rLo < 0.40 && Math.abs(rHi - 3.75) < 0.02,
    'minimum range ' + rLo.toFixed(3) + ' m (gate-limited), maximum ' + rHi.toFixed(2) + ' m');
  check('chirp is 15 ms of Hann-windowed LFM at 0.6 FS',
    p.chirp.length === 720 && Math.abs(Math.max.apply(null, Array.from(p.chirp)) - 0.6) < 0.01);

  // --- live DSP on synthetic audio
  const synth = synthRx(p, 1.5, 5e-3);
  let res = null;
  for (let i = 0; i < 14; i++) res = p.process(synthRx(p, 0, 0), { t: i * 50 });
  for (let i = 0; i < 6; i++) res = p.process(synth, { t: 700 + i * 50, classifier: EchoNet });
  check('pipeline recovers a synthetic 1.50 m echo',
    !!res.detection && Math.abs(res.detection.range_m - 1.5) < 0.08,
    res.detection ? res.detection.range_m.toFixed(3) + ' m, SNR ' + res.detection.snr_db.toFixed(1) + ' dB'
      : 'no detection (' + res.diagnostics.reason + ')');
  if (res.detection) {
    check('classifier window is 64 samples normalised to 1',
      res.diagnostics.window && res.diagnostics.window.length === 64
      && Math.abs(Math.max.apply(null, Array.from(res.diagnostics.window)) - 1) < 1e-6);
  }

  // --- coordinate conversion
  const w = polarToWorld({ x: 0, y: 0, heading: 90 }, 90, 2);
  check('polar -> world: bearing 90 deg is +2 m east', Math.abs(w.x - 2) < 1e-6 && Math.abs(w.y) < 1e-6,
    '(' + w.x.toFixed(3) + ', ' + w.y.toFixed(3) + ')');
  const w2 = polarToWorld({ x: 1, y: 1, heading: 0 }, 180, 1);
  check('polar -> world: bearing 180 deg is 1 m south', Math.abs(w2.x - 1) < 1e-6 && Math.abs(w2.y) < 1e-6,
    '(' + w2.x.toFixed(3) + ', ' + w2.y.toFixed(3) + ')');

  // --- occupancy grid
  const g = new OccupancyGrid({ cell: 0.1, halfExtent: 6 });
  g.integrate({ x: 0, y: 0, heading: 0 }, 0, 2.0, { weight: 1, className: 'WALL', beamwidth_deg: 30 });
  const occ = g.toCell(0, 2.0);
  const free = g.toCell(0, 1.0);
  check('occupancy marks the reflector cell occupied', g.get(occ.cx, occ.cy) > 0, 'log-odds ' + g.get(occ.cx, occ.cy).toFixed(2));
  check('occupancy marks the traversed beam free', g.get(free.cx, free.cy) < 0, 'log-odds ' + g.get(free.cx, free.cy).toFixed(2));
  const unknown = g.toCell(4, -4);
  check('unscanned space stays unknown, not free', g.get(unknown.cx, unknown.cy) === 0);
  const ser = g.serialize();
  const de = OccupancyGrid.deserialize(ser);
  check('occupancy grid survives serialise/deserialise',
    Math.abs(de.get(occ.cx, occ.cy) - g.get(occ.cx, occ.cy)) < 0.1, ser.idx.length + ' non-zero cells');

  // --- point cloud consolidation
  const cloud = new PointCloud();
  for (let i = 0; i < 5; i++) {
    cloud.add(normalizeDetection({
      range_m: 2, bearing_deg: 0, confidence: 0.8, obstacleClass: 'WALL',
      classConfidence: 0.9, classProbs: [0.9, 0.06, 0.04],
      phone: { x: 0.01 * i, y: 0, heading: 0 },
    }));
  }
  check('repeat observations consolidate instead of stacking', cloud.points.length === 1,
    cloud.points.length + ' point(s), ' + cloud.points[0].hits + ' hits');

  // --- temporal fusion
  const fus = new ClassFuser();
  let f = null;
  for (let i = 0; i < 4; i++) {
    f = fus.fuse(normalizeDetection({
      t: Date.now() + i * 50, range_m: 2, bearing_deg: 0, confidence: 0.8,
      classProbs: [0.7, 0.2, 0.1], phone: { x: 0, y: 0, heading: 0 },
    }));
  }
  check('fusion accumulates support and picks the consistent class',
    f.className === 'WALL' && f.support >= 4 && f.stable, 'support ' + f.support + ', conf ' + f.confidence.toFixed(3));
  const fus2 = new ClassFuser();
  let g2 = null;
  const alt = [[0.6, 0.3, 0.1], [0.1, 0.8, 0.1], [0.6, 0.3, 0.1], [0.1, 0.8, 0.1]];
  for (let i = 0; i < 4; i++) {
    g2 = fus2.fuse(normalizeDetection({
      t: Date.now() + i * 50, range_m: 2, bearing_deg: 0, confidence: 0.8,
      classProbs: alt[i], phone: { x: 0, y: 0, heading: 0 },
    }));
  }
  check('fusion reports disagreement as unstable', g2.stable === false, 'alternating predictions -> stable=false');

  // --- reconstruction
  const pts = [];
  for (let i = 0; i <= 24; i++) pts.push({ x: -1.2 + i * 0.1, y: 2, weight: 1.5, hits: 3, className: 'WALL', classConfidence: 0.9 });
  const rec = reconstruct(pts);
  check('reconstruction fits a straight wall to collinear echoes',
    rec.segments.length === 1 && rec.segments[0].length > 2.0,
    rec.segments.length + ' segment(s), ' + (rec.segments[0] ? rec.segments[0].length.toFixed(2) + ' m, rms '
      + rec.segments[0].rms.toFixed(3) : ''));
  // A door-width hole in a wall should surface as an opening candidate.
  const gapPts = pts.filter((q) => !(q.x > -0.35 && q.x < 0.45));
  const rec2 = reconstruct(gapPts);
  check('reconstruction proposes an opening where a wall has a door-width gap',
    (rec2.openings || []).some((o) => o.evidence === 'geometric-gap'),
    (rec2.openings || []).length + ' candidate(s)');
  // An L of points must not be fitted as one straight wall.
  const corner = [];
  for (let i = 0; i <= 14; i++) corner.push({ x: 0, y: i * 0.12, weight: 1.5, hits: 3, className: 'WALL', classConfidence: 0.9 });
  for (let i = 1; i <= 14; i++) corner.push({ x: i * 0.12, y: 1.68, weight: 1.5, hits: 3, className: 'WALL', classConfidence: 0.9 });
  const rec3 = reconstruct(corner);
  check('reconstruction splits a corner into two surfaces', rec3.segments.length >= 2,
    rec3.segments.length + ' segment(s), ' + rec3.corners.length + ' corner(s)');

  // --- protocol validation
  check('protocol rejects malformed JSON', validateMessage('{oops').ok === false);
  check('protocol rejects unknown message types', validateMessage({ type: 'nope' }).ok === false);
  check('protocol rejects a version mismatch', validateMessage({ v: 99, type: 'hello' }).ok === false);
  check('protocol rejects a detection with no usable range',
    validateMessage({ type: 'detection', detection: { range_m: 'x' } }).ok === false);
  const good = validateMessage({
    type: 'detection',
    detection: { range_m: 1.5, bearing_deg: 45, classProbs: [0.5, 0.3, 0.2], phone: { x: 0, y: 0, heading: 45 } },
  });
  check('protocol accepts and normalises a valid detection',
    good.ok && good.msg.detection.obstacleClass === 'WALL' && Number.isFinite(good.msg.detection.worldX),
    good.ok ? 'world (' + good.msg.detection.worldX.toFixed(2) + ', ' + good.msg.detection.worldY.toFixed(2) + ')' : '');
  const repaired = normalizeDetection({ range_m: 2, vel_mps: 999, confidence: 5, classProbs: [2, 0, 0] });
  check('protocol clamps out-of-range fields rather than throwing',
    repaired.vel_mps <= 5 && repaired.confidence <= 1 && Math.abs(repaired.classProbs[0] - 1) < 1e-6);

  out('SELF TEST: ' + pass + ' passed, ' + fail + ' failed.', fail ? 'fail' : 'pass');
  setVerdict(fail === 0
    ? { cls: 'ok', title: 'Self test passed', text: pass + ' internal checks passed. The DSP, geometry, fusion, reconstruction and protocol layers are all behaving. Device-specific results still depend on the microphone and orientation tests above.' }
    : { cls: 'err', title: 'Self test found problems', text: fail + ' of ' + (pass + fail) + ' checks failed. See the test output below — a failure here is a build problem, not a device problem.' });
  renderAll();
});

/** Full loop: server, simulation, detections, fusion and reconstruction. */
$('btnLoop').addEventListener('click', () => {
  out('TEST FULL LOOP (simulation through to reconstruction)', 'head');
  let dets = 0;
  let snaps = 0;
  let lastSnap = null;
  const probe = new WsClient({
    role: 'diagnostics',
    onMessage: (msg) => {
      if (msg.type === 'welcome') {
        out('Linked. Starting a simulated mission in the "room" scenario…', '');
        probe.send('set_mode', { mode: 'simulation' });
        probe.send('mission_start', { scenario: 'room', rateHz: 20, record: false });
      }
      if (msg.type === 'detection') {
        dets++;
        if (dets === 1) {
          const d = msg.detection;
          out('First detection: ' + d.range_m.toFixed(2) + ' m, bearing ' + Math.round(d.bearing_deg)
            + '°, class ' + d.obstacleClass + ' ' + (d.classConfidence * 100).toFixed(0) + '%, SNR '
            + d.snr_db.toFixed(1) + ' dB, source ' + d.source, 'pass');
        }
      }
      if (msg.type === 'state_snapshot') { snaps++; lastSnap = msg; }
    },
    onState: () => {},
  });
  probe.connect();

  setTimeout(() => {
    probe.send('mission_complete', {});
  }, 5000);

  setTimeout(() => {
    out('Detections received: ' + dets, dets > 30 ? 'pass' : dets > 0 ? 'warn' : 'fail');
    out('State snapshots received: ' + snaps, snaps > 1 ? 'pass' : 'warn');
    if (lastSnap) {
      out('Map state: ' + lastSnap.cloud.length + ' echo points, ' + lastSnap.grid.idx.length
        + ' grid cells, ' + lastSnap.trajectory.nodes.length + ' path nodes, '
        + (lastSnap.stats.distanceScanned || 0).toFixed(2) + ' m scanned.',
        lastSnap.cloud.length > 5 ? 'pass' : 'warn');
      const r = lastSnap.reconstruction;
      if (r) {
        out('Reconstruction: ' + r.segments.length + ' surfaces, ' + r.corners.length + ' corners, '
          + (r.openings || []).length + ' opening candidates, confidence ' + Math.round(r.confidence * 100) + '%.',
          r.segments.length > 0 ? 'pass' : 'warn');
      }
      state.loopResult = { dets, snaps, ok: dets > 30 };
    }
    out(dets > 30
      ? 'Full loop works end to end with no phone hardware. This is the fallback that guarantees a demo.'
      : 'The loop did not produce the expected detection rate — check the server console.',
      dets > 30 ? 'pass' : 'fail');
    probe.close();
    refreshServer();
  }, 6500);
});

function setVerdict(v) {
  const el = $('verdict');
  el.hidden = false;
  el.className = 'verdict ' + v.cls;
  el.innerHTML = '<b>' + v.title + '</b>' + v.text;
}

function makeSensor() {
  return new AcousticSensor({
    classifier: EchoNet,
    onLog: (m, l) => out(m, l === 'error' ? 'fail' : l === 'success' ? 'pass' : l === 'warn' ? 'warn' : ''),
    onDetection: (det, diag) => {
      state.live.det = det;
      state.live.diag = diag;
    },
  });
}

/** Synthetic RX window: leakage plus one echo, for the in-browser self-test. */
function synthRx(pipeline, rangeM, amp) {
  const rx = new Float32Array(FFT_SIZE);
  const chirp = pipeline.chirp;
  const leakAt = 200;
  for (let i = 0; i < chirp.length; i++) rx[leakAt + i] += chirp[i] * (0.08 / 0.6);
  if (rangeM > 0 && amp > 0) {
    const d = Math.round((2 * rangeM / SPEED_OF_SOUND) * 48000);
    for (let i = 0; i < chirp.length; i++) {
      const j = leakAt + d + i;
      if (j < FFT_SIZE) rx[j] += chirp[i] * (amp / 0.6);
    }
  }
  for (let i = 0; i < FFT_SIZE; i++) rx[i] += (Math.random() * 2 - 1) * 3e-4;
  return rx;
}

// ---------------------------------------------------------------------------
// A-scope: the live range profile
// ---------------------------------------------------------------------------
const scope = $('scope');
const sctx = scope.getContext('2d');
function drawScope() {
  const rect = scope.getBoundingClientRect();
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  if (scope.width !== Math.round(rect.width * dpr)) {
    scope.width = Math.round(rect.width * dpr);
    scope.height = Math.round(rect.height * dpr);
  }
  const w = rect.width;
  const h = rect.height;
  sctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  sctx.clearRect(0, 0, w, h);

  // Range graticule every 0.5 m out to 4 m.
  sctx.strokeStyle = 'rgba(86,130,170,0.12)';
  sctx.lineWidth = 1;
  sctx.font = '8px ui-monospace, monospace';
  sctx.fillStyle = 'rgba(85,103,126,0.8)';
  for (let m = 0; m <= 4; m += 0.5) {
    const x = (m / 4) * w;
    sctx.beginPath();
    sctx.moveTo(x, 0);
    sctx.lineTo(x, h - 11);
    sctx.stroke();
    if (m % 1 === 0) sctx.fillText(m + 'm', x + 2, h - 2);
  }

  const prof = sensor && sensor.running ? sensor.profile(240, 4.0) : null;
  if (!prof) {
    sctx.fillStyle = 'rgba(85,103,126,0.6)';
    sctx.fillText('run TEST MICROPHONE to see the live profile', 8, h / 2);
    $('scopeMeta').textContent = 'idle';
    return;
  }

  let mx = 1e-9;
  for (const v of prof) if (v > mx) mx = v;
  sctx.beginPath();
  for (let i = 0; i < prof.length; i++) {
    const x = (i / (prof.length - 1)) * w;
    const y = (h - 13) * (1 - prof[i] / mx);
    if (i === 0) sctx.moveTo(x, y);
    else sctx.lineTo(x, y);
  }
  sctx.strokeStyle = 'rgba(44,232,245,0.85)';
  sctx.lineWidth = 1.3;
  sctx.stroke();

  const d = state.live.det;
  if (d) {
    const x = (Math.min(4, d.range_m) / 4) * w;
    sctx.strokeStyle = 'rgba(255,180,84,0.8)';
    sctx.beginPath();
    sctx.moveTo(x, 0);
    sctx.lineTo(x, h - 11);
    sctx.stroke();
    $('scopeMeta').textContent = d.range_m.toFixed(2) + ' m · ' + d.snr_db.toFixed(0) + ' dB · '
      + (d.obstacleClass || '--') + ' ' + Math.round((d.classConfidence || 0) * 100) + '%';
  } else {
    $('scopeMeta').textContent = 'no detection · peak ' + mx.toExponential(1);
  }
}

// ---------------------------------------------------------------------------
// Server status
// ---------------------------------------------------------------------------
async function refreshServer() {
  try {
    const r = await fetch('/api/status', { cache: 'no-store' });
    state.serverStatus = await r.json();
  } catch (e) {
    state.serverStatus = null;
    out('Could not reach /api/status: ' + e.message, 'warn');
  }
  renderServer();
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
pose = new PoseEstimator({ onUpdate: () => {} });
guidance = new GuidanceEngine();
renderAll();
refreshServer();
setInterval(() => { renderLive(); drawScope(); }, 400);
setInterval(refreshServer, 6000);

const caps = AcousticSensor.capabilities();
out('Diagnostics ready on ' + shortUA(navigator.userAgent) + '.', 'head');
if (!caps.liveAudioPossible) {
  out('Live audio will not work here: ' + caps.reason, 'warn');
  setVerdict({
    cls: 'warn',
    title: 'Live audio unavailable on this device/URL',
    text: caps.reason + '. Everything else works: run RUN SELF TEST to verify the build, and use SIMULATION mode for the demo — the command center behaves identically because both modes emit the same detection schema.',
  });
} else {
  out('Live audio is possible here. Run the tests above to confirm the device actually delivers it.', '');
}
