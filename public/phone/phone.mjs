/**
 * Phone sensor application.
 *
 * Wires the sensor, pose estimator, guidance engine, calibrator and the
 * WebSocket uplink together, and renders the sonar.  Every failure here is
 * non-fatal by construction: a denied microphone, an absent magnetometer, a
 * dropped socket or a missing classifier each degrade one capability and leave
 * the rest of the instrument working.
 */
import { AcousticSensor } from './sensor.mjs';
import { PoseEstimator } from './pose.mjs';
import { GuidanceEngine, signedOffset } from './audio-guidance.mjs';
import { Calibrator } from './calibration.mjs';
import { WsClient } from '../shared/wsclient.mjs';
import { normalizeDetection, polarToWorld, wrapDeg, CLASSES } from '../shared/protocol.mjs';
import { ClassFuser } from '../shared/spatial.mjs';

const $ = (id) => document.getElementById(id);

const ui = {
  modeChip: $('modeChip'), connChip: $('connChip'), connText: $('connText'),
  scanState: $('scanState'), latencyText: $('latencyText'),
  rangeValue: $('rangeValue'), rangeSub: $('rangeSub'),
  headingValue: $('headingValue'), headingSub: $('headingSub'),
  classValue: $('classValue'), classSub: $('classSub'),
  confText: $('confText'), confFill: $('confFill'), probRows: $('probRows'),
  sonar: $('sonar'), sweepTag: $('sweepTag'),
  btnScan: $('btnScan'), btnScanText: $('btnScanText'), btnMission: $('btnMission'),
  btnCalibrate: $('btnCalibrate'), btnAudio: $('btnAudio'), btnVoice: $('btnVoice'), btnMode: $('btnMode'),
  poseMethod: $('poseMethod'), poseConf: $('poseConf'), poseNote: $('poseNote'),
  btnTurnL: $('btnTurnL'), btnTurnR: $('btnTurnR'), btnWalk: $('btnWalk'),
  voiceLabel: $('voiceLabel'), cueText: $('cueText'),
  log: $('log'), logCount: $('logCount'), logPanel: $('logPanel'), logToggle: $('logToggle'),
  gateSheet: $('gateSheet'), gateGo: $('gateGo'), gateSkip: $('gateSkip'), gateCaps: $('gateCaps'),
  calSheet: $('calSheet'), calStart: $('calStart'), calCancel: $('calCancel'),
  calDist: $('calDist'), calChoices: $('calChoices'), calFill: $('calFill'), calStatus: $('calStatus'),
};

const state = {
  mode: 'simulation',
  missionActive: false,
  scanning: false,
  lastDetection: null,
  lastDetectionAt: 0,
  audioOn: true,
  voiceOn: true,
  sweepAngle: 0,
  trail: [],                 // recent detections for the sonar's persistence
  logLines: 0,
  classifierReady: false,
  pollyLabel: 'LOCAL FALLBACK',
};

// EchoNet is loaded by a classic <script> tag, which sets globalThis.EchoNet.
const EchoNet = (typeof globalThis !== 'undefined' && globalThis.EchoNet) || null;

// ---------------------------------------------------------------------------
// Logging
// ---------------------------------------------------------------------------
function log(msg, level = 'info') {
  state.logLines++;
  ui.logCount.textContent = String(state.logLines);
  const line = document.createElement('div');
  line.className = 'log-line ' + level;
  const t = new Date().toLocaleTimeString([], { hour12: false });
  line.innerHTML = '<span class="t">' + t + '</span><span class="m"></span>';
  line.querySelector('.m').textContent = msg;
  ui.log.prepend(line);
  while (ui.log.childElementCount > 80) ui.log.lastElementChild.remove();
  if (level === 'error') console.error('[sentryshield]', msg);
}

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------
const fuser = new ClassFuser();

const pose = new PoseEstimator({
  onUpdate: (p) => {
    renderPose(p);
    // Pose-only updates keep the map's phone marker alive between detections.
    if (net && net.isOpen() && Date.now() - lastPoseSent > 120) {
      lastPoseSent = Date.now();
      net.send('pose', { pose: p });
    }
  },
});
let lastPoseSent = 0;

const guidance = new GuidanceEngine({
  voice: {
    sendToServer: (text, level) => net && net.send('speak', { text, level }),
  },
});

const calibrator = new Calibrator({
  onUpdate: (st) => renderCalibration(st),
});

const sensor = new AcousticSensor({
  classifier: EchoNet,
  onLog: (m, l) => log(m, l),
  onDetection: (det, diag) => onSensorPulse(det, diag),
});

const net = new WsClient({
  role: 'phone',
  device: deviceInfo(),
  onMessage: (msg) => onServerMessage(msg),
  onState: (s, d) => renderConnection(s, d),
});

function deviceInfo() {
  const n = typeof navigator !== 'undefined' ? navigator : {};
  return {
    userAgent: n.userAgent || '',
    platform: (n.userAgentData && n.userAgentData.platform) || n.platform || '',
    cores: n.hardwareConcurrency || null,
    screen: typeof screen !== 'undefined' ? screen.width + 'x' + screen.height : null,
    secureContext: typeof isSecureContext !== 'undefined' ? isSecureContext : null,
  };
}

// ---------------------------------------------------------------------------
// Detection path
// ---------------------------------------------------------------------------
function onSensorPulse(raw, diag) {
  if (calibrator.active && raw) {
    // Calibration samples must be the *uncalibrated* range.
    const rawRange = diag && diag.rawRange != null ? diag.rawRange : raw.range_m;
    calibrator.addSample(rawRange, raw.confidence);
  }

  if (!raw) {
    // No return this pulse. Say so rather than holding a stale number.
    if (Date.now() - state.lastDetectionAt > 900) {
      state.lastDetection = null;
      renderDetection(null, diag);
      guidance.update(null, pose.heading);
    }
    return;
  }

  const p = pose.pose();
  const det = normalizeDetection(Object.assign({}, raw, {
    bearing_deg: p.heading,
    beamwidth_deg: 30,
    phone: p,
    source: 'live',
  }), 'live');
  if (!det) return;

  // Local temporal fusion so the phone's own display is as stable as the map's.
  const f = fuser.fuse(det);
  det.fusedClass = f.className;
  det.fusedConfidence = f.confidence;
  det.fusedProbs = f.probs;
  det.fusedSupport = f.support;
  det.fusedStable = f.stable;

  state.lastDetection = det;
  state.lastDetectionAt = Date.now();
  pushTrail(det);
  renderDetection(det, diag);
  guidance.update(det, p.heading);

  if (state.mode === 'live') net.send('detection', { detection: det });
}

function pushTrail(det) {
  state.trail.push({
    r: det.range_m,
    bearing: det.bearing_deg,
    cls: det.fusedClass || det.obstacleClass,
    conf: det.confidence,
    classConf: det.fusedConfidence || det.classConfidence || 0,
    at: performance.now(),
  });
  if (state.trail.length > 90) state.trail.shift();
}

// ---------------------------------------------------------------------------
// Server messages
// ---------------------------------------------------------------------------
function onServerMessage(msg) {
  switch (msg.type) {
    case 'welcome':
      state.mode = msg.mode || state.mode;
      applyAwsStatus(msg.aws);
      if (msg.capabilities) {
        state.classifierReady = msg.capabilities.classifier === 'loaded';
      }
      renderMode();
      log('Connected to command center (session ' + msg.sessionId + ', protocol v' + msg.protocolVersion + ').', 'success');
      break;

    case 'status':
      if (msg.mode) state.mode = msg.mode;
      applyAwsStatus(msg.aws);
      state.missionActive = !!msg.missionActive;
      renderMode();
      break;

    case 'heartbeat':
      net.measureFromServer(msg.serverTime);
      ui.latencyText.textContent = net.latencyMs != null ? net.latencyMs + ' ms' : '-- ms';
      break;

    case 'mission_start':
      state.missionActive = true;
      fuser.reset();
      state.trail = [];
      sensor.resetClutter();
      renderMode();
      log('Mission started (' + (msg.mode || state.mode) + ').', 'success');
      break;

    case 'mission_summary':
      state.missionActive = false;
      renderMode();
      log('Mission complete: ' + msg.summary.stats.detections + ' detections over '
        + msg.summary.stats.distanceScanned.toFixed(1) + ' m.', 'success');
      break;

    case 'guidance':
      showCue(msg.text, msg.level);
      if (msg.voice) {
        state.pollyLabel = msg.voice.label || 'LOCAL FALLBACK';
        ui.voiceLabel.textContent = 'VOICE: ' + state.pollyLabel;
        guidance.voice.setPollyAvailable(msg.voice.provider === 'polly', msg.voice.reason);
      }
      // A Polly cue arrives as audio in the next frame; anything else we say
      // locally right now so the operator is never waiting on the network.
      if (state.voiceOn && (!msg.voice || msg.voice.provider !== 'polly')) {
        guidance.speak(msg.text, msg.level);
      }
      break;

    case 'voice_audio':
      if (state.voiceOn && msg.audio) guidance.voice.playPollyAudio(msg.audio, msg.text);
      break;

    case 'scan_start':
      if (state.mode !== 'live') log('Simulation scan running on the server.', 'info');
      break;

    case 'error':
      log('Server: ' + msg.error, 'warn');
      break;

    default:
      break;
  }
}

function applyAwsStatus(aws) {
  if (!aws || !aws.voice) return;
  state.pollyLabel = aws.voice.label;
  ui.voiceLabel.textContent = 'VOICE: ' + aws.voice.label;
  guidance.voice.setPollyAvailable(aws.voice.provider === 'polly', aws.voice.reason);
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------
function renderConnection(s, d) {
  ui.connChip.className = 'conn ' + s;
  ui.connText.textContent = s === 'connected' ? 'LINKED'
    : s === 'connecting' ? 'LINKING'
      : s === 'reconnecting' ? 'RELINK ' + (d && d.attempt ? d.attempt : '') : 'OFFLINE';
  if (s === 'connected') log('Uplink established.', 'success');
  else if (s === 'reconnecting') log('Uplink lost — retrying (attempt ' + (d && d.attempt) + ').', 'warn');
}

function renderMode() {
  ui.modeChip.textContent = state.mode === 'live' ? 'LIVE' : state.mode === 'hybrid' ? 'HYBRID' : 'SIM';
  ui.btnMode.textContent = 'MODE: ' + (state.mode === 'live' ? 'LIVE' : state.mode === 'hybrid' ? 'HYB' : 'SIM');
  ui.btnMission.textContent = state.missionActive ? 'END MISSION' : 'START MISSION';
  ui.btnMission.classList.toggle('on', state.missionActive);
  const scanning = sensor.running;
  ui.btnScanText.textContent = scanning ? 'STOP SCAN' : 'START SCAN';
  ui.btnScan.classList.toggle('on', scanning);
  const label = scanning ? 'ACOUSTIC SCANNING — ACTIVE ' + sensor.rateHz + ' HZ'
    : state.mode === 'live' ? 'ACOUSTIC SCANNING — STANDBY'
      : 'SENSOR IDLE — ' + state.mode.toUpperCase() + ' DATA FROM SERVER';
  ui.scanState.textContent = label;
  ui.scanState.parentElement.classList.toggle('active', scanning);
}

function renderDetection(det, diag) {
  if (!det) {
    ui.rangeValue.innerHTML = '--<span class="unit">m</span>';
    ui.rangeValue.className = 'ro-value';
    ui.rangeSub.textContent = diag && diag.reason ? diag.reason : 'no returns';
    ui.classValue.textContent = '--';
    ui.classValue.className = 'ro-value small c-none';
    ui.classSub.textContent = state.classifierReady || EchoNet ? 'EchoNet idle' : 'classifier unavailable';
    ui.confText.textContent = '--';
    ui.confFill.style.width = '0%';
    renderProbs(null, null);
    return;
  }

  const r = det.range_m;
  ui.rangeValue.innerHTML = r.toFixed(2) + '<span class="unit">m</span>';
  ui.rangeValue.className = 'ro-value ' + (r < 0.6 ? 'close' : r < 1.3 ? 'near' : 'live');

  const vel = det.vel_mps;
  const motion = Math.abs(vel) < 0.08 ? 'steady'
    : vel > 0 ? 'closing ' + vel.toFixed(2) + ' m/s' : 'opening ' + Math.abs(vel).toFixed(2) + ' m/s';
  const off = signedOffset(det.bearing_deg, det.phone.heading);
  const side = Math.abs(off) < 12 ? 'ahead' : off > 0 ? Math.round(off) + '° right' : Math.round(-off) + '° left';
  ui.rangeSub.textContent = motion + ' · ' + side + ' · ' + det.snr_db.toFixed(0) + ' dB'
    + (det.cfar_pass ? '' : ' · below CFAR');

  const cls = det.fusedClass || det.obstacleClass;
  const conf = det.fusedConfidence || det.classConfidence || 0;
  ui.classValue.textContent = cls || '--';
  ui.classValue.className = 'ro-value small ' + (cls ? 'c-' + cls : 'c-none');
  // The sub-line is where honesty lives: a single unstable look says so.
  ui.classSub.textContent = !cls ? 'no classification'
    : det.fusedStable ? 'fused over ' + det.fusedSupport + ' echoes'
      : 'unstable — ' + det.fusedSupport + ' echo' + (det.fusedSupport === 1 ? '' : 'es');

  ui.confText.textContent = (conf * 100).toFixed(0) + '%';
  ui.confFill.style.width = (conf * 100).toFixed(0) + '%';
  ui.confFill.className = 'meter-fill ' + (conf < 0.4 ? 'low' : conf < 0.65 ? 'mid' : '');
  renderProbs(det.fusedProbs || det.classProbs, cls);
}

function renderProbs(probs, winner) {
  if (!probs) { ui.probRows.innerHTML = ''; return; }
  if (ui.probRows.childElementCount !== 3) {
    ui.probRows.innerHTML = CLASSES.map((c) =>
      '<div class="prow" data-c="' + c + '"><span class="pname c-' + c + '">' + c
      + '</span><span class="pbar"><i></i></span><span class="pval">0.00</span></div>').join('');
  }
  CLASSES.forEach((c, i) => {
    const row = ui.probRows.children[i];
    const v = probs[i] || 0;
    row.classList.toggle('win', c === winner);
    row.querySelector('i').style.width = (v * 100).toFixed(1) + '%';
    row.querySelector('i').style.background = 'var(--' + c.toLowerCase() + ')';
    row.querySelector('.pval').textContent = v.toFixed(2);
  });
}

function renderPose(p) {
  ui.headingValue.innerHTML = Math.round(p.heading) + '<span class="unit">°</span>';
  const st = pose.status();
  ui.headingSub.textContent = st.headingSource === 'none' ? 'no sensor'
    : st.headingAbsolute ? 'compass' : st.headingSource === 'manual' ? 'manual' : 'relative';
  ui.poseMethod.textContent = p.method.toUpperCase();
  ui.poseConf.textContent = (p.confidence * 100).toFixed(0) + '%';
  ui.poseNote.textContent = st.note;
}

function renderCalibration(st) {
  ui.btnCalibrate.textContent = st.calibrated ? 'CAL: ACTIVE' : 'CALIBRATE';
  ui.btnCalibrate.classList.toggle('off', !st.calibrated);
  if (!ui.calSheet.hidden) {
    ui.calFill.style.width = ((st.collected / st.target) * 100).toFixed(0) + '%';
    if (st.active) {
      ui.calStatus.className = 'cal-status';
      ui.calStatus.textContent = 'Hold still — ' + st.collected + ' / ' + st.target + ' samples.';
    } else if (st.result) {
      const r = st.result;
      if (r.ok) {
        ui.calStatus.className = 'cal-status ' + (r.quality === 'poor' ? 'warn' : 'ok');
        ui.calStatus.textContent = 'Factor ' + r.factor.toFixed(3) + ' from ' + r.samples
          + ' samples (measured ' + r.measuredMedian.toFixed(2) + ' m, spread ±'
          + (r.spreadM * 100).toFixed(0) + ' cm, ' + r.quality + ').'
          + (r.warning ? ' ' + r.warning : '');
      } else {
        ui.calStatus.className = 'cal-status err';
        ui.calStatus.textContent = r.warning || r.error || 'Calibration failed.';
      }
    }
  }
}

function showCue(text, level) {
  ui.cueText.textContent = text;
  ui.cueText.className = 'cue ' + (level || 'info');
}

// ---------------------------------------------------------------------------
// Sonar rendering
// ---------------------------------------------------------------------------
const sonar = {
  canvas: ui.sonar,
  ctx: ui.sonar ? ui.sonar.getContext('2d') : null,
  w: 0, h: 0, dpr: 1,
  maxRange: 4.0,
};

function resizeSonar() {
  if (!sonar.canvas) return;
  const rect = sonar.canvas.getBoundingClientRect();
  if (!rect.width) return;
  sonar.dpr = Math.min(2, window.devicePixelRatio || 1);
  sonar.canvas.width = Math.round(rect.width * sonar.dpr);
  sonar.canvas.height = Math.round(rect.height * sonar.dpr);
  sonar.w = rect.width;
  sonar.h = rect.height;
}

function drawSonar(now) {
  const c = sonar.ctx;
  if (!c || !sonar.w) return;
  const { w, h, dpr } = sonar;
  c.setTransform(dpr, 0, 0, dpr, 0, 0);
  c.clearRect(0, 0, w, h);

  // Origin at the bottom centre: the sensor looks forward, and the display
  // shows the forward half-plane only.
  const ox = w / 2;
  const oy = h - 10;
  const radius = Math.min(w / 2 - 8, h - 22);

  // Range rings, labelled in metres.
  c.lineWidth = 1;
  c.font = '9px ui-monospace, monospace';
  for (let m = 1; m <= 4; m++) {
    const rr = (m / sonar.maxRange) * radius;
    c.beginPath();
    c.arc(ox, oy, rr, Math.PI, 2 * Math.PI);
    c.strokeStyle = m === 4 ? 'rgba(86,130,170,0.26)' : 'rgba(86,130,170,0.14)';
    c.stroke();
    c.fillStyle = 'rgba(85,103,126,0.85)';
    c.fillText(m + 'm', ox + 3, oy - rr + 10);
  }

  // Bearing graticule every 30 degrees of boresight offset.
  for (let a = -90; a <= 90; a += 30) {
    const rad = (a * Math.PI) / 180;
    c.beginPath();
    c.moveTo(ox, oy);
    c.lineTo(ox + Math.sin(rad) * radius, oy - Math.cos(rad) * radius);
    c.strokeStyle = a === 0 ? 'rgba(86,130,170,0.22)' : 'rgba(86,130,170,0.10)';
    c.stroke();
  }

  // The sweep. In live mode it is the phone's own boresight, so it is a real
  // indication of where the sensor is pointing, not an animation.
  const scanning = sensor.running || state.mode !== 'live';
  if (scanning) {
    if (state.mode === 'live') {
      state.sweepAngle = 0;                   // boresight is always dead ahead
    } else {
      state.sweepAngle = Math.sin(now / 1400) * 55;
    }
    const rad = (state.sweepAngle * Math.PI) / 180;
    const grad = c.createLinearGradient(ox, oy, ox + Math.sin(rad) * radius, oy - Math.cos(rad) * radius);
    grad.addColorStop(0, 'rgba(44,232,245,0.30)');
    grad.addColorStop(1, 'rgba(44,232,245,0.02)');
    // A beam wedge, drawn at the sensor's real ~30 degree beamwidth.
    c.beginPath();
    c.moveTo(ox, oy);
    c.arc(ox, oy, radius, -Math.PI / 2 + rad - 0.26, -Math.PI / 2 + rad + 0.26);
    c.closePath();
    c.fillStyle = grad;
    c.fill();
  }

  // Detection persistence: recent returns fade over 2.5 s.
  for (const p of state.trail) {
    const age = (now - p.at) / 2500;
    if (age > 1) continue;
    const off = signedOffset(p.bearing, pose.heading);
    if (Math.abs(off) > 92) continue;
    const rad = (off * Math.PI) / 180;
    const rr = Math.min(1, p.r / sonar.maxRange) * radius;
    const x = ox + Math.sin(rad) * rr;
    const y = oy - Math.cos(rad) * rr;
    const alpha = (1 - age) * (0.35 + 0.65 * p.conf);
    const col = p.cls === 'WALL' ? '78,168,255' : p.cls === 'SOFT' ? '255,180,84'
      : p.cls === 'OPENING' ? '155,140,255' : '44,232,245';

    // Arc of angular uncertainty: the beam is ~30 degrees wide, so a return
    // is genuinely "somewhere along this arc", and drawing it as a dot alone
    // would overstate what the sensor knows.
    c.beginPath();
    c.arc(ox, oy, rr, -Math.PI / 2 + rad - 0.24, -Math.PI / 2 + rad + 0.24);
    c.strokeStyle = 'rgba(' + col + ',' + (alpha * 0.32).toFixed(3) + ')';
    c.lineWidth = 2;
    c.stroke();

    c.beginPath();
    c.arc(x, y, 3.2, 0, Math.PI * 2);
    c.fillStyle = 'rgba(' + col + ',' + alpha.toFixed(3) + ')';
    c.fill();
  }

  // The newest detection gets a halo whose size encodes class confidence.
  const d = state.lastDetection;
  if (d && Date.now() - state.lastDetectionAt < 900) {
    const off = signedOffset(d.bearing_deg, pose.heading);
    const rad = (off * Math.PI) / 180;
    const rr = Math.min(1, d.range_m / sonar.maxRange) * radius;
    const x = ox + Math.sin(rad) * rr;
    const y = oy - Math.cos(rad) * rr;
    const conf = d.fusedConfidence || d.classConfidence || 0;
    c.beginPath();
    c.arc(x, y, 6 + 7 * conf, 0, Math.PI * 2);
    c.strokeStyle = 'rgba(44,232,245,0.6)';
    c.lineWidth = 1;
    c.stroke();
  }

  // Sensor origin.
  c.beginPath();
  c.arc(ox, oy, 3, 0, Math.PI * 2);
  c.fillStyle = 'rgba(44,232,245,0.9)';
  c.fill();
}

function frame(now) {
  drawSonar(now);
  requestAnimationFrame(frame);
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------
ui.btnScan.addEventListener('click', async () => {
  if (sensor.running) {
    await sensor.stop();
    net.send('scan_stop', {});
    renderMode();
    return;
  }
  const res = await sensor.start({ rateHz: 20 });
  if (!res.ok) {
    log(res.error, 'error');
    showCue('Live audio unavailable: ' + res.error + ' Simulation mode still works.', 'warning');
    renderMode();
    return;
  }
  // Starting the mic means this device is the sensor: switch to live mode.
  if (state.mode !== 'live') {
    state.mode = 'live';
    net.send('set_mode', { mode: 'live' });
  }
  sensor.setCalibration(calibrator.factor, calibrator.offsetM);
  net.send('scan_start', { rateHz: sensor.rateHz });
  renderMode();
});

ui.btnMission.addEventListener('click', () => {
  if (state.missionActive) {
    net.send('mission_complete', {});
    state.missionActive = false;
  } else {
    fuser.reset();
    state.trail = [];
    sensor.resetClutter();
    net.send('mission_start', { rateHz: 20 });
    state.missionActive = true;
  }
  renderMode();
});

ui.btnMode.addEventListener('click', () => {
  // live -> simulation -> hybrid -> live
  const order = ['live', 'simulation', 'hybrid'];
  const next = order[(order.indexOf(state.mode) + 1) % order.length];
  state.mode = next;
  net.send('set_mode', { mode: next });
  log('Mode set to ' + next + '.', 'info');
  renderMode();
});

ui.btnAudio.addEventListener('click', () => {
  state.audioOn = !state.audioOn;
  guidance.setEnabled(state.audioOn);
  ui.btnAudio.textContent = state.audioOn ? 'AUDIO ON' : 'AUDIO OFF';
  ui.btnAudio.classList.toggle('off', !state.audioOn);
});

ui.btnVoice.addEventListener('click', () => {
  state.voiceOn = !state.voiceOn;
  guidance.speechEnabled = state.voiceOn;
  if (!state.voiceOn) guidance.voice.cancel();
  ui.btnVoice.textContent = state.voiceOn ? 'VOICE ON' : 'VOICE OFF';
  ui.btnVoice.classList.toggle('off', !state.voiceOn);
});

ui.btnTurnL.addEventListener('click', () => pose.manualTurn(-15));
ui.btnTurnR.addEventListener('click', () => pose.manualTurn(15));
ui.btnWalk.addEventListener('click', () => pose.manualWalk(0.7));

ui.logToggle.addEventListener('click', () => ui.logPanel.classList.toggle('collapsed'));

// ---- calibration sheet ----
ui.btnCalibrate.addEventListener('click', () => {
  ui.calSheet.hidden = false;
  renderCalibration(calibrator.status());
  if (!sensor.running) {
    ui.calStatus.className = 'cal-status warn';
    ui.calStatus.textContent = 'Start the scan first — calibration needs live echoes.';
  }
});
ui.calCancel.addEventListener('click', () => {
  calibrator.cancel();
  ui.calSheet.hidden = true;
});
ui.calChoices.addEventListener('click', (e) => {
  const b = e.target.closest('button[data-d]');
  if (!b) return;
  Array.from(ui.calChoices.children).forEach((x) => x.classList.remove('on'));
  b.classList.add('on');
  ui.calDist.value = b.dataset.d;
});
ui.calStart.addEventListener('click', () => {
  if (!sensor.running) {
    ui.calStatus.className = 'cal-status err';
    ui.calStatus.textContent = 'The scan must be running to collect calibration samples.';
    return;
  }
  const d = parseFloat(ui.calDist.value);
  if (!(d > 0.2 && d < 5)) {
    ui.calStatus.className = 'cal-status err';
    ui.calStatus.textContent = 'Enter a distance between 0.2 and 5 m.';
    return;
  }
  calibrator.begin(d);
  log('Calibrating against a wall at ' + d.toFixed(2) + ' m.', 'info');
  const check = setInterval(() => {
    if (calibrator.active) return;
    clearInterval(check);
    const r = calibrator.result;
    if (r && r.ok) {
      sensor.setCalibration(r.factor, r.offsetM);
      net.send('calibration', {
        factor: r.factor, offsetM: r.offsetM, samples: r.samples, knownDistanceM: r.knownDistanceM,
      });
      log('Calibration applied: factor ' + r.factor.toFixed(3) + ' (' + r.quality + ').', 'success');
    } else {
      log('Calibration not applied.', 'warn');
    }
  }, 250);
});

// ---- first-run gate ----
function renderGateCaps() {
  const caps = AcousticSensor.capabilities();
  const ps = PoseEstimator.supported();
  const rows = [
    ['Secure context (HTTPS)', caps.secureContext, caps.secureContext ? 'yes' : 'required for mic'],
    ['Microphone API', caps.getUserMedia, caps.getUserMedia ? 'available' : 'unavailable'],
    ['AudioContext', caps.audioContext, caps.audioContext ? 'available' : 'unavailable'],
    ['AudioWorklet', caps.audioWorklet, caps.audioWorklet ? 'available' : 'ScriptProcessor fallback'],
    ['Orientation sensor', ps.orientation, ps.orientation ? 'available' : 'unavailable'],
    ['Motion sensor', ps.motion, ps.motion ? 'available' : 'unavailable'],
    ['EchoNet classifier', !!EchoNet, EchoNet ? 'loaded' : 'not loaded'],
  ];
  ui.gateCaps.innerHTML = rows.map(([name, ok, note]) =>
    '<div class="gate-cap ' + (ok ? 'yes' : 'no') + '"><span>' + name + '</span><b>' + note.toUpperCase() + '</b></div>'
  ).join('');
  if (!caps.liveAudioPossible) {
    ui.gateGo.textContent = 'CONTINUE (NO LIVE AUDIO)';
  }
}

async function enableSensors() {
  ui.gateSheet.hidden = true;

  // Both audio and orientation need to be requested inside the gesture.
  const perm = await pose.requestPermission();
  pose.start();
  if (perm.orientation !== 'granted') log('Orientation permission: ' + perm.orientation + ' — heading will be manual.', 'warn');

  const g = await guidance.init();
  if (!g.ok) log('Guidance audio unavailable: ' + g.error, 'warn');
  else log('Guidance audio ready' + (g.panning ? ' with stereo panning.' : '.'), 'success');

  if (EchoNet) {
    const st = EchoNet.selfTest();
    state.classifierReady = st.ok;
    log('EchoNet self-test ' + (st.ok ? 'passed' : 'FAILED') + ' (' + EchoNet.META.n_params
      + ' params, synthetic val ' + (EchoNet.META.val_accuracy * 100).toFixed(1) + '%).', st.ok ? 'success' : 'error');
  } else {
    log('EchoNet not loaded — ranges still work, classification will not.', 'warn');
  }

  const caps = AcousticSensor.capabilities();
  if (!caps.liveAudioPossible) {
    log('Live audio unavailable: ' + caps.reason, 'warn');
    showCue('Live audio unavailable here. Use SIMULATION mode from the command center — the map works identically.', 'warning');
  }
  renderMode();
}

ui.gateGo.addEventListener('click', enableSensors);
ui.gateSkip.addEventListener('click', () => {
  ui.gateSheet.hidden = true;
  pose.requestPermission().then(() => pose.start());
  log('Continuing without audio. Simulation and hybrid modes still work.', 'info');
});

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
window.addEventListener('resize', resizeSonar);
window.addEventListener('orientationchange', () => setTimeout(resizeSonar, 250));

// Suspending the tab suspends the AudioContext; resume when we come back.
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible' && sensor.audioCtx && sensor.audioCtx.state === 'suspended') {
    sensor.audioCtx.resume().catch(() => {});
    log('AudioContext resumed after returning to the tab.', 'info');
  }
});

renderGateCaps();
renderMode();
renderPose(pose.pose());
renderCalibration(calibrator.status());
resizeSonar();
requestAnimationFrame(frame);
net.connect();

// Keep the pose/heading readout live even with no sensors at all.
setInterval(() => {
  renderPose(pose.pose());
  if (state.lastDetection && Date.now() - state.lastDetectionAt > 1200) {
    state.lastDetection = null;
    renderDetection(null, null);
  }
}, 500);

log('Sensor page ready. ' + (AcousticSensor.capabilities().liveAudioPossible
  ? 'Live audio available.' : 'Live audio unavailable: ' + AcousticSensor.capabilities().reason), 'info');
