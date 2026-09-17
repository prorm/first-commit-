/**
 * Command center application.
 *
 * Holds a local copy of the fused world and renders it every frame.  Detections
 * arrive individually (so the map moves at sensor rate) and a full snapshot
 * arrives periodically (so the map can never drift from the server, and can
 * rebuild itself completely after a reconnect or a page reload mid-mission).
 *
 * The local fusion uses the same shared modules the server runs, so "smooth
 * local rendering" and "authoritative server state" are the same computation.
 */
import { WsClient } from '../shared/wsclient.mjs';
import { OccupancyGrid, PointCloud, Trajectory } from '../shared/spatial.mjs';
import { reconstruct } from '../shared/reconstruct.mjs';
import { CLASSES, makePose } from '../shared/protocol.mjs';
import { MapRenderer, worldBounds, CLASS_RGB } from './renderer.mjs';

const $ = (id) => document.getElementById(id);

const ui = {
  stage: $('stage'),
  modeChip: $('modeChip'), connChip: $('connChip'), connText: $('connText'), linkMeta: $('linkMeta'),
  missionLine: $('missionLine'), statGrid: $('statGrid'), reconLine: $('reconLine'),
  aiProbs: $('aiProbs'), aiHist: $('aiHist'), aiVerdict: $('aiVerdict'), aiNote: $('aiNote'), aiMeta: $('aiMeta'),
  btnMission: $('btnMission'), btnZero: $('btnZero'), btnRecon: $('btnRecon'),
  btnReplay: $('btnReplay'), btnRecord: $('btnRecord'), btnFit: $('btnFit'),
  selMode: $('selMode'), selScenario: $('selScenario'),
  layGrid: $('layGrid'), layCloud: $('layCloud'), laySurf: $('laySurf'), layTruth: $('layTruth'),
  voiceLabel: $('voiceLabel'), cueText: $('cueText'),
  zvBanner: $('zvBanner'), reconCard: $('reconCard'), rcFill: $('rcFill'), rcSub: $('rcSub'),
  summarySheet: $('summarySheet'), sumGrid: $('sumGrid'), sumCaveats: $('sumCaveats'),
  sumAi: $('sumAi'), sumAiText: $('sumAiText'), sumClose: $('sumClose'), sumReplay: $('sumReplay'),
  replayBar: $('replayBar'), replayFill: $('replayFill'), replayLabel: $('replayLabel'), replayStop: $('replayStop'),
};

const EchoNet = (typeof globalThis !== 'undefined' && globalThis.EchoNet) || null;

// ---------------------------------------------------------------------------
// Local world model
// ---------------------------------------------------------------------------
const world = {
  grid: new OccupancyGrid(),
  cloud: new PointCloud(),
  trajectory: new Trajectory(),
  pose: makePose({ confidence: 0.6 }),
  reconstruction: null,
  serverGrid: null,        // sparse grid straight from the snapshot
  truth: null,             // simulation ground truth, opt-in
  lastDetection: null,
  stats: null,
  missionActive: false,
  mode: 'simulation',
  recording: null,
  replaying: false,
  lastReconAt: 0,
  detections: 0,
  history: [],             // temporal class-confidence trail for the AI panel
};

const renderer = new MapRenderer(ui.stage);

// ---------------------------------------------------------------------------
// Network
// ---------------------------------------------------------------------------
const net = new WsClient({
  role: 'map',
  onMessage: onMessage,
  onState: (s, d) => {
    ui.connChip.className = 'conn ' + s;
    ui.connText.textContent = s === 'connected' ? 'LINKED'
      : s === 'connecting' ? 'LINKING'
        : s === 'reconnecting' ? 'RELINKING' : 'OFFLINE';
    // The map keeps whatever it already drew during a reconnect and asks for a
    // fresh snapshot once the link is back.
    if (s === 'connected') net.send('request_state', {});
  },
});

function onMessage(msg) {
  switch (msg.type) {
    case 'welcome':
      world.mode = msg.mode || world.mode;
      ui.selMode.value = world.mode;
      populateScenarios(msg.scenarios, msg.simulation);
      applyAws(msg.aws);
      applyCapabilities(msg.capabilities);
      renderChrome();
      break;

    case 'status':
      if (msg.mode) { world.mode = msg.mode; ui.selMode.value = msg.mode; }
      world.missionActive = !!msg.missionActive;
      world.recording = msg.recording || null;
      world.replaying = !!msg.replaying;
      if (msg.simulation) {
        world.simInfo = msg.simInfo || msg.simulation;
        world.truth = msg.simulation.surfaces || world.truth;
        if (msg.simulation.scenario) ui.selScenario.value = msg.simulation.scenario;
      }
      applyAws(msg.aws);
      renderChrome();
      break;

    case 'heartbeat':
      net.measureFromServer(msg.serverTime);
      ui.linkMeta.textContent = (net.latencyMs != null ? net.latencyMs + ' ms' : '-- ms')
        + ' · ' + (msg.peers ? msg.peers.phone + ' phone' : '');
      break;

    case 'detection':
      ingest(msg.detection);
      break;

    case 'pose':
      world.pose = msg.pose;
      world.trajectory.push(world.pose, Date.now());
      if (renderer.follow) renderer.centerOn(world.pose.x, world.pose.y);
      break;

    case 'state_snapshot':
      applySnapshot(msg);
      break;

    case 'mission_start':
      resetWorld();
      world.missionActive = true;
      if (msg.simulation) world.truth = msg.simulation.surfaces || null;
      renderer.clearReconstruction();
      renderChrome();
      break;

    case 'mission_summary':
      world.missionActive = false;
      showSummary(msg);
      renderChrome();
      break;

    case 'scan_summary':
      if (msg.available && msg.text) {
        ui.sumAi.hidden = false;
        ui.sumAiText.textContent = msg.text;
      } else {
        ui.sumAi.hidden = true;
      }
      break;

    case 'guidance':
      ui.cueText.textContent = msg.text;
      ui.cueText.className = 'tick-text ' + (msg.level || 'info');
      if (msg.voice) {
        ui.voiceLabel.textContent = 'VOICE: ' + msg.voice.label;
        ui.voiceLabel.classList.toggle('polly', msg.voice.provider === 'polly');
      }
      break;

    case 'replay_start':
      resetWorld();
      world.replaying = true;
      ui.replayBar.hidden = false;
      ui.replayLabel.textContent = 'REPLAY · ' + (msg.scenario || msg.mode || '');
      renderer.clearReconstruction();
      renderChrome();
      break;

    case 'replay_status':
      if (msg.state === 'playing') {
        ui.replayBar.hidden = false;
        ui.replayFill.style.width = ((msg.fraction || 0) * 100).toFixed(1) + '%';
      } else {
        world.replaying = false;
        ui.replayFill.style.width = '100%';
        setTimeout(() => { ui.replayBar.hidden = true; }, 1200);
        if (msg.state === 'finished') {
          // A finished replay ends on the reconstruction, which is the shape
          // of the story: raw echoes in, environment out.
          triggerReconstruct();
        }
        renderChrome();
      }
      break;

    case 'recording_list':
      world.recordings = msg.recordings || [];
      renderChrome();
      break;

    case 'error':
      ui.cueText.textContent = 'Server: ' + msg.error;
      ui.cueText.className = 'tick-text warning';
      break;

    default:
      break;
  }
}

/** One detection into the local world model. */
function ingest(det) {
  if (!det) return;
  world.lastDetection = det;
  world.pose = det.phone;
  world.detections++;
  world.cloud.add(det);
  world.trajectory.push(det.phone, det.t);
  world.grid.integrate(det.phone, det.bearing_deg, det.range_m, {
    weight: det.confidence * (0.35 + 0.65 * det.phone.confidence),
    className: det.fusedClass || det.obstacleClass,
    beamwidth_deg: det.beamwidth_deg,
  });
  // The snapshot's sparse grid is superseded once we are integrating locally.
  world.serverGrid = null;
  gridCacheAt = 0;

  renderer.noteDetection(det);
  if (renderer.follow) renderer.centerOn(det.phone.x, det.phone.y);
  pushHistory(det);
  renderAi(det);

  // Recompute geometry a few times a second, not per pulse.
  const now = performance.now();
  if (now - world.lastReconAt > 450) {
    world.lastReconAt = now;
    world.reconstruction = reconstruct(world.cloud.points);
    renderReconLine();
  }
}

function pushHistory(det) {
  const cls = det.fusedClass || det.obstacleClass;
  const conf = det.fusedConfidence || det.classConfidence || 0;
  world.history.push({ cls, conf, agree: cls === (det.obstacleClass || cls) });
  if (world.history.length > 16) world.history.shift();
}

/**
 * Adopt the server's authoritative state.  Used on connect, after a reconnect,
 * and periodically — so a page refresh mid-mission loses nothing.
 */
function applySnapshot(snap) {
  if (!snap) return;
  world.pose = snap.pose || world.pose;
  world.stats = snap.stats || world.stats;
  world.missionActive = !!snap.missionActive;
  if (snap.lastDetection) {
    world.lastDetection = snap.lastDetection;
    renderAi(snap.lastDetection);
  }

  // Take the server's cloud and trajectory wholesale: they are the truth.
  if (snap.cloud) world.cloud.loadSerialized(snap.cloud);
  if (snap.trajectory) world.trajectory.loadSerialized(snap.trajectory);
  if (snap.grid) {
    world.grid = OccupancyGrid.deserialize(snap.grid);
    world.serverGrid = snap.grid;
  }
  if (snap.reconstruction) world.reconstruction = snap.reconstruction;
  else world.reconstruction = reconstruct(world.cloud.points);

  if (snap.reconstructTrigger) triggerReconstruct();
  if (renderer.follow && world.pose) renderer.centerOn(world.pose.x, world.pose.y);
  renderChrome();
  renderReconLine();
}

function resetWorld() {
  world.grid = new OccupancyGrid();
  world.cloud = new PointCloud();
  world.trajectory = new Trajectory();
  world.reconstruction = null;
  world.serverGrid = null;
  world.lastDetection = null;
  world.detections = 0;
  world.history = [];
  world.stats = null;
  renderer.pulses = [];
  renderer.follow = true;
  renderAi(null);
  renderReconLine();
}

// ---------------------------------------------------------------------------
// Chrome / HUD rendering
// ---------------------------------------------------------------------------
function applyAws(aws) {
  if (!aws) return;
  world.aws = aws;
  ui.voiceLabel.textContent = 'VOICE: ' + aws.voice.label;
  ui.voiceLabel.classList.toggle('polly', aws.voice.provider === 'polly');
}

function applyCapabilities(caps) {
  if (!caps) return;
  world.caps = caps;
  const meta = caps.classifierMeta;
  ui.aiMeta.textContent = meta
    ? 'EchoNet · ' + meta.n_params + ' params · val ' + (meta.val_accuracy * 100).toFixed(1) + '%'
    : caps.classifier === 'loaded' ? 'EchoNet loaded' : 'classifier unavailable';
}

function populateScenarios(list, sim) {
  if (!list || !list.length) return;
  ui.selScenario.innerHTML = list.map((s) =>
    '<option value="' + s.id + '">' + s.label.split('—')[0].trim() + '</option>').join('');
  if (sim) {
    ui.selScenario.value = sim.scenario;
    world.truth = sim.surfaces || null;
  }
}

function renderChrome() {
  ui.modeChip.textContent = world.mode.toUpperCase();
  ui.btnMission.textContent = world.missionActive ? 'END MISSION' : 'START MISSION';
  ui.btnMission.classList.toggle('on', world.missionActive);
  ui.btnRecord.textContent = world.recording ? 'RECORDING' : 'RECORD';
  ui.btnRecord.classList.toggle('on', !!world.recording);
  ui.selScenario.disabled = world.mode === 'live';

  const st = world.stats;
  const observed = st ? (st.freeArea || 0) + (st.occupiedArea || 0) : 0;
  const envLabel = world.missionActive
    ? (observed > 1.5 ? 'ENVIRONMENT: MAPPING' : 'ENVIRONMENT: UNKNOWN')
    : 'ENVIRONMENT: STANDBY';
  const scanLabel = world.replaying ? 'REPLAY'
    : world.missionActive ? 'ACTIVE' : 'IDLE';
  ui.missionLine.innerHTML = envLabel + ' &nbsp;·&nbsp; VISIBILITY: '
    + (renderer.zeroVisibility ? '0%' : 'NORMAL') + ' &nbsp;·&nbsp; ACOUSTIC SCAN: ' + scanLabel;
  ui.missionLine.classList.toggle('active', world.missionActive || world.replaying);

  renderStats();
}

function renderStats() {
  const st = world.stats || {};
  const recon = world.reconstruction;
  const cells = [
    ['DETECTIONS', String(st.detections != null ? st.detections : world.detections), true],
    ['SCANNED', (st.distanceScanned || world.trajectory.distance || 0).toFixed(1), false, 'm'],
    ['ECHO POINTS', String(world.cloud.points.length), false],
    // Area observed, not "coverage": a percentage of the fixed 24 x 24 m grid
    // says nothing useful about a 6 x 4 m room.
    ['AREA OBSERVED', ((st.freeArea || 0) + (st.occupiedArea || 0)).toFixed(1), false, 'm²'],
    ['SURFACES', String(recon ? recon.segments.length : 0), false],
    ['OPENINGS?', String(recon ? (recon.openings || []).filter((o) => o.confidence > 0.25).length : 0), false],
    ['MEAN CONF', ((st.avgClassConfidence || 0) * 100).toFixed(0), false, '%'],
    ['CFAR PASS', ((st.cfarRate || 0) * 100).toFixed(0), false, '%'],
  ];
  ui.statGrid.innerHTML = cells.map(([k, v, hl, unit]) =>
    '<div class="stat' + (hl ? ' hl' : '') + '"><span class="k">' + k + '</span><span class="v">'
    + v + (unit ? '<small>' + unit + '</small>' : '') + '</span></div>').join('');
}

function renderReconLine() {
  const r = world.reconstruction;
  if (!r || !r.segments.length) {
    ui.reconLine.textContent = 'RECONSTRUCTION — INSUFFICIENT ECHOES';
    ui.reconLine.classList.remove('on');
    return;
  }
  ui.reconLine.textContent = 'RECONSTRUCTION CONFIDENCE ' + Math.round(r.confidence * 100)
    + '% · ' + r.segments.length + ' SURFACES · ' + (r.stats.totalLength || 0).toFixed(1) + ' M';
  ui.reconLine.classList.add('on');
}

/** The AI panel: the model's own numbers, with its uncertainty visible. */
function renderAi(det) {
  if (ui.aiProbs.childElementCount !== 3) {
    ui.aiProbs.innerHTML = CLASSES.map((c) =>
      '<div class="arow" data-c="' + c + '"><span class="aname c-' + c + '">' + c
      + '</span><span class="abar"><i></i></span><span class="aval">0.00</span></div>').join('');
  }
  if (!det) {
    CLASSES.forEach((c, i) => {
      const row = ui.aiProbs.children[i];
      row.classList.remove('win');
      row.querySelector('i').style.width = '0%';
      row.querySelector('.aval').textContent = '0.00';
    });
    ui.aiVerdict.textContent = '--';
    ui.aiVerdict.className = 'c-none';
    ui.aiHist.innerHTML = '';
    ui.aiNote.textContent = 'Waiting for echoes.';
    ui.aiNote.classList.remove('warn');
    return;
  }

  // Per-echo output is what the network said this pulse; the fused row is the
  // decision. Showing the single-echo probabilities keeps the model honest.
  const probs = det.classProbs || [0, 0, 0];
  const fused = det.fusedProbs || probs;
  const cls = det.fusedClass || det.obstacleClass;
  CLASSES.forEach((c, i) => {
    const row = ui.aiProbs.children[i];
    row.classList.toggle('win', c === cls);
    const bar = row.querySelector('i');
    bar.style.width = ((fused[i] || 0) * 100).toFixed(1) + '%';
    bar.style.background = 'rgba(' + (CLASS_RGB[c] || [44, 232, 245]).join(',') + ',0.85)';
    row.querySelector('.aval').textContent = (fused[i] || 0).toFixed(2);
  });

  ui.aiVerdict.textContent = cls || '--';
  ui.aiVerdict.className = cls ? 'c-' + cls : 'c-none';

  // Temporal history: the sequence of confidences that led here. Bars that
  // disagreed with the final decision are drawn in amber, so instability is
  // legible rather than averaged away.
  ui.aiHist.innerHTML = world.history.map((h) => {
    const height = Math.max(2, Math.round(h.conf * 20));
    return '<i class="' + (h.cls === cls ? '' : 'dis') + '" style="height:' + height + 'px"></i>';
  }).join('');

  const conf = det.fusedConfidence || det.classConfidence || 0;
  const support = det.fusedSupport || 1;
  if (cls === 'OPENING') {
    ui.aiNote.textContent = 'OPENING is the model\'s weakest class (43 % recall in synthetic validation). Treat as a candidate, not a confirmed doorway.';
    ui.aiNote.classList.add('warn');
  } else if (!det.fusedStable) {
    ui.aiNote.textContent = 'Unstable: ' + support + ' echo' + (support === 1 ? '' : 'es')
      + ' at this spot, predictions disagree. Needs more looks.';
    ui.aiNote.classList.add('warn');
  } else {
    ui.aiNote.textContent = 'Fused over ' + support + ' echoes at '
      + det.range_m.toFixed(2) + ' m · single-echo top prob ' + Math.max.apply(null, probs).toFixed(2)
      + ' · fused ' + conf.toFixed(2);
    ui.aiNote.classList.remove('warn');
  }
}

// ---------------------------------------------------------------------------
// The reconstruction moment
// ---------------------------------------------------------------------------
function triggerReconstruct() {
  world.reconstruction = reconstruct(world.cloud.points);
  renderReconLine();
  net.send('reconstruct', {});

  if (!world.reconstruction.segments.length) {
    ui.cueText.textContent = 'Not enough echo evidence to reconstruct surfaces yet — keep scanning.';
    ui.cueText.className = 'tick-text warning';
    return;
  }

  renderer.startReconstruction();
  ui.reconCard.hidden = false;
  ui.rcSub.textContent = 'fusing ' + world.cloud.points.length + ' echoes into inferred surfaces';

  // Hold the exact reconstruction this animation is describing.  A mission
  // restart or a replay can land mid-animation and replace (or clear) the
  // world's reconstruction; the card must then abort rather than narrate
  // geometry that no longer exists.
  const recon = world.reconstruction;
  const started = performance.now();
  const dur = 1500;
  const step = () => {
    if (world.reconstruction !== recon) { ui.reconCard.hidden = true; return; }
    const f = Math.min(1, (performance.now() - started) / dur);
    ui.rcFill.style.width = (f * 100).toFixed(0) + '%';
    if (f < 1) { requestAnimationFrame(step); return; }
    ui.rcSub.textContent = recon.segments.length + ' surfaces · ' + recon.corners.length + ' corners · '
      + (recon.openings || []).filter((o) => o.confidence > 0.25).length + ' opening candidates · confidence '
      + Math.round(recon.confidence * 100) + '%';
    setTimeout(() => { ui.reconCard.hidden = true; }, 2600);
  };
  requestAnimationFrame(step);

  // Frame the whole reconstructed space once the morph is underway.
  setTimeout(() => {
    const b = worldBounds(world.cloud.points, world.trajectory, world.pose);
    if (b) { renderer.follow = false; renderer.fitTo(b, 1.2); }
  }, 350);
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------
ui.btnMission.addEventListener('click', () => {
  if (world.missionActive) {
    net.send('mission_complete', {});
  } else {
    resetWorld();
    net.send('mission_start', {
      scenario: ui.selScenario.value || undefined,
      rateHz: 20,
      record: true,
    });
  }
});

ui.btnZero.addEventListener('click', () => {
  renderer.zeroVisibility = !renderer.zeroVisibility;
  ui.btnZero.classList.toggle('on', renderer.zeroVisibility);
  ui.zvBanner.hidden = !renderer.zeroVisibility;
  // Zero-visibility is a presentation mode: it suppresses the optical-world
  // context (grid, background) and leaves only what the sensor heard.
  renderer.layers.grid = !renderer.zeroVisibility && ui.layGrid.checked;
  renderChrome();
});

ui.btnRecon.addEventListener('click', () => {
  if (renderer.reconAnimTarget > 0) {
    renderer.clearReconstruction();
    ui.btnRecon.classList.remove('armed');
  } else {
    triggerReconstruct();
    ui.btnRecon.classList.add('armed');
  }
});

ui.btnReplay.addEventListener('click', () => {
  net.send('replay_start', { speed: 2.5 });
});

ui.btnRecord.addEventListener('click', () => {
  if (world.recording) net.send('record_stop', {});
  else net.send('record_start', { note: 'manual' });
});

ui.btnFit.addEventListener('click', () => {
  const b = worldBounds(world.cloud.points, world.trajectory, world.pose);
  renderer.follow = false;
  if (b) renderer.fitTo(b);
});

ui.selMode.addEventListener('change', () => {
  world.mode = ui.selMode.value;
  net.send('set_mode', { mode: world.mode });
  renderChrome();
});

ui.selScenario.addEventListener('change', () => {
  net.send('sim_control', { scenario: ui.selScenario.value, action: 'reset' });
  resetWorld();
});

ui.layGrid.addEventListener('change', () => { renderer.layers.grid = ui.layGrid.checked; });
ui.layCloud.addEventListener('change', () => { renderer.layers.cloud = ui.layCloud.checked; });
ui.laySurf.addEventListener('change', () => { renderer.layers.surfaces = ui.laySurf.checked; });
ui.layTruth.addEventListener('change', () => { renderer.layers.truth = ui.layTruth.checked; });

ui.sumClose.addEventListener('click', () => { ui.summarySheet.hidden = true; });
ui.sumReplay.addEventListener('click', () => {
  ui.summarySheet.hidden = true;
  net.send('replay_start', { speed: 2.5 });
});
ui.replayStop.addEventListener('click', () => net.send('replay_stop', {}));

function showSummary(msg) {
  const s = msg.summary.stats;
  const r = msg.summary.reconstruction;
  const rows = [
    ['DISTANCE SCANNED', (s.distanceScanned || 0).toFixed(1) + ' m'],
    ['DURATION', Math.round((s.elapsedMs || 0) / 1000) + ' s'],
    ['DETECTIONS', String(s.detections)],
    ['CFAR PASS RATE', Math.round((s.cfarRate || 0) * 100) + '%'],
    ['WALL / SOFT / OPENING', s.classCounts.WALL + ' / ' + s.classCounts.SOFT + ' / ' + s.classCounts.OPENING],
    ['MEAN CLASS CONFIDENCE', Math.round((s.avgClassConfidence || 0) * 100) + '%'],
    ['RANGE SPAN', (s.minRange != null ? s.minRange.toFixed(2) : '--') + ' – ' + (s.maxRange != null ? s.maxRange.toFixed(2) : '--') + ' m'],
    ['AREA OBSERVED', ((s.freeArea || 0) + (s.occupiedArea || 0)).toFixed(1) + ' m²'],
    ['OF WHICH FREE', (s.freeArea || 0).toFixed(1) + ' m²'],
    ['RECONSTRUCTED SURFACES', r.segments + ' (' + (r.totalWallLength || 0).toFixed(1) + ' m)'],
    ['CORNERS / OPENINGS', r.corners + ' / ' + r.openings],
    ['RECONSTRUCTION CONFIDENCE', Math.round((r.confidence || 0) * 100) + '%'],
  ];
  ui.sumGrid.innerHTML = rows.map(([k, v]) =>
    '<div class="sum-row"><span class="k">' + k + '</span><span class="v">' + v + '</span></div>').join('');
  ui.sumCaveats.innerHTML = (msg.summary.caveats || []).map((c) => '<li>' + c + '</li>').join('');
  ui.sumAi.hidden = true;
  ui.summarySheet.hidden = false;
  // Bedrock, if configured, adds one paragraph after the numbers are already up.
  if (msg.bedrockAvailable) {
    ui.sumAi.hidden = false;
    ui.sumAiText.textContent = 'Requesting interpretation…';
    net.send('request_summary', { summary: msg.summary });
  }
  triggerReconstruct();
}

// ---------------------------------------------------------------------------
// Pan / zoom / keys
// ---------------------------------------------------------------------------
let dragging = false;
let lastX = 0;
let lastY = 0;

ui.stage.addEventListener('pointerdown', (e) => {
  dragging = true;
  lastX = e.clientX;
  lastY = e.clientY;
  ui.stage.setPointerCapture(e.pointerId);
});
ui.stage.addEventListener('pointermove', (e) => {
  if (!dragging) return;
  renderer.panBy(e.clientX - lastX, e.clientY - lastY);
  lastX = e.clientX;
  lastY = e.clientY;
});
ui.stage.addEventListener('pointerup', (e) => {
  dragging = false;
  try { ui.stage.releasePointerCapture(e.pointerId); } catch (err) { /* ignore */ }
});
ui.stage.addEventListener('wheel', (e) => {
  e.preventDefault();
  renderer.zoomBy(e.deltaY < 0 ? 1.12 : 0.893, e.clientX, e.clientY);
}, { passive: false });
ui.stage.addEventListener('dblclick', () => {
  renderer.follow = true;
  renderer.targetScale = 62;
});

document.addEventListener('keydown', (e) => {
  if (e.target && /input|select|textarea/i.test(e.target.tagName)) return;
  switch (e.key.toLowerCase()) {
    case 'z': ui.btnZero.click(); break;
    case 'r': ui.btnRecon.click(); break;
    case 'm': ui.btnMission.click(); break;
    case 'f': ui.btnFit.click(); break;
    case 'p': ui.btnReplay.click(); break;
    case 'c': renderer.follow = true; break;
    default: break;
  }
});

// ---------------------------------------------------------------------------
// Frame loop
// ---------------------------------------------------------------------------
let lastFrame = performance.now();

// The occupancy grid is serialised for the renderer, not per frame: at 0.1 m
// cells a full scan is tens of thousands of cells, and rebuilding that 60
// times a second is the one thing that would make this drop frames.
let gridCache = null;
let gridCacheAt = 0;
function cachedGrid() {
  const now = performance.now();
  if (!gridCache || now - gridCacheAt > 220) {
    gridCache = world.grid.serialize();
    gridCacheAt = now;
  }
  return gridCache;
}

function frame(now) {
  const dt = Math.min(0.1, (now - lastFrame) / 1000);
  lastFrame = now;
  renderer.draw({
    grid: world.serverGrid || cachedGrid(),
    cloud: world.cloud.points,
    trajectory: world.trajectory,
    reconstruction: world.reconstruction,
    pose: world.pose,
    truth: world.truth,
    lastDetection: world.lastDetection,
  }, now, dt);
  requestAnimationFrame(frame);
}

window.addEventListener('resize', () => renderer.resize());

renderer.resize();
renderChrome();
renderAi(null);
requestAnimationFrame(frame);
net.connect();
setInterval(renderStats, 500);

if (EchoNet) {
  const st = EchoNet.selfTest();
  ui.aiMeta.textContent = 'EchoNet · ' + EchoNet.META.n_params + ' params · val '
    + (EchoNet.META.val_accuracy * 100).toFixed(1) + '%' + (st.ok ? '' : ' · SELF-TEST FAILED');
}
