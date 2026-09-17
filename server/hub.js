/**
 * Session hub — the message boundary in one place.
 *
 * Owns every socket, the authoritative MapState, the simulation engine, the
 * recorder, the replay player and the AWS adapter.  Phones publish detections;
 * command centers subscribe.  Nothing else in the system knows how those two
 * halves find each other.
 *
 * Robustness rules, all of them learned from demos going wrong:
 *   - A malformed frame produces an `error` message, never a closed socket.
 *   - Map state survives a phone disconnect; the map re-syncs from a snapshot.
 *   - A dead peer is detected by heartbeat, not by hoping `close` fires.
 *   - Simulation can run with no phone connected at all.
 */
const fs = require('fs');
const path = require('path');
const { WebSocketServer } = require('ws');
const { validateMessage, envelope, makePose, PROTOCOL_VERSION, MESSAGE_TYPES } = require('../public/shared/protocol.mjs');
const { MapState } = require('./mapstate');
const { SimulationEngine, SCENARIO_LIST, EchoNet, echonetError } = require('./simulation');
const { Recorder, ReplayPlayer } = require('./recorder');
const { GuidancePolicy } = require('./guidance');
const { AwsAdapter } = require('./aws/adapter');

const HEARTBEAT_MS = 4000;
const PEER_TIMEOUT_MS = 14000;
/** Broadcast the fused state this often; detections stream in between. */
const SNAPSHOT_MS = 1200;
/** Roles that consume the fused stream (the phone produces it). */
const OBSERVERS = ['map', 'diagnostics'];

class Hub {
  constructor(opts = {}) {
    this.clients = new Map();          // id -> client record
    this.nextId = 1;
    this.mode = 'simulation';          // safe default: works with no phone
    this.map = new MapState();
    this.recorder = new Recorder();
    this.guidance = new GuidancePolicy();
    this.aws = new AwsAdapter(process.env);
    this.replay = null;
    this.lastDetectionAt = 0;
    this.stats = { framesIn: 0, framesOut: 0, badFrames: 0, detections: 0, reconnects: 0 };
    this.log = opts.log || (() => {});
    this.servers = [];

    this.sim = new SimulationEngine({
      onDetection: (det, pose) => this.onSimFrame(det, pose),
      onStatus: (s) => this.log('sim', s),
    });

    this.heartbeatTimer = setInterval(() => this.heartbeat(), HEARTBEAT_MS);
    this.snapshotTimer = setInterval(() => this.pushSnapshot(), SNAPSHOT_MS);
    if (this.heartbeatTimer.unref) this.heartbeatTimer.unref();
    if (this.snapshotTimer.unref) this.snapshotTimer.unref();
  }

  attach(server) {
    const wss = new WebSocketServer({ server, path: '/ws' });
    wss.on('connection', (ws, req) => this.onConnection(ws, req));
    // One hub, potentially two listeners (HTTP for the laptop, HTTPS for the
    // phone's secure context). Both feed the same sessions.
    this.servers.push(wss);
    this.wss = wss;
    return wss;
  }

  // -------------------------------------------------------------------------
  // Connection lifecycle
  // -------------------------------------------------------------------------
  onConnection(ws, req) {
    const id = this.nextId++;
    const client = {
      id, ws, role: 'unknown', alive: true, lastSeen: Date.now(),
      latencyMs: null, ip: (req && req.socket && req.socket.remoteAddress) || '?',
      userAgent: (req && req.headers && req.headers['user-agent']) || '',
      connectedAt: Date.now(),
    };
    this.clients.set(id, client);

    ws.on('message', (data) => this.onMessage(client, data));
    ws.on('pong', () => { client.alive = true; client.lastSeen = Date.now(); });
    ws.on('close', () => this.onClose(client));
    ws.on('error', () => this.onClose(client));

    this.send(client, envelope('welcome', {
      sessionId: id,
      protocolVersion: PROTOCOL_VERSION,
      mode: this.mode,
      capabilities: this.capabilities(),
      scenarios: SCENARIO_LIST,
      simulation: this.sim.info(),
      aws: this.aws.status(),
      recordings: this.recorder.list().slice(0, 12),
      recording: this.recorder.activeInfo(),
      serverTime: Date.now(),
    }));
    this.log('ws', { event: 'connect', id, ip: client.ip });
    this.broadcastStatus();
  }

  onClose(client) {
    if (!this.clients.has(client.id)) return;
    this.clients.delete(client.id);
    this.log('ws', { event: 'disconnect', id: client.id, role: client.role });
    // Map state is deliberately NOT cleared: the phone may be reconnecting,
    // and losing the scan because WiFi blinked would be unforgivable mid-demo.
    this.broadcastStatus();
  }

  heartbeat() {
    const now = Date.now();
    for (const client of Array.from(this.clients.values())) {
      if (now - client.lastSeen > PEER_TIMEOUT_MS) {
        try { client.ws.terminate(); } catch (e) { /* already gone */ }
        this.onClose(client);
        continue;
      }
      try {
        client.ws.ping();
        this.send(client, envelope('heartbeat', { serverTime: now, peers: this.peerSummary() }));
      } catch (e) { this.onClose(client); }
    }
  }

  // -------------------------------------------------------------------------
  // Messages
  // -------------------------------------------------------------------------
  onMessage(client, data) {
    this.stats.framesIn++;
    const result = validateMessage(typeof data === 'string' ? data : data.toString('utf8'));
    if (!result.ok) {
      this.stats.badFrames++;
      // Report and carry on.  One bad packet must never take down a session.
      this.send(client, envelope('error', { error: result.error, fatal: false }));
      return;
    }
    client.lastSeen = Date.now();
    const msg = result.msg;

    try {
      this.dispatch(client, msg);
    } catch (e) {
      this.stats.badFrames++;
      this.send(client, envelope('error', { error: 'handler failed: ' + e.message, type: msg.type, fatal: false }));
      this.log('error', { where: 'dispatch', type: msg.type, message: e.message });
    }
  }

  dispatch(client, msg) {
    switch (msg.type) {
      case 'hello':
        client.role = msg.role || 'unknown';
        client.deviceInfo = msg.device || null;
        this.log('ws', { event: 'hello', id: client.id, role: client.role });
        this.send(client, envelope('status', this.statusPayload()));
        if (client.role === 'map') this.send(client, envelope('state_snapshot', this.map.snapshot({ reconstruct: true })));
        this.broadcastStatus();
        break;

      case 'heartbeat':
        if (msg.clientTime) client.latencyMs = Date.now() - msg.clientTime;
        break;

      case 'detection':
        this.ingestDetection(msg.detection, client);
        break;

      case 'pose': {
        const pose = this.map.applyPose(msg.pose || msg);
        if (this.mode === 'hybrid') this.sim.setExternalPose(pose);
        this.recorder.push('pose', pose);
        this.broadcast(envelope('pose', { pose: pose }), OBSERVERS);
        break;
      }

      case 'calibration':
        this.calibration = {
          factor: msg.factor, offsetM: msg.offsetM, samples: msg.samples,
          knownDistanceM: msg.knownDistanceM, at: Date.now(),
        };
        this.log('calibration', this.calibration);
        this.broadcast(envelope('status', this.statusPayload()));
        break;

      case 'set_mode':
        this.setMode(msg.mode);
        break;

      case 'scan_start':
        this.scanning = true;
        if (this.mode !== 'live') this.sim.start(msg.rateHz || 20);
        this.broadcast(envelope('scan_start', { mode: this.mode }));
        this.broadcastStatus();
        break;

      case 'scan_stop':
        this.scanning = false;
        this.sim.stop();
        this.broadcast(envelope('scan_stop', {}));
        this.broadcastStatus();
        break;

      case 'mission_start':
        this.map.startMission();
        this.guidance.reset();
        this.lastDetectionAt = Date.now();
        if (msg.scenario) {
          this.sim.setScenario(msg.scenario, msg.seed);
          if (this.mode === 'live') this.setMode('simulation');
        }
        if (this.mode !== 'live') this.sim.start(msg.rateHz || 20);
        this.scanning = true;
        if (msg.record !== false) this.startRecording();
        this.broadcast(envelope('mission_start', {
          mode: this.mode, simulation: this.sim.info(), at: Date.now(),
        }));
        this.pushSnapshot(true);
        break;

      case 'mission_complete':
        this.completeMission();
        break;

      case 'reconstruct': {
        const recon = this.map.buildReconstruction(true);
        this.broadcast(envelope('state_snapshot', Object.assign(this.map.snapshot(), { reconstruction: recon, reconstructTrigger: true })));
        break;
      }

      case 'sim_control':
        this.simControl(msg);
        break;

      case 'record_start':
        this.startRecording(msg.note);
        this.broadcastStatus();
        break;

      case 'record_stop': {
        const rec = this.recorder.stop();
        this.broadcast(envelope('recording_list', { recordings: this.recorder.list().slice(0, 12), saved: rec }));
        this.broadcastStatus();
        break;
      }

      case 'replay_start':
        this.startReplay(msg.id, msg.speed);
        break;

      case 'replay_stop':
        this.stopReplay('aborted');
        break;

      case 'request_state':
        this.send(client, envelope('state_snapshot', this.map.snapshot({ reconstruct: true })));
        break;

      case 'request_summary':
        this.requestAwsSummary(client, msg.summary);
        break;

      case 'speak':
        this.speak(client, msg.text, msg.level);
        break;

      case 'save_training_dataset': {
        try {
          const dataset = msg.dataset || {};
          const samples = Array.isArray(dataset.samples) ? dataset.samples : [];
          const nowStr = new Date().toISOString().replace(/[:.]/g, '-');
          const filename = `real_echo_dataset_${nowStr}.json`;
          const dir = path.join(__dirname, '..', 'recordings');
          if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
          const targetPath = path.join(dir, filename);
          const latestPath = path.join(dir, 'real_training_dataset_latest.json');
          const payload = {
            created_at: new Date().toISOString(),
            device: dataset.device || 'OnePlus',
            total_samples: samples.length,
            stats: dataset.stats || {},
            samples: samples,
          };
          fs.writeFileSync(targetPath, JSON.stringify(payload, null, 2));
          fs.writeFileSync(latestPath, JSON.stringify(payload, null, 2));
          this.send(client, envelope('training_dataset_saved', {
            ok: true,
            filename,
            count: samples.length,
            stats: dataset.stats,
          }));
        } catch (err) {
          this.send(client, envelope('error', { error: 'Failed to save dataset: ' + err.message, fatal: false }));
        }
        break;
      }

      default:
        // Known-but-unhandled types are server-originated; ignore quietly.
        if (MESSAGE_TYPES[msg.type] && MESSAGE_TYPES[msg.type].from === 'server') break;
        this.send(client, envelope('error', { error: 'unhandled type ' + msg.type, fatal: false }));
    }
  }

  // -------------------------------------------------------------------------
  // Detection path
  // -------------------------------------------------------------------------
  /**
   * The one funnel every detection passes through, whatever produced it:
   * live phone DSP, the digital twin, or a replay.  Fusion, mapping,
   * recording, guidance and broadcast all happen here exactly once.
   */
  ingestDetection(det, client) {
    if (!det) return;
    // In live mode, ignore synthetic frames and vice versa, so a stray sim
    // timer cannot pollute a live scan.
    const fused = this.map.applyDetection(det);
    this.stats.detections++;
    this.lastDetectionAt = Date.now();

    this.recorder.push('detection', fused);
    this.broadcast(envelope('detection', { detection: fused }), OBSERVERS);

    const cue = this.guidance.evaluate(fused);
    if (cue) this.emitGuidance(cue, fused);
    return fused;
  }

  onSimFrame(det, pose) {
    if (this.mode === 'live') return;          // live mode owns the pipeline
    if (!det) {
      if (pose) {
        this.map.applyPose(pose);
        this.broadcast(envelope('pose', { pose }), OBSERVERS);
      }
      // Sustained silence is itself a cue worth speaking.
      const quiet = Date.now() - this.lastDetectionAt;
      const cue = this.guidance.evaluate(null, { quietMs: quiet });
      if (cue) this.emitGuidance(cue, null);
      return;
    }
    this.ingestDetection(det);
  }

  /** Guidance goes to the phone (to speak) and the map (to display). */
  async emitGuidance(cue, det) {
    const payload = {
      text: cue.text, level: cue.level, key: cue.key, reason: cue.reason,
      range_m: det ? det.range_m : null,
      bearing_deg: det ? det.bearing_deg : null,
      obstacleClass: det ? det.fusedClass : null,
      classConfidence: det ? det.fusedConfidence : null,
    };

    // Resolve the voice provider here so both screens agree on the label.
    let voice = { provider: 'local', reason: this.aws.voiceUnavailableReason() };
    try {
      voice = await this.aws.speak(cue.text);
    } catch (e) {
      voice = { provider: 'local', reason: 'adapter threw: ' + e.message };
    }
    payload.voice = {
      provider: voice.provider,
      label: voice.provider === 'polly' ? 'AWS POLLY' : 'LOCAL FALLBACK',
      reason: voice.reason || null,
      cached: !!voice.cached,
    };
    this.broadcast(envelope('guidance', payload));
    if (voice.provider === 'polly' && voice.audio) {
      // Audio only to the phone: it is the thing with a speaker in hand.
      this.broadcast(envelope('voice_audio', {
        audio: voice.audio, format: voice.format, text: cue.text, voiceId: voice.voiceId,
      }), 'phone');
    }
  }

  async speak(client, text, level) {
    if (!text) return;
    let voice = { provider: 'local' };
    try { voice = await this.aws.speak(String(text).slice(0, 400)); } catch (e) { voice = { provider: 'local', reason: e.message }; }
    const payload = {
      text, level: level || 'info',
      voice: {
        provider: voice.provider,
        label: voice.provider === 'polly' ? 'AWS POLLY' : 'LOCAL FALLBACK',
        reason: voice.reason || null,
      },
    };
    this.send(client, envelope('guidance', payload));
    if (voice.provider === 'polly' && voice.audio) {
      this.send(client, envelope('voice_audio', { audio: voice.audio, format: voice.format, text }));
    }
    this.broadcast(envelope('guidance', payload), OBSERVERS);
  }

  // -------------------------------------------------------------------------
  // Mode / simulation
  // -------------------------------------------------------------------------
  setMode(mode) {
    this.mode = mode;
    this.sim.setMode(mode);
    if (mode === 'live') this.sim.stop();
    else if (this.scanning) this.sim.start();
    this.log('mode', { mode });
    this.broadcast(envelope('status', this.statusPayload()));
  }

  simControl(msg) {
    if (msg.scenario) this.sim.setScenario(msg.scenario, msg.seed);
    if (msg.speed != null) this.sim.setSpeed(Number(msg.speed));
    if (msg.steer !== undefined) this.sim.setSteer(msg.steer);
    if (msg.paused != null) this.sim.setPaused(!!msg.paused);
    if (msg.action === 'start') this.sim.start(msg.rateHz);
    if (msg.action === 'stop') this.sim.stop();
    if (msg.action === 'reset') { this.sim.setScenario(this.sim.scenarioId, msg.seed); this.map.reset(); }
    this.broadcast(envelope('status', this.statusPayload()));
  }

  // -------------------------------------------------------------------------
  // Mission / recording / replay
  // -------------------------------------------------------------------------
  startRecording(note) {
    if (this.recorder.isRecording()) return this.recorder.activeInfo();
    const info = this.recorder.start({
      mode: this.mode,
      scenario: this.mode === 'live' ? null : this.sim.scenarioId,
      note: note || '',
    });
    this.log('record', { event: 'start', id: info.id });
    return info;
  }

  completeMission() {
    const summary = this.map.endMission();
    this.scanning = false;
    this.sim.stop();
    this.recorder.setSummary(summary);
    const saved = this.recorder.isRecording() ? this.recorder.stop() : null;
    this.broadcast(envelope('mission_summary', {
      summary,
      recording: saved,
      recordings: this.recorder.list().slice(0, 12),
      bedrockAvailable: this.aws.bedrock.enabled,
    }));
    this.broadcastStatus();
    return summary;
  }

  startReplay(id, speed) {
    this.stopReplay('superseded');
    const rec = id ? this.recorder.load(id) : this.recorder.latest();
    if (!rec) {
      this.broadcast(envelope('error', { error: 'no recording available to replay', fatal: false }));
      return;
    }
    // A replay rebuilds the map from an empty canvas through the real
    // pipeline, so the reconstruction you see is recomputed, not a screenshot.
    this.map.reset();
    this.map.missionActive = true;
    this.map.stats.missionStart = Date.now();
    this.guidance.reset();
    this.sim.stop();
    this.scanning = false;

    this.broadcast(envelope('replay_start', {
      id: rec.id, frames: rec.frameCount || (rec.frames || []).length,
      mode: rec.mode, scenario: rec.scenario, speed: speed || 1.5,
      durationMs: rec.durationMs,
    }));

    this.replay = new ReplayPlayer(rec, {
      speed: speed || 1.5,
      onFrame: (f) => {
        if (f.k === 'detection') {
          const det = Object.assign({}, f.p, { source: 'replay' });
          const fused = this.map.applyDetection(det);
          this.broadcast(envelope('detection', { detection: fused, replay: true }), OBSERVERS);
        } else if (f.k === 'pose') {
          const pose = this.map.applyPose(f.p);
          this.broadcast(envelope('pose', { pose, replay: true }), OBSERVERS);
        }
      },
      onProgress: (p) => this.broadcast(envelope('replay_status', Object.assign({ state: 'playing' }, p))),
      onDone: (d) => {
        this.replay = null;
        this.map.buildReconstruction(true);
        this.pushSnapshot(true);
        this.broadcast(envelope('replay_status', { state: 'finished', id: d.id, summary: d.summary }));
      },
    });
    this.replay.start();
  }

  stopReplay(reason) {
    if (this.replay) {
      this.replay.stop();
      this.replay = null;
      this.broadcast(envelope('replay_status', { state: 'stopped', reason: reason || 'stopped' }));
    }
  }

  async requestAwsSummary(client, summaryOverride) {
    const summary = summaryOverride || this.map.summary();
    const res = await this.aws.summarizeScan(summary);
    this.broadcast(envelope('scan_summary', Object.assign({ summary }, res)));
  }

  // -------------------------------------------------------------------------
  // Broadcast helpers
  // -------------------------------------------------------------------------
  send(client, obj) {
    if (!client || !client.ws || client.ws.readyState !== 1) return false;
    try {
      client.ws.send(JSON.stringify(obj));
      this.stats.framesOut++;
      return true;
    } catch (e) {
      this.onClose(client);
      return false;
    }
  }

  /** @param {string|string[]} [role] restrict to one role, or any of several */
  broadcast(obj, role) {
    const json = JSON.stringify(obj);
    const roles = role == null ? null : Array.isArray(role) ? role : [role];
    for (const client of Array.from(this.clients.values())) {
      if (roles && roles.indexOf(client.role) < 0) continue;
      if (!client.ws || client.ws.readyState !== 1) continue;
      try { client.ws.send(json); this.stats.framesOut++; } catch (e) { this.onClose(client); }
    }
  }

  pushSnapshot(force) {
    if (!this.clients.size) return;
    const hasObserver = Array.from(this.clients.values()).some((c) => OBSERVERS.indexOf(c.role) >= 0);
    if (!hasObserver && !force) return;
    // Reconstruction is the expensive part; run it on the snapshot cadence
    // rather than per detection.
    this.broadcast(envelope('state_snapshot', this.map.snapshot({ reconstruct: true })), OBSERVERS);
  }

  peerSummary() {
    const out = { phone: 0, map: 0, diagnostics: 0, unknown: 0 };
    for (const c of this.clients.values()) out[c.role] = (out[c.role] || 0) + 1;
    return out;
  }

  capabilities() {
    return {
      classifier: EchoNet ? 'loaded' : 'error',
      classifierError: echonetError,
      classifierMeta: EchoNet ? EchoNet.META : null,
      simulation: true,
      replay: true,
      polly: this.aws.polly.enabled,
      bedrock: this.aws.bedrock.enabled,
      modes: ['live', 'simulation', 'hybrid'],
    };
  }

  statusPayload() {
    return {
      mode: this.mode,
      scanning: !!this.scanning,
      peers: this.peerSummary(),
      clients: Array.from(this.clients.values()).map((c) => ({
        id: c.id, role: c.role, latencyMs: c.latencyMs, ip: c.ip,
        connectedForMs: Date.now() - c.connectedAt,
      })),
      simulation: this.sim.info(),
      recording: this.recorder.activeInfo(),
      replaying: !!this.replay,
      calibration: this.calibration || null,
      aws: this.aws.status(),
      capabilities: this.capabilities(),
      stats: Object.assign({}, this.stats),
      missionActive: this.map.missionActive,
      guidanceHistory: this.guidance.history.slice(-8),
    };
  }

  broadcastStatus() { this.broadcast(envelope('status', this.statusPayload())); }

  close() {
    clearInterval(this.heartbeatTimer);
    clearInterval(this.snapshotTimer);
    this.sim.stop();
    this.stopReplay('shutdown');
    // wss.close() stops new connections but leaves established sockets open,
    // which would keep the process (and the test runner) alive forever.
    for (const client of Array.from(this.clients.values())) {
      try { client.ws.terminate(); } catch (e) { /* already gone */ }
    }
    this.clients.clear();
    for (const wss of this.servers) {
      try { wss.close(); } catch (e) { /* already closed */ }
    }
    this.servers = [];
  }
}

module.exports = { Hub };
