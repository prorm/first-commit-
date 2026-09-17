/**
 * Record / replay.
 *
 * The single most valuable demo-reliability feature in the project: a scan
 * that worked once can be replayed exactly, from an empty canvas, with no
 * phone, no microphone and no network.  If the venue's acoustics or WiFi
 * collapse during the presentation, the replay still tells the whole story.
 *
 * A recording is a flat, time-stamped frame list — detections and pose updates
 * in capture order — plus the mission summary.  Replaying it pushes those
 * frames back through the *same* MapState pipeline that live data uses, so the
 * occupancy grid, reconstruction and statistics are rebuilt rather than
 * replayed as pictures.
 */
const fs = require('node:fs');
const path = require('node:path');

const DIR = path.join(__dirname, '..', 'recordings');

class Recorder {
  constructor() {
    this.active = null;
    this.ensureDir();
  }

  ensureDir() {
    try { fs.mkdirSync(DIR, { recursive: true }); } catch (e) { /* non-fatal */ }
  }

  start(meta = {}) {
    this.active = {
      id: 'scan-' + new Date().toISOString().replace(/[:.]/g, '-'),
      startedAt: Date.now(),
      mode: meta.mode || 'unknown',
      scenario: meta.scenario || null,
      note: meta.note || '',
      frames: [],
      summary: null,
    };
    return { id: this.active.id, startedAt: this.active.startedAt };
  }

  /** @param {'detection'|'pose'} kind */
  push(kind, payload) {
    if (!this.active) return false;
    // Frames are stored with a relative offset so a replay can be re-timed.
    this.active.frames.push({ dt: Date.now() - this.active.startedAt, k: kind, p: payload });
    if (this.active.frames.length > 40000) this.active.frames.shift();
    return true;
  }

  setSummary(summary) { if (this.active) this.active.summary = summary; }

  stop() {
    if (!this.active) return null;
    const rec = this.active;
    rec.endedAt = Date.now();
    rec.durationMs = rec.endedAt - rec.startedAt;
    rec.frameCount = rec.frames.length;
    this.active = null;
    try {
      fs.writeFileSync(path.join(DIR, rec.id + '.json'), JSON.stringify(rec));
    } catch (e) {
      return Object.assign({ persistError: e.message }, this.describe(rec));
    }
    return this.describe(rec);
  }

  describe(rec) {
    return {
      id: rec.id, startedAt: rec.startedAt, endedAt: rec.endedAt,
      durationMs: rec.durationMs, frameCount: rec.frameCount,
      mode: rec.mode, scenario: rec.scenario, note: rec.note,
      detections: rec.frames ? rec.frames.filter((f) => f.k === 'detection').length : undefined,
    };
  }

  /** Newest first, so the UI's default "replay last scan" is index 0. */
  list() {
    this.ensureDir();
    let files = [];
    try { files = fs.readdirSync(DIR).filter((f) => f.endsWith('.json') && f !== 'README.json'); } catch (e) { return []; }
    const out = [];
    for (const f of files) {
      try {
        const rec = JSON.parse(fs.readFileSync(path.join(DIR, f), 'utf8'));
        const info = this.describe(rec);
        // The id is what `load()` resolves, so it must be the filename. A
        // hand-renamed or hand-edited recording would otherwise list fine and
        // then fail to open — which is exactly the kind of silent failure that
        // ruins a fallback you were relying on.
        info.id = path.basename(f, '.json');
        out.push(info);
      } catch (e) { /* skip unreadable recording */ }
    }
    return out.sort((a, b) => (b.startedAt || 0) - (a.startedAt || 0));
  }

  load(id) {
    const safe = String(id || '').replace(/[^a-zA-Z0-9._-]/g, '');
    if (!safe) return null;
    try {
      return JSON.parse(fs.readFileSync(path.join(DIR, safe + '.json'), 'utf8'));
    } catch (e) { return null; }
  }

  latest() {
    const l = this.list();
    return l.length ? this.load(l[0].id) : null;
  }

  isRecording() { return !!this.active; }

  activeInfo() {
    if (!this.active) return null;
    return {
      id: this.active.id,
      frames: this.active.frames.length,
      elapsedMs: Date.now() - this.active.startedAt,
    };
  }
}

/**
 * Plays a recording back at a chosen speed, re-emitting frames through a
 * callback.  Uses one self-rescheduling timer so playback stays aligned to the
 * recorded timing even if a frame's processing runs long.
 */
class ReplayPlayer {
  constructor(recording, opts = {}) {
    this.rec = recording;
    this.speed = opts.speed || 1.5;
    this.onFrame = opts.onFrame || (() => {});
    this.onProgress = opts.onProgress || (() => {});
    this.onDone = opts.onDone || (() => {});
    this.idx = 0;
    this.timer = null;
    this.startedAt = 0;
  }

  start() {
    this.stop();
    this.idx = 0;
    this.startedAt = Date.now();
    this.step();
  }

  step() {
    const frames = this.rec.frames || [];
    if (this.idx >= frames.length) { this.finish(); return; }

    const elapsed = (Date.now() - this.startedAt) * this.speed;
    // Emit every frame whose recorded time has arrived.
    let emitted = 0;
    while (this.idx < frames.length && frames[this.idx].dt <= elapsed && emitted < 60) {
      const f = frames[this.idx++];
      try { this.onFrame(f); } catch (e) { /* one bad frame must not stop a replay */ }
      emitted++;
    }
    this.onProgress({
      index: this.idx,
      total: frames.length,
      fraction: frames.length ? this.idx / frames.length : 1,
    });
    if (this.idx >= frames.length) { this.finish(); return; }

    const nextAt = frames[this.idx].dt;
    const wait = Math.max(8, (nextAt - elapsed) / this.speed);
    // Deliberately NOT unref'd: a replay in progress is real pending work with
    // a completion callback, and unref'ing it lets the process drain and
    // abandon the replay half-finished. `stop()` is the way to end it early.
    this.timer = setTimeout(() => this.step(), Math.min(wait, 250));
  }

  finish() {
    this.stop();
    this.onDone({ summary: this.rec.summary, id: this.rec.id });
  }

  stop() {
    if (this.timer) { clearTimeout(this.timer); this.timer = null; }
  }

  get running() { return !!this.timer; }
}

module.exports = { Recorder, ReplayPlayer, RECORDINGS_DIR: DIR };
