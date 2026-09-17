/**
 * Reconnecting WebSocket client, shared by all three pages.
 *
 * Venue WiFi is the single most likely thing to break a live demo, so this is
 * written to survive it:
 *   - exponential backoff with jitter, capped, and reconnection is automatic
 *   - the `hello` handshake is replayed on every reconnect, so the server
 *     re-learns the role without the page doing anything
 *   - outbound messages queue (bounded) while offline instead of being lost
 *   - round-trip latency is measured continuously from heartbeat echoes
 *   - state is reported as a single enum the UI can render directly
 */
import { envelope, validateMessage, PROTOCOL_VERSION } from './protocol.mjs';

export const STATES = ['offline', 'connecting', 'connected', 'reconnecting'];

export class WsClient {
  /**
   * @param {Object} opts
   * @param {string} opts.role 'phone' | 'map' | 'diagnostics'
   * @param {(msg:Object)=>void} opts.onMessage
   * @param {(state:string, detail:Object)=>void} [opts.onState]
   */
  constructor(opts = {}) {
    this.role = opts.role || 'map';
    this.onMessage = opts.onMessage || (() => {});
    this.onState = opts.onState || (() => {});
    this.device = opts.device || null;

    this.url = opts.url || defaultUrl();
    this.ws = null;
    this.state = 'offline';
    this.attempt = 0;
    this.queue = [];
    this.maxQueue = 200;
    this.latencyMs = null;
    this.latencySamples = [];
    this.lastServerTime = 0;
    this.sessionId = null;
    this.serverCapabilities = null;
    this.closedByUs = false;
    this.reconnectTimer = null;
    this.heartbeatTimer = null;
    this.stats = { sent: 0, received: 0, reconnects: 0, dropped: 0, errors: 0 };
  }

  connect() {
    this.closedByUs = false;
    if (this.ws && (this.ws.readyState === 0 || this.ws.readyState === 1)) return;
    this.setState(this.attempt === 0 ? 'connecting' : 'reconnecting');

    let ws;
    try {
      ws = new WebSocket(this.url);
    } catch (e) {
      this.stats.errors++;
      this.scheduleReconnect();
      return;
    }
    this.ws = ws;

    ws.onopen = () => {
      this.attempt = 0;
      this.setState('connected');
      // Re-announce on every connect, including reconnects: the server treats
      // this socket as brand new and needs the role again.
      this.sendNow(envelope('hello', {
        role: this.role,
        device: this.device,
        clientTime: Date.now(),
      }));
      this.flush();
      this.startHeartbeat();
    };

    ws.onmessage = (ev) => {
      this.stats.received++;
      const res = validateMessage(typeof ev.data === 'string' ? ev.data : String(ev.data));
      if (!res.ok) {
        // An unparsable frame from the server is logged, not thrown: the demo
        // continues on the frames that are fine.
        this.stats.errors++;
        return;
      }
      const msg = res.msg;
      if (msg.type === 'welcome') {
        this.sessionId = msg.sessionId;
        this.serverCapabilities = msg.capabilities || null;
      }
      if (msg.type === 'heartbeat') {
        this.lastServerTime = msg.serverTime || Date.now();
        // Echo it back stamped with our clock so the server can measure too.
        this.sendNow(envelope('heartbeat', { clientTime: Date.now() }));
      }
      if (msg.type === 'pong' || (msg.type === 'heartbeat' && msg.echo)) this.notePong(msg);
      this.onMessage(msg);
    };

    ws.onerror = () => { this.stats.errors++; };

    ws.onclose = () => {
      this.stopHeartbeat();
      if (this.closedByUs) { this.setState('offline'); return; }
      this.stats.reconnects++;
      this.scheduleReconnect();
    };
  }

  scheduleReconnect() {
    if (this.reconnectTimer || this.closedByUs) return;
    this.setState('reconnecting');
    this.attempt++;
    // Backoff with jitter: fast enough that a brief WiFi blip is invisible,
    // slow enough that a dead server is not hammered.
    const base = Math.min(8000, 400 * Math.pow(1.7, Math.min(this.attempt, 8)));
    const delay = base * (0.7 + Math.random() * 0.6);
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, delay);
  }

  startHeartbeat() {
    this.stopHeartbeat();
    this.heartbeatTimer = setInterval(() => {
      this.pingSentAt = Date.now();
      this.sendNow(envelope('heartbeat', { clientTime: this.pingSentAt }));
    }, 3000);
  }

  stopHeartbeat() {
    if (this.heartbeatTimer) { clearInterval(this.heartbeatTimer); this.heartbeatTimer = null; }
  }

  notePong() {
    if (!this.pingSentAt) return;
    const rtt = Date.now() - this.pingSentAt;
    this.latencySamples.push(rtt);
    if (this.latencySamples.length > 8) this.latencySamples.shift();
    const sorted = this.latencySamples.slice().sort((a, b) => a - b);
    this.latencyMs = sorted[Math.floor(sorted.length / 2)];   // median
  }

  /** Estimate RTT from the server's heartbeat cadence when no echo is available. */
  measureFromServer(serverTime) {
    if (!serverTime) return;
    const skew = Math.abs(Date.now() - serverTime);
    if (this.latencyMs == null || skew < this.latencyMs * 4) this.latencyMs = skew;
  }

  send(type, payload) {
    const msg = envelope(type, payload);
    if (this.isOpen()) return this.sendNow(msg);
    // Queue while offline.  Bounded, and detections are dropped in preference
    // to control messages — a stale detection is worthless, a queued
    // mission_start is not.
    if (this.queue.length >= this.maxQueue) {
      const i = this.queue.findIndex((m) => m.type === 'detection' || m.type === 'pose');
      if (i >= 0) this.queue.splice(i, 1);
      else this.queue.shift();
      this.stats.dropped++;
    }
    this.queue.push(msg);
    return false;
  }

  sendNow(msg) {
    if (!this.isOpen()) return false;
    try {
      this.ws.send(JSON.stringify(msg));
      this.stats.sent++;
      return true;
    } catch (e) {
      this.stats.errors++;
      return false;
    }
  }

  flush() {
    const pending = this.queue.splice(0, this.queue.length);
    for (const msg of pending) this.sendNow(msg);
  }

  isOpen() { return !!this.ws && this.ws.readyState === 1; }

  setState(s) {
    if (this.state === s) return;
    this.state = s;
    this.onState(s, {
      attempt: this.attempt,
      latencyMs: this.latencyMs,
      queued: this.queue.length,
      sessionId: this.sessionId,
    });
  }

  close() {
    this.closedByUs = true;
    this.stopHeartbeat();
    if (this.reconnectTimer) { clearTimeout(this.reconnectTimer); this.reconnectTimer = null; }
    if (this.ws) { try { this.ws.close(); } catch (e) { /* ignore */ } }
    this.setState('offline');
  }

  info() {
    return {
      state: this.state,
      url: this.url,
      latencyMs: this.latencyMs,
      queued: this.queue.length,
      sessionId: this.sessionId,
      protocolVersion: PROTOCOL_VERSION,
      stats: Object.assign({}, this.stats),
    };
  }
}

/** ws:// for http pages, wss:// for https — required for a secure context. */
export function defaultUrl() {
  if (typeof location === 'undefined') return 'ws://localhost:8000/ws';
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  return proto + '//' + location.host + '/ws';
}
