/**
 * End-to-end test: boot a real server, connect a real WebSocket as the command
 * center, run a simulated mission, and check that the whole chain produces
 * detections, a fused map, a reconstruction and a mission summary.
 *
 * This runs the actual server rather than mocking the hub — it is the test
 * that catches integration bugs between the twin, the fusion and the wire.
 */
const test = require('node:test');
const assert = require('node:assert');
const http = require('node:http');
const WebSocket = require('ws');

const fs = require('node:fs');
const path = require('node:path');

const staticServer = require('../server/static');
const { Hub } = require('../server/hub');
const { envelope } = require('../public/shared/protocol.mjs');

/**
 * Remove a recording a test created.  A stray test scan left on disk would
 * change which recording "REPLAY LAST" resolves to during an actual demo, and
 * the shipped reference scan is meant to be the one it finds.
 */
function cleanupRecording(id) {
  if (!id) return;
  try { fs.unlinkSync(path.join(__dirname, '..', 'recordings', id + '.json')); } catch (e) { /* already gone */ }
}

function boot() {
  return new Promise((resolve) => {
    const hub = new Hub({ log: () => {} });
    const server = http.createServer((req, res) => staticServer.serve(req, res, []));
    hub.attach(server);
    server.listen(0, '127.0.0.1', () => resolve({ hub, server, port: server.address().port }));
  });
}

function connect(port, role) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket('ws://127.0.0.1:' + port + '/ws');
    const frames = [];
    const waiters = [];
    ws.on('message', (raw) => {
      let msg;
      try { msg = JSON.parse(raw.toString('utf8')); } catch (e) { return; }
      frames.push(msg);
      for (let i = waiters.length - 1; i >= 0; i--) {
        if (waiters[i].pred(msg)) { waiters[i].resolve(msg); waiters.splice(i, 1); }
      }
    });
    ws.on('error', reject);
    ws.on('open', () => {
      ws.send(JSON.stringify(envelope('hello', { role })));
      resolve({
        ws,
        frames,
        send: (t, p) => ws.send(JSON.stringify(envelope(t, p))),
        of: (type) => frames.filter((f) => f.type === type),
        wait: (pred, ms) => new Promise((res2, rej2) => {
          const existing = frames.find(pred);
          if (existing) return res2(existing);
          const w = { pred, resolve: res2 };
          waiters.push(w);
          setTimeout(() => {
            const i = waiters.indexOf(w);
            if (i >= 0) { waiters.splice(i, 1); rej2(new Error('timeout waiting for frame')); }
          }, ms || 8000);
        }),
        close: () => ws.close(),
      });
    });
  });
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

test('simulated mission produces detections, a map and a summary', async (t) => {
  const { hub, server, port } = await boot();
  t.after(() => { hub.close(); server.close(); });

  const map = await connect(port, 'map');
  const welcome = await map.wait((m) => m.type === 'welcome');
  assert.equal(welcome.protocolVersion, 1);
  assert.equal(welcome.capabilities.classifier, 'loaded', 'EchoNet must load server-side');
  assert.ok(welcome.scenarios.length >= 5, 'scenarios should be advertised');

  map.send('set_mode', { mode: 'simulation' });
  map.send('mission_start', { scenario: 'room', rateHz: 20, record: true });

  await sleep(2500);

  const dets = map.of('detection');
  assert.ok(dets.length > 15, 'expected a stream of detections, got ' + dets.length);

  const d = dets[dets.length - 1].detection;
  assert.ok(d.range_m > 0 && d.range_m < 20, 'range must be physical');
  assert.ok(Number.isFinite(d.worldX) && Number.isFinite(d.worldY), 'world projection must be finite');
  assert.equal(d.source, 'simulation');
  assert.ok(['WALL', 'SOFT', 'OPENING', null].includes(d.obstacleClass));
  assert.ok(Array.isArray(d.classProbs) && d.classProbs.length === 3);
  const psum = d.classProbs.reduce((a, b) => a + b, 0);
  assert.ok(Math.abs(psum - 1) < 1e-3 || psum === 0, 'class probs must be a distribution');

  // The classifier must actually be running: scripted labels would not vary.
  const confs = new Set(dets.map((f) => Math.round(f.detection.classConfidence * 100)));
  assert.ok(confs.size > 3, 'class confidence should vary across echoes, not be scripted');

  const snap = await map.wait((m) => m.type === 'state_snapshot' && m.stats.detections > 10);
  assert.ok(snap.grid.idx.length > 30, 'occupancy grid should have accumulated cells');
  assert.ok(snap.cloud.length > 5, 'point cloud should have points');
  assert.ok(snap.trajectory.nodes.length > 2, 'trajectory should have moved');
  assert.ok(snap.stats.distanceScanned > 0.2, 'walker should have covered ground');

  map.send('mission_complete', {});
  const sum = await map.wait((m) => m.type === 'mission_summary');
  assert.ok(sum.summary.stats.detections > 15);
  assert.ok(sum.summary.caveats.length >= 3, 'summary must carry its honesty caveats');
  assert.ok(sum.summary.reconstruction.confidence >= 0 && sum.summary.reconstruction.confidence <= 1);
  assert.ok(sum.recording && sum.recording.id, 'mission should have saved a recording');
  cleanupRecording(sum.recording.id);

  map.close();
});

test('replay rebuilds the map from an empty canvas', async (t) => {
  const { hub, server, port } = await boot();
  t.after(() => { hub.close(); server.close(); });

  const map = await connect(port, 'map');
  await map.wait((m) => m.type === 'welcome');
  map.send('mission_start', { scenario: 'corridor', rateHz: 20 });
  await sleep(1800);
  map.send('mission_complete', {});
  const sum = await map.wait((m) => m.type === 'mission_summary');
  const recId = sum.recording.id;
  t.after(() => cleanupRecording(recId));
  assert.ok(map.of('detection').length > 10);

  map.send('replay_start', { id: recId, speed: 6 });
  const started = await map.wait((m) => m.type === 'replay_start');
  assert.equal(started.id, recId);
  assert.ok(started.frames > 10);

  const finished = await map.wait((m) => m.type === 'replay_status' && m.state === 'finished', 20000);
  assert.ok(finished);
  const replayed = map.of('detection').filter((f) => f.replay).length;
  assert.ok(replayed > 10, 'replay should re-emit detections, got ' + replayed);
});

test('malformed frames are reported, never fatal', async (t) => {
  const { hub, server, port } = await boot();
  t.after(() => { hub.close(); server.close(); });

  const map = await connect(port, 'map');
  await map.wait((m) => m.type === 'welcome');

  map.ws.send('this is not json');
  map.ws.send(JSON.stringify({ type: 'no_such_type' }));
  map.ws.send(JSON.stringify({ v: 99, type: 'hello' }));
  map.ws.send(JSON.stringify({ type: 'detection', detection: { range_m: 'banana' } }));

  const errs = [];
  for (let i = 0; i < 4; i++) {
    errs.push(await map.wait((m) => m.type === 'error' && errs.indexOf(m) === -1, 4000));
  }
  assert.equal(errs.length, 4);
  assert.ok(errs.every((e) => e.fatal === false));

  map.send('request_state', {});
  const snap = await map.wait((m) => m.type === 'state_snapshot');
  assert.ok(snap);
  assert.equal(map.ws.readyState, 1, 'socket should still be open after bad frames');
});

test('a live-mode phone detection flows through fusion to the map', async (t) => {
  const { hub, server, port } = await boot();
  t.after(() => { hub.close(); server.close(); });

  const map = await connect(port, 'map');
  const phone = await connect(port, 'phone');
  await map.wait((m) => m.type === 'welcome');
  await phone.wait((m) => m.type === 'welcome');

  map.send('set_mode', { mode: 'live' });
  map.send('mission_start', { record: false });
  await sleep(150);

  // Three looks at the same spot: fusion should accumulate support.
  for (let i = 0; i < 3; i++) {
    phone.send('detection', {
      detection: {
        t: Date.now(), range_m: 1.82, vel_mps: 0.3, confidence: 0.8,
        bearing_deg: 90, beamwidth_deg: 30, snr_db: 22, cfar_pass: true,
        obstacleClass: 'WALL', classConfidence: 0.94, classProbs: [0.94, 0.04, 0.02],
        phone: { x: 0, y: 0, heading: 90, confidence: 0.7, method: 'dead-reckoning' },
        source: 'live',
      },
    });
    await sleep(120);
  }

  const got = map.of('detection');
  assert.ok(got.length >= 3, 'map should receive every live detection');
  const last = got[got.length - 1].detection;
  assert.equal(last.source, 'live');
  assert.equal(last.fusedClass, 'WALL');
  assert.ok(last.fusedSupport >= 3, 'temporal fusion should accumulate support');
  assert.ok(Math.abs(last.worldX - 1.82) < 0.01, 'worldX should be 1.82 m east');
  assert.ok(Math.abs(last.worldY) < 0.01, 'worldY should be 0');
});
