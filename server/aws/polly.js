/**
 * Amazon Polly speech synthesis (AWS SDK v3).
 *
 * Wrapped so the rest of the system only ever sees { ok, audio } or
 * { ok:false, error }.  The SDK is an optional dependency: if it is not
 * installed, or credentials are absent, this object simply reports itself
 * unavailable and the phone speaks with the browser's SpeechSynthesis.
 *
 * Guidance phrases repeat constantly during a scan, so successful syntheses
 * are cached by text — that keeps the demo responsive and the API bill small.
 */
let PollyClient = null;
let SynthesizeSpeechCommand = null;
let sdkLoadError = null;
try {
  const sdk = require('@aws-sdk/client-polly');
  PollyClient = sdk.PollyClient;
  SynthesizeSpeechCommand = sdk.SynthesizeSpeechCommand;
} catch (e) {
  sdkLoadError = e.message;
}

class PollyVoice {
  constructor(opts = {}) {
    this.region = opts.region || '';
    this.voiceId = opts.voiceId || 'Matthew';
    this.engine = opts.engine || 'neural';
    this.sdkAvailable = !!PollyClient;
    this.sdkLoadError = sdkLoadError;
    this.enabled = !!(opts.enabled && this.sdkAvailable && this.region);
    this.client = null;
    this.cache = new Map();
    this.maxCache = 48;
  }

  ensureClient() {
    if (!this.client && this.enabled) {
      // No credentials are passed explicitly: the SDK's default provider chain
      // reads the environment, shared config, or the instance role.
      this.client = new PollyClient({ region: this.region });
    }
    return this.client;
  }

  /**
   * @returns {Promise<{ok:boolean, audio?:string, cached?:boolean, error?:string, fatal?:boolean}>}
   *          audio is base64 MP3.
   */
  async synthesize(text) {
    if (!this.enabled) return { ok: false, error: 'Polly not enabled', fatal: false };
    const key = this.voiceId + '|' + this.engine + '|' + text;
    const hit = this.cache.get(key);
    if (hit) return { ok: true, audio: hit, cached: true };

    try {
      const client = this.ensureClient();
      const out = await client.send(new SynthesizeSpeechCommand({
        Text: text,
        OutputFormat: 'mp3',
        VoiceId: this.voiceId,
        Engine: this.engine,
        SampleRate: '24000',
      }));
      const bytes = await streamToBuffer(out.AudioStream);
      const b64 = bytes.toString('base64');
      if (this.cache.size >= this.maxCache) this.cache.delete(this.cache.keys().next().value);
      this.cache.set(key, b64);
      return { ok: true, audio: b64 };
    } catch (e) {
      // Credential and permission errors will not fix themselves mid-demo, so
      // they switch the provider off for good; a timeout might, so it does not.
      const name = e.name || '';
      const fatal = /Credential|AccessDenied|UnrecognizedClient|InvalidSignature|AuthFailure|Unauthorized/i.test(name + ' ' + e.message);
      return { ok: false, error: (name ? name + ': ' : '') + e.message, fatal };
    }
  }
}

async function streamToBuffer(stream) {
  if (!stream) throw new Error('Polly returned no audio stream');
  if (typeof stream.transformToByteArray === 'function') {
    return Buffer.from(await stream.transformToByteArray());
  }
  const chunks = [];
  for await (const c of stream) chunks.push(Buffer.isBuffer(c) ? c : Buffer.from(c));
  return Buffer.concat(chunks);
}

module.exports = { PollyVoice };
