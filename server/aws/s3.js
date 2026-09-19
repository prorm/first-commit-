/**
 * Optional Amazon S3 archive for finished recordings.
 *
 * Recordings are always written to local disk first — that is what replay
 * reads, and it must keep working with no network.  S3 is a durable second
 * copy, uploaded after the local write succeeds and never awaited by the scan
 * loop.  If the SDK is missing, no bucket is configured, or an upload fails,
 * the recording is simply not archived; nothing else changes.
 */
let S3Client = null;
let PutObjectCommand = null;
let sdkLoadError = null;
try {
  const sdk = require('@aws-sdk/client-s3');
  S3Client = sdk.S3Client;
  PutObjectCommand = sdk.PutObjectCommand;
} catch (e) {
  sdkLoadError = e.message;
}

class S3Archive {
  constructor(opts = {}) {
    this.region = opts.region || '';
    this.bucket = opts.bucket || '';
    // Keys look like "<prefix><recording id>.json"; keep the prefix ending in "/".
    const p = (opts.prefix || 'recordings/').replace(/^\/+/, '');
    this.prefix = p && !p.endsWith('/') ? p + '/' : p;
    this.sdkAvailable = !!S3Client;
    this.sdkLoadError = sdkLoadError;
    this.enabled = !!(opts.enabled && this.sdkAvailable && this.region && this.bucket);
    this.client = opts.client || null;
  }

  ensureClient() {
    // Credentials come from the SDK's default provider chain, as for Polly.
    if (!this.client && this.enabled) this.client = new S3Client({ region: this.region });
    return this.client;
  }

  /** @returns {Promise<{ok:boolean, key?:string, error?:string}>} */
  async upload(id, body) {
    if (!this.enabled) return { ok: false, error: 'S3 archive not enabled' };
    const safe = String(id || '').replace(/[^a-zA-Z0-9._-]/g, '');
    if (!safe) return { ok: false, error: 'missing recording id' };
    const key = this.prefix + safe + '.json';
    try {
      await this.ensureClient().send(new PutObjectCommand({
        Bucket: this.bucket,
        Key: key,
        Body: body,
        ContentType: 'application/json',
      }));
      return { ok: true, key };
    } catch (e) {
      return { ok: false, error: (e.name ? e.name + ': ' : '') + e.message };
    }
  }
}

module.exports = { S3Archive };
