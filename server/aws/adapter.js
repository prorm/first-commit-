/**
 * AWS service adapter.
 *
 * Contract: the application never imports an AWS SDK directly and never fails
 * because AWS is absent.  This module reports what is actually available and
 * returns a `provider` field on every call, so the UI can state plainly
 * whether a given piece of speech came from Amazon Polly or the browser's own
 * synthesiser.  No credentials are ever created, guessed, or defaulted here —
 * they come from the environment or the feature stays off.
 *
 *   VOICE: AWS POLLY      credentials present, synthesis succeeded
 *   VOICE: LOCAL FALLBACK no credentials, or Polly returned an error
 */
const { PollyVoice } = require('./polly');
const { BedrockSummarizer } = require('./bedrock');
const { S3Archive } = require('./s3');

class AwsAdapter {
  constructor(env = process.env) {
    this.region = (env.AWS_REGION || env.AWS_DEFAULT_REGION || '').trim();
    this.hasStaticKeys = !!(env.AWS_ACCESS_KEY_ID && env.AWS_SECRET_ACCESS_KEY);
    // A container/SSO profile can supply credentials without static keys.
    // An EC2 instance role leaves no environment variable behind, so it has to
    // be declared: AWS_USE_INSTANCE_ROLE=true lets the SDK's chain find it.
    this.hasProfile = !!(env.AWS_PROFILE || env.AWS_CONTAINER_CREDENTIALS_RELATIVE_URI || env.AWS_WEB_IDENTITY_TOKEN_FILE
      || String(env.AWS_USE_INSTANCE_ROLE || '').toLowerCase() === 'true');
    this.credentialsLikely = !!this.region && (this.hasStaticKeys || this.hasProfile);

    this.voiceId = env.POLLY_VOICE_ID || 'Matthew';
    this.engine = env.POLLY_ENGINE || 'neural';
    this.bedrockModelId = env.BEDROCK_MODEL_ID || '';
    this.bedrockEnabled = String(env.ENABLE_BEDROCK || '').toLowerCase() === 'true' && !!this.bedrockModelId;

    this.polly = new PollyVoice({
      region: this.region,
      voiceId: this.voiceId,
      engine: this.engine,
      enabled: this.credentialsLikely,
    });
    this.bedrock = new BedrockSummarizer({
      region: this.region,
      modelId: this.bedrockModelId,
      enabled: this.credentialsLikely && this.bedrockEnabled,
    });

    this.s3Bucket = (env.S3_BUCKET || '').trim();
    this.archive = new S3Archive({
      region: this.region,
      bucket: this.s3Bucket,
      prefix: env.S3_PREFIX || 'recordings/',
      enabled: this.credentialsLikely,
    });

    this.stats = {
      pollyCalls: 0, pollyFailures: 0, bedrockCalls: 0, bedrockFailures: 0, cacheHits: 0,
      s3Uploads: 0, s3Failures: 0,
    };
  }

  /**
   * Second, durable copy of a finished recording.  Never throws and never has
   * to be awaited: the local file is the source of truth for replay.
   */
  async archiveRecording(id, body) {
    if (!this.archive.enabled) return { ok: false, error: this.s3UnavailableReason() };
    const res = await this.archive.upload(id, body);
    if (res.ok) {
      this.stats.s3Uploads++;
    } else {
      this.stats.s3Failures++;
      // Nothing awaits this, so without a log line a bad bucket name, region or
      // permission would fail invisibly.
      console.warn('[warn] S3 archive failed for ' + id + ': ' + res.error);
    }
    return res;
  }

  /**
   * Synthesise guidance speech.
   * @returns {Promise<{provider:'polly'|'local', audio?:string, format?:string,
   *                    voiceId?:string, reason?:string, cached?:boolean}>}
   *   provider 'local' tells the phone to use SpeechSynthesis itself.
   */
  async speak(text) {
    if (!text || !this.polly.enabled) {
      return { provider: 'local', reason: this.voiceUnavailableReason() };
    }
    this.stats.pollyCalls++;
    const res = await this.polly.synthesize(text);
    if (!res.ok) {
      this.stats.pollyFailures++;
      // One failure disables Polly for the session rather than stalling every
      // future cue behind a doomed network call — the phone keeps talking.
      if (res.fatal) this.polly.enabled = false;
      return { provider: 'local', reason: res.error };
    }
    if (res.cached) this.stats.cacheHits++;
    return {
      provider: 'polly', audio: res.audio, format: 'mp3',
      voiceId: this.voiceId, engine: this.engine, cached: !!res.cached,
    };
  }

  /**
   * Optional post-scan interpretation.  Deliberately never in the real-time
   * path: it runs once, after a mission, on the numeric summary only.
   */
  async summarizeScan(summary) {
    if (!this.bedrock.enabled) {
      return { available: false, reason: this.bedrockUnavailableReason() };
    }
    this.stats.bedrockCalls++;
    const res = await this.bedrock.summarize(summary);
    if (!res.ok) {
      this.stats.bedrockFailures++;
      return { available: false, reason: res.error };
    }
    return { available: true, text: res.text, modelId: this.bedrockModelId };
  }

  voiceUnavailableReason() {
    if (!this.region) return 'AWS_REGION not set';
    if (!this.hasStaticKeys && !this.hasProfile) return 'no AWS credentials in environment';
    if (!this.polly.sdkAvailable) return '@aws-sdk/client-polly not installed';
    if (!this.polly.enabled) return 'Polly disabled after a failed call';
    return 'Polly available';
  }

  bedrockUnavailableReason() {
    if (!this.credentialsLikely) return 'no AWS credentials in environment';
    if (!this.bedrockModelId) return 'BEDROCK_MODEL_ID not set';
    if (String(process.env.ENABLE_BEDROCK || '').toLowerCase() !== 'true') return 'ENABLE_BEDROCK is not true';
    if (!this.bedrock.sdkAvailable) return '@aws-sdk/client-bedrock-runtime not installed';
    return 'Bedrock available';
  }

  s3UnavailableReason() {
    if (!this.s3Bucket) return 'S3_BUCKET not set';
    if (!this.credentialsLikely) return 'no AWS credentials in environment';
    if (!this.archive.sdkAvailable) return '@aws-sdk/client-s3 not installed';
    return 'S3 archive available';
  }

  /** What the diagnostics page and the map's status bar display. */
  status() {
    return {
      region: this.region || null,
      credentialsDetected: this.credentialsLikely,
      credentialSource: this.hasStaticKeys ? 'static-env-keys' : this.hasProfile ? 'profile-or-role' : null,
      voice: {
        provider: this.polly.enabled ? 'polly' : 'local',
        label: this.polly.enabled ? 'AWS POLLY' : 'LOCAL FALLBACK',
        voiceId: this.voiceId,
        engine: this.engine,
        sdkAvailable: this.polly.sdkAvailable,
        reason: this.voiceUnavailableReason(),
      },
      bedrock: {
        enabled: this.bedrock.enabled,
        modelId: this.bedrockModelId || null,
        sdkAvailable: this.bedrock.sdkAvailable,
        reason: this.bedrockUnavailableReason(),
      },
      s3: {
        enabled: this.archive.enabled,
        bucket: this.s3Bucket || null,
        prefix: this.archive.prefix,
        sdkAvailable: this.archive.sdkAvailable,
        reason: this.s3UnavailableReason(),
      },
      stats: Object.assign({}, this.stats),
    };
  }
}

module.exports = { AwsAdapter };
