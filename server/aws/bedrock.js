/**
 * Optional Amazon Bedrock post-scan interpretation.
 *
 * Scope is deliberately narrow, and the narrowness is the point: Bedrock never
 * touches the real-time DSP loop.  It runs once, after a mission, on the
 * already-computed numeric summary, and turns it into one short paragraph an
 * operator could read aloud.  If it is unavailable the mission summary is
 * simply shown without that paragraph.
 *
 * The prompt is constrained to the measured numbers and instructed to preserve
 * uncertainty, because the underlying classifier is 69.5 % accurate and a
 * fluent sentence must not imply more confidence than the sensor earned.
 */
let BedrockRuntimeClient = null;
let InvokeModelCommand = null;
let sdkLoadError = null;
try {
  const sdk = require('@aws-sdk/client-bedrock-runtime');
  BedrockRuntimeClient = sdk.BedrockRuntimeClient;
  InvokeModelCommand = sdk.InvokeModelCommand;
} catch (e) {
  sdkLoadError = e.message;
}

class BedrockSummarizer {
  constructor(opts = {}) {
    this.region = opts.region || '';
    this.modelId = opts.modelId || '';
    this.sdkAvailable = !!BedrockRuntimeClient;
    this.sdkLoadError = sdkLoadError;
    this.enabled = !!(opts.enabled && this.sdkAvailable && this.region && this.modelId);
    this.client = null;
  }

  ensureClient() {
    if (!this.client && this.enabled) this.client = new BedrockRuntimeClient({ region: this.region });
    return this.client;
  }

  buildPrompt(summary) {
    const s = summary.stats || {};
    const r = summary.reconstruction || {};
    const openings = (r.openingDetail || [])
      .map((o) => `  - candidate at (${o.x} m, ${o.y} m), evidence ${o.evidence}, confidence ${(o.confidence * 100).toFixed(0)}%`)
      .join('\n');

    return [
      'You are summarising one acoustic reconnaissance scan from SentryShield, a research prototype that maps a space using near-ultrasonic chirps from a phone.',
      '',
      'MEASURED RESULTS',
      `- distance walked: ${(s.distanceScanned || 0).toFixed(1)} m`,
      `- detections: ${s.detections || 0} (${((s.cfarRate || 0) * 100).toFixed(0)}% cleared the CFAR threshold)`,
      `- range span: ${s.minRange ? s.minRange.toFixed(2) : 'n/a'} m to ${s.maxRange ? s.maxRange.toFixed(2) : 'n/a'} m`,
      `- classified returns: ${(s.classCounts && s.classCounts.WALL) || 0} WALL, ${(s.classCounts && s.classCounts.SOFT) || 0} SOFT, ${(s.classCounts && s.classCounts.OPENING) || 0} OPENING`,
      `- mean class confidence: ${((s.avgClassConfidence || 0) * 100).toFixed(0)}%`,
      `- reconstructed boundary: ${r.segments || 0} surfaces totalling ${(r.totalWallLength || 0).toFixed(1)} m, ${r.corners || 0} corners, ${r.corridors || 0} corridor(s)`,
      `- reconstruction confidence: ${((r.confidence || 0) * 100).toFixed(0)}%`,
      `- opening candidates: ${r.openings || 0}`,
      openings || '  (none above 25% confidence)',
      '',
      'SENSOR LIMITS YOU MUST RESPECT',
      '- Bearing is the phone boresight with roughly 30 degrees of beamwidth; angles are approximate.',
      '- Phone position is dead-reckoned or simulated, not surveyed.',
      '- The echo classifier scores 69.5% on synthetic validation: WALL recall 91%, SOFT 74%, OPENING 43%. OPENING calls are weak evidence.',
      '',
      'TASK',
      'Write 2-4 sentences an operator could read aloud, describing what the scan suggests about the space and where the uncertainty is. Use hedged language ("suggests", "approximately", "possible") for anything the sensor cannot establish firmly. Do not invent measurements that are not listed above. Do not make safety guarantees or claim this is navigation-certified. Plain prose, no headings, no bullet points.',
    ].join('\n');
  }

  /** @returns {Promise<{ok:boolean, text?:string, error?:string}>} */
  async summarize(summary) {
    if (!this.enabled) return { ok: false, error: 'Bedrock not enabled' };
    const prompt = this.buildPrompt(summary);
    try {
      const client = this.ensureClient();
      const body = this.isAnthropicModel()
        ? {
            anthropic_version: 'bedrock-2023-05-31',
            max_tokens: 400,
            temperature: 0.3,
            messages: [{ role: 'user', content: [{ type: 'text', text: prompt }] }],
          }
        : { inputText: prompt, textGenerationConfig: { maxTokenCount: 400, temperature: 0.3 } };

      const out = await client.send(new InvokeModelCommand({
        modelId: this.modelId,
        contentType: 'application/json',
        accept: 'application/json',
        body: JSON.stringify(body),
      }));
      const parsed = JSON.parse(Buffer.from(out.body).toString('utf8'));
      const text = this.extractText(parsed);
      if (!text) return { ok: false, error: 'model returned no text' };
      return { ok: true, text: text.trim() };
    } catch (e) {
      return { ok: false, error: (e.name ? e.name + ': ' : '') + e.message };
    }
  }

  isAnthropicModel() { return /anthropic|claude/i.test(this.modelId); }

  extractText(parsed) {
    if (!parsed) return '';
    if (Array.isArray(parsed.content)) {
      return parsed.content.map((c) => (typeof c === 'string' ? c : c.text || '')).join('').trim();
    }
    if (Array.isArray(parsed.results)) return (parsed.results[0] && parsed.results[0].outputText) || '';
    if (typeof parsed.completion === 'string') return parsed.completion;
    if (typeof parsed.outputText === 'string') return parsed.outputText;
    return '';
  }
}

module.exports = { BedrockSummarizer };
