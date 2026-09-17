/**
 * SentryShield echo synthesiser — the acoustic half of the digital twin.
 *
 * Simulation mode does NOT fabricate classifier output.  It synthesises a
 * matched-filter *envelope window* with the same physics the training set used,
 * then runs the real exported EchoNet on it.  A simulated WALL is therefore
 * classified by the same 2,339-parameter network that classifies a live echo,
 * and the model's genuine weaknesses — OPENING recall is 43 % — show up in
 * simulation exactly as they do in the field.
 *
 * Ported from src/classifier/generate_echonet_dataset.py, keeping the parts
 * that shape the 64-sample window the network actually sees:
 *
 *  1. Matched filtering the echo against the *ideal* chirp leaves the spectrum
 *     shaped by Hann^2 (the TX window) times every transfer function the echo
 *     passed through.  The complex point-spread function is therefore
 *     chi(tau) = integral of shape(f) * exp(j*2*pi*(f - f_mid)*tau) df,
 *     and a scatterer lands as A*exp(j*phi)*exp(j*2*pi*f_mid*tau)*chi(tau).
 *     Spectral tilt is what broadens and skews the lobe, so it is modelled
 *     rather than approximated away.
 *  2. Per-class transfer functions: rigid tilt (WALL), porous-absorber
 *     low-pass (f/F0)^-p with p = 5.5..9 (SOFT), Keller knife-edge diffraction
 *     ~1/sqrt(f) (OPENING).
 *  3. A per-device speaker/mic response the matched filter does not know about.
 *  4. Two-way air absorption, ISO 9613-1, evaluated per frequency bin.
 *  5. Clutter subtraction: profile = max(live - 40-pulse baseline, 0).  The
 *     half-wave rectification leaves a zero-clipped, ragged noise floor, which
 *     is a large part of what a weak OPENING window looks like.
 */

export const FS = 48000;
export const F0 = 17500;
export const F1 = 22000;
export const BW = F1 - F0;
export const F_MID = 0.5 * (F0 + F1);
export const C_SOUND = 343.2;
export const WIN = 64;
export const HALF = WIN / 2;
export const N_BASELINE_PULSES = 40;
/** Envelope amplitude from a unit reflector at 1 m, for TX 0.6 FS (A_REF). */
export const A_REF = 6.0e-4;

// ---------------------------------------------------------------------------
// Air absorption — ISO 9613-1:1993 eq. (3)-(5), same as the generator
// ---------------------------------------------------------------------------
export function airAbsorptionDbPerM(fHz, tC = 20, rh = 50, pKpa = 101.325) {
  const T = tC + 273.15;
  const T0 = 293.15;
  const T01 = 273.16;
  const p = pKpa / 101.325;
  const psat = Math.pow(10, -6.8346 * Math.pow(T01 / T, 1.261) + 4.6151);
  const h = (rh * psat) / p;
  const frO = p * (24 + (4.04e4 * h * (0.02 + h)) / (0.391 + h));
  const frN = p * Math.pow(T / T0, -0.5) * (9 + 280 * h * Math.exp(-4.17 * (Math.pow(T / T0, -1 / 3) - 1)));
  const f2 = fHz * fHz;
  return 8.686 * f2 * (
    (1.84e-11 / p) * Math.pow(T / T0, 0.5)
    + Math.pow(T / T0, -2.5) * (
      (0.01275 * Math.exp(-2239.1 / T)) / (frO + f2 / frO)
      + (0.1068 * Math.exp(-3352.0 / T)) / (frN + f2 / frN)
    )
  );
}

export const ALPHA_MID = airAbsorptionDbPerM(F_MID);

// ---------------------------------------------------------------------------
// Complex point-spread function, per spectral shape
// ---------------------------------------------------------------------------
const NF = 160;                    // frequency samples across the sweep band
const LAG_SPAN = 56;               // samples either side of a scatterer
const LAG_SUB = 8;                 // sub-sample LUT resolution
const LAG_N = LAG_SPAN * 2 * LAG_SUB + 1;

const BAND_F = new Float32Array(NF);
const BAND_HANN2 = new Float32Array(NF);
const BAND_ABS_EXCESS = new Float32Array(NF);   // alpha(f) - alpha(f_mid)
for (let k = 0; k < NF; k++) {
  const u = k / (NF - 1);
  const hann = 0.5 * (1 - Math.cos(2 * Math.PI * u));
  BAND_F[k] = F0 + u * BW;
  BAND_HANN2[k] = hann * hann;
  BAND_ABS_EXCESS[k] = airAbsorptionDbPerM(BAND_F[k]) - ALPHA_MID;
}

const psfCache = new Map();

/**
 * Complex PSF for a spectral shape.
 * @param {(f:number,k:number)=>number} shapeFn extra magnitude response
 * @param {string} key cache key (quantised shape parameters)
 */
function getPsf(key, shapeFn) {
  let hit = psfCache.get(key);
  if (hit) return hit;

  const w = new Float32Array(NF);
  let wsum = 0;
  for (let k = 0; k < NF; k++) {
    w[k] = BAND_HANN2[k] * (shapeFn ? shapeFn(BAND_F[k], k) : 1);
    wsum += w[k];
  }
  if (wsum <= 0) wsum = 1;

  const re = new Float32Array(LAG_N);
  const im = new Float32Array(LAG_N);
  for (let i = 0; i < LAG_N; i++) {
    const tau = (i / LAG_SUB - LAG_SPAN) / FS;
    let ar = 0, ai = 0;
    for (let k = 0; k < NF; k++) {
      const ph = 2 * Math.PI * (BAND_F[k] - F_MID) * tau;
      ar += w[k] * Math.cos(ph);
      ai += w[k] * Math.sin(ph);
    }
    re[i] = ar / wsum;
    im[i] = ai / wsum;
  }
  hit = { re, im };
  if (psfCache.size > 96) psfCache.clear();      // bounded; shapes are quantised
  psfCache.set(key, hit);
  return hit;
}

/** Interpolate the complex PSF at a fractional sample lag. */
function psfAt(p, lag, out) {
  const x = (lag + LAG_SPAN) * LAG_SUB;
  if (x <= 0 || x >= LAG_N - 1) { out[0] = 0; out[1] = 0; return false; }
  const i = Math.floor(x);
  const fr = x - i;
  out[0] = p.re[i] * (1 - fr) + p.re[i + 1] * fr;
  out[1] = p.im[i] * (1 - fr) + p.im[i + 1] * fr;
  return true;
}

export function rangeToSamples(rangeM) { return ((2 * rangeM) / C_SOUND) * FS; }
export function samplesToRange(n) { return (n / FS) * C_SOUND / 2; }

/** Two-way spreading + band-centre absorption for reflectivity R at range r. */
export function echoAmplitude(R, rangeM) {
  const r = Math.max(0.15, rangeM);
  return (A_REF * R * Math.pow(10, (-ALPHA_MID * 2 * r) / 20)) / (r * r);
}

// ---------------------------------------------------------------------------
// Spectral shapes
// ---------------------------------------------------------------------------
/** Residual spectral shape of air absorption over a path (band centre removed). */
function absShape(pathM) {
  return (f, k) => Math.min(2, Math.pow(10, (-BAND_ABS_EXCESS[k] * pathM) / 20));
}
/** Porous absorber: |H| = (f/F0)^-p above F0. */
function lowpassShape(p) { return (f) => Math.min(1, Math.pow(f / F0, -p)); }
/** Linear-in-dB tilt across the band (tiltDb at F1 relative to F0). */
function tiltShape(tiltDb) { return (f) => Math.pow(10, (tiltDb * (f - F0)) / BW / 20); }
/** Keller GTD knife edge: |D| ~ 1/sqrt(k) ~ 1/sqrt(f). */
function gtdShape() { return (f) => Math.min(2, Math.sqrt(F0 / f)); }

function combine(fns) {
  const list = fns.filter(Boolean);
  return (f, k) => {
    let v = 1;
    for (let i = 0; i < list.length; i++) v *= list[i](f, k);
    return v;
  };
}

/**
 * Per-device speaker + mic response.  The PWA matched-filters against the
 * ideal chirp, so this mismatch broadens the compressed pulse on real hardware
 * — and in simulation, which is the point of modelling it.
 * @returns {{shape:Function, key:string, tiltDb:number}}
 */
export function makeDeviceResponse(rng) {
  const rand = rng || Math.random;
  const tiltDb = -9 * rand();
  const fRes = 18000 + rand() * 10000;
  const q = 2 + rand() * 4;
  const gain = rand() * 2;
  const ripDb = rand() * 1.5;
  const period = 1500 + rand() * 2500;
  const phase = rand() * 2 * Math.PI;
  const shape = (f) => {
    let v = Math.pow(10, (tiltDb * (f - F0)) / BW / 20);
    v *= Math.pow(10, (ripDb * Math.sin((2 * Math.PI * f) / period + phase)) / 20);
    // 2nd-order band-pass added to flat (MEMS Helmholtz resonance).
    const s = (f * fRes) / q;
    const denRe = fRes * fRes - f * f;
    const mag = Math.hypot(denRe, s);
    v *= Math.abs(1 + (gain * s) / Math.max(mag, 1e-9));
    return v;
  };
  return {
    shape,
    key: 'dev' + Math.round(tiltDb) + '_' + Math.round(fRes / 1000) + '_' + Math.round(q) + '_' + Math.round(gain * 4) + '_' + Math.round(ripDb * 4) + '_' + Math.round(period / 500) + '_' + Math.round(phase * 2),
    tiltDb,
  };
}

// ---------------------------------------------------------------------------
// Scatterer -> complex profile -> rectified envelope
// ---------------------------------------------------------------------------
/**
 * Coherently sum scatterer groups into a complex matched-filter profile.
 * Each group shares one spectral shape (and therefore one cached PSF).
 *
 * @param {Array} groups  [{ scatterers:[{r,amp,phase}], shapeKey, shapeFn }]
 * @param {Object} opts   { centreRange, n }
 */
function accumulate(groups, opts) {
  const n = opts.n;
  const centreIdx = opts.centreIdx;
  const centreTau = rangeToSamples(opts.centreRange);
  const re = new Float32Array(n);
  const im = new Float32Array(n);
  const tmp = [0, 0];

  for (const g of groups) {
    const p = getPsf(g.shapeKey, g.shapeFn);
    for (const s of g.scatterers) {
      if (!(s.amp > 0)) continue;
      const tau = rangeToSamples(s.r) - centreTau + centreIdx;
      const lo = Math.max(0, Math.floor(tau - LAG_SPAN));
      const hi = Math.min(n - 1, Math.ceil(tau + LAG_SPAN));
      // The analytic MF output is z[i] = sum_s A_s e^{j*phi_s} chi(i/fs - tau_s)
      // with chi(u) = e^{j*2*pi*f_mid*u} E(u).  The e^{j*2*pi*f_mid*i/fs} part is
      // common to every scatterer and cancels under |z|, so all that survives
      // is a constant per-scatterer phase -2*pi*f_mid*tau_s/fs.  That is what
      // makes two nearby reflectors interfere (speckle) without putting a
      // carrier-rate ripple on the envelope itself.
      const ph = s.phase - 2 * Math.PI * F_MID * (tau / FS);
      const cosP = Math.cos(ph);
      const sinP = Math.sin(ph);
      for (let i = lo; i <= hi; i++) {
        if (!psfAt(p, i - tau, tmp)) continue;
        re[i] += s.amp * (cosP * tmp[0] - sinP * tmp[1]);
        im[i] += s.amp * (cosP * tmp[1] + sinP * tmp[0]);
      }
    }
  }
  return { re, im };
}

/** Box-Muller from a supplied uniform RNG, so runs stay reproducible. */
function gauss(rand) {
  let u = 0;
  while (u === 0) u = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}

/**
 * Matched-filter output noise.
 *
 * Critically, this is NOT white.  The matched filter only passes 17.5-22 kHz,
 * so its output noise is band-limited: after the analytic operation the noise
 * envelope wanders on the scale of the compressed pulse (~1/BW, about 10
 * samples), not sample to sample.  Adding white noise instead produces a
 * visibly jagged envelope that no real receiver ever sees — and the classifier
 * notices, because roughness is one of the few cues it has.
 *
 * Synthesised as a sum over the band's independent bins (spacing fs/n), each
 * Hann-weighted (the MF passes |S(f)|, i.e. one Hann, not Hann^2) with a
 * random complex coefficient, then scaled to the requested output RMS.
 */
function bandLimitedNoise(n, sigma, rand) {
  const re = new Float32Array(n);
  const im = new Float32Array(n);
  if (!(sigma > 0)) return { re, im };
  const binHz = FS / n;
  const nb = Math.max(4, Math.round(BW / binHz));
  for (let k = 0; k < nb; k++) {
    const u = nb === 1 ? 0.5 : k / (nb - 1);
    const w = 0.5 * (1 - Math.cos(2 * Math.PI * u));       // Hann across the band
    if (w <= 0) continue;
    const df = F0 + u * BW - F_MID;
    const ar = gauss(rand) * w;
    const ai = gauss(rand) * w;
    const wn = (2 * Math.PI * df) / FS;
    for (let i = 0; i < n; i++) {
      const c = Math.cos(wn * i);
      const s = Math.sin(wn * i);
      re[i] += ar * c - ai * s;
      im[i] += ar * s + ai * c;
    }
  }
  let ss = 0;
  for (let i = 0; i < n; i++) ss += re[i] * re[i] + im[i] * im[i];
  const rms = Math.sqrt(ss / (2 * n));
  const g = rms > 0 ? sigma / rms : 0;
  for (let i = 0; i < n; i++) { re[i] *= g; im[i] *= g; }
  return { re, im };
}

// ---------------------------------------------------------------------------
// Class-conditional scenes
// ---------------------------------------------------------------------------
/**
 * Build the scatterer groups for a surface of `kind` at range r.
 * Distributions follow build_scene() in the generator.
 */
export function buildScene(kind, r, rng, ctx = {}) {
  const rand = rng || Math.random;
  const u = (lo, hi) => lo + rand() * (hi - lo);
  const h = ctx.phoneHeight != null ? ctx.phoneHeight : u(0.9, 1.15);
  const point = ctx.pointing != null ? ctx.pointing : u(0.6, 1.0);
  const floorR = ctx.floorR != null ? ctx.floorR : u(0.2, 0.9);
  const dev = ctx.device || makeDeviceResponse(rand);
  const groups = [];
  let R = 0;

  // Room reverberation: many weak oblique scatterers.  Anything nearer than the
  // target is off-axis by definition, so it only contributes diffusely.
  const revSc = [];
  const S = 10 + Math.floor(rand() * 20);
  for (let i = 0; i < S; i++) {
    let ri = u(0.8, 8.0);
    let Ri = Math.pow(10, u(-3.0, -1.7));
    if (ri < r) Ri *= 0.15;
    revSc.push({ r: ri, amp: echoAmplitude(Ri, ri), phase: u(0, 2 * Math.PI) });
  }
  const revP = u(0, 3);
  groups.push({
    scatterers: revSc,
    shapeKey: dev.key + '|rev' + Math.round(revP * 2),
    shapeFn: combine([dev.shape, absShape(8.0), lowpassShape(revP)]),
  });

  if (kind === 'WALL') {
    R = u(0.85, 0.98);
    const sc = [{ r, amp: echoAmplitude(R * point, r), phase: 0 }];
    if (rand() < 0.35) {                                   // baseboard / door frame edge
      const r2 = r + u(0.03, 0.30);
      sc.push({ r: r2, amp: echoAmplitude(R * point * u(0.03, 0.20), r2), phase: u(0, 2 * Math.PI) });
    }
    if (rand() < 0.70) {                                   // floor/wall dihedral (retro-reflector)
      const rd = Math.hypot(r, h);
      sc.push({ r: rd, amp: echoAmplitude(R * floorR * point * u(0.1, 0.5), rd), phase: 0 });
    }
    const rfb = 0.5 * (Math.sqrt(r * r + 4 * h * h) + r);  // phone -> floor -> wall -> phone
    sc.push({ r: rfb, amp: echoAmplitude(2 * R * floorR * point * u(0.1, 0.4), rfb), phase: rand() < 0.5 ? 0 : Math.PI });
    const tilt = u(-1, 1);
    groups.push({
      scatterers: sc,
      shapeKey: dev.key + '|wall' + Math.round(tilt * 4) + '_' + Math.round(r * 4),
      shapeFn: combine([dev.shape, absShape(2 * r), tiltShape(tilt)]),
    });
  } else if (kind === 'SOFT') {
    R = u(0.08, 0.35);
    const pLp = u(5.5, 9.0);
    const spreadM = samplesToRange(u(15, 30));
    const layers = [
      [0, u(0.15, 0.50)],
      [u(0.2, 0.5) * spreadM, u(0.10, 0.40)],
      [u(0.5, 1.0) * spreadM, 1.0],
    ];
    const M = 8 + Math.floor(rand() * 9);
    for (let i = 0; i < M; i++) {
      const t = rand();
      layers.push([t * spreadM, Math.exp(-3 * t) * u(0.05, 0.30)]);
    }
    // Glint cluster: head, shoulders, belly, cushions — 0..0.25 m behind the
    // front surface.  This depth is the signature that separates a soft body
    // from a point-like door edge once the pulse has smeared the layers.
    const nGl = 2 + Math.floor(rand() * 4);
    for (let i = 0; i < nGl; i++) layers.push([spreadM + u(0, 0.25), u(0.15, 0.80)]);
    const sc = layers.map(([d, f]) => {
      const rr = r + d;
      return { r: rr, amp: echoAmplitude(R * f * point, rr), phase: u(0, 2 * Math.PI) };
    });
    if (rand() < 0.6) {                                    // swinging arms
      const nA = 1 + Math.floor(rand() * 2);
      for (let i = 0; i < nA; i++) {
        const rr = r + u(-0.05, 0.15);
        sc.push({ r: rr, amp: echoAmplitude(R * point * u(0.05, 0.25), rr), phase: u(0, 2 * Math.PI) });
      }
    }
    const rfb = 0.5 * (Math.sqrt(r * r + 4 * h * h) + r);
    sc.push({ r: rfb, amp: echoAmplitude(2 * R * floorR * point * u(0.1, 0.4), rfb), phase: u(0, 2 * Math.PI) });
    groups.push({
      scatterers: sc,
      shapeKey: dev.key + '|soft' + Math.round(pLp * 2) + '_' + Math.round(r * 4),
      shapeFn: combine([dev.shape, absShape(2 * r), lowpassShape(pLp)]),
    });
  } else {
    // OPENING — no specular return at all.  Two jamb knife edges (GTD, -pi/4),
    // a lintel, and the far room beyond the doorway.
    const W = u(0.75, 1.05);
    const off = u(-0.35, 0.35);
    const sc = [];
    for (const side of [1, -1]) {
      const rj = Math.hypot(r, 0.5 * W + side * off);
      const Rj = u(0.01, 0.08);
      R = Math.max(R, Rj);
      sc.push({ r: rj, amp: echoAmplitude(Rj * point, rj), phase: -Math.PI / 4 + (rand() < 0.5 ? 0 : Math.PI) });
    }
    if (rand() < 0.8) {
      const dz = u(0.85, 1.15);
      const rl = Math.hypot(r, dz);
      sc.push({ r: rl, amp: echoAmplitude(u(0.01, 0.06) * point * Math.pow(r / Math.hypot(r, dz), 1.5), rl), phase: -Math.PI / 4 });
    }
    if (rand() < 0.3) {                                     // threshold strip
      const rt = Math.hypot(r, h);
      sc.push({ r: rt, amp: echoAmplitude(u(0.005, 0.03), rt), phase: -Math.PI / 4 });
    }
    groups.push({
      scatterers: sc,
      shapeKey: dev.key + '|gtd' + Math.round(r * 4),
      shapeFn: combine([dev.shape, absShape(2 * r), gtdShape()]),
    });

    // Far room beyond the doorway.
    const rFar = ctx.beyond != null ? Math.max(3.9, ctx.beyond) : u(3.9, 7.5);
    const far = [{ r: rFar, amp: echoAmplitude(u(0.6, 0.98) * point, rFar), phase: 0 }];
    const nF = 1 + Math.floor(rand() * 4);
    for (let i = 0; i < nF; i++) {
      const rf = u(3.9, 7.5);
      far.push({ r: rf, amp: echoAmplitude(Math.pow(10, u(-2.0, -0.8)), rf), phase: u(0, 2 * Math.PI) });
    }
    const farP = u(0, 2);
    groups.push({
      scatterers: far,
      shapeKey: dev.key + '|far' + Math.round(farP * 2) + '_' + Math.round(rFar),
      shapeFn: combine([dev.shape, absShape(2 * rFar), lowpassShape(farP)]),
    });
  }

  // Floor ping straight below the phone — static clutter, mostly cancelled.
  groups.push({
    scatterers: [{ r: h, amp: echoAmplitude(floorR * Math.pow(10, u(-40, -26) / 20), h), phase: 0 }],
    shapeKey: dev.key + '|floor' + Math.round(h * 8),
    shapeFn: combine([dev.shape, absShape(2 * h)]),
  });

  return { groups, R, device: dev, pointing: point, phoneHeight: h, floorR };
}

// ---------------------------------------------------------------------------
// Full pulse
// ---------------------------------------------------------------------------
/**
 * One synthetic pulse, all the way to the 64-sample classifier window.
 *
 * `leak` is the fraction of the target that bleeds into the slow clutter
 * baseline (a stationary target partially cancels itself), matching the
 * generator's row_gain.
 *
 * @returns {{win:Float32Array, profile:Float32Array, peakIdx:number,
 *            snr_db:number, measuredRange:number, detectionConfidence:number,
 *            cfar_pass:boolean, R:number}}
 */
export function synthesizePulse(kind, rangeM, rng, ctx = {}) {
  const rand = rng || Math.random;
  const n = ctx.n || 224;
  const centreIdx = Math.round(n / 2);
  const scene = buildScene(kind, rangeM, rand, ctx);

  const live = accumulate(scene.groups, { n, centreIdx, centreRange: rangeM });

  // Receiver noise: MEMS thermal floor + in-band share of 1/f room noise,
  // referred to the matched-filter output (processing gain ~16x).
  const sigmaTh = Math.pow(10, (-104 + rand() * 12) / 20);
  const sigmaPink = Math.pow(10, (-100 + rand() * 16) / 20) * 0.34;   // in-band share
  const sigmaOut = Math.hypot(sigmaTh, sigmaPink) * 0.061;

  // Static clutter baseline: 40 averaged pulses, so its noise is 1/sqrt(40) of
  // the live pulse, and it contains `leak` of the target.
  const leak = ctx.leak != null ? ctx.leak : rand() * 0.25;
  const bScale = 1 / Math.sqrt(N_BASELINE_PULSES);

  const liveNoise = bandLimitedNoise(n, sigmaOut, rand);
  // Fluctuation that survives averaging 40 envelopes: sd(|N|) ~ 0.655*sigma,
  // divided by sqrt(40).
  const baseRipple = bandLimitedNoise(n, sigmaOut * 0.655 * bScale, rand);

  const profile = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const liveEnv = Math.hypot(live.re[i] + liveNoise.re[i], live.im[i] + liveNoise.im[i]);
    // The baseline is the mean of 40 *envelopes*, not the envelope of a mean,
    // so it keeps the noise floor's positive bias (Rayleigh mean 1.2533*sigma).
    // Subtracting it and rectifying is what fills the real training windows
    // with exact zeros in noise-only cells — ~4 % for a strong wall, ~10 % for
    // a faint doorway.  Rician mean, approximated as sqrt(nu^2 + (pi/2)sigma^2),
    // which is exact at both nu = 0 and nu >> sigma.
    const nu = leak * Math.hypot(live.re[i], live.im[i]);
    const baseEnv = Math.sqrt(nu * nu + 1.5708 * sigmaOut * sigmaOut) + baseRipple.re[i];
    profile[i] = Math.max(liveEnv - baseEnv, 0);       // clutter subtraction, rectified
  }
  // Residual noise amplitude after the mean floor has been subtracted off.
  const noiseAmp = sigmaOut * 0.655;

  // Strongest local maximum, as the CFAR stage would pick.
  let peak = centreIdx;
  let pv = -1;
  const lo = HALF + 2;
  const hi = n - HALF - 2;
  for (let i = lo; i < hi; i++) {
    if (profile[i] >= profile[i - 1] && profile[i] > profile[i + 1] && profile[i] > pv) {
      pv = profile[i];
      peak = i;
    }
  }

  const win = new Float32Array(WIN);
  let mx = 0;
  for (let i = 0; i < WIN; i++) {
    const src = peak - HALF + i;
    const v = src >= 0 && src < n ? profile[src] : 0;
    win[i] = v;
    if (v > mx) mx = v;
  }
  if (mx > 0) for (let i = 0; i < WIN; i++) win[i] /= mx;

  const snr_db = Math.min(60, 20 * Math.log10(Math.max(pv, 1e-30) / Math.max(noiseAmp, 1e-30)));
  const measuredRange = rangeM + samplesToRange(peak - centreIdx);

  return {
    win,
    profile,
    peakIdx: peak,
    peakVal: mx,
    snr_db,
    R: scene.R,
    measuredRange,
    // Detection (not class) confidence: how far the peak sits above the noise
    // floor.  A doorway's jamb diffraction is genuinely faint, so this drops —
    // which is the honest outcome, not a bug to paper over.
    detectionConfidence: clamp01((snr_db - 3) / 24),
    cfar_pass: snr_db > 8,
    device: scene.device,
  };
}

function clamp01(v) { return v < 0 ? 0 : v > 1 ? 1 : v; }
