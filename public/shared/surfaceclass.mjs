/**
 * What EchoNet is allowed to decide.
 *
 * EchoNet has three output heads because it was trained on three synthetic
 * classes.  Only two of them describe something the sensor can actually
 * measure:
 *
 *   WALL  a hard, specular return — most of the incident energy comes back
 *   SOFT  an absorptive return — a body, a curtain, upholstery
 *
 * The third head, OPENING, does not. An opening is not a sound texture; it is
 * the *absence* of a return. CFAR takes the strongest peak past the direct-path
 * gate, so when the beam passes through a doorway it locks onto whatever stands
 * behind the doorway — the far wall of the next room. The window EchoNet then
 * sees is a wall echo, and labelling it OPENING teaches the network that a
 * distant wall is a hole. The real recordings say exactly this: across the ten
 * sessions in recordings/, pulses labelled OPENING carry the *highest*
 * spreading-compensated target strength of any class (23.7 dB, against WALL's
 * 18.2 dB), which is physically backwards for a hole in a wall.
 *
 * So openings are recovered geometrically instead, by findGapOpenings() in
 * reconstruct.mjs: a door-width gap in an otherwise continuous run of wall.
 * That is evidence this sensor can genuinely produce.
 *
 * Both the phone pipeline and the digital twin route their classifier output
 * through here, so the two cannot disagree about what a class means.
 */

/** The classes EchoNet is permitted to assert. Index matches its output head. */
export const SURFACE_CLASSES = ['WALL', 'SOFT'];

/**
 * Below this separation between the two retained heads the call is a coin
 * flip, and we say so by returning null rather than inventing a label. Held
 * out by session, the model's real-echo accuracy is 39% against a 33% chance
 * floor, so an unconfident call carries close to no information.
 */
export const MIN_MARGIN = 0.10;

/**
 * Collapse a full EchoNet distribution to a surface call.
 *
 * @param {number[]|Float32Array|null} probs  [WALL, SOFT, OPENING] softmax
 * @returns {{className: string|null, confidence: number, margin: number, openingProb: number}}
 *   className is 'WALL', 'SOFT', or null when the two heads are too close to
 *   separate. confidence is renormalised over the retained heads only, so it
 *   is not inflated by probability mass parked on the discarded head.
 */
export function surfaceFromProbs(probs) {
  const none = { className: null, confidence: 0, margin: 0, openingProb: 0 };
  if (!probs || probs.length < 2) return none;

  const wall = Math.max(0, Number(probs[0]) || 0);
  const soft = Math.max(0, Number(probs[1]) || 0);
  const openingProb = Math.max(0, Number(probs[2]) || 0);
  const keep = wall + soft;
  if (!(keep > 0)) return none;

  const pWall = wall / keep;
  const pSoft = soft / keep;
  const margin = Math.abs(pWall - pSoft);
  if (margin < MIN_MARGIN) return { className: null, confidence: 0, margin, openingProb };

  return {
    className: pWall >= pSoft ? 'WALL' : 'SOFT',
    confidence: Math.max(pWall, pSoft),
    margin,
    openingProb,
  };
}

/** Convenience: run a classifier and collapse its output in one step. */
export function classifySurface(classifier, win) {
  if (!classifier || !win) return { className: null, confidence: 0, margin: 0, openingProb: 0, probs: [0, 0, 0] };
  const out = classifier.forward(win);
  const probs = Array.from(out.probs);
  return { ...surfaceFromProbs(probs), probs };
}
