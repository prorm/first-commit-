/**
 * Radix-2 FFT.
 *
 * Carried over unchanged from the working PITCHBLACK gate-test prototype —
 * this code already produced correct ranges on the target phone, so it is
 * reused rather than rewritten.  Only the module wrapper and typed-array
 * scratch buffers are new.
 */
export class FFT {
  constructor(size) {
    this.size = size;
    this.cosTable = new Float32Array(size / 2);
    this.sinTable = new Float32Array(size / 2);
    for (let i = 0; i < size / 2; i++) {
      const angle = (2 * Math.PI * i) / size;
      this.cosTable[i] = Math.cos(angle);
      this.sinTable[i] = Math.sin(angle);
    }
    this.reverseTable = new Uint32Array(size);
    let limit = 1;
    let bit = size >> 1;
    while (limit < size) {
      for (let i = 0; i < limit; i++) {
        this.reverseTable[i + limit] = this.reverseTable[i] + bit;
      }
      limit <<= 1;
      bit >>= 1;
    }
  }

  /** In-place complex transform. `inverse` scales by 1/n. */
  transform(real, imag, inverse = false) {
    const n = this.size;
    for (let i = 0; i < n; i++) {
      const j = this.reverseTable[i];
      if (j > i) {
        let temp = real[i]; real[i] = real[j]; real[j] = temp;
        temp = imag[i]; imag[i] = imag[j]; imag[j] = temp;
      }
    }
    for (let halfSize = 1; halfSize < n; halfSize <<= 1) {
      const step = n / (halfSize << 1);
      for (let i = 0; i < n; i += halfSize << 1) {
        for (let j = 0; j < halfSize; j++) {
          const k = j * step;
          const c = this.cosTable[k];
          const s = inverse ? -this.sinTable[k] : this.sinTable[k];
          const rOdd = real[i + j + halfSize];
          const iOdd = imag[i + j + halfSize];
          const tr = rOdd * c + iOdd * s;
          const ti = iOdd * c - rOdd * s;
          real[i + j + halfSize] = real[i + j] - tr;
          imag[i + j + halfSize] = imag[i + j] - ti;
          real[i + j] += tr;
          imag[i + j] += ti;
        }
      }
    }
    if (inverse) {
      const invN = 1 / n;
      for (let i = 0; i < n; i++) {
        real[i] *= invN;
        imag[i] *= invN;
      }
    }
  }
}
