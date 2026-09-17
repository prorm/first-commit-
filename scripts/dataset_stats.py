"""Summarise the real EchoNet training windows so the JS digital twin can be
matched against them (shape statistics only -- no training here).

  python scripts/dataset_stats.py
"""
import numpy as np
import sys

d = np.load("src/classifier/echonet_synthetic_dataset.npz", allow_pickle=True)
print("keys:", list(d.keys()))
for k in d.keys():
    print(f"  {k}: shape={d[k].shape} dtype={d[k].dtype}")

X = d["X_train"] if "X_train" in d else d["X"]
y = d["y_train"] if "y_train" in d else d["y"]
if y.ndim > 1:
    y = y[:, 0]
y = y.astype(int)
print("\nX", X.shape, "y", np.bincount(y))

names = ["WALL", "SOFT", "OPENING"]


def feats(w):
    """Shape descriptors of a 64-sample peak-normalised window."""
    n = len(w)
    pk = int(np.argmax(w))
    mass = w.sum()
    # -3 dB (0.5) width around the peak
    above = w >= 0.5
    width = above.sum()
    # fraction of exactly-zero samples (clutter-subtraction rectification)
    zeros = float((w <= 1e-6).mean())
    # energy in the tails vs the main lobe
    lobe = w[max(0, pk - 12):pk + 13].sum()
    tail = mass - lobe
    # roughness: mean |second difference|
    rough = float(np.abs(np.diff(w, 2)).mean())
    # number of local maxima above 0.25
    lm = 0
    for i in range(1, n - 1):
        if w[i] >= w[i - 1] and w[i] > w[i + 1] and w[i] > 0.25:
            lm += 1
    return dict(peak=pk, mass=float(mass), width=float(width), zeros=zeros,
                tailfrac=float(tail / max(mass, 1e-9)), rough=rough, lmax=lm)


print("\n           peakIdx    mass   w>0.5   zerofrac  tailfrac   rough   localmax")
for c in range(3):
    sel = X[y == c]
    fs = [feats(w) for w in sel[:1500]]
    def m(k):
        return float(np.mean([f[k] for f in fs]))
    print(f"  {names[c]:<8} {m('peak'):7.1f} {m('mass'):7.2f} {m('width'):7.2f} "
          f"{m('zeros'):9.3f} {m('tailfrac'):9.3f} {m('rough'):7.4f} {m('lmax'):8.2f}")

if "M_train" in d or "M" in d:
    M = d["M_train"] if "M_train" in d else d["M"]
    cols = list(d["meta_columns"]) if "meta_columns" in d else None
    print("\nmeta columns:", cols)
    if cols is not None and "snr_db" in [str(c) for c in cols]:
        si = [str(c) for c in cols].index("snr_db")
        for c in range(3):
            s = M[y == c][:, si]
            print(f"  {names[c]:<8} snr_db  mean {s.mean():6.1f}  p10 {np.percentile(s,10):6.1f}  p90 {np.percentile(s,90):6.1f}")
        ri = [str(c) for c in cols].index("range_m") if "range_m" in [str(c) for c in cols] else None
        if ri is not None:
            for c in range(3):
                s = M[y == c][:, ri]
                print(f"  {names[c]:<8} range_m mean {s.mean():5.2f}  min {s.min():5.2f}  max {s.max():5.2f}")
