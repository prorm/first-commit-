import numpy as np
import json
import glob
import os

# 1. Load synthetic data
synth = np.load('src/classifier/echonet_synthetic_dataset.npz', allow_pickle=True)
X_synth = synth['X_train']
y_synth = synth['y_train']
meta_synth = synth['meta_train']
cols = list(synth['meta_columns'])

print(f"Synthetic total: {len(X_synth)} samples")
for c in range(3):
    print(f"  Class {c} synth count: {np.sum(y_synth == c)}")

# 2. Load all real data
real_files = glob.glob('recordings/real_echo_dataset_*.json')
real_windows = []
real_labels = []
real_ranges = []
real_snrs = []
real_devices = []

seen = set()
for rf in real_files:
    with open(rf, 'r') as f:
        data = json.load(f)
    dev = data.get('device', 'Unknown')
    for s in data.get('samples', []):
        id_str = f"{s.get('t')}_{s.get('classIndex')}"
        if id_str in seen:
            continue
        seen.add(id_str)
        w = s.get('window', [])
        if len(w) == 64:
            real_windows.append(w)
            real_labels.append(s.get('classIndex'))
            real_ranges.append(s.get('range_m') if s.get('range_m') is not None else np.nan)
            real_snrs.append(s.get('snr_db') if s.get('snr_db') is not None else np.nan)
            real_devices.append(dev)

X_real = np.array(real_windows, dtype=np.float32)
y_real = np.array(real_labels, dtype=np.int64)
real_ranges = np.array(real_ranges, dtype=np.float32)
real_snrs = np.array(real_snrs, dtype=np.float32)

print(f"\nReal total unique: {len(X_real)} samples across {len(real_files)} files")
for c in range(3):
    print(f"  Class {c} real count: {np.sum(y_real == c)}")

# 3. Compare range and SNR stats
for c, name in enumerate(['WALL', 'SOFT', 'OPENING']):
    r_idx = np.where(y_real == c)[0]
    s_idx = np.where(y_synth == c)[0]
    
    r_snr_valid = real_snrs[r_idx][~np.isnan(real_snrs[r_idx])]
    s_snr = meta_synth[s_idx, cols.index('snr_db')]
    
    r_rng_valid = real_ranges[r_idx][~np.isnan(real_ranges[r_idx])]
    s_rng = meta_synth[s_idx, cols.index('range_m')]
    
    print(f"\n=== {name} (Class {c}) ===")
    if len(r_snr_valid) > 0:
        print(f"Real SNR (dB):  median {np.median(r_snr_valid):.1f} (IQR: {np.percentile(r_snr_valid, 25):.1f}..{np.percentile(r_snr_valid, 75):.1f})")
    print(f"Synth SNR (dB): median {np.median(s_snr):.1f} (IQR: {np.percentile(s_snr, 25):.1f}..{np.percentile(s_snr, 75):.1f})")
    
    if len(r_rng_valid) > 0:
        print(f"Real Range (m):  median {np.median(r_rng_valid):.2f} ({np.percentile(r_rng_valid, 10):.2f}..{np.percentile(r_rng_valid, 90):.2f})")
    print(f"Synth Range (m): median {np.median(s_rng):.2f} ({np.percentile(s_rng, 10):.2f}..{np.percentile(s_rng, 90):.2f})")

    # Measure cosine similarity between real samples and synthetic samples of the same class
    # Normalize vectors
    r_norm = X_real[r_idx] / (np.linalg.norm(X_real[r_idx], axis=1, keepdims=True) + 1e-8)
    s_norm = X_synth[s_idx] / (np.linalg.norm(X_synth[s_idx], axis=1, keepdims=True) + 1e-8)
    
    # Compute mean centroid of real class
    r_mean = np.mean(r_norm, axis=0)
    r_mean = r_mean / (np.linalg.norm(r_mean) + 1e-8)
    
    # Cosine sim of synthetic samples to real centroid
    sims = np.dot(s_norm, r_mean)
    print(f"Synth-to-Real Centroid Cosine Sim: mean {np.mean(sims):.3f}, max {np.max(sims):.3f}, 90th-pct {np.percentile(sims, 90):.3f}")
    
    # How many synthetic samples have high similarity (> 0.70)?
    high_sim = np.sum(sims > 0.70)
    print(f"Synthetic samples with Cosine Sim > 0.70: {high_sim} / {len(sims)} ({high_sim/len(sims)*100:.1f}%)")

print("\n=== FILTERING HIGH-FIDELITY SYNTHETIC PULSES FOR TRAINING ===")
snr_col = cols.index('snr_db')
rng_col = cols.index('range_m')

selected_synth_samples = []
for c in range(3):
    r_idx = np.where(y_real == c)[0]
    s_idx = np.where(y_synth == c)[0]
    
    r_norm = X_real[r_idx] / (np.linalg.norm(X_real[r_idx], axis=1, keepdims=True) + 1e-8)
    s_norm = X_synth[s_idx] / (np.linalg.norm(X_synth[s_idx], axis=1, keepdims=True) + 1e-8)
    
    r_centroid = np.mean(r_norm, axis=0)
    r_centroid /= (np.linalg.norm(r_centroid) + 1e-8)
    
    sims = np.dot(s_norm, r_centroid)
    snrs = meta_synth[s_idx, snr_col]
    ranges = meta_synth[s_idx, rng_col]
    
    # Filter matching realistic SNR (8-26 dB), distance (0.4-3.5m), and high waveform similarity (>0.88)
    mask = (sims >= 0.88) & (snrs >= 8.0) & (snrs <= 26.0) & (ranges >= 0.4) & (ranges <= 3.5)
    matched_s_idx = s_idx[mask]
    print(f"Class {c} ({['WALL', 'SOFT', 'OPENING'][c]}): selected {len(matched_s_idx)} realistic synthetic pulses")
    
    for idx in matched_s_idx:
        selected_synth_samples.append({
            "classIndex": int(c),
            "className": ['WALL', 'SOFT', 'OPENING'][c],
            "window": [float(x) for x in X_synth[idx]],
            "range_m": float(meta_synth[idx, rng_col]),
            "snr_db": float(meta_synth[idx, snr_col]),
            "is_synthetic_augmented": True
        })

print(f"\nTotal high-fidelity realistic synthetic samples selected: {len(selected_synth_samples)}")

out_path = "recordings/synthetic_matched_rehearsal.json"
with open(out_path, "w") as f:
    json.dump({
        "created_at": "2026-09-17T18:15:00Z",
        "device": "Synthetic_Acoustic_Twin",
        "total_samples": len(selected_synth_samples),
        "samples": selected_synth_samples
    }, f, indent=2)

print(f"Exported to {out_path} for trainer pooling!")

