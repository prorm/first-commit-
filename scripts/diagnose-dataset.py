"""
Diagnose the real echo dataset before trying to train on it.

Answers the only question that matters before tuning a model: how much
class information is actually in the collected windows, and how much of any
measured accuracy is leakage from the collection protocol?

Run with the 3.13 interpreter (needs sklearn):
  C:/Users/prana/AppData/Local/Programs/Python/Python313/python.exe scripts/diagnose-dataset.py
"""
import glob
import hashlib
import json
import os
from collections import Counter

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

CLASSES = ['WALL', 'SOFT', 'OPENING']


def load_sessions():
    sessions = []
    for f in sorted(glob.glob('recordings/real_echo_dataset_*.json')):
        d = json.load(open(f))
        X, y, r, s = [], [], [], []
        for smp in d.get('samples', []):
            if len(smp.get('window', [])) != 64:
                continue
            X.append(smp['window'])
            y.append(smp['classIndex'])
            r.append(smp.get('range_m') or 0.0)
            s.append(smp.get('snr_db') or 0.0)
        sessions.append({
            'name': os.path.basename(f)[19:34],
            'device': d.get('device', '?'),
            'X': np.array(X, np.float64), 'y': np.array(y),
            'r': np.array(r), 's': np.array(s),
            'hash': hashlib.md5(np.array(X, np.float32).tobytes()).hexdigest()[:12],
        })
    return sessions


def report_duplicates(sessions):
    print('=== duplicate sessions ===')
    byhash = {}
    for i, S in enumerate(sessions):
        byhash.setdefault(S['hash'], []).append(i)
    dupes = set()
    for h, idxs in byhash.items():
        names = [f"{sessions[i]['name']} ({sessions[i]['device']})" for i in idxs]
        if len(idxs) > 1:
            print(f'  DUPLICATE: {names}')
            dupes.update(idxs[1:])
        else:
            print(f'  ok:        {names[0]}')
    return dupes


def report_independence(sessions):
    """Each class is collected as one contiguous burst of pulses 50 ms apart.
    Consecutive pulses in a burst are near-identical, so a burst is closer to
    one observation than to `len(burst)` of them."""
    print('\n=== independent observations ===')
    total = sum(len(S['y']) for S in sessions)
    bursts = 0
    for S in sessions:
        lab = S['y']
        bursts += 1 + int((lab[1:] != lab[:-1]).sum())
    print(f'  pulses collected:        {total}')
    print(f'  contiguous class bursts: {bursts}   <- the real sample size')
    print(f'  pulses per burst:        ~{total // max(bursts, 1)}')


def report_physics(sessions):
    y = np.concatenate([S['y'] for S in sessions])
    r = np.concatenate([S['r'] for S in sessions])
    s = np.concatenate([S['s'] for S in sessions])
    X = np.vstack([S['X'] for S in sessions])
    ts = s + 40 * np.log10(np.maximum(r, 0.2))  # two-way spreading compensated
    print('\n=== per-class physics (does the label match the acoustics?) ===')
    print(f'  {"":8s} {"range_m":>8s} {"snr_db":>8s} {"target dB":>10s} {"FWHM":>6s}')
    for c, n in enumerate(CLASSES):
        m = y == c
        half = [np.flatnonzero(w >= 0.5) for w in X[m]]
        fwhm = np.median([(h[-1] - h[0] + 1) if len(h) else 0 for h in half])
        print(f'  {n:8s} {np.median(r[m]):8.2f} {np.median(s[m]):8.1f} '
              f'{np.median(ts[m]):10.1f} {fwhm:6.0f}')
    print('  An OPENING returns less energy than a WALL. If its target strength')
    print('  is not the lowest, the detector locked onto something behind it.')


def report_accuracy(sessions, dupes):
    keep = [S for i, S in enumerate(sessions) if i not in dupes]
    X = np.vstack([S['X'] for S in keep])
    y = np.concatenate([S['y'] for S in keep])
    g = np.concatenate([np.full(len(S['y']), i) for i, S in enumerate(keep)])
    r = np.concatenate([S['r'] for S in keep])
    s = np.concatenate([S['s'] for S in keep])

    print('\n=== how much of the accuracy is real? (chance = 33.3%) ===')
    rf = lambda: RandomForestClassifier(300, n_jobs=-1, random_state=0)

    leaky = cross_val_score(rf(), np.c_[X, r, s], y,
                            cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()
    print(f'  random 5-fold CV            {leaky * 100:5.1f}%   <- inflated: splits bursts')

    F = np.c_[X, r, s]
    accs = []
    for i in range(len(keep)):
        tr, te = g != i, g == i
        accs.append(rf().fit(F[tr], y[tr]).score(F[te], y[te]))
    print(f'  leave-one-session-out       {np.mean(accs) * 100:5.1f}%   <- honest')
    print(f'    per session               {[round(a * 100) for a in accs]}')

    accs_w = []
    for i in range(len(keep)):
        tr, te = g != i, g == i
        accs_w.append(rf().fit(X[tr], y[tr]).score(X[te], y[te]))
    print(f'  LOSO, window only           {np.mean(accs_w) * 100:5.1f}%   <- no range/snr crutch')

    accs_r = []
    for i in range(len(keep)):
        tr, te = g != i, g == i
        accs_r.append(rf().fit(r[tr, None], y[tr]).score(r[te, None], y[te]))
    print(f'  LOSO, range_m alone         {np.mean(accs_r) * 100:5.1f}%   <- pure collection artifact')


def main():
    sessions = load_sessions()
    y = np.concatenate([S['y'] for S in sessions])
    print(f'{len(sessions)} sessions, {len(y)} pulses, labels {dict(sorted(Counter(y.tolist()).items()))}\n')
    dupes = report_duplicates(sessions)
    report_independence(sessions)
    report_physics(sessions)
    report_accuracy(sessions, dupes)


if __name__ == '__main__':
    main()
