#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_echonet_dataset.py
===========================
PITCHBLACK / EchoNet -- first-principles synthetic dataset generator + trainer.

EchoNet is a tiny on-device 1-D CNN that looks at a 64-sample window of the
matched-filter envelope around the detected echo peak and answers:

    0 = WALL     rigid specular planar reflector (drywall, concrete, door, glass)
    1 = SOFT     diffuse, absorbing obstacle (person, coat, curtain, sofa)
    2 = OPENING  acoustic void (open doorway, hallway, clear path) -> only
                 knife-edge diffraction from the jambs, then far echoes

This script simulates the *whole* PITCHBLACK acoustic chain from physics,
then runs the exact receiver chain the PWA runs, so the training crops have
the same statistics the deployed classifier will see:

    TX chirp -> transducer response -> propagation (spreading + ISO 9613-1 air
    absorption) -> target impulse response (class physics) -> multipath
    (floor bounce, floor/wall dihedral, user body) -> room reverberation
    -> Doppler time dilation (walking) -> chassis direct-path leakage
    -> mic response -> thermal + 1/f + impulsive noise -> 16-bit ADC
    -> FFT matched filter -> analytic envelope -> direct-path gate
    -> 40-pulse clutter subtraction -> CA-CFAR -> 64-sample crop -> peak-norm.

Outputs (written to --out-dir, default = this file's directory):
    echonet_synthetic_dataset.npz   X_train/y_train/X_val/y_val/metadata (+ per-sample meta)
    echonet_tiny.pt                 PyTorch state_dict
    echonet.onnx                    ONNX graph for onnxruntime-web
    echonet_weights.js              dependency-free JS forward pass (BN folded)
    echonet_dataset_preview.png     (only with --plot, needs matplotlib)

Usage:
    python generate_echonet_dataset.py                       # 15k samples, train, export
    python generate_echonet_dataset.py --n-samples 600 --epochs 3 --workers 2   # smoke test
    python generate_echonet_dataset.py --skip-train          # dataset only
    python generate_echonet_dataset.py --head flatten        # positional head variant

Requirements: numpy, scipy (optional, faster FFTs), torch (training/export),
onnx (ONNX export), node (optional, verifies the JS forward pass bit-for-bit).

--------------------------------------------------------------------------
Engineering notes (read before changing constants)
--------------------------------------------------------------------------
* Air absorption is computed from the full Bass/Sutherland/ISO 9613-1
  relaxation model at 20 C, 50 % RH, 101.325 kPa.  That model gives
  ~0.42 dB/m at 17.5 kHz, ~0.52 dB/m at 20 kHz and ~0.61 dB/m at 22 kHz --
  roughly ten times larger than the 0.05-0.07 dB/m quoted in the brief.  The
  physics is kept as computed; the quadratic-in-frequency shape is identical.
* Spreading uses the point-target radar form A ~ 1/r^2 (power ~ 1/r^4) as
  specified.  For an infinite plane the image-source law A ~ 1/(2r) is the
  exact answer; 1/r^2 is therefore *pessimistic* for WALL SNR at range,
  which biases training toward harder examples.  SPREADING_EXP controls it.
* The classifier only ever sees the crop around the peak that the CFAR
  picked.  For OPENING scenes the detection search is capped at 3.75 m so
  the crop is centred on the jamb diffraction wavelet, not on the far wall.
  The PWA must apply the same cap (or treat detections beyond 3.5 m as
  OPENING) for the class semantics to match.
* The 40-pulse clutter baseline is emulated with --baseline-rows (default 8)
  explicit past pulses whose noise is scaled so the averaged noise variance
  equals a true 40-pulse mean.  Use --baseline-rows 40 for full fidelity
  (about 5x slower).
* Validation accuracy on this synthetic set is a statement about the
  simulator, not about the real world.  Treat it as a smoke test; the blind
  wall test in CLAUDE.md is the acceptance test.

Units: every physical quantity is annotated [unit].  [smp] = samples at FS.
[FS] = digital full scale (1.0 = 0 dBFS).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

import numpy as np

try:                                  # scipy.fft is ~2x faster than numpy.fft; API-compatible subset
    from scipy import fft as sfft
except ImportError:                   # pragma: no cover
    sfft = np.fft

# =============================================================================
# 1. Signal constants (must match src/audio chirp + src/dsp matched filter)
# =============================================================================
FS = 48_000.0                         # [Hz]  sample rate (NOT 44.1 k)
F0 = 17_500.0                         # [Hz]  chirp start frequency
F1 = 22_000.0                         # [Hz]  chirp stop frequency
BW = F1 - F0                          # [Hz]  4500 Hz sweep bandwidth
F_MID = 0.5 * (F0 + F1)               # [Hz]  19.75 kHz band centre
T_CHIRP = 0.015                       # [s]   chirp duration
N_CHIRP = int(round(T_CHIRP * FS))    # [smp] 720
CHIRP_RATE = BW / T_CHIRP             # [Hz/s] linear sweep rate
TX_AMP = 0.6                          # [FS]  TX amplitude (Hann window provides the ramp)
PRI = 0.050                           # [s]   pulse repetition interval (20 Hz)
N_RX = int(round(PRI * FS))           # [smp] 2400 receive window per pulse
NFFT = 4096                           # [smp] FFT size (>= N_RX + N_CHIRP - 1 -> linear correlation)
NF = NFFT // 2 + 1                    # rfft bins
FREQS = np.fft.rfftfreq(NFFT, 1.0 / FS)   # [Hz] rfft bin frequencies

C_SOUND = 343.2                       # [m/s] speed of sound at 20 C
T_AIR_C = 20.0                        # [C]
REL_HUMIDITY = 50.0                   # [%]
P_ATM_KPA = 101.325                   # [kPa]

LATENCY_S = 0.002                     # [s]   calibrated audio I/O latency offset (constant, any value works)
L0 = int(round(LATENCY_S * FS))       # [smp] 96
GATE_HALF = int(round(0.002 * FS))    # [smp] +-2 ms direct-path gate = 96
N_BASELINE_PULSES = 40                # pulses averaged for the static clutter baseline
WIN = 64                              # [smp] classifier window
HALF = WIN // 2

CLASSES = ["WALL", "SOFT", "OPENING"]
R_MIN, R_MAX = 0.40, 3.50             # [m] target range span
R_MAX_OPENING = 3.20                  # [m] door plane range cap (jambs are farther than the plane)
R_SEARCH_MIN, R_SEARCH_MAX = 0.30, 3.75   # [m] CFAR search span (covers Doppler range coupling)
V_MAX = 1.2                           # [m/s] walking speed span
SOFT_CLUSTER_DEPTH_M = 0.25           # [m]   range extent of the body/furniture glint cluster behind the front surface
A_REF = 6.0e-4                        # [FS]  echo amplitude from R=1 reflector at 1 m for TX 0.6 FS
                                      #       (~74 dB SPL @1 m speaker, -38 dBFS/Pa MEMS mic -> -64 dBFS)
SPREADING_EXP = 2.0                   # amplitude ~ r^-SPREADING_EXP (2.0 = point target, 1.0 = image source)
ADC_BITS = 16

META_COLUMNS = ["class", "range_m", "vel_mps", "R", "snr_db", "cfar_pass", "peak_idx",
                "range_detected_m", "sys_tilt_db", "noise_dbfs", "h_phone_m", "baseline_leak",
                "peak_amp_fs", "target_nearest_m", "n_tries"]


# =============================================================================
# 2. Waveform synthesis
# =============================================================================
def chirp_at(t, phase=0.0):
    """Hann-windowed linear FM chirp evaluated at arbitrary times.

    t     : array [s], time since chirp start (any shape); zero outside [0, T_CHIRP)
    phase : additional carrier phase [rad] (broadcastable) -> models reflection phase
    Returns unit-amplitude samples (multiply by amplitude in [FS]).
    Evaluating the closed form at arbitrary t gives exact fractional delays and
    exact Doppler time-dilation with no interpolation error.
    """
    inside = (t >= 0.0) & (t < T_CHIRP)
    tt = np.where(inside, t, 0.0)
    win = 0.5 * (1.0 - np.cos(2.0 * np.pi * tt / T_CHIRP))          # Hann
    ph = 2.0 * np.pi * (F0 * tt + 0.5 * CHIRP_RATE * tt * tt) + phase  # linear FM phase
    return np.where(inside, win * np.sin(ph), 0.0)


REF = TX_AMP * chirp_at(np.arange(N_CHIRP) / FS)      # [FS] reference chirp, 720 smp
REF_PAD = np.zeros(NFFT)
REF_PAD[:N_CHIRP] = REF                                # zero-padded to 4096
_REF_F = sfft.fft(REF_PAD)
# Matched filter, normalised so an echo of amplitude a [FS] yields envelope peak a [FS].
MF_F = np.conj(_REF_F) / (np.sum(REF ** 2) / TX_AMP)
N_CHUNK = N_CHIRP + 8                                  # [smp] local buffer per echo copy (Doppler-stretched chirp fits)


# =============================================================================
# 3. Atmospheric absorption -- ISO 9613-1 (Bass, Sutherland et al.)
# =============================================================================
def air_absorption_db_per_m(f_hz, t_c=T_AIR_C, rh=REL_HUMIDITY, p_kpa=P_ATM_KPA):
    """Pure-tone atmospheric absorption coefficient alpha(f) [dB/m].

    Classical (viscous/thermal, ~f^2) + O2 and N2 vibrational relaxation terms.
    ISO 9613-1:1993 eq. (3)-(5); Bass et al., JASA 97(1) 1995.
    """
    f = np.asarray(f_hz, dtype=float)
    T = t_c + 273.15                      # [K]
    T0 = 293.15                           # [K] reference
    T01 = 273.16                          # [K] triple point
    p = p_kpa / 101.325                   # normalised pressure
    psat = 10.0 ** (-6.8346 * (T01 / T) ** 1.261 + 4.6151)   # saturation vapour pressure ratio
    h = rh * psat / p                     # [%] molar concentration of water vapour
    fr_o = p * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h))                          # [Hz] O2 relaxation
    fr_n = p * (T / T0) ** -0.5 * (9.0 + 280.0 * h * np.exp(-4.170 * ((T / T0) ** (-1.0 / 3.0) - 1.0)))  # [Hz] N2
    f2 = f * f
    alpha = 8.686 * f2 * (
        1.84e-11 / p * (T / T0) ** 0.5
        + (T / T0) ** -2.5 * (
            0.01275 * np.exp(-2239.1 / T) / (fr_o + f2 / fr_o)
            + 0.1068 * np.exp(-3352.0 / T) / (fr_n + f2 / fr_n)
        )
    )
    return alpha


ALPHA = air_absorption_db_per_m(FREQS)                 # [dB/m] on the rfft grid
ALPHA_MID = float(air_absorption_db_per_m(F_MID))      # [dB/m] band-centre value (folded into amplitudes)
F_CLIP = np.clip(FREQS, F0, F1)                        # in-band frequency for tilt models


def h_abs_shape(path_m):
    """Residual *spectral shape* of air absorption over path_m [m] (band-centre value removed)."""
    return np.clip(10.0 ** (-(ALPHA - ALPHA_MID) * path_m / 20.0), 0.0, 2.0)


def h_tilt(tilt_db):
    """Linear-in-dB magnitude tilt across the sweep band (tilt_db at F1 relative to F0)."""
    return 10.0 ** (tilt_db * (F_CLIP - F0) / BW / 20.0)


def h_lowpass(p):
    """Porous-absorber reflection: |H| = (f/F0)^-p above F0, unity below (clamped)."""
    return np.minimum(1.0, (np.maximum(FREQS, 1.0) / F0) ** (-p))


def h_gtd():
    """Keller GTD knife-edge diffraction coefficient magnitude ~ 1/sqrt(k) ~ 1/sqrt(f)."""
    return np.minimum(2.0, np.sqrt(F0 / np.maximum(FREQS, 1.0)))


# =============================================================================
# 4. Transducer response (speaker roll-off + port ripple + MEMS mic resonance)
# =============================================================================
def make_system_response(rng):
    """Per-device speaker+mic transfer function on the rfft grid, band-mean |H| = 1.

    Phone speakers roll off above ~15 kHz (0..-9 dB across the sweep), ports add
    +-1.5 dB ripple, MEMS mics have a Helmholtz resonance at 18-28 kHz (Q 2-6).
    The PWA's matched filter uses the *ideal* chirp, so this mismatch broadens
    the compressed pulse exactly as it will on real hardware.
    """
    tilt_db = rng.uniform(-9.0, 0.0)
    H = h_tilt(tilt_db).astype(complex)
    for _ in range(int(rng.integers(1, 4))):
        ripple_db = rng.uniform(0.0, 1.5)
        period = rng.uniform(1500.0, 4000.0)                       # [Hz]
        H *= 10.0 ** (ripple_db * np.sin(2 * np.pi * F_CLIP / period + rng.uniform(0, 2 * np.pi)) / 20.0)
    f_res = rng.uniform(18e3, 28e3)                                # [Hz] mic resonance
    q = rng.uniform(2.0, 6.0)
    gain = rng.uniform(0.0, 2.0)                                   # up to +9.5 dB peak
    s = 1j * FREQS * f_res / q
    H *= 1.0 + gain * s / (f_res ** 2 - FREQS ** 2 + s)            # 2nd-order band-pass added to flat
    band = (FREQS >= F0) & (FREQS <= F1)
    H /= np.mean(np.abs(H[band]))
    return H, tilt_db


# =============================================================================
# 5. Scene construction (class physics)
# =============================================================================
def sample_velocity(rng):
    """Closing velocity [m/s] (+ = walking toward the obstacle). 20 % standing still."""
    if rng.random() < 0.20:
        return float(np.clip(rng.normal(0.0, 0.03), -0.1, 0.1))
    return float(rng.uniform(-V_MAX, V_MAX))


def tau_of(r_m):
    """Two-way delay [s] of an echo at range r_m [m] incl. calibrated latency."""
    return LATENCY_S + 2.0 * np.asarray(r_m, dtype=float) / C_SOUND


def amp_of(R, r_m):
    """Echo amplitude [FS]: reflection coeff x spreading x band-centre air absorption (two-way)."""
    r = np.asarray(r_m, dtype=float)
    return A_REF * R * r ** (-SPREADING_EXP) * 10.0 ** (-ALPHA_MID * 2.0 * r / 20.0)


def make_group(tau, gamma, amp, phi, H):
    """A set of scatterers sharing one spectral filter H(f). Arrays are (K rows, S scatterers)."""
    shp = np.broadcast(tau, gamma, amp, phi).shape
    return dict(tau=np.broadcast_to(np.asarray(tau, float), shp),
                gamma=np.broadcast_to(np.asarray(gamma, float), shp),
                amp=np.broadcast_to(np.asarray(amp, float), shp),
                phi=np.broadcast_to(np.asarray(phi, float), shp),
                H=H)


def stack_scatterers(sc, row_gain):
    """sc: list of (r_k (K,), R_eff, phase, radial velocity). Returns tau/gamma/amp/phi (K,S)."""
    r = np.stack([np.asarray(s[0], float) for s in sc], axis=1)           # (K,S) [m]
    R = np.array([s[1] for s in sc])                                        # (S,)
    phi = np.array([s[2] for s in sc])                                      # (S,) [rad]
    vel = np.array([s[3] for s in sc])                                      # (S,) [m/s]
    tau = tau_of(r)
    amp = amp_of(R[None, :], r) * row_gain[:, None]
    gamma = np.broadcast_to(1.0 + 2.0 * vel[None, :] / C_SOUND, tau.shape)  # Doppler time dilation
    return tau, gamma, amp, np.broadcast_to(phi[None, :], tau.shape)


def build_scene(rng, cls, ages):
    """Return (groups, info) for one sample. ages: (K,) pulse ages, ages[0]=0 is the live pulse."""
    K = len(ages)
    t_age = -ages * PRI                                   # [s] <= 0, time of each pulse w.r.t. live pulse
    v = sample_velocity(rng)                              # [m/s]
    r_max = R_MAX_OPENING if cls == 2 else R_MAX
    r = float(rng.uniform(R_MIN, r_max))                  # [m] range at the live pulse
    # Target range at each past pulse (K,). Clamped: a receding walker cannot have been *inside*
    # the obstacle 2 s ago, and r -> 0 would blow up the r^-2 spreading law.
    r_k = np.maximum(r + v * ages * PRI, 0.35)
    h = float(rng.uniform(0.90, 1.15))                    # [m] phone height above floor
    a_bob = 0.004 + 0.030 * abs(v) / V_MAX                # [m] walking bob amplitude
    h_k = h + a_bob * np.sin(2 * np.pi * rng.uniform(1.6, 2.2) * t_age + rng.uniform(0, 2 * np.pi))  # (K,)
    R_floor = float(rng.uniform(0.2, 0.9))                # tile ~0.9, carpet ~0.2
    floor_lp = rng.uniform(0.0, 4.0) if R_floor < 0.5 else rng.uniform(0.0, 1.0)
    D90 = 10.0 ** (rng.uniform(-40.0, -26.0) / 20.0)      # transducer directivity toward the floor (-40..-26 dB)
    point = float(rng.uniform(0.6, 1.0))                  # hand-held pointing loss
    leak = float(rng.uniform(0.0, 0.25))                  # fraction of the target leaking into the slow baseline
    row_gain = np.ones(K)
    row_gain[1:] = leak
    ones_k = np.ones(K)
    groups = []

    # --- (a) speaker -> mic chassis leakage: direct path + chassis reverberation (static, gated)
    #     TX and RX share one device clock, so pulse-to-pulse timing jitter is sub-sample
    #     (sigma 0.15 smp); grip changes modulate the amplitude by ~0.5 %.  Both residuals leak
    #     through the incoherent clutter subtraction just outside the gate edge (0.40-0.55 m).
    d_sm = float(rng.uniform(0.08, 0.16))                 # [m] speaker-mic spacing
    a_dp = 10.0 ** (rng.uniform(-40.0, -20.0) / 20.0)     # [FS] direct path amplitude
    n_cr = int(rng.integers(3, 7))
    delays = np.concatenate([[0.0], rng.uniform(0.05e-3, 0.5e-3, n_cr)])              # [s]
    amps = np.concatenate([[a_dp], a_dp * rng.uniform(0.05, 0.4, n_cr) * rng.choice([-1.0, 1.0], n_cr)])
    phis = np.concatenate([[0.0], rng.uniform(0, 2 * np.pi, n_cr)])
    jitter = rng.normal(0.0, 0.15, K) / FS                # [s] per-pulse I/O timing jitter
    grip = 1.0 + rng.normal(0.0, 0.005, K)                # per-pulse amplitude modulation
    tau_dp = LATENCY_S + d_sm / C_SOUND + delays[None, :] + jitter[:, None]
    groups.append(make_group(tau_dp, 1.0, amps[None, :] * grip[:, None], phis[None, :], np.ones(NF)))
    n_dp = int(round((LATENCY_S + d_sm / C_SOUND) * FS)) # [smp] calibrated direct-path index (gate centre)

    # --- (b) floor ping straight below the phone (static except for walking bob) -> clutter
    tau, gamma, amp, phi = stack_scatterers([(h_k, R_floor * D90, 0.0, 0.0)], ones_k)
    groups.append(make_group(tau, gamma, amp, phi, h_abs_shape(2 * h) * h_lowpass(floor_lp)))

    # --- (c) user's own torso behind the phone (quasi-static w.r.t. phone: 2 mm arm jitter).
    #     Assumed inside the +-2 ms gate (phone held <= 0.4 m from the body); a body return beyond
    #     the gate is an obstacle the tracker has to deal with, not a classifier problem.
    r_ub = float(rng.uniform(0.20, 0.40))
    r_ub_k = r_ub + rng.normal(0.0, 0.002, K)
    r_ub_k[0] = r_ub
    tau, gamma, amp, phi = stack_scatterers([(r_ub_k, rng.uniform(0.05, 0.30), rng.uniform(0, 2 * np.pi), 0.0)], ones_k)
    groups.append(make_group(tau, gamma, amp, phi, h_abs_shape(2 * r_ub) * h_lowpass(rng.uniform(3.0, 7.0))))

    # --- (d) room reverberation: many weak oblique scatterers, each with radial velocity v*cos(theta)
    #     Anything nearer than the target is off-axis by definition (else it would *be* the target),
    #     so near scatterers only contribute their weak diffuse component.
    S = int(rng.integers(15, 51))
    r_i = rng.uniform(0.8, 8.0, S)                        # [m]
    R_i = 10.0 ** rng.uniform(-3.0, -1.7, S)              # 0.001 .. 0.02 (oblique / diffuse returns)
    R_i = np.where(r_i < r, R_i * 0.15, R_i)
    v_i = v * rng.uniform(-1.0, 1.0, S) + rng.normal(0.0, 0.05, S)
    r_ks = np.maximum(r_i[None, :] + v_i[None, :] * ages[:, None] * PRI, 0.5)
    tau = tau_of(r_ks)
    amp = amp_of(R_i[None, :], r_ks)
    gamma = np.broadcast_to(1.0 + 2.0 * v_i[None, :] / C_SOUND, tau.shape)
    phi = np.broadcast_to(rng.uniform(0, 2 * np.pi, S)[None, :], tau.shape)
    groups.append(make_group(tau, gamma, amp, phi, h_abs_shape(8.0) * h_lowpass(rng.uniform(0.0, 3.0))))

    # --- (e) the target itself
    info = dict(cls=cls, range_m=r, vel_mps=v, h_phone_m=h, n_dp=n_dp, leak=leak)
    if cls == 0:  # ------------------------------------------------ WALL
        R = float(rng.uniform(0.85, 0.98))
        sc = [(r_k, R * point, 0.0, v)]                                     # specular main return, phase 0 (rigid)
        if rng.random() < 0.35:                                             # door frame / baseboard edge
            sc.append((r_k + rng.uniform(0.03, 0.30), R * point * rng.uniform(0.03, 0.20), rng.uniform(0, 2 * np.pi), v))
        if rng.random() < 0.70:                                             # floor/wall dihedral corner (retro-reflector)
            r_dh = np.sqrt(r_k ** 2 + h_k ** 2)
            sc.append((r_dh, R * R_floor * point * rng.uniform(0.1, 0.5), 0.0, v))
        r_fb = 0.5 * (np.sqrt(r_k ** 2 + 4.0 * h_k ** 2) + r_k)            # phone->floor->wall->phone (x2 reciprocal)
        sc.append((r_fb, 2.0 * R * R_floor * point * rng.uniform(0.1, 0.4), rng.choice([0.0, np.pi]), v))
        if rng.random() < 0.5:                                              # wall -> user's body -> wall double bounce
            sc.append((2.0 * r_k + r_ub, R * R * rng.uniform(0.05, 0.30), rng.uniform(0, 2 * np.pi), 2.0 * v))
        info["target_nearest_m"] = float(min(np.asarray(x[0])[0] for x in sc))
        tau, gamma, amp, phi = stack_scatterers(sc, row_gain)
        groups.append(make_group(tau, gamma, amp, phi, h_abs_shape(2 * r) * h_tilt(rng.uniform(-1.0, 1.0))))

    elif cls == 1:  # ---------------------------------------------- SOFT
        R = float(rng.uniform(0.08, 0.35))
        p_lp = float(rng.uniform(5.5, 9.0))                                 # -> 60..80 % attenuation at 20-22 kHz
        spread_m = rng.uniform(15.0, 30.0) / FS * C_SOUND / 2.0            # [m] 15-30 smp two-way coat dispersion
        layers = [(0.0, rng.uniform(0.15, 0.50)),                           # coat outer surface
                  (rng.uniform(0.2, 0.5) * spread_m, rng.uniform(0.10, 0.40)),   # lining
                  (rng.uniform(0.5, 1.0) * spread_m, 1.0)]                  # torso / front surface
        M = int(rng.integers(8, 17))                                        # diffuse micro-scatterer continuum
        u = rng.uniform(0.0, 1.0, M)
        layers += [(ui * spread_m, np.exp(-3.0 * ui) * rng.uniform(0.05, 0.30)) for ui in u]
        # Body / furniture glint cluster: a person is not a 5 cm deep target.  Head, shoulders,
        # belly, hands (or cushions, armrests, curtain folds) sit 0-SOFT_CLUSTER_DEPTH_M behind the
        # front surface, each a diffuse glint.  This spatial extent is the physical signature that
        # separates a SOFT obstacle from a point-like door edge once the Hann-LFM pulse (20 smp at
        # -3 dB) has smeared the coat dispersion into the main lobe.
        for _ in range(int(rng.integers(2, 6))):
            layers.append((spread_m + rng.uniform(0.0, SOFT_CLUSTER_DEPTH_M), rng.uniform(0.15, 0.80)))
        sc = [(r_k + d + dv * ages * PRI, R * f * point, rng.uniform(0, 2 * np.pi), v + dv)
              for (d, f), dv in zip(layers, rng.normal(0.0, 0.15, len(layers)))]       # micro-Doppler
        if rng.random() < 0.6:                                              # swinging arms
            for _ in range(int(rng.integers(1, 3))):
                dv = rng.uniform(-0.6, 0.6)
                sc.append((r_k + rng.uniform(-0.05, 0.15) + dv * ages * PRI, R * point * rng.uniform(0.05, 0.25),
                           rng.uniform(0, 2 * np.pi), v + dv))
        r_fb = 0.5 * (np.sqrt(r_k ** 2 + 4.0 * h_k ** 2) + r_k)
        sc.append((r_fb, 2.0 * R * R_floor * point * rng.uniform(0.1, 0.4), rng.uniform(0, 2 * np.pi), v))
        info["target_nearest_m"] = float(min(np.asarray(x[0])[0] for x in sc))
        tau, gamma, amp, phi = stack_scatterers(sc, row_gain)
        groups.append(make_group(tau, gamma, amp, phi, h_abs_shape(2 * r) * h_lowpass(p_lp)))

    else:  # --------------------------------------------------------- OPENING
        W = float(rng.uniform(0.75, 1.05))                                  # [m] door width
        d_off = float(rng.uniform(-0.35, 0.35))                             # [m] lateral offset from door centre
        R = 0.0
        sc = []
        for side in (+1.0, -1.0):                                           # two jamb knife edges (GTD, -pi/4 phase)
            r_j = np.sqrt(r_k ** 2 + (0.5 * W + side * d_off) ** 2)
            R_j = float(rng.uniform(0.01, 0.08))
            R = max(R, R_j)
            sc.append((r_j, R_j * point, -np.pi / 4 + rng.choice([0.0, np.pi]), v))
        if rng.random() < 0.8:                                              # lintel (top edge), directivity penalised
            dz = rng.uniform(0.85, 1.15)
            r_l = np.sqrt(r_k ** 2 + dz ** 2)
            sc.append((r_l, rng.uniform(0.01, 0.06) * point * float((r / math.hypot(r, dz)) ** 1.5), -np.pi / 4, v))
        if rng.random() < 0.3:                                              # threshold strip / floor edge
            sc.append((np.sqrt(r_k ** 2 + h_k ** 2), rng.uniform(0.005, 0.03), -np.pi / 4, v))
        info["target_nearest_m"] = float(min(np.asarray(x[0])[0] for x in sc))
        tau, gamma, amp, phi = stack_scatterers(sc, row_gain)
        groups.append(make_group(tau, gamma, amp, phi, h_abs_shape(2 * r) * h_gtd()))
        # far room beyond the doorway: wall > 3.9 m plus a few furniture scatterers
        r_far = float(rng.uniform(3.9, 7.5))
        far = [(r_far + v * ages * PRI, rng.uniform(0.6, 0.98) * point, 0.0, v)]
        for _ in range(int(rng.integers(1, 5))):
            far.append((rng.uniform(3.9, 7.5) + v * ages * PRI, 10.0 ** rng.uniform(-2.0, -0.8), rng.uniform(0, 2 * np.pi), v))
        tau, gamma, amp, phi = stack_scatterers(far, row_gain)
        groups.append(make_group(tau, gamma, amp, phi, h_abs_shape(2 * r_far) * h_lowpass(rng.uniform(0.0, 2.0))))

    info["R"] = R
    return groups, info


# =============================================================================
# 6. Receiver chain
# =============================================================================
def synth_rows(groups, K, H_sys):
    """Time-domain synthesis of K receive windows (K, N_RX) [FS] from scatterer groups."""
    Xsum = np.zeros((K, NF), dtype=complex)
    n_loc = np.arange(N_CHUNK)
    rows = np.arange(K)[:, None, None]
    for g in groups:
        tau, gamma, amp, phi = g["tau"], g["gamma"], g["amp"], g["phi"]
        n0 = np.floor(tau * FS).astype(np.int64)                    # (K,S) first sample of each echo copy
        idx = n0[..., None] + n_loc                                  # (K,S,N_CHUNK)
        t = idx / FS
        chunk = amp[..., None] * chirp_at(gamma[..., None] * (t - tau[..., None]), phi[..., None])
        valid = idx < NFFT
        flat = (np.broadcast_to(rows, idx.shape) * NFFT + idx)[valid]
        x = np.bincount(flat, weights=chunk[valid], minlength=K * NFFT).reshape(K, NFFT)
        Xsum += sfft.rfft(x, axis=1) * g["H"][None, :]
    Xsum *= H_sys[None, :]
    return sfft.irfft(Xsum, n=NFFT, axis=1)[:, :N_RX]


def add_noise(rx, rng, K):
    """Mic thermal (Johnson-Nyquist, white) + 1/f room background + occasional impulsive clicks."""
    Kr, N = rx.shape
    sigma_th = 10.0 ** (rng.uniform(-104.0, -92.0) / 20.0)          # [FS rms] MEMS mic EIN ~ -100 dBFS (+-6 dB)
    sigma_pink = 10.0 ** (rng.uniform(-100.0, -84.0) / 20.0)        # [FS rms] total 1/f power (in-band share ~ -15 dB)
    scale = np.ones((Kr, 1))
    scale[1:] = np.sqrt((Kr - 1) / N_BASELINE_PULSES)                # emulate a 40-pulse mean with Kr-1 rows
    white = rng.standard_normal((Kr, N)) * sigma_th
    Wn = sfft.rfft(rng.standard_normal((Kr, N)), axis=1)
    fr = np.fft.rfftfreq(N, 1.0 / FS)
    shp = 1.0 / np.sqrt(np.maximum(fr, 20.0))
    shp[0] = 0.0
    pink = sfft.irfft(Wn * shp[None, :], n=N, axis=1)
    pink *= sigma_pink / np.sqrt(np.mean(pink ** 2, axis=1, keepdims=True))
    rx = rx + (white + pink) * scale
    if rng.random() < 0.10:                                          # keys, footsteps, door clicks
        for _ in range(int(rng.integers(1, 4))):
            k = 0 if rng.random() < 0.5 else int(rng.integers(0, Kr))
            n0 = int(rng.integers(0, N - 16))
            L = int(rng.integers(3, 12))
            a = 10.0 ** (rng.uniform(-55.0, -35.0) / 20.0) * (1.0 if k == 0 else (Kr - 1) / N_BASELINE_PULSES)
            rx[k, n0:n0 + L] += a * rng.standard_normal(L) * np.exp(-np.arange(L) / rng.uniform(1.0, 4.0))
    return rx, sigma_th


def quantize(rx, bits=ADC_BITS):
    q = 2.0 ** (bits - 1)
    return np.clip(np.round(rx * q) / q, -1.0, 1.0 - 1.0 / q)


def matched_filter_envelope(rx):
    """FFT cross-correlation with the reference chirp, then analytic-signal envelope |z[n]|.

    Y = X * conj(S); the analytic spectrum (2*Y for f>0, 0 for f<0) is inverse
    transformed directly, which equals hilbert(correlate(x, s)) exactly.
    """
    X = sfft.fft(rx, n=NFFT, axis=1)
    Y = X * MF_F[None, :]
    Y[:, 1:NFFT // 2] *= 2.0
    Y[:, NFFT // 2 + 1:] = 0.0
    z = sfft.ifft(Y, axis=1)[:, :N_RX]
    return np.abs(z)


def cfar_detect(profile, n_lo, n_hi, guard=20, train=24, pfa=1e-3, peak_profile=None, valid=None):
    """Smallest-of cell-averaging CFAR on the power profile.

    Returns (peak index, passed, noise amplitude estimate).

    * Noise = min(mean of leading window, mean of lagging window).  Plain CA-CFAR
      self-masks extended targets (a coat + torso + swinging arm spans 60-90 smp,
      so both training windows sit on the target); SO-CFAR only needs one clean side.
    * guard=20 keeps the compressed pulse's own main lobe (Hann^2 spectrum ->
      -12 dB at +-19 smp) out of the training cells.
    * valid (bool mask) excludes gated cells from the statistics so the gate edge
      does not offer a zero-noise window; a side with < 8 valid cells is ignored.
    * peak_profile (optional) is the *un-gated* profile used only for the
      local-maximum test, so a clutter skirt cut off by the gate does not
      masquerade as a peak at the gate edge.
    * Threshold factor alpha = N (Pfa^(-1/N) - 1) (square-law CA-CFAR form).  The
      strongest local maximum crossing the threshold inside [n_lo, n_hi) wins; if
      nothing crosses, the strongest local bump is returned with passed=False (the
      PWA would tag such a detection low-confidence).
    """
    p2 = profile ** 2
    N = len(p2)
    v = np.ones(N, dtype=bool) if valid is None else valid
    cs = np.concatenate([[0.0], np.cumsum(p2 * v)])
    cv = np.concatenate([[0], np.cumsum(v.astype(np.int64))])
    idx = np.arange(N)

    def wmean(lo, hi):
        lo = np.clip(lo, 0, N)
        hi = np.clip(hi, 0, N)
        cnt = cv[hi] - cv[lo]
        m = (cs[hi] - cs[lo]) / np.maximum(cnt, 1)
        return np.where(cnt >= 8, m, np.nan), cnt

    m1, c1 = wmean(idx - guard - train, idx - guard)
    m2, c2 = wmean(idx + guard + 1, idx + guard + 1 + train)
    with np.errstate(invalid="ignore"):
        noise = np.fmin(m1, m2)                                       # smallest-of; NaN-aware
    fallback = np.median(p2[n_lo:n_hi][v[n_lo:n_hi]]) if v[n_lo:n_hi].any() else 0.0
    noise = np.where(np.isnan(noise), fallback, noise)
    n_eff = np.maximum(np.where(np.isnan(m1), 0, c1) + np.where(np.isnan(m2), 0, c2), 1)
    alpha = n_eff * (pfa ** (-1.0 / n_eff) - 1.0)
    det = p2 > alpha * noise
    pp = profile if peak_profile is None else peak_profile
    lmx = np.zeros(N, dtype=bool)
    lmx[1:-1] = (pp[1:-1] >= pp[:-2]) & (pp[1:-1] > pp[2:])
    region = np.zeros(N, dtype=bool)
    region[n_lo:n_hi] = True
    cand = det & lmx & region
    if cand.any():
        p = int(np.argmax(np.where(cand, profile, -1.0)))
        passed = True
    else:
        sub = np.where(lmx[n_lo:n_hi], profile[n_lo:n_hi], -1.0)
        p = n_lo + int(np.argmax(sub if sub.max() > 0 else profile[n_lo:n_hi]))
        passed = False
    return p, passed, float(np.sqrt(max(noise[p], 1e-30)))


def simulate_sample(rng, cls, k_base, return_profile=False):
    """Full chain for one labelled sample. Returns (crop float32[64], meta float32[len(META_COLUMNS)])."""
    ages = np.concatenate([[0.0], np.round(np.linspace(1, N_BASELINE_PULSES, k_base))])
    K = len(ages)
    H_sys, tilt_db = make_system_response(rng)
    groups, info = build_scene(rng, cls, ages)
    rx = synth_rows(groups, K, H_sys)
    rx, sigma_th = add_noise(rx, rng, K)
    rx = quantize(rx)
    env = matched_filter_envelope(rx)
    live = env[0].copy()
    base = env[1:].mean(axis=0)
    profile_raw = np.maximum(live - base, 0.0)                      # un-gated, for the local-maximum test only
    n_dp = info["n_dp"]
    g0, g1 = max(n_dp - GATE_HALF, 0), n_dp + GATE_HALF + 1
    live[g0:g1] = 0.0                                               # direct-path gate +-2 ms
    base[g0:g1] = 0.0
    profile = np.maximum(live - base, 0.0)                          # clutter subtraction (40-pulse static baseline)
    n_lo = max(g1, L0 + int(2.0 * R_SEARCH_MIN / C_SOUND * FS))
    n_hi = L0 + int(2.0 * R_SEARCH_MAX / C_SOUND * FS)
    valid = np.ones(N_RX, dtype=bool)
    valid[g0:g1] = False
    p, passed, noise_amp = cfar_detect(profile, n_lo, n_hi, peak_profile=profile_raw, valid=valid)
    seg = profile[p - HALF:p + HALF].astype(np.float32)
    pk = float(seg.max())
    if pk > 0:
        seg = seg / pk                                              # peak-normalise to [0, 1]
    snr_db = min(20.0 * math.log10(max(profile[p], 1e-30) / max(noise_amp, 1e-30)), 60.0)
    meta = np.array([cls, info["range_m"], info["vel_mps"], info["R"], snr_db, float(passed), p,
                     (p - L0) * C_SOUND / (2.0 * FS), tilt_db, 20.0 * math.log10(sigma_th),
                     info["h_phone_m"], info["leak"], profile[p], info["target_nearest_m"], 1.0], dtype=np.float32)
    if return_profile:
        return seg, meta, profile
    return seg, meta


MAX_TRIES = 12                        # rejection-sampling budget per sample
DETECT_TOL_M = 0.30                   # [m] detection must land this close to the nearest target scatterer


def generate_one(task):
    """One labelled sample. Rejection sampling keeps only crops that are actually centred on the
    labelled target (|detected range - nearest target scatterer| <= DETECT_TOL_M).  The deployed
    classifier only ever runs on tracker-confirmed detections, so crops centred on clutter or on a
    noise spike would carry a wrong label.  Pass keep_misdetections=True to disable (label noise)."""
    idx, seed, cls, k_base, keep_misdetections = task
    rng = np.random.default_rng([seed, idx])                        # deterministic per index, worker-count independent
    for attempt in range(1, MAX_TRIES + 1):
        seg, meta = simulate_sample(rng, cls, k_base)
        meta[META_COLUMNS.index("n_tries")] = attempt
        ok = abs(meta[META_COLUMNS.index("range_detected_m")] - meta[META_COLUMNS.index("target_nearest_m")]) <= DETECT_TOL_M
        if ok or keep_misdetections:
            return seg, meta
    meta[META_COLUMNS.index("n_tries")] = -attempt                  # gave up: flagged negative, sample kept
    return seg, meta


def generate_dataset(n_samples, seed, k_base, workers, keep_misdetections=False):
    n_per = n_samples // len(CLASSES)
    tasks = [(i, seed, i % len(CLASSES), k_base, keep_misdetections) for i in range(n_per * len(CLASSES))]
    X = np.zeros((len(tasks), WIN), dtype=np.float32)
    M = np.zeros((len(tasks), len(META_COLUMNS)), dtype=np.float32)
    t0 = time.time()
    step = max(1, len(tasks) // 20)
    if workers > 1:
        with mp.Pool(workers) as pool:
            for i, (seg, meta) in enumerate(pool.imap(generate_one, tasks, chunksize=16)):
                X[i], M[i] = seg, meta
                if (i + 1) % step == 0:
                    print(f"  generated {i + 1:6d}/{len(tasks)}  ({time.time() - t0:6.1f} s)", flush=True)
    else:
        for i, task in enumerate(tasks):
            X[i], M[i] = generate_one(task)
            if (i + 1) % step == 0:
                print(f"  generated {i + 1:6d}/{len(tasks)}  ({time.time() - t0:6.1f} s)", flush=True)
    print(f"  done: {len(tasks)} samples in {time.time() - t0:.1f} s "
          f"({1e3 * (time.time() - t0) / len(tasks):.1f} ms/sample wall)")
    return X, M


def split_dataset(X, M, seed, val_frac=0.2):
    """Stratified 80/20 split."""
    rng = np.random.default_rng(seed + 12345)
    y = M[:, 0].astype(np.int64)
    tr, va = [], []
    for c in range(len(CLASSES)):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        n_val = int(round(val_frac * len(idx)))
        va.append(idx[:n_val])
        tr.append(idx[n_val:])
    tr = np.concatenate(tr)
    va = np.concatenate(va)
    rng.shuffle(tr)
    rng.shuffle(va)
    return (X[tr], y[tr], M[tr]), (X[va], y[va], M[va])


def dataset_report(M):
    y = M[:, 0].astype(int)
    it = META_COLUMNS.index("n_tries")
    err = np.abs(M[:, META_COLUMNS.index("range_detected_m")] - M[:, META_COLUMNS.index("target_nearest_m")])
    print("  class     n    CFAR-pass  median SNR  range [m]   on-target  mean tries  gave-up")
    for c, name in enumerate(CLASSES):
        m = M[y == c]
        print(f"  {name:8s} {len(m):5d}  {100 * m[:, 5].mean():6.1f} %  {np.median(m[:, 4]):7.1f} dB"
              f"  {m[:, 1].min():.2f}-{m[:, 1].max():.2f}   {100 * np.mean(err[y == c] <= DETECT_TOL_M):5.1f} %"
              f"   {np.abs(m[:, it]).mean():6.2f}     {100 * np.mean(m[:, it] < 0):5.1f} %")


def physics_banner():
    x = np.zeros(N_RX)
    x[500:500 + N_CHIRP] = REF                                      # delayed copy so both skirts are inside the window
    env = matched_filter_envelope(x[None, :])[0]
    pk = env.max()
    w3 = int(np.sum(env > pk / math.sqrt(2.0)))
    w12 = int(np.sum(env > pk / 4.0))
    print("Physics summary")
    print(f"  chirp {F0 / 1e3:.1f}-{F1 / 1e3:.1f} kHz, {T_CHIRP * 1e3:.0f} ms, Hann, fs={FS:.0f} Hz, TB={BW * T_CHIRP:.1f}")
    print(f"  range resolution c/2B = {C_SOUND / (2 * BW) * 100:.1f} cm; 1/B = {FS / BW:.1f} smp (rectangular window); "
          f"actual Hann-LFM compressed pulse: -3 dB width {w3} smp, -12 dB width {w12} smp")
    for f in (F0, 20e3, F1):
        print(f"  air absorption ISO 9613-1 @ {f / 1e3:4.1f} kHz, 20 C, 50 % RH: {float(air_absorption_db_per_m(f)):.3f} dB/m")
    print(f"  A_REF = {A_REF:.1e} FS ({20 * math.log10(A_REF):.0f} dBFS) for R=1 at 1 m; spreading ~ r^-{SPREADING_EXP:.0f}")
    print(f"  gate: +-{GATE_HALF} smp; search {R_SEARCH_MIN}-{R_SEARCH_MAX} m; crop {WIN} smp = "
          f"{WIN / FS * 1e3:.2f} ms = {WIN / FS * C_SOUND / 2 * 100:.1f} cm of range")


def preview_plot(path, seed, k_base):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping preview plot")
        return
    n_ex = 6
    fig, axes = plt.subplots(4, n_ex, figsize=(3.2 * n_ex, 9))
    means = []
    for c, name in enumerate(CLASSES):
        crops = []
        for j in range(n_ex):
            rng = np.random.default_rng([seed + 777, c, j])
            seg, meta, prof = simulate_sample(rng, c, k_base, return_profile=True)
            crops.append(seg)
            ax = axes[c, j]
            ax.plot(seg, lw=1.2)
            ax.set_ylim(0, 1.05)
            ax.set_title(f"{name} r={meta[1]:.2f}m v={meta[2]:+.1f} SNR={meta[4]:.0f}dB", fontsize=8)
            ax.set_xticks([0, 32, 63])
        means.append(np.mean(crops, axis=0))
    for j in range(n_ex):
        axes[3, j].axis("off")
    ax = axes[3, 0]
    ax.axis("on")
    for c, name in enumerate(CLASSES):
        ax.plot(means[c], label=name)
    ax.legend(fontsize=8)
    ax.set_title("mean crop (6 examples each)", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"  preview written: {path}")


# =============================================================================
# 7. EchoNetTiny model + training (PyTorch)
# =============================================================================
def build_model(head="gap"):
    import torch.nn as nn

    class EchoNetTiny(nn.Module):
        """1-D CNN on (batch, 1, 64). ~2.3k params (gap) / ~10k (flatten). Sub-millisecond CPU inference."""

        def __init__(self, n_classes=3, head="gap"):
            super().__init__()
            self.head = head
            self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
            self.bn1 = nn.BatchNorm1d(16)
            self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm1d(32)
            self.pool = nn.MaxPool1d(2)
            self.act = nn.ReLU()
            self.fc1 = nn.Linear(32 if head == "gap" else 32 * (WIN // 4), 16)
            self.drop = nn.Dropout(0.2)
            self.fc2 = nn.Linear(16, n_classes)

        def forward(self, x):
            x = self.pool(self.act(self.bn1(self.conv1(x))))        # (B,16,32)
            x = self.pool(self.act(self.bn2(self.conv2(x))))        # (B,32,16)
            x = x.mean(dim=2) if self.head == "gap" else x.flatten(1)
            x = self.drop(self.act(self.fc1(x)))
            return self.fc2(x)

    return EchoNetTiny(len(CLASSES), head)


def augment_batch(xb, torch):
    """Train-time augmentation: +-2 smp peak-alignment jitter, gain jitter, additive noise, re-normalise."""
    import torch.nn.functional as F
    B = xb.shape[0]
    shifts = torch.randint(-2, 3, (B,), device=xb.device)
    xp = F.pad(xb, (2, 2), mode="replicate")
    idx = (torch.arange(WIN, device=xb.device)[None, :] + 2 + shifts[:, None])[:, None, :]
    xs = torch.gather(xp, 2, idx)
    xs = xs * (1.0 + 0.03 * torch.randn(B, 1, 1, device=xb.device)) + 0.01 * torch.randn_like(xs)
    xs = xs.clamp_min(0.0)
    return xs / xs.amax(dim=2, keepdim=True).clamp_min(1e-6)


def confusion_matrix(y_true, y_pred, n=3):
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def print_confusion(cm):
    w = max(8, max(len(c) for c in CLASSES) + 2)
    print("  Confusion matrix (rows = true, cols = predicted)")
    print("  " + " " * 14 + "".join(f"{c:>{w}s}" for c in CLASSES) + "   recall")
    for i, c in enumerate(CLASSES):
        rec = cm[i, i] / max(cm[i].sum(), 1)
        print(f"  {'true ' + c:14s}" + "".join(f"{cm[i, j]:>{w}d}" for j in range(len(CLASSES))) + f"   {100 * rec:6.1f} %")
    prec = [cm[j, j] / max(cm[:, j].sum(), 1) for j in range(len(CLASSES))]
    print("  " + "precision     " + "".join(f"{100 * p:>{w - 2}.1f} %" for p in prec))
    f1 = [2 * prec[i] * (cm[i, i] / max(cm[i].sum(), 1)) / max(prec[i] + cm[i, i] / max(cm[i].sum(), 1), 1e-9)
          for i in range(len(CLASSES))]
    print("  macro-F1 = {:.3f}   accuracy = {:.2f} %".format(np.mean(f1), 100 * np.trace(cm) / cm.sum()))


def train_model(X_train, y_train, X_val, y_val, epochs, head, seed, batch_size=128, lr=3e-3):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    model = build_model(head)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  EchoNetTiny(head={head}): {n_params} parameters")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    Xt = torch.from_numpy(X_train)[:, None, :]
    yt = torch.from_numpy(y_train)
    Xv = torch.from_numpy(X_val)[:, None, :]
    yv = torch.from_numpy(y_val)
    best_acc, best_state = -1.0, None
    print(f"  {'epoch':>5s} {'train_loss':>10s} {'val_loss':>9s} {'val_acc':>8s} {'lr':>9s}")
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xt))
        tot, n = 0.0, 0
        for i in range(0, len(perm), batch_size):
            b = perm[i:i + batch_size]
            xb = augment_batch(Xt[b], torch)
            loss = crit(model(xb), yt[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
            n += len(b)
        sched.step()
        model.eval()
        with torch.no_grad():
            logits = model(Xv)
            vloss = crit(logits, yv).item()
            vacc = (logits.argmax(1) == yv).float().mean().item()
        if vacc > best_acc:
            best_acc, best_state = vacc, copy.deepcopy(model.state_dict())
        print(f"  {ep:5d} {tot / n:10.4f} {vloss:9.4f} {100 * vacc:7.2f}% {opt.param_groups[0]['lr']:9.2e}")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(Xv).argmax(1).numpy()
    cm = confusion_matrix(y_val, pred)
    print(f"  best validation accuracy: {100 * best_acc:.2f} %")
    print_confusion(cm)
    # single-sample CPU latency (the on-device figure of merit)
    torch.set_num_threads(1)
    x1 = Xv[:1]
    with torch.no_grad():
        for _ in range(50):
            model(x1)
        t0 = time.perf_counter()
        for _ in range(500):
            model(x1)
        dt = (time.perf_counter() - t0) / 500
    print(f"  torch CPU inference (batch 1, 1 thread): {1e6 * dt:.0f} us")
    return model, best_acc, n_params, cm


# =============================================================================
# 8. Export: BN folding, numpy reference, ONNX, standalone JS
# =============================================================================
def fold_weights(model):
    """Fold BatchNorm (eval statistics) into the preceding conv: w' = w*g/sqrt(v+eps), b' = (b-m)*g/sqrt(v+eps)+beta."""
    sd = {k: v.detach().cpu().numpy().astype(np.float64) for k, v in model.state_dict().items()}

    def fold(prefix_c, prefix_b, eps):
        s = sd[f"{prefix_b}.weight"] / np.sqrt(sd[f"{prefix_b}.running_var"] + eps)
        w = sd[f"{prefix_c}.weight"] * s[:, None, None]
        b = (sd[f"{prefix_c}.bias"] - sd[f"{prefix_b}.running_mean"]) * s + sd[f"{prefix_b}.bias"]
        return w.astype(np.float32), b.astype(np.float32)

    w1, b1 = fold("conv1", "bn1", model.bn1.eps)
    w2, b2 = fold("conv2", "bn2", model.bn2.eps)
    return dict(conv1_w=w1, conv1_b=b1, conv2_w=w2, conv2_b=b2,
                fc1_w=sd["fc1.weight"].astype(np.float32), fc1_b=sd["fc1.bias"].astype(np.float32),
                fc2_w=sd["fc2.weight"].astype(np.float32), fc2_b=sd["fc2.bias"].astype(np.float32))


def numpy_forward(Wf, x, head):
    """Reference forward pass with folded weights (mirrors the JS exactly). x: (64,) -> logits (3,)."""
    def conv1d(x2, w, b, pad):
        cin, L = x2.shape
        k = w.shape[2]
        xp = np.pad(x2, ((0, 0), (pad, pad)))
        win = np.stack([xp[:, j:j + L] for j in range(k)], axis=-1)     # (cin, L, k)
        return np.einsum("oik,ilk->ol", w, win) + b[:, None]

    def pool2(a):
        return a.reshape(a.shape[0], a.shape[1] // 2, 2).max(axis=2)

    a = pool2(np.maximum(conv1d(x[None, :].astype(np.float32), Wf["conv1_w"], Wf["conv1_b"], 2), 0))
    a = pool2(np.maximum(conv1d(a, Wf["conv2_w"], Wf["conv2_b"], 1), 0))
    feat = a.mean(axis=1) if head == "gap" else a.reshape(-1)
    hdn = np.maximum(Wf["fc1_w"] @ feat + Wf["fc1_b"], 0)
    return Wf["fc2_w"] @ hdn + Wf["fc2_b"]


def export_onnx(model, path):
    import torch
    model.eval()
    dummy = torch.zeros(1, 1, WIN)
    kwargs = dict(input_names=["input"], output_names=["logits"],
                  dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)
    try:
        try:
            torch.onnx.export(model, dummy, path, dynamo=False, **kwargs)
        except TypeError:                                             # older torch: no dynamo kwarg
            torch.onnx.export(model, dummy, path, **kwargs)
    except Exception as e:                                            # pragma: no cover
        try:
            torch.onnx.export(model, dummy, path, dynamo=True, **kwargs)
        except Exception as e2:
            print(f"  ONNX export FAILED ({type(e).__name__}: {e}; dynamo retry: {e2}).\n"
                  f"  Install with: pip install onnx onnxscript")
            return False
    try:
        import onnx
        onnx.checker.check_model(onnx.load(path))
        print(f"  ONNX written and checked: {path} ({os.path.getsize(path)} bytes)")
    except ImportError:
        print(f"  ONNX written: {path} (onnx package not installed, checker skipped)")
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        out = sess.run(None, {"input": dummy.numpy()})[0]
        with torch.no_grad():
            ref = model(dummy).numpy()
        print(f"  onnxruntime check: max |diff| = {np.abs(out - ref).max():.2e}")
    except ImportError:
        pass
    return True


def _js_array(a):
    return "new Float32Array([" + ",".join(f"{float(v):.9g}" for v in np.asarray(a, np.float32).ravel()) + "])"


def export_js(Wf, head, path, meta, test_x, test_logits):
    """Write a dependency-free UMD module with the folded weights and a pure-JS forward pass."""
    shapes = {k: list(v.shape) for k, v in Wf.items()}
    weights_js = "\n".join(f"    {k}: {_js_array(v)}, // shape {shapes[k]}" for k, v in Wf.items())
    js = f"""/* echonet_weights.js -- auto-generated by generate_echonet_dataset.py ({meta['generated_utc']})
 * EchoNetTiny obstacle classifier for PITCHBLACK. Pure JavaScript, zero dependencies.
 *
 *   Input : Float32Array(64) -- matched-filter envelope window centred on the
 *           detected echo peak (peak at index 32), peak-normalised to [0, 1].
 *           Use EchoNet.normalize(win) if you pass a raw envelope crop.
 *   Output: {{ logits: Float32Array(3), probs: Float32Array(3), classIndex,
 *             className: 'WALL'|'SOFT'|'OPENING', confidence }}
 *
 * Architecture: Conv1d(1->16,k5,p2)+BN+ReLU+MaxPool2 -> Conv1d(16->32,k3,p1)+BN+ReLU
 *               +MaxPool2 -> {head.upper()} -> Linear(->16)+ReLU -> Linear(16->3).
 * BatchNorm layers are folded into the conv weights (inference only).
 * Works in browsers, Web Workers, AudioWorklets (globalThis) and Node (module.exports).
 * Synthetic validation accuracy: {100 * meta['val_accuracy']:.2f} %  |  params: {meta['n_params']}
 */
(function (root, factory) {{
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.EchoNet = factory();
}}(typeof globalThis !== 'undefined' ? globalThis : (typeof self !== 'undefined' ? self : this), function () {{
  'use strict';

  const CLASSES = {json.dumps(CLASSES)};
  const META = {json.dumps(meta)};
  const HEAD = '{head}';
  const WIN = {WIN};

  const W = {{
{weights_js}
  }};

  // y[co][n] = relu(b[co] + sum_ci sum_j w[co][ci][j] * x[ci][n + j - pad])   (zero padded)
  function conv1d(x, cIn, L, w, b, cOut, k, pad) {{
    const y = new Float32Array(cOut * L);
    for (let co = 0; co < cOut; co++) {{
      const bo = b[co];
      for (let n = 0; n < L; n++) {{
        let acc = bo;
        for (let ci = 0; ci < cIn; ci++) {{
          const xo = ci * L, wo = (co * cIn + ci) * k;
          for (let j = 0; j < k; j++) {{
            const m = n + j - pad;
            if (m >= 0 && m < L) acc += w[wo + j] * x[xo + m];
          }}
        }}
        y[co * L + n] = acc > 0 ? acc : 0;   // fused ReLU
      }}
    }}
    return y;
  }}

  function maxPool2(x, C, L) {{
    const Lo = L >> 1, y = new Float32Array(C * Lo);
    for (let c = 0; c < C; c++)
      for (let n = 0; n < Lo; n++) {{
        const a = x[c * L + 2 * n], b = x[c * L + 2 * n + 1];
        y[c * Lo + n] = a > b ? a : b;
      }}
    return y;
  }}

  function dense(x, w, b, nOut, nIn, relu) {{
    const y = new Float32Array(nOut);
    for (let o = 0; o < nOut; o++) {{
      let acc = b[o];
      const wo = o * nIn;
      for (let i = 0; i < nIn; i++) acc += w[wo + i] * x[i];
      y[o] = relu && acc < 0 ? 0 : acc;
    }}
    return y;
  }}

  /** Peak-normalise a raw envelope crop to [0, 1] (returns a copy). */
  function normalize(win) {{
    const x = Float32Array.from(win);
    let m = 0;
    for (let i = 0; i < x.length; i++) if (x[i] > m) m = x[i];
    if (m > 0) for (let i = 0; i < x.length; i++) x[i] /= m;
    return x;
  }}

  /** Forward pass. input: Float32Array(64), already peak-normalised. */
  function forward(input) {{
    if (!input || input.length !== WIN) throw new Error('EchoNet.forward expects a 64-sample window');
    let a = conv1d(input, 1, 64, W.conv1_w, W.conv1_b, 16, 5, 2);   // (16,64)
    a = maxPool2(a, 16, 64);                                          // (16,32)
    a = conv1d(a, 16, 32, W.conv2_w, W.conv2_b, 32, 3, 1);           // (32,32)
    a = maxPool2(a, 32, 32);                                          // (32,16)
    let feat;
    if (HEAD === 'gap') {{
      feat = new Float32Array(32);
      for (let c = 0; c < 32; c++) {{
        let s = 0;
        for (let n = 0; n < 16; n++) s += a[c * 16 + n];
        feat[c] = s / 16;
      }}
    }} else {{
      feat = a;                                                       // flatten (32*16)
    }}
    const h = dense(feat, W.fc1_w, W.fc1_b, 16, feat.length, true);
    const logits = dense(h, W.fc2_w, W.fc2_b, CLASSES.length, 16, false);
    let mx = -Infinity;
    for (let i = 0; i < logits.length; i++) if (logits[i] > mx) mx = logits[i];
    const probs = new Float32Array(logits.length);
    let z = 0;
    for (let i = 0; i < logits.length; i++) {{ probs[i] = Math.exp(logits[i] - mx); z += probs[i]; }}
    let best = 0;
    for (let i = 0; i < logits.length; i++) {{ probs[i] /= z; if (probs[i] > probs[best]) best = i; }}
    return {{ logits, probs, classIndex: best, className: CLASSES[best], confidence: probs[best] }};
  }}

  // Self-test vector captured from the PyTorch model at export time.
  const TEST_INPUT = {_js_array(test_x)};
  const TEST_LOGITS = {_js_array(test_logits)};
  function selfTest() {{
    const out = forward(TEST_INPUT);
    let err = 0;
    for (let i = 0; i < TEST_LOGITS.length; i++) err = Math.max(err, Math.abs(out.logits[i] - TEST_LOGITS[i]));
    return {{ ok: err < 1e-3, maxAbsError: err, className: out.className }};
  }}

  return {{ forward, normalize, selfTest, CLASSES, META, W, HEAD, WIN }};
}}));
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(js)
    print(f"  JS written: {path} ({os.path.getsize(path)} bytes)")


def verify_js_with_node(js_path, X, ref_logits):
    node = shutil.which("node")
    if node is None:
        print("  node not found on PATH, JS verification skipped (selfTest() is embedded in the file)")
        return
    with tempfile.TemporaryDirectory() as td:
        runner = os.path.join(td, "run.js")
        inp = os.path.join(td, "in.json")
        with open(inp, "w") as f:
            json.dump(X.tolist(), f)
        with open(runner, "w") as f:
            f.write("const E=require(process.argv[2]);const xs=JSON.parse(require('fs').readFileSync(process.argv[3],'utf8'));"
                    "const st=E.selfTest();const t0=process.hrtime.bigint();let o=null;"
                    "for(let r=0;r<200;r++)for(const x of xs)o=E.forward(Float32Array.from(x));"
                    "const us=Number(process.hrtime.bigint()-t0)/1e3/(200*xs.length);"
                    "console.log(JSON.stringify({st,us,out:xs.map(x=>Array.from(E.forward(Float32Array.from(x)).logits))}));")
        try:
            res = subprocess.run([node, runner, os.path.abspath(js_path), inp], capture_output=True, text=True, timeout=120)
        except Exception as e:                                        # pragma: no cover
            print(f"  node verification failed to run: {e}")
            return
        if res.returncode != 0:
            print(f"  node verification error:\n{res.stderr}")
            return
        out = json.loads(res.stdout)
        diff = np.abs(np.array(out["out"]) - ref_logits).max()
        print(f"  node JS forward vs torch: max |diff| = {diff:.2e} over {len(X)} samples; "
              f"selfTest ok={out['st']['ok']}; JS latency {out['us']:.1f} us/inference")


# =============================================================================
# 9. Main
# =============================================================================
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-samples", type=int, default=15000, help="total samples (split evenly over 3 classes)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--baseline-rows", type=int, default=8, help="explicit past pulses used to emulate the 40-pulse baseline")
    ap.add_argument("--head", choices=["gap", "flatten"], default="gap", help="classifier head (spec: gap)")
    ap.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument("--plot", action="store_true", help="write echonet_dataset_preview.png (needs matplotlib)")
    ap.add_argument("--keep-misdetections", action="store_true",
                    help="disable rejection sampling: keep crops whose detection missed the target (adds label noise)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    t_start = time.time()
    print("=" * 78)
    print("PITCHBLACK EchoNet synthetic dataset generator")
    print("=" * 78)
    physics_banner()

    print(f"\n[1/4] Generating {args.n_samples} samples (seed {args.seed}, {args.workers} worker(s), "
          f"{args.baseline_rows} baseline rows)")
    X, M = generate_dataset(args.n_samples, args.seed, args.baseline_rows, args.workers, args.keep_misdetections)
    (Xtr, ytr, Mtr), (Xva, yva, Mva) = split_dataset(X, M, args.seed)
    dataset_report(M)
    metadata = dict(
        generated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        classes=CLASSES, window=WIN, peak_index=HALF, fs_hz=FS, f0_hz=F0, f1_hz=F1, chirp_s=T_CHIRP,
        pri_s=PRI, nfft=NFFT, c_mps=C_SOUND, air_temp_c=T_AIR_C, rel_humidity_pct=REL_HUMIDITY,
        alpha_db_per_m={"17500": float(air_absorption_db_per_m(F0)), "20000": float(air_absorption_db_per_m(20e3)),
                        "22000": float(air_absorption_db_per_m(F1))},
        spreading_exponent=SPREADING_EXP, a_ref_fs=A_REF, gate_half_smp=GATE_HALF,
        search_range_m=[R_SEARCH_MIN, R_SEARCH_MAX], baseline_pulses=N_BASELINE_PULSES,
        baseline_rows_simulated=args.baseline_rows, adc_bits=ADC_BITS, seed=args.seed,
        rejection_sampling=not args.keep_misdetections, detect_tol_m=DETECT_TOL_M, max_tries=MAX_TRIES,
        n_train=int(len(Xtr)), n_val=int(len(Xva)), meta_columns=META_COLUMNS,
        normalisation="crop / max(crop)  ->  [0, 1], peak at index 32",
    )
    npz_path = os.path.join(args.out_dir, "echonet_synthetic_dataset.npz")
    np.savez_compressed(npz_path, X_train=Xtr, y_train=ytr, X_val=Xva, y_val=yva,
                        metadata=np.array(json.dumps(metadata)), meta_train=Mtr, meta_val=Mva,
                        meta_columns=np.array(META_COLUMNS))
    print(f"  dataset written: {npz_path}  X_train {Xtr.shape}  X_val {Xva.shape}")
    if args.plot:
        preview_plot(os.path.join(args.out_dir, "echonet_dataset_preview.png"), args.seed, args.baseline_rows)

    if args.skip_train:
        print("\n--skip-train given; done.")
        return 0

    print(f"\n[2/4] Training EchoNetTiny for {args.epochs} epochs")
    try:
        import torch
    except ImportError:
        print("  torch not installed -> cannot train. pip install torch")
        return 1
    model, val_acc, n_params, cm = train_model(Xtr, ytr, Xva, yva, args.epochs, args.head, args.seed)
    pt_path = os.path.join(args.out_dir, "echonet_tiny.pt")
    torch.save(model.state_dict(), pt_path)
    print(f"  state_dict written: {pt_path}")

    print("\n[3/4] Folding BatchNorm and verifying the numpy reference forward pass")
    Wf = fold_weights(model)
    with torch.no_grad():
        ref = model(torch.from_numpy(Xva[:64])[:, None, :]).numpy()
    np_out = np.stack([numpy_forward(Wf, x, args.head) for x in Xva[:64]])
    print(f"  numpy(folded) vs torch logits: max |diff| = {np.abs(np_out - ref).max():.2e}")

    if args.skip_export:
        print("\n--skip-export given; done.")
        return 0

    print("\n[4/4] Exporting")
    export_onnx(model, os.path.join(args.out_dir, "echonet.onnx"))
    js_meta = dict(generated_utc=metadata["generated_utc"], val_accuracy=float(val_acc), n_params=int(n_params),
                   fs_hz=FS, f0_hz=F0, f1_hz=F1, window=WIN, peak_index=HALF, classes=CLASSES,
                   confusion_matrix=cm.tolist())
    js_path = os.path.join(args.out_dir, "echonet_weights.js")
    export_js(Wf, args.head, js_path, js_meta, Xva[0], ref[0])
    verify_js_with_node(js_path, Xva[:16], ref[:16])
    print(f"\nAll done in {time.time() - t_start:.1f} s.")
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
