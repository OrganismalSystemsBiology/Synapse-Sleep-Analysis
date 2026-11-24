#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import json
import shutil
import traceback
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import welch, find_peaks, peak_widths

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from multiprocessing import get_context

# python3 random_search_sync_to_desync_N80_cv0_all.py 1


# ============================================================
# 基本設定
# ============================================================

print("E and I")
SEED = int(sys.argv[1]) #if len(sys.argv) > 1 else 140000
print("seed:", SEED)

# 固定ネットワークサイズ（ランダムサーチしない）
N_E = 64
N_I = 16
N = N_E + N_I

# シミュレーション長
T_ms = 11500.0
dt = 0.1
T_steps = int(round(T_ms / dt))

# 解析窓（CV と CV0 の評価窓）
t_start_ms = 1500.0
t_end_ms = T_ms
cv_w = 50.0
moving = 50.0

# 結合確率
p_EE = 0.20
p_IE = 0.50
p_EI = 0.40
p_II = 0.50

# LIF
EL = -65.0
Vreset = -65.0
tau_m = 20.0
refrac_ms = 2.0
Vth = -55.0

# シナプス
tau_fac_E = 0.0
tau_fac_I = 0.0
tau_s_E = 3.0
tau_s_I = 5.0
Erev_E = 0.0
Erev_I = -70.0

# スケーリング基準
N_E0 = 64
N_I0 = 16

# Uex sweep（昇順）
USE_Es = [0.01, 0.02, 0.05, 0.1, 0.3, 0.5, 0.7]

# release
STOCHASTIC_RELEASE = False
N_RELEASE_SITES = 2
QUANTAL_ALPHA = 1.0

# 出力先
base_fol = "random_search_syncorr3/"
os.makedirs(os.path.join(os.getcwd(), base_fol), exist_ok=True)
OUT_ROOT = Path(base_fol) / f"seed_{SEED}"
CSV_DIR = OUT_ROOT / "csv"
PDF_DIR = OUT_ROOT / "pdf"
HIT_DIR = OUT_ROOT / "hits"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)
PDF_DIR.mkdir(parents=True, exist_ok=True)
HIT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# ランダムサーチ設定
# ============================================================

@dataclass(frozen=True)
class RandomSearchConfig:
    n_samples: int = 100000
    master_seed: int = SEED + 12345
    n_proc: int = 15
    chunksize: int = 1
    start_method: str = "fork"   # Linux なら fork が速い。安全重視なら spawn
    overwrite: bool = False


@dataclass(frozen=True)
class ParamRange:
    g_E_min: float = 0.05
    g_E_max: float = 4.0

    g_I_min: float = 0.05
    g_I_max: float = 4.0

    tau_rec_E_min: float = 200.0
    tau_rec_E_max: float = 5000.0

    tau_rec_I_min: float = 200.0
    tau_rec_I_max: float = 5000.0

    USE_I_min: float = 0.01
    USE_I_max: float = 0.90

    ext_rate_E_min: float = 0.5
    ext_rate_E_max: float = 25.0

    ext_rate_I_min: float = 1.0
    ext_rate_I_max: float = 50.0

    bg_mu_min: float = 0.0002  #0.0002
    bg_mu_max: float = 0.03   #0.01

    bg_std_min: float = 0.0001
    bg_std_max: float = 0.03  #0.005

    max_std_to_mu_ratio: float = 20.0


RS_CFG = RandomSearchConfig(
    n_samples=100000,
    master_seed=SEED + 12345,
    n_proc=12,
    chunksize=1,
    start_method="fork",
    overwrite=False,
)


# ============================================================
# hit 判定設定（CV0 + sync_corr 併用）
# ※ 同期/脱同期判定は全ニューロン（E+I）の
#    CV_all と sync_corr_all の両方で行う
# ============================================================

@dataclass(frozen=True)
class HitCriterion:
    indicator: str = "cv_all"
    rate_indicator: str = "rate_all_hz"
    cv0_indicator: str = "cv0_all"
    ratio_indicator: str = "cv_ratio_all"

    sync_indicator: str = "sync_corr_all"
    sync_hi_hit_indicator: str = "sync_hi_hit_all"
    sync_lo_hit_indicator: str = "sync_lo_hit_all"

    hi_hit_indicator: str = "hi_hit_all"
    lo_hit_indicator: str = "lo_hit_all"

    # 動的閾値: CV0 = 1/sqrt(N * rate_hz * (cv_w_ms/1000))
    sync_factor: float = 1.8
    desync_factor: float = 1.3

    # 固定 sync_corr 閾値
    sync_corr_hi: float = 0.15
    sync_corr_lo: float = 0.05

    decrease_eps: float = 0.01
    min_rate_hz: float = 0.1
    require_monotonic_cv_drop: bool = True


HIT_CFG = HitCriterion()

# sync_corr 計算設定
SYNC_BIN_MS = 5.0
SYNC_BINARY = True
SYNC_ON_POPULATION = "ALL"   # この版では ALL 固定で使う


# ============================================================
# 背景 OU 設定
# ============================================================

@dataclass(frozen=True)
class BGConductanceOUConfig:
    enabled: bool = True
    tau_ge_ms: float = 5.0
    tau_gi_ms: float = 10.0
    rho_shared_ge: float = 0.0
    rho_shared_gi: float = 0.0
    clip_nonnegative: bool = True
    init_from_stationary: bool = True
    target: str = "all"  # "all" / "E" / "I"


BG_OU_CFG = BGConductanceOUConfig()


# ============================================================
# 条件データクラス
# ============================================================

@dataclass(frozen=True)
class Cond:
    sample_id: int
    g_E_base: float
    g_I_base: float
    tau_rec_E: float
    tau_rec_I: float
    USE_I: float
    ext_rate_E: float
    ext_rate_I: float
    bg_mu: float
    bg_std: float


# ============================================================
# Utility
# ============================================================

def _seed_from_cond(cond: Cond, base_seed: int = SEED) -> int:
    def q(x: float, scale: int) -> int:
        return int(round(float(x) * scale))

    vals = [
        int(base_seed),
        int(cond.sample_id),
        q(cond.g_E_base, 10**6),
        q(cond.g_I_base, 10**6),
        q(cond.tau_rec_E, 10**2),
        q(cond.tau_rec_I, 10**2),
        q(cond.USE_I, 10**6),
        q(cond.ext_rate_E, 10**3),
        q(cond.ext_rate_I, 10**3),
        q(cond.bg_mu, 10**7),
        q(cond.bg_std, 10**7),
    ]
    x = 0
    for v in vals:
        x = (x * 1664525 + (v & 0xFFFFFFFF) + 1013904223) & 0xFFFFFFFF
    return int(x)


def sample_random_conditions(cfg: RandomSearchConfig, pr: ParamRange) -> List[Cond]:
    rng = np.random.default_rng(cfg.master_seed)
    conds: List[Cond] = []

    for sample_id in range(cfg.n_samples):
        mu = float(rng.uniform(pr.bg_mu_min, pr.bg_mu_max))
        std_hi = min(pr.bg_std_max, pr.max_std_to_mu_ratio * mu)
        std_lo = min(pr.bg_std_min, std_hi)

        if std_hi <= 0:
            std = 0.0
        elif std_hi <= std_lo:
            std = float(std_hi)
        else:
            std = float(rng.uniform(std_lo, std_hi))

        conds.append(
            Cond(
                sample_id=sample_id,
                g_E_base=float(rng.uniform(pr.g_E_min, pr.g_E_max)),
                g_I_base=float(rng.uniform(pr.g_I_min, pr.g_I_max)),
                tau_rec_E=float(rng.uniform(pr.tau_rec_E_min, pr.tau_rec_E_max)),
                tau_rec_I=float(rng.uniform(pr.tau_rec_I_min, pr.tau_rec_I_max)),
                USE_I=float(rng.uniform(pr.USE_I_min, pr.USE_I_max)),
                ext_rate_E=float(rng.uniform(pr.ext_rate_E_min, pr.ext_rate_E_max)),
                ext_rate_I=float(rng.uniform(pr.ext_rate_I_min, pr.ext_rate_I_max)),
                bg_mu=mu,
                bg_std=std,
            )
        )

    return conds


def _sample_prefix(cond: Cond) -> str:
    return (
        f"sample{cond.sample_id:06d}"
        f"_gE{cond.g_E_base:.4f}"
        f"_gI{cond.g_I_base:.4f}"
        f"_tauE{cond.tau_rec_E:.1f}"
        f"_tauI{cond.tau_rec_I:.1f}"
        f"_UI{cond.USE_I:.3f}"
        f"_Er{cond.ext_rate_E:.3f}"
        f"_Ir{cond.ext_rate_I:.3f}"
        f"_mu{cond.bg_mu:.5f}"
        f"_std{cond.bg_std:.5f}"
    )


def _bg_ou_file_tag(cfg: BGConductanceOUConfig) -> str:
    if not cfg.enabled:
        return "bgOUoff"
    return f"bgOU_tau{cfg.tau_ge_ms:.1f}-{cfg.tau_gi_ms:.1f}"


def _make_raster_fig(spike_times, is_exc, title: str, tlim_ms=None):
    fig, ax = plt.subplots(figsize=(10, 4))

    if tlim_ms is not None:
        t0, t1 = float(tlim_ms[0]), float(tlim_ms[1])
    else:
        t0, t1 = None, None

    row = 0
    for n in range(len(spike_times)):
        st = np.asarray(spike_times[n], dtype=float)
        if st.size > 0 and t0 is not None:
            st = st[(st >= t0) & (st <= t1)]
        if st.size > 0:
            color = "tab:blue" if is_exc[n] else "tab:orange"
            ax.vlines(st, row, row + 0.5, color=color, lw=1.0, rasterized=True)
        row += 1

    ax.set_title(title)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Neuron index")
    if tlim_ms is not None:
        ax.set_xlim((t0, t1))
    fig.tight_layout()
    return fig


def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if win <= 1 or x.size == 0:
        return x.copy()
    k = np.ones(int(win), dtype=float) / float(win)
    return np.convolve(x, k, mode="same")


# ============================================================
# 解析関数
# ============================================================

def spike_times_to_binary(spike_times, T_ms=3000.0, dt_ms=0.1, dtype=np.uint8):
    Nn = len(spike_times)
    n_bins = int(np.floor(T_ms / dt_ms))
    out = np.zeros((Nn, n_bins), dtype=dtype)

    for i, times in enumerate(spike_times):
        if times is None or len(times) == 0:
            continue
        t = np.asarray(times, dtype=float)
        mask = (t >= 0.0) & (t < T_ms)
        if not np.any(mask):
            continue
        idx = np.rint(t[mask] / dt_ms).astype(int)
        idx = np.clip(idx, 0, n_bins - 1)
        idx = np.unique(idx)
        out[i, idx] = 1
    return out


def evaluate_population_cv_from_spts(
    spts: np.ndarray,
    *,
    N_sel: int,
    dt_ms: float,
    t_start_ms: float,
    t_end_ms: float,
    cv_w_ms: float,
    moving_ms: float,
) -> Tuple[float, float]:
    """
    集団の「窓内総スパイク数CV」と 1ニューロンあたり平均発火率[Hz] を返す。
    """
    spts = np.asarray(spts)
    if spts.ndim != 2:
        raise ValueError("spts must be 2D (N,T)")
    if N_sel <= 0 or N_sel > spts.shape[0]:
        return np.nan, 0.0

    start = int(round(t_start_ms / dt_ms))
    end = int(round(t_end_ms / dt_ms))
    start = max(0, min(spts.shape[1], start))
    end = max(0, min(spts.shape[1], end))
    if end <= start:
        return np.nan, 0.0

    pop = spts[:N_sel, start:end].sum(axis=0).astype(np.float64)
    dur_s = (end - start) * dt_ms * 1e-3
    if dur_s <= 0.0:
        return np.nan, 0.0

    total_spikes = float(pop.sum())
    rate_hz = total_spikes / (dur_s * float(N_sel))

    win = int(round(cv_w_ms / dt_ms))
    step = int(round(moving_ms / dt_ms))
    if win <= 0 or step <= 0 or pop.size < win:
        return np.nan, rate_hz

    starts = np.arange(0, pop.size - win + 1, step, dtype=int)
    if starts.size < 2:
        return np.nan, rate_hz

    csum = np.concatenate(([0.0], np.cumsum(pop)))
    win_sums = csum[starts + win] - csum[starts]
    m = float(win_sums.mean())
    if m <= 0.0:
        cv = 0.0
    else:
        cv = float(win_sums.std(ddof=0) / m)
    return cv, float(rate_hz)


def cv0_from_rate(rate_hz: float, N_sel: int, cv_w_ms: float) -> float:
    """
    非同期独立ポアソン近似:
    CV0 = 1 / sqrt(lambda)
    lambda = N_sel * rate_hz * (cv_w_ms / 1000)
    """
    lam = float(N_sel) * float(rate_hz) * (float(cv_w_ms) / 1000.0)
    if not np.isfinite(lam) or lam <= 0.0:
        return np.nan
    return float(1.0 / np.sqrt(lam))


def build_binned_spike_matrix(
    spk_times,
    neuron_indices,
    t_start_ms,
    t_end_ms,
    bin_ms=5.0,
    binary=True,
):
    """
    指定した neuron 群のスパイク列を bin 化して行列化する
    rows   : neurons
    cols   : time bins
    values : binary=True なら 0/1, False なら spike count
    """
    neuron_indices = np.asarray(neuron_indices, dtype=int)

    if t_end_ms <= t_start_ms or bin_ms <= 0:
        return np.empty((0, 0), dtype=float), np.empty(0), np.empty(0)

    edges = np.arange(t_start_ms, t_end_ms + bin_ms, bin_ms, dtype=float)
    if edges.size < 2:
        edges = np.array([t_start_ms, t_end_ms], dtype=float)
    if edges[-1] < t_end_ms:
        edges = np.append(edges, t_end_ms)

    nbins = len(edges) - 1
    X = np.zeros((len(neuron_indices), nbins), dtype=float)

    for r, idx in enumerate(neuron_indices):
        ts = np.asarray(spk_times[idx], dtype=float)
        if ts.size == 0:
            continue
        ts = ts[(ts >= t_start_ms) & (ts < t_end_ms)]
        if ts.size == 0:
            continue

        counts, _ = np.histogram(ts, bins=edges)
        if binary:
            X[r, :] = (counts > 0).astype(float)
        else:
            X[r, :] = counts.astype(float)

    centers = 0.5 * (edges[:-1] + edges[1:])
    return X, edges, centers


def mean_pairwise_zero_lag_corr(X):
    """
    行列 X (neurons x bins) について、
    ニューロン間のゼロラグ Pearson 相関の平均を返す
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 2:
        return np.nan, 0

    keep = np.std(X, axis=1) > 0
    Xk = X[keep]

    if Xk.shape[0] < 2:
        return np.nan, int(Xk.shape[0])

    C = np.corrcoef(Xk)
    iu = np.triu_indices(C.shape[0], k=1)
    if len(iu[0]) == 0:
        return np.nan, int(Xk.shape[0])

    return float(np.nanmean(C[iu])), int(Xk.shape[0])


def detect_bursts(
    spikes,
    dt_ms=0.1,
    window_ms=10.0,
    moving_ms=5.0,
    threshold=80,
    merge_adjacent=True,
    max_duration_ms=250.0,
):
    spikes = np.asarray(spikes, dtype=np.uint8)
    if spikes.ndim != 2:
        raise ValueError("spikes must be 2D (N,T).")

    _, Tn = spikes.shape
    win_bins = int(round(window_ms / dt_ms))
    step_bins = int(round(moving_ms / dt_ms))
    if win_bins <= 0 or step_bins <= 0:
        raise ValueError("window_ms and moving_ms must be > 0.")

    pop = spikes.sum(axis=0)
    raw = []
    for start in range(0, Tn - win_bins + 1, step_bins):
        end = start + win_bins
        if int(pop[start:end].sum()) > int(threshold):
            raw.append([start * dt_ms, end * dt_ms])

    if not raw:
        return []

    merged = [raw[0][:]]
    for s_ms, e_ms in raw[1:]:
        ps_ms, pe_ms = merged[-1]
        overlap = (s_ms <= pe_ms) if merge_adjacent else (s_ms < pe_ms)
        if overlap:
            merged[-1][1] = max(pe_ms, e_ms)
        else:
            merged.append([s_ms, e_ms])

    if max_duration_ms is not None:
        merged = [b for b in merged if (b[1] - b[0]) <= max_duration_ms]

    return merged


def burst_dur_freq(bursts, spts, dt_ms):
    if bursts is None or len(bursts) == 0:
        return 0.0, 0.0

    burst_durs = []
    burst_freqs = []

    for burst in bursts:
        start_ms, end_ms = float(burst[0]), float(burst[1])
        start = int(round(start_ms / dt_ms))
        end = int(round(end_ms / dt_ms))
        if end <= start:
            continue

        burst_durs.append(end_ms - start_ms)
        burst_sps = np.sum(spts[:, start:end], axis=1)
        burst_N = int(np.sum(burst_sps > 0))
        if burst_N > 0:
            dur_s = (end_ms - start_ms) * 1e-3
            burst_freqs.append(float(np.sum(burst_sps) / dur_s / burst_N))

    if len(burst_durs) == 0:
        return 0.0, 0.0

    m_burst_dur = float(np.mean(burst_durs))
    m_burst_freq = float(np.mean(burst_freqs)) if len(burst_freqs) else 0.0
    return m_burst_dur, m_burst_freq


def _as_neuron_time(spts, Nn):
    spts = np.asarray(spts)
    if spts.ndim != 2:
        raise ValueError(f"spts must be 2D. got shape={spts.shape}")

    if spts.shape[0] == Nn:
        return spts
    if spts.shape[1] == Nn:
        return spts.T
    raise ValueError(f"spts shape {spts.shape} doesn't match N={Nn} on any axis.")


def pop_total_rate_and_max_for_group(
    spts, dt_ms, group_mask,
    window_ms=100.0, step_ms=1.0,
    t_start_ms=0.0, t_end_ms=None
):
    group_mask = np.asarray(group_mask, dtype=bool)
    Nn = group_mask.size
    n_group = int(group_mask.sum())
    if n_group == 0:
        raise ValueError("group_mask has no True entries.")

    spts = _as_neuron_time(spts, Nn)
    _, Tn = spts.shape

    start_idx = int(round(t_start_ms / dt_ms))
    end_idx = int(round(t_end_ms / dt_ms)) if t_end_ms is not None else Tn
    start_idx = max(0, min(Tn, start_idx))
    end_idx = max(0, min(Tn, end_idx))
    if end_idx <= start_idx:
        raise ValueError("t_end_ms must be > t_start_ms.")

    w = int(round(window_ms / dt_ms))
    step = int(round(step_ms / dt_ms))
    if w <= 0 or step <= 0:
        raise ValueError("window_ms/step_ms too small.")

    pop_spikes = spts[group_mask, start_idx:end_idx].sum(axis=0).astype(np.int64)
    L = pop_spikes.size
    if L < w:
        raise ValueError(f"segment too short: L={L}, window={w}")

    cs = np.concatenate(([0], np.cumsum(pop_spikes, dtype=np.int64)))
    counts = cs[w:] - cs[:-w]
    counts = counts[::step]
    k0 = np.arange(counts.size, dtype=np.int64) * step

    t_center_ms = (start_idx + k0 + 0.5 * w) * dt_ms
    rate_total = counts / (window_ms * 1e-3)

    i_peak = int(np.argmax(rate_total))
    max_total = float(rate_total[i_peak])
    t_peak_ms = float(t_center_ms[i_peak])

    return t_center_ms, rate_total, max_total, t_peak_ms, n_group


def population_rate_hz_from_spts(
    spts: np.ndarray,
    *,
    Nn: int,
    dt_ms: float,
    bin_ms: float = 5.0,
    smooth_ms: float = 0.0,
    neuron_mask: np.ndarray,
):
    sptsNT = _as_neuron_time(spts, Nn)
    neuron_mask = np.asarray(neuron_mask, dtype=bool)
    if neuron_mask.shape != (Nn,):
        raise ValueError(f"neuron_mask must have shape ({Nn},), got {neuron_mask.shape}")

    X = sptsNT[neuron_mask].astype(np.float32)
    n_sel = X.shape[0]
    if n_sel == 0:
        raise ValueError("neuron_mask selects 0 neurons")

    bin_steps = max(int(round(bin_ms / dt_ms)), 1)
    T_steps_local = X.shape[1]
    T_trim = (T_steps_local // bin_steps) * bin_steps
    if T_trim < bin_steps * 10:
        raise ValueError("Too short data for given bin_ms.")

    X = X[:, :T_trim]
    nbins = T_trim // bin_steps
    Xb = X.reshape(n_sel, nbins, bin_steps)
    spikes_per_bin = Xb.sum(axis=(0, 2))

    bin_s = (bin_steps * dt_ms) * 1e-3
    rate_hz = spikes_per_bin / n_sel / bin_s
    t_s = (np.arange(nbins) + 0.5) * bin_s

    if smooth_ms and smooth_ms > 0.0:
        smooth_bins = max(int(round(smooth_ms / bin_ms)), 1)
        if smooth_bins > 1:
            k = np.ones(smooth_bins, dtype=np.float32) / smooth_bins
            rate_hz = np.convolve(rate_hz, k, mode="same")

    return t_s, rate_hz


def estimate_freq_from_time_peaks(
    t: np.ndarray,
    f: np.ndarray,
    *,
    fmin: float,
    fmax: float,
    smooth_ms: float = 50.0,
) -> Tuple[float, Dict[str, Any]]:
    t = np.asarray(t, dtype=float)
    f = np.asarray(f, dtype=float)
    info: Dict[str, Any] = {"method": "time_peaks", "n_peaks": 0}

    if t.size < 10 or f.size != t.size:
        return np.nan, info

    dt_s = float(np.median(np.diff(t)))
    if dt_s <= 0.0:
        return np.nan, info

    smooth_bins = max(1, int(round((smooth_ms * 1e-3) / dt_s)))
    fs = _moving_average(f, smooth_bins)

    amp = float(np.percentile(fs, 95) - np.percentile(fs, 5))
    prom = max(1e-6, 0.10 * amp, 0.20 * float(np.std(fs)))
    min_dist = max(1, int(np.floor((1.0 / max(fmax, 1e-6)) / dt_s * 0.5)))

    peaks, props = find_peaks(fs, distance=min_dist, prominence=prom)
    info["n_peaks"] = int(peaks.size)

    if peaks.size < 2:
        return np.nan, info

    periods = np.diff(t[peaks])
    per_min = 1.0 / max(fmax, 1e-6)
    per_max = 1.0 / max(fmin, 1e-6)
    periods = periods[(periods >= 0.5 * per_min) & (periods <= 1.5 * per_max)]

    if periods.size == 0:
        return np.nan, info

    freq = float(1.0 / np.median(periods))
    info["periods"] = periods.tolist()
    info["prominence_mean"] = (
        float(np.mean(props["prominences"]))
        if "prominences" in props and len(props["prominences"])
        else 0.0
    )
    return freq, info


def estimate_freq_from_autocorr(
    t: np.ndarray,
    f: np.ndarray,
    *,
    fmin: float,
    fmax: float,
    smooth_ms: float = 50.0,
) -> Tuple[float, Dict[str, Any]]:
    t = np.asarray(t, dtype=float)
    f = np.asarray(f, dtype=float)
    info: Dict[str, Any] = {"method": "acf", "acf_peak": 0.0}

    if t.size < 10 or f.size != t.size:
        return np.nan, info

    dt_s = float(np.median(np.diff(t)))
    if dt_s <= 0.0:
        return np.nan, info

    smooth_bins = max(1, int(round((smooth_ms * 1e-3) / dt_s)))
    x = _moving_average(f, smooth_bins)
    x = x - float(np.mean(x))
    sx = float(np.std(x))
    if sx <= 1e-12:
        return np.nan, info
    x = x / sx

    acf = np.correlate(x, x, mode="full")
    acf = acf[acf.size // 2:]
    if acf[0] <= 0.0:
        return np.nan, info
    acf = acf / acf[0]

    lag_min = max(1, int(np.floor((1.0 / max(fmax, 1e-6)) / dt_s)))
    lag_max = min(len(acf) - 1, int(np.ceil((1.0 / max(fmin, 1e-6)) / dt_s)))
    if lag_max <= lag_min + 1:
        return np.nan, info

    seg = acf[lag_min:lag_max + 1]
    if seg.size < 3:
        return np.nan, info

    prom = max(0.02, 0.10 * float(np.std(seg)))
    peaks, _ = find_peaks(seg, prominence=prom)

    if peaks.size > 0:
        k = int(peaks[np.argmax(seg[peaks])])
    else:
        k = int(np.argmax(seg))

    lag = lag_min + k
    if lag <= 0:
        return np.nan, info

    freq = float(1.0 / (lag * dt_s))
    info["acf_peak"] = float(seg[k])
    return freq, info


def estimate_freq_from_psd(
    t: np.ndarray,
    f: np.ndarray,
    *,
    fmin: float,
    fmax: float,
) -> Tuple[float, Dict[str, Any]]:
    t = np.asarray(t, dtype=float)
    f = np.asarray(f, dtype=float)
    info: Dict[str, Any] = {"method": "psd", "snr_db": np.nan, "Q": np.nan}

    if t.size < 16 or f.size != t.size:
        return np.nan, info

    dt_s = float(np.median(np.diff(t)))
    if dt_s <= 0.0:
        return np.nan, info

    x = f - float(np.mean(f))
    if float(np.std(x)) <= 1e-12:
        return np.nan, info

    fs = 1.0 / dt_s
    n = len(x)

    nper = n
    nfft_target = max(8 * n, int(np.ceil(8.0 * fs / max(fmin, 1e-6))))
    nfft = 1
    while nfft < nfft_target:
        nfft *= 2

    f_psd, Pxx = welch(
        x,
        fs=fs,
        window="hann",
        nperseg=nper,
        noverlap=nper // 2,
        detrend="constant",
        scaling="density",
        nfft=nfft,
    )

    band = (f_psd >= fmin) & (f_psd <= fmax)
    if not np.any(band):
        return np.nan, info

    f_b = f_psd[band]
    P_b = Pxx[band]
    if P_b.size < 3:
        return np.nan, info

    dom = int(np.argmax(P_b))
    freq = float(f_b[dom])

    rest = np.delete(P_b, dom)
    med_noise = float(np.median(rest)) if rest.size else 1e-20
    med_noise = max(med_noise, 1e-20)
    snr_db = 10.0 * np.log10(P_b[dom] / med_noise)

    try:
        if 0 < dom < (len(P_b) - 1):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                widths, _, _, _ = peak_widths(P_b, [dom], rel_height=0.5)
            df_psd = float(f_b[1] - f_b[0]) if f_b.size > 1 else np.inf
            fwhm = float(widths[0] * df_psd) if widths.size > 0 else np.inf
            Q = freq / max(fwhm, 1e-12)
        else:
            Q = np.nan
    except Exception:
        Q = np.nan

    info["snr_db"] = float(snr_db)
    info["Q"] = float(Q) if np.isfinite(Q) else np.nan
    return freq, info


def detect_oscillation_and_freq(t, f, cfg):
    """
    周波数推定はできるだけ返す。
    ただし hit 判定には使わない。
    """
    t = np.asarray(t, dtype=float)
    f = np.asarray(f, dtype=float)

    if t.size < 16 or f.size != t.size:
        return False, np.nan, {"method": "none"}

    last_seconds = cfg.get("last_seconds", None)
    if last_seconds is not None and np.isfinite(last_seconds) and last_seconds > 0:
        t0 = max(0.0, float(t[-1]) - float(last_seconds))
        mask = t >= t0
        tt = t[mask]
        ff = f[mask]
    else:
        tt = t
        ff = f

    if ff.size < 16:
        return False, np.nan, {"method": "none"}

    smooth_ms = float(cfg.get("smooth_ms", 50.0))
    dt_s = float(np.median(np.diff(tt)))
    smooth_bins = max(1, int(round((smooth_ms * 1e-3) / max(dt_s, 1e-12))))
    ff_s = _moving_average(ff, smooth_bins)

    amp_rob = float(np.percentile(ff_s, 95) - np.percentile(ff_s, 5))
    mean_rate = float(np.mean(ff_s))
    rel_amp = amp_rob / max(mean_rate, 1e-6)

    fmin = float(cfg["fmin"])
    fmax = float(cfg["fmax"])

    f_time, info_time = estimate_freq_from_time_peaks(
        tt, ff_s, fmin=fmin, fmax=fmax, smooth_ms=smooth_ms
    )
    f_acf, info_acf = estimate_freq_from_autocorr(
        tt, ff_s, fmin=fmin, fmax=fmax, smooth_ms=smooth_ms
    )
    f_psd, info_psd = estimate_freq_from_psd(
        tt, ff_s, fmin=fmin, fmax=fmax
    )

    freq = np.nan
    method = "none"

    if np.isfinite(f_time):
        freq = float(f_time)
        method = "time_peaks"
    elif np.isfinite(f_acf):
        freq = float(f_acf)
        method = "acf"
    elif np.isfinite(f_psd):
        freq = float(f_psd)
        method = "psd"

    snr_db = float(info_psd.get("snr_db", np.nan))
    q_val = float(info_psd.get("Q", np.nan))
    n_peaks = int(info_time.get("n_peaks", 0))
    acf_peak = float(info_acf.get("acf_peak", 0.0))

    is_osc = (
        np.isfinite(freq)
        and (fmin <= freq <= fmax)
        and (amp_rob >= float(cfg["amp_thr_abs"]))
        and (rel_amp >= float(cfg["amp_thr_rel"]))
        and (
            (np.isfinite(snr_db) and snr_db >= float(cfg["snr_db_thr"]))
            or (acf_peak >= float(cfg.get("acf_peak_thr", 0.05)))
            or (n_peaks >= int(cfg["min_peaks"]))
        )
        and ((not np.isfinite(q_val)) or (q_val >= float(cfg["q_min"])))
    )

    info = {
        "method": method,
        "freq_time": float(f_time) if np.isfinite(f_time) else np.nan,
        "freq_acf": float(f_acf) if np.isfinite(f_acf) else np.nan,
        "freq_psd": float(f_psd) if np.isfinite(f_psd) else np.nan,
        "n_peaks": n_peaks,
        "acf_peak": acf_peak,
        "snr_db": snr_db,
        "Q": q_val,
        "amp_rob": amp_rob,
        "rel_amp": rel_amp,
    }
    return bool(is_osc), (float(freq) if np.isfinite(freq) else np.nan), info


def get_detection_config(profile: str):
    if profile == "very_strict":
        return dict(
            last_seconds=12.0,
            min_peaks=10,
            amp_thr_abs=4.0,
            amp_thr_rel=0.35,
            snr_db_thr=15.0,
            cv_p2p_thr=0.25,
            fmin=1.0, fmax=80.0,
            q_min=3.0,
            acf_peak_thr=0.20,
            smooth_ms=50.0,
        )
    if profile == "strict":
        return dict(
            last_seconds=10.0,
            min_peaks=3,
            amp_thr_abs=0.5,
            amp_thr_rel=0.10,
            snr_db_thr=5.0,
            cv_p2p_thr=0.10,
            fmin=0.005, fmax=20.0,
            q_min=1.0,
            acf_peak_thr=0.10,
            smooth_ms=50.0,
        )
    if profile == "loose":
        return dict(
            last_seconds=10.0,
            min_peaks=2,
            amp_thr_abs=0.25,
            amp_thr_rel=0.04,
            snr_db_thr=-2.0,
            cv_p2p_thr=0.60,
            fmin=0.1, fmax=80.0,
            q_min=0.4,
            acf_peak_thr=0.05,
            smooth_ms=50.0,
        )
    if profile == "very_loose":
        return dict(
            last_seconds=10.0,
            min_peaks=2,
            amp_thr_abs=0.10,
            amp_thr_rel=0.02,
            snr_db_thr=-8.0,
            cv_p2p_thr=1.20,
            fmin=0.1, fmax=80.0,
            q_min=0.0,
            acf_peak_thr=0.03,
            smooth_ms=50.0,
        )
    return dict(
        last_seconds=8.0,
        min_peaks=6,
        amp_thr_abs=1.5,
        amp_thr_rel=0.15,
        snr_db_thr=6.0,
        cv_p2p_thr=0.45,
        fmin=0.5, fmax=80.0,
        q_min=2.0,
        acf_peak_thr=0.08,
        smooth_ms=50.0,
    )


# ============================================================
# hit 判定（CV0 + sync_corr）
# ※ Oscillation frequency は使わない
# ============================================================

def find_decreasing_segment_with_dynamic_thresholds(
    U_list: List[float],
    cv_list: List[float],
    valid_list: List[bool],
    hi_hit_list: List[bool],
    lo_hit_list: List[bool],
    decrease_eps: float = 0.01,
) -> Optional[Dict[str, Any]]:
    """
    条件:
    - 連続した U インデックス区間 [i..j], j>=i+1
    - cv が単調減少: cv[k+1] <= cv[k] - decrease_eps
    - 区間内に hi_hit=True が少なくとも1つ
    - 区間内に lo_hit=True が少なくとも1つ
    - 区間内の点は valid=True
    """
    n = len(U_list)
    best = None
    best_drop = -np.inf

    for i in range(n - 1):
        if not valid_list[i]:
            continue
        j = i
        while j + 1 < n:
            if not valid_list[j + 1]:
                break
            if not (cv_list[j + 1] <= cv_list[j] - decrease_eps):
                break
            j += 1

            if j - i + 1 >= 2:
                seg_hi = hi_hit_list[i:j + 1]
                seg_lo = lo_hit_list[i:j + 1]
                if any(seg_hi) and any(seg_lo):
                    drop = float(cv_list[i] - cv_list[j])
                    if drop > best_drop:
                        rel_hi = [k for k, x in enumerate(seg_hi) if x]
                        rel_lo = [k for k, x in enumerate(seg_lo) if x]
                        sleep_rel = rel_hi[0]

                        wake_rel = None
                        for rr in rel_lo:
                            if rr > sleep_rel:
                                wake_rel = rr
                                break
                        if wake_rel is None:
                            wake_rel = rel_lo[-1]

                        best_drop = drop
                        best = {
                            "start_idx": i,
                            "end_idx": j,
                            "sleep_idx": i + sleep_rel,
                            "wake_idx": i + wake_rel,
                            "drop": drop,
                            "max_cv": float(np.nanmax(cv_list[i:j + 1])),
                            "min_cv": float(np.nanmin(cv_list[i:j + 1])),
                        }
    return best


def find_sync_to_desync_transition(
    cc_df: pd.DataFrame,
    hit_cfg: HitCriterion,
) -> Dict[str, Any]:
    """
    CV_all と sync_corr_all の両方で同期→脱同期遷移を判定する。
    osc, osc_freq は判定に使わない。
    """
    cv_vals = cc_df[hit_cfg.indicator].to_numpy(dtype=float).tolist()

    valid = (
        cc_df[hit_cfg.indicator].replace([np.inf, -np.inf], np.nan).notna()
        & cc_df[hit_cfg.cv0_indicator].replace([np.inf, -np.inf], np.nan).notna()
        & cc_df[hit_cfg.sync_indicator].replace([np.inf, -np.inf], np.nan).notna()
        & (cc_df[hit_cfg.rate_indicator].to_numpy(dtype=float) >= hit_cfg.min_rate_hz)
    ).tolist()

    hi_hit = cc_df[hit_cfg.hi_hit_indicator].astype(bool).tolist()
    lo_hit = cc_df[hit_cfg.lo_hit_indicator].astype(bool).tolist()

    seg = find_decreasing_segment_with_dynamic_thresholds(
        U_list=cc_df["Uex"].to_numpy(dtype=float).tolist(),
        cv_list=cv_vals,
        valid_list=valid,
        hi_hit_list=hi_hit,
        lo_hit_list=lo_hit,
        decrease_eps=hit_cfg.decrease_eps,
    )

    out = {
        "hit": False,
        "sleep_idx": -1,
        "wake_idx": -1,
        "sleep_Uex": np.nan,
        "wake_Uex": np.nan,
        "start_idx": -1,
        "end_idx": -1,
        "drop": np.nan,
        "max_cv": np.nan,
        "min_cv": np.nan,
    }
    if seg is None:
        return out

    out.update(
        hit=True,
        sleep_idx=int(seg["sleep_idx"]),
        wake_idx=int(seg["wake_idx"]),
        sleep_Uex=float(cc_df.at[int(seg["sleep_idx"]), "Uex"]),
        wake_Uex=float(cc_df.at[int(seg["wake_idx"]), "Uex"]),
        start_idx=int(seg["start_idx"]),
        end_idx=int(seg["end_idx"]),
        drop=float(seg["drop"]),
        max_cv=float(seg["max_cv"]),
        min_cv=float(seg["min_cv"]),
    )
    return out


def summarize_hit_row(
    cond: Cond,
    cc_df: pd.DataFrame,
    trans: Dict[str, Any],
) -> Dict[str, Any]:
    sleep_idx = int(trans["sleep_idx"])
    wake_idx = int(trans["wake_idx"])

    base = {
        "sample_id": cond.sample_id,
        "gE": cond.g_E_base,
        "gI": cond.g_I_base,
        "tau_rec_E": cond.tau_rec_E,
        "tau_rec_I": cond.tau_rec_I,
        "UI": cond.USE_I,
        "Erate": cond.ext_rate_E,
        "Irate": cond.ext_rate_I,
        "bg_mu": cond.bg_mu,
        "bg_std": cond.bg_std,
        "seg_start_idx": int(trans.get("start_idx", -1)),
        "seg_end_idx": int(trans.get("end_idx", -1)),
        "cv_drop": float(trans.get("drop", np.nan)) if np.isfinite(trans.get("drop", np.nan)) else np.nan,
    }

    if sleep_idx < 0 or wake_idx < 0:
        base.update({
            "wake_Uex": np.nan,
            "sleep_Uex": np.nan,
            "wake_osc_freq": np.nan,
            "sleep_osc_freq": np.nan,

            "wake_mean_freq": np.nan,
            "sleep_mean_freq": np.nan,
            "wake_mean_freq_ex": np.nan,
            "sleep_mean_freq_ex": np.nan,

            "wake_rate_all_hz": np.nan,
            "sleep_rate_all_hz": np.nan,
            "wake_cv0_all": np.nan,
            "sleep_cv0_all": np.nan,
            "wake_cv_ratio_all": np.nan,
            "sleep_cv_ratio_all": np.nan,
            "wake_hi_thr_all": np.nan,
            "sleep_hi_thr_all": np.nan,
            "wake_lo_thr_all": np.nan,
            "sleep_lo_thr_all": np.nan,
            "wake_sync_corr_all": np.nan,
            "sleep_sync_corr_all": np.nan,

            "wake_max_freq": np.nan,
            "sleep_max_freq": np.nan,
            "wake_max_freq_ex": np.nan,
            "sleep_max_freq_ex": np.nan,

            "wake_m_burst_dur": np.nan,
            "sleep_m_burst_dur": np.nan,
            "wake_m_burst_dur_ex": np.nan,
            "sleep_m_burst_dur_ex": np.nan,

            "wake_m_burst_freq": np.nan,
            "sleep_m_burst_freq": np.nan,
            "wake_m_burst_freq_ex": np.nan,
            "sleep_m_burst_freq_ex": np.nan,

            "wake_cv_all": np.nan,
            "sleep_cv_all": np.nan,
            "bifur_flag": 0,
        })
        return base

    base.update({
        "wake_Uex": float(cc_df.at[wake_idx, "Uex"]),
        "sleep_Uex": float(cc_df.at[sleep_idx, "Uex"]),
        "wake_osc_freq": float(cc_df.at[wake_idx, "osc_freq"]) if np.isfinite(cc_df.at[wake_idx, "osc_freq"]) else np.nan,
        "sleep_osc_freq": float(cc_df.at[sleep_idx, "osc_freq"]) if np.isfinite(cc_df.at[sleep_idx, "osc_freq"]) else np.nan,

        "wake_mean_freq": float(cc_df.at[wake_idx, "mean_freq"]),
        "sleep_mean_freq": float(cc_df.at[sleep_idx, "mean_freq"]),
        "wake_mean_freq_ex": float(cc_df.at[wake_idx, "mean_freq_ex"]),
        "sleep_mean_freq_ex": float(cc_df.at[sleep_idx, "mean_freq_ex"]),

        "wake_rate_all_hz": float(cc_df.at[wake_idx, "rate_all_hz"]),
        "sleep_rate_all_hz": float(cc_df.at[sleep_idx, "rate_all_hz"]),
        "wake_cv0_all": float(cc_df.at[wake_idx, "cv0_all"]) if np.isfinite(cc_df.at[wake_idx, "cv0_all"]) else np.nan,
        "sleep_cv0_all": float(cc_df.at[sleep_idx, "cv0_all"]) if np.isfinite(cc_df.at[sleep_idx, "cv0_all"]) else np.nan,
        "wake_cv_ratio_all": float(cc_df.at[wake_idx, "cv_ratio_all"]) if np.isfinite(cc_df.at[wake_idx, "cv_ratio_all"]) else np.nan,
        "sleep_cv_ratio_all": float(cc_df.at[sleep_idx, "cv_ratio_all"]) if np.isfinite(cc_df.at[sleep_idx, "cv_ratio_all"]) else np.nan,
        "wake_hi_thr_all": float(cc_df.at[wake_idx, "cv_hi_thr_all"]) if np.isfinite(cc_df.at[wake_idx, "cv_hi_thr_all"]) else np.nan,
        "sleep_hi_thr_all": float(cc_df.at[sleep_idx, "cv_hi_thr_all"]) if np.isfinite(cc_df.at[sleep_idx, "cv_hi_thr_all"]) else np.nan,
        "wake_lo_thr_all": float(cc_df.at[wake_idx, "cv_lo_thr_all"]) if np.isfinite(cc_df.at[wake_idx, "cv_lo_thr_all"]) else np.nan,
        "sleep_lo_thr_all": float(cc_df.at[sleep_idx, "cv_lo_thr_all"]) if np.isfinite(cc_df.at[sleep_idx, "cv_lo_thr_all"]) else np.nan,
        "wake_sync_corr_all": float(cc_df.at[wake_idx, "sync_corr_all"]) if np.isfinite(cc_df.at[wake_idx, "sync_corr_all"]) else np.nan,
        "sleep_sync_corr_all": float(cc_df.at[sleep_idx, "sync_corr_all"]) if np.isfinite(cc_df.at[sleep_idx, "sync_corr_all"]) else np.nan,

        "wake_max_freq": float(cc_df.at[wake_idx, "max_freq/100ms"]),
        "sleep_max_freq": float(cc_df.at[sleep_idx, "max_freq/100ms"]),
        "wake_max_freq_ex": float(cc_df.at[wake_idx, "max_freq_ex/100ms"]),
        "sleep_max_freq_ex": float(cc_df.at[sleep_idx, "max_freq_ex/100ms"]),

        "wake_m_burst_dur": float(cc_df.at[wake_idx, "mean_burst_duration"]),
        "sleep_m_burst_dur": float(cc_df.at[sleep_idx, "mean_burst_duration"]),
        "wake_m_burst_dur_ex": float(cc_df.at[wake_idx, "mean_burst_duration_ex"]),
        "sleep_m_burst_dur_ex": float(cc_df.at[sleep_idx, "mean_burst_duration_ex"]),

        "wake_m_burst_freq": float(cc_df.at[wake_idx, "mean_burst_freq"]),
        "sleep_m_burst_freq": float(cc_df.at[sleep_idx, "mean_burst_freq"]),
        "wake_m_burst_freq_ex": float(cc_df.at[wake_idx, "mean_burst_freq_ex"]),
        "sleep_m_burst_freq_ex": float(cc_df.at[sleep_idx, "mean_burst_freq_ex"]),

        "wake_cv_all": float(cc_df.at[wake_idx, "cv_all"]) if np.isfinite(cc_df.at[wake_idx, "cv_all"]) else np.nan,
        "sleep_cv_all": float(cc_df.at[sleep_idx, "cv_all"]) if np.isfinite(cc_df.at[sleep_idx, "cv_all"]) else np.nan,
        "bifur_flag": 1,
    })
    return base


# ============================================================
# 1 条件の実行
# ============================================================

def run_one_condition(cond: Cond):
    prefix = _sample_prefix(cond)

    out_csv = CSV_DIR / f"{prefix}_cc.csv"
    out_pdf = PDF_DIR / f"{prefix}_raster.pdf"
    out_meta = CSV_DIR / f"{prefix}_meta.json"

    tmp_csv = out_csv.with_suffix(out_csv.suffix + ".tmp")
    tmp_pdf = out_pdf.with_suffix(out_pdf.suffix + ".tmp")
    tmp_meta = out_meta.with_suffix(out_meta.suffix + ".tmp")

    if (not RS_CFG.overwrite) and out_csv.exists() and out_pdf.exists() and out_meta.exists():
        with open(out_meta, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["status"] = "skip"
        return meta

    try:
        rng = np.random.default_rng(_seed_from_cond(cond))
        rng_net = np.random.default_rng(3)  # 全条件でネットワーク結線を固定

        is_exc = np.zeros(N, dtype=bool)
        is_exc[:N_E] = True
        idx_E = np.arange(N_E)
        idx_I = np.arange(N_E, N)

        # rows=post, cols=pre
        W = np.zeros((N, N), dtype=bool)
        W[np.ix_(idx_E, idx_E)] = (rng_net.random((N_E, N_E)) < p_EE)
        W[np.ix_(idx_E, idx_I)] = (rng_net.random((N_E, N_I)) < p_IE)
        W[np.ix_(idx_I, idx_E)] = (rng_net.random((N_I, N_E)) < p_EI)
        W[np.ix_(idx_I, idx_I)] = (rng_net.random((N_I, N_I)) < p_II)
        np.fill_diagonal(W, False)
        posts_of = [np.flatnonzero(W[:, j]) for j in range(N)]

        g_E_eff = float(cond.g_E_base) * (N_E0 - 1) / max(1, (N_E - 1))
        g_I_eff = float(cond.g_I_base) * (N_I0 - 1) / max(1, (N_I - 1))

        mu_ge_E = float(cond.bg_mu) * (N_E0 - 1) / max(1, (N_E - 1))
        std_ge_E = float(cond.bg_std) * (N_E0 - 1) / max(1, (N_E - 1))

        profile = "very_loose"
        osc_cfg = get_detection_config(profile)

        # 背景OUの前計算
        bg_enabled = bool(BG_OU_CFG.enabled)
        if bg_enabled:
            tau_ge = float(BG_OU_CFG.tau_ge_ms)
            tau_gi = float(BG_OU_CFG.tau_gi_ms)
            alpha_ge = np.float32(np.exp(-dt / tau_ge))
            alpha_gi = np.float32(np.exp(-dt / tau_gi))

            sqrt_one_minus_a2_ge = np.float32(np.sqrt(max(0.0, 1.0 - float(alpha_ge) ** 2)))
            sqrt_one_minus_a2_gi = np.float32(np.sqrt(max(0.0, 1.0 - float(alpha_gi) ** 2)))

            mu_ge = np.where(is_exc, mu_ge_E, mu_ge_E).astype(np.float32)
            std_ge = np.where(is_exc, std_ge_E, std_ge_E).astype(np.float32)
            mu_gi = np.where(is_exc, mu_ge_E, mu_ge_E).astype(np.float32)
            std_gi = np.where(is_exc, std_ge_E, std_ge_E).astype(np.float32)

            tgt = str(BG_OU_CFG.target).lower()
            if tgt == "e":
                bg_mask = is_exc
            elif tgt == "i":
                bg_mask = ~is_exc
            else:
                bg_mask = np.ones(N, dtype=bool)

            mu_ge = np.where(bg_mask, mu_ge, 0.0).astype(np.float32)
            std_ge = np.where(bg_mask, std_ge, 0.0).astype(np.float32)
            mu_gi = np.where(bg_mask, mu_gi, 0.0).astype(np.float32)
            std_gi = np.where(bg_mask, std_gi, 0.0).astype(np.float32)

            step_ge = std_ge * sqrt_one_minus_a2_ge
            step_gi = std_gi * sqrt_one_minus_a2_gi

            rho_ge = float(np.clip(BG_OU_CFG.rho_shared_ge, 0.0, 1.0))
            rho_gi = float(np.clip(BG_OU_CFG.rho_shared_gi, 0.0, 1.0))
            sqrt_rho_ge = np.float32(np.sqrt(rho_ge))
            sqrt_1mrho_ge = np.float32(np.sqrt(1.0 - rho_ge))
            sqrt_rho_gi = np.float32(np.sqrt(rho_gi))
            sqrt_1mrho_gi = np.float32(np.sqrt(1.0 - rho_gi))

            clip_nonneg = bool(BG_OU_CFG.clip_nonnegative)
            init_stationary = bool(BG_OU_CFG.init_from_stationary)
            ErevE_f32 = np.float32(Erev_E)
            ErevI_f32 = np.float32(Erev_I)

        # 全ニューロン hit 判定用指標
        cv_all_li = []
        rate_all_li = []
        cv0_all_li = []
        cv_hi_thr_all_li = []
        cv_lo_thr_all_li = []
        cv_ratio_all_li = []

        sync_corr_all_li = []
        sync_n_valid_all_li = []
        sync_hi_hit_all_li = []
        sync_lo_hit_all_li = []

        hi_hit_all_li = []
        lo_hit_all_li = []
        valid_all_li = []

        # 付随指標
        m_burst_dur_li = []
        m_burst_freq_li = []
        m_burst_dur_ex_li = []
        m_burst_freq_ex_li = []
        mean_freqs = []
        mean_freqs_ex = []
        max_freqs = []
        max_freqs_ex = []
        is_osc_li = []
        f_osc_li = []

        with PdfPages(tmp_pdf) as pdf:
            for USE_E in USE_Es:
                USE_col = np.where(is_exc, USE_E, float(cond.USE_I)).astype(np.float32)
                tau_rec_col = np.where(is_exc, cond.tau_rec_E, cond.tau_rec_I).astype(np.float32)
                tau_fac_col = np.where(is_exc, tau_fac_E, tau_fac_I).astype(np.float32)
                tau_s_col = np.where(is_exc, tau_s_E, tau_s_I).astype(np.float32)
                g_col = np.where(is_exc, g_E_eff, g_I_eff).astype(np.float32)
                Erev_col = np.where(is_exc, Erev_E, Erev_I).astype(np.float32)

                V = np.full(N, EL, dtype=np.float32)
                refrac_count = np.zeros(N, dtype=np.int32)
                refrac_steps = int(round(refrac_ms / dt))

                # rows=post, cols=pre
                r = np.ones((N, N), dtype=np.float32)
                u = np.broadcast_to(USE_col[None, :], (N, N)).copy()
                s = np.zeros((N, N), dtype=np.float32)

                spike_times = [[] for _ in range(N)]

                p_ext = np.where(is_exc, cond.ext_rate_E, cond.ext_rate_I).astype(np.float32) * (dt * 1e-3)
                spiked_prev = np.zeros(N, dtype=bool)

                dt_over_tau_rec = (dt / tau_rec_col).astype(np.float32)
                dt_over_tau_s = (dt / tau_s_col).astype(np.float32)

                has_fac = bool(np.any(tau_fac_col > 0))
                if has_fac:
                    dt_over_tau_fac = np.zeros_like(tau_fac_col, dtype=np.float32)
                    m_fac = tau_fac_col > 0
                    dt_over_tau_fac[m_fac] = (dt / tau_fac_col[m_fac]).astype(np.float32)

                if bg_enabled:
                    if init_stationary:
                        gE_bg = mu_ge + std_ge * rng.standard_normal(N).astype(np.float32)
                        gI_bg = mu_gi + std_gi * rng.standard_normal(N).astype(np.float32)
                    else:
                        gE_bg = mu_ge.copy()
                        gI_bg = mu_gi.copy()

                    if clip_nonneg:
                        gE_bg = np.maximum(gE_bg, 0.0)
                        gI_bg = np.maximum(gI_bg, 0.0)

                for t in range(1, T_steps):
                    # STP 連続更新
                    r += (1.0 - r) * dt_over_tau_rec[None, :]
                    if has_fac:
                        u += (USE_col[None, :] - u) * dt_over_tau_fac[None, :]
                    s -= s * dt_over_tau_s[None, :]

                    # 外部 Poisson
                    ext_spike = (rng.random(N) < p_ext)

                    # 1dt 遅延で到着するネットワーク内 presyn spike
                    pres = np.flatnonzero(spiked_prev)
                    for j in pres:
                        posts = posts_of[j]
                        if posts.size == 0:
                            continue

                        u_minus = u[posts, j]
                        r_minus = r[posts, j]

                        if STOCHASTIC_RELEASE:
                            k = rng.binomial(N_RELEASE_SITES, np.clip(u_minus, 0.0, 1.0))
                            release_amount = k.astype(np.float32) * np.float32(QUANTAL_ALPHA)
                        else:
                            release_amount = u_minus * r_minus

                        s[posts, j] += release_amount
                        r[posts, j] = r_minus * (1.0 - u_minus)

                        if tau_fac_col[j] > 0.0:
                            u[posts, j] = u_minus + USE_col[j] * (1.0 - u_minus)

                    gs = s * g_col[None, :]
                    sum_gs = gs.sum(axis=1)
                    sum_gsE = (gs * Erev_col[None, :]).sum(axis=1)
                    Isyn = sum_gsE - V * sum_gs

                    # 外部入力は E 型 conductance として各ニューロンに直接注入
                    if np.any(ext_spike):
                        release_ext = 0.5 * float(USE_E)
                        Isyn += (g_E_eff * release_ext) * (Erev_E - V) * ext_spike.astype(np.float32)

                    if bg_enabled:
                        z = rng.standard_normal(N).astype(np.float32)
                        if BG_OU_CFG.rho_shared_ge > 0.0:
                            z_shared = np.float32(rng.standard_normal())
                            z = sqrt_1mrho_ge * z + sqrt_rho_ge * z_shared
                        gE_bg = mu_ge + alpha_ge * (gE_bg - mu_ge) + step_ge * z
                        if clip_nonneg:
                            gE_bg = np.maximum(gE_bg, 0.0)

                        z = rng.standard_normal(N).astype(np.float32)
                        if BG_OU_CFG.rho_shared_gi > 0.0:
                            z_shared = np.float32(rng.standard_normal())
                            z = sqrt_1mrho_gi * z + sqrt_rho_gi * z_shared
                        gI_bg = mu_gi + alpha_gi * (gI_bg - mu_gi) + step_gi * z
                        if clip_nonneg:
                            gI_bg = np.maximum(gI_bg, 0.0)

                        Isyn += gE_bg * (ErevE_f32 - V) + gI_bg * (ErevI_f32 - V)

                    dV = ((EL - V) / tau_m + Isyn) * dt
                    active = (refrac_count == 0)

                    V[active] += dV[active]
                    V[~active] = Vreset
                    refrac_count[~active] -= 1

                    spiked = (V >= Vth) & (refrac_count == 0)
                    if np.any(spiked):
                        V[spiked] = Vreset
                        refrac_count[spiked] = refrac_steps
                        t_ms_now = t * dt
                        for n_idx in np.flatnonzero(spiked):
                            spike_times[n_idx].append(t_ms_now)

                    spiked_prev = spiked

                fig = _make_raster_fig(
                    spike_times,
                    is_exc,
                    title=(
                        f"sample={cond.sample_id}  Uex={USE_E:.2f}  "
                        f"(NE={N_E}, NI={N_I})  "
                        f"{_bg_ou_file_tag(BG_OU_CFG)}"
                    ),
                    tlim_ms=(0, t_end_ms),
                )
                pdf.savefig(fig, dpi=200)
                plt.close(fig)

                spts = spike_times_to_binary(spike_times, T_ms=T_ms, dt_ms=dt).astype(np.uint8)

                # ------------------------------------------------------------
                # CV / rate / CV0 : 全ニューロン（E+I）で計算
                # ------------------------------------------------------------
                cv_all, rate_all_hz = evaluate_population_cv_from_spts(
                    spts,
                    N_sel=N,
                    dt_ms=dt,
                    t_start_ms=t_start_ms,
                    t_end_ms=t_end_ms,
                    cv_w_ms=cv_w,
                    moving_ms=moving,
                )

                cv0_all = cv0_from_rate(rate_hz=rate_all_hz, N_sel=N, cv_w_ms=cv_w)

                hi_thr_all = HIT_CFG.sync_factor * cv0_all if np.isfinite(cv0_all) else np.nan
                lo_thr_all = HIT_CFG.desync_factor * cv0_all if np.isfinite(cv0_all) else np.nan

                ratio_all = (
                    cv_all / cv0_all
                    if np.isfinite(cv_all) and np.isfinite(cv0_all) and cv0_all > 0.0
                    else np.nan
                )

                # ------------------------------------------------------------
                # sync_corr : 全ニューロン（E+I）で計算
                # ------------------------------------------------------------
                sync_neurons = np.arange(N, dtype=int)

                Xsync, _, _ = build_binned_spike_matrix(
                    spk_times=spike_times,
                    neuron_indices=sync_neurons,
                    t_start_ms=t_start_ms,
                    t_end_ms=t_end_ms,
                    bin_ms=SYNC_BIN_MS,
                    binary=SYNC_BINARY,
                )
                sync_corr_all, n_valid_sync_all = mean_pairwise_zero_lag_corr(Xsync)

                sync_hi_hit_all = bool(
                    np.isfinite(sync_corr_all) and (sync_corr_all >= HIT_CFG.sync_corr_hi)
                )
                sync_lo_hit_all = bool(
                    np.isfinite(sync_corr_all) and (sync_corr_all <= HIT_CFG.sync_corr_lo)
                )

                # 最終判定: 全ニューロンの CV_all と sync_corr_all の両方
                hi_hit_all = bool(
                    np.isfinite(cv_all)
                    and np.isfinite(hi_thr_all)
                    and (cv_all >= hi_thr_all)
                    and sync_hi_hit_all
                )
                lo_hit_all = bool(
                    np.isfinite(cv_all)
                    and np.isfinite(lo_thr_all)
                    and (cv_all <= lo_thr_all)
                    and sync_lo_hit_all
                )
                valid_all = bool(
                    np.isfinite(cv_all)
                    and np.isfinite(cv0_all)
                    and np.isfinite(sync_corr_all)
                    and (rate_all_hz >= HIT_CFG.min_rate_hz)
                )

                cv_all_li.append(cv_all)
                rate_all_li.append(rate_all_hz)
                cv0_all_li.append(cv0_all)
                cv_hi_thr_all_li.append(hi_thr_all)
                cv_lo_thr_all_li.append(lo_thr_all)
                cv_ratio_all_li.append(ratio_all)

                sync_corr_all_li.append(sync_corr_all)
                sync_n_valid_all_li.append(n_valid_sync_all)
                sync_hi_hit_all_li.append(sync_hi_hit_all)
                sync_lo_hit_all_li.append(sync_lo_hit_all)

                hi_hit_all_li.append(hi_hit_all)
                lo_hit_all_li.append(lo_hit_all)
                valid_all_li.append(valid_all)

                bursts = detect_bursts(spts, dt_ms=dt, window_ms=10.0, moving_ms=5.0, threshold=80)
                m_burst_dur, m_burst_freq = burst_dur_freq(bursts, spts, dt)
                m_burst_dur_li.append(m_burst_dur)
                m_burst_freq_li.append(m_burst_freq)

                bursts_ex = detect_bursts(spts[:N_E, :], dt_ms=dt, window_ms=10.0, moving_ms=5.0, threshold=80)
                m_burst_dur_ex, m_burst_freq_ex = burst_dur_freq(bursts_ex, spts[:N_E, :], dt)
                m_burst_dur_ex_li.append(m_burst_dur_ex)
                m_burst_freq_ex_li.append(m_burst_freq_ex)

                dur_s_total = T_ms * 1e-3
                mean_freqs.append(float(np.mean([len(spike_times[i]) / dur_s_total for i in range(N)])))
                mean_freqs_ex.append(float(np.mean([len(spike_times[i]) / dur_s_total for i in range(N_E)])))

                E_mask = is_exc
                ALL_mask = np.ones_like(E_mask, dtype=bool)

                _, _, maxE, _, _ = pop_total_rate_and_max_for_group(
                    spts, dt_ms=dt, group_mask=E_mask,
                    window_ms=100.0, step_ms=1.0,
                    t_start_ms=t_start_ms, t_end_ms=t_end_ms,
                )
                _, _, maxA, _, _ = pop_total_rate_and_max_for_group(
                    spts, dt_ms=dt, group_mask=ALL_mask,
                    window_ms=100.0, step_ms=1.0,
                    t_start_ms=t_start_ms, t_end_ms=t_end_ms,
                )
                max_freqs.append(maxA)
                max_freqs_ex.append(maxE)

                # oscillation frequency は従来どおり E 集団レートで計算するが、hit 判定には使わない
                t_rate, rate_hz_trace = population_rate_hz_from_spts(
                    spts,
                    Nn=N,
                    dt_ms=dt,
                    bin_ms=5.0,
                    smooth_ms=0.0,
                    neuron_mask=is_exc,
                )
                is_osc, f_osc, _osc_info = detect_oscillation_and_freq(t_rate, rate_hz_trace, osc_cfg)
                is_osc_li.append(bool(is_osc))
                f_osc_li.append(float(f_osc) if np.isfinite(f_osc) else np.nan)

        df = pd.DataFrame(
            {
                "Uex": USE_Es,
                "osc": np.asarray(is_osc_li, dtype=np.uint8),
                "osc_freq": np.asarray(f_osc_li, dtype=float),

                "mean_freq": np.asarray(mean_freqs, dtype=float),
                "max_freq/100ms": np.asarray(max_freqs, dtype=float),
                "mean_burst_duration": np.asarray(m_burst_dur_li, dtype=float),
                "mean_burst_freq": np.asarray(m_burst_freq_li, dtype=float),

                "cv_all": np.asarray(cv_all_li, dtype=float),
                "rate_all_hz": np.asarray(rate_all_li, dtype=float),
                "cv0_all": np.asarray(cv0_all_li, dtype=float),
                "cv_hi_thr_all": np.asarray(cv_hi_thr_all_li, dtype=float),
                "cv_lo_thr_all": np.asarray(cv_lo_thr_all_li, dtype=float),
                "cv_ratio_all": np.asarray(cv_ratio_all_li, dtype=float),

                "sync_corr_all": np.asarray(sync_corr_all_li, dtype=float),
                "sync_n_valid_all": np.asarray(sync_n_valid_all_li, dtype=int),
                "sync_hi_hit_all": np.asarray(sync_hi_hit_all_li, dtype=np.uint8),
                "sync_lo_hit_all": np.asarray(sync_lo_hit_all_li, dtype=np.uint8),

                "hi_hit_all": np.asarray(hi_hit_all_li, dtype=np.uint8),
                "lo_hit_all": np.asarray(lo_hit_all_li, dtype=np.uint8),
                "valid_all": np.asarray(valid_all_li, dtype=np.uint8),

                "mean_freq_ex": np.asarray(mean_freqs_ex, dtype=float),
                "max_freq_ex/100ms": np.asarray(max_freqs_ex, dtype=float),
                "mean_burst_duration_ex": np.asarray(m_burst_dur_ex_li, dtype=float),
                "mean_burst_freq_ex": np.asarray(m_burst_freq_ex_li, dtype=float),
            }
        )
        df.to_csv(tmp_csv, index=False)

        trans = find_sync_to_desync_transition(df, HIT_CFG)
        hit_row = summarize_hit_row(cond, df, trans)

        meta = {
            "status": "ok",
            "sample_id": cond.sample_id,
            "params": asdict(cond),
            "csv_path": str(out_csv),
            "pdf_path": str(out_pdf),
            "hit": bool(trans["hit"]),
            "sleep_idx": int(trans["sleep_idx"]),
            "wake_idx": int(trans["wake_idx"]),
            "sleep_Uex": None if not np.isfinite(hit_row.get("sleep_Uex", np.nan)) else float(hit_row["sleep_Uex"]),
            "wake_Uex": None if not np.isfinite(hit_row.get("wake_Uex", np.nan)) else float(hit_row["wake_Uex"]),
            "transition": trans,
            "hit_row": hit_row,
            "hit_cfg": asdict(HIT_CFG),
            "sync_cfg": {
                "SYNC_BIN_MS": SYNC_BIN_MS,
                "SYNC_BINARY": SYNC_BINARY,
                "SYNC_ON_POPULATION": SYNC_ON_POPULATION,
            },
        }
        with open(tmp_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        os.replace(tmp_csv, out_csv)
        os.replace(tmp_pdf, out_pdf)
        os.replace(tmp_meta, out_meta)

        if trans["hit"]:
            hit_subdir = HIT_DIR / f"sample{cond.sample_id:06d}"
            hit_subdir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(out_csv, hit_subdir / out_csv.name)
            shutil.copy2(out_pdf, hit_subdir / out_pdf.name)
            with open(hit_subdir / "condition.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "params": asdict(cond),
                        "transition": trans,
                        "hit_row": hit_row,
                        "hit_cfg": asdict(HIT_CFG),
                        "sync_cfg": {
                            "SYNC_BIN_MS": SYNC_BIN_MS,
                            "SYNC_BINARY": SYNC_BINARY,
                            "SYNC_ON_POPULATION": SYNC_ON_POPULATION,
                        },
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

        return meta

    except Exception as e:
        traceback.print_exc()

        err_meta = {
            "status": "err",
            "sample_id": cond.sample_id,
            "params": asdict(cond),
            "csv_path": str(out_csv),
            "pdf_path": str(out_pdf),
            "hit": False,
            "sleep_idx": -1,
            "wake_idx": -1,
            "sleep_Uex": None,
            "wake_Uex": None,
            "error": repr(e),
        }
        try:
            with open(tmp_meta, "w", encoding="utf-8") as f:
                json.dump(err_meta, f, ensure_ascii=False, indent=2)
            os.replace(tmp_meta, out_meta)
        except Exception:
            pass

        return err_meta


# ============================================================
# main
# ============================================================

def main():
    pr = ParamRange()

    print(
        f"[random-search] n_samples={RS_CFG.n_samples}, "
        f"n_proc={RS_CFG.n_proc}, start_method={RS_CFG.start_method}"
    )
    print(
        f"[hit-cfg] sync_factor={HIT_CFG.sync_factor}, desync_factor={HIT_CFG.desync_factor}, "
        f"cv_w={cv_w} ms, moving={moving} ms, min_rate_hz={HIT_CFG.min_rate_hz}"
    )
    print(
        f"[hit-cfg] sync_corr_hi={HIT_CFG.sync_corr_hi}, "
        f"sync_corr_lo={HIT_CFG.sync_corr_lo}, "
        f"sync_bin_ms={SYNC_BIN_MS} ms, sync_binary={SYNC_BINARY}, "
        f"sync_population={SYNC_ON_POPULATION}"
    )
    print("[hit-cfg] synchrony/desynchrony decision uses BOTH cv_all and sync_corr_all on ALL neurons (E+I); osc_freq is recorded but not used.")

    conds = sample_random_conditions(RS_CFG, pr)

    results = []
    ctx = get_context(RS_CFG.start_method)
    with ctx.Pool(processes=RS_CFG.n_proc, maxtasksperchild=1) as pool:
        for i, meta in enumerate(
            pool.imap_unordered(run_one_condition, conds, chunksize=RS_CFG.chunksize),
            start=1
        ):
            results.append(meta)
            if i % 10 == 0:
                print(
                    f"[random-search] done={i}/{len(conds)}  "
                    f"sample={meta.get('sample_id')}  "
                    f"status={meta.get('status')}  "
                    f"hit={meta.get('hit')}"
                )

    results = sorted(results, key=lambda x: int(x.get("sample_id", -1)))

    all_rows = []
    hit_rows = []
    for meta in results:
        row = {
            "sample_id": meta.get("sample_id"),
            "status": meta.get("status"),
            "hit": int(bool(meta.get("hit", False))),
            "csv_path": meta.get("csv_path"),
            "pdf_path": meta.get("pdf_path"),
        }

        params = meta.get("params", {})
        row.update({
            "gE": params.get("g_E_base"),
            "gI": params.get("g_I_base"),
            "tau_rec_E": params.get("tau_rec_E"),
            "tau_rec_I": params.get("tau_rec_I"),
            "UI": params.get("USE_I"),
            "Erate": params.get("ext_rate_E"),
            "Irate": params.get("ext_rate_I"),
            "bg_mu": params.get("bg_mu"),
            "bg_std": params.get("bg_std"),
        })

        hit_row = meta.get("hit_row", {})
        if isinstance(hit_row, dict):
            row.update(hit_row)
            if bool(meta.get("hit", False)):
                hit_rows.append(hit_row)

        all_rows.append(row)

    all_df = pd.DataFrame(all_rows)
    all_df.to_csv(OUT_ROOT / "all_samples_summary.csv", index=False)

    if len(hit_rows) > 0:
        hit_df = pd.DataFrame(hit_rows).sort_values(["sample_id"]).reset_index(drop=True)
    else:
        hit_df = pd.DataFrame(columns=[
            "sample_id", "gE", "gI", "tau_rec_E", "tau_rec_I", "UI", "Erate", "Irate",
            "bg_mu", "bg_std", "seg_start_idx", "seg_end_idx", "cv_drop",
            "wake_Uex", "sleep_Uex", "wake_osc_freq", "sleep_osc_freq",
            "wake_mean_freq", "sleep_mean_freq", "wake_mean_freq_ex", "sleep_mean_freq_ex",
            "wake_rate_all_hz", "sleep_rate_all_hz", "wake_cv0_all", "sleep_cv0_all",
            "wake_cv_ratio_all", "sleep_cv_ratio_all", "wake_hi_thr_all", "sleep_hi_thr_all",
            "wake_lo_thr_all", "sleep_lo_thr_all",
            "wake_sync_corr_all", "sleep_sync_corr_all",
            "wake_max_freq", "sleep_max_freq", "wake_max_freq_ex", "sleep_max_freq_ex",
            "wake_m_burst_dur", "sleep_m_burst_dur", "wake_m_burst_dur_ex", "sleep_m_burst_dur_ex",
            "wake_m_burst_freq", "sleep_m_burst_freq", "wake_m_burst_freq_ex", "sleep_m_burst_freq_ex",
            "wake_cv_all", "sleep_cv_all", "bifur_flag",
        ])
    hit_df.to_csv(OUT_ROOT / "hits_summary.csv", index=False)

    n_ok = int(np.sum(all_df["status"] == "ok")) if len(all_df) else 0
    n_hit = int(np.sum(all_df["hit"] == 1)) if len(all_df) else 0

    print(f"[random-search] finished. ok={n_ok}, hit={n_hit}, total={len(all_df)}")
    print(f"[random-search] all summary : {OUT_ROOT / 'all_samples_summary.csv'}")
    print(f"[random-search] hit summary : {OUT_ROOT / 'hits_summary.csv'}")
    print(f"[random-search] hit dir     : {HIT_DIR}")


if __name__ == "__main__":
    main()