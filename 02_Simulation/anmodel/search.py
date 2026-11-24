# -- coding: utf-8 --
"""
search.py

- RandomParamSearch (AN / SAN / X / NET_SAN_STP)
- SynOnlySearchFromSANIntrinsic (time-transition: early->late, CV-based)
- SynOnlySearchFromSANIntrinsic_UList (U sweep transition: CV decreases with U)
- SynOnlySearchFromSANIntrinsic_UList_bifur (bifurcation Excel -> intrinsic)
- SynOnlySearchFromANIntrinsic_UList (U sweep transition: CV decreases with U)

同期/脱同期の判定は、E集団の「50ms窓での総スパイク数」のCV（=std/mean）を用いる。
  - 同期(バースト強い) : 窓ごとの総スパイク数が揺れやすい -> CVが大きい
  - 脱同期(一様)       : 窓ごとの総スパイク数が揺れにくい -> CVが小さい
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from datetime import datetime
from pathlib import Path
from time import time
from typing import Dict, List, Optional, Tuple, Union, Any
import glob
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import multiprocessing as mp
from multiprocessing import Pool

import traceback
import matplotlib.pyplot as plt

import models
import analysis


# =============================================================================
# Utilities: spike extraction, correlation-based sync (kept), and CV-based metric
# =============================================================================

def _spike_bin_counts(
    v_mat: np.ndarray,
    dt_ms: float,
    thr_mV: float = -20.0,
    bin_ms: float = 5.0,
) -> np.ndarray:
    """
    v_mat: (T, N) membrane voltages
    return: counts (N, B) spike counts per bin
    """
    T, N = v_mat.shape
    above = v_mat > thr_mV
    crossings = np.logical_and(above[1:, :], np.logical_not(above[:-1, :]))  # (T-1, N)

    bin_steps = max(1, int(round(bin_ms / dt_ms)))
    B = (T - 1) // bin_steps
    if B <= 2:
        return np.zeros((N, max(B, 1)), dtype=float)

    counts = np.zeros((N, B), dtype=float)
    for b in range(B):
        s = b * bin_steps
        e = (b + 1) * bin_steps
        counts[:, b] = crossings[s:e, :].sum(axis=0)
    return counts


def _mean_pairwise_corr(X: np.ndarray) -> float:
    """
    X: (N, B)  neuron x timebins
    return: mean of upper-triangular pairwise correlations
    """
    v = X.var(axis=1)
    keep = v > 0
    X2 = X[keep]
    if X2.shape[0] < 3:
        return np.nan
    C = np.corrcoef(X2)
    iu = np.triu_indices(C.shape[0], k=1)
    vals = C[iu]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    return float(vals.mean())


def evaluate_synchrony(
    v_mat: np.ndarray,
    dt_ms: float,
    NE: int,
    eval_window_ms: Tuple[float, float] = (2000.0, 12000.0),
    thr_mV: float = -20.0,
    bin_ms: float = 5.0,
) -> Tuple[float, float]:
    """
    (従来の指標を残しておく)：
    E集団のスパイクビン系列の平均ペア相関を同期性とする。
    返り値: (sync_metric, mean_rate_hz)
    """
    T, N = v_mat.shape
    t_ms = np.arange(T) * dt_ms
    t0, t1 = eval_window_ms
    idx = np.where((t_ms >= t0) & (t_ms < t1))[0]
    if idx.size < 10:
        return np.nan, 0.0

    Vm = v_mat[idx, :NE]  # E only
    X = _spike_bin_counts(Vm, dt_ms=dt_ms, thr_mV=thr_mV, bin_ms=bin_ms)  # (NE, B)
    sync = _mean_pairwise_corr(X)

    dur_s = (t1 - t0) / 1000.0
    spikes_per_neuron = X.sum(axis=1)
    rate_hz = float(np.mean(spikes_per_neuron / max(dur_s, 1e-9)))
    return sync, rate_hz


def _population_indices(NE: int, N: int, on_population: str = "ALL") -> np.ndarray:
    """
    on_population:
      - "E"   : excitatory only
      - "ALL" : all neurons (E + I)
    """
    on_population = str(on_population).upper()
    if on_population == "E":
        return np.arange(NE, dtype=int)
    if on_population == "ALL":
        return np.arange(N, dtype=int)
    raise ValueError("on_population must be 'E' or 'ALL'")


def cv0_from_rate(rate_hz: float, NE: int, cv_w_ms: float) -> float:
    """
    非同期(独立ポアソン)の期待CV:
      CV0 = 1/sqrt(lambda),  lambda = N_sel * rate_hz * (cv_w_ms/1000)

    注意:
      引数名は後方互換のため NE のままだが、
      実際には「評価に使ったニューロン数 N_sel」を渡す。
      Eのみなら N_sel=NE,
      全ニューロンなら N_sel=NE+NI。
    """
    lam = float(NE) * float(rate_hz) * (float(cv_w_ms) / 1000.0)
    if not np.isfinite(lam) or lam <= 0.0:
        return np.nan
    return float(1.0 / np.sqrt(lam))


def evaluate_population_cv(
    v_mat: np.ndarray,
    dt_ms: float,
    NE: int,
    eval_window_ms: Tuple[float, float] = (2000.0, 12000.0),
    thr_mV: float = -20.0,
    cv_w_ms: float = 50.0,
    moving_ms: float = 50.0,
    on_population: str = "ALL",
) -> Tuple[float, float]:
    """
    窓内総スパイク数のCV (= std/mean)

    on_population:
      - "E"   : excitatoryのみ
      - "ALL" : 全ニューロン (E+I)

    Returns
    -------
    cv : float
        windowごとの選択集団総スパイク数のCV
    rate_hz : float
        選択集団における1ニューロンあたり平均発火率[Hz]
    """
    T, N = v_mat.shape
    if NE <= 0 or NE > N:
        return np.nan, 0.0
    if T < 3:
        return np.nan, 0.0

    sel_idx = _population_indices(NE=NE, N=N, on_population=on_population)
    N_sel = len(sel_idx)
    if N_sel <= 0:
        return np.nan, 0.0

    t0, t1 = float(eval_window_ms[0]), float(eval_window_ms[1])
    t_max = (T - 1) * float(dt_ms)
    t0 = max(0.0, min(t0, t_max))
    t1 = max(0.0, min(t1, t_max))
    if t1 <= t0:
        return np.nan, 0.0

    # threshold crossing: v[t-1] < thr <= v[t]
    v0 = v_mat[:-1, :]
    v1 = v_mat[1:, :]
    cross = (v0 < thr_mV) & (v1 >= thr_mV)  # (T-1, N)

    # restrict to eval window using time at step index (k+1)
    t_cross = np.arange(1, T, dtype=float) * float(dt_ms)
    mask = (t_cross >= t0) & (t_cross < t1)
    if not np.any(mask):
        return np.nan, 0.0

    cross_sel = cross[:, sel_idx]          # (T-1, N_sel)
    cross_win = cross_sel[mask, :]         # (Tw, N_sel)
    pop_counts_dt = cross_win.sum(axis=1).astype(np.float64)

    dur_s = (t1 - t0) / 1000.0
    total_spikes = float(pop_counts_dt.sum())
    rate_hz = total_spikes / (max(dur_s, 1e-12) * float(N_sel))

    win_steps = max(1, int(round(cv_w_ms / float(dt_ms))))
    step_steps = max(1, int(round(moving_ms / float(dt_ms))))
    L = pop_counts_dt.size
    if L < win_steps + 1:
        return np.nan, rate_hz

    starts = np.arange(0, L - win_steps + 1, step_steps, dtype=int)
    if starts.size < 2:
        return np.nan, rate_hz

    csum = np.concatenate([[0.0], np.cumsum(pop_counts_dt)])
    win_sums = csum[starts + win_steps] - csum[starts]

    m = float(win_sums.mean())
    if m <= 0.0:
        cv = 0.0
    else:
        cv = float(win_sums.std(ddof=0) / m)

    return cv, rate_hz


def evaluate_population_cv_and_sync_corr(
    v_mat: np.ndarray,
    dt_ms: float,
    NE: int,
    eval_window_ms: Tuple[float, float] = (2000.0, 12000.0),
    thr_mV: float = -20.0,
    cv_w_ms: float = 50.0,
    moving_ms: float = 50.0,
    sync_bin_ms: float = 5.0,
    sync_binary: bool = True,
    cv_on_population: str = "ALL",
    sync_on_population: str = "ALL",
) -> Tuple[float, float, float, int]:
    """
    窓内総スパイク数CVと、5 ms bin のゼロラグ平均相関 sync_corr を同時に評価する。

    cv_on_population:
      - "E"   : CVをEのみで計算
      - "ALL" : CVを全ニューロンで計算

    sync_on_population:
      - "E"   : sync_corrをEのみで計算
      - "ALL" : sync_corrを全ニューロンで計算

    Returns
    -------
    cv : float
        選択集団の窓内総スパイク数CV
    rate_hz : float
        CV評価に使った集団における1ニューロンあたり平均発火率 [Hz]
    sync_corr : float
        指定集団の 5 ms bin ゼロラグ平均相関
    n_valid_sync : int
        sync_corr 計算に使われた「分散 > 0」の有効ニューロン数
    """
    T, N = v_mat.shape
    if NE <= 0 or NE > N:
        return np.nan, 0.0, np.nan, 0
    if T < 3:
        return np.nan, 0.0, np.nan, 0

    cv_neurons = _population_indices(NE=NE, N=N, on_population=cv_on_population)
    sync_neurons = _population_indices(NE=NE, N=N, on_population=sync_on_population)

    if len(cv_neurons) == 0 or len(sync_neurons) == 0:
        return np.nan, 0.0, np.nan, 0

    t0, t1 = float(eval_window_ms[0]), float(eval_window_ms[1])
    t_max = (T - 1) * float(dt_ms)
    t0 = max(0.0, min(t0, t_max))
    t1 = max(0.0, min(t1, t_max))
    if t1 <= t0:
        return np.nan, 0.0, np.nan, 0

    # threshold crossing: v[t-1] < thr <= v[t]
    v0 = v_mat[:-1, :]
    v1 = v_mat[1:, :]
    cross = (v0 < thr_mV) & (v1 >= thr_mV)   # (T-1, N)
    t_cross = np.arange(1, T, dtype=float) * float(dt_ms)

    # 評価窓で制限
    mask = (t_cross >= t0) & (t_cross < t1)
    if not np.any(mask):
        return np.nan, 0.0, np.nan, 0

    # ---- CV ----
    cross_win = cross[mask, :]  # (Tw, N)
    pop_counts_dt = cross_win[:, cv_neurons].sum(axis=1).astype(np.float64)

    dur_s = (t1 - t0) / 1000.0
    total_spikes = float(pop_counts_dt.sum())
    rate_hz = total_spikes / (max(dur_s, 1e-12) * float(len(cv_neurons)))

    win_steps = max(1, int(round(cv_w_ms / float(dt_ms))))
    step_steps = max(1, int(round(moving_ms / float(dt_ms))))
    L = pop_counts_dt.size

    if L < win_steps + 1:
        cv = np.nan
    else:
        starts = np.arange(0, L - win_steps + 1, step_steps, dtype=int)
        if starts.size < 2:
            cv = np.nan
        else:
            csum = np.concatenate([[0.0], np.cumsum(pop_counts_dt)])
            win_sums = csum[starts + win_steps] - csum[starts]
            m = float(win_sums.mean())
            if m <= 0.0:
                cv = 0.0
            else:
                cv = float(win_sums.std(ddof=0) / m)

    # ---- sync_corr ----
    spk_times = [t_cross[mask & cross[:, i]] for i in range(N)]

    Xsync, _, _ = build_binned_spike_matrix(
        spk_times=spk_times,
        neuron_indices=sync_neurons,
        t_start_ms=t0,
        t_end_ms=t1,
        bin_ms=sync_bin_ms,
        binary=sync_binary,
    )
    sync_corr, n_valid_sync = mean_pairwise_zero_lag_corr(Xsync)

    return cv, rate_hz, sync_corr, n_valid_sync


def find_decreasing_segment_with_dynamic_thresholds(
    U_list: List[float],
    cv_list: List[float],
    valid_list: List[bool],
    hi_hit_list: List[bool],   # cv >= (sync_factor * CV0) を満たしたか
    lo_hit_list: List[bool],   # cv <= (desync_factor * CV0) を満たしたか
    decrease_eps: float = 0.01,
) -> Optional[Dict[str, Any]]:
    """
    条件:
      - 連続したUインデックス区間 [i..j], j>=i+1
      - cv が単調減少: cv[k+1] <= cv[k] - decrease_eps
      - 区間内に hi_hit=True が少なくとも1つ（同期側を含む）
      - 区間内に lo_hit=True が少なくとも1つ（脱同期側を含む）
      - 区間内の点は valid=True

    戻り値: dict(start_idx, end_idx, drop, max_cv, min_cv) / なしなら None
    """
    n = len(U_list)
    best = None
    best_drop = -np.inf

    for i in range(n - 1):
        if not valid_list[i]:
            continue
        j = i

        # 右に伸ばして「単調減少」区間を作る
        while j + 1 < n:
            if not valid_list[j + 1]:
                break
            if not (cv_list[j + 1] <= cv_list[j] - decrease_eps):
                break
            j += 1

            if j - i + 1 >= 2:
                seg_cv = cv_list[i:j+1]
                seg_hi = hi_hit_list[i:j+1]
                seg_lo = lo_hit_list[i:j+1]

                if (any(seg_hi)) and (any(seg_lo)):
                    drop = float(cv_list[i] - cv_list[j])
                    if drop > best_drop:
                        best_drop = drop
                        best = {
                            "start_idx": i,
                            "end_idx": j,
                            "drop": drop,
                            "max_cv": float(np.nanmax(seg_cv)),
                            "min_cv": float(np.nanmin(seg_cv)),
                        }
    return best


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

    # 全binで一定値の neuron（全0など）は相関が定義できないので除外
    keep = np.std(X, axis=1) > 0
    Xk = X[keep]

    if Xk.shape[0] < 2:
        return np.nan, int(Xk.shape[0])

    C = np.corrcoef(Xk)
    iu = np.triu_indices(C.shape[0], k=1)
    if len(iu[0]) == 0:
        return np.nan, int(Xk.shape[0])

    return float(np.nanmean(C[iu])), int(Xk.shape[0])



def find_decreasing_segment_with_cv_sync_desync(
    U_list: List[float],
    cv_list: List[float],
    valid_list: List[bool],
    cv_sync_hi: float,
    cv_desync_lo: float,
    decrease_eps: float = 0.01,
) -> Optional[Dict[str, Any]]:
    """
    条件:
      - 連続したUインデックス区間 [i..j], j>=i+1
      - cv が単調減少: cv[k+1] <= cv[k] - decrease_eps
      - 区間内に cv>=cv_sync_hi を満たす点が少なくとも1つ
      - 区間内に cv<=cv_desync_lo を満たす点が少なくとも1つ
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
                seg = cv_list[i:j+1]
                seg_max = float(np.nanmax(seg))
                seg_min = float(np.nanmin(seg))
                if (seg_max >= cv_sync_hi) and (seg_min <= cv_desync_lo):
                    drop = float(cv_list[i] - cv_list[j])
                    if drop > best_drop:
                        best_drop = drop
                        best = {
                            "start_idx": i,
                            "end_idx": j,
                            "drop": drop,
                            "max_cv": seg_max,
                            "min_cv": seg_min,
                        }
    return best


def detect_sync_to_desync_transition(
    v_mat: np.ndarray,
    dt_ms: float,
    NE: int,
    early_ms: Tuple[float, float] = (3000.0, 8000.0),
    late_ms: Tuple[float, float] = (14000.0, 19000.0),
    thr_mV: float = -20.0,
    cv_w_ms: float = 50.0,
    moving_ms: float = 50.0,
    cv_sync_hi: float = 0.2,
    cv_desync_lo: float = 0.07,
    decrease_eps: float = 0.01,
    min_rate_hz: float = 0.1,
) -> Tuple[bool, Dict[str, float]]:
    """
    時間軸で early->late の同期→脱同期をCVで判定する。

    ok条件:
      - early_cv >= cv_sync_hi
      - late_cv  <= cv_desync_lo
      - late_cv <= early_cv - decrease_eps
      - early_rate, late_rate >= min_rate_hz
    """
    early_cv, early_rate = evaluate_population_cv(
        v_mat=v_mat, dt_ms=dt_ms, NE=NE,
        eval_window_ms=early_ms, thr_mV=thr_mV,
        cv_w_ms=cv_w_ms, moving_ms=moving_ms,
    )
    late_cv, late_rate = evaluate_population_cv(
        v_mat=v_mat, dt_ms=dt_ms, NE=NE,
        eval_window_ms=late_ms, thr_mV=thr_mV,
        cv_w_ms=cv_w_ms, moving_ms=moving_ms,
    )

    ok = (
        np.isfinite(early_cv) and np.isfinite(late_cv)
        and (early_cv >= cv_sync_hi)
        and (late_cv <= cv_desync_lo)
        and (late_cv <= early_cv - decrease_eps)
        and (early_rate >= min_rate_hz)
        and (late_rate >= min_rate_hz)
    )

    metrics = {
        "early_cv": float(early_cv) if np.isfinite(early_cv) else np.nan,
        "late_cv": float(late_cv) if np.isfinite(late_cv) else np.nan,
        "early_rate_hz": float(early_rate),
        "late_rate_hz": float(late_rate),
    }
    return bool(ok), metrics


# =============================================================================
# Random search (original + NET_SAN_STP support)
# =============================================================================

def _random_search_worker(cfg: Dict) -> None:
    """
    Spawn-safe worker. cfg must be picklable dict (only primitives/lists/dicts).
    """
    core = cfg["core"]
    now = cfg["now"]
    time_start = cfg["time_start"]
    rand_seed = int(cfg["rand_seed"])

    model_type = cfg["model_type"]
    pattern = cfg["pattern"]
    model_name = cfg["model_name"]

    samp_freq = int(cfg["samp_freq"])
    samp_len = int(cfg["samp_len"])
    hr = float(cfg["hr"])

    # output path
    date = f"{now.year}_{now.month}_{now.day}"
    p = Path.cwd()
    res_p = p / "results" / f"{pattern}_params" / f"{date}_{model_name}"
    res_p.mkdir(parents=True, exist_ok=True)
    save_p = res_p / f"{pattern}_{date}_{core}.pickle"

    np.random.seed(rand_seed)

    # instantiate model inside worker
    if model_type == "AN":
        model = models.ANmodel(cfg.get("ion", False), cfg.get("concentration", None))
    elif model_type == "SAN":
        model = models.SANmodel(cfg.get("ion", False), cfg.get("concentration", None))
    elif model_type == "X":
        model = models.Xmodel(cfg["channel_bool"], cfg.get("ion", False), cfg.get("concentration", None))
    elif model_type == "NET_SAN_STP":
        model = models.NET_SAN_STP(
            NE=int(cfg.get("NE", 16)),
            NI=int(cfg.get("NI", 4)),
            net_seed=int(cfg.get("net_seed", 0)),
            p_ee=float(cfg.get("p_ee", 0.2)),
            p_ei=float(cfg.get("p_ei", 0.2)),
            p_ie=float(cfg.get("p_ie", 0.3)),
            p_ii=float(cfg.get("p_ii", 0.3)),
            dt_ms=float(cfg.get("dt_ms", 0.1)),
        )
        net_check = analysis.NetTransitionCheck(
            early_ms=tuple(cfg.get("early_ms", (3000.0, 8000.0))),
            late_ms=tuple(cfg.get("late_ms", (14000.0, 19000.0))),
            R_sync=float(cfg.get("R_sync", 0.70)),
            R_desync=float(cfg.get("R_desync", 0.45)),
            spike_thresh=float(cfg.get("spike_thresh", -20.0)),
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # checkers for single neuron models
    if model_type in ("AN", "SAN", "X"):
        wave_check = analysis.WaveCheck(samp_freq=samp_freq)

    param_df = pd.DataFrame([])
    niter = 0
    nhit = 0
    st = time()

    while True:
        niter += 1

        # random params
        new_params = pd.DataFrame.from_dict(model.set_rand_params(), orient="index").T

        # run
        if model_type == "NET_SAN_STP":
            s, info = model.run_odeint(samp_freq=samp_freq, samp_len=samp_len, dt_ms=cfg.get("dt_ms", 0.1))
            v_mat = s
            t_ms = info.get("t_ms", None)
            if t_ms is None:
                # fallback
                dt0 = float(info.get("dt_ms", cfg.get("dt_ms", 0.1)))
                rec = int(info.get("record_every", 1))
                t_ms = np.arange(v_mat.shape[0], dtype=float) * dt0 * rec
            ev = net_check.evaluate(v_mat, t_ms)

            ok = bool(ev["ok"])
            new_params["R_early"] = ev.get("R_early", np.nan)
            new_params["R_late"] = ev.get("R_late", np.nan)

        else:
            s, info = model.run_odeint(samp_freq=samp_freq, samp_len=samp_len)

            if info.get("message", "") == "Excess work done on this call (perhaps wrong Dfun type).":
                ok = False
            else:
                v = s[samp_freq * samp_len // 2 :, 0]
                pat = wave_check.pattern(v=v)
                ok = (pat.name == pattern)

        if ok:
            nhit += 1
            param_df = pd.concat([param_df, new_params], ignore_index=True)
            print(f"Hit! core={core} iter={niter} hits={nhit}")

        # save every 1 hour
        md = time()
        if (md - st) > 60 * 60:
            st = time()
            with open(str(save_p), "wb") as f:
                pickle.dump(niter, f)
                pickle.dump(param_df, f)
            print(datetime.now(), f"Core {core}: {len(param_df)} hits pickled -> {save_p}")

        # terminate
        if (md - time_start) > 60 * 60 * hr:
            with open(str(save_p), "wb") as f:
                pickle.dump(niter, f)
                pickle.dump(param_df, f)
            print(f"Core {core}: {hr} hours passed, terminated. Final hits={len(param_df)}")
            break


class RandomParamSearch:
    """
    Random parameter search unified for:
      - AN, SAN, X (original)
      - NET_SAN_STP (new)
    """

    def __init__(
        self,
        model: str,
        pattern: str = "SWS",
        ncore: int = 1,
        hr: int = 48,
        samp_freq: int = 1000,
        samp_len: int = 10,
        channel_bool: Optional[List[bool]] = None,
        model_name: Optional[str] = None,
        ion: bool = False,
        concentration: Optional[Dict] = None,

        # multiprocessing
        start_method: str = "spawn",

        # NET_SAN_STP options (ignored for other models)
        NE: int = 16,
        NI: int = 4,
        net_seed: int = 0,
        p_ee: float = 0.2,
        p_ei: float = 0.2,
        p_ie: float = 0.3,
        p_ii: float = 0.3,
        dt_ms: float = 0.1,
        early_ms=(3000.0, 8000.0),
        late_ms=(14000.0, 19000.0),
        R_sync: float = 0.70,
        R_desync: float = 0.45,
        spike_thresh: float = -20.0,
    ) -> None:
        self.model_type = model
        self.pattern = pattern
        self.ncore = int(ncore)
        self.hr = float(hr)
        self.samp_freq = int(samp_freq)
        self.samp_len = int(samp_len)

        self.channel_bool = channel_bool
        self.model_name = model_name or model

        self.ion = ion
        self.concentration = concentration

        self.start_method = start_method

        # NET config
        self.NE = int(NE)
        self.NI = int(NI)
        self.net_seed = int(net_seed)
        self.p_ee = float(p_ee)
        self.p_ei = float(p_ei)
        self.p_ie = float(p_ie)
        self.p_ii = float(p_ii)
        self.dt_ms = float(dt_ms)
        self.early_ms = tuple(early_ms)
        self.late_ms = tuple(late_ms)
        self.R_sync = float(R_sync)
        self.R_desync = float(R_desync)
        self.spike_thresh = float(spike_thresh)

        if self.model_type == "X" and self.channel_bool is None:
            raise TypeError("Designate channel_bool when model='X'.")

    def multi_singleprocess(self) -> None:
        now = datetime.now()
        time_start = time()

        cfgs = []
        for core in range(self.ncore):
            cfg = {
                "core": int(core),
                "now": now,
                "time_start": float(time_start),
                "rand_seed": int(np.random.randint(0, 2**16 - 1)),

                "model_type": self.model_type,
                "pattern": self.pattern,
                "model_name": self.model_name,

                "samp_freq": self.samp_freq,
                "samp_len": self.samp_len,
                "hr": self.hr,

                "ion": self.ion,
                "concentration": self.concentration,
            }
            if self.model_type == "X":
                cfg["channel_bool"] = self.channel_bool

            if self.model_type == "NET_SAN_STP":
                cfg.update({
                    "NE": self.NE,
                    "NI": self.NI,
                    "net_seed": self.net_seed + core,  # 固定がよければ +core を消す
                    "p_ee": self.p_ee,
                    "p_ei": self.p_ei,
                    "p_ie": self.p_ie,
                    "p_ii": self.p_ii,
                    "dt_ms": self.dt_ms,
                    "early_ms": self.early_ms,
                    "late_ms": self.late_ms,
                    "R_sync": self.R_sync,
                    "R_desync": self.R_desync,
                    "spike_thresh": self.spike_thresh,
                })

            cfgs.append(cfg)

        print(f"Random search: model={self.model_type} pattern={self.pattern} ncore={self.ncore} start_method={self.start_method}")

        ctx = mp.get_context(self.start_method)
        with ctx.Pool(processes=self.ncore) as pool:
            pool.map(_random_search_worker, cfgs)


# =============================================================================
# Syn-only search from SAN intrinsic (time-transition version, CV-based)
# =============================================================================

class SynOnlySearchFromSANIntrinsic:
    """
    SAN の intrinsic パラメータセット（RandomParamSearch出力pickle）を読み込み、
    intrinsic固定で NET_SAN_STP の syn だけ探索。

    ここでは「時間方向の early->late の同期→脱同期」を
    E集団スパイク総数のCVで判定して hit とする。
    """

    def __init__(
        self,
        intrinsic_pickle: str,
        pattern: str = "TRANS",
        ncore: int = 8,
        hr: int = 24,
        NE: int = 16,
        NI: int = 4,
        dt_ms: float = 0.1,
        T_ms: float = 20000.0,
        net_seed_base: int = 0,
        syn_trials_per_intrinsic: int = 200,
        max_intrinsic: Optional[int] = None,
        shuffle_intrinsic: bool = True,
        # connection probs
        p_ee: float = 0.2, p_ei: float = 0.2, p_ie: float = 0.3, p_ii: float = 0.3,

        # CV-based transition params
        early_ms: Tuple[float, float] = (3000.0, 8000.0),
        late_ms: Tuple[float, float] = (14000.0, 19000.0),
        thr_mV: float = -20.0,
        cv_w_ms: float = 50.0,
        moving_ms: float = 50.0,
        cv_sync_hi: float = 0.2,
        cv_desync_lo: float = 0.07,
        decrease_eps: float = 0.01,
        min_rate_hz: float = 0.1,
    ) -> None:

        self.intrinsic_pickle = str(intrinsic_pickle)
        self.pattern = pattern
        self.ncore = int(ncore)
        self.hr = int(hr)

        self.NE = int(NE)
        self.NI = int(NI)
        self.dt_ms = float(dt_ms)
        self.T_ms = float(T_ms)

        self.net_seed_base = int(net_seed_base)
        self.syn_trials_per_intrinsic = int(syn_trials_per_intrinsic)

        self.max_intrinsic = max_intrinsic
        self.shuffle_intrinsic = bool(shuffle_intrinsic)

        self.p_ee, self.p_ei, self.p_ie, self.p_ii = float(p_ee), float(p_ei), float(p_ie), float(p_ii)

        self.early_ms = tuple(early_ms)
        self.late_ms = tuple(late_ms)
        self.thr_mV = float(thr_mV)
        self.cv_w_ms = float(cv_w_ms)
        self.moving_ms = float(moving_ms)
        self.cv_sync_hi = float(cv_sync_hi)
        self.cv_desync_lo = float(cv_desync_lo)
        self.decrease_eps = float(decrease_eps)
        self.min_rate_hz = float(min_rate_hz)

    @staticmethod
    def _load_intrinsic_df(pickle_path: str) -> pd.DataFrame:
        with open(pickle_path, "rb") as f:
            _ = pickle.load(f)       # niter
            df = pickle.load(f)      # param_df
        if not isinstance(df, pd.DataFrame):
            raise TypeError("intrinsic_pickle does not contain a DataFrame at 2nd object.")
        need = ["g_leak", "g_kvhh", "g_cav", "g_kca", "g_nap", "t_ca"]
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise KeyError(f"Intrinsic DF missing columns: {missing}")
        return df.reset_index(drop=True)

    def _result_dir(self, now: datetime) -> Path:
        date = f"{now.year}_{now.month}_{now.day}"
        p = Path.cwd()
        res_p = p / "results" / f"{self.pattern}_params" / f"{date}_NET_SAN_STP_SYNONLY"
        res_p.mkdir(parents=True, exist_ok=True)
        return res_p

    def singleprocess(self, args: Tuple[int, datetime, float, int]) -> None:
        core, now, time_start, rand_seed = args
        rng = np.random.default_rng(int(rand_seed))

        intr_df = self._load_intrinsic_df(self.intrinsic_pickle)

        idx_all = np.arange(len(intr_df))
        if self.shuffle_intrinsic:
            rng.shuffle(idx_all)
        if self.max_intrinsic is not None:
            idx_all = idx_all[: int(self.max_intrinsic)]

        idx_core = idx_all[core::self.ncore]

        res_p = self._result_dir(now)
        date = f"{now.year}_{now.month}_{now.day}"
        save_p = res_p / f"{self.pattern}_{date}_core{core}.pickle"

        hit_df = pd.DataFrame([])
        niter = 0
        st = time()

        for intrinsic_id in idx_core:
            intrinsic = intr_df.loc[int(intrinsic_id)].to_dict()

            model = models.NET_SAN_STP(
                NE=self.NE, NI=self.NI,
                net_seed=self.net_seed_base + int(intrinsic_id),
                p_ee=self.p_ee, p_ei=self.p_ei, p_ie=self.p_ie, p_ii=self.p_ii,
                dt_ms=self.dt_ms,
            )
            model.set_intrinsic_fixed(intrinsic)
            model.set_U_I(self.U_I_fixed)

            for _trial in range(self.syn_trials_per_intrinsic):
                niter += 1
                syn = model.gen_syn_params()
                model.set_syn_params(syn)

                try:
                    v_mat, info = model.run_odeint(samp_freq=1000, samp_len=self.T_ms)
                    dt_rec_ms = float(info.get("dt_ms", self.dt_ms)) * float(info.get("record_every", 1))
                    ok, metrics = detect_sync_to_desync_transition(
                        v_mat=v_mat,
                        dt_ms=dt_rec_ms,
                        NE=self.NE,
                        early_ms=self.early_ms,
                        late_ms=self.late_ms,
                        thr_mV=self.thr_mV,
                        cv_w_ms=self.cv_w_ms,
                        moving_ms=self.moving_ms,
                        cv_sync_hi=self.cv_sync_hi,
                        cv_desync_lo=self.cv_desync_lo,
                        decrease_eps=self.decrease_eps,
                        min_rate_hz=self.min_rate_hz,
                    )
                except Exception:
                    traceback.print_exc()
                    ok = False
                    metrics = {}

                if ok:
                    row = {}
                    row.update({k: float(intrinsic[k]) for k in ["g_leak","g_kvhh","g_cav","g_kca","g_nap","t_ca"]})
                    row.update({k: float(syn[k]) for k in syn.keys()})
                    row["intrinsic_id"] = int(intrinsic_id)
                    row["core"] = int(core)
                    for mk, mv in metrics.items():
                        row[mk] = float(mv)

                    hit_df = pd.concat([hit_df, pd.DataFrame([row])], ignore_index=True)
                    print(f"[core {core}] Hit transition! intrinsic_id={intrinsic_id}")
                    break

                md = time()
                if (md - st) > 60 * 60:
                    st = time()
                    with open(str(save_p), "wb") as f:
                        pickle.dump(niter, f)
                        pickle.dump(hit_df, f)
                    print(datetime.now(), f"[core {core}] saved {len(hit_df)} hits")

                if (md - time_start) > 60 * 60 * self.hr:
                    print(f"[core {core}] {self.hr} hours passed -> terminate.")
                    with open(str(save_p), "wb") as f:
                        pickle.dump(niter, f)
                        pickle.dump(hit_df, f)
                    return

            md = time()
            if (md - st) > 60 * 60:
                st = time()
                with open(str(save_p), "wb") as f:
                    pickle.dump(niter, f)
                    pickle.dump(hit_df, f)
                print(datetime.now(), f"[core {core}] saved {len(hit_df)} hits")

            if (md - time_start) > 60 * 60 * self.hr:
                print(f"[core {core}] {self.hr} hours passed -> terminate.")
                with open(str(save_p), "wb") as f:
                    pickle.dump(niter, f)
                    pickle.dump(hit_df, f)
                return

        with open(str(save_p), "wb") as f:
            pickle.dump(niter, f)
            pickle.dump(hit_df, f)
        print(f"[core {core}] finished all intrinsic. total hits={len(hit_df)}")

    def multi_singleprocess(self) -> None:
        now = datetime.now()
        time_start = time()

        args: List[Tuple[int, datetime, float, int]] = []
        for core in range(self.ncore):
            args.append((int(core), now, float(time_start), int(np.random.randint(0, 2**16 - 1))))

        print(f"Syn-only search from SAN intrinsic: using {self.ncore} cores, hr={self.hr}")
        with Pool(processes=self.ncore) as pool:
            pool.map(self.singleprocess, args)


# =============================================================================
# Syn-only search from SAN intrinsic (U-list version, CV-based)
# =============================================================================
class SynOnlySearchFromSANIntrinsic_UList:
    """
    intrinsic固定して NET_SAN_STP の synだけ探索。
    U_list の各Uで、
      - 窓内総スパイク数CV
      - 5 ms bin ゼロラグ平均相関 sync_corr
    を評価する。

    ★動的CV閾値:
      CV0(U) = 1/sqrt( N_sel * rate_hz(U) * (cv_w_ms/1000) )
      CV同期側:   CV(U) >= sync_factor   * CV0(U)
      CV脱同期側: CV(U) <= desync_factor * CV0(U)

    ★相関閾値:
      同期側:   sync_corr(U) >= sync_corr_hi
      脱同期側: sync_corr(U) <= sync_corr_lo

    ★U点ごとの同期/脱同期判定:
      同期側ヒット   = [CV同期側] and [corr同期側]
      脱同期側ヒット = [CV脱同期側] and [corr脱同期側]

    ★hit条件:
      - 2点以上の連続Uで CV が単調減少
      - その区間内に「同期側ヒット」と「脱同期側ヒット」が両方含まれる
    """

    def __init__(
        self,
        intrinsic_sources: Union[str, List[str]],
        U_list: List[float],
        pattern: str = "U_TRANS",
        fname: Optional[str] = None,
        ncore: int = 8,
        hr: int = 24,
        NE: int = 16,
        NI: int = 4,
        dt_ms: float = 0.1,
        T_ms: float = 15000.0,
        net_seed_base: int = 0,
        syn_trials_per_intrinsic: int = 200,
        max_intrinsic: Optional[int] = None,
        shuffle_intrinsic: bool = True,
        

        # connection probs
        p_ee: float = 0.2, p_ei: float = 0.2, p_ie: float = 0.3, p_ii: float = 0.3,

        # evaluation window & spike threshold
        eval_window_ms: Tuple[float, float] = (2000.0, 12000.0),
        thr_mV: float = -20.0,

        # CV params
        cv_w_ms: float = 50.0,
        moving_ms: float = 50.0,
        cv_on_population: str = "ALL",

        # sync_corr params
        sync_bin_ms: float = 5.0,
        sync_binary: bool = True,
        sync_on_population: str = "ALL",
        sync_corr_hi: float = 0.15,
        sync_corr_lo: float = 0.05,

        # 動的閾値係数
        sync_factor: float = 2.0,
        desync_factor: float = 1.2,

        decrease_eps: float = 0.01,
        min_rate_hz: float = 0.5,

        U_I_fixed: float = 0.3,
    ) -> None:

        self.intrinsic_sources = intrinsic_sources
        self.U_list = [float(u) for u in U_list]
        self.U_I_fixed = float(U_I_fixed)

        self.pattern = pattern
        self.fname = fname or "SAN"
        self.ncore = int(ncore)
        self.hr = int(hr)

        self.NE = int(NE)
        self.NI = int(NI)
        self.dt_ms = float(dt_ms)
        self.T_ms = float(T_ms)

        self.net_seed_base = int(net_seed_base)
        self.syn_trials_per_intrinsic = int(syn_trials_per_intrinsic)

        self.max_intrinsic = max_intrinsic
        self.shuffle_intrinsic = bool(shuffle_intrinsic)

        self.p_ee = float(p_ee)
        self.p_ei = float(p_ei)
        self.p_ie = float(p_ie)
        self.p_ii = float(p_ii)

        self.eval_window_ms = tuple(eval_window_ms)
        self.thr_mV = float(thr_mV)

        self.cv_w_ms = float(cv_w_ms)
        self.moving_ms = float(moving_ms)
        self.cv_on_population = str(cv_on_population).upper()

        self.sync_bin_ms = float(sync_bin_ms)
        self.sync_binary = bool(sync_binary)
        self.sync_on_population = str(sync_on_population).upper()
        self.sync_corr_hi = float(sync_corr_hi)
        self.sync_corr_lo = float(sync_corr_lo)

        self.sync_factor = float(sync_factor)
        self.desync_factor = float(desync_factor)

        self.decrease_eps = float(decrease_eps)
        self.min_rate_hz = float(min_rate_hz)

    @staticmethod
    def _read_one_pickle(path: str) -> pd.DataFrame:
        with open(path, "rb") as f:
            _ = pickle.load(f)
            df = pickle.load(f)
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{path} does not contain a DataFrame at 2nd object.")
        return df

    def _load_intrinsic_df(self) -> pd.DataFrame:
        paths: List[str] = []
        if isinstance(self.intrinsic_sources, str):
            src = self.intrinsic_sources
            if os.path.isdir(src):
                paths = sorted(glob.glob(os.path.join(src, "*.pickle")))
            else:
                paths = [src]
        else:
            for s in self.intrinsic_sources:
                if os.path.isdir(s):
                    paths.extend(sorted(glob.glob(os.path.join(s, "*.pickle"))))
                else:
                    paths.append(s)
            paths = sorted(list(dict.fromkeys(paths)))

        if len(paths) == 0:
            raise FileNotFoundError("No intrinsic pickle files found.")

        dfs = [self._read_one_pickle(p) for p in paths]
        df = pd.concat(dfs, ignore_index=True)

        need = ["g_leak", "g_kvhh", "g_cav", "g_kca", "g_nap", "t_ca"]
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise KeyError(f"Intrinsic DF missing columns: {missing}")

        df = df.dropna(subset=need).reset_index(drop=True)
        return df

    def _result_dir(self, now: datetime) -> Path:
        date = f"{now.year}_{now.month}_{now.day}"
        p = Path.cwd()
        res_p = p / "results" / f"{self.pattern}_params" / f"{date}_NET_SAN_STP_ULIST_{self.fname}"
        res_p.mkdir(parents=True, exist_ok=True)
        return res_p

    @staticmethod
    def _u_tag(u: float) -> str:
        return f"{u}".replace(".", "p")

    def singleprocess(self, args: Tuple[int, datetime, float, int]) -> None:
        core, now, time_start, rand_seed = args
        rng = np.random.default_rng(int(rand_seed))

        intr_df = self._load_intrinsic_df()

        idx_all = np.arange(len(intr_df))
        if self.shuffle_intrinsic:
            rng.shuffle(idx_all)
        if self.max_intrinsic is not None:
            idx_all = idx_all[: int(self.max_intrinsic)]

        idx_core = idx_all[core::self.ncore]

        res_p = self._result_dir(now)
        date = f"{now.year}_{now.month}_{now.day}"
        save_p = res_p / f"{self.pattern}_{date}_core{core}.pickle"

        hit_df = pd.DataFrame([])
        niter = 0
        st = time()

        for intrinsic_id in idx_core:
            intrinsic = intr_df.loc[int(intrinsic_id)].to_dict()

            # print("[DEBUG] self.U_I_fixed =", self.U_I_fixed, type(self.U_I_fixed))
            model = models.NET_SAN_STP(
            NE=self.NE, NI=self.NI,
            net_seed=self.net_seed_base + int(intrinsic_id),
            p_ee=self.p_ee, p_ei=self.p_ei, p_ie=self.p_ie, p_ii=self.p_ii,
            dt_ms=self.dt_ms,
            U_I_fixed=self.U_I_fixed,
        )
            model.set_intrinsic_fixed(intrinsic)
            model.set_U_I(self.U_I_fixed)

            for _trial in range(self.syn_trials_per_intrinsic):
                niter += 1
                syn = model.gen_syn_params()
                model.set_syn_params(syn)

                cv_vals: List[float] = []
                rate_vals: List[float] = []
                cv0_vals: List[float] = []
                cv_hi_thr_vals: List[float] = []
                cv_lo_thr_vals: List[float] = []

                sync_corr_vals: List[float] = []
                n_valid_sync_vals: List[int] = []

                cv_hi_hit_vals: List[bool] = []
                cv_lo_hit_vals: List[bool] = []
                corr_hi_hit_vals: List[bool] = []
                corr_lo_hit_vals: List[bool] = []

                hi_hit: List[bool] = []
                lo_hit: List[bool] = []
                valid: List[bool] = []

                for u in self.U_list:
                    # Sweep excitatory presynaptic release-use parameter only.
                    model.set_U_E(u)

                    # Keep inhibitory presynaptic release-use parameter fixed.
                    model.set_U_I(self.U_I_fixed)

                    try:
                        v_mat, info = model.run_odeint(samp_freq=1000, samp_len=self.T_ms)
                        dt_rec_ms = float(info.get("dt_ms", self.dt_ms)) * float(info.get("record_every", 1))

                        cv, rate, sync_corr, n_valid_sync = evaluate_population_cv_and_sync_corr(
                            v_mat=v_mat,
                            dt_ms=dt_rec_ms,
                            NE=self.NE,
                            eval_window_ms=self.eval_window_ms,
                            thr_mV=self.thr_mV,
                            cv_w_ms=self.cv_w_ms,
                            moving_ms=self.moving_ms,
                            sync_bin_ms=self.sync_bin_ms,
                            sync_binary=self.sync_binary,
                            cv_on_population=self.cv_on_population,
                            sync_on_population=self.sync_on_population,
                        )
                    except Exception:
                        traceback.print_exc()
                        cv, rate, sync_corr, n_valid_sync = np.nan, 0.0, np.nan, 0

                    cv_n_sel = self.NE if self.cv_on_population == "E" else (self.NE + self.NI)
                    cv0 = cv0_from_rate(rate_hz=rate, NE=cv_n_sel, cv_w_ms=self.cv_w_ms)

                    cv_hi_thr = self.sync_factor * cv0 if np.isfinite(cv0) else np.nan
                    cv_lo_thr = self.desync_factor * cv0 if np.isfinite(cv0) else np.nan

                    cv_hi_hit = bool(np.isfinite(cv) and np.isfinite(cv_hi_thr) and (cv >= cv_hi_thr))
                    cv_lo_hit = bool(np.isfinite(cv) and np.isfinite(cv_lo_thr) and (cv <= cv_lo_thr))

                    corr_hi_hit = bool(np.isfinite(sync_corr) and (sync_corr >= self.sync_corr_hi))
                    corr_lo_hit = bool(np.isfinite(sync_corr) and (sync_corr <= self.sync_corr_lo))

                    # 同期/脱同期は CV と sync_corr の両方を満たす必要あり
                    hi = bool(cv_hi_hit and corr_hi_hit)
                    lo = bool(cv_lo_hit and corr_lo_hit)

                    # valid は CV0 と sync_corr が有限で、発火率も十分、相関に使えるニューロン数も2以上
                    v_ok = bool(
                        np.isfinite(cv)
                        and np.isfinite(cv0)
                        and np.isfinite(sync_corr)
                        and (rate >= self.min_rate_hz)
                        and (n_valid_sync >= 2)
                    )

                    cv_vals.append(cv)
                    rate_vals.append(rate)
                    cv0_vals.append(cv0)
                    cv_hi_thr_vals.append(cv_hi_thr)
                    cv_lo_thr_vals.append(cv_lo_thr)

                    sync_corr_vals.append(sync_corr)
                    n_valid_sync_vals.append(int(n_valid_sync))

                    cv_hi_hit_vals.append(cv_hi_hit)
                    cv_lo_hit_vals.append(cv_lo_hit)
                    corr_hi_hit_vals.append(corr_hi_hit)
                    corr_lo_hit_vals.append(corr_lo_hit)

                    hi_hit.append(hi)
                    lo_hit.append(lo)
                    valid.append(v_ok)

                seg = find_decreasing_segment_with_dynamic_thresholds(
                    U_list=self.U_list,
                    cv_list=cv_vals,
                    valid_list=valid,
                    hi_hit_list=hi_hit,
                    lo_hit_list=lo_hit,
                    decrease_eps=self.decrease_eps,
                )

                if seg is not None:
                    row: Dict[str, Any] = {}

                    # intrinsic
                    for k in ["g_leak", "g_kvhh", "g_cav", "g_kca", "g_nap", "t_ca"]:
                        row[k] = float(intrinsic[k])

                    # syn
                    for k in syn.keys():
                        row[k] = float(syn[k])

                    row["intrinsic_id"] = int(intrinsic_id)
                    row["core"] = int(core)

                    # segment info
                    row.update(seg)
                    i0 = int(seg["start_idx"])
                    i1 = int(seg["end_idx"])
                    row["U_I_fixed"] = float(self.U_I_fixed)

                    # More explicit aliases for the swept parameter
                    row["U_E_start"] = float(self.U_list[i0])
                    row["U_E_end"] = float(self.U_list[i1])

                    seg_sync = np.asarray(sync_corr_vals[i0:i1+1], dtype=float)
                    row["seg_max_sync_corr"] = float(np.nanmax(seg_sync)) if seg_sync.size > 0 else np.nan
                    row["seg_min_sync_corr"] = float(np.nanmin(seg_sync)) if seg_sync.size > 0 else np.nan

                    row["cv_on_population"] = str(self.cv_on_population)
                    row["sync_corr_hi"] = float(self.sync_corr_hi)
                    row["sync_corr_lo"] = float(self.sync_corr_lo)
                    row["sync_bin_ms"] = float(self.sync_bin_ms)
                    row["sync_binary"] = bool(self.sync_binary)
                    row["sync_on_population"] = str(self.sync_on_population)

                    # store per-U metrics
                    for (
                        u, cv, r, cv0, cv_hi_thr, cv_lo_thr,
                        sc, nvs,
                        cv_hi, cv_lo, corr_hi, corr_lo,
                        hh, ll, vv
                    ) in zip(
                        self.U_list,
                        cv_vals, rate_vals, cv0_vals, cv_hi_thr_vals, cv_lo_thr_vals,
                        sync_corr_vals, n_valid_sync_vals,
                        cv_hi_hit_vals, cv_lo_hit_vals, corr_hi_hit_vals, corr_lo_hit_vals,
                        hi_hit, lo_hit, valid
                    ):
                        tag = self._u_tag(u)
                        row[f"cv_u{tag}"] = float(cv) if np.isfinite(cv) else np.nan
                        row[f"rate_u{tag}"] = float(r)
                        row[f"cv0_u{tag}"] = float(cv0) if np.isfinite(cv0) else np.nan
                        row[f"cv_hi_thr_u{tag}"] = float(cv_hi_thr) if np.isfinite(cv_hi_thr) else np.nan
                        row[f"cv_lo_thr_u{tag}"] = float(cv_lo_thr) if np.isfinite(cv_lo_thr) else np.nan

                        row[f"sync_corr_u{tag}"] = float(sc) if np.isfinite(sc) else np.nan
                        row[f"n_valid_sync_u{tag}"] = int(nvs)

                        row[f"cv_hi_hit_u{tag}"] = bool(cv_hi)
                        row[f"cv_lo_hit_u{tag}"] = bool(cv_lo)
                        row[f"corr_hi_hit_u{tag}"] = bool(corr_hi)
                        row[f"corr_lo_hit_u{tag}"] = bool(corr_lo)

                        row[f"hi_hit_u{tag}"] = bool(hh)
                        row[f"lo_hit_u{tag}"] = bool(ll)
                        row[f"valid_u{tag}"] = bool(vv)

                    hit_df = pd.concat([hit_df, pd.DataFrame([row])], ignore_index=True)
                    print(
                        f"[core {core}] HIT: intrinsic_id={intrinsic_id}, "
                        f"seg=({row['U_start']}->{row['U_end']})"
                    )
                    break  # intrinsicあたり1ヒットで次へ

                # hourly save
                md = time()
                if (md - st) > 60 * 60:
                    st = time()
                    with open(str(save_p), "wb") as f:
                        pickle.dump(niter, f)
                        pickle.dump(hit_df, f)
                    print(datetime.now(), f"[core {core}] saved hits={len(hit_df)}")

                # stop after hr
                if (md - time_start) > 60 * 60 * self.hr:
                    print(f"[core {core}] {self.hr} hours passed -> terminate.")
                    with open(str(save_p), "wb") as f:
                        pickle.dump(niter, f)
                        pickle.dump(hit_df, f)
                    return

            # intrinsic loop checkpoint save
            md = time()
            if (md - st) > 60 * 60:
                st = time()
                with open(str(save_p), "wb") as f:
                    pickle.dump(niter, f)
                    pickle.dump(hit_df, f)
                print(datetime.now(), f"[core {core}] saved hits={len(hit_df)}")

            if (md - time_start) > 60 * 60 * self.hr:
                print(f"[core {core}] {self.hr} hours passed -> terminate.")
                with open(str(save_p), "wb") as f:
                    pickle.dump(niter, f)
                    pickle.dump(hit_df, f)
                return

        with open(str(save_p), "wb") as f:
            pickle.dump(niter, f)
            pickle.dump(hit_df, f)
        print(f"[core {core}] finished. total hits={len(hit_df)}")

    def multi_singleprocess(self) -> None:
        now = datetime.now()
        time_start = time()

        print(
            f"Syn-only U-list search (dynamic CV0 + sync_corr): "
            f"ncore={self.ncore}, hr={self.hr}, U_list={self.U_list}, "
            f"cv_on_population={self.cv_on_population}, "
            f"sync_on_population={self.sync_on_population}"
        )

        if self.ncore == 1:
            rand_seed = int(np.random.randint(0, 2**16 - 1))
            self.singleprocess((0, now, float(time_start), rand_seed))
            return

        args: List[Tuple[int, datetime, float, int]] = []
        for core in range(self.ncore):
            args.append((int(core), now, float(time_start), int(np.random.randint(0, 2**16 - 1))))

        with Pool(processes=self.ncore) as pool:
            pool.map(self.singleprocess, args)
# =============================================================================
# Syn-only search from SAN intrinsic (U-list version, bifurcation Excel source)
# =============================================================================

class SynOnlySearchFromSANIntrinsic_UList_bifur(SynOnlySearchFromSANIntrinsic_UList):
    """
    bifurcation_Total.xlsx で抽出したパラメータから intrinsic を作り、
    intrinsic固定で NET_SAN_STP の syn探索 + CV(U)単調減少判定。

    ※親クラス (SynOnlySearchFromSANIntrinsic_UList) の判定ロジックは同じ。
    """

    def __init__(
        self,
        bifur_fol: str,
        g_str: str,
        U_list: List[float],
        orders: Optional[List[int]] = None,
        pattern: str = "U_TRANS",
        fname: Optional[str] = None,
        ncore: int = 8,
        hr: int = 24,
        NE: int = 16,
        NI: int = 4,
        dt_ms: float = 0.1,
        T_ms: float = 15000.0,
        net_seed_base: int = 0,
        syn_trials_per_intrinsic: int = 200,
        max_intrinsic: Optional[int] = None,
        shuffle_intrinsic: bool = True,

        p_ee: float = 0.2, p_ei: float = 0.2, p_ie: float = 0.3, p_ii: float = 0.3,

        eval_window_ms: Tuple[float, float] = (2000.0, 12000.0),
        thr_mV: float = -20.0,
        cv_w_ms: float = 50.0,
        moving_ms: float = 50.0,

        sync_hi: float = 0.25,
        sync_lo: float = 0.10,
        decrease_eps: float = 0.01,
        min_rate_hz: float = 0.5,
    ) -> None:
        self.bifur_fol = str(bifur_fol)
        self.g_str = str(g_str)
        self.orders = orders

        # 親の intrinsic_sources は使わない（Excelから読む）
        super().__init__(
            intrinsic_sources=[],
            U_list=U_list,
            pattern=pattern,
            fname=fname or "bifur",
            ncore=ncore,
            hr=hr,
            NE=NE,
            NI=NI,
            dt_ms=dt_ms,
            T_ms=T_ms,
            net_seed_base=net_seed_base,
            syn_trials_per_intrinsic=syn_trials_per_intrinsic,
            max_intrinsic=max_intrinsic,
            shuffle_intrinsic=shuffle_intrinsic,
            p_ee=p_ee, p_ei=p_ei, p_ie=p_ie, p_ii=p_ii,
            eval_window_ms=eval_window_ms,
            thr_mV=thr_mV,
            cv_w_ms=cv_w_ms,
            moving_ms=moving_ms,
            sync_hi=sync_hi,
            sync_lo=sync_lo,
            decrease_eps=decrease_eps,
            min_rate_hz=min_rate_hz,
        )

    def _load_intrinsic_df(self) -> pd.DataFrame:
        """
        results_bifurcation/<bifur_fol>/<bifur_fol>_<g_str>/bifurcation_Total.xlsx
        から、条件でフィルタした行に対応する RandomParamSearch の pickle を開いて抽出。
        """
        bifur_fol_b = "results_bifurcation"
        date = "_".join(self.bifur_fol.split("_")[0:4])

        # RandomParamSearch 側の保存場所（あなたの元コード互換）
        base_fname = "results/SWS_params"
        path = self.bifur_fol + "/SWS_" + date  # 例: <bifur_fol>/SWS_YYYY_M_D

        res_path = f"{bifur_fol_b}/{self.bifur_fol}/{self.bifur_fol}_{self.g_str}"
        xlsx_path = f"{res_path}/bifurcation_Total.xlsx"

        df_bifur = pd.read_excel(xlsx_path)
        df_bifur = df_bifur[
            (df_bifur["wake_count"] > 0) &
            (df_bifur["sleep_to_wake"] == 2) &
            (df_bifur["vmin"] == 1) &
            (df_bifur["exclude"] == 0)
        ]

        cps = []
        for i in df_bifur["file"]:
            s = str(i).replace("core", "").replace("param", "")
            cps.append((int(s.split("_")[0]), int(s.split("_")[1])))

        dfs = []
        for n, (core_id, param_id) in enumerate(cps):
            if self.orders is not None and (n not in self.orders):
                continue

            fpath = f"{base_fname}/{path}_{core_id}.pickle"
            with open(fpath, "rb") as f:
                _iters = pickle.load(f)
                params = pickle.load(f)

            pd_param = params.iloc[int(param_id), 0:]
            dfs.append(pd_param.T)

        if len(dfs) == 0:
            raise RuntimeError("No intrinsic rows extracted from bifurcation excel / pickles.")

        df = pd.concat(dfs, axis=1, ignore_index=True).T

        # 必須列チェック（SAN intrinsic）
        need = ["g_leak", "g_kvhh", "g_cav", "g_kca", "g_nap", "g_ampar", "g_gabar", "t_ca"]
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise KeyError(f"Intrinsic DF missing columns: {missing}")

        df = df.dropna(subset=need).reset_index(drop=True)
        return df

    def _result_dir(self, now: datetime) -> Path:
        date = f"{now.year}_{now.month}_{now.day}"
        p = Path.cwd()
        res_p = p / "results" / f"{self.pattern}_params" / f"{date}_NET_SAN_STP_SYNONLY_ULIST_{self.fname}"
        res_p.mkdir(parents=True, exist_ok=True)
        return res_p


# =============================================================================
# Syn-only search from AN intrinsic (U-list, CV-based)
# =============================================================================

class SynOnlySearchFromANIntrinsic_UList:
    """
    事前にANで集めた intrinsic（dict 1個）を固定して、
    NET_AN_STP_NMDA で syn (g_ampar, tau_noise, sigma_noise) だけ探索。

    U_list の各 U で CV を評価し、CVがUとともに単調減少する区間が
    (cv>=cv_sync_hi と cv<=cv_desync_lo を両方含む)なら hit として保存。
    """

    def __init__(
        self,
        intrinsic: Dict[str, float],
        U_list: List[float],
        pattern: str = "U_TRANS_AN",
        fname: Optional[str] = None,
        ncore: int = 8,
        hr: int = 24,

        NE: int = 16,
        NI: int = 4,
        dt_ms: float = 0.1,
        T_ms: float = 8000.0,
        net_seed_base: int = 0,
        syn_trials_per_intrinsic: int = 50,

        # connection probs
        p_ee: float = 0.2, p_ei: float = 0.2, p_ie: float = 0.3, p_ii: float = 0.3,

        # evaluation
        eval_window_ms: Tuple[float, float] = (2000.0, 7000.0),
        thr_mV: float = -20.0,
        cv_w_ms: float = 50.0,
        moving_ms: float = 50.0,
        min_rate_hz: float = 0.5,

        # CV thresholds
        sync_hi: float = 0.20,   # cv_sync_hi
        sync_lo: float = 0.07,   # cv_desync_lo
        decrease_eps: float = 0.01,
    ) -> None:

        self.intrinsic = dict(intrinsic)
        self.U_list = [float(u) for u in U_list]

        self.pattern = pattern
        self.fname = fname or "AN"
        self.ncore = int(ncore)
        self.hr = int(hr)

        self.NE = int(NE)
        self.NI = int(NI)
        self.dt_ms = float(dt_ms)
        self.T_ms = float(T_ms)
        self.net_seed_base = int(net_seed_base)

        self.syn_trials_per_intrinsic = int(syn_trials_per_intrinsic)

        self.p_ee, self.p_ei, self.p_ie, self.p_ii = float(p_ee), float(p_ei), float(p_ie), float(p_ii)

        self.eval_window_ms = tuple(eval_window_ms)
        self.thr_mV = float(thr_mV)
        self.cv_w_ms = float(cv_w_ms)
        self.moving_ms = float(moving_ms)
        self.min_rate_hz = float(min_rate_hz)

        self.cv_sync_hi = float(sync_hi)
        self.cv_desync_lo = float(sync_lo)
        self.decrease_eps = float(decrease_eps)

    def _result_dir(self, now: datetime) -> Path:
        date = f"{now.year}_{now.month}_{now.day}"
        p = Path.cwd()
        res_p = p / "results" / f"{self.pattern}_params" / f"{date}_NET_AN_STP_ULIST_{self.fname}"
        res_p.mkdir(parents=True, exist_ok=True)
        return res_p

    @staticmethod
    def _u_tag(u: float) -> str:
        return f"{u}".replace(".", "p")

    def singleprocess(self, args: Tuple[int, datetime, float, int, Dict[str, float]]) -> None:
        core, now, time_start, rand_seed, intrinsic = args
        _ = np.random.default_rng(int(rand_seed))

        res_p = self._result_dir(now)
        date = f"{now.year}_{now.month}_{now.day}"
        save_p = res_p / f"{self.pattern}_{date}_core{core}.pickle"

        hit_df = pd.DataFrame([])
        niter = 0
        st = time()

        model = models.NET_AN_STP_NMDA(
            NE=self.NE, NI=self.NI,
            net_seed=self.net_seed_base,
            p_ee=self.p_ee, p_ei=self.p_ei, p_ie=self.p_ie, p_ii=self.p_ii,
            dt_ms=self.dt_ms,
        )
        model.set_intrinsic_fixed(intrinsic)
        model.set_U_I(self.U_I_fixed)

        for _trial in range(self.syn_trials_per_intrinsic):
            niter += 1

            syn = model.gen_syn_params()
            model.set_syn_params(syn)

            cv_vals: List[float] = []
            rate_vals: List[float] = []
            valid: List[bool] = []

            for u in self.U_list:
                model.set_U(u)
                try:
                    v_mat, info = model.run_odeint(samp_freq=1000, samp_len=int(self.T_ms), dt_ms=self.dt_ms)

                    # dtはt_msから推定（安全）
                    t_ms = info.get("t_ms", None)
                    if t_ms is not None and len(t_ms) >= 2:
                        dt_rec = float(np.median(np.diff(t_ms)))
                    else:
                        dt_rec = float(self.dt_ms)

                    cv, rate = evaluate_population_cv(
                        v_mat=v_mat,
                        dt_ms=dt_rec,
                        NE=self.NE,
                        eval_window_ms=self.eval_window_ms,
                        thr_mV=self.thr_mV,
                        cv_w_ms=self.cv_w_ms,
                        moving_ms=self.moving_ms,
                    )
                except Exception:
                    traceback.print_exc()
                    cv, rate = np.nan, 0.0

                cv_vals.append(cv)
                rate_vals.append(rate)
                valid.append(bool(np.isfinite(cv) and (rate >= self.min_rate_hz)))

            try:
                seg = find_decreasing_segment_with_cv_sync_desync(
                    U_list=self.U_list,
                    cv_list=cv_vals,
                    valid_list=valid,
                    cv_sync_hi=self.cv_sync_hi,
                    cv_desync_lo=self.cv_desync_lo,
                    decrease_eps=self.decrease_eps,
                )
            except Exception:
                traceback.print_exc()
                seg = None

            if seg is not None:
                row: Dict[str, Any] = {}

                # intrinsic（モデル側が持つキー集合を優先）
                if hasattr(models.NET_AN_STP_NMDA, "INTRINSIC_KEYS"):
                    for k in models.NET_AN_STP_NMDA.INTRINSIC_KEYS:
                        row[k] = float(intrinsic[k])
                else:
                    for k, v in intrinsic.items():
                        row[k] = float(v)

                # syn（探索対象）
                row["g_ampar"] = float(syn["g_ampar"])
                row["tau_noise"] = float(syn["tau_noise"])
                row["sigma_noise"] = float(syn["sigma_noise"])

                row["core"] = int(core)

                row.update(seg)
                row["U_start"] = float(self.U_list[int(seg["start_idx"])])
                row["U_end"]   = float(self.U_list[int(seg["end_idx"])])

                for u, c, r, v in zip(self.U_list, cv_vals, rate_vals, valid):
                    tag = self._u_tag(u)
                    row[f"cv_u{tag}"] = float(c) if np.isfinite(c) else np.nan
                    row[f"rate_u{tag}"] = float(r)
                    row[f"valid_u{tag}"] = bool(v)

                hit_df = pd.concat([hit_df, pd.DataFrame([row])], ignore_index=True)
                print(f"[core {core}] HIT: seg=({row['U_start']}->{row['U_end']})")
                break

            md = time()
            if (md - st) > 60 * 60:
                st = time()
                with open(str(save_p), "wb") as f:
                    pickle.dump(niter, f)
                    pickle.dump(hit_df, f)
                print(datetime.now(), f"[core {core}] saved hits={len(hit_df)}")

            if (md - time_start) > 60 * 60 * self.hr:
                print(f"[core {core}] {self.hr} hours passed -> terminate.")
                with open(str(save_p), "wb") as f:
                    pickle.dump(niter, f)
                    pickle.dump(hit_df, f)
                return

        with open(str(save_p), "wb") as f:
            pickle.dump(niter, f)
            pickle.dump(hit_df, f)
        print(f"[core {core}] finished. total hits={len(hit_df)}")

    def multi_singleprocess(self) -> None:
        now = datetime.now()
        time_start = time()

        args: List[Tuple[int, datetime, float, int, Dict[str, float]]] = []
        for core in range(self.ncore):
            args.append((int(core), now, float(time_start), int(np.random.randint(0, 2**16 - 1)), self.intrinsic))

        print(f"AN syn-only U-list search (CV): ncore={self.ncore}, hr={self.hr}, U_list={self.U_list}")
        with Pool(processes=self.ncore) as pool:
            pool.map(self.singleprocess, args)