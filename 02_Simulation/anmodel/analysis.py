# -*- coding: utf-8 -*-

""" 
analysis.py
This is the analysis module for Averaged Neuron (AN) model. In this module, 
you can analyze firing patterns from AN model, mainly using frequency and
spike analysis. 
"""

__author__ = 'Fumiya Tatsuki, Kensuke Yoshida, Tetsuya Yamada, \
              Takahiro Katsumata, Shoi Shi, Hiroki R. Ueda'
__status__ = 'Published'
__version__ = '1.0.0'
__date__ = '15 May 2020'

# import Dict
from typing import Dict, Iterator, List, Optional, Union
import os
import sys
"""
LIMIT THE NUMBER OF THREADS!
change local env variables BEFORE importing numpy
"""
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from enum import Flag, auto
import numpy as np
from scipy.signal import periodogram
from scipy import signal
from scipy.signal import find_peaks, welch, peak_widths

import os
import gc
import sys


from enum import Flag, auto
import numpy as np
from scipy.signal import periodogram, find_peaks
from scipy import signal
from typing import Optional
import matplotlib.pyplot as plt
from multiprocessing import Pool, Array, Manager
import traceback
# import models
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


class WavePattern(Flag):
    """ Enumeration class that distinguish different wave pattern.
    """
    SWS = auto()
    SWS_FEW_SPIKES = auto()
    AWAKE = auto()
    RESTING = auto()
    EXCLUDED = auto()
    ERROR = auto()
    SWS_HIGH_FR= auto()
    AWAKE_HIGH_FR= auto()


class WaveCheck:
    """ Check which wave pattern the neuronal firing belong to.

    Parameters
    ----------
    samp_freq : int
        sampling frequency of neuronal recordings (Hz)
    
    Attributes
    ----------
    wave_patters : WavePattern
        choices of wave pattern: enumeration objects
    samp_freq : int
        sampling frequency of neuronal recordings (Hz)
    freq_spike : FreqSpike
        contains helper functions for analyzing firing pattern
        using those frequency and spikes
    """
    def __init__(self, samp_freq: int=1000) -> None:
        self.wave_pattern = WavePattern
        self.samp_freq = samp_freq
        self.freq_spike = FreqSpike(samp_freq=samp_freq)
    
    def pattern(self, v: np.ndarray, T, dt) -> WavePattern:
        """

        Parameters
        ----------
        v : np.ndarray
            membrane potential over time

        Returns
        ----------
        WavePattern
            which wave pattern `v` belong to 
        """
        if np.any(np.isinf(v)) or np.any(np.isnan(v)):
            return self.wave_pattern.EXCLUDED
        detv: np.ndarray = signal.detrend(v)
        max_potential: float = max(detv)
        f: np.ndarray  # Array of sample frequencies
        spw: np.ndarray  # Array of power spectral density or power spectrum
        f, spw = periodogram(detv, fs=1/dt*1000)
        maxamp: float = max(spw)
        nummax: int = spw.tolist().index(maxamp)
        maxfre: float = f[nummax]
#         plt.plot(v)
#         plt.show()
     
        numfire: float = self.freq_spike.count_spike(v)
        # if pr ==True:
        #     print("maxfre", maxfre)
        #     print("numfire", numfire)

        ot = float(T/1000)  #sec
        
        if np.min(v) > -30 or  np.max(v) >200:
            return self.wave_pattern.EXCLUDED
        elif (maxfre < 0.1) or (numfire < 0.1*ot) :
            return self.wave_pattern.RESTING
        elif (0.1 < maxfre) and (numfire > ot*4*maxfre):
            if numfire < 30*ot :
                return self.wave_pattern.SWS
            elif numfire >= 30*ot and np.min(v)<-30 and numfire < 100*ot:
                return self.wave_pattern.SWS_HIGH_FR
            else:
                return self.wave_pattern.EXCLUDED
#         elif (0.5 < maxfre < 25) and (numfire <= 1.5*ot*maxfre):
#             return self.wave_pattern.SWS_FEW_SPIKES
        
        elif (0.1 < maxfre) and numfire <= ot * 2*maxfre:  # and numfire > ot * 0.5*maxfre  # (np.sum(v < -70)/len(v))*100 < 0.1:
           
                if numfire < 30*ot :
                    return self.wave_pattern.AWAKE
                elif numfire >= 30*ot:
                    return self.wave_pattern.AWAKE_HIGH_FR
                else:
                    return self.wave_pattern.EXCLUDED

                        # return self.wave_pattern.SWS_FEW_SPIKES
             
        else:
            return self.wave_pattern.EXCLUDED


class FreqSpike:
    """ 

    Parameters
    ----------
    samp_freq : int
        sampling frequency of neuronal recordings (Hz)

    Attributes
    ----------
    samp_freq : int
        sampling frequency of neuronal recordings (Hz)
    """
    def __init__(self, samp_freq: int) -> None:
        self.samp_freq = samp_freq

    def count_spike(self, v: np.ndarray) -> int:
        """ Count how many times a neuron fired.

        If neuron traverse -20 mV in a very short time range (1ms), 
        traverse count is added 1. Here, spike count is calculated as 
        traverse count // 2. 

        Parameter
        ---------
        v : np.ndarray
            membrane potential of a neuron
        
        Return
        ---------
        int
            spike count
        """
        ntraverse: int = 0
        ms: int =  1 # int(self.samp_freq / 1000)
        for i in range(len(v)-1):
            if (v[i]+20) * (v[i+ms]+20) < 0:
                ntraverse += 1
        nspike: int = int(ntraverse//2)
        return nspike

    


import numpy as np
from typing import Dict, Tuple

def detect_spikes_upcross(v: np.ndarray, thresh: float = -20.0) -> np.ndarray:
    v0 = v[:-1]
    v1 = v[1:]
    return np.where((v0 < thresh) & (v1 >= thresh))[0] + 1

def kuramoto_order_from_v(
    v_mat: np.ndarray,
    t_ms: np.ndarray,
    thresh: float = -20.0,
) -> Dict[str, np.ndarray]:
    T, N = v_mat.shape
    spike_times = []
    for i in range(N):
        sp_idx = detect_spikes_upcross(v_mat[:, i], thresh=thresh)
        spike_times.append(t_ms[sp_idx])

    phase = np.full((T, N), np.nan, dtype=float)
    for i in range(N):
        st = spike_times[i]
        if len(st) < 2:
            continue
        k = 0
        for ti in range(T):
            tt = t_ms[ti]
            while (k + 1 < len(st)) and (tt >= st[k + 1]):
                k += 1
                if k + 1 >= len(st):
                    break
            if k + 1 >= len(st):
                break
            t0, t1 = st[k], st[k + 1]
            if t1 > t0:
                phase[ti, i] = 2.0 * np.pi * (tt - t0) / (t1 - t0)

    R_t = np.full(T, np.nan, dtype=float)
    for ti in range(T):
        ph = phase[ti, :]
        m = ~np.isnan(ph)
        if m.sum() < 2:
            continue
        z = np.exp(1j * ph[m]).mean()
        R_t[ti] = np.abs(z)

    return {"R_t": R_t, "R_mean": np.nanmean(R_t)}

class NetTransitionCheck:
    """
    Early window: high synchrony (R >= R_sync)
    Late window:  low synchrony (R <= R_desync)
    """
    def __init__(
        self,
        early_ms: Tuple[float, float] = (3000.0, 8000.0),
        late_ms: Tuple[float, float] = (14000.0, 19000.0),
        R_sync: float = 0.70,
        R_desync: float = 0.45,
        spike_thresh: float = -20.0,
    ) -> None:
        self.early_ms = early_ms
        self.late_ms = late_ms
        self.R_sync = R_sync
        self.R_desync = R_desync
        self.spike_thresh = spike_thresh

    def evaluate(self, v_mat: np.ndarray, t_ms: np.ndarray) -> Dict[str, float]:
        sync = kuramoto_order_from_v(v_mat, t_ms, thresh=self.spike_thresh)
        R_t = sync["R_t"]

        e0, e1 = self.early_ms
        l0, l1 = self.late_ms

        mE = (t_ms >= e0) & (t_ms <= e1)
        mL = (t_ms >= l0) & (t_ms <= l1)

        R_early = float(np.nanmean(R_t[mE])) if np.any(mE) else float("nan")
        R_late  = float(np.nanmean(R_t[mL])) if np.any(mL) else float("nan")

        ok = (R_early >= self.R_sync) and (R_late <= self.R_desync)
        return {"ok": float(ok), "R_early": R_early, "R_late": R_late}



fs=25
fs_l=18

def _as_neuron_time(spts, N):
    """
    spts を (N, T) 形状に揃える。
    - すでに (N,T) ならそのまま
    - (T,N) なら転置
    """
    spts = np.asarray(spts)
    if spts.ndim != 2:
        raise ValueError(f"spts must be 2D. got shape={spts.shape}")

    if spts.shape[0] == N:
        return spts
    if spts.shape[1] == N:
        return spts.T

    raise ValueError(f"spts shape {spts.shape} doesn't match N={N} on any axis.")


def pop_total_rate_and_max_for_group(
    spts, dt_ms, group_mask,
    window_ms=100.0, step_ms=1,
    t_start_ms=0.0, t_end_ms=None
):
    """
    group_mask で指定した集団の population rate (total: spikes/s) を計算し、
    時間方向の最大値(max)とその時刻も返す。

    Returns:
      t_center_ms: (K,) 各窓の中心時刻 [ms]
      rate_total:  (K,) population rate total [spikes/s]
      max_total:        最大値 [spikes/s]
      t_peak_ms:        最大時刻 [ms]
      n_group:          集団ニューロン数
    """
    group_mask = np.asarray(group_mask, dtype=bool)
    N = group_mask.size
    n_group = int(group_mask.sum())
    if n_group == 0:
        raise ValueError("group_mask has no True entries (empty group).")

    spts = _as_neuron_time(spts, N)  # (N,T)
    _, T = spts.shape

    # 解析区間（ms -> index）
    start_idx = int(np.round(t_start_ms / dt_ms))
    end_idx = int(np.round(t_end_ms / dt_ms)) if t_end_ms is not None else T
    start_idx = max(0, min(T, start_idx))
    end_idx   = max(0, min(T, end_idx))
    if end_idx <= start_idx:
        raise ValueError("t_end_ms must be > t_start_ms (after rounding to dt).")

    # window/step（ms -> steps）
    w = int(np.round(window_ms / dt_ms))
    step = int(np.round(step_ms / dt_ms))
    if w <= 0:
        raise ValueError("window_ms is too small for dt_ms (w<=0).")
    if step <= 0:
        raise ValueError("step_ms is too small for dt_ms (step<=0).")

    # 集団の「各 dt 内のスパイク総数」
    pop_spikes = spts[group_mask, start_idx:end_idx].sum(axis=0).astype(np.int64)  # (L,)
    L = pop_spikes.size
    if L < w:
        raise ValueError(f"segment too short: L={L} steps, window={w} steps")

    # スライディング窓で「窓内スパイク総数」を高速計算（cumsum）
    cs = np.concatenate(([0], np.cumsum(pop_spikes, dtype=np.int64)))  # (L+1,)
    counts = cs[w:] - cs[:-w]  # (L-w+1,)

    # step ごとに間引く
    counts = counts[::step]
    k0 = np.arange(counts.size, dtype=np.int64) * step  # 窓開始位置（解析区間内）

    # 窓中心時刻
    t_center_ms = (start_idx + k0 + 0.5 * w) * dt_ms

    # total rate: spikes/s
    window_s = window_ms * 1e-3
    rate_total = counts / window_s

    # max
    i_peak = int(np.argmax(rate_total))
    max_total = float(rate_total[i_peak])
    t_peak_ms = float(t_center_ms[i_peak])

    return t_center_ms, rate_total, max_total, t_peak_ms, n_group

def population_rate_hz_from_spts(
    spts: np.ndarray,
    *,
    N: int,
    dt_ms: float,
    bin_ms: float = 5.0,
    smooth_ms: float = 0.0,
    neuron_mask: np.ndarray
):
    """
    spts (N,T) or (T,N) のスパイク0/1から集団平均発火率[Hz]を作る。

    Returns
    -------
    t_s : (nbins,) seconds
    rate_hz : (nbins,) Hz  （1ニューロンあたりの平均発火率）
    """
    sptsNT = _as_neuron_time(spts, N)  # (N,T)
    if neuron_mask is None:
        sel = slice(None)
        n_sel = sptsNT.shape[0]
    else:
        neuron_mask = np.asarray(neuron_mask, dtype=bool)
        if neuron_mask.shape != (N,):
            raise ValueError(f"neuron_mask must have shape ({N},), got {neuron_mask.shape}")
        sel = neuron_mask
        n_sel = int(neuron_mask.sum())
        if n_sel == 0:
            raise ValueError("neuron_mask selects 0 neurons")

    X = sptsNT[sel].astype(np.float32)  # (n_sel, T)

    # ビニング
    bin_steps = int(np.round(bin_ms / dt_ms))
    bin_steps = max(bin_steps, 1)
    T_steps = X.shape[1]
    T_trim = (T_steps // bin_steps) * bin_steps
    if T_trim < bin_steps * 10:
        raise ValueError("Too short data for given bin_ms; increase T or decrease bin_ms.")

    X = X[:, :T_trim]
    nbins = T_trim // bin_steps

    # (n_sel, nbins, bin_steps) にして bin 内スパイクを合計
    Xb = X.reshape(n_sel, nbins, bin_steps)
    spikes_per_bin = Xb.sum(axis=(0, 2))  # (nbins,)

    bin_s = (bin_steps * dt_ms) * 1e-3
    rate_hz = spikes_per_bin / n_sel / bin_s  # 1ニューロンあたりのHz

    # 時間（bin中心）を seconds で
    t_s = (np.arange(nbins) + 0.5) * bin_s

    # 平滑化（移動平均）
    if smooth_ms and smooth_ms > 0.0:
        smooth_bins = int(np.round(smooth_ms / bin_ms))
        smooth_bins = max(smooth_bins, 1)
        if smooth_bins > 1:
            k = np.ones(smooth_bins, dtype=np.float32) / smooth_bins
            rate_hz = np.convolve(rate_hz, k, mode="same")

    return t_s, rate_hz

# ====== 発振検出（厳格） ======
def per_cycle_amp_and_periods(t, f, last_seconds: float):
    t0 = max(0.0, t[-1] - last_seconds)
    mask = t >= t0
    tt = t[mask]; ff = f[mask]
    if ff.size < 200:
        return np.nan, [], 0.0, ff  # amp_med, periods, mean_rate, windowed f

    mean_rate = float(ff.mean())
    d = np.diff(ff)
    peak_idx = np.where((d[:-1] > 0) & (d[1:] <= 0))[0] + 1
    if peak_idx.size < 4:
        return np.nan, [], mean_rate, ff

    # 閾値で弱いピークを除外
    amp_global = float(ff.max() - ff.min())
    thr = float(ff.min() + 0.3 * amp_global)
    peak_idx = peak_idx[ff[peak_idx] >= thr]
    if peak_idx.size < 3:
        return np.nan, [], mean_rate, ff

    amps = []
    periods = []
    for k in range(peak_idx.size - 1):
        i0, i1 = peak_idx[k], peak_idx[k+1]
        if i1 <= i0 + 2:
            continue
        seg = ff[i0:i1+1]
        amps.append(float(seg.max() - seg.min()))
        periods.append(float(tt[i1] - tt[i0]))

    amp_med = float(np.median(amps)) if amps else np.nan
    return amp_med, periods, mean_rate, ff

def detect_oscillation_and_freq(t, f, cfg):
    # 時間窓・基本指標
    amp_med, periods, mean_rate, ff = per_cycle_amp_and_periods(t, f, cfg["last_seconds"])
    if not np.isfinite(amp_med) or len(periods) < cfg["min_peaks"] - 1:
        return False, np.nan

    # 周期安定性
    periods = np.array(periods, dtype=float)
    periods = periods[periods > 1e-6]
    if periods.size < cfg["min_peaks"] - 1:
        return False, np.nan
    cv = float(np.std(periods) / (np.mean(periods) + 1e-12))

    # しきい値（絶対＆相対）
    rel_amp = amp_med / max(mean_rate, 1e-6)

    # Welch PSD によるSNRとQ
    dt = float(np.median(np.diff(t)))
    fs = 1.0 / dt
    nper = int(min(len(ff), max(256, int(fs * 4.0))))  # ~4秒セグメント
    f_psd, Pxx = welch(ff - ff.mean(), fs=fs, nperseg=nper, noverlap=nper//2,
                       detrend="constant", scaling="density")
    band = (f_psd >= cfg["fmin"]) & (f_psd <= cfg["fmax"])
    if not np.any(band):
        return False, np.nan
    f_psd_b = f_psd[band]; Pxx_b = Pxx[band]
    if Pxx_b.size < 8:
        return False, np.nan

    # 支配ピーク
    pk_idx, _ = find_peaks(Pxx_b)
    if pk_idx.size == 0:
        return False, np.nan
    dom = pk_idx[np.argmax(Pxx_b[pk_idx])]
    f_peak = float(f_psd_b[dom])

    # SNR[dB]
    med_noise = float(np.median(Pxx_b[np.setdiff1d(np.arange(Pxx_b.size), [dom])]))
    med_noise = max(med_noise, 1e-20)
    snr_db = 10.0 * np.log10(Pxx_b[dom] / med_noise)

    # Q = f_peak / FWHM
    widths, _, _, _ = peak_widths(Pxx_b, [dom], rel_height=0.5)
    df = float(f_psd_b[1] - f_psd_b[0])
    fwhm = float(widths[0] * df) if widths.size > 0 else np.inf
    Q = f_peak / max(fwhm, 1e-12)

    # 時間領域の周波数（頑健な中央値）
    freq_time = 1.0 / np.median(periods)
    freq = float(freq_time) if np.isfinite(freq_time) else float(f_peak)

    # 判定
    is_osc = (
        (amp_med >= cfg["amp_thr_abs"]) and
        (rel_amp >= cfg["amp_thr_rel"]) and
        (len(periods) + 1 >= cfg["min_peaks"]) and
        (cv <= cfg["cv_p2p_thr"]) and
        (snr_db >= cfg["snr_db_thr"]) and
        (Q >= cfg["q_min"]) and
        (cfg["fmin"] <= freq <= cfg["fmax"])
    )
    return bool(is_osc), (freq if is_osc else np.nan)


def get_detection_config(profile: str):
    if profile == "very_strict":
        return dict(
            last_seconds=12.0,   # 末尾ウィンドウ
            min_peaks=10,        # 必要ピーク数
            amp_thr_abs=4.0,     # 絶対振幅 [Hz]
            amp_thr_rel=0.35,    # 相対振幅 (=振幅/平均発火率)
            snr_db_thr=15.0,     # Welch SNR[dB]
            cv_p2p_thr=0.25,     # 周期の変動係数
            fmin=1.0, fmax=80.0, # 周波数帯域 [Hz]
            q_min=3.0            # Q= f_peak / FWHM の最小
        )
    if profile == "strict":
        return dict(
            last_seconds=10.0,
            min_peaks=3,
            amp_thr_abs=0.5,
            amp_thr_rel=0.10,
            snr_db_thr=5.0,
            cv_p2p_thr=0.10,
            fmin=0.005, fmax=20.0, # 帯域（必要に応じて調整）
            q_min=1.0
        )
    # ★追加：緩い
    if profile == "loose":
        return dict(
            last_seconds=7.0,     # ★5〜12秒だけ見たいなら 7.0 推奨（末尾7秒）
            min_peaks=2,          # ★少なく
            amp_thr_abs=0.3,      # ★下げる（Hz）
            amp_thr_rel=0.05,     # ★下げる
            snr_db_thr=0.0,       # ★下げる（0dBでもOK扱い）
            cv_p2p_thr=0.60,      # ★周期の乱れ許容を増やす
            fmin=0.5, fmax=80.0,  # ★帯域は広め（必要なら元のままでもOK）
            q_min=0.7             # ★ピークの鋭さ要件を緩める
        )

    # ★追加：かなり緩い（誤検出は増えます）
    if profile == "very_loose":
        return dict(
            last_seconds=7.0,
            min_peaks=2,
            amp_thr_abs=0.2,
            amp_thr_rel=0.03,
            snr_db_thr=-5.0,      # ★SNRが負でも通す（ほぼ無視に近い）
            cv_p2p_thr=1.00,      # ★かなり雑でもOK
            fmin=0.5, fmax=80.0,
            q_min=0.5
        )
    # default
    return dict(
        last_seconds=8.0,
        min_peaks=6,
        amp_thr_abs=1.5,
        amp_thr_rel=0.15,
        snr_db_thr=6.0,
        cv_p2p_thr=0.45,
        fmin=0.5, fmax=80.0,
        q_min=2.0
    )
class eval_wave:
    def __init__(self,model, date_i,NE,NI,T, Tp,  drc, fig_dir,exs):
        manager = Manager()
        
        self.date_i = date_i
        self.N = NE + NI
        self.NE = NE
        self.NI = NI
        self.exs = exs
        
        self.tbs = int(T/Tp)
        
        
        self.T = T
        self.Tp=Tp
        dt = 0.01
        # self.con_log = con_log
        self.drc = drc
        
        # self.fig_dir = fig_dir
        # self.x = np.arange(0, T, dt)/1000
        
#         self.df_np = np.zeros((len(exs),13))
        self.df_np = manager.Array('d', range(len(exs)*16))

        
        # try:
        #     os.makedirs(fig_dir+"/"+model+"/")
        # except FileExistsError:
        #     print("{} is already exist".format(fig_dir+"/"+model+"/"))
            
        # if con_log == "True":
        self.savef = fig_dir
        # else:
        #     self.savef = fig_dir+"/{}/{}_NE{}_NI{}_conth{}_{}_{}_{}_gnap{}_gabac{}_T{}".format(model, self.date_i, self.NE,self.NI,self.con_ex_ex,self.con_ex_in,self.con_in_ex, self.con_in_in,self.g_nap_SD,self.gabac,self.T)
    
    
    def graph_saving(self, exs, pdf_save=False, graph_show=False):
        if pdf_save ==True:
            pdf = PdfPages(self.savef+".pdf")
            
        for num, ex in enumerate(exs):
            if type(ex) != str:
                ex = round(ex,2)
                # ex_in = round((ex), 2)
            # else:
                # ex_in = float(ex)
                # ex_in = f"{ex_in:,.2f}"
            print("exp", ex)

            # print("param_i",i)
#             print("NE", self.NE)
#             print("NI", self.NI)
#             print("g_nap_SD", self.g_nap_SD)
#             print("gaba_c", self.gabac)
           
#             print("con_M", self.con_M)
#             print("con_M_in", self.con_M_in)
#             print("con_ex_ex", self.con_ex_ex)
#             # print("con_ex_in", con_ex_in)
#             # print("con_in_ex", con_in_ex)
#             print("con_in_in", self.con_in_in)
            
            try:
                v_file = self.drc+ "Uex"+str(ex)+"_"+"N"+str(0)+"_"+str(0)+".bin" 
                f2= open(v_file, "rb")
                rectype = np.dtype(np.float64)
                v_tb = np.fromfile(f2, dtype=rectype)
                dt = round(self.Tp/v_tb.shape[0],4)
                print("ex {}, dt{}".format(ex, dt))
                self.x = np.arange(0, self.T, dt)/1000
            except:
                print("file_Error, ex{}".format(ex))
                traceback.print_exc()
#             sp_c_li = []
            if pdf_save ==True or graph_show ==True:
                for n in range(self.N):
                    try:
                        if self.N > 60 and self.N < 200:
                            if n%10 ==0:
                                for tb in range(self.tbs):
                                    v_file = self.drc+ "Uex"+str(ex)+"_"+"N"+str(n)+"_"+str(tb)+".bin" 
                                    f2= open(v_file, "rb")
                                    rectype = np.dtype(np.float64)
                                    v_tb = np.fromfile(f2, dtype=rectype).astype("float32")
                                    if tb==0:
                                        v=v_tb
                                    else:
                                        v=np.append(v, v_tb)
                                    f2.close()
                                if graph_show ==True:
                                    plt.figure(figsize=(15, 4))
                                    plt.title("Neuron{}".format(n))
                                    plt.plot(v)   #time[0:int(T/dt)-1],
                                    plt.ylim(-100, 40)
                                    plt.xlabel('Time (sec)',fontsize = fs)
                            #             plt.xlim(0,2)
                                        #plt.ylim(0,1)
                                    plt.xticks(fontsize = fs)
                                    plt.yticks(fontsize = fs)

                                    plt.show()
                                if pdf_save==True:
                                    fig, axes = plt.subplots(figsize=(20, 8))
                                    axes.set_title("param {}\n ex {}, Neuron{}".format(self.date_i, ex, n))
                                    axes.plot(v)   #time[0:int(T/dt)-1],
                                    axes.set_xlabel('Time (sec)')
                                    axes.set_ylim(-110,40)
                                    axes.set_rasterized(True)
                                    pdf.savefig(fig)
                                    plt.close()
                                    del v 
                                    gc.collect()
                                # print("max_fre", maxfre)
                                # print("sp_num", sp_c)
                                # print(pattern)
                        elif self.N > 200:
                            if n%200 ==0:
                                for tb in range(self.tbs):
                                    v_file = self.drc+ "Uex"+str(ex)+"_"+"N"+str(n)+"_"+str(tb)+".bin" 
                                    f2= open(v_file, "rb")
                                    rectype = np.dtype(np.float64)
                                    v_tb = np.fromfile(f2, dtype=rectype).astype("float32")
                                    if tb==0:
                                        v=v_tb
                                    else:
                                        v=np.append(v, v_tb)
                                    f2.close()
                                if graph_show ==True:
                                    plt.figure(figsize=(15, 4))
                                    plt.title("Neuron{}".format(n))
                                    plt.plot(v)   #time[0:int(T/dt)-1],
                                    plt.ylim(-100, 40)
                                    plt.xlabel('Time (sec)',fontsize = fs)
                            #             plt.xlim(0,2)
                                        #plt.ylim(0,1)
                                    plt.xticks(fontsize = fs)
                                    plt.yticks(fontsize = fs)

                                    plt.show()
                                if pdf_save==True:
                                    fig, axes = plt.subplots(figsize=(20, 8))
                                    axes.set_title("param {}\n ex {}, Neuron{}".format(self.date_i, ex, n))
                                    axes.plot(v)   #time[0:int(T/dt)-1],
                                    axes.set_xlabel('Time (sec)')
                                    axes.set_ylim(-110,40)
                                    axes.set_rasterized(True)
                                    pdf.savefig(fig)
                                    plt.close()
                                    del v 
                                    gc.collect()
                                # print("max_fre", maxfre)
                                # print("sp_num", sp_c)
                                # print(pattern)
                        else:
                            if n%3==0:
                                for tb in range(self.tbs):
                                    v_file = self.drc+ "Uex"+str(ex)+"_"+"N"+str(n)+"_"+str(tb)+".bin" 
                                    f2= open(v_file, "rb")
                                    rectype = np.dtype(np.float64)
                                    v_tb = np.fromfile(f2, dtype=rectype).astype("float32")
                                    if tb==0:
                                        v=v_tb
                                    else:
                                        v=np.append(v, v_tb)
                                    f2.close()
                                if graph_show ==True:
                                    plt.figure(figsize=(15, 4))
                                    plt.title("Neuron{}".format(n))
                                    plt.plot(v)   #time[0:int(T/dt)-1],
                                    plt.ylim(-100, 40)
                                    plt.xlabel('Time (sec)',fontsize = fs)
                            #             plt.xlim(0,2)
                                        #plt.ylim(0,1)
                                    plt.xticks(fontsize = fs)
                                    plt.yticks(fontsize = fs)

                                    plt.show()
                                if pdf_save==True:
                                    fig, axes = plt.subplots(figsize=(20, 8))
                                    axes.set_title("param {}\n ex {}, Neuron{}".format(self.date_i, ex, n))
                                    axes.plot(v)   #time[0:int(T/dt)-1],
                                    axes.set_xlabel('Time (sec)')
                                    axes.set_ylim(-110,40)
                                    axes.set_rasterized(True)
                                    pdf.savefig(fig)
                                    plt.close()   
                                    del v 
                                    gc.collect()     
                    except:
                        traceback.print_exc()
        if pdf_save ==True:
            pdf.close()
            print("pdf_saving_end")
        
        
    def multi_cal(self, cores, exs, cv_w, moving, v_off, ISI_M_r, counts, T_w="None", stat = False, raster=False):
        args = []
        if T_w == "None":
            T = self.T
        else:
            T = T_w
        
        if stat ==True:
            # dt = 0.01
            p=os.getcwd()  #/net_search
            try:
                for i in range(counts):
                    cv_dis=np.load(p+"/CV_random_dis/ISIr_{}_{}/N{}_T{}_dt{}_3/CV_w_{}.npy".format(ISI_M_r[0],ISI_M_r[1], self.N, T, 0.01, i)).astype('float32')
                    if i==0:
                        cv_dis_all = cv_dis
                    else:
                        cv_dis_all = np.concatenate([cv_dis_all, cv_dis]).astype('float32')
                
                self.cv_dis_all = np.array(sorted(cv_dis_all))  # small-> large
            except:
                traceback.print_exc()
                self.cv_dis_all = np.zeros(1)
        for num, ex in enumerate(exs):
            args.append((num, ex, T, cv_w, moving, v_off, stat, raster))
        print(f'analyze networks: using {cores} cores')
        with Pool(processes=cores) as pool:
            pool.map(self.calc_values, args)
            
    
    def calc_values(self, args):
        num, ex, T, cv_w, moving, v_off, stat, raster = args
        self.df_np[num*16]=float(ex)
        
        if type(ex) != str:
            ex = round(ex,2)
        #     ex_in = round((ex), 2)
        # else:
        #     ex_in = float(ex) 
        #     ex_in = f"{ex_in:,.2f}"
            
        try:
            v_file = self.drc+ "Uex"+str(ex)+"_"+"N"+str(0)+"_"+str(0)+".bin" 
            f2= open(v_file, "rb")
            rectype = np.dtype(np.float64)
            v_tb = np.fromfile(f2, dtype=rectype)
            dt = round(self.Tp/v_tb.shape[0],4)
        except:
            print("file_Error, ex{}".format(ex))
            traceback.print_exc()
            dt = 0.1
            
        flag=0
#             pattern_li = []
#             pattern_id=[]
        wake_c = 0
        sleep_c = 0
        wake_ex_c =0
        sleep_ex_c=0
        
        
        print("analyze exp", ex)
        
#             sp_c_li = []
        sp_fre_li = np.zeros(self.N)
        spts = np.zeros((self.N, int((T-v_off)/dt)), dtype = "int8")
        for n in range(self.N):
            try:
                for tb in range(self.tbs):
                    v_file = self.drc+ "Uex"+str(ex)+"_"+"N"+str(n)+"_"+str(tb)+".bin" 
                    f2= open(v_file, "rb")
                    rectype = np.dtype(np.float64)
                    v_tb = np.fromfile(f2, dtype=rectype).astype("float32")
                    if tb==0:
                        v=v_tb
                    else:
                        v=np.append(v, v_tb)
                    f2.close()

                pattern, maxfre, sp_c, peak_np = self.fre_spike(v[int(v_off/dt):int(T/dt)], int(T-v_off), dt)
                
                if np.any(np.isinf(v)) or np.any(np.isnan(v)):
                    flag = 1
                    print("nan_Error, can't make DF")
                    self.df_np[num*16+0]=1
#                     self.df_np[num][1:] = -1
                    for i in range(2,16):
#                         if i==0:
#                             continue
                        self.df_np[num*16 +i] = -1
                    break  #N
                
                if flag==1:
                    continue

                if pattern == "SWS" or pattern == "SWS_HIGH_FR":
                    sleep_c+=1
                    if n<self.NE:
                        sleep_ex_c+=1
                if pattern == "AWAKE" or pattern == "AWAKE_HIGH_FR":
                    wake_c+=1
                    if n<self.NE:
                        wake_ex_c+=1

#                     pattern_li.append(pattern)
                sp_fre_li[n]=(sp_c/(T-v_off)*1000)

                try:
                    # if maxfre!=0 and sp_c!=0:
                    #     if pattern!= "ERROR" and pattern != "EXCLUDED": 
                    #     # if pattern == "SWS" or pattern == "SWS_HIGH_FR" or pattern == "AWAKE" or pattern == "AWAKE_HIGH_FR" or pattern == "SWS_FEW_SPIKES" or pattern == "RES":
                    spts[n]= peak_np
        #                                                             print("{}, sp_add".format(n))
#                                     pattern_id.append(n)
                except:
                    print("file_Error, cant make DF")
                    traceback.print_exc()
                    flag = 1
                    self.df_np[num*16+1]=2
                    for i in range(2,16):
#                         if i==0:
#                             continue
                        self.df_np[num*16 +i] = -1
                    break  #N end
            except:
                print("file_Error, cant make DF")
                traceback.print_exc()
                flag=1
                self.df_np[num*16+1]=2
                for i in range(2,16):
#                         if i==0:
#                             continue
                        self.df_np[num*16 +i] = -1
                break  #N end
            
        if raster == True:
            plt.figure(figsize=(23, 15))
            Lt = int((T-v_off)/dt)
            x2=  np.arange(0, T-v_off, dt)
#                 spT = spts[spts.sum(axis=1) > 0., :]  #spikeあるものだけ選ぶ
            #color_set = ['r', 'b', 'k', 'orange', 'c']
            for i in range(spts.shape[0]):
                t_sp = x2[spts[i, :] > 0.5]   # spike times
                plt.plot(t_sp/1000, i*np.ones(len(t_sp)), '.',
                        ms=8, markeredgewidth=0.1, color ="black")
            plt.title("ex: {}".format(ex) ,fontsize = fs)
            plt.xlabel('Time (sec)',fontsize = fs)
            plt.ylabel('Neuron ID', fontsize = fs)
            #plt.xlim(0, 1)
            plt.xticks(fontsize = fs_l)
            plt.yticks(fontsize = fs_l)
            plt.show()
                
        if flag!=1 :
            m_sp_fre =np.mean(sp_fre_li)
            var_sp_fre =np.var(sp_fre_li)
            m_sp_fre_ex =np.mean(sp_fre_li[0:self.NE])
            var_sp_fre_ex =np.var(sp_fre_li[0:self.NE])
            sleep_per = sleep_c/(self.N)*100
            wake_per = wake_c/(self.N)*100
            sleep_per_ex = sleep_ex_c/(self.NE)*100
            wake_per_ex = wake_ex_c/(self.NE)*100

            self.df_np[num*16+1] = 0  #%nan
            self.df_np[num*16+2] = sleep_per
            self.df_np[num*16+3] = wake_per
            self.df_np[num*16+4] = m_sp_fre
            self.df_np[num*16+5] = var_sp_fre
            self.df_np[num*16+8] = sleep_per_ex
            self.df_np[num*16+9] = wake_per_ex
            self.df_np[num*16+10] = m_sp_fre_ex
            self.df_np[num*16+11] = var_sp_fre_ex
            # print("mean_fre_ex", m_sp_fre_ex)
            # print("var_sp_fre_ex", var_sp_fre_ex)
            # print("%sleep_ex", sleep_per_ex)
            # print("%wake_ex", wake_per_ex)

            sps_cv = self.calc_cv(self.N,spts, T-v_off, dt, cv_w,  moving)
#             print("CV_sp_count/w", sps_cv)
            self.df_np[num*16+6] = sps_cv
            sps_cv_ex = -1#self.calc_cv(self.NE,spts, T-v_off, dt, cv_w,  moving)
            # print("CV_sp_count_ex/w", sps_cv_ex)
            self.df_np[num*16+12] = -1#sps_cv_ex
            
            if stat ==True:
                p = self.get_cv_pv(sps_cv)
                # p_ex = self.get_cv_pv(sps_cv_ex)
                self.df_np[num*16+7] = p
                self.df_np[num*16+13] = -1#p_ex
            else:
                self.df_np[num*16+7] = -1
                self.df_np[num*16+13] = -1
            
            try:
                
                # 全ニューロンを集団として使う（必要ならEだけ: np.arange(N)<self.NE など）
                neuron_mask = np.ones(self.N, dtype=bool)

                # population rate (1 neuronあたりHz) を作る
                t_s, rate_hz = population_rate_hz_from_spts(
                    spts,
                    N=self.N,
                    dt_ms=dt,
                    bin_ms=5.0,        # 必要なら調整
                    smooth_ms=0.0,     # 必要なら調整（例: 10.0）
                    neuron_mask=neuron_mask
                )

                cfg = get_detection_config("very_loose")  # "loose" 等に変更可
                is_osc, freq = detect_oscillation_and_freq(t_s, rate_hz, cfg)

                self.df_np[num*16 + 14] = 1.0 if is_osc else 0.0
                self.df_np[num*16 + 15] = float(freq) if is_osc and np.isfinite(freq) else -1.0

            except Exception:
                # 発振判定だけ失敗しても他指標は生かす
                traceback.print_exc()
                self.df_np[num*16 + 14] = -1.0
                self.df_np[num*16 + 15] = -1.0


    def get_cv_pv(self, cv):
        if cv >= np.max(self.cv_dis_all):
            pv = 0
        else:
            minx = np.argmin(np.abs(self.cv_dis_all-cv))

            # low_p = minx/len(self.cv_dis_all)*100
            up_p = (len(self.cv_dis_all)-minx)/len(self.cv_dis_all)*100
            # if low_p<up_p:
            #     pv=low_p
            #     # print(low_p)
            # else:
            pv=up_p
                # print(up_p)
            print("p_value", pv)
        return pv
        
    def make_df(self):
        df_np = np.array(self.df_np).reshape(len(self.exs), 16)
#         exs = np.array(self.exs).reshape(-1,1)
        print(df_np.shape)
#         print(exs)
#         df_np = np.concatenate([exs, self.df_np], axis=1)
        df_cc = pd.DataFrame(df_np, columns=["exp","%nan","%sleep","%wake","mean_spike_fre","var_spike_fre",  "cv", "p", "%sleep_ex","%wake_ex","mean_spike_fre_ex","var_spike_fre_ex", "cv_ex", "p_ex","pop_osc", "pop_osc_freq"])
        # df_cc.to_excel(self.drc + "cc.xlsx")
        df_cc.to_excel(self.savef + "_cc2.xlsx")
        print(df_cc)
        nan_exs = df_cc[df_cc["%nan"]==1]["exp"]
        nan_exs = nan_exs.tolist()
        print(("nan_exp: {}\n".format(nan_exs)))
        error_exs = df_cc[df_cc["%nan"]==2]["exp"]
        error_exs = error_exs.tolist()
        print(("error_exp: {}\n".format(error_exs)))          

        return df_cc
            
            
    def calc_cv(self, N, spts, T, dt, cv_w,  moving):
        wc = int((T-cv_w)/moving+1)
        sp_li = []
        for m in range(wc):
            if m*moving/dt>0 and m*moving/dt + cv_w/dt  < (T)/dt:
                sp_sum = np.sum(spts[0:N,int(m*moving/dt):int(m*moving/dt + cv_w/dt)])
                sp_li.append(sp_sum)
        if np.mean(sp_li)!=0:
            sps_cv = np.std(sp_li)/np.mean(sp_li)
        else:
            sps_cv =0
        print("CV_sp_count/w", sps_cv)
        return sps_cv
    
    def fre_spike(self, v, T, dt):
    
        if np.any(np.isinf(v)) or np.any(np.isnan(v)):
            print("fre_spike: nan")
            return "Excluded",0, 0, np.zeros(int(T/dt), dtype="int8")
        else:
            wavech =WaveCheck(1/dt*1000)
            pattern = wavech.pattern(v, T, dt)
            detv: np.ndarray = signal.detrend(v)
            max_potential: float = max(detv)
            f: np.ndarray  # Array of sample frequencies
            spw: np.ndarray  # Array of power spectral density or power spectrum
            f, spw = periodogram(detv, fs=1/dt*1000)
            maxamp: float = max(spw)
            nummax: int = spw.tolist().index(maxamp)
            maxfre: float = f[nummax]

    #         ntraverse: int = 0
    #         ms = 1
    #         for k in range(len(v)-1):
    #             if k+ms < len(v)-1:
    #                 if (v[k]+20) * (v[k+ms]+20) < 0:
    #                     ntraverse += 1
    #         nspike= (ntraverse/2)/(T/(2*1000))/20
            peaks = find_peaks(v, height=-20)[0]
    #         print(peaks)
            peak_np = np.zeros(int(T/dt), dtype="int8")
            peak_np[peaks]=1
            return pattern.name, maxfre, len(peaks), peak_np
