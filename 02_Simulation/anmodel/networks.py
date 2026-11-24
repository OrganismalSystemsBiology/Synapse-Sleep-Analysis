# -- coding: utf-8 --
"""
networks.py
SAN network + AMPAR/GABAR + vesicle STP (Tsodyks-like) + OU noise
Designed to be instantiated inside multiprocessing worker safely.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np

import channels
import params


def rate_proxy(v: float) -> float:
    # same sigmoid as receptor f(V) style
    return 1.0 / (1.0 + np.exp(-(v - 20.0) / 2.0))


@dataclass
class UProfile:
    U0: float = 0.02
    U1: float = 0.98
    t0_ms: float = 2000.0
    t1_ms: float = 18000.0

    def __call__(self, t_ms: float) -> float:
        if t_ms <= self.t0_ms:
            return self.U0
        if t_ms >= self.t1_ms:
            return self.U1
        a = (t_ms - self.t0_ms) / (self.t1_ms - self.t0_ms)
        return self.U0 + a * (self.U1 - self.U0)

# -- coding: utf-8 --
"""
SAN network + AMPAR/GABAR + vesicle STP + OU noise
E/I-separated synaptic utilization version.

Important:
    U_E is used only for excitatory synapses.
    U_I is used only for inhibitory synapses.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any

import numpy as np

import channels
import params


def rate_proxy(v: float) -> float:
    """
    Same sigmoid as receptor f(V).
    This is used as a continuous presynaptic activity proxy.
    """
    return 1.0 / (1.0 + np.exp(-(v - 20.0) / 2.0))


def _scalar_float(x: Any, name: str = "value") -> float:
    """
    Convert scalar-like objects to float.

    This also accepts accidental one-element tuples/lists/arrays, e.g. (0.3,).
    """
    if isinstance(x, (tuple, list, np.ndarray)):
        if len(x) != 1:
            raise ValueError(f"{name} must be scalar or one-element sequence, got {x}")
        x = x[0]
    return float(x)


@dataclass
class UProfile:
    U0: float = 0.02
    U1: float = 0.98
    t0_ms: float = 2000.0
    t1_ms: float = 18000.0

    def __call__(self, t_ms: float) -> float:
        if t_ms <= self.t0_ms:
            return self.U0
        if t_ms >= self.t1_ms:
            return self.U1
        a = (t_ms - self.t0_ms) / (self.t1_ms - self.t0_ms)
        return self.U0 + a * (self.U1 - self.U0)


class SANVesicleNetwork:
    """
    NE excitatory + NI inhibitory neurons.

    Intrinsic model:
        Simplified averaged neuron model with Leak, KvHH, Cav, KCa, and NaP.

    Synapses:
        E -> all: AMPAR
        I -> all: GABAR

    Short-term synaptic depression:
        R_E and R_I are tracked separately.

        dR_E/dt = (1 - R_E) / tau_rec_e - U_E * R_E * rate_proxy(V_pre_E)
        dR_I/dt = (1 - R_I) / tau_rec_i - U_I * R_I * rate_proxy(V_pre_I)

    Postsynaptic drives:
        driveE = rE @ [W_E * U_E * R_E]
        driveI = rI @ [W_I * U_I * R_I]

    Here, U_E can be swept and U_I can be held fixed.
    """

    def __init__(
        self,
        NE: int = 16,
        NI: int = 4,
        seed: int = 0,
        p_ee: float = 0.2,
        p_ei: float = 0.2,
        p_ie: float = 0.3,
        p_ii: float = 0.3,
        allow_self: bool = False,
    ) -> None:
        self.NE = int(NE)
        self.NI = int(NI)
        self.N = self.NE + self.NI

        self.E_idx = np.arange(self.NE, dtype=int)
        self.I_idx = np.arange(self.NE, self.N, dtype=int)

        self.rng = np.random.default_rng(seed)
        self.const = params.Constants()

        # SAN intrinsic channels
        self.leak = channels.Leak()
        self.kvhh = channels.KvHH()
        self.cav = channels.Cav()
        self.kca = channels.KCa()
        self.nap = channels.NaP()

        # default SAN intrinsic parameters
        self.set_intrinsic_params(params.TypicalParam().san_sws_ampagaba)

        # synaptic conductances
        self.g_ampar = 0.04
        self.g_gabar = 0.12

        # STP recovery
        self.tau_rec_e = 800.0
        self.tau_rec_i = 400.0

        # E/I-separated utilization
        self.U_E = 0.1
        self.U_I = 0.3

        # backward-compatible alias
        self.U = self.U_E

        # old UProfile kept only for backward compatibility
        self.u_profile = UProfile(
            U0=self.U_E,
            U1=self.U_E,
            t0_ms=0.0,
            t1_ms=0.0,
        )

        # OU noise
        self.tau_noise = 5.0
        self.sigma_noise = 0.03

        # external current-like term
        self.I_app = np.zeros(self.N, dtype=float)

        # connectivity weights
        self.W_E = self._make_W_E(p_ee, p_ei, allow_self)
        self.W_I = self._make_W_I(p_ie, p_ii, allow_self)

        # synaptic resources
        self.R_E = np.ones((self.NE, self.N), dtype=float)
        self.R_I = np.ones((self.NI, self.N), dtype=float)

        self.reset_state()

    # ============================================================
    # U setters
    # ============================================================

    def set_U(self, U: float) -> None:
        """
        Backward-compatible setter.
        In this E/I-separated implementation, set_U changes only excitatory U.
        """
        self.set_U_E(U)

    def set_U_E(self, U_E: float) -> None:
        self.U_E = _scalar_float(U_E, "U_E")
        self.U = self.U_E
        if hasattr(self, "u_profile"):
            self.u_profile = UProfile(
                U0=self.U_E,
                U1=self.U_E,
                t0_ms=0.0,
                t1_ms=0.0,
            )

    def set_U_I(self, U_I: float) -> None:
        self.U_I = _scalar_float(U_I, "U_I")

    def set_U_pair(self, U_E: float, U_I: float) -> None:
        self.set_U_E(U_E)
        self.set_U_I(U_I)

    # ============================================================
    # connectivity
    # ============================================================

    def _make_W_E(self, p_ee: float, p_ei: float, allow_self: bool) -> np.ndarray:
        """
        E presynaptic neurons to all postsynaptic neurons.
        Shape: (NE, N), rows are presynaptic E neurons, columns are postsynaptic neurons.
        """
        W = np.zeros((self.NE, self.N), dtype=float)

        for j in range(self.NE):
            mask_ee = self.rng.random(self.NE) < p_ee
            if not allow_self:
                mask_ee[j] = False
            W[j, self.E_idx] = mask_ee.astype(float)

            mask_ei = self.rng.random(self.NI) < p_ei
            W[j, self.I_idx] = mask_ei.astype(float)

        indeg = W.sum(axis=0)
        indeg[indeg < 1.0] = 1.0
        W = W / indeg

        return W

    def _make_W_I(self, p_ie: float, p_ii: float, allow_self: bool) -> np.ndarray:
        """
        I presynaptic neurons to all postsynaptic neurons.
        Shape: (NI, N), rows are presynaptic I neurons, columns are postsynaptic neurons.
        """
        W = np.zeros((self.NI, self.N), dtype=float)

        for k in range(self.NI):
            mask_ie = self.rng.random(self.NE) < p_ie
            W[k, self.E_idx] = mask_ie.astype(float)

            mask_ii = self.rng.random(self.NI) < p_ii
            if not allow_self:
                mask_ii[k] = False
            W[k, self.I_idx] = mask_ii.astype(float)

        indeg = W.sum(axis=0)
        indeg[indeg < 1.0] = 1.0
        W = W / indeg

        return W

    # ============================================================
    # parameter setters
    # ============================================================

    def set_intrinsic_params(self, p: Dict) -> None:
        self.leak.set_g(float(p["g_leak"]))
        self.kvhh.set_g(float(p["g_kvhh"]))
        self.cav.set_g(float(p["g_cav"]))
        self.kca.set_g(float(p["g_kca"]))
        self.nap.set_g(float(p["g_nap"]))
        self.tau_ca = float(p["t_ca"])

    def set_network_params(self, p: Dict) -> None:
        """
        Network/synaptic parameters.

        U-related behavior:
            U_E or U changes excitatory utilization only.
            U_I or U_I_fixed changes inhibitory utilization.
        """
        if "g_ampar" in p:
            self.g_ampar = _scalar_float(p["g_ampar"], "g_ampar")

        if "g_gabar" in p:
            self.g_gabar = _scalar_float(p["g_gabar"], "g_gabar")

        if "tau_rec_e" in p:
            self.tau_rec_e = _scalar_float(p["tau_rec_e"], "tau_rec_e")

        if "tau_rec_i" in p:
            self.tau_rec_i = _scalar_float(p["tau_rec_i"], "tau_rec_i")

        # E utilization
        if "U_E" in p:
            self.set_U_E(p["U_E"])
        elif "U" in p:
            self.set_U_E(p["U"])

        # I utilization
        if "U_I" in p:
            self.set_U_I(p["U_I"])
        elif "U_I_fixed" in p:
            self.set_U_I(p["U_I_fixed"])

        # old U ramp support; now interpreted as excitatory U profile only
        if any(k in p for k in ("U0", "U1", "t0_ms", "t1_ms")):
            self.u_profile = UProfile(
                U0=_scalar_float(p.get("U0", self.u_profile.U0), "U0"),
                U1=_scalar_float(p.get("U1", self.u_profile.U1), "U1"),
                t0_ms=_scalar_float(p.get("t0_ms", self.u_profile.t0_ms), "t0_ms"),
                t1_ms=_scalar_float(p.get("t1_ms", self.u_profile.t1_ms), "t1_ms"),
            )
            # keep current U_E at U0 for fixed-step U sweep unless explicitly set later
            self.set_U_E(self.u_profile.U0)

        if "tau_noise" in p:
            self.tau_noise = _scalar_float(p["tau_noise"], "tau_noise")

        if "sigma_noise" in p:
            self.sigma_noise = _scalar_float(p["sigma_noise"], "sigma_noise")

        if "I_app" in p:
            arr = np.asarray(p["I_app"], dtype=float)
            if arr.shape == ():
                self.I_app[:] = float(arr)
            else:
                assert arr.shape == (self.N,)
                self.I_app = arr.copy()

    # ============================================================
    # state
    # ============================================================

    def reset_state(self) -> None:
        v0, n0, ca0 = self.const.san_ini

        self.v = np.full(self.N, v0, dtype=float)
        self.n = np.full(self.N, n0, dtype=float)
        self.ca = np.full(self.N, ca0, dtype=float)

        self.sA = np.zeros(self.N, dtype=float)
        self.sG = np.zeros(self.N, dtype=float)
        self.eta = np.zeros(self.N, dtype=float)

        self.R_E[:] = 1.0
        self.R_I[:] = 1.0

    # ============================================================
    # simulation step
    # ============================================================

    def step(self, t_ms: float, dt_ms: float) -> None:
        """
        One fixed-step Euler update.

        E synapses:
            use U_E

        I synapses:
            use U_I
        """
        U_E = float(getattr(self, "U_E", getattr(self, "U", 0.1)))
        U_I = float(getattr(self, "U_I", 0.3))

        rE = np.array([rate_proxy(self.v[j]) for j in self.E_idx], dtype=float)
        rI = np.array([rate_proxy(self.v[j]) for j in self.I_idx], dtype=float)

        # ------------------------------------------------------------
        # STP resource update
        # ------------------------------------------------------------
        self.R_E += dt_ms * (
            (1.0 - self.R_E) / self.tau_rec_e
            - (U_E * self.R_E) * rE[:, None]
        )

        self.R_I += dt_ms * (
            (1.0 - self.R_I) / self.tau_rec_i
            - (U_I * self.R_I) * rI[:, None]
        )

        np.clip(self.R_E, 0.0, 1.0, out=self.R_E)
        np.clip(self.R_I, 0.0, 1.0, out=self.R_I)

        # ------------------------------------------------------------
        # postsynaptic aggregate drive
        # ------------------------------------------------------------
        driveE = rE @ (self.W_E * (U_E * self.R_E))
        driveI = rI @ (self.W_I * (U_I * self.R_I))

        # ------------------------------------------------------------
        # receptor gating variables
        # ------------------------------------------------------------
        dsA = self.const.s_a_ampar * driveE - self.sA / self.const.s_tau_ampar
        dsG = self.const.s_a_gabar * driveI - self.sG / self.const.s_tau_gabar

        self.sA += dt_ms * dsA
        self.sG += dt_ms * dsG

        self.sA[self.sA < 0.0] = 0.0
        self.sG[self.sG < 0.0] = 0.0

        # ------------------------------------------------------------
        # OU noise
        # ------------------------------------------------------------
        if self.sigma_noise > 0.0:
            self.eta += (
                dt_ms * (-self.eta / self.tau_noise)
                + self.sigma_noise
                * np.sqrt(2.0 * dt_ms / self.tau_noise)
                * self.rng.standard_normal(self.N)
            )

        cm = self.const.cm
        area = self.const.area
        a_ca = self.const.a_ca

        for i in range(self.N):
            v = float(self.v[i])
            n = float(self.n[i])
            ca = float(self.ca[i])

            i_leak = self.leak.i(v)
            i_kvhh = self.kvhh.i(v, n=n)
            i_cav = self.cav.i(v)
            i_kca = self.kca.i(v, ca=ca)
            i_nap = self.nap.i(v)

            i_ampa = self.g_ampar * self.sA[i] * (v - self.const.vAMPAR)
            i_gaba = self.g_gabar * self.sG[i] * (v - self.const.vGABAR)

            dvdt = (
                (-10.0 * area * (i_leak + i_kvhh + i_cav + i_kca + i_nap))
                - (i_ampa + i_gaba)
                + self.I_app[i]
                + self.eta[i]
            ) / (10.0 * cm * area)

            dndt = self.kvhh.dndt(v=v, n=n)
            dCadt = -a_ca * (10.0 * area * i_cav) - ca / self.tau_ca

            self.v[i] = v + dt_ms * dvdt
            self.n[i] = n + dt_ms * dndt
            self.ca[i] = ca + dt_ms * dCadt

    def run_fixed_step(
        self,
        T_ms: float = 20000.0,
        dt_ms: float = 0.1,
        record_every: int = 10,
        reset: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if reset:
            self.reset_state()

        nstep = int(np.round(T_ms / dt_ms))

        t_rec = []
        v_rec = []

        for k in range(nstep):
            t = (k + 1) * dt_ms
            self.step(t_ms=t, dt_ms=dt_ms)

            if (k % record_every) == 0:
                t_rec.append(t)
                v_rec.append(self.v.copy())

        return np.asarray(t_rec, dtype=float), np.vstack(v_rec)
        
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np
import channels
import params

def rate_proxy(v: float) -> float:
    return 1.0 / (1.0 + np.exp(-(v - 20.0) / 2.0))


class ANVesicleNetwork:
    """
    NE excit + NI inhib network with AN intrinsic channels:
      Leak, NavHH, KvHH, KvA, KvSI, Cav, KCa, NaP, KIR
    Synapses:
      - E -> all: AMPAR + NMDAR (both driven by the same vesicle release)
      - I -> all: GABAR
    STP (depression only):
      dR/dt = (1-R)/tau_rec - U*R*rate_proxy(v_pre)
    OU noise:
      d eta = -(eta/tau_noise) dt + sigma * sqrt(2 dt / tau_noise) * N(0,1)
    State per neuron:
      v, h_nav, n_kvhh, h_kva, m_kvsi, ca, sA, sN, xN, sG, eta
    Plus per-connection resources:
      R_E (NE, N), R_I (NI, N)
    """

    def __init__(
        self,
        NE: int = 16,
        NI: int = 4,
        seed: int = 0,
        p_ee: float = 0.2,
        p_ei: float = 0.2,
        p_ie: float = 0.3,
        p_ii: float = 0.3,
        allow_self: bool = False,
    ) -> None:
        self.NE = int(NE)
        self.NI = int(NI)
        self.N = self.NE + self.NI
        self.E_idx = np.arange(self.NE, dtype=int)
        self.I_idx = np.arange(self.NE, self.N, dtype=int)

        self.rng = np.random.default_rng(seed)
        self.const = params.Constants()

        # intrinsic channels (AN)
        self.leak = channels.Leak()
        self.nav  = channels.NavHH()
        self.kvhh = channels.KvHH()
        self.kva  = channels.KvA()
        self.kvsi = channels.KvSI()
        self.cav  = channels.Cav()
        self.kca  = channels.KCa()
        self.nap  = channels.NaP()
        self.kir  = channels.KIR()

        # NMDAR object (for Mg-block kinetics xN + current)
        self.nmdar = channels.NMDAR()

        # default intrinsic: AN typical SWS (あれば)
        # ない場合は an_sws がある想定。なければ set_intrinsic_params を必ず呼ぶ運用でOK。
        try:
            self.set_intrinsic_params(params.TypicalParam().an_sws)
        except Exception:
            pass

        # syn params (g_gabar, g_nmdar は intrinsic 側で固定運用)
        self.g_ampar = 0.05   # search対象
        self.g_gabar = 0.10
        self.g_nmdar = 0.10

        # STP
        self.tau_rec_e = 800.0
        self.tau_rec_i = 500.0
        self.U = 0.1  # 固定U（U_list で変える）

        # noise
        self.tau_noise = 5.0
        self.sigma_noise = 0.03
        self.I_app = np.zeros(self.N, dtype=float)

        # connectivity
        self.W_E = self._make_W_E(p_ee, p_ei, allow_self)
        self.W_I = self._make_W_I(p_ie, p_ii, allow_self)

        # resources
        self.R_E = np.ones((self.NE, self.N), dtype=float)
        self.R_I = np.ones((self.NI, self.N), dtype=float)

        # syn gate kinetics constants（params.Constants に無い場合のフォールバック）
        self.s_a_ampar   = float(getattr(self.const, "s_a_ampar", 0.5))
        self.s_tau_ampar = float(getattr(self.const, "s_tau_ampar", 2.0))

        self.s_a_gabar   = float(getattr(self.const, "s_a_gabar", 0.5))
        self.s_tau_gabar = float(getattr(self.const, "s_tau_gabar", 5.0))

        self.s_a_nmdar   = float(getattr(self.const, "s_a_nmdar", 0.2))
        self.s_tau_nmdar = float(getattr(self.const, "s_tau_nmdar", 80.0))

        self.reset_state()

    def set_U(self, U: float) -> None:
        self.U = float(U)

    def _make_W_E(self, p_ee: float, p_ei: float, allow_self: bool) -> np.ndarray:
        W = np.zeros((self.NE, self.N), dtype=float)
        for j in range(self.NE):
            mask_ee = self.rng.random(self.NE) < p_ee
            if not allow_self:
                mask_ee[j] = False
            W[j, self.E_idx] = mask_ee.astype(float)

            mask_ei = self.rng.random(self.NI) < p_ei
            W[j, self.I_idx] = mask_ei.astype(float)

        indeg = W.sum(axis=0)
        indeg[indeg < 1.0] = 1.0
        W = W / indeg
        return W

    def _make_W_I(self, p_ie: float, p_ii: float, allow_self: bool) -> np.ndarray:
        W = np.zeros((self.NI, self.N), dtype=float)
        for k in range(self.NI):
            mask_ie = self.rng.random(self.NE) < p_ie
            W[k, self.E_idx] = mask_ie.astype(float)

            mask_ii = self.rng.random(self.NI) < p_ii
            if not allow_self:
                mask_ii[k] = False
            W[k, self.I_idx] = mask_ii.astype(float)

        indeg = W.sum(axis=0)
        indeg[indeg < 1.0] = 1.0
        W = W / indeg
        return W

    def set_intrinsic_params(self, p: Dict) -> None:
        """
        AN intrinsic を設定（g_ampar は固定しない運用：searchで上書きする）
        必須（固定対象）:
          g_leak,g_nav,g_kvhh,g_kva,g_kvsi,g_cav,g_kca,g_nap,g_kir,t_ca,g_nmdar,g_gabar
        """
        self.leak.set_g(p["g_leak"])
        self.nav.set_g(p["g_nav"])
        self.kvhh.set_g(p["g_kvhh"])
        self.kva.set_g(p["g_kva"])
        self.kvsi.set_g(p["g_kvsi"])
        self.cav.set_g(p["g_cav"])
        self.kca.set_g(p["g_kca"])
        self.nap.set_g(p["g_nap"])
        self.kir.set_g(p["g_kir"])

        self.tau_ca = float(p["t_ca"])

        # 固定: g_nmdar, g_gabar は intrinsic 側の値で固定
        self.g_nmdar = float(p["g_nmdar"])
        self.g_gabar = float(p["g_gabar"])

        # channels.NMDAR の内部 g を合わせておく（i() に使うため）
        try:
            self.nmdar.set_g(self.g_nmdar)
        except Exception:
            pass

    def set_network_params(self, p: Dict) -> None:
        # search対象: g_ampar, tau_noise, sigma_noise
        if "g_ampar" in p:
            self.g_ampar = float(p["g_ampar"])
        if "tau_noise" in p:
            self.tau_noise = float(p["tau_noise"])
        if "sigma_noise" in p:
            self.sigma_noise = float(p["sigma_noise"])

        # 固定で良いが、必要なら外から変更可能にしておく
        if "tau_rec_e" in p:
            self.tau_rec_e = float(p["tau_rec_e"])
        if "tau_rec_i" in p:
            self.tau_rec_i = float(p["tau_rec_i"])
        if "I_app" in p:
            arr = np.asarray(p["I_app"], dtype=float)
            if arr.shape == ():
                self.I_app[:] = float(arr)
            else:
                assert arr.shape == (self.N,)
                self.I_app = arr.copy()

    def reset_state(self) -> None:
        # params.Constants().an_ini の並び:
        # [v, h_nav, n_kvhh, h_kva, m_kvsi, s_ampar, x_nmdar, s_nmdar, s_gabar, ca]
        v0, h0, n0, hk0, m0, sA0, xN0, sN0, sG0, ca0 = self.const.an_ini

        self.v = np.full(self.N, v0, dtype=float)
        self.h_nav = np.full(self.N, h0, dtype=float)
        self.n_kvhh = np.full(self.N, n0, dtype=float)
        self.h_kva = np.full(self.N, hk0, dtype=float)
        self.m_kvsi = np.full(self.N, m0, dtype=float)
        self.ca = np.full(self.N, ca0, dtype=float)

        self.sA = np.full(self.N, sA0, dtype=float)  # AMPAR gate（driveで更新）
        self.xN = np.full(self.N, xN0, dtype=float)  # NMDAR Mg/dynamics
        self.sN = np.full(self.N, sN0, dtype=float)  # NMDAR syn gate（driveで更新）
        self.sG = np.full(self.N, sG0, dtype=float)  # GABAR gate（driveで更新）

        self.eta = np.zeros(self.N, dtype=float)

        self.R_E[:] = 1.0
        self.R_I[:] = 1.0

    def step(self, t_ms: float, dt_ms: float) -> None:
        U = float(self.U)

        rE = np.array([rate_proxy(self.v[j]) for j in self.E_idx], dtype=float)
        rI = np.array([rate_proxy(self.v[j]) for j in self.I_idx], dtype=float)

        # STP update (resource)
        self.R_E += dt_ms * ((1.0 - self.R_E) / self.tau_rec_e - (U * self.R_E) * rE[:, None])
        self.R_I += dt_ms * ((1.0 - self.R_I) / self.tau_rec_i - (U * self.R_I) * rI[:, None])
        np.clip(self.R_E, 0.0, 1.0, out=self.R_E)
        np.clip(self.R_I, 0.0, 1.0, out=self.R_I)

        # drives
        driveE = rE @ (self.W_E * (U * self.R_E))  # (N,)
        driveI = rI @ (self.W_I * (U * self.R_I))  # (N,)

        # syn gates (E release drives AMPAR + NMDAR, I release drives GABAR)
        self.sA += dt_ms * (self.s_a_ampar * driveE - self.sA / self.s_tau_ampar)
        self.sN += dt_ms * (self.s_a_nmdar * driveE - self.sN / self.s_tau_nmdar)
        self.sG += dt_ms * (self.s_a_gabar * driveI - self.sG / self.s_tau_gabar)
        self.sA[self.sA < 0.0] = 0.0
        self.sN[self.sN < 0.0] = 0.0
        self.sG[self.sG < 0.0] = 0.0

        # OU noise
        if self.sigma_noise > 0.0:
            self.eta += (
                dt_ms * (-self.eta / self.tau_noise)
                + self.sigma_noise * np.sqrt(2.0 * dt_ms / self.tau_noise) * self.rng.standard_normal(self.N)
            )

        cm = self.const.cm
        area = self.const.area
        a_ca = self.const.a_ca

        # reversal potentials（無い場合は channels 側のEを使うより const 側が安全）
        vAMPA = float(getattr(self.const, "vAMPAR", 0.0))
        vGABA = float(getattr(self.const, "vGABAR", -70.0))
        vNMDA = float(getattr(self.const, "vNMDAR", 0.0))

        for i in range(self.N):
            v = float(self.v[i])
            h = float(self.h_nav[i])
            n = float(self.n_kvhh[i])
            hk = float(self.h_kva[i])
            m = float(self.m_kvsi[i])
            ca = float(self.ca[i])
            xN = float(self.xN[i])

            # intrinsic currents
            i_leak = self.leak.i(v)
            i_nav  = self.nav.i(v, h=h)
            i_kvhh = self.kvhh.i(v, n=n)
            i_kva  = self.kva.i(v, h=hk)
            i_kvsi = self.kvsi.i(v, m=m)
            i_cav  = self.cav.i(v)
            i_kca  = self.kca.i(v, ca=ca)
            i_nap  = self.nap.i(v)
            i_kir  = self.kir.i(v)

            # syn currents
            i_ampa = self.g_ampar * self.sA[i] * (v - vAMPA)

            # NMDAR current with x-dynamics (Mg等が channels.NMDAR 内にある想定)
            try:
                i_nmda = self.nmdar.i(v, s=self.sN[i], x=xN)
            except Exception:
                # フォールバック（Mgなし線形）
                i_nmda = self.g_nmdar * self.sN[i] * (v - vNMDA)

            i_gaba = self.g_gabar * self.sG[i] * (v - vGABA)

            dvdt = (
                (-10.0 * area * (i_leak + i_nav + i_kvhh + i_kva + i_kvsi + i_cav + i_kca + i_nap + i_kir))
                - (i_ampa + i_nmda + i_gaba)
                + self.I_app[i]
                + self.eta[i]
            ) / (10.0 * cm * area)

            # gating derivatives
            dhdt  = self.nav.dhdt(v=v, h=h)
            dndt  = self.kvhh.dndt(v=v, n=n)
            dhkdt = self.kva.dhdt(v=v, h=hk)
            dmdt  = self.kvsi.dmdt(v=v, m=m)

            # NMDAR x-dynamics
            try:
                dxdt = self.nmdar.dxdt(v=v, x=xN)
            except Exception:
                dxdt = 0.0

            # Ca dynamics: Cav + NMDAR + pump
            dCadt = -a_ca * (10.0 * area * i_cav) - a_ca * i_nmda - ca / float(self.tau_ca)

            # update
            self.v[i] = v + dt_ms * dvdt
            self.h_nav[i] = h + dt_ms * dhdt
            self.n_kvhh[i] = n + dt_ms * dndt
            self.h_kva[i] = hk + dt_ms * dhkdt
            self.m_kvsi[i] = m + dt_ms * dmdt
            self.xN[i] = xN + dt_ms * dxdt
            self.ca[i] = ca + dt_ms * dCadt

    def run_fixed_step(
        self,
        T_ms: float = 20000.0,
        dt_ms: float = 0.1,
        record_every: int = 10,
        reset: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if reset:
            self.reset_state()

        nstep = int(np.round(T_ms / dt_ms))
        t_rec = []
        v_rec = []
        for k in range(nstep):
            t = (k + 1) * dt_ms
            self.step(t_ms=t, dt_ms=dt_ms)
            if (k % record_every) == 0:
                t_rec.append(t)
                v_rec.append(self.v.copy())
        return np.asarray(t_rec, dtype=float), np.vstack(v_rec)

