# -- coding: utf-8 --
"""
Network SAN + (AMPAR, GABAR) + vesicle dynamics (Tsodyks-type) + OU noise.

Designed to be compatible with existing anmodel.* modules:
- channels.py: channel/receptor objects
- params.py: constants + typical params
- analysis.py: can be extended for network synchrony measures
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

import channels
import params


def rate_proxy(v: float) -> float:
    """Same shape as channels.AMPAR.f / channels.GABAR.f (scalar)."""
    return 1.0 / (1.0 + np.exp(-(v - 20.0) / 2.0))


@dataclass
class UProfile:
    """Time-dependent U(t): ramp from U0 to U1 between [t0, t1] (ms)."""
    U0: float = 0.05
    U1: float = 0.45
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
    Intrinsic: SAN (Leak, KvHH, Cav, KCa, NaP)
    Synapses: AMPAR (E) + GABAR (I)
    Presyn drive uses rate_proxy(v_pre) for continuous-time integration.
    Vesicle depletion/recovery per connection (R matrix).
    Noise: OU current-like term eta added into dvdt numerator.

    State variables per neuron:
      v, n_kvhh, ca, sA (AMPA gate), sG (GABA gate), eta (OU noise)
    Plus synaptic resource matrices:
      R_E: (NE, N) for E->all
      R_I: (NI, N) for I->all
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

        # Intrinsic channels (SAN set)
        self.leak = channels.Leak()
        self.kvhh = channels.KvHH()
        self.cav = channels.Cav()
        self.kca = channels.KCa()
        self.nap = channels.NaP()

        # Receptors for current calculation (we keep their reversal potentials)
        self.ampar = channels.AMPAR()
        self.gabar = channels.GABAR()

        # Default: SAN typical SWS params
        self.set_intrinsic_params(params.TypicalParam().san_sws)

        # Network/synapse params (defaults)
        self.g_ampar = 0.04
        self.g_gabar = 0.12
        self.tau_rec_e = 800.0  # ms
        self.tau_rec_i = 500.0  # ms
        self.u_profile = UProfile()

        # OU noise
        self.tau_noise = 5.0    # ms
        self.sigma_noise = 0.03 # (arbitrary current units consistent w/ dvdt numerator)

        # External bias current (optional)
        self.I_app = np.zeros(self.N, dtype=float)

        # Build random connectivity (weights)
        self.W_E = self._make_W_E(p_ee, p_ei, allow_self=allow_self)
        self.W_I = self._make_W_I(p_ie, p_ii, allow_self=allow_self)

        # Synaptic resources
        self.R_E = np.ones((self.NE, self.N), dtype=float)
        self.R_I = np.ones((self.NI, self.N), dtype=float)

        # Neuron states
        self.reset_state()

    # ---------- connectivity ----------
    def _make_W_E(self, p_ee: float, p_ei: float, allow_self: bool) -> np.ndarray:
        """
        W_E shape: (NE, N). Row: presyn E neuron j, col: postsyn i.
        Two probabilities: E->E uses p_ee, E->I uses p_ei.
        """
        W = np.zeros((self.NE, self.N), dtype=float)
        # E targets
        for j in range(self.NE):
            # to E
            mask_ee = self.rng.random(self.NE) < p_ee
            if not allow_self:
                mask_ee[j] = False
            W[j, self.E_idx] = mask_ee.astype(float)
            # to I
            mask_ei = self.rng.random(self.NI) < p_ei
            W[j, self.I_idx] = mask_ei.astype(float)

        # Normalize roughly by expected indegree to stabilize scaling
        # (You can change this later to lognormal weights etc.)
        indeg = W.sum(axis=0)
        indeg[indeg < 1.0] = 1.0
        W = W / indeg  # postsyn-normalization
        return W

    def _make_W_I(self, p_ie: float, p_ii: float, allow_self: bool) -> np.ndarray:
        """
        W_I shape: (NI, N). Row: presyn I neuron k (0..NI-1), col: postsyn i.
        I->E uses p_ie, I->I uses p_ii.
        """
        W = np.zeros((self.NI, self.N), dtype=float)
        for k in range(self.NI):
            pre_global = self.I_idx[k]
            # to E
            mask_ie = self.rng.random(self.NE) < p_ie
            W[k, self.E_idx] = mask_ie.astype(float)
            # to I
            mask_ii = self.rng.random(self.NI) < p_ii
            if not allow_self:
                mask_ii[k] = False
            W[k, self.I_idx] = mask_ii.astype(float)

        indeg = W.sum(axis=0)
        indeg[indeg < 1.0] = 1.0
        W = W / indeg
        return W

    # ---------- parameter setters ----------
    def set_intrinsic_params(self, p: Dict) -> None:
        """SAN intrinsic parameter dict: g_leak, g_kvhh, g_cav, g_kca, g_nap, t_ca."""
        self.leak.set_g(p["g_leak"])
        self.kvhh.set_g(p["g_kvhh"])
        self.cav.set_g(p["g_cav"])
        self.kca.set_g(p["g_kca"])
        self.nap.set_g(p["g_nap"])
        self.tau_ca = p["t_ca"]

    def set_network_params(self, p: Dict) -> None:
        """
        Network/synapse/noise parameters.
        Expected keys (examples):
          g_ampar, g_gabar,
          tau_rec_e, tau_rec_i,
          U0, U1, t0_ms, t1_ms,
          tau_noise, sigma_noise
        """
        if "g_ampar" in p:
            self.g_ampar = float(p["g_ampar"])
        if "g_gabar" in p:
            self.g_gabar = float(p["g_gabar"])
        if "tau_rec_e" in p:
            self.tau_rec_e = float(p["tau_rec_e"])
        if "tau_rec_i" in p:
            self.tau_rec_i = float(p["tau_rec_i"])
        if any(k in p for k in ["U0", "U1", "t0_ms", "t1_ms"]):
            U0 = float(p.get("U0", self.u_profile.U0))
            U1 = float(p.get("U1", self.u_profile.U1))
            t0 = float(p.get("t0_ms", self.u_profile.t0_ms))
            t1 = float(p.get("t1_ms", self.u_profile.t1_ms))
            self.u_profile = UProfile(U0=U0, U1=U1, t0_ms=t0, t1_ms=t1)
        if "tau_noise" in p:
            self.tau_noise = float(p["tau_noise"])
        if "sigma_noise" in p:
            self.sigma_noise = float(p["sigma_noise"])
        if "I_app" in p:
            arr = np.asarray(p["I_app"], dtype=float)
            if arr.shape == ():
                self.I_app[:] = float(arr)
            else:
                assert arr.shape == (self.N,)
                self.I_app = arr.copy()

    # ---------- state ----------
    def reset_state(self) -> None:
        """Reset neuron states and synaptic resources."""
        # SAN initial values (from params.Constants.san_ini)
        v0, n0, ca0 = self.const.san_ini
        self.v = np.full(self.N, v0, dtype=float)
        self.n = np.full(self.N, n0, dtype=float)
        self.ca = np.full(self.N, ca0, dtype=float)

        # postsyn gating variables
        self.sA = np.zeros(self.N, dtype=float)
        self.sG = np.zeros(self.N, dtype=float)

        # OU noise state
        self.eta = np.zeros(self.N, dtype=float)

        # vesicle resources reset to full
        self.R_E[:] = 1.0
        self.R_I[:] = 1.0

    # ---------- one step integration ----------
    def step(self, t_ms: float, dt_ms: float) -> None:
        # current U(t) shared across synapses (you can extend to separate Ue/Ui)
        U = float(self.u_profile(t_ms))

        # presyn rate proxies
        rE = np.array([rate_proxy(self.v[j]) for j in self.E_idx], dtype=float)  # (NE,)
        rI = np.array([rate_proxy(self.v[j]) for j in self.I_idx], dtype=float)  # (NI,)

        # ---- vesicle resource update (vectorized) ----
        # dR = (1-R)/tau_rec - U*R*r_pre
        self.R_E += dt_ms * ((1.0 - self.R_E) / self.tau_rec_e - (U * self.R_E) * rE[:, None])
        self.R_I += dt_ms * ((1.0 - self.R_I) / self.tau_rec_i - (U * self.R_I) * rI[:, None])
        # keep in [0,1]
        np.clip(self.R_E, 0.0, 1.0, out=self.R_E)
        np.clip(self.R_I, 0.0, 1.0, out=self.R_I)

        # ---- postsyn drive ----
        # driveE_i = sum_j W_E[j,i] * (U*R_E[j,i]) * rE_j
        M_E = self.W_E * (U * self.R_E)
        driveE = rE @ M_E  # (N,)

        M_I = self.W_I * (U * self.R_I)
        driveI = rI @ M_I  # (N,)

        # ---- synaptic gating update ----
        # use same constants as receptor classes, but with network drive
        dsA = self.const.s_a_ampar * driveE - self.sA / self.const.s_tau_ampar
        dsG = self.const.s_a_gabar * driveI - self.sG / self.const.s_tau_gabar
        self.sA += dt_ms * dsA
        self.sG += dt_ms * dsG
        # avoid negative gates
        self.sA[self.sA < 0.0] = 0.0
        self.sG[self.sG < 0.0] = 0.0

        # ---- OU noise update ----
        if self.sigma_noise > 0.0:
            # eta += dt*(-eta/tau) + sigma*sqrt(2*dt/tau)*N(0,1)
            self.eta += dt_ms * (-self.eta / self.tau_noise) + self.sigma_noise * np.sqrt(
                2.0 * dt_ms / self.tau_noise
            ) * self.rng.standard_normal(self.N)

        # ---- neuron intrinsic updates (loop; N=20 so OK) ----
        cm = self.const.cm
        area = self.const.area
        a_ca = self.const.a_ca

        for i in range(self.N):
            v = float(self.v[i])
            n = float(self.n[i])
            ca = float(self.ca[i])

            # intrinsic currents (SAN)
            i_leak = self.leak.i(v)
            i_kvhh = self.kvhh.i(v, n=n)
            i_cav = self.cav.i(v)
            i_kca = self.kca.i(v, ca=ca)
            i_nap = self.nap.i(v)

            # syn currents (postsyn aggregate gates)
            # I = g * s * (v - Erev)
            i_ampa = self.g_ampar * self.sA[i] * (v - self.const.vAMPAR)
            i_gaba = self.g_gabar * self.sG[i] * (v - self.const.vGABAR)

            # dv/dt (match SAN/AN style: intrinsic part scaled by -10*area, syn part subtracted)
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

    # ---------- run ----------
    def run_fixed_step(
        self,
        T_ms: float = 20000.0,
        dt_ms: float = 0.1,
        record_every: int = 10,
        reset: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fixed-step integration.
        Returns:
          t_ms: (Trec,)
          v_rec: (Trec, N)
        record_every: store every k steps (so recorded dt = k*dt_ms)
        """
        if reset:
            self.reset_state()

        nstep = int(np.round(T_ms / dt_ms))
        rec = []
        t_rec = []
        for k in range(nstep):
            t = (k + 1) * dt_ms
            self.step(t_ms=t, dt_ms=dt_ms)
            if (k % record_every) == 0:
                rec.append(self.v.copy())
                t_rec.append(t)

        return np.array(t_rec, dtype=float), np.vstack(rec)
