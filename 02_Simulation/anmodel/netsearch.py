# -- coding: utf-8 --
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from time import time
from datetime import datetime

from network import SANVesicleNetwork
from analysis import kuramoto_order_from_v


@dataclass
class TransitionCriterion:
    # early window should be synchronous
    t_early0: float = 3000.0
    t_early1: float = 8000.0
    R_sync: float = 0.70

    # late window should be desynchronous
    t_late0: float = 14000.0
    t_late1: float = 19000.0
    R_desync: float = 0.45


class NetworkTransitionSearch:
    """
    Random search for parameter sets that show:
      early (low U) synchrony  -> late (high U) desynchrony
    """

    def __init__(
        self,
        out_dir: str = "results_net",
        NE: int = 16,
        NI: int = 4,
        seed: int = 0,
        hr: float = 2.0,
        T_ms: float = 20000.0,
        dt_ms: float = 0.1,
        record_dt_ms: float = 1.0,
        criterion: Optional[TransitionCriterion] = None,
    ) -> None:
        self.NE = NE
        self.NI = NI
        self.seed = seed
        self.hr = float(hr)

        self.T_ms = float(T_ms)
        self.dt_ms = float(dt_ms)
        self.record_every = int(np.round(record_dt_ms / dt_ms))

        self.criterion = criterion or TransitionCriterion()
        self.out_dir = Path(out_dir)

        self.rng = np.random.default_rng(seed)

        # fixed connectivity across trials (important for comparability)
        self.net = SANVesicleNetwork(NE=NE, NI=NI, seed=seed)

        self.out_dir.mkdir(parents=True, exist_ok=True)

    def gen_params(self) -> Dict:
        """
        Log-uniform sampling for synaptic parameters.
        (Intrinsic SAN is fixed to san_sws; you can extend later.)
        """
        def logu(a, b):
            # sample in [10^a, 10^b]
            x = self.rng.uniform(a, b)
            return 10.0 ** x

        p = {}
        p["g_ampar"] = logu(-2.0, -0.8)   # ~0.01..0.16
        p["g_gabar"] = logu(-1.5, -0.5)   # ~0.03..0.32
        p["tau_rec_e"] = logu(2.6, 3.1)   # ~400..1250 ms
        p["tau_rec_i"] = logu(2.5, 3.0)   # ~316..1000 ms

        # U ramp range: U0 small, U1 larger
        p["U0"] = self.rng.uniform(0.02, 0.10)
        p["U1"] = self.rng.uniform(0.25, 0.60)
        p["t0_ms"] = 2000.0
        p["t1_ms"] = 18000.0

        # noise
        p["tau_noise"] = self.rng.uniform(3.0, 10.0)
        p["sigma_noise"] = self.rng.uniform(0.0, 0.05)

        return p

    def evaluate(self, p: Dict) -> Dict:
        self.net.set_network_params(p)
        t_ms, v_rec = self.net.run_fixed_step(
            T_ms=self.T_ms, dt_ms=self.dt_ms,
            record_every=self.record_every, reset=True
        )

        sync = kuramoto_order_from_v(v_rec, t_ms)
        R_t = sync["R_t"]

        c = self.criterion
        m_early = (t_ms >= c.t_early0) & (t_ms <= c.t_early1)
        m_late  = (t_ms >= c.t_late0)  & (t_ms <= c.t_late1)

        R_early = float(np.nanmean(R_t[m_early]))
        R_late  = float(np.nanmean(R_t[m_late]))

        ok = (R_early >= c.R_sync) and (R_late <= c.R_desync)
        return {"ok": ok, "R_early": R_early, "R_late": R_late}

    def run(self) -> None:
        now = datetime.now()
        date = f"{now.year}_{now.month}_{now.day}"
        save_p = self.out_dir / f"transition_params_{date}.pickle"

        hits = []
        niter = 0
        t0 = time()
        last_save = time()

        while True:
            niter += 1
            p = self.gen_params()
            res = self.evaluate(p)

            if res["ok"]:
                row = dict(p)
                row.update(res)
                hits.append(row)
                print("HIT", niter, "R_early", res["R_early"], "R_late", res["R_late"])

            # save hourly
            if time() - last_save > 3600:
                last_save = time()
                df = pd.DataFrame(hits)
                with open(str(save_p), "wb") as f:
                    pickle.dump({"niter": niter, "hits": df}, f)
                print("saved:", save_p, "hits:", len(df))

            # stop after hr
            if time() - t0 > 3600 * self.hr:
                df = pd.DataFrame(hits)
                with open(str(save_p), "wb") as f:
                    pickle.dump({"niter": niter, "hits": df}, f)
                print("done:", save_p, "hits:", len(df))
                break
