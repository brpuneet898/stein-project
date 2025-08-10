# scripts/run_experiment.py
import time
import os
import argparse
import pandas as pd
import numpy as np
import arviz as az

from data_utils import load_covtype_100pc
from monitor import MemoryMonitor
from svgd_jax import run_svgd
from hmc_pymc import run_hmc

def compute_ess_for_array(samples, var_name="w"):
    """
    samples: if InferenceData, let arviz compute ESS.
    if numpy array shaped (n_samples, dim), convert to (chain, draw, dim) = (1, n_samples, dim).
    Return the minimal ESS across dimensions (bulk) and the full per-dim ESS array.
    """
    if hasattr(samples, "posterior") or hasattr(samples, "to_inference_data"):
        # an arviz InferenceData
        ess = az.ess(samples)
        # ess may be dict-like; return mean or min across vars
        return ess
    else:
        # numpy array (n_samples, dim)
        if samples.ndim == 2:
            n, dim = samples.shape
            arr = samples.reshape((1, n, dim))
            # convert to InferenceData via az.convert_to_inference_data?
            ds = az.convert_to_dataset(arr, dims={"chain": [0], "draw": list(range(n)), "w_dim": list(range(dim))})
            ess = az.ess(arr)
            return ess
        else:
            return az.ess(samples)

def run_and_log(results_csv="results.csv", subset=None, **kwargs):
    # prepare data
    X_train, X_test, y_train, y_test, scaler, pca = load_covtype_100pc(subset=subset)

    rows = []

    # --- SVGD ---
    mem = MemoryMonitor()
    mem.start()
    t0 = time.time()
    particles, duration = run_svgd(X_train, y_train, n_particles=kwargs.get("svgd_particles", 200),
                                   n_iter=kwargs.get("svgd_iters", 1000),
                                   step_size=kwargs.get("svgd_step", 5e-4),
                                   seed=kwargs.get("seed", 0))
    t1 = time.time()
    mem.stop()
    svgd_peak = mem.get_peak()
    svgd_time = t1 - t0  # overall wall-clock
    # compute ESS: treat particles as (chain=1, draws=particles.shape[0], dims)
    try:
        ess_svgd = az.ess(particles)  # will handle numpy input shape (chain, draw) expectations
    except Exception:
        # fallback: compute per-dim ess by calling az.ess on each dimension split into (1, draws)
        from collections import defaultdict
        ess_svgd = {}
        ess_vals = []
        for d in range(particles.shape[1]):
            a = particles[:, d]
            ess_vals.append(az.ess(a))
        # aggregate min
        ess_svgd = np.array([v for v in ess_vals])
        ess_svgd_min = np.nanmin(ess_svgd)
    try:
        # if ess returned xarray/dict-like, try to compute a scalar summary
        ess_svgd_min = float(np.nanmin(ess_svgd)) if hasattr(ess_svgd, "__iter__") else float(ess_svgd)
    except Exception:
        ess_svgd_min = float(ess_svgd)

    rows.append({
        "method": "SVGD",
        "n_particles": particles.shape[0],
        "time_sec": svgd_time,
        "ess": ess_svgd_min,
        "ess_per_sec": (ess_svgd_min / svgd_time) if svgd_time > 0 else np.nan,
        "peak_rss_bytes": svgd_peak
    })

    # --- HMC ---
    mem = MemoryMonitor()
    mem.start()
    t0 = time.time()
    w_np, idata = run_hmc(X_train, y_train,
                          n_samples=kwargs.get("hmc_draws", 1000),
                          n_tune=kwargs.get("hmc_tune", 500),
                          chains=kwargs.get("hmc_chains", 4),
                          target_accept=kwargs.get("hmc_target_accept", 0.9),
                          seed=kwargs.get("seed", 0))
    t1 = time.time()
    mem.stop()
    hmc_peak = mem.get_peak()
    hmc_time = t1 - t0

    # compute ESS from InferenceData (recommended)
    try:
        ess_hmc = az.ess(idata)
        # pick minimal ESS across parameters (we'll pick the 'w' variable)
        # az.ess returns xarray with variable dims
        # we'll flatten to numeric min
        import numpy as _np
        ess_vals = []
        # If ess_hmc is xarray dataset, extract numeric values
        try:
            for var in ess_hmc.data_vars:
                ess_arr = np.array(ess_hmc[var])
                ess_vals.append(np.nanmin(ess_arr))
            ess_hmc_min = float(np.nanmin(ess_vals))
        except Exception:
            ess_hmc_min = float(np.nanmin(ess_hmc))
    except Exception as e:
        print("Error computing ESS for HMC:", e)
        ess_hmc_min = np.nan

    rows.append({
        "method": "HMC",
        "n_particles": np.nan,
        "time_sec": hmc_time,
        "ess": ess_hmc_min,
        "ess_per_sec": (ess_hmc_min / hmc_time) if hmc_time > 0 else np.nan,
        "peak_rss_bytes": hmc_peak
    })

    df = pd.DataFrame(rows)
    if os.path.exists(results_csv):
        prev = pd.read_csv(results_csv)
        df = pd.concat([prev, df], ignore_index=True)
    df.to_csv(results_csv, index=False)
    print("Saved results to", results_csv)
    return df
