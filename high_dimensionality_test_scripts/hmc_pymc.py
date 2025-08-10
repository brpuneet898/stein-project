import numpy as np
import arviz as az

def run_hmc(X, y, n_samples=2000, n_tune=1000, chains=4, target_accept=0.9, seed=0):
    import pymc as pm
    n, dim = X.shape
    X_shared = X

    with pm.Model() as model:
        w = pm.Normal("w", mu=0.0, sigma=1.0, shape=dim)
        logits = pm.math.dot(X_shared, w)
        y_obs = pm.Bernoulli("y_obs", logit_p=logits, observed=y)

        idata = pm.sample(
            draws=n_samples,
            tune=n_tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=seed,
            cores=1,
            progressbar=True
        )

    # Convert to numpy — works for 1 or multiple chains
    posterior_w = idata.posterior["w"].values  # shape: (chain, draw, dim)
    w_np = posterior_w.reshape(-1, dim)
    return w_np, idata
