# scripts/svgd_jax.py
"""
SVGD implementation in JAX for Bayesian logistic regression.
Takes X (N x D) and y (N,), returns particles (n_particles x D),
and timing information.
"""
import time
from typing import Tuple
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, vmap, jit
from functools import partial
from tqdm import trange

# Prior sigma
PRIOR_STD = 1.0

def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))

def log_prior(w):
    # Gaussian prior N(0, PRIOR_STD^2)
    return -0.5 * jnp.sum((w / PRIOR_STD) ** 2)

def log_likelihood_single(w, x, y):
    logits = jnp.dot(x, w)
    # Bernoulli log-likelihood
    return y * jnp.log(sigmoid(logits) + 1e-12) + (1 - y) * jnp.log(1 - sigmoid(logits) + 1e-12)

def log_posterior(w, X, y):
    # w: (D,)
    ll = jnp.sum(vmap(lambda xi, yi: log_likelihood_single(w, xi, yi))(X, y))
    return ll + log_prior(w)

def rbf_kernel_matrix(theta):
    # theta: (n_particles, dim)
    pairwise_dists = jnp.sum((theta[:, None, :] - theta[None, :, :]) ** 2, axis=-1)
    # median heuristic
    h = jnp.median(pairwise_dists)
    h = jnp.where(h <= 0.0, 1.0, h)
    h = h / jnp.log(theta.shape[0] + 1.0)  # slightly shrink
    K = jnp.exp(-pairwise_dists / (h + 1e-8))
    return K, h

@partial(jit, static_argnums=(3,4))
def svgd_update(theta, X, y, n_particles, step_size, rng_key):
    # theta: (n_particles, dim)
    dim = theta.shape[1]

    # gradient of log posterior for each particle
    grad_logp = jax.vmap(lambda w: jax.grad(lambda ww: log_posterior(ww, X, y))(w))(theta)  # (n_particles, dim)

    K, h = rbf_kernel_matrix(theta)
    # kernel gradient term
    grad_K = -2 * (theta[:, None, :] - theta[None, :, :]) * (K[:, :, None] / (h + 1e-8))
    # combine
    phi = (K @ grad_logp) / n_particles + jnp.sum(grad_K, axis=1) / n_particles
    theta_new = theta + step_size * phi
    return theta_new, phi

def run_svgd(X: np.ndarray, y: np.ndarray, n_particles=100, n_iter=1000, step_size=1e-3, seed=0):
    X_j = jnp.array(X)
    y_j = jnp.array(y)
    n_particles = int(n_particles)
    dim = X.shape[1]
    rng = jax.random.PRNGKey(seed)
    rng, sub = jax.random.split(rng)
    # init particles from prior
    theta = jax.random.normal(sub, (n_particles, dim)) * PRIOR_STD

    start = time.time()
    for it in trange(n_iter, desc="SVGD iters"):
        rng, sub = jax.random.split(rng)
        theta, phi = svgd_update(theta, X_j, y_j, n_particles, step_size, sub)
    duration = time.time() - start
    # return particles as numpy
    return np.array(theta), duration
