import numpy as np

def log_posterior(x):
    """Log probability of a standard 2D Gaussian."""
    return -0.5 * np.sum(x**2, axis=1)