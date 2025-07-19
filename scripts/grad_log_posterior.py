import numpy as np

def grad_log_posterior(x):
    """Gradient of the log probability (standard Gaussian)."""
    return -x