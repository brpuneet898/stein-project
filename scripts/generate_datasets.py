import numpy as np

def generate_gaussian_mixture(n_samples=1000):
    """Generates a 2-component Gaussian mixture."""
    mean1, mean2 = np.array([2, 2]), np.array([-2, -2])
    cov = np.eye(2)
    samples = []
    for _ in range(n_samples):
        if np.random.rand() > 0.5:
            samples.append(np.random.multivariate_normal(mean1, cov))
        else:
            samples.append(np.random.multivariate_normal(mean2, cov))
    return np.array(samples)

def generate_banana_posterior(n_samples=1000, a=1, b=0.1):
    """Banana-shaped distribution."""
    x = np.random.normal(0, 1, size=(n_samples, 2))
    x[:, 1] = x[:, 1] + b * (x[:, 0]**2 - a**2)
    return x