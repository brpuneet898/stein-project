import numpy as np

class SVGD:
    def __init__(self, bandwidth=None, step_size=1e-2):
        self.bandwidth = bandwidth
        self.step_size = step_size

    def _rbf_kernel(self, X):
        n, d = X.shape
        pairwise_dists = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
        h = self.bandwidth
        if h is None:
            h = np.median(pairwise_dists)
            h = 0.5 * h / np.log(n + 1)

        K = np.exp(-pairwise_dists / h)
        grad_K = -np.matmul(K, X) + X * np.sum(K, axis=1, keepdims=True)
        grad_K *= (2 / h)
        return K, grad_K

    def update(self, particles, grad_log_p):
        score = grad_log_p(particles)
        K, grad_K = self._rbf_kernel(particles)
        phi = (np.matmul(K, score) + grad_K) / particles.shape[0]
        return particles + self.step_size * phi