import numpy as np

def imq_kernel(x, y, alpha=0.5, c=1.0):
    """Inverse Multiquadric kernel: k(x, y) = (c + ||x - y||^2)^(-alpha)"""
    dist_sq = np.sum((x - y) ** 2)
    return (c + dist_sq) ** (-alpha)

def linear_kernel(x, y):
    """Linear kernel: k(x, y) = x^T y"""
    return np.dot(x, y)

def compute_kernel_matrix(X, kernel_fn, **kwargs):
    """Computes the full kernel matrix."""
    n = X.shape[0]
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = kernel_fn(X[i], X[j], **kwargs) if kwargs else kernel_fn(X[i], X[j])
    return K