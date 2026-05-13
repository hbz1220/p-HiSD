"""
Preconditioner implementations for p-HiSD.

Each preconditioner returns a dict with:
    'mat'   : M matrix (ndarray, SPD)
    'solve' : callable b -> M^{-1} b
    'update': (optional) callable (x, H) -> new precond dict
"""
import numpy as np
from scipy.linalg import cholesky, cho_solve, eigh


def identity_precond(n):
    """Identity preconditioner (equivalent to standard HiSD)."""
    return None


def spectral_precond(H, eps_ratio=1e-6, frozen=False):
    """Spectral preconditioner: M = Q diag(|lambda_i| + eps) Q^T.

    Parameters
    ----------
    H : ndarray (n, n)
        Hessian matrix.
    eps_ratio : float
        Regularization: eps = eps_ratio * max|lambda_i|.
    frozen : bool
        If True, do not update M with new Hessians.
    """
    eigvals, Q = eigh(H)
    eps = eps_ratio * np.max(np.abs(eigvals))
    d = np.abs(eigvals) + eps
    M = Q @ np.diag(d) @ Q.T
    M = 0.5 * (M + M.T)

    d_inv = 1.0 / d
    M_inv_Q = Q @ np.diag(d_inv)

    def solve(b):
        return M_inv_Q @ (Q.T @ b)

    result = {'mat': M, 'solve': solve}
    if not frozen:
        def update(x, H_new):
            return spectral_precond(H_new, eps_ratio=eps_ratio, frozen=False)
        result['update'] = update
    return result


def spectral_precond_custom(H, eps):
    """Spectral preconditioner with explicit eps value."""
    eigvals, Q = eigh(H)
    d = np.abs(eigvals) + eps
    M = Q @ np.diag(d) @ Q.T
    M = 0.5 * (M + M.T)

    d_inv = 1.0 / d
    M_inv_Q = Q @ np.diag(d_inv)

    def solve(b):
        return M_inv_Q @ (Q.T @ b)

    def update(x, H_new):
        return spectral_precond_custom(H_new, eps)

    return {'mat': M, 'solve': solve, 'update': update}


def frozen_spectral_precond(H0, eps_ratio=1e-6):
    """Frozen spectral preconditioner: computed once from H0."""
    return spectral_precond(H0, eps_ratio=eps_ratio, frozen=True)


def jacobi_precond(H, eps=1e-6, frozen=False):
    """Jacobi (diagonal) preconditioner: M = diag(|H_ii| + eps)."""
    d = np.abs(np.diag(H)) + eps
    M = np.diag(d)
    d_inv = 1.0 / d

    def solve(b):
        return d_inv * b

    result = {'mat': M, 'solve': solve}
    if not frozen:
        def update(x, H_new):
            return jacobi_precond(H_new, eps=eps, frozen=False)
        result['update'] = update
    return result


def block_jacobi_precond(H, block_sizes, eps=1e-6, frozen=False):
    """Block Jacobi preconditioner.

    Parameters
    ----------
    H : ndarray (n, n)
    block_sizes : list of int
        Sizes of diagonal blocks, must sum to n.
    eps : float
        Regularization parameter.
    """
    n = H.shape[0]
    M = np.zeros((n, n))
    M_inv = np.zeros((n, n))

    idx = 0
    for bs in block_sizes:
        block = H[idx:idx+bs, idx:idx+bs]
        # |block| + eps*I via eigendecomposition
        ev, Q = eigh(block)
        d = np.abs(ev) + eps
        M_block = Q @ np.diag(d) @ Q.T
        M_block_inv = Q @ np.diag(1.0 / d) @ Q.T
        M[idx:idx+bs, idx:idx+bs] = M_block
        M_inv[idx:idx+bs, idx:idx+bs] = M_block_inv
        idx += bs

    def solve(b):
        return M_inv @ b

    result = {'mat': M, 'solve': solve}
    if not frozen:
        def update(x, H_new):
            return block_jacobi_precond(H_new, block_sizes, eps=eps, frozen=False)
        result['update'] = update
    return result


def shifted_cholesky_precond(H, sigma=None, sigma0=0.1, frozen=False):
    """Shifted (incomplete) Cholesky preconditioner: M = L L^T where L L^T ≈ H + sigma*I.

    For small dense problems, this is a complete Cholesky factorization.
    """
    if sigma is None:
        eigvals = eigh(H, eigvals_only=True, subset_by_index=[0, 0])
        sigma = -eigvals[0] + sigma0

    H_shifted = H + sigma * np.eye(H.shape[0])
    H_shifted = 0.5 * (H_shifted + H_shifted.T)
    L = cholesky(H_shifted, lower=True)
    M = L @ L.T

    def solve(b):
        return cho_solve((L, True), b)

    result = {'mat': M, 'solve': solve}
    if not frozen:
        def update(x, H_new):
            return shifted_cholesky_precond(H_new, sigma0=sigma0, frozen=False)
        result['update'] = update
    return result


def subspace_inertial_precond(H, k, alpha=0.7, a_weights=None, eps=1e-2,
                               V_old=None):
    """Subspace-Inertial preconditioner.

    Parameters
    ----------
    H : ndarray (n, n)
    k : int
        Number of unstable directions.
    alpha : float
        Inertia parameter in [0, 1).
    a_weights : ndarray (k,) or None
        Spectral stiffness weights. If None, uses uniform weights with
        constraint a_bar + beta = 1.
    eps : float
        Regularization.
    V_old : ndarray (n, k) or None
        Previous basis. If None, uses eigenvectors directly.
    """
    n = H.shape[0]
    eigvals, eigvecs = eigh(H)
    # k smallest eigenpairs
    lam = eigvals[:k]
    V_new = eigvecs[:, :k].copy()
    lam_kp1 = eigvals[k] if k < n else eigvals[-1]

    if a_weights is None:
        a_weights = np.ones(k) * 0.49  # default

    # Inertial update
    if V_old is not None and V_old.shape == (n, k):
        for i in range(k):
            sign_i = np.sign(np.dot(V_new[:, i], V_old[:, i]))
            if sign_i == 0:
                sign_i = 1.0
            V_new[:, i] = (1 - alpha) * sign_i * V_new[:, i] + alpha * V_old[:, i]
        # Orthonormalize
        for i in range(k):
            for j in range(i):
                V_new[:, i] -= V_new[:, j] * np.dot(V_new[:, j], V_new[:, i])
            V_new[:, i] /= np.linalg.norm(V_new[:, i])

    # Construct M
    a_bar = np.mean(a_weights)
    beta = 1.0 - a_bar
    mu = np.array([a_weights[i] * np.abs(lam[i]) + eps for i in range(k)])
    mu_rest = beta * np.abs(lam_kp1) + eps

    # M = mu_rest * I + sum_i (mu_i - mu_rest) * v_i v_i^T
    M = mu_rest * np.eye(n)
    for i in range(k):
        M += (mu[i] - mu_rest) * np.outer(V_new[:, i], V_new[:, i])
    M = 0.5 * (M + M.T)

    # Solve M^{-1} b directly for robustness
    # For small n, direct solve is fine; for large n, use Woodbury
    M_sym = 0.5 * (M + M.T)
    try:
        from scipy.linalg import cho_factor, cho_solve as _cho_solve
        cho = cho_factor(M_sym)
        def solve(b):
            return _cho_solve(cho, b)
    except Exception:
        M_inv = np.linalg.inv(M_sym)
        def solve(b):
            return M_inv @ b

    V_old_copy = V_new.copy()

    def update(x, H_new):
        return subspace_inertial_precond(H_new, k, alpha=alpha,
                                          a_weights=a_weights, eps=eps,
                                          V_old=V_old_copy)

    return {'mat': M, 'solve': solve, 'update': update}


def diagonal_precond(d):
    """Generic diagonal preconditioner from a vector d (all positive)."""
    M = np.diag(d)
    d_inv = 1.0 / d

    def solve(b):
        return d_inv * b

    return {'mat': M, 'solve': solve}
