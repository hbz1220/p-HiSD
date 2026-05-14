"""
Core HiSD and p-HiSD solvers for finding index-k saddle points.

Implements:
  - Standard HiSD (High-index Saddle Dynamics)
  - p-HiSD (Preconditioned HiSD) following Algorithm 1 in the paper
"""
import numpy as np
from scipy.linalg import eigh


def _update_v_iterative(V, H, dt_v, k, M=None, v_iter=1):
    """Update eigenvector frame V via iterative Rayleigh quotient minimization.

    For standard HiSD (M=None):
        v_i <- v_i - dt_v * P_i * H * v_i,  then orthonormalize.
    For p-HiSD (M given):
        v_i <- v_i - dt_v * P_i^M * M^{-1} * H * v_i,  then M-orthonormalize.
    """
    n = V.shape[0]
    for _ in range(v_iter):
        for i in range(k):
            if M is not None:
                Hv = M['solve'](H @ V[:, i])
            else:
                Hv = H @ V[:, i]

            # P_i * Hv:  subtract projections
            proj = Hv.copy()
            if M is not None:
                proj -= V[:, i] * (V[:, i] @ (M['mat'] @ Hv))
                for j in range(i):
                    proj -= 2.0 * V[:, j] * (V[:, j] @ (M['mat'] @ Hv))
            else:
                proj -= V[:, i] * (V[:, i] @ Hv)
                for j in range(i):
                    proj -= 2.0 * V[:, j] * (V[:, j] @ Hv)

            V[:, i] -= dt_v * proj

            # Orthonormalize
            if M is not None:
                Mmat = M['mat']
                for j in range(i):
                    V[:, i] -= V[:, j] * (V[:, j] @ (Mmat @ V[:, i]))
                norm_M = np.sqrt(V[:, i] @ (Mmat @ V[:, i]))
                V[:, i] /= norm_M
            else:
                for j in range(i):
                    V[:, i] -= V[:, j] * (V[:, j] @ V[:, i])
                V[:, i] /= np.linalg.norm(V[:, i])
    return V


def _update_v_eig(x, H, k, M_mat=None):
    """Update eigenvector frame by solving the (generalized) eigenvalue problem directly."""
    if M_mat is not None:
        eigvals, eigvecs = eigh(H, M_mat, subset_by_index=[0, k - 1])
    else:
        eigvals, eigvecs = eigh(H, subset_by_index=[0, k - 1])
    return eigvecs, eigvals


def hisd_solve(x0, grad_E, hess_E, k=1,
               dt_x=0.01, dt_v=0.05, v_iter=3,
               tol=1e-6, max_iter=10000,
               precond=None,
               use_eig=False,
               record_traj=False):
    """Unified HiSD / p-HiSD solver.

    Parameters
    ----------
    x0 : ndarray, shape (n,)
        Initial point.
    grad_E : callable x -> ndarray
        Gradient of the energy.
    hess_E : callable x -> ndarray (n,n)
        Hessian of the energy.
    k : int
        Target Morse index.
    dt_x : float
        Step size for x-update.
    dt_v : float
        Step size for v-update (iterative mode).
    v_iter : int
        Number of inner v-update iterations per outer step.
    tol : float
        Convergence tolerance on ||grad||.
    max_iter : int
        Maximum number of iterations.
    precond : dict or None
        Preconditioner. If None, standard HiSD.
        Keys:
          'mat'   : M matrix (ndarray)
          'solve' : callable b -> M^{-1} b
          'update': (optional) callable (x, H) -> new precond dict
    use_eig : bool
        If True, solve eigenvalue problem directly instead of iterative v-update.
    record_traj : bool
        If True, record full x trajectory.

    Returns
    -------
    result : dict with keys:
        'x'         : final point
        'converged' : bool
        'iterations': int
        'grad_norms': list of gradient norms
        'trajectory': list of x arrays (if record_traj)
    """
    n = len(x0)
    x = x0.copy().astype(float)

    # Initialize V from eigendecomposition
    H = hess_E(x)
    if precond is not None:
        M_mat = precond['mat']
        eigvals, eigvecs = eigh(H, M_mat, subset_by_index=[0, k - 1])
    else:
        M_mat = None
        eigvals, eigvecs = eigh(H, subset_by_index=[0, k - 1])
    V = eigvecs.copy()

    grad_norms = []
    trajectory = [x.copy()] if record_traj else []

    converged = False
    for m in range(max_iter):
        g = grad_E(x)
        gn = np.linalg.norm(g)
        grad_norms.append(gn)

        if gn < tol:
            converged = True
            break

        # x-update
        if precond is not None:
            M_solve = precond['solve']
            g_tilde = M_solve(g)               # M^{-1} g
            d = -g_tilde + 2.0 * V @ (V.T @ g) # -R_V^M M^{-1} g
        else:
            d = -g + 2.0 * V @ (V.T @ g)        # -R_V g

        x = x + dt_x * d

        if record_traj:
            trajectory.append(x.copy())

        # Optional: update preconditioner
        if precond is not None and 'update' in precond:
            precond = precond['update'](x, hess_E(x))

        # v-update
        H = hess_E(x)
        if use_eig:
            V, _ = _update_v_eig(x, H, k, M_mat=M_mat if precond else None)
        else:
            M_dict = {'mat': M_mat, 'solve': precond['solve']} if precond else None
            V = _update_v_iterative(V, H, dt_v, k, M=M_dict, v_iter=v_iter)

    return {
        'x': x,
        'converged': converged,
        'iterations': m + 1 if not converged else m,
        'grad_norms': grad_norms,
        'trajectory': trajectory,
    }
