"""
Example 3: Stiff Diatomic Chain (Figure 3)

N=50 molecule pairs, n=100.
K=10^4, delta=1.  Stiffness ratio 10^4.
Compares: Standard HiSD, Block Jacobi, IC, Frozen Spectral.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import eigh, cholesky, cho_solve


def setup_diatomic(N, K, delta):
    n = 2 * N

    def gradient(x):
        g = np.zeros(n)
        u = x[0::2]
        v = x[1::2]
        for i in range(N):
            g[2*i] = K * (u[i] - v[i]) + 4 * u[i] * (u[i]**2 - 1)
            g[2*i+1] = K * (v[i] - u[i]) + 4 * v[i] * (v[i]**2 - 1)
        for i in range(N - 1):
            g[2*i+1] += delta * (v[i] - u[i+1])
            g[2*(i+1)] += delta * (u[i+1] - v[i])
        return g

    def hessian(x):
        H = np.zeros((n, n))
        u = x[0::2]
        v = x[1::2]
        for i in range(N):
            ui, vi = 2*i, 2*i+1
            H[ui, ui] = K + 12*u[i]**2 - 4
            H[vi, vi] = K + 12*v[i]**2 - 4
            H[ui, vi] = -K
            H[vi, ui] = -K
        for i in range(N - 1):
            vi = 2*i + 1
            ui1 = 2*(i+1)
            H[vi, vi] += delta
            H[ui1, ui1] += delta
            H[vi, ui1] = -delta
            H[ui1, vi] = -delta
        return H

    return gradient, hessian


def make_block_jacobi(H, N, eps=1e-2):
    """Build block Jacobi preconditioner from 2x2 diagonal blocks."""
    n = 2 * N
    M = np.zeros((n, n))
    M_inv = np.zeros((n, n))
    for i in range(N):
        idx = slice(2*i, 2*i+2)
        block = H[idx, idx]
        ev, Q = eigh(block)
        d = np.abs(ev) + eps
        M[idx, idx] = Q @ np.diag(d) @ Q.T
        M_inv[idx, idx] = Q @ np.diag(1.0/d) @ Q.T
    return M, M_inv


def make_shifted_cholesky(H, sigma0=0.1):
    """Build shifted complete Cholesky preconditioner."""
    ev_min = eigh(H, eigvals_only=True, subset_by_index=[0, 0])[0]
    sigma = max(-ev_min + sigma0, sigma0)
    H_shifted = H + sigma * np.eye(H.shape[0])
    H_shifted = 0.5 * (H_shifted + H_shifted.T)
    L = cholesky(H_shifted, lower=True)
    return L


def make_frozen_spectral(H, eps_ratio=1e-4):
    """Build frozen spectral preconditioner."""
    eigvals, Q = eigh(H)
    eps = eps_ratio * np.max(np.abs(eigvals))
    d = np.abs(eigvals) + eps
    M = Q @ np.diag(d) @ Q.T
    d_inv = 1.0 / d
    return M, Q, d_inv


def run_phisd(x0, grad_fn, hess_fn, solve_fn, M_mat, dt_x, dt_v, k=1,
              tol=1e-6, max_iter=2000, update_precond=None):
    """Run p-HiSD. solve_fn: b -> M^{-1}b. M_mat: M matrix for eig."""
    x = x0.copy()
    H = hess_fn(x)

    if M_mat is not None:
        _, V = eigh(H, M_mat, subset_by_index=[0, k-1])
    else:
        _, V = eigh(H, subset_by_index=[0, k-1])
    V = V.reshape(-1, k)

    gnorms = []
    for it in range(max_iter):
        g = grad_fn(x)
        gn = np.linalg.norm(g)
        gnorms.append(gn)
        if gn < tol:
            break
        if not np.isfinite(gn) or gn > 1e15:
            break

        if solve_fn is not None:
            g_tilde = solve_fn(g)
            d = -g_tilde + 2.0 * V @ (V.T @ g)
        else:
            d = -g + 2.0 * V @ (V.T @ g)

        x = x + dt_x * d
        if not np.all(np.isfinite(x)):
            break

        H = hess_fn(x)

        # Update preconditioner if needed
        if update_precond is not None:
            solve_fn, M_mat = update_precond(x, H)

        # v-update via eigensolve
        try:
            if M_mat is not None:
                _, V = eigh(H, M_mat, subset_by_index=[0, k-1])
            else:
                _, V = eigh(H, subset_by_index=[0, k-1])
            V = V.reshape(-1, k)
        except Exception:
            pass  # keep old V

    return x, gnorms


def run():
    N = 50
    n = 2 * N
    K = 1e4
    delta_param = 1.0
    k = 1
    dt_v = 0.05
    tol = 1e-6
    max_iter_all = 2000

    gradient, hessian = setup_diatomic(N, K, delta_param)

    np.random.seed(123)
    x0 = np.zeros(n)
    for i in range(N):
        x0[2*i] = (-1)**i + 0.1 * np.random.randn()
        x0[2*i+1] = x0[2*i] + 0.1 * np.random.randn()

    g0_norm = np.linalg.norm(gradient(x0))
    print(f"Initial ||g|| = {g0_norm:.2e}")
    H0 = hessian(x0)

    results = {}

    # 1. Standard HiSD: dt_x = 5e-5
    print("Running Standard HiSD...")
    _, gn = run_phisd(x0, gradient, hessian, None, None,
                      dt_x=5e-5, dt_v=dt_v, k=k, tol=tol, max_iter=max_iter_all)
    results['Standard HiSD'] = gn
    print(f"  {len(gn)} iters, final ||g||={gn[-1]:.2e}")

    # 2. Block Jacobi: dt_x = 0.5
    print("Running Block Jacobi p-HiSD...")
    M_bj, M_bj_inv = make_block_jacobi(H0, N)
    solve_bj = lambda b: M_bj_inv @ b
    _, gn = run_phisd(x0, gradient, hessian, solve_bj, M_bj,
                      dt_x=0.5, dt_v=dt_v, k=k, tol=tol, max_iter=max_iter_all)
    results['Block Jacobi'] = gn
    print(f"  {len(gn)} iters, final ||g||={gn[-1]:.2e}")

    # 3. IC (shifted Cholesky, frozen): dt_x = 0.5
    print("Running IC p-HiSD...")
    L_ic = make_shifted_cholesky(H0, sigma0=1.0)
    M_ic = L_ic @ L_ic.T
    solve_ic = lambda b: cho_solve((L_ic, True), b)
    _, gn = run_phisd(x0, gradient, hessian, solve_ic, M_ic,
                      dt_x=0.5, dt_v=dt_v, k=k, tol=tol, max_iter=max_iter_all)
    results['IC'] = gn
    print(f"  {len(gn)} iters, final ||g||={gn[-1]:.2e}")

    # 4. Frozen Spectral: dt_x = 0.5
    print("Running Frozen Spectral p-HiSD...")
    M_fs, Q_fs, d_inv_fs = make_frozen_spectral(H0)
    solve_fs = lambda b: Q_fs @ (d_inv_fs * (Q_fs.T @ b))
    _, gn = run_phisd(x0, gradient, hessian, solve_fs, M_fs,
                      dt_x=0.5, dt_v=dt_v, k=k, tol=tol, max_iter=max_iter_all)
    results['Frozen Spectral'] = gn
    print(f"  {len(gn)} iters, final ||g||={gn[-1]:.2e}")

    # --- Plot Figure 3 ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))

    styles = {
        'Standard HiSD':   {'color': '#1f77b4', 'marker': 'o'},
        'Block Jacobi':    {'color': '#d62728', 'marker': 's'},
        'IC':              {'color': '#2ca02c', 'marker': '^'},
        'Frozen Spectral': {'color': '#ff7f0e', 'marker': 'D'},
    }
    labels = {
        'Standard HiSD': 'Standard HiSD ($\\Delta t_x=5{\\times}10^{-5}$)',
        'Block Jacobi': 'Block Jacobi ($\\Delta t_x=0.5$)',
        'IC': 'IC ($\\Delta t_x=0.5$)',
        'Frozen Spectral': 'Frozen Spectral ($\\Delta t_x=0.5$)',
    }

    for name, gn in results.items():
        s = styles[name]
        iters = np.arange(len(gn))
        ax.semilogy(iters, gn, color=s['color'],
                    marker=s['marker'], markersize=3,
                    markevery=max(1, len(gn)//20),
                    linewidth=1.5, label=labels[name])

    ax.set_xlabel('Iteration $m$', fontsize=13)
    ax.set_ylabel('$\\|\\nabla E(\\mathbf{x}_m)\\|$', fontsize=13)
    ax.set_title('Stiff Diatomic Chain ($N=50$, $K/\\delta=10^4$)', fontsize=14)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, '3.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, '3.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure 3 saved to {out_dir}/3.pdf")


if __name__ == '__main__':
    run()
