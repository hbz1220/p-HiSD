# Section 7.1: verify linear convergence rates for an index-1 quadratic saddle.
# Compare HiSD with two diagonal spectral metrics; save Figure 1 to outputs/7.1/1.pdf.

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.preconditioners import diagonal_precond

def run():
    np.random.seed(42)
    n = 100

    diag_H = np.arange(2, 101, dtype=float)
    diag_H = np.insert(diag_H, 0, -1.0)
    H = np.diag(diag_H)

    mu = 1.0   # min |lambda_i|
    L = 100.0  # max |lambda_i|
    kappa = L / mu

    def grad_E(x):
        return H @ x

    # Starting point with ||grad|| = 1
    x0=np.zeros(n)
    x0[0]=0.5
    x0[-1]=0.5
    x0=x0/np.linalg.norm(H@x0)

    k = 1  # index-1 saddle at origin
    tol = 1e-6
    delta_ind = 1e-6
    # Exact ordinary spectrum: this diagonal Hessian is constant at every endpoint.
    eigenvalues = np.sort(diag_H)
    index_resolved = bool(np.all(np.isfinite(eigenvalues))
                          and np.all(np.abs(eigenvalues) > delta_ind))
    morse_index = int(np.count_nonzero(eigenvalues < -delta_ind)) if index_resolved else None
    index_ok = bool(index_resolved and eigenvalues[k-1] < -delta_ind
                    and eigenvalues[k] > delta_ind)

    # Optimal fixed step for the reflected quadratic spectrum [mu, L].
    eta1 = 2.0 / (L + mu)
    rho1 = (kappa - 1) / (kappa + 1)

    # Choose spectral regularization to attain the prescribed condition number.
    kappa_M2 = 2.0
    # For M = |H| + eps I, kappa_M = L(mu + eps) / (mu(L + eps)).
    eps2 = mu * L * (1 - kappa_M2) / (kappa_M2 * mu - L)
    d2 = np.abs(diag_H) + eps2
    L_eff2 = L / (L + eps2)
    mu_eff2 = mu / (mu + eps2)
    eta2 = 2.0 / (L_eff2 + mu_eff2)
    rho2 = (kappa_M2 - 1) / (kappa_M2 + 1)

    # Repeat with a metric closer to the absolute Hessian.
    kappa_M3 = 1.01
    eps3 = mu * L * (1 - kappa_M3) / (kappa_M3 * mu - L)
    d3 = np.abs(diag_H) + eps3
    L_eff3 = L / (L + eps3)
    mu_eff3 = mu / (mu + eps3)
    eta3 = 2.0 / (L_eff3 + mu_eff3)
    rho3 = (kappa_M3 - 1) / (kappa_M3 + 1)

    configs = [
        ('HiSD ($\\kappa=100$)', None, eta1, rho1, 2000000),
        ('p-HiSD ($\\kappa_M=2.0$)', d2, eta2, rho2, 2000000),
        ('p-HiSD ($\\kappa_M=1.01$)', d3, eta3, rho3, 2000000),
    ]

    results = {}
    for name, d, eta, rho_theory, max_it in configs:
        x = x0.copy()
        # Eigenvector for smallest eigenvalue: e_0 (index 0, lambda=-1)
        v = np.zeros(n)
        v[0] = 1.0

        if d is not None:
            d_inv = 1.0 / d
            # M-normalize v: v^T M v = 1 => v = e_0 / sqrt(M_00)
            v_M = v / np.sqrt(d[0])
        else:
            v_M = v.copy()

        gnorms = []
        for it in range(max_it):
            g = H @ x
            gn = np.linalg.norm(g)
            gnorms.append(gn)
            # The residual stops iteration; the constant spectrum certifies the index.
            if gn < tol:
                break

            if d is not None:
                # p-HiSD: d = -M^{-1}g + 2*v*(v^T g)
                g_tilde = d_inv * g
                direction = -g_tilde + 2.0 * v_M * (v_M @ g)
            else:
                # Standard: d = -g + 2*v*(v^T g)
                direction = -g + 2.0 * v * (v @ g)

            x = x + eta * direction

        gnorms.append(np.linalg.norm(H @ x))

        # Certify the actual endpoint using its final residual and ordinary Hessian index.
        final_residual = float(gnorms[-1])
        if not np.isfinite(final_residual) or final_residual >= tol:
            status = 'residual_not_converged'
        elif not index_resolved:
            status = 'index_unresolved'
        elif not index_ok:
            status = 'wrong_index'
        else:
            status = 'converged'

        # Estimate contraction from consecutive residuals, omitting the first ratio.
        if len(gnorms) > 2:
            rates = [gnorms[i+1] / gnorms[i] for i in range(len(gnorms)-1) if gnorms[i] > 1e-15]
            obs_rate = np.median(rates[1:]) if len(rates) > 1 else rates[0]
        else:
            obs_rate = rho_theory

        results[name] = {
            'gnorms': gnorms,
            'rho_theory': rho_theory,
            'rho_obs': obs_rate,
            'iters': len(gnorms) - 1,
            'final_residual': final_residual,
            'morse_index': morse_index,
            'index_resolved': index_resolved,
            'delta_ind': delta_ind,
            'status': status,
            'converged': status == 'converged',
        }
        print(f"{name}: {len(gnorms)-1} iters, observed rate={obs_rate:.4f}, theory={rho_theory:.4f}")
        print(f"  final ||g||={final_residual:.2e}, Morse index={morse_index}, "
              f"delta_ind={delta_ind:.1e}, status={status}")

    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))
    colors = ['#1f77b4', '#d62728', '#2ca02c']
    for (name, _, _, rho, _), color in zip(configs, colors):
        r = results[name]
        iters = np.arange(len(r['gnorms']))
        label = f"{name}"
        ax.semilogy(iters, r['gnorms'], color=color,
                    linewidth=1.5, label=label)

    ax.set_xlabel('Iteration $m$', fontsize=17)
    ax.set_ylabel('$\\|\\nabla E(x_m)\\|$', fontsize=17)
    ax.set_title('Quadratic Model: Rate Verification ', fontsize=17)
    ax.legend(fontsize=15, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=1e-8)

    plt.tight_layout()
    out_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', 'outputs', '7.1'
    ))
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, '1.pdf'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure 2 saved to {out_dir}/1.pdf")


if __name__ == '__main__':
    run()
