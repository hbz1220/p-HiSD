"""
Example 2: Verification of Convergence Rate (Figure 2)

Quadratic model E(x) = 1/2 x^T H x with H = diag(-1, lambda_2, ..., lambda_n).
Tests: standard HiSD (kappa=100), p-HiSD (kappa_M=2), p-HiSD (kappa_M=1.01).
"""
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

    # H = diag(-1, 1.5, 3, 10, 20, 30, 50, 60, 80, 100)
    #diag_H = np.linspace(1, 100, n-1)
    #diag_H = np.insert(diag_H, 0, -1.0)
    diag_H = np.arange(2, 101, dtype=float)  # 生成 2.0 到 100.0，共 99 个数
    diag_H = np.insert(diag_H, 0, -1.0)      # 在开头插入 -1.0，凑齐 100 个数
    H = np.diag(diag_H)

    mu = 1.0   # min |lambda_i|
    L = 100.0  # max |lambda_i|
    kappa = L / mu  # = 100

    def grad_E(x):
        return H @ x

    # Starting point with ||grad|| = 1
    x0=np.zeros(n)
    x0[0]=0.5 #最小特征值方向
    x0[-1]=0.5 #最大特征值方向
    x0=x0/np.linalg.norm(H@x0)

    k = 1  # index-1 saddle at origin
    tol = 1e-8

    # --- Config 1: Standard HiSD (kappa=100) ---
    eta1 = 2.0 / (L + mu)
    rho1 = (kappa - 1) / (kappa + 1)

    # --- Config 2: p-HiSD (kappa_M=2.0) ---
    kappa_M2 = 2.0
    # Spectral precond with eps: kappa_M = L(mu+eps)/(mu(L+eps))
    # kappa_M * mu * (L+eps) = L * (mu+eps)
    # kappa_M * mu * L + kappa_M * mu * eps = L*mu + L*eps
    # eps * (kappa_M * mu - L) = L*mu - kappa_M*mu*L = mu*L*(1 - kappa_M)
    # eps = mu*L*(1-kappa_M) / (kappa_M*mu - L)
    eps2 = mu * L * (1 - kappa_M2) / (kappa_M2 * mu - L)
    d2 = np.abs(diag_H) + eps2
    L_eff2 = L / (L + eps2)
    mu_eff2 = mu / (mu + eps2)
    eta2 = 2.0 / (L_eff2 + mu_eff2)
    rho2 = (kappa_M2 - 1) / (kappa_M2 + 1)

    # --- Config 3: p-HiSD (kappa_M=1.01) ---
    kappa_M3 = 1.01
    eps3 = mu * L * (1 - kappa_M3) / (kappa_M3 * mu - L)
    d3 = np.abs(diag_H) + eps3
    L_eff3 = L / (L + eps3)
    mu_eff3 = mu / (mu + eps3)
    eta3 = 2.0 / (L_eff3 + mu_eff3)
    rho3 = (kappa_M3 - 1) / (kappa_M3 + 1)

    configs = [
        ('HiSD ($\\kappa=100$)', None, eta1, rho1, 1200),
        ('p-HiSD ($\\kappa_M=2.0$)', d2, eta2, rho2, 30),
        ('p-HiSD ($\\kappa_M=1.01$)', d3, eta3, rho3, 10),
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

        # Add final gradient norm
        gnorms.append(np.linalg.norm(H @ x))

        # Compute observed rate
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
        }
        print(f"{name}: {len(gnorms)-1} iters, observed rate={obs_rate:.4f}, theory={rho_theory:.4f}")

    # --- Plot Figure 2 ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))
    colors = ['#1f77b4', '#d62728', '#2ca02c']
    for (name, _, _, rho, _), color in zip(configs, colors): # 去掉了 markers 相关的循环变量
        r = results[name]
        iters = np.arange(len(r['gnorms']))
        label = f"{name}"
        # 这里删除了 marker, markersize 和 markevery 三个参数
        ax.semilogy(iters, r['gnorms'], color=color,
                    linewidth=1.5, label=label)

    ax.set_xlabel('Iteration $m$', fontsize=17)
    ax.set_ylabel('$\\|\\nabla E(x_m)\\|$', fontsize=17)
    ax.set_title('Quadratic Model: Rate Verification ', fontsize=17)
    ax.legend(fontsize=15, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=1e-10)
    #ax.set_xlim(0, 500)

    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, '1.pdf'), dpi=300, bbox_inches='tight')
    #fig.savefig(os.path.join(out_dir, '1.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure 2 saved to {out_dir}/2.pdf")


if __name__ == '__main__':
    run()
