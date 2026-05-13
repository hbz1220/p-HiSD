"""
Allen-Cahn Equation: Standard HiSD vs Adaptive p-HiSD
Correct M-normalization implementation (v^T M v = 1)
Using Cholesky decomposition for SPD preconditioners
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse import diags, kron, eye as speye, csc_matrix
from scipy.sparse.linalg import eigsh, LinearOperator, spsolve_triangular
import time
import os
# Try to import scikit-sparse for efficient sparse Cholesky
try:
    from sksparse.cholmod import cholesky as cholmod_cholesky  # type: ignore
    USE_CHOLMOD = True
    print("Using CHOLMOD (scikit-sparse) for Cholesky decomposition")
except ImportError:
    USE_CHOLMOD = False
    print("CHOLMOD not available, using scipy's sparse triangular solver")
    # Fallback: use scipy's own sparse Cholesky via cholesky_AAt
    from scipy.sparse.linalg import splu
    print("Warning: Using LU decomposition (install scikit-sparse for better performance)")


def build_laplacian_neumann(N):
    """Build 2D discrete Laplacian with Neumann BC on [0,1]^2, N x N grid."""
    h = 1.0 / (N - 1)
    e = np.ones(N)
    L1d = diags([-e, 2*e, -e], [-1, 0, 1], shape=(N, N), format='lil')
    L1d[0, 0] = 1.0; L1d[0, 1] = -1.0
    L1d[N-1, N-1] = 1.0; L1d[N-1, N-2] = -1.0
    L1d = L1d.tocsc() / h**2
    I_N = speye(N, format='csc')
    L2d = kron(I_N, L1d, format='csc') + kron(L1d, I_N, format='csc')
    return L2d, h


def grad_fn(u, L2d, eps):
    """Gradient: -Delta u + (u^3 - u)/eps^2"""
    return L2d @ u + (u**3 - u) / eps**2


def hess_matvec(u, v, L2d, eps):
    """Hessian-vector product: -Delta v + (3u^2 - 1)/eps^2 * v"""
    return L2d @ v + (3*u**2 - 1) / eps**2 * v


def cholesky_factor(M):
    """
    Compute Cholesky factorization of SPD matrix M.
    Returns a solver function that computes M^{-1} b by solving triangular systems.
    """
    if USE_CHOLMOD:
        # Use CHOLMOD from scikit-sparse (most efficient)
        factor = cholmod_cholesky(M)
        return factor, lambda b: factor(b)
    else:
        # Fallback to LU decomposition (scipy doesn't have sparse Cholesky)
        factor = splu(M)
        return factor, lambda b: factor.solve(b)


def run():
    # Parameters
    N = 80
    n = N * N
    eps = 0.07
    dt_x_std = 1e-5
    dt_x_p = 0.2
    dt_v = 1e-2
    v_iter = 5
    max_iter_std = 800
    max_iter_p = 500
    tol = 1e-6

    L2d, h = build_laplacian_neumann(N)
    print(f"Allen-Cahn: N={N}, n={n}, eps={eps}, h={h:.5f}")

    # Initial guess
    np.random.seed(456)
    xx = np.linspace(0, 1, N)
    X, Y = np.meshgrid(xx, xx)
    u0 = np.tanh((X.ravel() - 0.5) / (eps * np.sqrt(2))) + 0.05 * np.random.randn(n)

    # ========= Standard HiSD =========
    print("Running Standard HiSD (dt_x=1e-5)...")
    u = u0.copy()

    # Initialize v (standard eigenproblem)
    H_op = LinearOperator((n, n), matvec=lambda v: hess_matvec(u, v, L2d, eps))
    try:
        _, v_std = eigsh(H_op, k=1, which='SA')
        v_std = v_std[:, 0]
    except:
        v_std = np.random.randn(n)
    v_std /= np.linalg.norm(v_std)

    gnorms_std = []
    for it in range(max_iter_std):
        g = grad_fn(u, L2d, eps)
        gn = np.linalg.norm(g)
        gnorms_std.append(gn)
        if gn < tol:
            break

        # x-update: d = -g + 2v(v^T g)
        d = -g + 2.0 * v_std * (v_std @ g)
        u = u + dt_x_std * d

        # v-update: v = v - dt_v * (Hv - v(v^T Hv))
        for _ in range(v_iter):
            Hv = hess_matvec(u, v_std, L2d, eps)
            proj = Hv - v_std * (v_std @ Hv)
            v_std = v_std - dt_v * proj
            v_std /= np.linalg.norm(v_std)

        if (it + 1) % 200 == 0:
            print(f"  Std iter {it+1}: ||g||={gn:.4e}")

    print(f"  Standard: {len(gnorms_std)} iters, final ||g||={gnorms_std[-1]:.4e}")
    u_std_final = u.copy()  # Save final solution from standard HiSD

    # ========= Adaptive p-HiSD =========
    print("Running Adaptive p-HiSD (dt_x=0.2)...")
    u = u0.copy()

    # Initialize v (generalized eigenproblem with frozen M)
    print("  Constructing Phase 1 preconditioner (Cholesky)...")
    mu_bj = 2.0 / eps**2
    M_frozen = L2d + mu_bj * speye(n, format='csc')
    M_frozen_csc = csc_matrix(M_frozen)
    M_frozen_factor, M_frozen_solve = cholesky_factor(M_frozen_csc)

    H0 = L2d + diags((3 * u**2 - 1) / eps**2, 0, format='csc')
    try:
        _, v_p = eigsh(H0, k=1, M=M_frozen_csc, which='SA')
        v_p = v_p[:, 0]
    except:
        v_p = np.random.randn(n)
    # M-normalization: v^T M v = 1
    Mv = M_frozen_csc @ v_p
    v_p = v_p / np.sqrt(v_p @ Mv)

    gnorms_p = []
    switch_iter = None
    phase = 1
    M_solve = M_frozen_solve  # Use Cholesky solver
    M_mat = M_frozen_csc

    for it in range(max_iter_p):
        g = grad_fn(u, L2d, eps)
        gn = np.linalg.norm(g)
        gnorms_p.append(gn)
        if gn < tol:
            break

        # Adaptive switching: check stagnation
        if phase == 1 and it > 20:
            recent = gnorms_p[-10:]
            if len(recent) >= 10 and recent[-1] / (recent[0] + 1e-30) > 0.90:
                print(f"  Switching preconditioner at iteration {it} (||g||={gn:.4e})")
                print(f"  Constructing Phase 2 preconditioner (Cholesky)...")
                switch_iter = it
                phase = 2
                # Build shifted Cholesky preconditioner
                H_sparse = L2d + diags((3*u**2 - 1) / eps**2, 0, format='csc')
                try:
                    eig_min = eigsh(H_sparse, k=1, which='SA', return_eigenvectors=False)[0]
                except:
                    eig_min = -100.0
                sigma = max(-2.0 * eig_min, -eig_min + 1.0)
                H_shifted = H_sparse + sigma * speye(n, format='csc')
                H_shifted_csc = csc_matrix(H_shifted)
                M_chol_factor, M_chol_solve = cholesky_factor(H_shifted_csc)
                M_solve = M_chol_solve  # Use Cholesky solver
                M_mat = H_shifted_csc
                print(f"  ✓ Preconditioner FROZEN at iteration {it} (will NOT be updated)")
                print(f"  ✓ Using shifted Hessian: M = H(u_{it}) + {sigma:.4e}*I")

        # [REMOVED] Periodic preconditioner update - now using FROZEN preconditioner
        # The preconditioner is constructed ONCE at switch and never updated again

        # x-update: d = -M^{-1}g + 2v(v^T g)  [v is M-normalized]
        g_tilde = M_solve(g)
        d = -g_tilde + 2.0 * v_p * (v_p @ g)
        dt_x_eff = 1 if phase == 2 else dt_x_p
        u = u + dt_x_eff * d

        # v-update: v = v - dt_v * (M^{-1}Hv - v(v^T Hv))
        if phase == 2 and gn < 1e-2:
            # Recompute v via eigenproblem when close to saddle
            H_op_v = LinearOperator((n, n), matvec=lambda w: hess_matvec(u, w, L2d, eps))
            try:
                _, v_p = eigsh(H_op_v, k=1, which='SA', v0=v_p)
                v_p = v_p[:, 0]
            except:
                pass
        else:
            n_v_iter = 10 if phase == 2 else v_iter
            for _ in range(n_v_iter):
                Hv = hess_matvec(u, v_p, L2d, eps)
                Minv_Hv = M_solve(Hv)
                proj = Minv_Hv - v_p * (v_p @ Hv)
                v_p = v_p - dt_v * proj
                # M-normalization
                Mv = M_mat @ v_p
                v_p = v_p / np.sqrt(v_p @ Mv)

        if (it + 1) % 50 == 0:
            print(f"  p-HiSD iter {it+1}: ||g||={gn:.4e} (phase {phase})")

    print(f"  p-HiSD: {len(gnorms_p)} iters, final ||g||={gnorms_p[-1]:.4e}")
    u_p_final = u.copy()  # Save final solution from p-HiSD

    # ========= Verify if both methods converged to the same saddle point =========
    print("\n" + "="*60)
    print("VERIFICATION: Comparing the two saddle points")
    print("="*60)

    # Compute energy functional
    def energy(u, L2d, eps):
        grad_term = 0.5 * u @ (L2d @ u)
        potential_term = np.sum((u**2 - 1)**2) / (4 * eps**2)
        return grad_term + potential_term / (N * N)  # normalized by grid points

    E_std = energy(u_std_final, L2d, eps)
    E_p = energy(u_p_final, L2d, eps)

    print(f"\n1. Energy comparison:")
    print(f"   E(u_std)  = {E_std:.10e}")
    print(f"   E(u_p)    = {E_p:.10e}")
    print(f"   |E_std - E_p| = {abs(E_std - E_p):.10e}")

    # Compute difference norms
    diff = u_std_final - u_p_final
    l2_diff = np.linalg.norm(diff)
    linf_diff = np.max(np.abs(diff))
    rel_l2_diff = l2_diff / np.linalg.norm(u_std_final)

    print(f"\n2. Solution difference:")
    print(f"   ||u_std - u_p||_2     = {l2_diff:.10e}")
    print(f"   ||u_std - u_p||_inf   = {linf_diff:.10e}")
    print(f"   Relative L2 error     = {rel_l2_diff:.10e}")

    # Check Morse index (number of negative eigenvalues)
    print(f"\n3. Morse index verification:")
    H_std = L2d + diags((3*u_std_final**2 - 1) / eps**2, 0, format='csc')
    H_p = L2d + diags((3*u_p_final**2 - 1) / eps**2, 0, format='csc')

    try:
        eigs_std = eigsh(H_std, k=5, which='SA', return_eigenvectors=False)
        eigs_p = eigsh(H_p, k=5, which='SA', return_eigenvectors=False)

        n_neg_std = np.sum(eigs_std < 0)
        n_neg_p = np.sum(eigs_p < 0)

        print(f"   Standard HiSD: smallest 5 eigenvalues = {eigs_std}")
        print(f"   p-HiSD:        smallest 5 eigenvalues = {eigs_p}")
        print(f"   Standard HiSD: Morse index = {n_neg_std}")
        print(f"   p-HiSD:        Morse index = {n_neg_p}")
    except Exception as e:
        print(f"   Warning: Could not compute eigenvalues: {e}")

    # Conclusion
    print(f"\n4. Conclusion:")
    if rel_l2_diff < 1e-3 and abs(E_std - E_p) < 1e-6:
        print(f"   ✓ Both methods converged to the SAME saddle point!")
        print(f"   ✓ Relative difference: {rel_l2_diff:.2e} < 0.1%")
    else:
        print(f"   ✗ Methods may have converged to DIFFERENT saddle points.")
        print(f"   ✗ Relative difference: {rel_l2_diff:.2e}")

    print("="*60 + "\n")

    # Save solutions for visualization
    np.savez('saddle_points.npz',
             u_std=u_std_final.reshape(N, N),
             u_p=u_p_final.reshape(N, N),
             diff=diff.reshape(N, N),
             X=X, Y=Y)
    print("Solutions saved to: saddle_points.npz")

    # ========= Plot (same style as original) =========
    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))

    iters_std = np.arange(len(gnorms_std))
    iters_p = np.arange(len(gnorms_p))

    ax.semilogy(iters_std, gnorms_std, color='#1f77b4', linewidth=1.5,
                label=f'HiSD ', zorder=2)
    ax.semilogy(iters_p, gnorms_p, color='#d62728', linewidth=1.5,
                label=f'p-HiSD ', zorder=3)

    if switch_iter is not None:
        ax.axvline(x=switch_iter, color='black', linestyle='--', linewidth=1.2, alpha=0.6)
        ax.text(switch_iter + 2, gnorms_p[switch_iter] * 2,
                f'Switch at iter {switch_iter}', fontsize=15, color='black',bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))

    ax.set_xlabel('Iteration $m$', fontsize=17)
    ax.set_ylabel('$\\|\\nabla E(u_m)\\|$', fontsize=17)
    ax.set_title('Allen-Cahn Equation ($\\xi=0.07$, $N=80$)', fontsize=17)
    ax.legend(fontsize=15)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # --- 新增的保存路径设置 ---
    output_dir = '/Users/huangbingzhang/Documents/code/python/pre-hisd/figures'
    os.makedirs(output_dir, exist_ok=True)  # 确保文件夹存在
    pdf_path = os.path.join(output_dir, '7.pdf')
    # --------------------------

    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved: {pdf_path}")


if __name__ == '__main__':
    run()
