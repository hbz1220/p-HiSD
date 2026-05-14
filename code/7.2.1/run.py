"""
Example 1: Butterfly Function (Figure 1)

E(x,y) = x^4 - 2x^2 + y^4 + y^2 - 3/2 x^2 y^2 + x^2 y - c y^3, c=1.
Compares Standard HiSD (cyan), Spectral p-HiSD (magenta),
Subspace-Inertial p-HiSD (orange).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.preconditioners import spectral_precond_custom, subspace_inertial_precond


def E_butterfly(x, y, c=1.0):
    return x**4 - 2*x**2 + y**4 + y**2 - 1.5*x**2*y**2 + x**2*y - c*y**3


def grad_butterfly(xy, c=1.0):
    x, y = xy
    dEdx = 4*x**3 - 4*x - 3*x*y**2 + 2*x*y
    dEdy = 4*y**3 + 2*y - 3*x**2*y + x**2 - 3*c*y**2
    return np.array([dEdx, dEdy])


def hess_butterfly(xy, c=1.0):
    x, y = xy
    H11 = 12*x**2 - 4 - 3*y**2 + 2*y
    H12 = -6*x*y + 2*x
    H22 = 12*y**2 + 2 - 3*x**2 - 6*c*y
    return np.array([[H11, H12], [H12, H22]])


def run_hisd_2d(x0, grad_fn, hess_fn, k, dt_x, dt_v, tol, max_iter,
                precond_fn=None, precond_kwargs=None, bound=5.0, v_iter=3):
    """Run HiSD on a 2D problem with trajectory recording and overflow protection."""
    from scipy.linalg import eigh
    x = np.array(x0, dtype=float)
    H = hess_fn(x)

    if precond_fn is not None:
        pc = precond_fn(H, **precond_kwargs) if precond_kwargs else precond_fn(H)
        eigvals, eigvecs = eigh(H, pc['mat'], subset_by_index=[0, 0])
        v = eigvecs[:, 0]
    else:
        pc = None
        eigvals, eigvecs = eigh(H, subset_by_index=[0, 0])
        v = eigvecs[:, 0]

    traj = [x.copy()]
    gnorms = []

    for it in range(max_iter):
        g = grad_fn(x)
        gn = np.linalg.norm(g)
        gnorms.append(gn)

        # Overflow / divergence check
        if not np.isfinite(gn) or np.max(np.abs(x)) > bound:
            break
        if gn < tol:
            break

        if pc is not None:
            try:
                g_tilde = pc['solve'](g)
                d = -g_tilde + 2.0 * v * (v @ g)
            except Exception:
                break
        else:
            d = -g + 2.0 * v * (v @ g)

        x = x + dt_x * d
        traj.append(x.copy())

        if not np.all(np.isfinite(x)):
            break

        H = hess_fn(x)
        if not np.all(np.isfinite(H)):
            break

        if pc is not None and 'update' in pc:
            try:
                pc = pc['update'](x, H)
            except Exception:
                break

        # v-update: J inner iterations of Rayleigh quotient minimization
        try:
            for _ in range(v_iter):
                if pc is not None:
                    Hv = pc['solve'](H @ v)
                    M_mat = pc['mat']
                    proj = Hv - v * (v @ (M_mat @ Hv))
                    v = v - dt_v * proj
                    norm_M = np.sqrt(max(v @ (M_mat @ v), 1e-30))
                    v = v / norm_M
                else:
                    Hv = H @ v
                    proj = Hv - v * (v @ Hv)
                    v = v - dt_v * proj
                    nm = np.linalg.norm(v)
                    if nm > 1e-15:
                        v = v / nm
        except Exception:
            break

    return np.array(traj), gnorms


def run():
    c = 1.0
    k = 1
    dt_x = 0.01
    dt_v = 0.05
    tol = 1e-6
    max_iter = 10000

    grad_fn = lambda xy: grad_butterfly(xy, c)
    hess_fn = lambda xy: hess_butterfly(xy, c)

    # --- Standard HiSD: start at (1.44, -0.95) (expected to fail / drift) ---
    traj_std, gn_std = run_hisd_2d(
        [1.44, -0.95], grad_fn, hess_fn, k, dt_x, dt_v, tol, max_iter, bound=4.0
    )
    print(f"Standard HiSD: {len(gn_std)} iters, final ||g||={gn_std[-1]:.2e}, diverged={np.max(np.abs(traj_std[-1]))>3}")

    # --- Spectral p-HiSD: start at (-1.44, -0.95) ---
    eps_spec = 1e-2
    traj_spec, gn_spec = run_hisd_2d(
        [-1.44, -0.95], grad_fn, hess_fn, k, dt_x, dt_v, tol, max_iter,
        precond_fn=lambda H: spectral_precond_custom(H, eps_spec)
    )
    print(f"Spectral p-HiSD: {len(gn_spec)} iters, final ||g||={gn_spec[-1]:.2e}")

    # --- Subspace-Inertial p-HiSD: start at (1.44, -0.95) ---
    from scipy.linalg import eigh as eigh_fn
    x = np.array([1.44, -0.95])
    H = hess_fn(x)
    pc = subspace_inertial_precond(H, k=1, alpha=0.7, a_weights=np.array([0.49]), eps=1e-2)
    eigvals, eigvecs = eigh_fn(H, pc['mat'], subset_by_index=[0, 0])
    v = eigvecs[:, 0]

    traj_si = [x.copy()]
    gn_si = []
    for it in range(max_iter):
        g = grad_fn(x)
        gn = np.linalg.norm(g)
        gn_si.append(gn)
        if not np.isfinite(gn) or np.max(np.abs(x)) > 5.0:
            break
        if gn < tol:
            break

        try:
            g_tilde = pc['solve'](g)
            d = -g_tilde + 2.0 * v * (v @ g)
        except Exception:
            break

        x = x + dt_x * d
        traj_si.append(x.copy())

        if not np.all(np.isfinite(x)):
            break

        H = hess_fn(x)
        if not np.all(np.isfinite(H)):
            break

        try:
            pc = pc['update'](x, H)
        except Exception:
            break

        try:
            for _ in range(3):  # inner iterations for v
                Hv = pc['solve'](H @ v)
                M_mat = pc['mat']
                proj = Hv - v * (v @ (M_mat @ Hv))
                v = v - dt_v * proj
                norm_M = np.sqrt(max(v @ (M_mat @ v), 1e-30))
                v = v / norm_M
        except Exception:
            break

    traj_si = np.array(traj_si)
    print(f"Subspace-Inertial p-HiSD: {len(gn_si)} iters, final ||g||={gn_si[-1]:.2e}")

    # --- Plot Figure 1 ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))

    xx = np.linspace(-2.0, 2.0, 400)
    yy = np.linspace(-1.8, 1.8, 400)
    X, Y = np.meshgrid(xx, yy)
    Z = E_butterfly(X, Y, c)
    Z_clip = np.clip(Z, -5, 10)

    levels = np.linspace(-5, 10, 40)
    ax.contourf(X, Y, Z_clip, levels=levels, cmap='RdYlBu_r', alpha=0.6)
    ax.contour(X, Y, Z_clip, levels=levels, colors='gray', linewidths=0.3, alpha=0.5)

    # Mask out-of-bounds segments to avoid messy connecting lines
    def mask_traj(traj, xlim=(-2.0, 2.0), ylim=(-1.8, 1.8)):
        t = traj.copy().astype(float)
        oob = (t[:, 0] < xlim[0]) | (t[:, 0] > xlim[1]) | \
              (t[:, 1] < ylim[0]) | (t[:, 1] > ylim[1])
        t[oob, :] = np.nan
        return t

    traj_std_m = mask_traj(traj_std)
    traj_spec_m = mask_traj(traj_spec)
    traj_si_m = mask_traj(traj_si)

    ax.plot(traj_std_m[:, 0], traj_std_m[:, 1], '-', color='cyan', linewidth=1.8,
            label='HiSD (fails)', zorder=3)
    ax.plot(traj_spec_m[:, 0], traj_spec_m[:, 1], '-', color='magenta', linewidth=1.8,
            label='Spectral p-HiSD', zorder=3)
    ax.plot(traj_si_m[:, 0], traj_si_m[:, 1], '-', color='orange', linewidth=1.8,
            label='Subspace-Inertial p-HiSD', zorder=3)

    # Starting points
    ax.plot(1.44, -0.95, 'ko', markersize=7, zorder=5)
    ax.plot(-1.44, -0.95, 'ko', markersize=7, zorder=5)

    # Find saddle points
    from scipy.optimize import fsolve
    saddle_found = []
    for x0_guess in [[0, 0.5], [0, -0.3], [1.0, 0.5], [-1, 0.5], [0.5, 0], [-0.5, 0]]:
        sol = fsolve(grad_fn, x0_guess, full_output=True)
        if sol[2] == 1:
            xsol = sol[0]
            if np.linalg.norm(grad_fn(xsol)) < 1e-8:
                H_sol = hess_fn(xsol)
                ev = np.linalg.eigvalsh(H_sol)
                if np.sum(ev < 0) == 1:
                    dup = False
                    for s in saddle_found:
                        if np.linalg.norm(xsol - s) < 0.1:
                            dup = True
                    if not dup:
                        saddle_found.append(xsol)

    for s in saddle_found:
        ax.plot(s[0], s[1], '*', color='lime', markersize=14, markeredgecolor='black',
                markeredgewidth=0.8, zorder=6)
        print(f"  Saddle found at ({s[0]:.4f}, {s[1]:.4f})")

    ax.set_xlabel('$x$', fontsize=15)
    ax.set_ylabel('$y$', fontsize=15)
    ax.set_title('Butterfly Function ($c=1$)', fontsize=15)
    ax.legend(fontsize=13, loc='upper right')
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-1.8, 1.8)
    ax.set_aspect('equal')

    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, '2.1.pdf'), dpi=300, bbox_inches='tight')
    #fig.savefig(os.path.join(out_dir, '1.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure 1 saved to {out_dir}/1.pdf")


if __name__ == '__main__':
    run()
