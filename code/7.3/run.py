# -*- coding: utf-8 -*-
import os, sys
from dataclasses import dataclass
from typing import Optional
# Avoid local copy.py shadowing stdlib copy (can break sympy/mpmath import)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR in sys.path: sys.path.remove(_THIS_DIR)
import copy as _stdlib_copy
sys.modules["copy"] = _stdlib_copy
if _THIS_DIR not in sys.path: sys.path.insert(0, _THIS_DIR)
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
@dataclass(frozen=True)
class ExperimentConfig:
    d: int = 1000
    k: int = 5
    s_neg: float = -50000.0
    s_pos: float = 1.0
    eps_M: float = 1e-2
    eta_x_std: float = 1e-5
    eta_v_std: float = 1e-5
    inner_steps_std: int = 5
    eta_x_ph: float = 0.2
    eta_v_ph: float = 1e-5
    inner_steps_ph: int = 5
    block_size: int = 20
    max_iter_std: int = 10000
    max_iter_ph: int = 10000
    plot_max_iter: int = 600
    tol_grad: float = 1e-6
    tol_dist: float = 1e-2
    diverge_norm: float = 1e8
    trust_max_step: Optional[float] = None
    rng_seed: int = 1
    fig_name: str = "3.pdf"
def grad(x, s):
    g = np.zeros_like(x)
    g[0] = -400.0 * x[0] * (x[1] - x[0] ** 2) - 2.0 * (1.0 - x[0])
    g[1:-1] = 200.0 * (x[1:-1] - x[:-2] ** 2) - 400.0 * x[1:-1] * (x[2:] - x[1:-1] ** 2) - 2.0 * (1.0 - x[1:-1])
    g[-1] = 200.0 * (x[-1] - x[-2] ** 2)
    z = x - 1.0
    g += 2.0 * s * np.arctan(z) / (1.0 + z ** 2)
    return g
def hess_diagonals(x, s):
    diag = np.empty(x.size, dtype=float)
    off = -400.0 * x[:-1]
    diag[0] = 1200.0 * x[0] ** 2 - 400.0 * x[1] + 2.0
    diag[1:-1] = 1200.0 * x[1:-1] ** 2 - 400.0 * x[2:] + 202.0
    diag[-1] = 200.0
    z = x - 1.0
    diag += 2.0 * s * (1.0 - 2.0 * z * np.arctan(z)) / (1.0 + z ** 2) ** 2
    return diag, off
def hvp(x, p, s):
    diag, off = hess_diagonals(x, s)
    y = diag * p
    y[:-1] += off * p[1:]
    y[1:] += off * p[:-1]
    return y
def build_sparse_hessian(x, s):
    diag, off = hess_diagonals(x, s)
    return sp.diags([off, diag, off], offsets=[-1, 0, 1], format="csr")
class FrozenSpectralPreconditioner:
    def __init__(self, H_ref, eps_M=1e-2):
        lam, self.Q = la.eigh(H_ref)
        self.weights = np.abs(lam) + eps_M
    def _q(self, v):
        return self.Q.T @ v
    def apply_M(self, v):
        qv = self._q(v)
        return self.Q @ (self.weights * qv)
    def solve_M(self, v):
        qv = self._q(v)
        return self.Q @ (qv / self.weights)
    def M_inner(self, u, v):
        return float((self.weights * self._q(u)) @ self._q(v))
    def M_norm(self, v):
        return float(np.sqrt(max(self.M_inner(v, v), 0.0)))
class FrozenBlockJacobiPreconditioner:
    def __init__(self, H_ref, block_size=20, eps_M=1e-2):
        self.block_size, self.blocks, self.block_Q, self.block_w = int(block_size), [], [], []
        for st in range(0, H_ref.shape[0], self.block_size):
            ed = min(st + self.block_size, H_ref.shape[0])
            lam, Q = la.eigh(H_ref[st:ed, st:ed])
            self.blocks.append((st, ed)); self.block_Q.append(Q); self.block_w.append(np.abs(lam) + eps_M)
        all_w = np.concatenate(self.block_w) if self.block_w else np.array([np.nan])
        self.min_weight, self.max_weight, self.num_blocks = float(np.nanmin(all_w)), float(np.nanmax(all_w)), len(self.blocks)
    def _transform(self, v, inv=False):
        y = np.zeros_like(v)
        for (st, ed), Q, w in zip(self.blocks, self.block_Q, self.block_w):
            qv = Q.T @ v[st:ed]
            y[st:ed] = Q @ (qv / w if inv else w * qv)
        return y
    def apply_M(self, v):
        return self._transform(v, inv=False)
    def solve_M(self, v):
        return self._transform(v, inv=True)
    def M_inner(self, u, v):
        return float(u @ self.apply_M(v))
    def M_norm(self, v):
        return float(np.sqrt(max(self.M_inner(v, v), 0.0)))
class FrozenScalarJacobiPreconditioner:
    def __init__(self, diag_ref, eps_M=1e-2):
        self.weights = np.abs(diag_ref.astype(float)) + float(eps_M)
    def apply_M(self, v):
        return self.weights * v
    def solve_M(self, v):
        return v / self.weights
    def M_inner(self, u, v):
        return float(np.dot(u, self.weights * v))
    def M_norm(self, v):
        return float(np.sqrt(self.M_inner(v, v)))
def euclidean_orthonormalize(V):
    return np.linalg.qr(V)[0][:, : V.shape[1]]
def M_orthonormalize(V, precon):
    n, k = V.shape
    Q = np.zeros((n, k), dtype=float)
    for i in range(k):
        vi = V[:, i].copy()
        for j in range(i): vi -= Q[:, j] * precon.M_inner(Q[:, j], vi)
        nrm = precon.M_norm(vi)
        if (not np.isfinite(nrm)) or nrm < 1e-14: return None
        Q[:, i] = vi / nrm
    return Q
def _finite(*arrs):
    return all(np.isfinite(a).all() for a in arrs)
def _clip(step, max_step):
    if max_step is None: return step, True
    nrm = np.linalg.norm(step)
    if not np.isfinite(nrm): return step, False
    if nrm > max_step: step = step / nrm * max_step
    return step, True
def _run_core(x0, V0, x_star, s, eta_x, eta_v, inner_steps, max_iter, tol_grad, tol_dist, precon=None, max_step=None, diverge_norm=1e8):
    k, x, V = V0.shape[1], x0.copy(), V0.copy()
    res_hist, dist_hist, status = [], [], "max_iter"
    for _it in range(max_iter):
        g = grad(x, s)
        res, dist = np.linalg.norm(g), np.linalg.norm(x - x_star)
        res_hist.append(res); dist_hist.append(dist)
        if (not np.isfinite(res)) or (not np.isfinite(dist)) or (not _finite(x, V, g)): status = "diverged"; break
        if res < tol_grad and dist < tol_dist: status = "converged"; break
        for _ in range(inner_steps):
            for i in range(k):
                y = hvp(x, V[:, i], s)
                if precon is not None: y = precon.solve_M(y)
                if not _finite(y): status = "diverged"; break
                if precon is None:
                    pi_y = y - V[:, i] * (V[:, i] @ y)
                    if i > 0: pi_y -= 2.0 * (V[:, :i] @ (V[:, :i].T @ y))
                else:
                    pi_y = y - V[:, i] * precon.M_inner(V[:, i], y)
                    if i > 0: pi_y -= 2.0 * (V[:, :i] @ np.array([precon.M_inner(V[:, j], y) for j in range(i)]))
                V[:, i] += -eta_v * pi_y
            if status == "diverged": break
            V = euclidean_orthonormalize(V) if precon is None else M_orthonormalize(V, precon)
            if V is None or (not _finite(V)): status = "diverged"; break
        if status == "diverged": break
        if precon is None:
            dx = -g + 2.0 * V @ (V.T @ g)
        else:
            Minv_g = precon.solve_M(g)
            if not _finite(Minv_g): status = "diverged"; break
            dx = -Minv_g + 2.0 * V @ (V.T @ g)
        step, ok = _clip(eta_x * dx, max_step)
        if (not ok) or (not _finite(step)): status = "diverged"; break
        x = x + step
        if (not _finite(x)) or (np.linalg.norm(x) > diverge_norm): status = "diverged"; break
    return {
        "x": x, "V": V, "res_hist": np.array(res_hist), "dist_hist": np.array(dist_hist),
        "status": status, "iterations": len(res_hist), "eta_x": eta_x, "eta_v": eta_v, "inner_steps": inner_steps,
    }
def _print_summary(name, out, s):
    H = build_sparse_hessian(out["x"], s).toarray()
    if not np.isfinite(H).all():
        idx, evals = None, None
    else:
        evals = la.eigvalsh(H)
        idx = int(np.sum(evals < -1e-6))
    print("-" * 90)
    print(f"Method: {name}")
    print(f"  eta_x={out['eta_x']}, eta_v={out['eta_v']}, inner_steps={out['inner_steps']}")
    print(f"  status={out['status']}, iterations={out['iterations']}")
    print(f"  final ||grad||={out['res_hist'][-1]:.6e}, final ||x-x_star||={out['dist_hist'][-1]:.6e}")
    if idx is None:
        print("  final Hessian index: N/A (non-finite Hessian)")
        print("  first 10 Hessian eigenvalues: N/A")
    else:
        print(f"  final Hessian index: {idx}")
        print(f"  first 10 Hessian eigenvalues: {evals[:10]}")
def plot_results(out_std, method_series, save_path, plot_max_iter):
    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))
    std_hist = out_std["res_hist"][:plot_max_iter]
    ax.semilogy(np.arange(len(std_hist)), std_hist, lw=2.0, label="HiSD")
    for label, out in method_series:
        hist = out["res_hist"][:plot_max_iter]
        ax.semilogy(np.arange(len(hist)), hist, lw=2.0, label=label)
    ax.set_xlabel('Iteration $m$', fontsize=17); ax.set_ylabel('$\\|\\nabla E(x_m)\\|$', fontsize=17)
    ax.set_title('Modified Rosenbrock Function', fontsize=17)
    ax.tick_params(axis="both", labelsize=17)
    ax.set_ylim(1e-7, 1e3); ax.set_xlim(0, plot_max_iter); ax.grid(True, alpha=0.3); ax.legend(fontsize=15, loc='lower right')
    plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {save_path}")
def main():
    cfg = ExperimentConfig()
    x_star = np.ones(cfg.d)
    s = np.full(cfg.d, cfg.s_pos); s[: cfg.k] = cfg.s_neg
    rng = np.random.default_rng(cfg.rng_seed)
    n = rng.standard_normal(cfg.d)
    x0 = x_star + 0.1 * n / np.linalg.norm(n)
    H0_sparse = build_sparse_hessian(x0, s)
    evals0, evecs0 = spla.eigsh(H0_sparse, k=cfg.k, which="SA")
    V0_raw = evecs0[:, np.argsort(evals0)]
    V0_std = euclidean_orthonormalize(V0_raw)
    H_ref = build_sparse_hessian(x0, s).toarray()
    precon_spec = FrozenSpectralPreconditioner(H_ref, eps_M=cfg.eps_M)
    precon_jac = FrozenScalarJacobiPreconditioner(hess_diagonals(x0, s)[0], eps_M=cfg.eps_M)
    precon_blk = FrozenBlockJacobiPreconditioner(H_ref, block_size=cfg.block_size, eps_M=cfg.eps_M)
    print("Jacobi preconditioner:\n"
          f"  min(weight)={precon_jac.weights.min():.6e}\n"
          f"  max(weight)={precon_jac.weights.max():.6e}")
    print("Block-Jacobi preconditioner: "
          f"block_size={precon_blk.block_size}, number_of_blocks={precon_blk.num_blocks}, "
          f"min_local_weight={precon_blk.min_weight:.6e}, max_local_weight={precon_blk.max_weight:.6e}")
    V0_spec = M_orthonormalize(V0_raw, precon_spec)
    V0_jac = M_orthonormalize(V0_raw, precon_jac)
    V0_blk = M_orthonormalize(V0_raw, precon_blk)
    if V0_spec is None: raise RuntimeError("Spectral: initial M-orthonormalization failed.")
    if V0_jac is None: raise RuntimeError("Jacobi: initial M-orthonormalization failed.")
    if V0_blk is None: raise RuntimeError("Block-Jacobi: initial M-orthonormalization failed.")
    def _run(name, v0, eta_x, eta_v, inner_steps, max_iter, precon=None):
        out = _run_core(
            x0, v0, x_star, s,
            eta_x=eta_x, eta_v=eta_v, inner_steps=inner_steps, max_iter=max_iter,
            tol_grad=cfg.tol_grad, tol_dist=cfg.tol_dist,
            precon=precon, max_step=(cfg.trust_max_step if precon is not None else None),
            diverge_norm=cfg.diverge_norm,
        )
        print(f"{name}: status={out['status']}, iters={out['iterations']}, "
              f"final_res={out['res_hist'][-1]:.3e}, final_dist={out['dist_hist'][-1]:.3e}")
        return out
    out_std = _run("Standard HiSD", V0_std, cfg.eta_x_std, cfg.eta_v_std, cfg.inner_steps_std, cfg.max_iter_std)
    method_specs = [
        ("Spectral", precon_spec, V0_spec),
        ("Jacobi", precon_jac, V0_jac),
        ("Block Jacobi", precon_blk, V0_blk),
    ]
    method_outputs = [
        (label, _run(label, v0, cfg.eta_x_ph, cfg.eta_v_ph, cfg.inner_steps_ph, cfg.max_iter_ph, precon=precon))
        for label, precon, v0 in method_specs
    ]
    _print_summary("Standard HiSD", out_std, s)
    for label, out in method_outputs: _print_summary(label, out, s)
    out_dir = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "figures")
    )
    os.makedirs(out_dir, exist_ok=True)

    save_path = os.path.join(out_dir, cfg.fig_name)
    plot_results(out_std, method_outputs, save_path, cfg.plot_max_iter)
if __name__ == "__main__":
    main()
