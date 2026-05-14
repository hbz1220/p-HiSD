import argparse, time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import root

# ============================================================
# User-adjustable parameters
# ============================================================
# Discretization / model
H_GRID = 2.0**-8
N_GRID = 255
LAMBDA = 0.02

# Initial condition
U0_AMPLITUDE = 0.5
U0_MODE = 1

# Stopping / numerical differentiation
TOL = 1e-6
MAX_ITER = 10000
PLOT_MAX_ITER = 300
HV_EPS = 1e-5

# State solve
STATE_TOL = 1e-10
STATE_MAXIT = 30

# Divergence safety
DIVERGE_GRAD = 1e8
DIVERGE_UINF = 1e4

# Standard HiSD steps
STANDARD_ETA = 1e-4
COMMON_TAU = 0.001
DIRECTION_ITERS = 5

# p-HiSD step size shared by all p-HiSD methods
PHISD_ETA = 1

# Method switches
INCLUDE_H1 = True
INCLUDE_SHIFTED_CHOLESKY = True
INCLUDE_FROZEN_SPECTRAL = True

# Shifted-Cholesky settings
SHIFT_FACTOR = 2.0

# Regularization / SPD shifts
EPS_HESSIAN_FACTOR = 1e-6
EPS_DENSE_METRIC_FACTOR = 1e-12

# Output
OUTPUT_FIG = "8.pdf"

h, N, lam = H_GRID, N_GRID, LAMBDA
x = np.linspace(h, 1.0 - h, N)
yd = -2.0 * np.sin(np.pi * x)
L = sp.diags([(-1 / h**2) * np.ones(N - 1), (2 / h**2) * np.ones(N), (-1 / h**2) * np.ones(N - 1)], [-1, 0, 1], format="csr")
I = sp.eye(N, format="csr")
LI = (L + I).tocsr()
H1 = LI.tocsr()
LABELS = {
    "standard_hisd": "HiSD",
    "p_hisd_H1": "Laplacian",
    "p_hisd_shifted_cholesky": "shifted Cholesky",
    "p_hisd_frozen_spectral": "Spectral",
}
IDM = {"kind": "I", "solve": lambda b: np.asarray(b, float), "apply": lambda b: np.asarray(b, float), "dense": np.eye(N)}

def rhs_control(u): return 0.001 * u**2 + np.cos(2 * np.pi * u)
def rhs_prime(u): return 0.002 * u - 2 * np.pi * np.sin(2 * np.pi * u)
def rhs_second(u): return 0.002 - 4 * (np.pi**2) * np.cos(2 * np.pi * u)

def solve_state(u, y0=None, tol=STATE_TOL, maxit=STATE_MAXIT):
    u, y = np.asarray(u, float), (np.zeros(N) if y0 is None else np.asarray(y0, float).copy())
    for _ in range(maxit):
        F = (L @ y) + y + y**3 - rhs_control(u)
        if np.linalg.norm(F, np.inf) < tol: return y
        J = (LI + sp.diags(3 * y**2, 0, format="csr")).tocsc(); dy = spla.spsolve(J, -F)
        if not np.all(np.isfinite(dy)): break
        y += dy
        if np.linalg.norm(dy, np.inf) <= tol * (1 + np.linalg.norm(y, np.inf)): return y
    sol = root(lambda yy: (L @ yy) + yy + yy**3 - rhs_control(u), y,
               jac=lambda yy: (LI + sp.diags(3 * yy**2, 0, format="csr")).toarray(), method="hybr", tol=tol)
    if not sol.success: raise RuntimeError(f"State solve failed: {sol.message}")
    return np.asarray(sol.x, float)

def solve_adjoint(y):
    A = (LI + sp.diags(3 * y**2, 0, format="csr")).tocsc()
    return np.asarray(spla.spsolve(A, y - yd), float)

def grad_full(u, y0=None):
    y = solve_state(u, y0=y0); p = solve_adjoint(y)
    return y, p, np.asarray(lam * (LI @ u) + p * rhs_prime(u), float)

def cost(u, y):
    e = y - yd
    return 0.5 * h * float(e @ e) + 0.5 * lam * h * float(u @ (LI @ u))

def hvp(u, v, eps, y_ref=None):
    v = np.asarray(v, float); nv = float(np.linalg.norm(v))
    if nv <= 1e-16: return np.zeros_like(v)
    vh = v / nv; y_ref = solve_state(u, y0=np.zeros(N)) if y_ref is None else y_ref
    gp = grad_full(u + eps * vh, y0=y_ref)[2]; gm = grad_full(u - eps * vh, y0=y_ref)[2]
    return nv * (gp - gm) / (2 * eps)

def dense_hessian(u, eps):
    H, E = np.zeros((N, N)), np.eye(N); yref = solve_state(u, y0=np.zeros(N))
    for j in range(N): H[:, j] = hvp(u, E[:, j], eps, y_ref=yref)
    return 0.5 * (H + H.T)

def make_metric(A=None, kind="I"):
    if kind == "I": return IDM
    if kind == "sparse":
        Ac = A.tocsc(); solver = spla.factorized(Ac)
        return {"kind": "sparse", "solve": lambda b, s=solver: np.asarray(s(b), float), "apply": lambda b, M=Ac: np.asarray(M @ b, float), "dense": Ac.toarray()}
    A0 = 0.5 * (np.asarray(A, float) + np.asarray(A, float).T); step, shift = EPS_DENSE_METRIC_FACTOR * max(1.0, float(np.linalg.norm(A0, 2))), 0.0
    while True:
        At = A0 + shift * np.eye(A0.shape[0])
        try:
            cho = la.cho_factor(At, lower=True, check_finite=False)
            return {"kind": "dense", "solve": lambda b, c=cho: la.cho_solve(c, b, check_finite=False), "apply": lambda b, M=At: M @ b, "dense": At}
        except la.LinAlgError:
            shift = step if shift == 0.0 else 10.0 * shift
            if shift > 1e6: raise RuntimeError("Failed SPD dense metric")

def normalize(v, m):
    v = np.asarray(v, float)
    if m["kind"] == "I":
        n = float(np.linalg.norm(v))
        if n <= 1e-16: raise RuntimeError("Euclidean normalization failed")
        return v / n
    q = float(v @ m["apply"](v))
    if q <= 1e-18: raise RuntimeError("M-normalization failed")
    return v / np.sqrt(q)

def kappa(vals):
    a = np.abs(np.asarray(vals, float)); mx = float(np.max(a)); nz = a[a > 1e-12 * max(1.0, mx)]
    return (np.inf if nz.size == 0 else float(mx / np.min(nz))), mx

def make_method(name, metric, standard, H0, eta, tau, block_size=-1, alpha=np.nan):
    if standard:
        vals, vecs = np.linalg.eigh(H0); i = int(np.argmin(vals)); v0 = normalize(vecs[:, i], IDM); ge, Lm = np.asarray(vals, float), float(np.max(np.abs(vals)))
    else:
        ge, V = la.eigh(H0, metric["dense"]); i = int(np.argmin(ge)); v0 = normalize(V[:, i], metric); ge = np.asarray(ge, float); _, Lm = kappa(ge)
    kap, _ = kappa(ge)
    return {"name": name, "label": LABELS[name], "standard": standard, "solve": metric["solve"], "apply": metric["apply"], "v0": v0,
            "eta": float(eta), "tau": float(tau), "kappa_M0": float(kap), "L_M0": float(Lm), "block_size": int(block_size), "alpha": float(alpha)}

def final_index(u, eps):
    ev = np.linalg.eigvalsh(dense_hessian(u, eps)); mx = float(np.max(np.abs(ev))); th = 1e-6 * max(1.0, mx)
    return int(np.sum(ev < -th)), int(np.sum(np.abs(ev) <= th)), float(np.min(ev)), mx

def run(m, u0, tol, max_iter, hv_eps, direction_iters):
    u, v = np.asarray(u0, float).copy(), np.asarray(m["v0"], float).copy()
    eta, tau, t0, hist, status, conv = m["eta"], m["tau"], time.perf_counter(), [], "max_iter", False
    y = np.zeros_like(u)
    for it in range(max_iter + 1):
        y, _, g = grad_full(u); gn, J = float(np.linalg.norm(g)), float(cost(u, y))
        hist.append({"iter": it, "time": float(time.perf_counter() - t0), "grad_norm": gn})
        if (not np.isfinite(gn)) or (not np.isfinite(J)) or gn > DIVERGE_GRAD or float(np.linalg.norm(u, np.inf)) > DIVERGE_UINF: status = "diverged"; break
        if gn <= tol: status, conv = "converged", True; break
        if it == max_iter: break
        if m["standard"]:
            for _ in range(direction_iters):
                Hv = hvp(u, v, hv_eps, y_ref=y)
                ray = float(v @ Hv)
                v = normalize(v - tau * (Hv - v * ray), IDM)
            d = -g + 2 * v * float(v @ g)
        else:
            for _ in range(direction_iters):
                Hv = hvp(u, v, hv_eps, y_ref=y)
                z = m["solve"](Hv)
                ray = float(v @ Hv)
                v = normalize(v - tau * (z - v * ray), {"kind": "M", "apply": m["apply"]})
            d = -m["solve"](g) + 2 * v * float(v @ g)
        u = u + eta * d
        if (not np.all(np.isfinite(u))) or (not np.all(np.isfinite(v))): status = "diverged"; break
    fi, _, _, _ = final_index(u, hv_eps)
    return {"method": m["name"], "hist": hist, "u": u, "y": y, "grad": gn, "J": J, "iter": it, "status": status, "converged": conv, "final_index": fi, "time": float(hist[-1]["time"])}

def fmt_kappa(k): return f"{k:.2e}" if k >= 100 else f"{k:.2f}"

def plot_grad(outputs, tol, path, plot_max_iter):
    plt.figure(figsize=(8, 5.5))
    for o in outputs:
        it = np.array([r["iter"] for r in o["hist"]], float); gn = np.array([r["grad_norm"] for r in o["hist"]], float)
        mk = it <= float(plot_max_iter)
        it, gn = it[mk], gn[mk]
        if it.size == 0: continue
        plt.semilogy(it, gn, lw=2.0, label=f"{LABELS[o['method']]}")
    plt.xlabel("Iteration $m$", fontsize=17); plt.ylabel(r"$\|\nabla \hat{J}(u_m)\|_2$", fontsize=17); plt.yscale("log"); plt.title("Non-convex Optimal Control Problem", fontsize=17)
    plt.tick_params(axis="both", labelsize=17)
    plt.grid(True, ls="--", alpha=0.5); plt.legend(fontsize=15, loc="upper right", bbox_to_anchor=(1, 0.6)); plt.tight_layout(); plt.savefig(path, format="pdf", bbox_inches="tight"); plt.close()

def main():
    ap = argparse.ArgumentParser(description="OCP Example 4.1")
    ap.add_argument("--tol", type=float, default=TOL)
    ap.add_argument("--max-iter", type=int, default=MAX_ITER)
    ap.add_argument("--plot-max-iter", type=int, default=PLOT_MAX_ITER)
    ap.add_argument("--hv-eps", type=float, default=HV_EPS)
    ap.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / ".." / ".." / "figures")
    )
    a = ap.parse_args()
    outdir = Path(a.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print("Active configuration:")
    print(f"  N = {N_GRID}")
    print(f"  h = {H_GRID}")
    print(f"  lambda = {LAMBDA}")
    print(f"  u0 amplitude = {U0_AMPLITUDE}")
    print(f"  tol = {a.tol}")
    print(f"  max_iter = {a.max_iter}")
    print(f"  plot_max_iter = {a.plot_max_iter}")
    print(f"  hv_eps = {a.hv_eps}")
    print(f"  standard eta = {STANDARD_ETA}")
    print(f"  common tau = {COMMON_TAU}")
    print(f"  direction_iters = {DIRECTION_ITERS}")
    print(f"  p-HiSD eta = {PHISD_ETA}")
    print(f"  include H1/shifted-Chol/spectral = {INCLUDE_H1}/{INCLUDE_SHIFTED_CHOLESKY}/{INCLUDE_FROZEN_SPECTRAL}")
    print(f"  shifted factor = {SHIFT_FACTOR}")

    u0 = U0_AMPLITUDE * np.sin(2 * np.pi * U0_MODE * x); _, _, g0 = grad_full(u0)
    print("Building dense reduced Hessian H0 ...")
    H0 = dense_hessian(u0, a.hv_eps); eig_ord, Q0 = np.linalg.eigh(H0)
    print(f"Initial ||grad(u0)||_2 = {float(np.linalg.norm(g0)):.6e}")
    print(f"Initial ordinary negative count = {int(np.sum(eig_ord < 0.0))}")

    lam_min = float(np.min(eig_ord))
    h0_norm = float(np.max(np.abs(eig_ord)))
    methods = [make_method("standard_hisd", IDM, True, H0, eta=STANDARD_ETA, tau=COMMON_TAU)]
    if INCLUDE_H1:
        methods.append(make_method("p_hisd_H1", make_metric(H1, "sparse"), False, H0, eta=PHISD_ETA, tau=COMMON_TAU))
    if INCLUDE_SHIFTED_CHOLESKY:
        delta = SHIFT_FACTOR * max(0.0, -lam_min) + EPS_HESSIAN_FACTOR * max(1.0, h0_norm)
        M_shifted_chol = 0.5 * (H0 + delta * np.eye(N) + (H0 + delta * np.eye(N)).T)
        methods.append(
            make_method(
                "p_hisd_shifted_cholesky",
                make_metric(M_shifted_chol, "dense"),
                False,
                H0,
                eta=PHISD_ETA,
                tau=COMMON_TAU,
            )
        )
    if INCLUDE_FROZEN_SPECTRAL:
        eps_spec = EPS_HESSIAN_FACTOR * max(1.0, h0_norm)
        M_spec = Q0 @ np.diag(np.abs(eig_ord) + eps_spec) @ Q0.T
        M_spec = 0.5 * (M_spec + M_spec.T)
        methods.append(
            make_method(
                "p_hisd_frozen_spectral",
                make_metric(M_spec, "dense"),
                False,
                H0,
                eta=PHISD_ETA,
                tau=COMMON_TAU,
            )
        )
    print("\nMethod step setup:")
    for m in methods:
        print(f"  {LABELS[m['name']]}: eta={m['eta']:.6g}, tau={m['tau']:.6g}, direction_iters={DIRECTION_ITERS}, kappa_M0={fmt_kappa(m['kappa_M0'])}, L_M0={m['L_M0']:.2e}")
    print(f"\nRuntime policy: max_iter budget={a.max_iter}, one state update per outer iteration, {DIRECTION_ITERS} direction updates per outer iteration")

    outs = [run(m, u0, a.tol, a.max_iter, a.hv_eps, DIRECTION_ITERS) for m in methods]

    print("\nMethod | kappa_M0 | iterations | final_grad | final_index | status")
    mm = {m["name"]: m for m in methods}
    for o in outs:
        print(f"{LABELS[o['method']]}: kappa_M0={fmt_kappa(mm[o['method']]['kappa_M0'])}, iter={o['iter']}, grad={o['grad']:.2e}, index={o['final_index']}, status={o['status']}")

    fig = outdir / OUTPUT_FIG; plot_grad(outs, a.tol, fig, a.plot_max_iter)
    print("\nSaved file:")
    print(f"  {fig}")
    print(f"  Plot includes iter <= {a.plot_max_iter}")

if __name__ == "__main__":
    main()
