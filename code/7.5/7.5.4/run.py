#!/usr/bin/env python3
# Section 7.5.4: full-Q Landau-de Gennes dynamics toward the index-1 WORS saddle.
# Read reference data, the common initial state and settings from LDG_inputs.npz.
# Save Figure 8, Table 7 data and certification summaries to outputs/7.5/7.5.4/.

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext, redirect_stdout
import csv
import ctypes
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass
from typing import Any


HERE = Path(__file__).resolve().parent
INPUT_PATH = HERE / 'LDG_inputs.npz'
OUTPUT_DIR = (HERE / '..' / '..' /'..' / 'outputs' / '7.5' / '7.5.4').resolve()


def _select_python_for_editor_run() -> None:
    missing = [name for name in ('numpy', 'scipy', 'matplotlib', 'threadpoolctl') if importlib.util.find_spec(name) is None]
    if not missing:
        return
    for parent in (HERE, *HERE.parents):
        venv = parent / '.shared_venv'
        executable = venv / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        if executable.is_file():
            if Path(sys.prefix).absolute() != venv.absolute():
                print(f'Using the existing Python environment: {executable}', flush=True)
                completed = subprocess.run([str(executable), '-u', str(Path(__file__).resolve()), *sys.argv[1:]])
                raise SystemExit(completed.returncode)
            break
    if missing:
        raise SystemExit('Missing Python packages: ' + ', '.join(missing) + '. Select an interpreter with these packages installed.')


if __name__ == '__main__':
    _select_python_for_editor_run()

THREAD_ENV = {
    'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1',
}
os.environ.update(THREAD_ENV)
sys.dont_write_bytecode = True
_MPL_TEMP = tempfile.TemporaryDirectory(prefix='ldg-mpl-')
os.environ['MPLCONFIGDIR'] = _MPL_TEMP.name

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.linalg import eigh
from scipy.sparse import bmat, csc_matrix, diags, eye, kron
from scipy.sparse.linalg import LinearOperator, eigsh, splu


PDF_PATH = OUTPUT_DIR / '5.4.pdf'
OUTPUT_NAMES = ('5.4.pdf', 'LDG_results.csv', 'LDG_summary.json')


def _basis() -> np.ndarray:
    """Return the Frobenius-orthonormal basis E1,...,E5."""

    rt2 = math.sqrt(2.0)
    rt6 = math.sqrt(6.0)
    return np.asarray(
        [
            [[1.0 / rt2, 0.0, 0.0], [0.0, -1.0 / rt2, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 1.0 / rt2, 0.0], [1.0 / rt2, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[-1.0 / rt6, 0.0, 0.0], [0.0, -1.0 / rt6, 0.0], [0.0, 0.0, 2.0 / rt6]],
            [[0.0, 0.0, 1.0 / rt2], [0.0, 0.0, 0.0], [1.0 / rt2, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0 / rt2], [0.0, 1.0 / rt2, 0.0]],
        ],
        dtype=float,
    )


BASIS = _basis()


BASIS_NAMES = np.asarray(["E1", "E2", "E3", "E4", "E5"])


SOURCE_Q_TO_U_FACTORS = np.asarray(
    [math.sqrt(2.0), math.sqrt(2.0), math.sqrt(6.0), math.sqrt(2.0), math.sqrt(2.0)]
)


def u_to_Q(u: np.ndarray) -> np.ndarray:
    """Convert last-axis orthonormal coordinates (...,5) to (...,3,3)."""

    u = np.asarray(u, dtype=float)
    if u.shape[-1] != 5:
        raise ValueError(f"expected last coordinate axis of length 5, got {u.shape}")
    return np.einsum("...a,aij->...ij", u, BASIS, optimize=True)


def Q_to_u(Q: np.ndarray) -> np.ndarray:
    """Project symmetric traceless tensors (...,3,3) onto E1,...,E5."""

    Q = np.asarray(Q, dtype=float)
    if Q.shape[-2:] != (3, 3):
        raise ValueError(f"expected trailing tensor axes (3,3), got {Q.shape}")
    return np.einsum("...ij,aij->...a", Q, BASIS, optimize=True)


def source_q_to_u(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    if q.shape[-1] != 5:
        raise ValueError("source q coordinate axis must have length five")
    return q * SOURCE_Q_TO_U_FACTORS


def u_to_source_q(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    if u.shape[-1] != 5:
        raise ValueError("orthonormal coordinate axis must have length five")
    return u / SOURCE_Q_TO_U_FACTORS


@dataclass(frozen=True)
class Config:
    N: int = 41
    B: float = 0.64e4
    C: float = 0.35e4
    L: float = 4.0e-11
    lambda_sq: float = 7.0
    epsilon_bc: float = 0.1
    seed: int = 20260905
    newton_tolerance_l2: float = 1.0e-11
    gate2_tolerance_l2: float = 1.0e-9
    newton_max_iterations: int = 40
    eig_tolerance: float = 1.0e-10
    eig_max_iterations: int = 100000
    eigenpairs: int = 6

    @property
    def A(self) -> float:
        return -(self.B**2) / (3.0 * self.C)

    @property
    def s_plus(self) -> float:
        return self.B / self.C


class FullLdGModel:
    """Five-component full-Q finite-difference model with fixed Dirichlet data."""

    def __init__(self, cfg: Config):
        if cfg.N != 41:
            raise ValueError("This experiment is fixed at N=41 total grid points")
        if not (0.0 < cfg.epsilon_bc <= 1.0):
            raise ValueError("epsilon_bc must lie in (0,1]")
        self.cfg = cfg
        self.N = cfg.N
        self.ni = cfg.N - 2
        self.m = self.ni**2
        self.dofs = 5 * self.m
        self.h = 2.0 / (cfg.N - 1)
        self.coordinates = np.linspace(-1.0, 1.0, cfg.N)
        self.X, self.Y = np.meshgrid(self.coordinates, self.coordinates, indexing="xy")
        self.identity3 = np.eye(3)
        self.alpha = cfg.A / (2.0 * cfg.C)
        self.beta = cfg.B / (2.0 * cfg.C)

        main = 2.0 * np.ones(self.ni)
        off = -np.ones(self.ni - 1)
        T = diags([off, main, off], [-1, 0, 1], format="csc") / self.h**2
        I = eye(self.ni, format="csc")
        self.A_h = csc_matrix(kron(I, T, format="csc") + kron(T, I, format="csc"))

        # Boundary tensors are fixed; all five components evolve only at interior nodes.
        self.boundary_Q = np.zeros((self.N, self.N, 3, 3), dtype=float)
        for iy, y in enumerate(self.coordinates):
            Tv = self.T_epsilon(y)
            Qv = (cfg.s_plus / 3.0) * np.diag([-Tv, 2.0 * Tv, -Tv])
            self.boundary_Q[iy, 0] = Qv
            self.boundary_Q[iy, -1] = Qv
        for ix, x in enumerate(self.coordinates):
            Th = self.T_epsilon(x)
            Qh = (cfg.s_plus / 3.0) * np.diag([2.0 * Th, -Th, -Th])
            self.boundary_Q[0, ix] = Qh
            self.boundary_Q[-1, ix] = Qh
        self.boundary_u = np.moveaxis(Q_to_u(self.boundary_Q), -1, 0)
        self.boundary_load = np.vstack(
            [self._boundary_load(self.boundary_u[a]) for a in range(5)]
        )

    def T_epsilon(self, t: float | np.ndarray) -> float | np.ndarray:
        t_arr = np.asarray(t, dtype=float)
        eps = self.cfg.epsilon_bc
        out = np.ones_like(t_arr)
        left = t_arr <= -1.0 + eps
        right = t_arr >= 1.0 - eps
        out[left] = (1.0 + t_arr[left]) / eps
        out[right] = (1.0 - t_arr[right]) / eps
        out = np.clip(out, 0.0, 1.0)
        if np.ndim(t) == 0:
            return float(out)
        return out

    def _boundary_load(self, boundary: np.ndarray) -> np.ndarray:
        load = np.zeros((self.ni, self.ni), dtype=float)
        load[:, 0] += boundary[1:-1, 0] / self.h**2
        load[:, -1] += boundary[1:-1, -1] / self.h**2
        load[0, :] += boundary[0, 1:-1] / self.h**2
        load[-1, :] += boundary[-1, 1:-1] / self.h**2
        return load.ravel(order="C")

    def split(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=float)
        if state.shape != (self.dofs,):
            raise ValueError(f"expected state shape {(self.dofs,)}, got {state.shape}")
        return state.reshape(5, self.m)

    def join(self, components: np.ndarray) -> np.ndarray:
        components = np.asarray(components, dtype=float)
        if components.shape != (5, self.m):
            raise ValueError(f"expected component array {(5, self.m)}, got {components.shape}")
        return components.reshape(self.dofs)

    def tensor_nodes(self, state: np.ndarray) -> np.ndarray:
        return u_to_Q(self.split(state).T)

    def full_u_fields(self, state: np.ndarray) -> np.ndarray:
        fields = self.boundary_u.copy()
        fields[:, 1:-1, 1:-1] = self.split(state).reshape(5, self.ni, self.ni)
        return fields

    def full_Q_field(self, state: np.ndarray) -> np.ndarray:
        return u_to_Q(np.moveaxis(self.full_u_fields(state), 0, -1))

    def bulk_potential(self, Q: np.ndarray) -> np.ndarray:
        s = np.einsum("...ij,...ij->...", Q, Q, optimize=True)
        Q2 = np.matmul(Q, Q)
        trQ3 = np.einsum("...ij,...ji->...", Q2, Q, optimize=True)
        return (
            self.cfg.A / (4.0 * self.cfg.C) * s
            - self.cfg.B / (6.0 * self.cfg.C) * trQ3
            + 0.125 * s**2
        )

    def bulk_gradient_coordinates(self, components: np.ndarray) -> np.ndarray:
        Q = u_to_Q(components.T)
        s = np.einsum("nij,nij->n", Q, Q, optimize=True)
        Q2 = np.matmul(Q, Q)
        G = (
            self.alpha * Q
            - self.beta * (Q2 - (s / 3.0)[:, None, None] * self.identity3)
            + 0.5 * s[:, None, None] * Q
        )
        return np.einsum("aij,nij->an", BASIS, G, optimize=True)

    def bulk_hvp_coordinates(
        self, components: np.ndarray, perturbation: np.ndarray
    ) -> np.ndarray:
        Q = u_to_Q(components.T)
        W = u_to_Q(perturbation.T)
        s = np.einsum("nij,nij->n", Q, Q, optimize=True)
        q_dot_w = np.einsum("nij,nij->n", Q, W, optimize=True)
        QW = np.matmul(Q, W)
        WQ = np.matmul(W, Q)
        HW = (
            self.alpha * W
            - self.beta
            * (QW + WQ - (2.0 / 3.0) * q_dot_w[:, None, None] * self.identity3)
            + q_dot_w[:, None, None] * Q
            + 0.5 * s[:, None, None] * W
        )
        return np.einsum("aij,nij->an", BASIS, HW, optimize=True)

    def local_bulk_hessian(self, components: np.ndarray) -> np.ndarray:
        blocks = np.empty((self.m, 5, 5), dtype=float)
        for b in range(5):
            direction = np.zeros_like(components)
            direction[b, :] = 1.0
            blocks[:, :, b] = self.bulk_hvp_coordinates(components, direction).T
        return blocks

    def energy(self, state: np.ndarray, lambda_sq: float | None = None) -> float:
        lam = self.cfg.lambda_sq if lambda_sq is None else float(lambda_sq)
        U = self.split(state)
        elastic = 0.0
        for a in range(5):
            elastic += 0.5 * float(U[a] @ (self.A_h @ U[a]))
            elastic -= float(self.boundary_load[a] @ U[a])
        return elastic + lam * float(np.sum(self.bulk_potential(u_to_Q(U.T))))

    def gradient(self, state: np.ndarray, lambda_sq: float | None = None) -> np.ndarray:
        lam = self.cfg.lambda_sq if lambda_sq is None else float(lambda_sq)
        U = self.split(state)
        G = np.vstack([self.A_h @ U[a] - self.boundary_load[a] for a in range(5)])
        G += lam * self.bulk_gradient_coordinates(U)
        return self.join(G)

    # Combine scalar Laplacian actions with the analytic local bulk Hessian action.
    def hvp(
        self, state: np.ndarray, vector: np.ndarray, lambda_sq: float | None = None
    ) -> np.ndarray:
        lam = self.cfg.lambda_sq if lambda_sq is None else float(lambda_sq)
        U = self.split(state)
        W = self.split(vector)
        HW = np.vstack([self.A_h @ W[a] for a in range(5)])
        HW += lam * self.bulk_hvp_coordinates(U, W)
        return self.join(HW)

    def hessian(self, state: np.ndarray, lambda_sq: float | None = None) -> csc_matrix:
        lam = self.cfg.lambda_sq if lambda_sq is None else float(lambda_sq)
        local = self.local_bulk_hessian(self.split(state))
        blocks: list[list[csc_matrix]] = []
        for a in range(5):
            row: list[csc_matrix] = []
            for b in range(5):
                block = diags(lam * local[:, a, b], format="csc")
                if a == b:
                    block = self.A_h + block
                row.append(csc_matrix(block))
            blocks.append(row)
        return csc_matrix(bmat(blocks, format="csc"))

    def restricted_hessian(
        self, state: np.ndarray, lambda_sq: float, indices: tuple[int, ...] = (0, 2)
    ) -> csc_matrix:
        local = self.local_bulk_hessian(self.split(state))
        blocks: list[list[csc_matrix]] = []
        for a in indices:
            row: list[csc_matrix] = []
            for b in indices:
                block = diags(lambda_sq * local[:, a, b], format="csc")
                if a == b:
                    block = self.A_h + block
                row.append(csc_matrix(block))
            blocks.append(row)
        return csc_matrix(bmat(blocks, format="csc"))

    def residual_l2(self, state: np.ndarray, lambda_sq: float | None = None) -> float:
        return self.h * float(np.linalg.norm(self.gradient(state, lambda_sq)))


# Acceptance uses the state residual, frame residual, normalization and target distance.
STATE_TOL = 1.0e-6


FRAME_TOL = 1.0e-7


FRAME_NORM_TOL = 1.0e-10


REFERENCE_TOL = 1.0e-6


METHOD_AGREEMENT_TOL = 2.0e-6


J_INNER = 5


EIG_TOL = 1.0e-10


EIG_MAXITER = 100000


SEED_INITIAL = 20260906


SEED_EIG = 2026090601


RECONSTRUCTION_TOL = 1.0e-14


CANONICAL_U0_RELATIVE_PATH = "inputs/common_U0.npz"


STATE_ORDERING = "component-major [u1,u2,u3,u4,u5], each interior C-ravel"


PHASES = ("initialization", "iteration", "verification")


COUNT_KEYS = (
    "gradient_evaluations",
    "analytic_hvp_evaluations",
    "eigensolver_hessian_actions",
    "sparse_hessian_assemblies",
    "m_apply_calls",
    "full_vector_m_solve_calls",
    "scalar_rhs_solves",
    "metric_factorizations",
)


class GateFailure(RuntimeError):
    pass


def reconstruction_consistency(
    rebuilt: np.ndarray, canonical: np.ndarray, h: float
) -> dict[str, Any]:
    """Check reconstruction against the saved initial state numerically."""
    if rebuilt.shape != canonical.shape or not (
        np.all(np.isfinite(rebuilt)) and np.all(np.isfinite(canonical))
    ):
        raise GateFailure("canonical U0 reconstruction has invalid shape/nonfinite values")
    difference = rebuilt - canonical
    max_abs_diff = float(np.max(np.abs(difference)))
    l2_h_diff = h * float(np.linalg.norm(difference))
    passed = max_abs_diff <= RECONSTRUCTION_TOL and l2_h_diff <= RECONSTRUCTION_TOL
    diagnostic = {
        "status": "PASS" if passed else "FAIL",
        "max_abs_diff": max_abs_diff,
        "l2_h_diff": l2_h_diff,
        "tolerance": RECONSTRUCTION_TOL,
        "bitwise_equality_required": False,
    }
    if not passed:
        raise GateFailure(f"canonical U0 reconstruction exceeds roundoff tolerance: {diagnostic}")
    return diagnostic


# Check the common initial array, basis and component ordering before either method runs.
def load_canonical_u0(
    model: Any, selected: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    initial = selected.get("initial_state", {})
    specification = initial.get("canonical_U0", {})
    if specification.get("path") != CANONICAL_U0_RELATIVE_PATH:
        raise GateFailure("missing or invalid canonical U0 input path")
    contents = embedded_bytes("canonical")
    with np.load(io.BytesIO(contents), allow_pickle=False) as data:
        metadata = {key: data[key].copy() for key in data.files}
    expected_scalars = {
        "N": model.N, "h": model.h, "full_Q_dofs": model.dofs,
        "lambda_sq": model.cfg.lambda_sq, "epsilon_bc": model.cfg.epsilon_bc,
        "A": model.cfg.A, "B": model.cfg.B, "C": model.cfg.C,
        "L": model.cfg.L, "s_plus": model.cfg.s_plus,
        "seed": SEED_INITIAL,
        "negative_mode_fraction_of_s_plus": 0.02,
        "smooth_mode_fraction_of_s_plus": 0.01,
    }
    for key, value in expected_scalars.items():
        if key not in metadata or not np.array_equal(metadata[key], np.asarray(value)):
            raise GateFailure(f"canonical U0 metadata mismatch for {key}")
    for key in ("seed", "negative_mode_fraction_of_s_plus", "smooth_mode_fraction_of_s_plus"):
        if initial.get(key) != expected_scalars[key]:
            raise GateFailure(f"selected initial-state metadata mismatch for {key}")
    expected_arrays = {
        "basis_matrices": BASIS,
        "basis_names": BASIS_NAMES,
        "source_q_to_u_factors": SOURCE_Q_TO_U_FACTORS,
        "state_ordering": np.asarray(STATE_ORDERING),
        "sine_mode_coefficients": np.asarray(initial.get("sine_mode_coefficients")),
    }
    for key, value in expected_arrays.items():
        if key not in metadata or not np.array_equal(metadata[key], value):
            raise GateFailure(f"canonical U0 basis/ordering/coefficients mismatch for {key}")
    U0 = metadata.get("U0", np.empty(0))
    if U0.shape != (model.dofs,) or U0.dtype.str != "<f8" or not np.all(np.isfinite(U0)):
        raise GateFailure("canonical U0 has invalid shape, dtype, or nonfinite values")
    U0.setflags(write=False)
    return U0, {
        "status": "PASS", "source_path": CANONICAL_U0_RELATIVE_PATH, "storage": "canonical payload in LDG_inputs.npz",
        "metadata_basis_ordering_match": True,
        "production_initial_state": "accepted canonical U0 (fresh copy for each method)",
    }


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def empty_counts() -> dict[str, dict[str, int]]:
    return {phase: {key: 0 for key in COUNT_KEYS} for phase in PHASES}


def total_counts(counts: dict[str, dict[str, int]]) -> dict[str, int]:
    return {key: sum(counts[phase][key] for phase in PHASES) for key in COUNT_KEYS}


class CountedModelOps:
    def __init__(self, model: Any):
        self.model = model
        self.counts = empty_counts()

    def gradient(self, state: np.ndarray, phase: str) -> np.ndarray:
        self.counts[phase]["gradient_evaluations"] += 1
        return self.model.gradient(state)

    def hvp(
        self,
        state: np.ndarray,
        vector: np.ndarray,
        phase: str,
        *,
        eigensolver: bool = False,
    ) -> np.ndarray:
        self.counts[phase]["analytic_hvp_evaluations"] += 1
        if eigensolver:
            self.counts[phase]["eigensolver_hessian_actions"] += 1
        return self.model.hvp(state, vector)

    def assemble_hessian(self, state: np.ndarray, phase: str) -> csc_matrix:
        self.counts[phase]["sparse_hessian_assemblies"] += 1
        return self.model.hessian(state)


class FixedMetric:
    """M=I_5 tensor (A_h+I), with one scalar SuperLU factorization."""

    def __init__(self, model: Any, counts: dict[str, dict[str, int]], phase: str):
        self.model = model
        self.counts = counts
        self.K = csc_matrix(model.A_h + eye(model.m, format="csc"))
        self.lu = splu(self.K)
        self.counts[phase]["metric_factorizations"] += 1
        self.factorization_count = 1

    def apply(self, vector: np.ndarray, phase: str) -> np.ndarray:
        self.counts[phase]["m_apply_calls"] += 1
        blocks = self.model.split(np.asarray(vector, dtype=float))
        return self.model.join(np.vstack([self.K @ blocks[a] for a in range(5)]))

    def solve(self, vector: np.ndarray, phase: str) -> np.ndarray:
        self.counts[phase]["full_vector_m_solve_calls"] += 1
        self.counts[phase]["scalar_rhs_solves"] += 5
        blocks = self.model.split(np.asarray(vector, dtype=float))
        solved = self.lu.solve(blocks.T).T
        return self.model.join(solved)

    def as_operators(self, phase: str) -> tuple[LinearOperator, LinearOperator]:
        shape = (self.model.dofs, self.model.dofs)
        apply = lambda x: self.apply(x, phase)
        solve = lambda x: self.solve(x, phase)
        M = LinearOperator(shape, matvec=apply, rmatvec=apply, dtype=float)
        Minv = LinearOperator(shape, matvec=solve, rmatvec=solve, dtype=float)
        return M, Minv


class CostTracker:
    BUCKETS = ('t_problem_setup', 't_M_setup', 't_M_refresh', 't_M_apply',
               't_M_solve', 't_eigensolver_net', 't_other', 't_endpoint_certification')

    def __init__(self, clock=None):
        self.clock = time.perf_counter if clock is None else clock
        self.exclusive = dict.fromkeys(self.BUCKETS, 0.0)
        self.inclusive = dict.fromkeys(('t_eigensolver_init', 't_eigensolver_update'), 0.0)
        self.eigen_net = dict.fromkeys(self.inclusive, 0.0)
        self.calls = dict.fromkeys((*self.BUCKETS, *self.inclusive), 0)
        self.current = 't_other'
        self.eigen_phase = None
        self.depth = 0

    def start(self):
        self.started = self.last = self.clock()

    def _tick(self, now):
        elapsed = now - self.last
        self.exclusive[self.current] += elapsed
        if self.current == 't_eigensolver_net' and self.eigen_phase is not None:
            self.eigen_net[self.eigen_phase] += elapsed
        self.last = now

    @contextmanager
    def region(self, bucket, diagnostic=None):
        entered = self.clock()
        self._tick(entered)
        parent, parent_phase = self.current, self.eigen_phase
        self.current = bucket
        self.calls[bucket] += 1
        if diagnostic is not None:
            self.eigen_phase = diagnostic
            self.calls[diagnostic] += 1
        self.depth += 1
        try:
            yield
        finally:
            ended = self.clock()
            self._tick(ended)
            if diagnostic is not None:
                self.inclusive[diagnostic] += ended - entered
            self.current, self.eigen_phase = parent, parent_phase
            self.depth -= 1

    def stop(self):
        if self.depth:
            raise RuntimeError('unclosed timing region')
        ended = self.clock()
        self._tick(ended)
        times = {'t_total': ended - self.started, **self.exclusive, **self.inclusive}
        times['t_M_setup_update'] = times['t_M_setup'] + times['t_M_refresh']
        times['t_M_apply_solve'] = times['t_M_apply'] + times['t_M_solve']
        times['t_eigensolver'] = times['t_eigensolver_init'] + times['t_eigensolver_update']
        times['t_eigensolver_init_net'] = self.eigen_net['t_eigensolver_init']
        times['t_eigensolver_update_net'] = self.eigen_net['t_eigensolver_update']
        times['t_eigensolver_nested_M'] = times['t_eigensolver'] - times['t_eigensolver_net']
        times['accounting_sum'] = sum(self.exclusive.values())
        times['accounting_residual'] = times['t_total'] - times['accounting_sum']
        times['accounting_relative_residual'] = abs(times['accounting_residual']) / max(times['t_total'], 1e-30)
        times['accounting_PASS'] = (times['accounting_relative_residual'] <= 1e-9
                                    and min(self.exclusive.values()) >= 0)
        return times


class TimedFixedMetric(FixedMetric):
    def __init__(self, model, counts, phase, cost):
        self.cost = cost
        super().__init__(model, counts, phase)

    def apply(self, vector, phase):
        with self.cost.region('t_M_apply'):
            return super().apply(vector, phase)

    def solve(self, vector, phase):
        with self.cost.region('t_M_solve'):
            return super().solve(vector, phase)


def metric_norm_sq(vector: np.ndarray, method: str, metric: FixedMetric | None, phase: str) -> float:
    if method == "p_hisd":
        assert metric is not None
        return float(vector @ metric.apply(vector, phase))
    return float(vector @ vector)


def normalize_frame(
    vector: np.ndarray, method: str, metric: FixedMetric | None, phase: str
) -> np.ndarray:
    norm_sq = metric_norm_sq(vector, method, metric, phase)
    if not math.isfinite(norm_sq) or norm_sq <= 0.0:
        raise FloatingPointError(f"invalid {method} frame norm squared: {norm_sq}")
    return vector / math.sqrt(norm_sq)


def canonicalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float).copy()
    pivot = int(np.argmax(np.abs(vector)))
    if vector[pivot] < 0.0:
        vector *= -1.0
    return vector


def gate_a_reference(model: Any) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    required = {
        "U_star",
        "most_negative_full_Q_eigenvector",
        "hessian_eigenvalues",
        "eigenpair_residuals",
        "basis_matrices",
        "basis_names",
        "source_q_to_u_factors",
        "state_ordering",
        "N",
        "h",
        "lambda_sq",
        "epsilon_bc",
        "A",
        "B",
        "C",
        "L",
        "s_plus",
    }
    with np.load(io.BytesIO(embedded_bytes("reference")), allow_pickle=False) as data:
        actual_keys = sorted(data.files)
        missing = sorted(required - set(data.files))
        if missing:
            raise GateFailure(f"Gate A: reference NPZ missing keys {missing}")
        U_star = np.asarray(data["U_star"], dtype=float).copy()
        negative = np.asarray(data["most_negative_full_Q_eigenvector"], dtype=float).copy()
        metadata = {key: np.asarray(data[key]).copy() for key in data.files}

    numeric_finite = {
        key: bool(np.all(np.isfinite(value)))
        for key, value in metadata.items()
        if np.issubdtype(value.dtype, np.number)
    }
    expected_metadata = {
        "N": model.N,
        "h": model.h,
        "lambda_sq": model.cfg.lambda_sq,
        "epsilon_bc": model.cfg.epsilon_bc,
        "A": model.cfg.A,
        "B": model.cfg.B,
        "C": model.cfg.C,
        "L": model.cfg.L,
        "s_plus": model.cfg.s_plus,
    }
    metadata_errors = {
        key: float(abs(float(metadata[key]) - float(value)))
        for key, value in expected_metadata.items()
    }
    ordering = str(metadata["state_ordering"])
    expected_ordering = "component-major [u1,u2,u3,u4,u5], each interior C-ravel"
    basis_error = float(np.max(np.abs(metadata["basis_matrices"] - BASIS)))
    basis_names_match = bool(
        np.array_equal(metadata["basis_names"].astype(str), BASIS_NAMES.astype(str))
    )
    source_coordinate_factor_error = float(
        np.max(
            np.abs(
                metadata["source_q_to_u_factors"]
                - SOURCE_Q_TO_U_FACTORS
            )
        )
    )

    if U_star.shape != (model.dofs,) or negative.shape != (model.dofs,):
        raise GateFailure(
            f"Gate A: invalid state/mode shapes {U_star.shape}, {negative.shape}; expected {(model.dofs,)}"
        )
    if not all(numeric_finite.values()):
        raise GateFailure("Gate A: nonfinite numeric data in reference NPZ")

    gradient = model.gradient(U_star)
    full_residual = model.h * float(np.linalg.norm(gradient))
    per_component = [model.h * float(np.linalg.norm(block)) for block in model.split(gradient)]
    H = model.hessian(U_star)
    negative = canonicalize(negative)
    negative_norm = float(np.linalg.norm(negative))
    negative_rayleigh = float(negative @ (H @ negative) / (negative @ negative))
    negative_residual = float(np.linalg.norm(H @ negative - negative_rayleigh * negative))

    action_counter = {"SA": 0, "LA": 0}
    shape = H.shape
    Hop_sa = LinearOperator(
        shape,
        matvec=lambda x: _counted_sparse_action(H, x, action_counter, "SA"),
        dtype=float,
    )
    rng = np.random.default_rng(SEED_EIG)
    values, vectors = eigsh(
        Hop_sa,
        k=6,
        which="SA",
        tol=EIG_TOL,
        maxiter=EIG_MAXITER,
        v0=rng.standard_normal(model.dofs),
        ncv=32,
    )
    order = np.argsort(values)
    values = values[order]
    vectors = vectors[:, order]
    eig_residuals = np.asarray(
        [np.linalg.norm(H @ vectors[:, i] - values[i] * vectors[:, i]) for i in range(6)]
    )
    Hop_la = LinearOperator(
        shape,
        matvec=lambda x: _counted_sparse_action(H, x, action_counter, "LA"),
        dtype=float,
    )
    lambda_max = float(
        eigsh(
            Hop_la,
            k=1,
            which="LA",
            tol=EIG_TOL,
            maxiter=EIG_MAXITER,
            v0=np.random.default_rng(SEED_EIG + 1).standard_normal(model.dofs),
            return_eigenvectors=False,
        )[0]
    )
    rmax = float(np.max(eig_residuals))
    delta = 1.0e-6
    resolved_index = int(np.count_nonzero(values < -delta)) if values[-1] > delta else None
    index_pass = bool(values[0] < -delta and values[1] > delta)
    content_pass = bool(
        max(metadata_errors.values()) <= 1.0e-14
        and basis_error <= 1.0e-14
        and basis_names_match
        and source_coordinate_factor_error <= 1.0e-14
        and ordering == expected_ordering
        and full_residual <= 1.0e-9
        and abs(negative_norm - 1.0) <= 1.0e-10
        and negative_residual <= 1.0e-7
        and index_pass
    )
    result = {
        "status": "PASS" if content_pass else "FAIL",
        "actual_npz_keys": actual_keys,
        "missing_npz_keys": missing,
        "state_shape": list(U_star.shape),
        "negative_mode_shape": list(negative.shape),
        "all_numeric_finite": all(numeric_finite.values()),
        "numeric_finite_by_key": numeric_finite,
        "metadata_errors": metadata_errors,
        "basis_max_error": basis_error,
        "basis_names_match": basis_names_match,
        "source_q_to_u_factors_max_error": source_coordinate_factor_error,
        "ordering": ordering,
        "ordering_matches": ordering == expected_ordering,
        "full_residual_l2": full_residual,
        "per_component_residual_l2": per_component,
        "saved_negative_mode_euclidean_norm": negative_norm,
        "saved_negative_mode_rayleigh": negative_rayleigh,
        "saved_negative_mode_actual_residual": negative_residual,
        "smallest_six_eigenvalues": values.tolist(),
        "eigenpair_residuals": eig_residuals.tolist(),
        "delta_ind": delta,
        "resolved_index": resolved_index,
        "lambda_max": lambda_max,
        "eta_linear_euclidean": 2.0 / max(abs(values[0]), abs(lambda_max)),
        "tau_linear_euclidean": 2.0 / (lambda_max - values[0]),
        "eigensolver_sparse_actions": action_counter,
    }
    if not content_pass:
        raise GateFailure("Gate A: mathematical/reference integrity checks failed")
    return result, U_star, negative


def _counted_sparse_action(
    matrix: csc_matrix, vector: np.ndarray, counter: dict[str, int], key: str
) -> np.ndarray:
    counter[key] += 1
    return matrix @ vector


def nodewise_amplitude(model: Any, vector: np.ndarray) -> float:
    blocks = model.split(np.asarray(vector, dtype=float))
    return float(np.max(np.sqrt(np.sum(blocks**2, axis=0))))


def component_max_abs(model: Any, vector: np.ndarray) -> list[float]:
    return [float(np.max(np.abs(block))) for block in model.split(vector)]


def gate_b_initial_state(
    model: Any, U_star: np.ndarray, negative: np.ndarray,
    canonical_U0: np.ndarray | None = None,
    saved_coefficients: np.ndarray | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    negative = canonicalize(negative)
    amp_negative = nodewise_amplitude(model, negative)
    if amp_negative <= 0.0:
        raise GateFailure("Gate B: saved negative mode has zero nodewise amplitude")
    D_minus = negative / amp_negative

    coords = model.coordinates[1:-1]
    X, Y = np.meshgrid(coords, coords, indexing="xy")
    rng = np.random.default_rng(SEED_INITIAL)
    psi_blocks = np.zeros((5, model.m), dtype=float)
    coefficients = (
        np.zeros((5, 3, 3), dtype=float) if saved_coefficients is None
        else np.asarray(saved_coefficients, dtype=float).copy()
    )
    if coefficients.shape != (5, 3, 3) or not np.all(np.isfinite(coefficients)) or np.any(coefficients[1]):
        raise GateFailure("Gate B: invalid saved smooth-mode coefficients")
    for component in (0, 2, 3, 4):
        for p in (1, 2, 3):
            sx = np.sin(p * math.pi * (X + 1.0) / 2.0)
            for q in (1, 2, 3):
                coefficient = (
                    float(rng.standard_normal()) if saved_coefficients is None
                    else float(coefficients[component, p - 1, q - 1])
                )
                coefficients[component, p - 1, q - 1] = coefficient
                sy = np.sin(q * math.pi * (Y + 1.0) / 2.0)
                psi_blocks[component] += (coefficient * sx * sy).ravel(order="C")
    Psi_raw = model.join(psi_blocks)
    amp_psi_raw = nodewise_amplitude(model, Psi_raw)
    if amp_psi_raw <= 0.0:
        raise GateFailure("Gate B: generated smooth perturbation has zero amplitude")
    Psi = Psi_raw / amp_psi_raw
    scale_minus = 0.02 * model.cfg.s_plus
    scale_psi = 0.01 * model.cfg.s_plus
    U0_rebuilt = U_star + scale_minus * D_minus + scale_psi * Psi
    reconstruction = None
    if canonical_U0 is not None:
        reconstruction = reconstruction_consistency(U0_rebuilt, canonical_U0, model.h)
    U0 = U0_rebuilt if canonical_U0 is None else canonical_U0.copy()

    full_star = model.full_u_fields(U_star)
    full_u0 = model.full_u_fields(U0)
    boundary_error = float(
        max(
            np.max(np.abs(full_u0[:, 0, :] - full_star[:, 0, :])),
            np.max(np.abs(full_u0[:, -1, :] - full_star[:, -1, :])),
            np.max(np.abs(full_u0[:, :, 0] - full_star[:, :, 0])),
            np.max(np.abs(full_u0[:, :, -1] - full_star[:, :, -1])),
        )
    )
    gradient = model.gradient(U0)
    residual = model.h * float(np.linalg.norm(gradient))
    per_component_residual = [
        model.h * float(np.linalg.norm(block)) for block in model.split(gradient)
    ]
    U0_component_max = component_max_abs(model, U0)
    perturb_component_max = component_max_abs(model, U0 - U_star)
    psi_component_max = component_max_abs(model, Psi)
    dminus_component_max = component_max_abs(model, D_minus)
    u4_nonzero = perturb_component_max[3] > 1.0e-6
    u5_nonzero = perturb_component_max[4] > 1.0e-6
    passed = bool(
        boundary_error <= 1.0e-14
        and residual >= 100.0 * STATE_TOL
        and u4_nonzero
        and u5_nonzero
        and abs(nodewise_amplitude(model, D_minus) - 1.0) <= 1.0e-13
        and abs(nodewise_amplitude(model, Psi) - 1.0) <= 1.0e-13
    )
    result = {
        "status": "PASS" if passed else "FAIL",
        "seed": SEED_INITIAL,
        "negative_mode_scale": scale_minus,
        "smooth_perturbation_scale": scale_psi,
        "negative_mode_raw_nodewise_amplitude": amp_negative,
        "D_minus_nodewise_amplitude": nodewise_amplitude(model, D_minus),
        "Psi_raw_nodewise_amplitude": amp_psi_raw,
        "Psi_nodewise_amplitude": nodewise_amplitude(model, Psi),
        "sine_mode_coefficients": coefficients.tolist(),
        "smooth_components": ["u1", "u3", "u4", "u5"],
        "smooth_u2_exactly_zero": bool(np.all(psi_blocks[1] == 0.0)),
        "boundary_max_error": boundary_error,
        "initial_full_residual_l2": residual,
        "initial_per_component_residual_l2": per_component_residual,
        "U0_component_max_abs": U0_component_max,
        "perturbation_component_max_abs": perturb_component_max,
        "Psi_component_max_abs": psi_component_max,
        "D_minus_component_max_abs": dminus_component_max,
        "u4_initially_nonzero": u4_nonzero,
        "u5_initially_nonzero": u5_nonzero,
    }
    if reconstruction is not None:
        result["reconstruction_consistency"] = reconstruction
    if not passed:
        raise GateFailure("Gate B: common initial-state checks failed")
    return result, U0, D_minus, Psi


def gate_c_metric(model: Any) -> dict[str, Any]:
    counts = empty_counts()
    started = time.perf_counter()
    metric = FixedMetric(model, counts, "verification")
    setup_time = time.perf_counter() - started
    eig_min = float(
        eigsh(metric.K, k=1, which="SA", tol=EIG_TOL, return_eigenvectors=False)[0]
    )
    eig_max = float(
        eigsh(metric.K, k=1, which="LA", tol=EIG_TOL, return_eigenvectors=False)[0]
    )
    rng = np.random.default_rng(SEED_INITIAL + 100)
    rhs = rng.standard_normal(model.dofs)
    solution = metric.solve(rhs, "verification")
    solve_residual = float(
        np.linalg.norm(metric.apply(solution, "verification") - rhs) / np.linalg.norm(rhs)
    )
    vector = rng.standard_normal(model.dofs)
    normalized = normalize_frame(vector, "p_hisd", metric, "verification")
    normalization_error = abs(
        float(normalized @ metric.apply(normalized, "verification")) - 1.0
    )
    passed = bool(
        eig_min > 0.0
        and solve_residual <= 1.0e-10
        and normalization_error <= 1.0e-12
        and metric.factorization_count == 1
        and counts["verification"]["scalar_rhs_solves"]
        == 5 * counts["verification"]["full_vector_m_solve_calls"]
    )
    result = {
        "status": "PASS" if passed else "FAIL",
        "definition": "M=I_5 tensor K_h, K_h=A_h+I_m",
        "K_shape": list(metric.K.shape),
        "K_nnz": int(metric.K.nnz),
        "K_smallest_eigenvalue": eig_min,
        "K_largest_eigenvalue": eig_max,
        "random_rhs_relative_solve_residual": solve_residual,
        "normalization_error": normalization_error,
        "factorization_backend": "SciPy SuperLU sparse LU (not called Cholesky)",
        "factorization_count": metric.factorization_count,
        "one_scalar_factor_reused_for_five_components": True,
        "setup_seconds": setup_time,
        "counters": counts,
    }
    if not passed:
        raise GateFailure("Gate C: fixed metric or solve checks failed")
    return result


def tiny_unit_tests() -> dict[str, Any]:
    rng = np.random.default_rng(SEED_INITIAL + 200)
    n = 5
    A = rng.standard_normal((n, n))
    H = 0.5 * (A + A.T)
    g = rng.standard_normal(n)
    x = rng.standard_normal(n)
    v0 = rng.standard_normal(n)
    eta = 0.037
    tau = 0.021

    def generic_step(
        state: np.ndarray,
        frame: np.ndarray,
        gradient: np.ndarray,
        hessian: np.ndarray,
        metric_matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        v = frame / math.sqrt(float(frame @ metric_matrix @ frame))
        for _ in range(J_INNER):
            w = hessian @ v
            z = np.linalg.solve(metric_matrix, w)
            rho = float(v @ w)
            trial = v - tau * (z - rho * v)
            v = trial / math.sqrt(float(trial @ metric_matrix @ trial))
        direction = -np.linalg.solve(metric_matrix, gradient) + 2.0 * v * float(v @ gradient)
        return state + eta * direction, v

    # A: generic p-HiSD with M=I versus a separately written standard update.
    xp, vp = generic_step(x, v0, g, H, np.eye(n))
    vh = v0 / np.linalg.norm(v0)
    for _ in range(J_INNER):
        w = H @ vh
        rho = float(vh @ w)
        trial = vh - tau * (w - rho * vh)
        vh = trial / np.linalg.norm(trial)
    xh = x + eta * (-g + 2.0 * vh * float(vh @ g))
    identity_state_error = float(np.max(np.abs(xp - xh)))
    identity_frame_error = float(np.max(np.abs(vp - vh)))

    # B: non-diagonal SPD fixed-coordinate equivalence y=Sx, w=Sv.
    R = rng.standard_normal((n, n))
    M = R.T @ R + 0.75 * np.eye(n)
    eigenvalues, eigenvectors = eigh(M)
    S = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
    Sinv = (eigenvectors * (1.0 / np.sqrt(eigenvalues))) @ eigenvectors.T
    v_metric = v0 / math.sqrt(float(v0 @ M @ v0))
    x_metric, v_metric_new = generic_step(x, v_metric, g, H, M)

    y = S @ x
    w = S @ v_metric
    gy = Sinv @ g
    Hy = Sinv @ H @ Sinv
    w = w / np.linalg.norm(w)
    for _ in range(J_INNER):
        Hw = Hy @ w
        rho = float(w @ Hw)
        trial = w - tau * (Hw - rho * w)
        w = trial / np.linalg.norm(trial)
    y_new = y + eta * (-gy + 2.0 * w * float(w @ gy))
    coordinate_state_error = float(np.max(np.abs(S @ x_metric - y_new)))
    coordinate_frame_error = float(np.max(np.abs(S @ v_metric_new - w)))
    passed = bool(
        max(identity_state_error, identity_frame_error) <= 1.0e-13
        and max(coordinate_state_error, coordinate_frame_error) <= 2.0e-12
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "identity_metric_state_max_error": identity_state_error,
        "identity_metric_frame_max_error": identity_frame_error,
        "non_diagonal_spd_coordinate_state_max_error": coordinate_state_error,
        "non_diagonal_spd_coordinate_frame_max_error": coordinate_frame_error,
        "tiny_sqrt_used_only_here": True,
    }


# Use the lowest ordinary or generalized mode, with a fixed eigensolver start.
def initial_frame(
    model: Any,
    state: np.ndarray,
    method: str,
    ops: CountedModelOps,
    metric: FixedMetric | None,
    *,
    compute_largest: bool,
) -> dict[str, Any]:
    shape = (model.dofs, model.dofs)
    H_operator = LinearOperator(
        shape,
        matvec=lambda x: ops.hvp(
            state, x, "initialization", eigensolver=True
        ),
        rmatvec=lambda x: ops.hvp(
            state, x, "initialization", eigensolver=True
        ),
        dtype=float,
    )
    rng = np.random.default_rng(SEED_EIG)
    kwargs: dict[str, Any] = {}
    if method == "p_hisd":
        assert metric is not None
        M, Minv = metric.as_operators("initialization")
        kwargs.update(M=M, Minv=Minv)
    values, vectors = eigsh(
        H_operator,
        k=2,
        which="SA",
        tol=EIG_TOL,
        maxiter=EIG_MAXITER,
        v0=rng.standard_normal(model.dofs),
        ncv=24,
        **kwargs,
    )
    order = np.argsort(values)
    values = values[order]
    vectors = vectors[:, order]
    frame = canonicalize(vectors[:, 0])
    frame = normalize_frame(frame, method, metric, "initialization")
    Hv = ops.hvp(state, frame, "initialization")
    if method == "p_hisd":
        assert metric is not None
        Mv = metric.apply(frame, "initialization")
    else:
        Mv = frame
    rho = float(frame @ Hv / (frame @ Mv))
    raw_residual = float(np.linalg.norm(Hv - rho * Mv))
    normalization_error = abs(float(frame @ Mv) - 1.0)
    largest = None
    largest_residual = None
    if compute_largest:
        value_la, vector_la = eigsh(
            H_operator,
            k=1,
            which="LA",
            tol=EIG_TOL,
            maxiter=EIG_MAXITER,
            v0=np.random.default_rng(SEED_EIG + 1).standard_normal(model.dofs),
            **kwargs,
        )
        largest = float(value_la[0])
        v_la = vector_la[:, 0]
        Hv_la = ops.hvp(state, v_la, "initialization")
        Mv_la = metric.apply(v_la, "initialization") if metric is not None else v_la
        largest_residual = float(np.linalg.norm(Hv_la - largest * Mv_la))
    return {
        "eigenvalues": values.tolist(),
        "frame": frame,
        "rho": rho,
        "raw_residual": raw_residual,
        "normalization_error": normalization_error,
        "largest_eigenvalue": largest,
        "largest_raw_residual": largest_residual,
    }


def gate_d_algorithm(model: Any, U0: np.ndarray) -> dict[str, Any]:
    units = tiny_unit_tests()
    if units["status"] != "PASS":
        raise GateFailure("Gate D: tiny algorithm equivalence tests failed")

    ordinary_ops = CountedModelOps(model)
    ordinary_started = time.perf_counter()
    ordinary = initial_frame(
        model, U0, "hisd", ordinary_ops, None, compute_largest=True
    )
    ordinary_seconds = time.perf_counter() - ordinary_started

    preconditioned_ops = CountedModelOps(model)
    metric_started = time.perf_counter()
    metric = FixedMetric(model, preconditioned_ops.counts, "initialization")
    metric_setup = time.perf_counter() - metric_started
    generalized_started = time.perf_counter()
    generalized = initial_frame(
        model, U0, "p_hisd", preconditioned_ops, metric, compute_largest=True
    )
    generalized_seconds = time.perf_counter() - generalized_started

    ordinary_min = float(ordinary["eigenvalues"][0])
    ordinary_max = float(ordinary["largest_eigenvalue"])
    generalized_min = float(generalized["eigenvalues"][0])
    generalized_max = float(generalized["largest_eigenvalue"])
    initial_frames_pass = bool(
        ordinary["eigenvalues"][0] < 0.0 < ordinary["eigenvalues"][1]
        and generalized["eigenvalues"][0] < 0.0 < generalized["eigenvalues"][1]
        and ordinary["raw_residual"] <= 1.0e-7
        and generalized["raw_residual"] <= 1.0e-7
        and ordinary["normalization_error"] <= 1.0e-10
        and generalized["normalization_error"] <= 1.0e-10
        and metric.factorization_count == 1
    )
    if not initial_frames_pass:
        raise GateFailure("Gate D: ordinary/generalized initial-frame diagnostics failed")
    return {
        "status": "PASS",
        "algorithm_interpretation": (
            "fixed-M k=1 Algorithm 5.1; standard HiSD is its M=I finite-J specialization"
        ),
        "J": J_INNER,
        "unit_tests": units,
        "ordinary_initial_frame": {
            key: value for key, value in ordinary.items() if key != "frame"
        },
        "generalized_initial_frame": {
            key: value for key, value in generalized.items() if key != "frame"
        },
        "ordinary_spectral_estimates_U0": {
            "lambda_min": ordinary_min,
            "lambda_max": ordinary_max,
            "eta_linear_boundary": 2.0 / max(abs(ordinary_min), abs(ordinary_max)),
            "tau_linear_boundary": 2.0 / (ordinary_max - ordinary_min),
        },
        "generalized_spectral_estimates_U0": {
            "lambda_min": generalized_min,
            "lambda_max": generalized_max,
            "eta_linear_boundary": 2.0 / max(abs(generalized_min), abs(generalized_max)),
            "tau_linear_boundary": 2.0 / (generalized_max - generalized_min),
        },
        "ordinary_initial_frame_seconds": ordinary_seconds,
        "fixed_metric_setup_seconds": metric_setup,
        "generalized_initial_frame_seconds": generalized_seconds,
        "ordinary_counters": ordinary_ops.counts,
        "generalized_counters": preconditioned_ops.counts,
    }


def frame_and_state_metrics(
    model: Any,
    ops: CountedModelOps,
    method: str,
    state: np.ndarray,
    frame: np.ndarray,
    gradient: np.ndarray,
    U_star: np.ndarray,
    metric: FixedMetric | None,
    phase: str,
) -> dict[str, Any]:
    Hv = ops.hvp(state, frame, phase)
    if method == "p_hisd":
        assert metric is not None
        Mv = metric.apply(frame, phase)
    else:
        Mv = frame
    denominator = float(frame @ Mv)
    rho = float(frame @ Hv / denominator)
    error = Hv - rho * Mv
    if method == "p_hisd":
        assert metric is not None
        solved_error = metric.solve(error, phase)
        natural_sq = float(error @ solved_error)
        frame_residual = math.sqrt(max(0.0, natural_sq))
    else:
        frame_residual = float(np.linalg.norm(error))
    state_residual = model.h * float(np.linalg.norm(gradient))
    component_residuals = [
        model.h * float(np.linalg.norm(block)) for block in model.split(gradient)
    ]
    blocks = model.split(state)
    return {
        "state_residual": state_residual,
        "component_state_residuals": component_residuals,
        "frame_residual": frame_residual,
        "frame_raw_euclidean_residual": float(np.linalg.norm(error)),
        "frame_normalization_error": abs(denominator - 1.0),
        "rho": rho,
        "reference_distance": model.h * float(np.linalg.norm(state - U_star)),
        "u4_max_abs": float(np.max(np.abs(blocks[3]))),
        "u5_max_abs": float(np.max(np.abs(blocks[4]))),
    }


# Require state and frame convergence at the same state, a negative mode and the target saddle.
def stopping_pass(metrics: dict[str, Any]) -> bool:
    return bool(
        metrics["state_residual"] <= STATE_TOL
        and metrics["frame_residual"] <= FRAME_TOL
        and metrics["frame_normalization_error"] <= FRAME_NORM_TOL
        and metrics["rho"] < 0.0
        and metrics["reference_distance"] <= REFERENCE_TOL
    )


@dataclass(frozen=True)
class MethodParameters:
    eta: float
    tau: float
    J: int = J_INNER


def run_method(
    model: Any,
    method: str,
    params: MethodParameters,
    U0: np.ndarray,
    U_star: np.ndarray,
    *,
    max_updates: int,
    max_seconds: float,
    run_kind: str,
    history_stride: int,
    cost: CostTracker | None = None,
) -> dict[str, Any]:
    if method not in {"hisd", "p_hisd"}:
        raise ValueError(method)
    if params.J != J_INNER:
        raise ValueError("This experiment fixes J=5")
    ops = CountedModelOps(model)
    metric: FixedMetric | None = None
    setup_seconds = 0.0
    if method == "p_hisd":
        started = time.perf_counter()
        with cost.region('t_M_setup') if cost is not None else nullcontext():
            metric = (TimedFixedMetric(model, ops.counts, "initialization", cost)
                      if cost is not None else FixedMetric(model, ops.counts, "initialization"))
        setup_seconds = time.perf_counter() - started

    init_started = time.perf_counter()
    with cost.region('t_eigensolver_net', 't_eigensolver_init') if cost is not None else nullcontext():
        init = initial_frame(model, U0.copy(), method, ops, metric, compute_largest=False)
    state = U0.copy()
    initial_frame_vector = np.asarray(init["frame"], dtype=float).copy()
    frame = initial_frame_vector.copy()
    gradient = ops.gradient(state, "initialization")
    metrics = frame_and_state_metrics(
        model, ops, method, state, frame, gradient, U_star, metric, "initialization"
    )
    initialization_seconds = time.perf_counter() - init_started
    history: list[dict[str, Any]] = []

    def add_history(iteration: int, iteration_elapsed: float, status: str = "RUNNING") -> None:
        history.append(
            {
                "method": method,
                "iteration": iteration,
                "method_elapsed_seconds": setup_seconds
                + initialization_seconds
                + iteration_elapsed,
                "iteration_elapsed_seconds": iteration_elapsed,
                "state_residual": metrics["state_residual"],
                "frame_residual": metrics["frame_residual"],
                "frame_raw_euclidean_residual": metrics[
                    "frame_raw_euclidean_residual"
                ],
                "frame_normalization_error": metrics["frame_normalization_error"],
                "rho": metrics["rho"],
                "reference_distance": metrics["reference_distance"],
                "u4_max_abs": metrics["u4_max_abs"],
                "u5_max_abs": metrics["u5_max_abs"],
                "eta": params.eta,
                "tau": params.tau,
                "J": params.J,
                "status": status,
            }
        )

    add_history(0, 0.0)
    initial_metrics = dict(metrics)
    iteration_started = time.perf_counter()
    status = "RUNNING"
    reason = ""
    updates = 0
    if stopping_pass(metrics):
        status = (
            "CONVERGED"
            if metrics["reference_distance"] <= REFERENCE_TOL
            else "WRONG_TARGET"
        )
        reason = "initial state already met stopping checks"

    while status == "RUNNING" and updates < max_updates:
        elapsed = time.perf_counter() - iteration_started
        if elapsed >= max_seconds:
            status = "BUDGET_EXHAUSTED"
            reason = f"run time limit {max_seconds:g} s reached"
            break
        try:
            with cost.region('t_eigensolver_net', 't_eigensolver_update') if cost is not None else nullcontext():
                frame = normalize_frame(frame, method, metric, "iteration")
                # Evolve and normalize the frame J times at the current state.
                for _ in range(params.J):
                    w = ops.hvp(state, frame, "iteration")
                    if method == "p_hisd":
                        assert metric is not None
                        z = metric.solve(w, "iteration")
                    else:
                        z = w
                    rho = float(frame @ w)
                    trial = frame - params.tau * (z - rho * frame)
                    frame = normalize_frame(trial, method, metric, "iteration")

            if method == "p_hisd":
                assert metric is not None
                preconditioned_gradient = metric.solve(gradient, "iteration")
            else:
                preconditioned_gradient = gradient
            # Reflect the gradient using the updated frame, then advance all five components.
            direction = -preconditioned_gradient + 2.0 * frame * float(frame @ gradient)
            state = state + params.eta * direction
            updates += 1
            if not np.all(np.isfinite(state)) or not np.all(np.isfinite(frame)):
                raise FloatingPointError("state or frame became nonfinite")
            gradient = ops.gradient(state, "iteration")
            metrics = frame_and_state_metrics(
                model,
                ops,
                method,
                state,
                frame,
                gradient,
                U_star,
                metric,
                "iteration",
            )
        except Exception as exc:
            status = "NUMERICAL_FAILURE"
            reason = f"iteration exception: {type(exc).__name__}: {exc}"
            break

        if not all(
            math.isfinite(float(metrics[key]))
            for key in (
                "state_residual",
                "frame_residual",
                "frame_normalization_error",
                "rho",
                "reference_distance",
            )
        ):
            status = "NUMERICAL_FAILURE"
            reason = "nonfinite online diagnostic"
        elif (
            metrics["state_residual"]
            > max(1.0e8, 1.0e8 * initial_metrics["state_residual"])
            or np.linalg.norm(state) > 1.0e8
        ):
            status = "NUMERICAL_FAILURE"
            reason = "clear numerical divergence"
        elif stopping_pass(metrics):
            if metrics["reference_distance"] <= REFERENCE_TOL:
                status = "CONVERGED"
                reason = "same-state residual, frame, normalization, Rayleigh, and target checks passed"
            else:
                status = "WRONG_TARGET"
                reason = "stationary/frame checks passed but reference distance failed"

        elapsed = time.perf_counter() - iteration_started
        if updates % history_stride == 0 or status != "RUNNING":
            add_history(updates, elapsed, status)

    if status == "RUNNING":
        status = "PILOT_INCONCLUSIVE" if run_kind == "pilot" else "BUDGET_EXHAUSTED"
        reason = f"maximum {max_updates} outer updates reached"
        elapsed = time.perf_counter() - iteration_started
        if history[-1]["iteration"] != updates:
            add_history(updates, elapsed, status)
        else:
            history[-1]["status"] = status

    iteration_seconds = time.perf_counter() - iteration_started
    if history[-1]["iteration"] != updates:
        add_history(updates, iteration_seconds, status)
    else:
        history[-1]["status"] = status
    method_total = setup_seconds + initialization_seconds + iteration_seconds
    counts = ops.counts
    metric_factorizations = total_counts(counts)["metric_factorizations"]
    if method == "p_hisd" and metric_factorizations != 1:
        status = "NUMERICAL_FAILURE"
        reason = f"expected one metric factorization, observed {metric_factorizations}"
    if method == "hisd" and metric_factorizations != 0:
        status = "NUMERICAL_FAILURE"
        reason = "HiSD unexpectedly factorized a metric"
    return {
        "method": method,
        "run_kind": run_kind,
        "parameters": {"eta": params.eta, "tau": params.tau, "J": params.J},
        "status": status,
        "reason": reason,
        "outer_state_updates": updates,
        "initial_metrics": initial_metrics,
        "final_metrics": metrics,
        "initial_frame_eigenvalues": init["eigenvalues"],
        "initial_frame_raw_residual": init["raw_residual"],
        "initial_frame_normalization_error": init["normalization_error"],
        "timing": {
            "metric_setup_seconds": setup_seconds,
            "initial_frame_seconds": initialization_seconds,
            "iteration_seconds": iteration_seconds,
            "method_total_seconds": method_total,
        },
        "counters_by_phase": counts,
        "counters_total": total_counts(counts),
        "history": history,
        "initial_frame": initial_frame_vector,
        "final_frame": frame.copy(),
        "final_state": state.copy(),
    }


def load_selected_parameters(
    model: Any
) -> tuple[dict[str, Any], MethodParameters, MethodParameters]:
    data = json.loads(embedded_bytes("selected").decode("utf-8"))
    model_data = data.get("model", {})
    for key, value in {
        "N": model.N,
        "h": model.h,
        "dofs": model.dofs,
        "lambda_sq": model.cfg.lambda_sq,
        "epsilon_bc": model.cfg.epsilon_bc,
        "A": model.cfg.A,
        "B": model.cfg.B,
        "C": model.cfg.C,
        "L": model.cfg.L,
    }.items():
        if model_data.get(key) != value:
            raise GateFailure(f"selected parameter model mismatch for {key}")
    p = data["p_hisd"]
    h = data["hisd"]
    p_params = MethodParameters(float(p["eta"]), float(p["tau"]), int(p["J"]))
    h_params = MethodParameters(float(h["eta"]), float(h["tau"]), int(h["J"]))
    if p_params.J != J_INNER or h_params.J != J_INNER:
        raise GateFailure("selected parameters do not preserve J=5")
    return data, p_params, h_params


# Independently certify index 1 in the full five-component space using six eigenpairs.
def verify_final_state(
    model: Any, result: dict[str, Any], U_star: np.ndarray
) -> dict[str, Any]:
    state = result["final_state"]
    started = time.perf_counter()
    H = model.hessian(state)
    assembly_seconds = time.perf_counter() - started
    action_count = 0

    def action(vector: np.ndarray) -> np.ndarray:
        nonlocal action_count
        action_count += 1
        return H @ vector

    eig_started = time.perf_counter()
    H_operator = LinearOperator(H.shape, matvec=action, dtype=float)
    values, vectors = eigsh(
        H_operator,
        k=6,
        which="SA",
        tol=EIG_TOL,
        maxiter=EIG_MAXITER,
        v0=np.random.default_rng(SEED_EIG + 500).standard_normal(model.dofs),
        ncv=32,
    )
    order = np.argsort(values)
    values = values[order]
    vectors = vectors[:, order]
    residuals = np.asarray(
        [np.linalg.norm(H @ vectors[:, i] - values[i] * vectors[:, i]) for i in range(6)]
    )
    eig_seconds = time.perf_counter() - eig_started
    rmax = float(np.max(residuals))
    delta = 1.0e-6
    index_pass = bool(values[0] < -delta and values[1] > delta)
    resolved_index = int(np.count_nonzero(values < -delta)) if values[-1] > delta else None
    metrics = result["final_metrics"]
    reference_distance = model.h * float(np.linalg.norm(state - U_star))
    frame_acceptance = bool(
        metrics["state_residual"] <= STATE_TOL
        and metrics["frame_residual"] <= FRAME_TOL
        and metrics["frame_normalization_error"] <= FRAME_NORM_TOL
        and metrics["rho"] < 0.0
    )
    passed = bool(
        result["status"] == "CONVERGED"
        and frame_acceptance
        and reference_distance <= REFERENCE_TOL
        and index_pass
        and rmax <= 1.0e-7
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "state_and_frame_acceptance": frame_acceptance,
        "reference_distance": reference_distance,
        "boundary_preserved_by_interior_only_state": True,
        "all_five_components_free_no_projection_or_clipping": True,
        "hessian_shape": list(H.shape),
        "hessian_nnz": int(H.nnz),
        "smallest_six_eigenvalues": values.tolist(),
        "eigenpair_residuals": residuals.tolist(),
        "max_eigenpair_residual": rmax,
        "delta_ind": delta,
        "resolved_index": resolved_index,
        "near_zero_mode_present": bool(np.any(np.abs(values) <= delta)),
        "sparse_hessian_assemblies": 1,
        "eigensolver_hessian_actions": action_count,
        "assembly_seconds": assembly_seconds,
        "eigensolve_seconds": eig_seconds,
        "final_verification_seconds": assembly_seconds + eig_seconds,
    }


def environment_info() -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "thread_environment": {key: os.environ.get(key) for key in THREAD_ENV},
    }


def public_method_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: jsonable(value)
        for key, value in result.items()
        if key not in {"history", "initial_frame", "final_frame", "final_state"}
    }


def load_input_bundle() -> dict[str, bytes]:
    try:
        contents = INPUT_PATH.read_bytes()
        with np.load(io.BytesIO(contents), allow_pickle=False) as data:
            if set(data.files) != {'reference', 'canonical', 'selected'}:
                raise ValueError('expected exactly reference, canonical and selected payloads')
            payloads = {}
            for name in ('reference', 'canonical', 'selected'):
                array = data[name]
                if array.dtype != np.dtype('uint8') or array.ndim != 1:
                    raise ValueError(f'{name} must be a one-dimensional uint8 payload')
                payload = array.tobytes()
                payloads[name] = payload
        return payloads
    except Exception as exc:
        raise GateFailure(
            f'Cannot read or verify the companion input file {INPUT_PATH}: {exc}. '
            'Keep LDG_inputs.npz beside run.py; do not regenerate the initial state.'
        ) from exc

def embedded_bytes(name: str) -> bytes:
    return load_input_bundle()[name]


def _validate_settings() -> None:
    for name, expected in FROZEN_CONSTANTS.items():
        if globals()[name] != expected:
            raise GateFailure(f'frozen numerical setting changed: {name}')


def validate_inputs_and_gates(model: Any) -> tuple[dict[str, Any], dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _validate_settings()
    selected, p_params, h_params = load_selected_parameters(model)
    if p_params != MethodParameters(0.25, 0.1, 5) or h_params != MethodParameters(5e-4, 2.5e-4, 5):
        raise GateFailure('accepted eta/tau/J changed')
    canonical, canonical_info = load_canonical_u0(model, selected)
    gates = {}
    gates['A'], reference, negative = gate_a_reference(model)
    print(f"[1/6] Reference PASS: residual={gates['A']['full_residual_l2']:.3e}, full-Q index={gates['A']['resolved_index']}", flush=True)
    gates['B'], initial, D_minus, Psi = gate_b_initial_state(
        model, reference, negative, canonical,
        np.asarray(selected['initial_state']['sine_mode_coefficients']),
    )
    gates['B']['canonical_input'] = canonical_info
    check = gates['B']['reconstruction_consistency']
    print(f"[2/6] Canonical U0 PASS: reconstruction max={check['max_abs_diff']:.3e}, h-L2={check['l2_h_diff']:.3e}", flush=True)
    gates['C'] = gate_c_metric(model)
    print('[3/6] Fixed metric and sparse solves PASS', flush=True)
    gates['D'] = gate_d_algorithm(model, initial)
    print('[4/6] Algorithm and ordinary/generalized initial spectra PASS', flush=True)
    return gates, selected, reference, initial, D_minus, Psi


TIMING_PROTOCOL = {
    'initial_repeats': 3, 'fast_threshold_seconds': 0.1, 'fast_repeats_total': 9,
    'order_rng_seed': 6090603, 'seeds': [20260906],
    'scope': 'Only the two frozen LDG methods and one accepted canonical U0; no new seeds/initial conditions.',
}


def timing_environment(*, configure=False) -> dict[str, Any]:
    native = None
    if platform.system() == 'Darwin':
        library = ctypes.CDLL('/System/Library/Frameworks/Accelerate.framework/Accelerate')
        library.BLASSetThreading.argtypes = [ctypes.c_uint]
        library.BLASSetThreading.restype = ctypes.c_int
        library.BLASGetThreading.argtypes = []
        library.BLASGetThreading.restype = ctypes.c_uint
        if configure and library.BLASSetThreading(1) != 0:
            raise RuntimeError('Accelerate single-thread policy could not be established')
        if library.BLASGetThreading() != 1:
            raise RuntimeError('Accelerate single-thread policy changed or was not configured')
        native = {'backend': 'Accelerate', 'current_enum': int(library.BLASGetThreading()),
                  'single_thread_enum': 1, 'scope': 'calling solver thread'}
    if any(os.environ.get(key) != '1' for key in THREAD_ENV):
        raise RuntimeError('BLAS/OpenMP thread environment changed')
    config_text = io.StringIO()
    with warnings.catch_warnings(record=True), redirect_stdout(config_text):
        np.show_config()
        scipy.show_config()
    pools = None
    if importlib.util.find_spec('threadpoolctl') is not None:
        import threadpoolctl
        pools = threadpoolctl.threadpool_info()
        if any(pool.get('num_threads') != 1 for pool in pools):
            raise RuntimeError('non-single-thread BLAS/OpenMP pool detected')
    backend_names = {
        name: {kind: getattr(module.__config__, 'CONFIG', {}).get('Build Dependencies', {})
                     .get(kind, {}).get('name', 'unknown').lower()
               for kind in ('blas', 'lapack')}
        for name, module in (('numpy', np), ('scipy', scipy))
    }
    native_verified = native is not None and all(
        backend_names[name][kind] == 'accelerate'
        for name in ('numpy', 'scipy') for kind in ('blas', 'lapack'))
    if not native_verified and not pools:
        raise RuntimeError('native BLAS thread policy not verifiable in this environment')
    info = environment_info()
    info.update(logical_cpu_count=os.cpu_count(), matplotlib=matplotlib.__version__,
                native_thread_policy=native, threadpool_inventory=pools,
                threadpoolctl_available=pools is not None,
                build_configuration=config_text.getvalue(), backend_names=backend_names,
                execution_policy='serial CPU float64; no concurrent solvers; same environment for both methods')
    if platform.system() == 'Darwin':
        try:
            info['cpu_model'] = subprocess.check_output(
                ['sysctl', '-n', 'machdep.cpu.brand_string'], text=True, stderr=subprocess.PIPE).strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            info['cpu_model'] = None
            info['cpu_model_query_error'] = f'{type(exc).__name__}: {exc}'
    return info


def timed_solver_run(method, settings, canonical, reference):
    cost = CostTracker()
    result = None
    verification = None
    error = None
    status = 'NUMERICAL_FAILURE'
    cost.start()
    try:
        with cost.region('t_problem_setup'):
            model = FullLdGModel(Config())
            U0, U_star = canonical.copy(), reference.copy()
            params = MethodParameters(settings['eta'], settings['tau'], settings['J'])
        result = run_method(model, method, params, U0, U_star,
                            max_updates=2000000,
                            max_seconds=600.0 if method == 'p_hisd' else 1800.0,
                            run_kind='final_measured', history_stride=2 if method == 'p_hisd' else 100,
                            cost=cost)
        with cost.region('t_endpoint_certification'):
            verification = verify_final_state(model, result, U_star)
            status = result['status'] if verification['status'] == 'PASS' else 'CERTIFICATION_FAILED'
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    times = cost.stop()
    counts = {} if result is None else result['counters_total']
    cert_actions = (0 if verification is None else verification['eigensolver_hessian_actions']
                    + len(verification['eigenpair_residuals']))
    work = {
        'accepted_outer_updates': None if result is None else result['outer_state_updates'],
        'gradient_evaluations': counts.get('gradient_evaluations'),
        'analytic_HVP_vector_equivalents': counts.get('analytic_hvp_evaluations'),
        'endpoint_sparse_HVP_vector_equivalents': cert_actions,
        'HVP_vector_equivalents': None if result is None else counts['analytic_hvp_evaluations'] + cert_actions,
        'preconditioner_setup_count': cost.calls['t_M_setup'],
        'preconditioner_refresh_update_count': cost.calls['t_M_refresh'],
        'M_apply_RHS_count': cost.calls['t_M_apply'],
        'M_solve_RHS_count': cost.calls['t_M_solve'],
        'scalar_block_M_apply_RHS_count': 5 * cost.calls['t_M_apply'],
        'scalar_block_M_solve_RHS_count': 5 * cost.calls['t_M_solve'],
        'eigensolver_init_calls': cost.calls['t_eigensolver_init'],
        'frame_solver_update_calls': cost.calls['t_eigensolver_update'],
        'eigensolver_frame_solver_calls': cost.calls['t_eigensolver_init'] + cost.calls['t_eigensolver_update'],
        'endpoint_eigensolver_calls': int(verification is not None),
        'sparse_Hessian_assemblies': int(verification is not None),
        'QR_calls': 0, 'M_QR_calls': 0,
        'final_residual': None if result is None else result['final_metrics']['state_residual'],
        'final_frame_residual': None if result is None else result['final_metrics']['frame_residual'],
        'final_Morse_index': None if verification is None else verification['resolved_index'],
        'certification_status': 'NOT_COMPLETED' if verification is None else verification['status'],
    }
    counter_check = result is not None and (
        work['M_apply_RHS_count'] == counts['m_apply_calls']
        and work['M_solve_RHS_count'] == counts['full_vector_m_solve_calls']
        and work['preconditioner_setup_count'] == counts['metric_factorizations']
        and work['frame_solver_update_calls'] == result['outer_state_updates']
    )
    if not times['accounting_PASS'] or not counter_check:
        status = 'INSTRUMENTATION_FAILED'
    record = {'method': method, 'seed': SEED_INITIAL, 'parameters': settings, 'status': status,
              'times': times, 'work': work, 'counter_crosscheck_PASS': counter_check,
              'error': error, 'verification': verification}
    if result is not None:
        result['cost_timing'], result['work_counters'] = times, work
    return record, result


def table_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(jsonable(v), sort_keys=True) if isinstance(v, (dict, list, tuple)) else v
                             for k, v in row.items()})


def summarize_cost_records(records):
    per_seed, aggregate, breakdown, representatives = [], [], [], {}
    for method in ('hisd', 'p_hisd'):
        rows = [r for r in records if r['method'] == method]
        first = [r for r in rows if r['repeat'] < TIMING_PROTOCOL['initial_repeats']]
        required = (9 if len(first) == 3 and np.median([r['times']['t_total'] for r in first]) < .1 else 3)
        if not rows:
            per_seed.append({'method': method, 'seed': SEED_INITIAL, 'repetitions': 0,
                             'required_repetitions': required, 'all_repetitions_certified': False})
            continue
        rep = sorted(rows, key=lambda r: (r['times']['t_total'], r['run_id']))[len(rows)//2]
        representatives[method] = rep['run_id']
        values = np.asarray([r['times']['t_total'] for r in rows])
        complete = len(rows) == required and {r['repeat'] for r in rows} == set(range(required))
        certified = sum(r['status'] == 'CONVERGED' and r['work']['certification_status'] == 'PASS' for r in rows)
        row = {'method': method, 'seed': SEED_INITIAL, 'repetitions': len(rows), 'required_repetitions': required,
               'formal_repetitions_complete': complete, 'certified_repetitions': certified,
               'all_repetitions_certified': complete and certified == required,
               'all_statuses': [r['status'] for r in rows],
               'median_t_total': float(np.median(values)), 'min_t_total': float(np.min(values)),
               'max_t_total': float(np.max(values)), 'IQR_t_total': float(np.quantile(values, .75)-np.quantile(values, .25)),
               'MAD_t_total': float(np.median(np.abs(values-np.median(values)))),
               'representative_run_id': rep['run_id'], 'representative_t_total': rep['times']['t_total'], **rep['work']}
        per_seed.append(row)
        aggregate.append({'method': method, 'seed_count': 1, 'seeds': [SEED_INITIAL],
                          'scope': 'ONE frozen seed; aggregate equals its per-seed median, NOT cross-seed robustness',
                          'total_repetitions': len(rows), 'all_formal_runs_certified': row['all_repetitions_certified'],
                          'geomean_per_seed_median_t_total': row['median_t_total'] if complete else None,
                          'median_per_seed_median_t_total': row['median_t_total'] if complete else None,
                          'representative_run_id': rep['run_id']})
        breakdown.append({'method': method, 'seed': SEED_INITIAL, 'representative_run_id': rep['run_id'],
                          'statistic': 'actual median-total-time repetition; all buckets/counters from THIS run',
                          **rep['times'], **rep['work']})
    return per_seed, aggregate, breakdown, representatives


# Randomize method order with a fixed seed; rebuild problem, metric and frame for each run.
# Use three repetitions, extending to nine when the initial median is below 0.1 seconds.
def run_timing_campaign(selected, canonical, reference):
    environment = timing_environment(configure=True)
    eigh(np.array([[2., -.2], [-.2, 1.]]))
    np.linalg.qr(np.ones((8, 2)) + np.arange(16).reshape(8, 2) * .01)
    schedule = [(method, rep) for method in ('hisd', 'p_hisd') for rep in range(3)]
    rng = np.random.default_rng(TIMING_PROTOCOL['order_rng_seed'])
    rng.shuffle(schedule)
    records, full_results = [], {}
    def perform(method, repeat):
        if timing_environment() != environment:
            raise GateFailure('timing environment changed within campaign')
        run_id = f'{method}_seed{SEED_INITIAL}_r{repeat}'
        print(f'Timing {run_id}: fresh problem/metric/frame; no I/O inside solver timer', flush=True)
        record, result = timed_solver_run(method, selected[method], canonical, reference)
        record.update(run_id=run_id, repeat=repeat)
        try:
            if timing_environment() != environment:
                raise RuntimeError('timing environment changed within campaign')
        except Exception as exc:
            record['status'] = 'ENVIRONMENT_CHANGED'
            record['post_environment_error'] = f'{type(exc).__name__}: {exc}'
        if result is not None:
            record['method_result'] = public_method_result(result)
            full_results[run_id] = result
        records.append(record)
        print(f"  {record['status']}: {record['work']['accepted_outer_updates']} updates, t_total={record['times']['t_total']:.6f}s, accounting residual={record['times']['accounting_residual']:.3e}s", flush=True)
        if record['status'] == 'ENVIRONMENT_CHANGED':
            raise GateFailure(f"environment verification failed: {record['post_environment_error']}")

    for method, repeat in schedule:
        perform(method, repeat)
    supplemental = []
    for method in ('hisd', 'p_hisd'):
        first = [r['times']['t_total'] for r in records if r['method'] == method]
        if np.median(first) < .1:
            supplemental.extend((method, rep) for rep in range(3, 9))
    rng.shuffle(supplemental)
    for method, repeat in supplemental:
        perform(method, repeat)
    per_seed, aggregate, breakdown, representatives = summarize_cost_records(records)
    passed = all(r['all_repetitions_certified'] for r in per_seed) and len(per_seed) == 2
    if not passed:
        raise GateFailure('formal timing repetitions failed; no outputs published')
    representative_results = {method: full_results[run_id] for method, run_id in representatives.items()}
    verifications = {method: next(r['verification'] for r in records if r['run_id'] == run_id)
                     for method, run_id in representatives.items()}
    return representative_results, verifications, {
        'records': records, 'per_seed': per_seed,
        'representative_run_ids': representatives,
    }, environment


def make_paper_pdf(results: dict[str, dict[str, Any]]) -> None:
    with plt.rc_context({'pdf.fonttype': 42, 'ps.fonttype': 42}):
        fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))
        for method, color, label in (
            ('hisd', '#1f77b4', 'HiSD'),
            ('p_hisd', '#d62728', 'p-HiSD'),
        ):
            history = results[method]['history']
            iterations = np.asarray([r['iteration'] for r in history], dtype=float)
            residual = np.asarray([r['state_residual'] for r in history], dtype=float)
            ax.loglog(iterations + 1, residual, color=color, linewidth=1.5, label=label)
        ax.set_xlabel(r'Iteration $m+1$', fontsize=17)
        ax.set_ylabel(r'$h\|g(U_m)\|_2$', fontsize=17)
        ax.set_title('Full-Q Landau-de Gennes Model', fontsize=17)
        ax.legend(fontsize=15, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=1)
        ax.set_ylim(bottom=1e-7)
        fig.tight_layout()
        fig.savefig(PDF_PATH, dpi=300, bbox_inches='tight', metadata={
            'Title': 'Full-Q Landau-de Gennes saddle recovery',
            'Subject': 'N=41; lambda_sq=7; epsilon_bc=0.1. Residual versus iteration; timing/cost evidence saved numerically, not plotted.',
            'Author': 'Self-contained LDG reproduction',
        })
        plt.close(fig)


def compact_outputs(model, selected, initial, results, verifications, campaign, environment, agreement):
    methods, rows = {}, []
    statistics = {row['method']: row for row in campaign['per_seed']}
    for method, label in (('hisd', 'HiSD'), ('p_hisd', 'p-HiSD')):
        result, verification = results[method], verifications[method]
        stat = statistics[method]
        representative = next(record for record in campaign['records']
                              if record['run_id'] == stat['representative_run_id'])
        times, work = representative['times'], representative['work']
        eigenvalues = verification['smallest_six_eigenvalues']
        row = {
            'method': label, **result['parameters'],
            'outer_updates': result['outer_state_updates'],
            'final_state_residual': result['final_metrics']['state_residual'],
            'final_frame_residual': result['final_metrics']['frame_residual'],
            'final_Morse_index': verification['resolved_index'],
            'lambda_1': eigenvalues[0], 'lambda_2': eigenvalues[1],
            'median_t_total': stat['median_t_total'],
            'representative_t_total': times['t_total'],
            **{key: times[key] for key in (
                't_endpoint_certification', 't_M_setup', 't_M_refresh', 't_M_setup_update',
                't_M_apply', 't_M_solve', 't_M_apply_solve',
                't_eigensolver_init', 't_eigensolver_update', 't_eigensolver')},
            **{key: work[key] for key in (
                'gradient_evaluations', 'HVP_vector_equivalents',
                'M_apply_RHS_count', 'M_solve_RHS_count',
                'preconditioner_setup_count', 'preconditioner_refresh_update_count')},
        }
        rows.append(row)
        methods[method] = {
            'parameters': result['parameters'], 'status': result['status'],
            'outer_updates': row['outer_updates'],
            'final_state_residual': row['final_state_residual'],
            'final_frame_residual': row['final_frame_residual'],
            'final_Morse_index': row['final_Morse_index'],
            'smallest_six_eigenvalues': eigenvalues,
            'endpoint_certification_status': verification['status'],
            'median_t_total': row['median_t_total'],
            'representative_t_total': row['representative_t_total'],
            'representative_run_id': representative['run_id'],
            'repetitions': stat['repetitions'],
            'all_repetitions_certified': stat['all_repetitions_certified'],
            'representative_timing': dict(times),
            'work_counters': {key: work[key] for key in (
                'gradient_evaluations', 'analytic_HVP_vector_equivalents',
                'endpoint_sparse_HVP_vector_equivalents', 'HVP_vector_equivalents',
                'M_apply_RHS_count', 'M_solve_RHS_count',
                'scalar_block_M_apply_RHS_count', 'scalar_block_M_solve_RHS_count',
                'preconditioner_setup_count', 'preconditioner_refresh_update_count')},
        }
    methods['p_hisd']['metric'] = selected['metric']
    summary = {
        'model': {'model_name': 'Full-Q Landau-de Gennes',
                  'N': model.N, 'h': model.h, 'dofs': model.dofs,
                  **{key: getattr(model.cfg, key) for key in ('A', 'B', 'C', 'L', 'lambda_sq', 'epsilon_bc')}},
        'initial_state': {
            **{key: selected['initial_state'][key] for key in (
                'seed', 'negative_mode_fraction_of_s_plus', 'smooth_mode_fraction_of_s_plus')},
        },
        'methods': methods,
        'agreement': {'final_state_hL2_distance': agreement, 'threshold': METHOD_AGREEMENT_TOL},
        'timing_protocol': {
            **TIMING_PROTOCOL, 'single_thread_verified': True,
            't_total_definition': 'Problem setup, preconditioner setup, initial frame/eigenspace, outer iterations, all M apply/solve operations, and endpoint certification.',
            'endpoint_certification_included_in_t_total': True,
            'io_and_plotting_excluded': True,
            'printing_serialization_and_output_postprocessing_excluded': True,
            'representative_selection': 'Actual run sorted by (t_total, run_id) at index len(runs)//2; every decomposition bucket and work counter comes from that run.',
        },
        'runtime': {**{key: environment.get(key) for key in (
            'platform', 'machine', 'cpu_model', 'python', 'numpy', 'scipy')},
            'thread_policy': 'single-thread'},
        'decision': 'PASS',
    }
    return rows, summary


def _publication_data(value):
    import re
    if isinstance(value, dict):
        return {key: _publication_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_publication_data(item) for item in value]
    if isinstance(value, str):
        def portable(match):
            path = match.group(0).replace('\\', '/')
            for marker in ('/outputs/', '/code/'):
                if marker in path:
                    return marker[1:] + path.rsplit(marker, 1)[1]
            return path.rstrip('/').rsplit('/', 1)[-1]
        value = re.sub(r"(?:/(?:Users|home|bicmr|private|tmp|opt|Applications)/|[A-Za-z]:[\\/](?:Users|Program Files)[\\/])[^\s\"'`<>;,]+", portable, value)
        return value
    return value


def publish_outputs(results, rows, summary):
    global PDF_PATH
    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    original_pdf_path = PDF_PATH
    try:
        with tempfile.TemporaryDirectory(prefix='.LDG-stage-', dir=OUTPUT_DIR.parent,
                                         ignore_cleanup_errors=True) as staging:
            stage = Path(staging)
            ready, previous = stage / 'ready', stage / 'previous'
            ready.mkdir()
            PDF_PATH = ready / '5.4.pdf'
            make_paper_pdf(results)
            table_csv(ready / 'LDG_results.csv', rows)
            write_json(ready / 'LDG_summary.json', _publication_data(summary))
            had_previous = OUTPUT_DIR.exists()
            if had_previous:
                os.replace(OUTPUT_DIR, previous)
            try:
                os.replace(ready, OUTPUT_DIR)
            except Exception:
                if had_previous:
                    os.replace(previous, OUTPUT_DIR)
                raise
    finally:
        PDF_PATH = original_pdf_path


def main() -> int:
    parser = argparse.ArgumentParser(description='Run the accepted Full-Q LdG paper experiment; no tuning.')
    parser.add_argument('--check-only', action='store_true', help='validate companion inputs and Gates A-D without producing outputs')
    args = parser.parse_args()
    print('Full-Q LdG: N=41, canonical common U0, frozen eta/tau/J.', flush=True)
    print(f'Output directory: {OUTPUT_DIR}', flush=True)
    try:
        model = FullLdGModel(Config())
        gates, selected, reference, initial, D_minus, Psi = validate_inputs_and_gates(model)
        if args.check_only:
            print('CHECK PASS: companion inputs, settings, reconstruction, and Gates A-D.', flush=True)
            return 0
        results, verifications, campaign, environment = run_timing_campaign(selected, initial, reference)
        for method, gate in (('p_hisd', 'E'), ('hisd', 'F')):
            result = results[method]
            verification = verifications[method]
            print(f"[{5 if method == 'p_hisd' else 6}/6] {method} PASS: {result['outer_state_updates']} updates, state={result['final_metrics']['state_residual']:.3e}, frame={result['final_metrics']['frame_residual']:.3e}, index={verification['resolved_index']}", flush=True)
        agreement = model.h * float(np.linalg.norm(results['hisd']['final_state'] - results['p_hisd']['final_state']))
        if agreement > METHOD_AGREEMENT_TOL:
            raise GateFailure(f'two methods disagree: h-L2={agreement:g}')
        rows, summary = compact_outputs(
            model, selected, initial, results, verifications, campaign, environment, agreement)
        publish_outputs(results, rows, summary)
        print(f'Decision: PASS\nFinal-state h-L2 agreement: {agreement:.12e}', flush=True)
        for name in OUTPUT_NAMES:
            print(f'Output: {OUTPUT_DIR / name}', flush=True)
        return 0
    except Exception as exc:
        print(f'Decision: NOT PASSED: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        return 1


FROZEN_CONSTANTS = {'STATE_TOL': 1e-06, 'FRAME_TOL': 1e-07, 'FRAME_NORM_TOL': 1e-10, 'REFERENCE_TOL': 1e-06, 'METHOD_AGREEMENT_TOL': 2e-06, 'J_INNER': 5, 'EIG_TOL': 1e-10, 'EIG_MAXITER': 100000, 'SEED_INITIAL': 20260906, 'SEED_EIG': 2026090601, 'RECONSTRUCTION_TOL': 1e-14, 'CANONICAL_U0_RELATIVE_PATH': 'inputs/common_U0.npz', 'STATE_ORDERING': 'component-major [u1,u2,u3,u4,u5], each interior C-ravel', 'PHASES': ('initialization', 'iteration', 'verification'), 'COUNT_KEYS': ('gradient_evaluations', 'analytic_hvp_evaluations', 'eigensolver_hessian_actions', 'sparse_hessian_assemblies', 'm_apply_calls', 'full_vector_m_solve_calls', 'scalar_rhs_solves', 'metric_factorizations')}

MODEL_REFERENCE = 'arXiv:2204.00478v2; Nonlinearity 36 (2023), 2631-2654'


if __name__ == "__main__":
    raise SystemExit(main())
