#!/usr/bin/env python3
# Section 7.3: compare five index-5 solvers on the modified Rosenbrock energy.
# Use the fixed configurations below for Figure 4 and Tables 3 and 4.
# Save histories, per-run outcomes and summaries to outputs/7.3/.

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from math import fsum
from pathlib import Path
import argparse
import copy
import csv
import ctypes
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import tempfile
import time
import traceback
import warnings


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
OUTPUT_DIR = SCRIPT_DIR.parent.parent / "outputs" / "7.3"
FIGURE_NAME = "3.pdf"
PLOT_MAX_ITER = 5e5


def _ensure_dependencies():
    packages = ("numpy", "scipy", "matplotlib", "threadpoolctl")
    missing = [name for name in packages if importlib.util.find_spec(name) is None]
    if not missing:
        return
    if os.environ.get("SECTION73_REEXEC") != "1":
        for parent in (SCRIPT_DIR, *SCRIPT_DIR.parents):
            for name in (".shared_venv", ".venv"):
                candidate = parent / name / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
                if candidate.exists():
                    os.environ["SECTION73_REEXEC"] = "1"
                    print("Using existing environment: " + str(candidate), flush=True)
                    os.execv(str(candidate), [str(candidate), str(SCRIPT_PATH), *sys.argv[1:]])
    raise SystemExit("Missing Python dependencies: " + ", ".join(missing)
                     + ". Select an interpreter with these packages installed.")


if __name__ == "__main__":
    _ensure_dependencies()

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
_accelerate = None
_initial_policy = None


def set_limits(cache_dir=None):
    global _accelerate, _initial_policy
    already_loaded = any(k in sys.modules for k in ("numpy", "scipy"))
    for name in THREAD_VARIABLES:
        os.environ[name] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    os.environ["MPLBACKEND"] = "Agg"
    if cache_dir is not None:
        os.environ["MPLCONFIGDIR"] = str(Path(cache_dir).resolve())
    elif "MPLCONFIGDIR" not in os.environ:
        os.environ["MPLCONFIGDIR"] = str(Path(__file__).resolve().parent / "artifacts" / "matplotlib_cache")
    info = {"environment": {name: os.environ[name] for name in THREAD_VARIABLES},
            "numpy_or_scipy_loaded_when_first_configured": already_loaded,
            "bytecode_disabled": sys.dont_write_bytecode}
    if platform.system() == "Darwin":
        _accelerate = ctypes.CDLL("/System/Library/Frameworks/Accelerate.framework/Accelerate")
        _accelerate.BLASSetThreading.argtypes = [ctypes.c_uint]
        _accelerate.BLASSetThreading.restype = ctypes.c_int
        _accelerate.BLASGetThreading.argtypes = []
        _accelerate.BLASGetThreading.restype = ctypes.c_uint
        native = {"before": int(_accelerate.BLASGetThreading()),
                  "set_return": int(_accelerate.BLASSetThreading(1)),
                  "after": int(_accelerate.BLASGetThreading()),
                  "required_enum": 1,
                  "scope": "Calling thread: subsequent BLAS and LAPACK calls"}
        if native["set_return"] != 0 or native["after"] != 1:
            raise RuntimeError("Failed to establish native Accelerate single-thread policy")
        info["accelerate"] = native
    if _initial_policy is None:
        _initial_policy = info
    return info


def verify_limits():
    import threadpoolctl
    pools = threadpoolctl.threadpool_info()
    if any(pool["num_threads"] != 1 for pool in pools):
        raise RuntimeError(f"Non-single-thread BLAS/OMP configuration: {pools}")
    native = None if _accelerate is None else int(_accelerate.BLASGetThreading())
    if native is not None and native != 1:
        raise RuntimeError("Accelerate native threading policy changed")
    if any(os.environ.get(name) != "1" for name in THREAD_VARIABLES):
        raise RuntimeError("Thread environment changed")
    return {"threadpools": pools, "accelerate_current_enum": native,
            "threadpoolctl_observability": ("Native single-thread policy checked directly." if _accelerate is not None
                                               else "Thread pools checked with threadpoolctl.")}


def environment_info():
    import contextlib
    import importlib.metadata
    import io
    import subprocess
    import warnings
    import numpy as np
    import scipy
    text = io.StringIO()
    with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stdout(text):
        np.show_config()
        scipy.show_config()
    cpu = platform.processor()
    if platform.system() == "Darwin":
        try:
            cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True, stderr=subprocess.PIPE).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return {"python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "machine": platform.machine(),
            "cpu": cpu, "logical_cpu_count": os.cpu_count(),
            "versions": {name: importlib.metadata.version(name) for name in
                         ("numpy", "scipy", "matplotlib", "threadpoolctl")},
            "initial_policy": _initial_policy, "current_policy": verify_limits(),
            "build_configuration": text.getvalue(),
            "configuration_warnings": [str(w.message) for w in caught]}


get_environment_info = environment_info
LIMIT_POLICY = set_limits(cache_dir=Path(tempfile.gettempdir()) / "section73_matplotlib_cache")

FROZEN_PROTOCOL = {'protocol_version': 'ROSENBROCK_PC_V2',
 'problem': {'d': 1000,
             'index': 5,
             's_neg': -50000.0,
             's_pos': 1.0,
             'initial_radius': 0.1,
             'metric': 'once_frozen_block_jacobi',
             'block_size': 20,
             'epsilon_M': 0.01},
 'seeds': {'historical_and_main': 1,
           'development': [101, 102],
           'held_out': [11, 12, 13, 14],
           'production': [1, 11, 12, 13, 14]},
 'stopping': {'raw_gradient_tolerance': 1e-06,
              'target_distance_tolerance': 0.01,
              'index_delta_abs': 1e-06,
              'state_norm_guard': 100000000.0,
              'frame_frobenius_norm_guard': 100000000.0},
 'budgets': {'screen_max_updates': 4000,
             'screen_max_seconds': 60,
             'full_max_updates': 2000000,
             'full_max_seconds': 300,
             'tuning_total_seconds': 7200,
             'production_total_seconds': 7200},
 'production_timing': {'initial_repeats': 3,
                       'fast_threshold_seconds': 0.1,
                       'fast_repeats_total': 9,
                       'order_rng_seed': 6090603}}
# Prescribed method parameters; this driver does not repeat parameter selection.
LOCKED_CONFIGS = {'AHiSD': {'J': 5,
           'backend': 'RQ',
           'eta_x': 3.150000000000001e-05,
           'method': 'AHiSD',
           'momentum': 0.98,
           'tau_v': 1e-05},
 'BBHiSD': {'J': 5,
            'backend': 'RQ',
            'displacement_cap': 2.0,
            'eta_x_initial': 1e-05,
            'method': 'BBHiSD',
            'tau_v': 1e-05},
 'HiSD': {'J': 5, 'backend': 'RQ', 'eta_x': 2e-05, 'method': 'HiSD', 'tau_v': 1e-05},
 'PCHiSD': {'backend': 'PC', 'eta_v': 2.5e-06, 'eta_x': 2e-05, 'method': 'PCHiSD'},
 'pHiSD': {'J': 5, 'backend': 'RQ', 'eta_x': 0.95, 'method': 'pHiSD', 'tau_v': 0.4}}


for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as sla
from threadpoolctl import threadpool_limits

IMPLEMENTATION_VERSION = 'rosenbrock-core-v2'
COUNTER_NAMES = (
    'attempted_updates', 'accepted_updates', 'n_energy_evaluations',
    'n_gradient_evaluations', 'n_hvp_vector_equivalents',
    'n_hessian_diagonal_builds', 'n_explicit_dense_hessian_constructions',
    'n_tridiag_eigensolver_calls', 'n_small_block_eigendecompositions',
    'n_validation_small_eigensolver_calls',
    'n_init_eigensolver_calls', 'n_init_retries', 'n_frame_solver_calls',
    'n_rq_sweeps', 'n_lobpcg_calls', 'n_M_setup', 'n_M_refresh',
    'n_M_apply_rhs', 'n_M_solve_rhs', 'n_QR', 'n_MQR',
    'n_pc_predictor_stages', 'n_pc_corrector_stages', 'n_bb_fallback',
    'n_bb_cap_active', 'n_rejected_trials', 'n_linear_solves',
    'n_nonlinear_solves',
)


COST_TIMING_VERSION = 'rosenbrock-timing-v1'
COST_EXCLUSIVE_FIELDS = (
    't_problem_setup', 't_M_setup', 't_M_refresh', 't_M_apply', 't_M_solve',
    't_eigensolver_init_net', 't_eigensolver_update_net', 't_other',
    't_endpoint_certification',
)
COST_DIAGNOSTIC_FIELDS = (
    't_eigensolver_init', 't_eigensolver_update', 't_pc_joint_update',
)
_ACTIVE_COST_TIMER = ContextVar('section73_cost_timer', default=None)


# Account for exclusive solver costs separately from inclusive frame timings.
class _CostTimer:
    def __init__(self, start):
        self.start = self.last = start
        self.end = None
        self.stack = ['t_other']
        self.exclusive = dict.fromkeys(COST_EXCLUSIVE_FIELDS, 0.0)
        self.inclusive = dict.fromkeys(COST_DIAGNOSTIC_FIELDS, 0.0)
        self.scope_calls = {}

    def _charge(self, now):
        self.exclusive[self.stack[-1]] += now - self.last
        self.last = now

    def enter(self, bucket, diagnostic):
        if self.end is not None:
            return None
        now = time.perf_counter()
        self._charge(now)
        self.stack.append(bucket if bucket is not None else self.stack[-1])
        name = diagnostic if diagnostic is not None else bucket
        self.scope_calls[name] = self.scope_calls.get(name, 0) + 1
        return now

    def leave(self, started, diagnostic):
        if started is None or self.end is not None:
            return
        now = time.perf_counter()
        self._charge(now)
        self.stack.pop()
        if diagnostic is not None:
            self.inclusive[diagnostic] += now - started

    def stop(self):
        if self.end is None:
            if len(self.stack) != 1:
                raise RuntimeError('Unbalanced cost timing scopes')
            self._charge(time.perf_counter())
            self.end = time.perf_counter()

    def values(self):
        if self.end is None:
            raise RuntimeError('Cost timing has not reached final solver status')
        row = {**self.exclusive, **self.inclusive, 't_total': self.end - self.start}
        row['t_M_setup_update'] = row['t_M_setup'] + row['t_M_refresh']
        row['t_M_apply_solve'] = row['t_M_apply'] + row['t_M_solve']
        row['t_eigensolver'] = row['t_eigensolver_init'] + row['t_eigensolver_update']
        row['t_eigensolver_net'] = (row['t_eigensolver_init_net']
                                   + row['t_eigensolver_update_net'])
        row['t_accounting_sum'] = fsum(self.exclusive.values())
        row['t_accounting_residual'] = row['t_total'] - row['t_accounting_sum']
        row['t_accounting_relative'] = (
            row['t_accounting_residual'] / row['t_total'] if row['t_total'] else 0.0)
        return row


class _CostScope:
    def __init__(self, bucket=None, diagnostic=None):
        self.timer = _ACTIVE_COST_TIMER.get()
        self.bucket, self.diagnostic, self.started = bucket, diagnostic, None

    def __enter__(self):
        if self.timer is not None:
            self.started = self.timer.enter(self.bucket, self.diagnostic)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.timer is not None:
            self.timer.leave(self.started, self.diagnostic)
        return False


def _finish_cost_timing():
    timer = _ACTIVE_COST_TIMER.get()
    if timer is not None:
        timer.stop()


class Counters:
    def __init__(self):
        self.phase = 'init'
        self._phases = {}

    def add(self, name, n=1):
        row = self._phases.setdefault(self.phase, {})
        row[name] = row.get(name, 0) + int(n)

    def totals(self, include_diagnostics=False):
        result = dict.fromkeys(COUNTER_NAMES, 0)
        for phase, row in self._phases.items():
            if phase == 'diagnostics' and not include_diagnostics:
                continue
            for key, value in row.items():
                result[key] = result.get(key, 0) + value
        return result

    def by_phase(self):
        return {phase: {**dict.fromkeys(COUNTER_NAMES, 0), **self._phases.get(phase, {})}
                for phase in ('problem_setup', 'metric_setup', 'init',
                              'iteration', 'certification', 'diagnostics')}


def _count(counters, name, n=1):
    if counters is not None:
        counters.add(name, n)


def energy(x, s):
    return float(np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1.0-x[:-1])**2)
                 + np.sum(s * np.arctan(x-1.0)**2))


def gradient(x, s):
    g = np.zeros_like(x)
    r = x[1:] - x[:-1]**2
    g[:-1] += -400.0*x[:-1]*r + 2.0*(x[:-1]-1.0)
    g[1:] += 200.0*r
    z = x-1.0
    g += 2.0*s*np.arctan(z)/(1.0+z*z)
    return g


# The nearest-neighbor energy gives a tridiagonal Hessian; store its two diagonals.
def hessian_diagonals(x, s):
    diag = np.empty_like(x)
    diag[:-1] = 1200.0*x[:-1]**2 - 400.0*x[1:] + 2.0
    diag[1:-1] += 200.0
    diag[-1] = 200.0
    z = x-1.0
    diag += 2.0*s*(1.0-2.0*z*np.arctan(z))/(1.0+z*z)**2
    return diag, -400.0*x[:-1]


def tridiag_apply(diag, off, v):
    if v.ndim == 1:
        out = diag*v
        out[:-1] += off*v[1:]
        out[1:] += off*v[:-1]
    else:
        out = diag[:, None]*v
        out[:-1] += off[:, None]*v[1:]
        out[1:] += off[:, None]*v[:-1]
    return out


class Problem:
    """Cache keyed by exact caller-owned state/stage identity, never proximity.

    A token denotes an immutable mathematical state. The solver makes a fresh
    accepted token after every update and a distinct predictor token per step.
    """
    def __init__(self, s, counters=None):
        self.s = np.array(s, dtype=np.float64, copy=True)
        self.counters = counters if counters is not None else Counters()
        self.cache = {}

    def energy(self, x, token=None):
        row = self.cache.setdefault(token, {}) if token is not None else {}
        if 'energy' not in row:
            self.counters.add('n_energy_evaluations')
            row['energy'] = energy(x, self.s)
        return row['energy']

    def gradient(self, x, token=None, force=False):
        row = self.cache.setdefault(token, {}) if token is not None else {}
        if force or 'gradient' not in row:
            self.counters.add('n_gradient_evaluations')
            row['gradient'] = gradient(x, self.s)
        return row['gradient']

    def diagonals(self, x, token=None, force=False):
        row = self.cache.setdefault(token, {}) if token is not None else {}
        if force or 'diagonals' not in row:
            self.counters.add('n_hessian_diagonal_builds')
            row['diagonals'] = hessian_diagonals(x, self.s)
        return row['diagonals']

    def hvp(self, x, v, token=None):
        diag, off = self.diagonals(x, token)
        self.counters.add('n_hvp_vector_equivalents', 1 if v.ndim == 1 else v.shape[1])
        return tridiag_apply(diag, off, v)

    def clear_except(self, token):
        row = self.cache.get(token, {})
        self.cache = {token: row}


class NumericalFailure(ArithmeticError):
    def __init__(self, status, stage, details=None):
        super().__init__(f'{status}: {stage}')
        self.status, self.stage = status, stage
        self.details = {} if details is None else details


class IdentityMetric:
    def apply(self, z):
        return z.copy()

    def solve(self, z):
        return z.copy()


class BlockMetric:
    """Frozen blocks of |H_BB(x0)| + eps I, with no global dense operator."""
    def __init__(self, diag, off, block_size=20, eps=.01, counters=None):
        with _CostScope('t_M_setup'):
            if len(diag) % block_size or eps <= 0:
                raise ValueError('Complete blocks and positive epsilon are required')
            self.n, self.b, self.counters = len(diag), block_size, counters
            _count(counters, 'n_M_setup')
            mats, invs = [], []
            for i in range(0, self.n, self.b):
                block = (np.diag(diag[i:i+self.b])
                         + np.diag(off[i:i+self.b-1], 1)
                         + np.diag(off[i:i+self.b-1], -1))
                _count(counters, 'n_small_block_eigendecompositions')
                w, q = la.eigh(block)
                w = np.abs(w)+eps
                mats.append((q*w)@q.T)
                invs.append((q*(1.0/w))@q.T)
            self.mats, self.invs = np.stack(mats), np.stack(invs)

    def _op(self, mats, z):
        q = 1 if z.ndim == 1 else z.shape[1]
        return (mats@z.reshape(-1, self.b, q)).reshape(z.shape)

    def apply(self, z):
        with _CostScope('t_M_apply'):
            _count(self.counters, 'n_M_apply_rhs', 1 if z.ndim == 1 else z.shape[1])
            return self._op(self.mats, z)

    def solve(self, z):
        with _CostScope('t_M_solve'):
            _count(self.counters, 'n_M_solve_rhs', 1 if z.ndim == 1 else z.shape[1])
            return self._op(self.invs, z)


# Two-pass modified Gram-Schmidt enforces V.T M V = I and detects rank loss.
def mqr(v, metric, counters=None):
    _count(counters, 'n_MQR')
    q = v.copy()
    mq = metric.apply(q)
    for i in range(q.shape[1]):
        for _ in range(2):
            for j in range(i):
                c = q[:, j]@mq[:, i]
                q[:, i] -= c*q[:, j]
                mq[:, i] -= c*mq[:, j]
        n2 = q[:, i]@mq[:, i]
        if not np.isfinite(n2) or n2 <= 1e-28:
            raise NumericalFailure('FRAME_BREAKDOWN', 'MGS2', {'column': i, 'norm2': float(n2)})
        q[:, i] /= np.sqrt(n2)
        mq[:, i] /= np.sqrt(n2)
    gram_error = float(la.norm(q.T@mq-np.eye(q.shape[1]), 2))
    if not np.isfinite(gram_error) or gram_error > 1e-8:
        raise NumericalFailure('FRAME_BREAKDOWN', 'MGS2_return_gram',
                               {'gram_defect_2': gram_error})
    return q


def euclidean_qr(v, counters=None):
    _count(counters, 'n_QR')
    if not np.isfinite(v).all():
        raise NumericalFailure('DIVERGED_NONFINITE', 'QR_input')
    q, r = la.qr(v, mode='economic')
    if (not np.isfinite(r).all()
            or np.min(np.abs(np.diag(r))) <= 1e-14*max(1., la.norm(v, 'fro'))):
        raise NumericalFailure('FRAME_BREAKDOWN', 'QR_rank_loss')
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return q*signs


def fix_initial_signs(v):
    q = v.copy()
    inds = np.argmax(np.abs(q), axis=0)
    signs = np.sign(q[inds, np.arange(q.shape[1])])
    signs[signs == 0] = 1.0
    return q*signs


def _hvp_from_diagonals(diag, off, v, counters):
    _count(counters, 'n_hvp_vector_equivalents', 1 if v.ndim == 1 else v.shape[1])
    return tridiag_apply(diag, off, v)


def invariant_validation(diag, off, v, metric=None, counters=None):
    hv = _hvp_from_diagonals(diag, off, v, counters)
    mv = v if metric is None else metric.apply(v)
    gram_error = float(la.norm(v.T@mv-np.eye(v.shape[1]), 2))
    lam = v.T@hv
    residual = hv-mv@lam
    if metric is None:
        numerator = la.norm(residual, 'fro')
        denominator = max(1., la.norm(hv, 'fro'))
    else:
        mir = metric.solve(residual)
        mihv = metric.solve(hv)
        numerator = np.sqrt(max(0., float(np.sum(residual*mir))))
        denominator = max(1., np.sqrt(max(0., float(np.sum(hv*mihv)))))
    value = float(numerator/denominator)
    _count(counters, 'n_validation_small_eigensolver_calls')
    return {'gram_defect_2': gram_error, 'invariant_residual': value,
            'rayleigh_eigenvalues': la.eigvalsh(lam).tolist(),
            'passed': bool(np.isfinite(value) and np.isfinite(gram_error)
                           and gram_error <= 1e-10 and value <= 1e-8)}


# Initialize the k lowest ordinary or generalized modes and check their subspace residual.
def init_frame(diag, off, k, metric=None, counters=None, return_validation=False):
    with _CostScope('t_eigensolver_init_net', 't_eigensolver_init'):
        n = len(diag)
        if metric is None:
            _count(counters, 'n_init_eigensolver_calls')
            _count(counters, 'n_tridiag_eigensolver_calls')
            try:
                w, v = la.eigh_tridiagonal(diag, off, select='i', select_range=(0, k-1))
            except la.LinAlgError as exc:
                raise NumericalFailure('INIT_EIG_FAIL', 'euclidean_initial_eigensolver',
                                       {'exception': str(exc), 'traceback': traceback.format_exc()}) from exc
            v = fix_initial_signs(v)
            validation = invariant_validation(diag, off, v, counters=counters)
            validation['eigenvalues'] = w.tolist()
            validation['attempts'] = [{'backend': 'eigh_tridiagonal', 'passed': validation['passed']}]
            if not validation['passed'] or not np.all(w < 0):
                raise NumericalFailure('INIT_EIG_FAIL', 'euclidean_initial_validation', validation)
            return (v, validation) if return_validation else v
        a = sla.LinearOperator((n, n), dtype=np.float64,
                matvec=lambda z: _hvp_from_diagonals(diag, off, z, counters),
                matmat=lambda z: _hvp_from_diagonals(diag, off, z, counters))
        b = sla.LinearOperator((n, n), matvec=metric.apply, matmat=metric.apply, dtype=np.float64)
        bi = sla.LinearOperator((n, n), matvec=metric.solve, matmat=metric.solve, dtype=np.float64)
        attempts = []
        for retry, (ncv, maxiter) in enumerate(((20, 300), (40, 1000))):
            if retry:
                _count(counters, 'n_init_retries')
            _count(counters, 'n_init_eigensolver_calls')
            try:
                w, v = sla.eigsh(a, k=k, M=b, Minv=bi, which='SA', tol=1e-10,
                                ncv=ncv, maxiter=maxiter, v0=np.ones(n)/np.sqrt(n))
                order = np.argsort(w)
                v = fix_initial_signs(mqr(v[:, order], metric, counters))
                validation = invariant_validation(diag, off, v, metric, counters)
                validation['eigenvalues'] = w[order].tolist()
                passed = validation['passed'] and bool(np.all(w < 0))
                attempts.append({'ncv': ncv, 'maxiter': maxiter, 'passed': passed,
                                 'validation': dict(validation)})
                if passed:
                    validation['attempts'] = attempts
                    return (v, validation) if return_validation else v
            except (sla.ArpackError, sla.ArpackNoConvergence, NumericalFailure, la.LinAlgError) as exc:
                attempts.append({'ncv': ncv, 'maxiter': maxiter, 'passed': False,
                                 'exception_type': type(exc).__name__, 'exception': str(exc),
                                 'traceback': traceback.format_exc()})
        raise NumericalFailure('INIT_EIG_FAIL', 'generalized_initial_validation', {'attempts': attempts})


# Reflection uses V.T g because the frame is orthonormal in the chosen metric.
def reflected_direction(g, v, metric=None):
    invg = g if metric is None else metric.solve(g)
    return -invg+2.0*v@(v.T@g)


def state_step(x, xprev, g, v, h, momentum=0., metric=None):
    return x+h*reflected_direction(g, v, metric)+momentum*(x-xprev)


# Sweep through the unstable directions, then restore Euclidean or metric orthogonality.
def frame_sweep(diag, off, v, tau_v, inner_steps, metric=None, counters=None):
    with _CostScope('t_eigensolver_update_net', 't_eigensolver_update'):
        q = v.copy()
        for _ in range(inner_steps):
            _count(counters, 'n_rq_sweeps')
            for i in range(q.shape[1]):
                u = q[:, i].copy()
                hu = _hvp_from_diagonals(diag, off, u, counters)
                y = hu if metric is None else metric.solve(hu)
                r = y-u*(u@hu)
                if i:
                    r -= 2.0*q[:, :i]@(q[:, :i].T@hu)
                q[:, i] = u-tau_v*r
            q = euclidean_qr(q, counters) if metric is None else mqr(q, metric, counters)
        return q


def lobpcg_frame(diag, off, v, maxiter, metric=None, counters=None, tol=1e-8):
    with _CostScope('t_eigensolver_update_net', 't_eigensolver_update'):
        _count(counters, 'n_lobpcg_calls')
        n = len(diag)
        a = sla.LinearOperator((n, n), dtype=np.float64,
                matvec=lambda z: _hvp_from_diagonals(diag, off, z, counters),
                matmat=lambda z: _hvp_from_diagonals(diag, off, z, counters))
        b = None if metric is None else sla.LinearOperator(
                (n, n), matvec=metric.apply, matmat=metric.apply, dtype=np.float64)
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter('always')
            w, q, residual_history = sla.lobpcg(a, v.copy(), B=b, M=None, tol=tol,
                    maxiter=maxiter, largest=False, retResidualNormsHistory=True)
        q = q[:, np.argsort(w)]
        q = euclidean_qr(q, counters) if metric is None else mqr(q, metric, counters)
        if not np.isfinite(q).all():
            raise NumericalFailure('DIVERGED_NONFINITE', 'LOBPCG_return')
        # QR/MQR return is checked without an extra M action by MGS2 itself;
        # the Euclidean Gram check requires no oracle operations.
        if metric is None and la.norm(q.T@q-np.eye(q.shape[1]), 2) > 1e-8:
            raise NumericalFailure('FRAME_BREAKDOWN', 'LOBPCG_return_gram')
        return q, [str(record.message) for record in records]


def field_from_g_hv(g, hv, v):
    c = v.T@hv
    a = 2.0*np.triu(c, k=1)+np.diag(np.diag(c))
    return -g+2.0*v@(v.T@g), -hv+v@a


# Explicit predictor-corrector step for the coupled state and frame dynamics.
def pc_step(x, v, a, b, oracle):
    with _CostScope(diagnostic='t_pc_joint_update'):
        g, hv = oracle(x, v)
        f0, q0 = field_from_g_hv(g, hv, v)
        xp, vp = x+a*f0, v+b*q0
        gp, hvp = oracle(xp, vp)
        fp, qp = field_from_g_hv(gp, hvp, vp)
        # Both correctors use the previous accepted state as their base.
        return .5*(xp+x+a*fp), .5*(vp+v+b*qp)


@dataclass
class BBHistory:
    previous_x: np.ndarray | None = None
    previous_direction: np.ndarray | None = None


# Estimate the BB step from displacement and direction differences, then cap displacement.
def bb_step(x, direction, history, h0, cap):
    if h0 <= 0 or cap <= 0:
        raise ValueError('h0 and cap must be positive')
    h, fallback = h0, False
    if history.previous_x is not None:
        if history.previous_direction is None:
            raise ValueError('Incomplete BB history')
        s = x-history.previous_x
        y = direction-history.previous_direction
        scale = float(np.max(np.abs(y)))
        if not np.isfinite(scale) or scale < np.finfo(float).tiny:
            fallback = True
        else:
            ys = y/scale
            h = abs(float(s@ys)/(scale*float(ys@ys)))
            if not np.isfinite(h) or h <= 0:
                h, fallback = h0, True
    norm = float(la.norm(direction))
    hcap = cap/norm if norm > 0 else np.inf
    active = bool(h > hcap)
    h = min(h, hcap)
    return x+h*direction, BBHistory(x.copy(), direction.copy()), {
        'step': h, 'bb_fallback': fallback, 'cap_active': active}


def _sturm_count(diag, off, shift=0.):
    """Count eigenvalues below shift using scalar tridiagonal LDL inertia."""
    pivmin = np.finfo(float).tiny*max(1., float(np.max(off*off, initial=0.)))
    pivot = float(diag[0]-shift)
    if abs(pivot) < pivmin:
        pivot = -pivmin
    count = int(pivot < 0)
    for i in range(1, len(diag)):
        pivot = float(diag[i]-shift-off[i-1]*(off[i-1]/pivot))
        if abs(pivot) < pivmin:
            pivot = -pivmin
        count += int(pivot < 0)
    return count


def certify(problem, x, protocol):
    """Independent fresh gradient/diagonals and first k+1 eigenvalues."""
    stop = protocol['stopping']
    k = protocol['problem']['index']
    token = ('endpoint_certification',)
    g = problem.gradient(x, token, force=True)
    diag, off = problem.diagonals(x, token, force=True)
    if not (np.isfinite(g).all() and np.isfinite(diag).all() and np.isfinite(off).all()):
        raise NumericalFailure('DIVERGED_NONFINITE', 'endpoint_certification_operators')
    problem.counters.add('n_tridiag_eigensolver_calls')
    eigenvalues = la.eigvalsh_tridiagonal(diag, off, select='i', select_range=(0, k))
    row_sum = np.abs(diag)
    row_sum[:-1] += np.abs(off)
    row_sum[1:] += np.abs(off)
    hnorm = float(np.max(row_sum))
    delta = stop['index_delta_abs']
    index = _sturm_count(diag, off)
    # Inertia at both sign margins detects eigenvalues too close to zero to resolve.
    below_negative_delta = _sturm_count(diag, off, -delta)
    below_positive_delta = _sturm_count(diag, off, delta)
    resolved = below_negative_delta == below_positive_delta
    index_ok = bool(eigenvalues[k-1] < -delta and eigenvalues[k] > delta)
    residual = float(la.norm(g))
    distance = float(la.norm(x-1.0))
    return {'raw_residual': residual, 'residual': residual, 'distance': distance,
            'index': index, 'eigenvalues': eigenvalues.tolist(),
            'eigenvalues_first6': eigenvalues.tolist(),
            'sign_margin': float(min(-eigenvalues[k-1], eigenvalues[k])),
            'delta': delta, 'hessian_inf_norm': hnorm, 'index_ok': index_ok,
            'index_resolved': bool(resolved), 'inertia_below_minus_delta': below_negative_delta,
            'inertia_below_plus_delta': below_positive_delta,
            'index_computation': 'tridiagonal_sturm_inertia',
            'raw_tolerance_satisfied': bool(residual < stop['raw_gradient_tolerance']),
            'target_distance_satisfied': bool(distance < stop['target_distance_tolerance'])}


def _guard(x, v, stop, stage):
    if not (np.isfinite(x).all() and np.isfinite(v).all()):
        raise NumericalFailure('DIVERGED_NONFINITE', stage)
    xn, vn = float(la.norm(x)), float(la.norm(v, 'fro'))
    if xn > stop['state_norm_guard'] or vn > stop['frame_frobenius_norm_guard']:
        raise NumericalFailure('DIVERGED_NORM', stage, {'state_norm': xn, 'frame_norm': vn})


def _failure_category(status):
    if status == 'CONVERGED_TARGET':
        return None
    if status in ('BUDGET_ITER', 'BUDGET_TIME', 'SCREEN_BUDGET', 'RESOURCE_LIMIT'):
        return 'BUDGET_LIMITED'
    if status.startswith('DIVERGED_'):
        return 'DIVERGED_NUMERICALLY'
    if status == 'CONVERGED_OTHER_INDEX5':
        return 'WRONG_SADDLE'
    if status in ('WRONG_INDEX', 'UNRESOLVED_INDEX'):
        return status
    if status in ('INIT_EIG_FAIL', 'FRAME_BREAKDOWN'):
        return status
    return 'IMPLEMENTATION_FAILURE'


def _diagnostic_record(problem, x, v, token, metric, updates, is_pc):
    counters = problem.counters
    oldphase = counters.phase
    counters.phase = 'diagnostics'
    started = time.perf_counter()
    try:
        diag, off = problem.diagonals(x, ('diagnostic', updates), force=True)
        if is_pc:
            raw_gram = float(la.norm(v.T@v-np.eye(v.shape[1]), 2))
            q = euclidean_qr(v.copy(), counters)
            row = invariant_validation(diag, off, q, counters=counters)
            row['diagnostic_copy_gram_defect_2'] = row['gram_defect_2']
            row['gram_defect_2'] = raw_gram
            row['column_norms'] = la.norm(v, axis=0).tolist()
        else:
            row = invariant_validation(diag, off, v, metric, counters)
        return {'updates': updates, **row, 'diagnostic_seconds': time.perf_counter()-started}
    finally:
        counters.phase = oldphase


def _solve(config, x0, protocol, max_updates, max_seconds, diagnostic=False, _start=None):
    """Fresh run from a preconstructed x0; no setup is shared across calls.

    Final frame may describe the previous accepted state: the frozen protocol
    stops before its unnecessary final EigenSol. Endpoint index is independent.
    """
    start = time.perf_counter() if _start is None else _start
    counters = Counters()
    counters.phase = 'problem_setup'
    times = dict.fromkeys(('t_problem_setup', 't_metric_setup', 't_init_eigenspace',
                          't_iteration', 't_endpoint_certification'), 0.)
    phase_start, phase_name = start, 't_problem_setup'
    method = config['method']
    is_pc, is_p = method == 'PCHiSD', method == 'pHiSD'
    status, failure_stage, failure_details = 'BUDGET_ITER', None, {}
    x = np.array(x0, dtype=np.float64, copy=True)
    v = np.empty((x.size, 0), dtype=np.float64)
    g, metric, problem = None, None, None
    updates, attempts, frame_state_update = 0, 0, None
    history, warning_records, bb_steps, diagnostic_history = [], [], [], []
    initial_validation, certification = {}, {}
    previous_x = x.copy()
    bb_history = BBHistory()
    token = ('accepted', 0)
    stop = protocol['stopping']

    def transition(name, counter_phase):
        nonlocal phase_name, phase_start
        now = time.perf_counter()
        times[phase_name] += now-phase_start
        phase_name, phase_start = name, now
        counters.phase = counter_phase

    def record(final=False, step=None):
        totals = counters.totals()
        row = {'updates': updates, 'residual': None if g is None else float(np.linalg.norm(g)),
               'elapsed': time.perf_counter()-start, 'final': final,
               **{name: totals[name] for name in (
                   'n_gradient_evaluations', 'n_hvp_vector_equivalents',
                   'n_M_apply_rhs', 'n_M_solve_rhs')}}
        if step is not None:
            row['state_step'] = float(step)
        if history and history[-1]['updates'] == updates:
            if 'state_step' in history[-1] and step is None:
                row['state_step'] = history[-1]['state_step']
            history[-1] = row
        else:
            history.append(row)

    try:
        with _CostScope('t_problem_setup'):
            if method not in ('HiSD', 'AHiSD', 'PCHiSD', 'BBHiSD', 'pHiSD'):
                raise ValueError(f'Unknown method {method!r}')
            if x.shape != (protocol['problem']['d'],):
                raise ValueError('x0 shape conflicts with frozen protocol')
            if max_updates < 0 or max_seconds <= 0:
                raise ValueError('Invalid run budgets')
            p = protocol['problem']
            s = np.full(x.size, p['s_pos'], dtype=np.float64)
            s[:p['index']] = p['s_neg']
            problem = Problem(s, counters)
            _guard(x, v, stop, 'initial_state')
            g = problem.gradient(x, token)
            diag, off = problem.diagonals(x, token)
            if not np.isfinite(g).all():
                raise NumericalFailure('DIVERGED_NONFINITE', 'initial_gradient')
        if is_p:
            transition('t_metric_setup', 'metric_setup')
            metric = BlockMetric(diag, off, p['block_size'], p['epsilon_M'], counters)
        transition('t_init_eigenspace', 'init')
        v, initial_validation = init_frame(diag, off, p['index'], metric, counters, True)
        _guard(x, v, stop, 'initial_frame')
        frame_state_update = 0
        transition('t_iteration', 'iteration')
        record()
        if diagnostic:
            diagnostic_history.append(_diagnostic_record(problem, x, v, token, metric, 0, is_pc))
        while True:
            raw_residual = float(la.norm(g))
            # Stop on the raw gradient; certify the target and Morse index afterward.
            if raw_residual < stop['raw_gradient_tolerance']:
                status = 'RESIDUAL_CONVERGED'
                break
            if updates >= max_updates:
                status = 'BUDGET_ITER'
                break
            if time.perf_counter()-start >= max_seconds:
                status = 'BUDGET_TIME'
                break
            if not is_pc and updates > 0:
                diag, off = problem.diagonals(x, token)
                counters.add('n_frame_solver_calls')
                backend = config['backend'].upper()
                if backend == 'RQ':
                    v_candidate = frame_sweep(diag, off, v, config['tau_v'], config['J'], metric, counters)
                elif backend == 'LOBPCG':
                    v_candidate, warns = lobpcg_frame(diag, off, v, config['maxiter'], metric,
                                                       counters, config.get('tol', 1e-8))
                    warning_records.extend({'update': updates, 'stage': 'frame_lobpcg', 'message': msg}
                                           for msg in warns)
                else:
                    raise ValueError(f'Unknown EigenSol backend {backend!r}')
                _guard(x, v_candidate, stop, 'frame_update')
                v = v_candidate
                frame_state_update = updates
            attempts += 1
            counters.add('attempted_updates')
            next_token = ('accepted', updates+1)
            step = None
            if is_pc:
                stage_calls = 0

                def oracle(stage_x, stage_v):
                    nonlocal stage_calls
                    if stage_calls == 0:
                        stage_token = token
                        counters.add('n_pc_predictor_stages')
                        stage_name = 'pc_predictor_rhs'
                    else:
                        # Guard the entire predictor before paying for the second stage.
                        _guard(stage_x, stage_v, stop, 'pc_predictor_state')
                        stage_token = ('predictor', updates)
                        counters.add('n_pc_corrector_stages')
                        stage_name = 'pc_corrector_rhs'
                    stage_calls += 1
                    gg = problem.gradient(stage_x, stage_token)
                    hv = problem.hvp(stage_x, stage_v, stage_token)
                    if not (np.isfinite(gg).all() and np.isfinite(hv).all()):
                        raise NumericalFailure('DIVERGED_NONFINITE', stage_name)
                    return gg, hv

                xn, vn = pc_step(x, v, config['eta_x'], config['eta_v'], oracle)
                step = config['eta_x']
            else:
                direction = reflected_direction(g, v, metric)
                if not np.isfinite(direction).all():
                    raise NumericalFailure('DIVERGED_NONFINITE', 'state_direction')
                if method == 'BBHiSD':
                    xn, next_bb_history, info = bb_step(x, direction, bb_history,
                            config['eta_x_initial'], config['displacement_cap'])
                    step = info['step']
                    bb_steps.append(step)
                    counters.add('n_bb_fallback', info['bb_fallback'])
                    counters.add('n_bb_cap_active', info['cap_active'])
                else:
                    step = config['eta_x']
                    momentum = config['momentum'] if method == 'AHiSD' else 0.
                    xn = x+step*direction+momentum*(x-previous_x)
                vn = v
            _guard(xn, vn, stop, 'accepted_candidate')
            gn = problem.gradient(xn, next_token)
            if not np.isfinite(gn).all():
                raise NumericalFailure('DIVERGED_NONFINITE', 'accepted_candidate_gradient')
            previous_x = x
            x, v, g, token = xn, vn, gn, next_token
            if method == 'BBHiSD':
                bb_history = next_bb_history
            updates += 1
            counters.add('accepted_updates')
            if is_pc:
                frame_state_update = updates
            problem.clear_except(token)
            if updates <= 10 or updates % 10 == 0:
                record(step=step)
            if diagnostic and (updates <= 10 or updates % 1000 == 0):
                diagnostic_history.append(_diagnostic_record(problem, x, v, token, metric, updates, is_pc))
    except NumericalFailure as exc:
        status, failure_stage, failure_details = exc.status, exc.stage, exc.details
        if status == 'INIT_EIG_FAIL':
            initial_validation = dict(exc.details)
    except Exception as exc:
        status, failure_stage = 'IMPLEMENTATION_FAILURE', phase_name
        failure_details = {'exception_type': type(exc).__name__, 'exception': str(exc),
                           'traceback': traceback.format_exc()}

    with _CostScope('t_endpoint_certification'):
        transition('t_endpoint_certification', 'certification')
        if problem is not None and np.isfinite(x).all() and float(la.norm(x)) <= stop['state_norm_guard']:
            try:
                certification = certify(problem, x, protocol)
                # Align the actual returned residual to independent certification.
                g = problem.cache[('endpoint_certification',)]['gradient']
                if status == 'RESIDUAL_CONVERGED':
                    if not certification['raw_tolerance_satisfied']:
                        status = 'CERTIFICATION_RESIDUAL_FAILED'
                    elif not certification['index_resolved']:
                        status = 'UNRESOLVED_INDEX'
                    elif not certification['index_ok']:
                        status = 'WRONG_INDEX'
                    elif not certification['target_distance_satisfied']:
                        status = 'CONVERGED_OTHER_INDEX5'
                    else:
                        status = 'CONVERGED_TARGET'
            except Exception as exc:
                detail = {'exception_type': type(exc).__name__, 'exception': str(exc),
                          'traceback': traceback.format_exc()}
                if failure_stage is None:
                    if isinstance(exc, NumericalFailure):
                        status, failure_stage = exc.status, exc.stage
                    else:
                        status, failure_stage = 'IMPLEMENTATION_FAILURE', 'endpoint_certification'
                    failure_details = detail
                else:
                    failure_details['endpoint_exception'] = detail
    _finish_cost_timing()
    if diagnostic and v.shape[1] and np.isfinite(x).all() and np.isfinite(v).all():
        try:
            if (diagnostic_history and diagnostic_history[-1]['updates'] == updates
                    and failure_stage is None):
                # The sample at this same accepted state already contains the
                # required final check. Do not compute it twice.
                diagnostic_history[-1]['final'] = True
            else:
                row = _diagnostic_record(problem, x, v, token, metric, updates, is_pc)
                row['final'] = True
                diagnostic_history.append(row)
        except Exception as exc:
            diagnostic_history.append({'updates': updates, 'final': True,
                'diagnostic_failure': str(exc), 'traceback': traceback.format_exc()})
    record(final=True)
    totals = counters.totals()
    bb_statistics = ({'count': len(bb_steps), 'min': float(np.min(bb_steps)),
                      'median': float(np.median(bb_steps)), 'p90': float(np.quantile(bb_steps, .9)),
                      'max': float(np.max(bb_steps))} if bb_steps else
                     {'count': 0, 'min': None, 'median': None, 'p90': None, 'max': None})
    summary = {
        'implementation_version': IMPLEMENTATION_VERSION,
        'method': method, 'config': dict(config), 'status': status,
        'failure_category': _failure_category(status), 'failure_stage': failure_stage,
        'failure_details': failure_details, 'accepted_updates': updates,
        'attempted_updates': attempts,
        'final_residual': None if g is None else float(np.linalg.norm(g)),
        'distance': float(la.norm(x-1.0)) if np.isfinite(x).all() else None,
        'certification': certification, 'times': times, 'counters': totals,
        'counters_by_phase': counters.by_phase(), 'warning_records': warning_records,
        'bb_step_statistics': bb_statistics, 'initial_frame_validation': initial_validation,
        'eigensolver_iterations': {'ARPACK': 'unavailable', 'LOBPCG': 'unavailable',
            'LOBPCG_internal_small_eigendecompositions': 'unavailable'},
        'final_frame_state_update': frame_state_update,
        'final_frame_note': ('Native PC accepted frame' if is_pc else
                            'Last EigenSol frame; no unnecessary final frame update'),
        'diagnostic': bool(diagnostic),
        'diagnostic_seconds': sum(r.get('diagnostic_seconds', 0.) for r in diagnostic_history),
    }
    result = {'summary': summary, 'history': history, 'final_x': x.copy(),
              'final_frame': v.copy(), 'bb_steps': np.array(bb_steps, dtype=np.float64),
              'diagnostic_history': diagnostic_history,
              'last_finite_state': {'x': x.copy(), 'frame': v.copy(), 'updates': updates}}
    end = time.perf_counter()
    times[phase_name] += end-phase_start
    times['t_total'] = end-start
    history[-1]['elapsed'] = times['t_total']
    return result


def solve(config, x0, protocol, max_updates, max_seconds, diagnostic=False):
    start = time.perf_counter()
    _cost_timer = _CostTimer(start)
    _cost_context_token = _ACTIVE_COST_TIMER.set(_cost_timer)
    try:
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter('always')
            result = _solve(config, x0, protocol, max_updates, max_seconds, diagnostic, _start=start)
        result['summary']['warning_records'].extend(
            {'stage': 'numerical_kernel', 'category': record.category.__name__,
             'message': str(record.message), 'filename': record.filename, 'line': record.lineno}
            for record in records)
        end = time.perf_counter()
        times = result['summary']['times']
        wrapper_seconds = (end-start)-times['t_total']
        times['t_endpoint_certification'] += wrapper_seconds
        times['t_total'] = end-start
        result['history'][-1]['elapsed'] = times['t_total']
        _cost_values = _cost_timer.values()
        result['summary']['timing_legacy_phases'] = dict(times)
        times.clear()
        times.update(_cost_values)
        result['history'][-1]['elapsed'] = times['t_total']
        result['summary']['timing_accounting'] = {
            'version': COST_TIMING_VERSION,
            'exclusive_fields': list(COST_EXCLUSIVE_FIELDS),
            'nested_diagnostic_fields': list(COST_DIAGNOSTIC_FIELDS),
            'scope_calls': dict(_cost_timer.scope_calls),
            'total_start': 'solve entry, before problem/metric/frame initialization',
            'total_end': 'final certification/status, before final record/statistics/result packaging',
            'other_measurement': 'direct sum of uncovered event intervals; not total minus components',
            'residual_definition': 'independent t_total minus exclusive event-partition sum',
            'pc_frame_classification': 'native coupled state/frame step is in t_other; t_pc_joint_update is an inclusive diagnostic',
            'formal_timing_eligible': not diagnostic,
        }
        return result
    finally:
        _ACTIVE_COST_TIMER.reset(_cost_context_token)


METHODS = ("HiSD", "AHiSD", "PCHiSD", "BBHiSD", "pHiSD")
LABELS = dict(zip(METHODS, ("HiSD", "A-HiSD", "PC-HiSD", "BB-HiSD", "p-HiSD")))
COLORS = dict(zip(METHODS, ("#4477AA", "#EE6677", "#AA3377", "#228833", "#222222")))
ADDITIVE_BUCKETS = ("t_problem_setup", "t_M_setup_update", "t_M_apply_solve",
                    "t_eigensolver_net", "t_other", "t_endpoint_certification")
TIMING_COLUMNS = ("t_total", "t_problem_setup", "t_M_setup", "t_M_refresh", "t_M_apply", "t_M_solve",
                  "t_eigensolver_init", "t_eigensolver_update", "t_endpoint_certification",
                  "t_M_setup_update", "t_M_apply_solve", "t_eigensolver",
                  "t_eigensolver_init_net", "t_eigensolver_update_net", "t_eigensolver_net",
                  "t_other", "t_pc_joint_update", "t_accounting_sum", "t_accounting_residual", "t_accounting_relative")


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


def _save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_publication_data(_clean(value)), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _clean(value):
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _save_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(_clean(v), ensure_ascii=False) if isinstance(v, (dict, list, tuple))
                             else _clean(v) for k, v in row.items()})


def _initial_state(seed):
    p = FROZEN_PROTOCOL["problem"]
    z = np.random.default_rng(seed).standard_normal(p["d"])
    return np.ones(p["d"]) + p["initial_radius"] * z / np.linalg.norm(z)


def _geomean(values):
    return float(np.exp(np.mean(np.log(values)))) if values and all(v > 0 for v in values) else None


def _draw_figure(representatives, curves, output_path, mode, group_certified=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcdefaults()
    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))
    markers = {"HiSD": "o", "AHiSD": "s", "PCHiSD": "^", "BBHiSD": "D", "pHiSD": "v"}
    for method in METHODS:
        r = representatives[(method, 1)]
        h = curves[r["run_id"]]
        label = LABELS[method]
        if r["status"] != "CONVERGED_TARGET":
            label += " [" + r["status"] + "]"
        elif group_certified is not None and not group_certified.get((method, 1), False):
            label += " [not all repeats certified]"
        line, = ax.semilogy(h[:, 0], h[:, 1], lw=2.0, label=label,
                    marker=markers[method], markersize=6, markerfacecolor="none",
                    markevery=(0.065 if method == "PCHiSD" else 0.02, 0.09))
        if method == "HiSD":
            line.set_color("#222222")
    ax.set_xlabel('Iteration $m$', fontsize=17)
    ax.set_ylabel('$\\|\\nabla E(x_m)\\|$', fontsize=17)
    ax.set_title('Modified Rosenbrock Function', fontsize=17)
    ax.tick_params(axis="both", labelsize=17)
    ax.set_ylim(1e-7, 1e4)
    max_iter = max(representatives[(method, 1)]["accepted_updates"] for method in METHODS)
    ax.set_xscale("symlog", base=10, linthresh=1, linscale=1)
    ax.set_xlim(0, PLOT_MAX_ITER if PLOT_MAX_ITER is not None else max(10, max_iter))
    from matplotlib.ticker import SymmetricalLogLocator, LogFormatterMathtext
    ax.xaxis.set_major_locator(SymmetricalLogLocator(base=10, linthresh=1, subs=(1.0,)))
    ax.xaxis.set_minor_locator(SymmetricalLogLocator(base=10, linthresh=1, subs=np.arange(2, 10)))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10, linthresh=1))
    from matplotlib.ticker import FuncFormatter

    ax.xaxis.set_minor_formatter(FuncFormatter(lambda x, pos: ""))
    ax.tick_params(axis="x", which="minor", labelsize=17)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=15, loc='upper right')
    if mode == "quick":
        ax.text(.015, .035, "SMOKE CHECK: three updates; not manuscript data", transform=ax.transAxes, fontsize=8, color="#555555")
    if group_certified is not None and not all(group_certified.values()):
        ax.text(.015, .085, "Some runs did not certify; see result tables", transform=ax.transAxes, fontsize=8, color="#AA3333")
    plt.tight_layout()
    fig.savefig(output_path, format="pdf", dpi=300, bbox_inches="tight", metadata={"Title": "Five-method Rosenbrock convergence by iteration",
                "Subject": "Seed 1: " + ("single fresh run" if mode == "quick" else "actual median-time repetitions")})
    plt.close(fig)


def _audit_timing(record):
    t = record["times"]
    required = set(TIMING_COLUMNS)
    if not required.issubset(t) or not all(np.isfinite(t[key]) for key in required):
        raise RuntimeError("Missing or non-finite timing bucket: " + record["run_id"])
    if any(t[key] < 0 for key in TIMING_COLUMNS if "residual" not in key and "relative" not in key):
        raise RuntimeError("Negative timing bucket: " + record["run_id"])
    if abs(t["t_accounting_sum"] - math.fsum(t[key] for key in ADDITIVE_BUCKETS)) > 1e-8:
        raise RuntimeError("Additive buckets do not match the event partition: " + record["run_id"])
    equations = {"t_M_setup_update": t["t_M_setup"] + t["t_M_refresh"],
                 "t_M_apply_solve": t["t_M_apply"] + t["t_M_solve"],
                 "t_eigensolver": t["t_eigensolver_init"] + t["t_eigensolver_update"],
                 "t_eigensolver_net": t["t_eigensolver_init_net"] + t["t_eigensolver_update_net"],
                 "t_accounting_residual": t["t_total"] - math.fsum(t[key] for key in ADDITIVE_BUCKETS),
                 "t_accounting_relative": t["t_accounting_residual"] / t["t_total"]}
    if any(abs(t[key] - value) > 1e-8 for key, value in equations.items()):
        raise RuntimeError("Derived timing identity failed: " + record["run_id"])
    if abs(t["t_accounting_residual"]) > max(1e-4, 1e-3 * t["t_total"]):
        raise RuntimeError("Timing accounting residual exceeds tolerance: " + record["run_id"])
    if t["t_M_refresh"] != 0 or record["counters"]["n_M_refresh"] != 0:
        raise RuntimeError("Unexpected refresh of the frozen metric: " + record["run_id"])
    if not record["timing_accounting"]["formal_timing_eligible"]:
        raise RuntimeError("Runs with extra numerical diagnostics are not eligible for formal timing.")


SENSITIVITY_METHODS = ("HiSD", "pHiSD")
SENSITIVITY_MULTIPLIERS = (0.5, 0.75, 1.0, 1.25, 1.5)
SUCCESS = "CONVERGED_TARGET"


def _required_repeats(rows, smoke=False):
    if smoke:
        return 1
    timing = FROZEN_PROTOCOL["production_timing"]
    first = [r for r in rows if r["repeat"] <= timing["initial_repeats"]]
    if len(first) < timing["initial_repeats"]:
        return timing["initial_repeats"]
    return (timing["fast_repeats_total"] if np.median([r["T_total"] for r in first])
            < timing["fast_threshold_seconds"] else timing["initial_repeats"])


def _production_schedule(seeds, smoke=False):
    repeats = 1 if smoke else FROZEN_PROTOCOL["production_timing"]["initial_repeats"]
    tasks = [(method, seed, repeat) for method in METHODS for seed in seeds
             for repeat in range(1, repeats + 1)]
    np.random.default_rng(FROZEN_PROTOCOL["production_timing"]["order_rng_seed"]).shuffle(tasks)
    return tasks


def _supplemental_schedule(records, seeds, initial, smoke=False):
    tasks = []
    for method in METHODS:
        for seed in seeds:
            rows = [r for r in records if r["experiment"] == "comparison"
                    and r["method"] == method and r["seed"] == seed]
            required = _required_repeats(rows, smoke)
            tasks.extend((method, seed, repeat) for repeat in range(len(rows) + 1, required + 1))
    rng = np.random.default_rng(FROZEN_PROTOCOL["production_timing"]["order_rng_seed"])
    dummy = list(initial)
    rng.shuffle(dummy)
    rng.shuffle(tasks)
    return tasks


def _compact_record(result, experiment, method, seed, repeat, x0, multiplier=None):
    summary = result["summary"]
    run_id = f"{experiment}_{method}_seed{seed}_repeat{repeat}"
    if multiplier is not None:
        run_id += f"_r{multiplier:g}"
    summary["run_id"] = run_id
    certification = summary["certification"]
    return _clean({
        "run_id": run_id, "experiment": experiment, "method": method,
        "seed": seed, "repeat": repeat, "r": multiplier,
        "status": summary["status"], "failure_category": summary["failure_category"],
        "failure_stage": summary["failure_stage"], "failure_details": summary["failure_details"],
        "config": summary["config"], "accepted_updates": summary["accepted_updates"],
        "n_gradient_evaluations": summary["counters"]["n_gradient_evaluations"],
        "n_hvp_vector_equivalents": summary["counters"]["n_hvp_vector_equivalents"],
        "T_total": summary["times"]["t_total"], "final_residual": summary["final_residual"],
        "distance": summary["distance"], "certification": certification,
        "initial_frame_validation": summary["initial_frame_validation"],
        "warning_records": summary["warning_records"],
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
    })


# Aggregate repeated runs per seed before combining seeds; retain unsuccessful outcomes.
def _comparison_summary(records, seeds, smoke=False):
    table, representatives, certified, medians = [], {}, {}, {}
    for method in METHODS:
        selected = []
        for seed in seeds:
            rows = [r for r in records if r["experiment"] == "comparison"
                    and r["method"] == method and r["seed"] == seed]
            required = _required_repeats(rows, smoke)
            if len(rows) != required or {r["repeat"] for r in rows} != set(range(1, required + 1)):
                raise RuntimeError(f"Incomplete comparison group: {method}/seed{seed}")
            rep = sorted(rows, key=lambda r: (r["T_total"], r["run_id"]))[len(rows)//2]
            representatives[(method, seed)] = rep
            certified[(method, seed)] = all(r["status"] == SUCCESS for r in rows)
            medians[(method, seed)] = float(np.median([r["T_total"] for r in rows]))
            selected.append(rep)
        all_rows = [r for r in records if r["experiment"] == "comparison" and r["method"] == method]
        table.append({
            "method": method,
            "outer_updates": float(np.median([r["accepted_updates"] for r in selected])),
            "gradient_evaluations": float(np.median([r["n_gradient_evaluations"] for r in selected])),
            "HVPs": float(np.median([r["n_hvp_vector_equivalents"] for r in selected])),
            "total_time_seconds": float(np.median([medians[(method, seed)] for seed in seeds])),
            "certified_seeds": sum(certified[(method, seed)] for seed in seeds),
            "total_seeds": len(seeds), "runs": len(all_rows),
            "failure_statuses": sorted({r["status"] for r in all_rows if r["status"] != SUCCESS}),
            "representative_run_ids": {str(seed): representatives[(method, seed)]["run_id"] for seed in seeds},
        })
    paired = [{"seed": seed,
               "BB_median_seconds": medians[("BBHiSD", seed)],
               "p_median_seconds": medians[("pHiSD", seed)],
               "BB_over_p": (medians[("BBHiSD", seed)] / medians[("pHiSD", seed)]
                             if certified[("BBHiSD", seed)] and certified[("pHiSD", seed)] else None)}
              for seed in seeds]
    ratio = {"paired_seed_ratios": paired,
             "geometric_mean_BB_over_p": _geomean([r["BB_over_p"] for r in paired])
             if all(r["BB_over_p"] is not None for r in paired) else None,
             "requires_all_BB_and_p_repetitions_certified": True}
    return table, representatives, certified, ratio


# Compare the prescribed step multipliers using only the development seeds.
def _sensitivity_summary(records, seeds, repeats):
    table = []
    for multiplier in SENSITIVITY_MULTIPLIERS:
        for method in SENSITIVITY_METHODS:
            rows = [r for r in records if r["experiment"] == "sensitivity"
                    and r["method"] == method and r["r"] == multiplier]
            by_seed = {seed: [r for r in rows if r["seed"] == seed] for seed in seeds}
            complete = all(len(part) == repeats and {r["repeat"] for r in part}
                           == set(range(1, repeats + 1)) for part in by_seed.values())
            successful = {seed: len(part) == repeats and all(r["status"] == SUCCESS for r in part)
                          for seed, part in by_seed.items()}
            medians = {str(seed): float(np.median([r["T_total"] for r in part])) if len(part) == repeats else None
                       for seed, part in by_seed.items()}
            table.append({"r": multiplier, "method": method,
                          "eta": multiplier * LOCKED_CONFIGS[method]["eta_x"],
                          "S": sum(successful.values()), "total_seeds": len(seeds),
                          "runs": len(rows), "complete": complete,
                          "T_dev": _geomean(list(medians.values())) if complete and all(successful.values()) else None,
                          "seed_median_seconds": medians,
                          "failure_statuses": sorted({r["status"] for r in rows if r["status"] != SUCCESS})})
    return table


def _outcome(row):
    labels = {"BUDGET_TIME": "T", "BUDGET_ITER": "I"}
    failures = ", ".join(labels.get(status, status) for status in row["failure_statuses"])
    return f"{row['S']}/{row['total_seeds']}" + (f" ({failures})" if failures else "")


def _write_report(output, meta, table3, table4, ratio):
    mode = meta["mode"]
    lines = ["# Section 7.3 manuscript results", "", f"Mode: {mode}; status: {meta['status']}.",
             "Frozen configurations; the earlier parameter search is not rerun.",
             "Times are fresh measurements on the environment recorded in protocol.json.", ""]
    if mode == "smoke":
        lines += ["SMOKE CHECK ONLY: reduced budgets/repetitions; these are not manuscript measurements.", ""]
    lines += ["## Table 3", "", "| Method | Outer updates | Gradient evaluations | HVPs | Total time (s) | Certified seeds |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in table3:
        lines.append(f"| {LABELS[row['method']]} | {row['outer_updates']:g} | {row['gradient_evaluations']:g} | "
                     f"{row['HVPs']:g} | {row['total_time_seconds']:.6g} | {row['certified_seeds']}/{row['total_seeds']} |")
    for row in table3:
        if row["failure_statuses"]:
            lines.append(f"\n{LABELS[row['method']]} outcomes include: {', '.join(row['failure_statuses'])}.\n")
    value = ratio["geometric_mean_BB_over_p"]
    lines += ["", "Paired BB-HiSD/p-HiSD geometric mean time ratio: "
              + (f"{value:.6g}." if value is not None else "unavailable (some required runs were not certified)."),
              "", "## Table 4", "", "| r | HiSD outcome | T_dev (s) | p-HiSD outcome | T_dev (s) |",
              "|---:|---|---:|---|---:|"]
    lookup = {(r["method"], r["r"]): r for r in table4}
    for multiplier in SENSITIVITY_MULTIPLIERS:
        values = [f"{multiplier:g}"]
        for method in SENSITIVITY_METHODS:
            row = lookup[(method, multiplier)]
            values += [_outcome(row), f"{row['T_dev']:.6g}" if row["T_dev"] is not None else "--"]
        lines.append("| " + " | ".join(values) + " |")
    lines += ["", "T: time budget exhausted. I: iteration budget exhausted. All other failures retain the solver status.",
              "Figure 3 uses the actual median-time repetition for seed 1; figure3_history.csv contains its plotted data.",
              "raw_runs.jsonl retains every prescribed run, including failures, endpoint certification."]
    (output / "paper_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_comparison_outputs(records, curves, seeds, output, smoke):
    table, reps, certified, ratio = _comparison_summary(records, seeds, smoke)
    _save_csv(output / "table3.csv", table)
    _save_json(output / "bb_over_p.json", ratio)
    _save_csv(output / "figure3_history.csv", [
        {"method": method, "run_id": reps[(method, 1)]["run_id"],
         "iteration": int(iteration), "residual": residual}
        for method in METHODS for iteration, residual in curves[reps[(method, 1)]["run_id"]]])
    _draw_figure(reps, curves, output / FIGURE_NAME, "quick" if smoke else "formal", certified)
    return table, ratio


@contextmanager
def _timing_lock():
    import errno
    name = hashlib.sha256(os.path.normcase(str(SCRIPT_DIR)).encode()).hexdigest()[:16]
    with (Path(tempfile.gettempdir()) / f"section73-{name}.lock").open("a+b") as stream:
        try:
            if os.name == "nt":
                import msvcrt
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                raise RuntimeError("Another Section 7.3 timing experiment is running.") from exc
            raise
        yield


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, help="result directory (matching result files are overwritten)")
    parser.add_argument("--dry-run", action="store_true", help="print protocol without running or writing results")
    parser.add_argument("--smoke", action="store_true", help="bounded pipeline check; never manuscript data")
    args = parser.parse_args()
    mode = "smoke" if args.smoke else "formal"
    production_seeds = [1] if args.smoke else list(FROZEN_PROTOCOL["seeds"]["production"])
    development_seeds = [101] if args.smoke else list(FROZEN_PROTOCOL["seeds"]["development"])
    repeats = 1 if args.smoke else 3
    max_updates = 3 if args.smoke else FROZEN_PROTOCOL["budgets"]["full_max_updates"]
    max_seconds = 5 if args.smoke else FROZEN_PROTOCOL["budgets"]["full_max_seconds"]
    initial = _production_schedule(production_seeds, args.smoke)
    sensitivity_tasks = [(method, multiplier, seed, repeat)
                         for multiplier in (1.0, .5, .75, 1.25, 1.5)
                         for seed in development_seeds for repeat in range(1, repeats + 1)
                         for method in SENSITIVITY_METHODS]
    spec = {
        "mode": mode, "numerical_protocol": FROZEN_PROTOCOL, "locked_configs": LOCKED_CONFIGS,
        "production_seeds": production_seeds, "development_seeds": development_seeds,
        "max_updates_per_solve": max_updates, "max_seconds_per_solve": max_seconds,
        "sensitivity_multipliers": SENSITIVITY_MULTIPLIERS, "sensitivity_repetitions": repeats,
        "initial_comparison_solves": len(initial), "sensitivity_solves": len(sensitivity_tasks),
        "comparison_extra_repetitions": "none" if args.smoke else "3 to 9 if first-three median time < 0.1 s",
        "tuning": "Previously locked configurations only; historical parameter tuning is not rerun.",
        "budget_scope": "Per-solve manuscript limits only; historical tuning/batch resource budgets are not applied.",
        "outcomes": "Retain all runs at the prespecified manuscript settings, regardless of outcome.",
        "table3_aggregation": "Median over seeds of actual median-time-run counts and per-seed median total times.",
        "table4_aggregation": "Geometric mean of per-seed median times only when all prescribed runs succeed.",
        "total_time_scope": "Original event-clock total: problem setup, metric/frame operations, iteration and endpoint certification; excludes output/plotting.",
        "products": [FIGURE_NAME, "figure3_history.csv", "table3.csv", "table4.csv", "bb_over_p.json",
                     "paper_summary.md", "raw_runs.jsonl", "protocol.json"],
    }
    if args.dry_run:
        print(json.dumps(_clean(spec), indent=2, allow_nan=False))
        return
    if args.output_dir:
        output = args.output_dir.expanduser().resolve()
    elif args.smoke:
        output = Path(tempfile.mkdtemp(prefix="section73-smoke-"))
    else:
        output = OUTPUT_DIR
    with _timing_lock(), threadpool_limits(limits=1):
        if args.smoke and output.exists() and any(output.iterdir()):
            parser.error("--smoke requires a new or empty --output-dir to keep test results separate")
        output.mkdir(parents=True, exist_ok=True)
        verify_limits()
        states = {seed: _initial_state(seed) for seed in production_seeds + development_seeds}
        meta = {"status": "RUNNING", "mode": mode, "started_utc": datetime.now(timezone.utc).isoformat(),
                "script": str(SCRIPT_PATH),
                "output_dir": str(output), "spec": spec, "environment": environment_info(),
                "schedule": {"comparison_initial": initial, "comparison_supplemental": None,
                             "sensitivity": sensitivity_tasks}, "completed_runs": 0}
        _save_json(output / "protocol.json", meta)
        records, curves = [], {}
        print(f"Mode={mode}; output={output}", flush=True)
        try:
            la.eigh(np.array([[2., -.2], [-.2, 1.]]))
            np.linalg.qr(np.ones((8, 2)) + np.arange(16).reshape(8, 2) * .01)
            with (output / "raw_runs.jsonl").open("w", encoding="utf-8") as raw:
                def execute(experiment, method, seed, repeat, multiplier=None):
                    config = copy.deepcopy(LOCKED_CONFIGS[method])
                    if multiplier is not None:
                        config["eta_x"] *= multiplier
                    suffix = "" if multiplier is None else f" r={multiplier:g}"
                    print(f"[{len(records)+1}] {experiment}: {LABELS[method]} seed={seed} repeat={repeat}{suffix}", flush=True)
                    verify_limits()
                    result = solve(config, states[seed].copy(), FROZEN_PROTOCOL, max_updates, max_seconds, diagnostic=False)
                    record = _compact_record(result, experiment, method, seed, repeat, states[seed], multiplier)
                    if experiment == "comparison" and seed == 1:
                        curves[record["run_id"]] = np.array([[r["updates"], r["residual"]]
                                                           for r in result["history"]], dtype=float)
                    records.append(record)
                    raw.write(json.dumps(_publication_data(record), allow_nan=False, separators=(",", ":")) + "\n")
                    raw.flush()
                    meta["completed_runs"] = len(records)
                    _save_json(output / "protocol.json", meta)
                    _audit_timing(result["summary"])
                    if record["status"] == SUCCESS and not all(record["certification"].get(key, False) for key in
                            ("index_ok", "index_resolved", "raw_tolerance_satisfied", "target_distance_satisfied")):
                        raise RuntimeError("Inconsistent success certification: " + record["run_id"])
                    print(f"  {record['status']}; updates={record['accepted_updates']}; T_total={record['T_total']:.6g}s", flush=True)
                    verify_limits()
                    if record["status"] == "IMPLEMENTATION_FAILURE":
                        raise RuntimeError("Solver implementation failure: " + str(record["failure_details"]))
                for method, seed, repeat in initial:
                    execute("comparison", method, seed, repeat)
                extra = _supplemental_schedule(records, production_seeds, initial, args.smoke)
                meta["schedule"]["comparison_supplemental"] = extra
                _save_json(output / "protocol.json", meta)
                for method, seed, repeat in extra:
                    execute("comparison", method, seed, repeat)
                table3, ratio = _write_comparison_outputs(records, curves, production_seeds, output, args.smoke)
                curves.clear()
                for method, multiplier, seed, repeat in sensitivity_tasks:
                    execute("sensitivity", method, seed, repeat, multiplier)
                    _save_csv(output / "table4.csv", _sensitivity_summary(records, development_seeds, repeats))
            table4 = _sensitivity_summary(records, development_seeds, repeats)
            if not all(row["complete"] for row in table4):
                raise RuntimeError("Incomplete sensitivity schedule")
            meta.update(status="COMPLETE" if all(r["status"] == SUCCESS for r in records) else "COMPLETE_WITH_FAILURES",
                        finished_utc=datetime.now(timezone.utc).isoformat(), final_thread_policy=verify_limits())
            _save_json(output / "protocol.json", meta)
            _write_report(output, meta, table3, table4, ratio)
            print(f"Finished ({meta['status']}): {output / 'paper_summary.md'}", flush=True)
        except BaseException as exc:
            meta.update(status="INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "ERROR",
                        exception=str(exc), traceback=traceback.format_exc(),
                        finished_utc=datetime.now(timezone.utc).isoformat())
            _save_json(output / "protocol.json", meta)
            print(f"Run incomplete; all completed records are retained in {output}", flush=True)
            raise


if __name__ == "__main__":
    main()
