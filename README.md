# p-HiSD

Code and experimental results accompanying the paper  
**Computing Saddle Points in Stiff Problems via a Preconditioned High-index Saddle Dynamics Method**.

Bingzhang Huang, Hua Su, Lei Zhang, Jin Zhao

Preprint: [arXiv:2603.25390](https://arxiv.org/abs/2603.25390) (2026).

This repository contains the implementations and saved results for the numerical
experiments in Section 7 of the revised manuscript. The experiments compare p-HiSD
with HiSD and other saddle-search methods on stiff problems.

## Repository Structure

```text
code/
├── run_all.py    # Sequential launcher for the ten experiments
├── 7.1/          # Quadratic model
├── 7.2/          # Two-dimensional test problems
│   ├── 7.2.1/    # Butterfly function
│   └── 7.2.2/    # Modified Muller potential
├── 7.3/          # Modified Rosenbrock problem
├── 7.4/          # Stiff coupled bistable chain
├── 7.5/          # Laplacian-dominated PDE discretizations
│   ├── 7.5.1/    # 1D semilinear elliptic problem
│   ├── 7.5.2/    # 2D Lane-Emden-type elliptic equation
│   ├── 7.5.3/    # Allen-Cahn equation
│   └── 7.5.4/    # Landau–de Gennes model
│       ├── run.py
│       └── LDG_inputs.npz  # Required reference, initial-state and parameter data
├── 7.6/          # Non-convex optimal control problem
└── utils/        # Shared helper routines

outputs/          # Saved figures, tables and numerical records, organized by section
```

The input bundle
[LDG_inputs.npz](code/7.5/7.5.4/LDG_inputs.npz) must remain beside the 7.5.4
`run.py` script.

## Experiments and Paper Correspondence

Figure and table numbers below refer to the revised manuscript. PDF filenames
retain section-based names: for example, `outputs/7.3/3.pdf` is Figure 4.

| Section and code | Experiment | Manuscript figure or table | Main results |
|---|---|---|---|
| [7.1](code/7.1/) | Quadratic model | Figure 1 | [1.pdf](outputs/7.1/1.pdf) |
| [7.2.1](code/7.2/7.2.1/) | Butterfly function | Figure 2 | [2.1.pdf](outputs/7.2/7.2.1/2.1.pdf) |
| [7.2.2](code/7.2/7.2.2/) | Modified Muller potential | Figure 3; endpoint distributions in the text | [2.2.pdf](outputs/7.2/7.2.2/2.2.pdf), [table_main.csv](outputs/7.2/7.2.2/table_main.csv) |
| [7.3](code/7.3/) | Modified Rosenbrock problem | Figure 4; Tables 3 and 4 | [3.pdf](outputs/7.3/3.pdf), [table3.csv](outputs/7.3/table3.csv), [table4.csv](outputs/7.3/table4.csv) |
| [7.4](code/7.4/) | Stiff coupled bistable chain | Figure 5; Table 5 | [4.pdf](outputs/7.4/4.pdf), [timing_summary.csv](outputs/7.4/timing_summary.csv) |
| [7.5.1](code/7.5/7.5.1/) | 1D semilinear elliptic problem | Figure 6 | [5.1.pdf](outputs/7.5/7.5.1/5.1.pdf), [summary.csv](outputs/7.5/7.5.1/summary.csv) |
| [7.5.2](code/7.5/7.5.2/) | 2D Lane-Emden-type elliptic equation | Table 6(a), (b) | [fixed_grid_cost_summary.csv](outputs/7.5/7.5.2/fixed_grid_cost_summary.csv), [scalability_summary.csv](outputs/7.5/7.5.2/scalability_summary.csv) |
| [7.5.3](code/7.5/7.5.3/) | Allen-Cahn equation | Figure 7 | [5.3.pdf](outputs/7.5/7.5.3/5.3.pdf), [timing753_summary.csv](outputs/7.5/7.5.3/timing753_summary.csv) |
| [7.5.4](code/7.5/7.5.4/) | Landau–de Gennes model | Figure 8; Table 7 | [5.4.pdf](outputs/7.5/7.5.4/5.4.pdf), [LDG_results.csv](outputs/7.5/7.5.4/LDG_results.csv), [LDG_summary.json](outputs/7.5/7.5.4/LDG_summary.json) |
| [7.6](code/7.6/) | Non-convex optimal control problem | Figure 9 | [6.pdf](outputs/7.6/6.pdf), [timing76_summary.csv](outputs/7.6/timing76_summary.csv) |

## Requirements

Sections **7.4** and **7.5.1** use MATLAB (`run.m`) for computation and Python
(`fig.py`) for plotting. The other eight experiments use Python `run.py` scripts.
The supplied results record **Python 3.13.5** and **MATLAB R2025b**, with these
Python dependencies:

| Package | Version | Use |
|---|---|---|
| `numpy` | 2.4.1 | Numerical arrays and linear algebra |
| `scipy` | 1.17.0 | Sparse operators, factorizations and eigensolvers |
| `matplotlib` | 3.10.8 | PDF figures |
| `sympy` | 1.14.0 | Symbolic derivatives in Section 7.2.2 |
| `threadpoolctl` | 3.6.0 | Thread control and checks in Sections 7.3 and 7.5.4 |
| `psutil` | >= 5.9 | Resource monitoring for Section 7.5.2 on Linux and Windows |

### Installation

Install Python 3.13, then create and activate a virtual environment from the
repository root (see the [Python venv documentation](https://docs.python.org/3.13/library/venv.html)).

On macOS or Linux (bash or zsh):

```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

On Windows (Command Prompt):

```bat
py -3.13 -m venv .venv
.venv\Scripts\activate.bat
```

In the activated environment, install the packages:

```bash
python -m pip install numpy==2.4.1 scipy==1.17.0 matplotlib==3.10.8 sympy==1.14.0 threadpoolctl==3.6.0
```

For Section 7.5.2 on Linux or Windows, also install:

```bash
python -m pip install "psutil>=5.9"
```

Section 7.5.3 uses `sksparse.cholmod.cholesky` when available, otherwise SciPy
`splu`. The installation above selects SciPy LU, matching the supplied results.
Optional CHOLMOD requires [scikit-sparse and SuiteSparse](https://scikit-sparse.readthedocs.io/en/latest/overview.html#installation);
switching solvers can change rounding and timings.

## Running

Run all commands below from the repository root with the Python environment activated.

### A single experiment

For a Python experiment, run its `run.py` directly. For example:

```bash
python code/7.1/run.py
```

For Sections 7.4 and 7.5.1, run MATLAB followed by Python (`matlab` must be
available in the terminal):

```bash
matlab -batch "run('code/7.4/run.m')"
python code/7.4/fig.py
matlab -batch "run('code/7.5/7.5.1/run.m')"
python code/7.5/7.5.1/fig.py
```

With existing MATLAB data, run only `fig.py` to regenerate the figure.
Section 7.2.2 can rebuild its tables, summary and figure from a complete compatible
checkpoint set without new solver trajectories:

```bash
python code/7.2/7.2.2/run.py --plot-only
```

### The experiment launcher

The [launcher](code/run_all.py) runs experiments in paper order. Check the plan
and dependencies before starting:

```bash
python code/run_all.py --python python --dry-run
```

To run all ten experiments:

```bash
python code/run_all.py --python python
```

`--python python` selects the activated environment. Append options as needed:

| Option | Effect |
|---|---|
| `--sections 7.3 7.5.4` | Run only the selected sections. |
| `--python-only` | Run the eight Python experiments; skip MATLAB sections and their plots. |
| `--from-section 7.5.2` | Run Section 7.5.2 and all subsequent sections. |
| `--dry-run` | Check dependencies, scripts and required inputs without running experiments. |
| `--matlab PATH` | Select the MATLAB executable. |

Use `python code/run_all.py --help` for all options. MATLAB license availability
is checked when MATLAB starts. The launcher stops on a script error; numerical success
or failure is recorded in each experiment's results.

Reruns can overwrite files in `outputs/`. For a fresh full run, keep the supplied results
separately and start without that directory. Section 7.2.2 otherwise resumes
compatible `_checkpoints`; use `--output` with a new directory for a fresh scan.

## Where to Find the Results

**Additional data download:** download `timing753_raw.npz` from the
[revised-manuscript release](https://github.com/hbz1220/p-HiSD/releases/tag/revised-manuscript)
and place it in `outputs/7.5/7.5.3/`, beside `timing753_raw.json`. This
separately attached data file is not included in GitHub's automatic source downloads.

Open the PDFs linked above to view figures without running experiments. CSV,
JSON and JSONL files contain numerical summaries and run records; `.mat` and
`.npz` files store MATLAB and NumPy data. Sections 7.1 and 7.2.1 save only figures
(7.1 prints residual and index details). Section 7.5.2 produces tables, with no figure.

Keep the Section 7.2.2 `_checkpoints` directory: it contains scientific trajectory
records needed for endpoint distributions and resuming the scan.

<details>
<summary>Supporting files and endpoint distributions</summary>

The following files accompany the main results in each section's output directory:

| Section | Supporting data |
|---|---|
| 7.3 | `figure3_history.csv`: plotted histories; `raw_runs.jsonl`: individual runs; `bb_over_p.json`: paired time ratios. |
| 7.4 | `experiment_results.mat`: plotting data; `timing_runs.csv`: individual timings. |
| 7.5.1 | `history_*.csv`: residual histories; `timing751_summary.csv`: repeated timings. |
| 7.5.2 | `cost_breakdown.csv`: timing components; `memory_structure.csv`: storage measurements; [RESULTS_OVERVIEW.md](outputs/7.5/7.5.2/RESULTS_OVERVIEW.md): result summary. |
| 7.5.3 | `saddle_points.npz`: endpoint fields and full residual histories. |
| 7.5.4 | `LDG_summary.json`: certification, method agreement and detailed Table 7 costs. |

`timing*_raw.mat` files retain MATLAB repetitions. In Sections 7.5.3 and 7.6,
keep each `timing*_raw.json` with its matching `.npz`: the JSON refers to arrays
in that file. Summary MAT/JSON files hold the aggregated statistics. A successful
Section 7.5.2 run retains four CSV files and `RESULTS_OVERVIEW.md`; temporary
individual-run and memory-sampling records are validated before cleanup.

In Section 7.2.2, `table_main.csv` gives success rates and endpoint counts for
each method/step-size pair. `N_H` and `N_L` count certified high- and low-energy
saddles; `N_other` counts other certified index-1 endpoints. Their sum is
`success`; `unsuccessful` counts the remaining starts. All 512 starts remain in
the denominator of `success_percent = 100 * success / N`.

The 82 `_checkpoints/*.csv` files contain initial/final coordinates, gradient
norms, two Hessian eigenvalues, update counts and terminal categories. Keep them
with `initial_points.csv` and `protocol.json`. `progress.csv` tracks completed
and planned starts; `paper_summary.json` summarizes the reported ranges.

</details>

<details>
<summary>Fields, residual norms and memory units</summary>

| Fields | Meaning |
|---|---|
| `eta`, `tau`, `J` | State step size, frame step size and number of inner frame sweeps. |
| `outer_updates`, `actual_outer_updates`, `n_outer_updates` | Executed state updates; summary columns may carry a `median_` prefix. |
| `final_residual`, `grad_norm`, `final_state_residual` | Reported gradient residual in the experiment's norm. `endpoint_residual` and `fresh_final_residual` are recomputed at the returned state. |
| `final_frame_residual` | Residual of the unstable-direction eigenproblem in Section 7.5.4. |
| `lambda_1`, `lambda_2`, `morse_index`, `final_Morse_index`, `final_index` | Ordinary-Hessian eigenvalues and the reported Morse index; read these together with the residual and certification status. |
| `status`, `endpoint_status`, `terminal_category` | Convergence, exhausted budgets, divergence or certification outcomes. Unsuccessful attempts remain part of the experiment record. |
| `HVPs`, `HVP_vector_equivalents`, `M_apply_RHS_count`, `M_solve_RHS_count` | Hessian-vector work and right-hand sides processed by metric applications or solves. |
| `Median`/`median`, `Min`/`min`, `Max`/`max` | Repetition statistics for the named `Metric`, `metric` or `quantity`; Section 7.5.2 uses prefixes such as `median_T_total`. Times are in seconds. |

The residual is `sqrt(h) * ||g||_2` in Section 7.5.1 and `h * ||g||_2` in Sections
7.5.2 and 7.5.4. Other experiments use the Euclidean gradient norm. In Section
7.2.2, `N` denotes the number of starts; in Section 7.5.2 it is the number of
interior grid points per axis, with `n = N^2`.

Stored residual samples and executed updates can differ. In Section 7.5.3,
`iterations` counts residual samples; `state_updates` in the JSON timing records
counts executed updates. At an exhausted budget, the last saved residual can
precede the returned endpoint; the JSON records also include `endpoint_residual`.

Section 7.5.2's `median_solver_peak_rss_bytes` and `median_full_peak_rss_bytes`
summarize sampled peak resident memory during the solver and solver-plus-certification
intervals. Divide bytes by `2^20` for MiB. Sampling is requested every 20 ms and
can miss brief peaks. Missing measurements remain blank or null. `nnz_*` counts
sparse nonzeros, `F_n` is `nnz(L) + nnz(U)`, and `fill_ratio = F_n / nnz(M)`;
stored sparse-array sizes and process RSS measure different quantities.

</details>

<details>
<summary>Timing and aggregation</summary>

The total-time definition is specific to each experiment:

| Result file or section | Total-time scope |
|---|---|
| 7.3 `table3.csv` | `total_time_seconds` includes problem/metric setup, initialization, iterations and endpoint certification. |
| 7.4 `timing_summary.csv` | `T_total` includes endpoint certification. The separate `timing74_summary.csv` uses a solver-only `T_total` that excludes it. |
| 7.5.1 | `T_total` includes index checks needed to determine the solver status; the additional post-solver verification is recorded as `T_cert`. |
| 7.5.2 | `T_total` covers the solver; `T_cert` covers independent endpoint verification; `T_total_with_cert` is their sum. |
| 7.5.3 | `T_total` excludes post-solver verification. `T_cert` describes one shared verification block and is repeated in both method records. |
| 7.5.4 | `median_t_total` and `representative_t_total` include endpoint certification. |
| 7.6 | `T_total` is the sum of shared setup, method setup and iteration times, with shared setup charged once to each method; `T_cert` is separate. |

Inclusive timing components can overlap. For Table 7, the exclusive eigensolver
cost is `representative_timing.t_eigensolver_net` under each method in
`LDG_summary.json`; the CSV field `t_eigensolver` includes nested metric work.
Independent column medians need not sum to the median total.

Section 7.3 reports the median across seeds of the per-seed median time in
`table3.csv`. In `table4.csv`, `r` scales the prescribed state step, `S` counts
successful development seeds, and `T_dev` is the geometric mean of per-seed median
times when all prescribed runs succeed. Section 7.5.4 reports the median total
time and takes its cost components from an actual median-time repetition.

In Section 7.5.2, the main statistics use verified runs; `record_count`,
`verified_repeat_count` and `failed_or_censored_count` retain the outcome counts.
Different `timing_mode` groups are separate series. For an unsuccessful method,
the recorded runtime is time spent until termination.

</details>

## Reproducibility

Default runs cover the ten experiments, including the 7.2.2 paired step-size
scan, 7.3 sensitivity study and 7.5.2 mesh refinement. They exclude earlier
parameter searches and the separate larger-step HiSD instability tests discussed
in Sections 7.5.1--7.5.3 and 7.6.

For comparison, retain the prescribed inputs, seeds, parameters, stopping
tolerances, iteration/resource limits and Morse-index checks; use the versions
in [Requirements](#requirements). Timing repetitions reuse each seed's initial
data and do not constitute independent starting points.

Numerical libraries and sparse solvers can affect trajectories and stopping
iterations. Times and peak RSS depend on computing resources and system load;
time- or memory-limited runs can stop differently. Compare residuals, indices,
iteration counts and success/failure statuses using the definitions above,
and retain unsuccessful outcomes in reported results.

## Citation

If you use this code or the accompanying results, please cite:

```bibtex
@misc{huang2026computing,
  title = {{Computing Saddle Points in Stiff Problems via a Preconditioned High-index Saddle Dynamics Method}},
  author = {Bingzhang Huang and Hua Su and Lei Zhang and Jin Zhao},
  year = {2026},
  eprint = {2603.25390},
  archivePrefix = {arXiv},
  primaryClass = {math.NA},
  url = {https://arxiv.org/abs/2603.25390}
}
```

## License

This repository is released under the MIT License. See [LICENSE](LICENSE) for details.
