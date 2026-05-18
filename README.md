# p-HiSD

Code for the numerical experiments in the paper  
**Preconditioned High-Index Saddle Dynamics for Computing Saddle Points**.

## Repository Structure

The code is organized according to Section 7 of the paper. Each folder contains the scripts and data needed to reproduce the corresponding numerical experiment.

```text
code/
├── 7.1/          # Quadratic model; reproduces Figure 1
├── 7.2/          # Two-dimensional test problems
│   ├── 7.2.1/    # Butterfly function; reproduces Figure 2
│   └── 7.2.2/    # Modified Mueller--Brown potential; reproduces Table 2
├── 7.3/          # Modified Rosenbrock problem; reproduces Figure 3
├── 7.4/          # Stiff coupled bistable chain; reproduces Figure 4 and the performance table
├── 7.5/          # Laplacian-dominated PDE discretizations
│   ├── 7.5.1/    # 1D semilinear elliptic problem; reproduces Figure 5
│   ├── 7.5.2/    # 2D Lane--Emden-type elliptic equation; reproduces Table 3(a) and Table 3(b)
│   └── 7.5.3/    # Allen--Cahn equation; reproduces Figure 6
├── 7.6/          # Non-convex optimal control problem; reproduces Figure 7
└── utils/        # Shared helper routines

figures/          # Generated PDF figures
```

Sections `7.5.1`--`7.5.3` correspond to the subsection
*Laplacian-dominated PDE discretizations*.

Some folders contain precomputed `.csv`, `.npz`, or `.mat` files. These files are
included either because they are used by the plotting scripts or because they allow
the reported figures and tables to be regenerated without rerunning all experiments.

## MATLAB and Python

Most experiments are implemented in Python. Some experiments involving the
incomplete Cholesky preconditioner use MATLAB to generate the numerical data,
because MATLAB provides convenient routines for incomplete Cholesky
factorization. The corresponding Python scripts then read the generated data and
produce the PDF figures.

Typical workflow:

```text
MATLAB run script -> CSV/data files -> Python plotting script -> PDF figure
```

## Requirements

The code was developed and tested using the following software versions. While newer versions will likely work, using similar versions is recommended for reproducibility:

* **MATLAB**: tested with MATLAB R2025b (Required for experiments where the main computation script is `run.m`, particularly for incomplete Cholesky factorization).
* **Python**: 3.13.5
  * `numpy` (v2.3.5)
  * `scipy` (v1.17.0)
  * `matplotlib` (v3.10.8)
  * `scikit-sparse` (optional, for sparse linear algebra)

Some scripts may use optional sparse linear algebra packages such as
`scikit-sparse` when available. MATLAB is required for experiments whose main
computation script is `run.m`.

## Running

Run each experiment from its own directory. For example,

```bash
cd code/7.1
python run.py
```

For experiments with MATLAB-generated data, run the MATLAB script first and then
the plotting script:

```bash
cd code/7.5/7.5.1
matlab -batch "run"
python fig.py
```

## Where to Find the Results

Depending on the experiment, numerical results are output in different formats to match the paper:

- **Figures (Most experiments)**  
  Generated PDF figures are automatically saved to the top-level `figures/` directory.

- **Table 2 (Experiment 7.2.2)**  
  The hit statistics are printed directly to the console after the script finishes.

- **Table 3 (Experiment 7.5.2)**  
  Numerical data for mesh refinement tests are saved as `.npz` and `.csv` files  
  in the `code/7.5/7.5.2/` directory.

## Reproducibility

Random seeds are fixed where randomized initial data are used. Iteration counts
and residual histories should reproduce the reported results up to standard
machine- and library-dependent numerical variation.

## License

This repository is released under the MIT License. See `LICENSE` for details.
