# p-HiSD

Code for the numerical experiments in the paper

**Preconditioned High-Index Saddle Dynamics for Computing Saddle Points**.

The folders under `code/` follow the numbering of Section 7 in the revised manuscript.

```text
code/
├── 7.1/      # Quadratic model -> Reproduces Figure 1
├── 7.2.1/    # Butterfly function -> Reproduces Figure 2
├── 7.2.2/    # Modified Mueller--Brown potential -> Reproduces Table 2
├── 7.3/      # Modified Rosenbrock problem -> Reproduces Figure 3
├── 7.4/      # Stiff coupled bistable chain -> Reproduces Figure 4 & Performance Table
├── 7.5.1/    # 1D semilinear elliptic problem -> Reproduces Figure 5
├── 7.5.2/    # 2D Lane--Emden-type elliptic equation -> Reproduces Table 3 (a & b)
├── 7.5.3/    # Allen--Cahn equation -> Reproduces Figure 6
├── 7.6/      # Non-convex optimal control problem -> Reproduces Figure 7
└── utils/    # Shared helper routines

figures/      # Generated PDF figures
```

Sections `7.5.1`--`7.5.3` correspond to the subsection
*Laplacian-dominated PDE discretizations*.

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

The code was developed and tested using the following software versions. While newer versions will likely work, using these specific versions ensures maximum reproducibility:

* **MATLAB**: R2025b (Required for experiments where the main computation script is `run.m`, particularly for incomplete Cholesky factorization).
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
cd code/7.5.1
matlab -batch "run"
python fig.py
```

Generated figures are saved to the repository-level `figures/` directory.

## Reproducibility

Random seeds are fixed where randomized initial data are used. Iteration counts
and residual histories should reproduce the reported results up to standard
machine- and library-dependent numerical variation.

## License

This repository is released under the MIT License. See `LICENSE` for details.
