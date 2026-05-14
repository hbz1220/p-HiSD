import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PLOT_MAX_ITER = 1500

SERIES = [
    ("history_standard.csv", "HiSD"),
    ("history_spectral.csv", "Spectral"),
    ("history_block-jacobi.csv", "Block Jacobi"),
    ("history_shifted-ic.csv", "Incomplete Cholesky"),
    ("history_h1-reaction.csv", "Laplacian Reaction"),
]


def read_history_csv(path: Path):
    iterations = []
    residuals = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            iterations.append(int(row["iteration"]))
            residuals.append(float(row["residual"]))
    return np.array(iterations, dtype=int), np.array(residuals, dtype=float)


def print_summary_if_exists(data_dir: Path) -> None:
    summary_path = data_dir / "summary.csv"
    if not summary_path.exists():
        return

    print(f"Found summary: {summary_path}")
    with summary_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("summary.csv is empty")
        return

    print("method          status            iter      final_res")
    for r in rows:
        method = r.get("method", "")
        status = r.get("status", "")
        iters = r.get("iterations", "")
        fres = r.get("final_residual", "")
        print(f"{method:<14}{status:<18}{iters:<10}{fres}")


def plot_all(data_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for filename, label in SERIES:
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")

        it, rs = read_history_csv(path)
        mask = it <= PLOT_MAX_ITER
        it_plot = it[mask]
        rs_plot = rs[mask]
        if it_plot.size == 0:
            continue

        ax.semilogy(it_plot, rs_plot, linewidth=2.0, label=label)

    ax.set_xlabel(r"Iteration $m$", fontsize=17)
    ax.set_ylabel(r"$\sqrt{h}\|\nabla E_h(u_m)\|_2$", fontsize=17)
    ax.set_title("Stiff Semilinear Elliptic Problem", fontsize=17)
    ax.tick_params(axis="both", labelsize=17)
    ax.set_xlim(0, PLOT_MAX_ITER)
    ax.set_ylim(1e-7, 1e3)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center right", bbox_to_anchor=(1, 0.54), fontsize=14)

    out_dir = (Path(__file__).resolve().parent /".." / ".." / ".." / "figures").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "5.1.pdf"
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot convergence curves from MATLAB-generated history CSV files."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing history_*.csv and optional summary.csv (default: script directory).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir.resolve()
    print(f"Data directory: {data_dir}")

    print_summary_if_exists(data_dir)
    out_path = plot_all(data_dir)
    print(f"Saved figure: {out_path}")


if __name__ == "__main__":
    main()
