# Section 7.4: read experiment_results.mat produced by run.m.
# Plot the saved residual histories as Figure 5 in outputs/7.4/4.pdf.


import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

def plot_matlab_results():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.abspath(os.path.join(
        script_dir, '..', '..', 'outputs', '7.4'
    ))
    os.makedirs(output_dir, exist_ok=True)

    mat_path = os.path.join(output_dir, 'experiment_results.mat')
    print(f"Reading data: {mat_path}")
    data = scipy.io.loadmat(mat_path)

    hist_standard = data['hist_standard']
    hist_block_jacobi = data['hist_block_jacobi']
    hist_ic = data['hist_ic']
    hist_frozen_spectral = data['hist_frozen_spectral']

    grad_norm_std = hist_standard['grad_norm'][0, 0].ravel()
    grad_norm_bj = hist_block_jacobi['grad_norm'][0, 0].ravel()
    grad_norm_ic = hist_ic['grad_norm'][0, 0].ravel()
    grad_norm_fs = hist_frozen_spectral['grad_norm'][0, 0].ravel()

    dt_x_std = float(data['dt_x_standard'][0, 0])
    dt_x_bj = float(data['dt_x_block_jacobi'][0, 0])
    dt_x_ic = float(data['dt_x_ic'][0, 0])
    dt_x_fs = float(data['dt_x_frozen_spectral'][0, 0])

    N = int(data['N'][0, 0])
    K = float(data['K'][0, 0])

    print("Data loaded successfully!")
    print(f"  System parameters: N={N}, K={K:.0e}")
    print(f"  HiSD: {len(grad_norm_std)} iterations")
    print(f"  Block Jacobi: {len(grad_norm_bj)} iterations")
    print(f"  Incomplete Cholesky: {len(grad_norm_ic)} iterations")
    print(f"  Frozen Spectral: {len(grad_norm_fs)} iterations")

    results = {
        'HiSD': grad_norm_std,
        'Block Jacobi': grad_norm_bj,
        'Incomplete Cholesky': grad_norm_ic,
        'Frozen Spectral': grad_norm_fs,
    }

    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))

    styles = {
        'HiSD':   {'color': '#1f77b4', 'marker': 'o'},
        'Block Jacobi':    {'color': '#d62728', 'marker': 's'},
        'Incomplete Cholesky':              {'color': '#2ca02c', 'marker': '^'},
        'Frozen Spectral': {'color': '#ff7f0e', 'marker': 'D'},
    }

    labels = {
        'HiSD': f'HiSD ($\\eta={dt_x_std:.0e}$)',
        'Block Jacobi': f'Block Jacobi ($\\eta={dt_x_bj}$)',
        'Incomplete Cholesky': f'Incomplete Cholesky ($\\eta={dt_x_ic}$)',
        'Frozen Spectral': f'Frozen Spectral ($\\eta={dt_x_fs}$)',
    }

    # This display limit leaves the complete MATLAB histories unchanged.
    plot_limit = 300

    for name, gn in results.items():
        s = styles[name]
        gn_plot = gn[:plot_limit]
        iters = np.arange(len(gn_plot))
        ax.semilogy(iters, gn_plot, color=s['color'],
                    linewidth=1.5, label=labels[name])

    ax.set_xlabel('Iteration $m$', fontsize=17)
    ax.set_ylabel('$\\|\\nabla E(x_m)\\|$', fontsize=17)
    ax.set_title(f'Stiff Coupled Bistable Chain ($N={N}$, $K/\\delta={K:.0e}$)', fontsize=17)
    ax.legend(fontsize=15, loc='center right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=1e-7, top=1e4)
    ax.set_xlim(0, 299)

    plt.tight_layout()

    pdf_path = os.path.join(output_dir, '4.pdf')

    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print("\nFigure saved successfully:")
    print(f"  PDF: {pdf_path}")

if __name__ == '__main__':
    plot_matlab_results()