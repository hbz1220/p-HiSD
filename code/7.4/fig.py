"""
绘制 MATLAB 实验结果
读取 experiment_results.mat 并使用与 run.py 相同的绘图风格
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

def plot_matlab_results():
    # 读取 .mat 文件
    # 读取 .mat 文件
    mat_path = '/Users/huangbingzhang/Documents/code/matlab/pre-hisd/刚性双原子链/example2/experiment_results.mat'
    print(f"正在读取数据: {mat_path}")
    data = scipy.io.loadmat(mat_path)

    # 提取历史记录
    hist_standard = data['hist_standard']
    hist_block_jacobi = data['hist_block_jacobi']
    hist_ic = data['hist_ic']
    hist_frozen_spectral = data['hist_frozen_spectral']

    # 提取梯度范数（注意：MATLAB 结构体数组的访问方式）
    # grad_norm 的 shape 是 (1, n_iters)，需要转换为 (n_iters,)
    grad_norm_std = hist_standard['grad_norm'][0, 0].ravel()
    grad_norm_bj = hist_block_jacobi['grad_norm'][0, 0].ravel()
    grad_norm_ic = hist_ic['grad_norm'][0, 0].ravel()
    grad_norm_fs = hist_frozen_spectral['grad_norm'][0, 0].ravel()

    # 提取步长参数
    dt_x_std = float(data['dt_x_standard'][0, 0])
    dt_x_bj = float(data['dt_x_block_jacobi'][0, 0])
    dt_x_ic = float(data['dt_x_ic'][0, 0])
    dt_x_fs = float(data['dt_x_frozen_spectral'][0, 0])

    # 提取系统参数
    N = int(data['N'][0, 0])
    K = float(data['K'][0, 0])

    print(f"数据读取成功!")
    print(f"  系统参数: N={N}, K={K:.0e}")
    print(f"  HiSD: {len(grad_norm_std)} 次迭代")
    print(f"  Block Jacobi: {len(grad_norm_bj)} 次迭代")
    print(f"  Incomplete Cholesky: {len(grad_norm_ic)} 次迭代")
    print(f"  Frozen Spectral: {len(grad_norm_fs)} 次迭代")

    # 组织数据
    results = {
        'HiSD': grad_norm_std,
        'Block Jacobi': grad_norm_bj,
        'Incomplete Cholesky': grad_norm_ic,
        'Frozen Spectral': grad_norm_fs,
    }

    # --- 绘图（使用与 run.py 相同的风格）---
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

    for name, gn in results.items():
        s = styles[name]
        iters = np.arange(len(gn))
        # 删除了 marker, markersize 和 markevery 参数
        ax.semilogy(iters, gn, color=s['color'],
                    linewidth=1.5, label=labels[name])

    ax.set_xlabel('Iteration $m$', fontsize=17)
    ax.set_ylabel('$\\|\\nabla E(x_m)\\|$', fontsize=17)
    ax.set_title(f'Stiff Coupled Bistable Chain ($N={N}$, $K/\\delta={K:.0e}$)', fontsize=17)
    ax.legend(fontsize=15, loc='center right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=1e-7, top=1e4)

    plt.tight_layout()

    # 保存图片到当前目录
    # 保存图片到指定目录
    output_dir = '/Users/huangbingzhang/Documents/code/python/pre-hisd/figures'
    os.makedirs(output_dir, exist_ok=True)  # 确保文件夹存在，如果不存在会自动创建
    pdf_path = os.path.join(output_dir, '4.pdf')

    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    #fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"\n图片已保存:")
    print(f"  PDF: {pdf_path}")
    #print(f"  PNG: {png_path}")


if __name__ == '__main__':
    plot_matlab_results()
