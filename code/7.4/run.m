% ============================================================================
% Stiff Diatomic Chain: Standard vs Block Jacobi vs IC vs Frozen Spectral
% 新增第四种方法：Frozen Spectral p-HiSD
% ============================================================================

clear; close all; clc;

fprintf('======================================================================\n');
fprintf('Stiff Diatomic Chain: 四种方法对比（新增 Frozen Spectral）\n');
fprintf('======================================================================\n');

%% 问题设置（共用参数）
N = 50;           % 分子对数
K = 10000.0;      % 刚性系数
n_dof = 2 * N;    % 自由度数

fprintf('系统: %d对分子, K=%.0e\n', N, K);

%% 步长设置
dt_x_standard = 5e-5;        % Standard HiSD - x 步长
dt_v_standard = 0.05;        % Standard HiSD - v 步长

dt_x_block_jacobi = 0.5;     % Block Jacobi p-HiSD - x 步长
dt_v_block_jacobi = 0.05;    % Block Jacobi p-HiSD - v 步长

dt_x_ic = 0.5;               % IC p-HiSD - x 步长
dt_v_ic = 0.05;              % IC p-HiSD - v 步长

dt_x_frozen_spectral = 0.5;  % Frozen Spectral p-HiSD - x 步长
dt_v_frozen_spectral = 0.05; % Frozen Spectral p-HiSD - v 步长

v_iter = 5;                  % 所有方法共用：x 迭代1次，v 迭代 v_iter 次

%% 创建初始猜测（四种方法共用）
rng(42);
x = ones(N, 1);
x(1:2:end) = -1.0;
y = x;

perturb = 0.1 * randn(N, 1);
x = x + perturb;
y = y + perturb + 0.01 * randn(N, 1);

X0 = zeros(n_dof, 1);
X0(1:2:end) = x;
X0(2:2:end) = y;

g0 = compute_gradient(X0, N, K);
fprintf('初始梯度范数 |g| = %.4e\n\n', norm(g0));

%% 运行 Standard HiSD
fprintf('[1] 正在求解 Standard HiSD (dt_x=%.0e, dt_v=%.2f, v_iter=%d)...\n', dt_x_standard, dt_v_standard, v_iter);
hist_standard = run_hisd(X0, N, K, dt_x_standard, dt_v_standard, v_iter, 2000, 1e-6, 'standard');

%% 运行 Block Jacobi p-HiSD
fprintf('\n[2] 正在求解 Block Jacobi p-HiSD (dt_x=%.2f, dt_v=%.2f, v_iter=%d)...\n', dt_x_block_jacobi, dt_v_block_jacobi, v_iter);
hist_block_jacobi = run_hisd(X0, N, K, dt_x_block_jacobi, dt_v_block_jacobi, v_iter, 300, 1e-6, 'block_jacobi');

%% 运行 IC p-HiSD
fprintf('\n[3] 正在求解 IC p-HiSD (dt_x=%.2f, dt_v=%.2f, v_iter=%d)...\n', dt_x_ic, dt_v_ic, v_iter);
hist_ic = run_hisd(X0, N, K, dt_x_ic, dt_v_ic, v_iter, 300, 1e-6, 'ic');

%% 运行 Frozen Spectral p-HiSD
fprintf('\n[4] 正在求解 Frozen Spectral p-HiSD (dt_x=%.2f, dt_v=%.2f, v_iter=%d)...\n', dt_x_frozen_spectral, dt_v_frozen_spectral, v_iter);
hist_frozen_spectral = run_hisd(X0, N, K, dt_x_frozen_spectral, dt_v_frozen_spectral, v_iter, 300, 1e-6, 'frozen_spectral');

%% 绘制对比图
plot_comparison_four(hist_standard, hist_block_jacobi, hist_ic, hist_frozen_spectral, ...
                     dt_x_standard, dt_x_block_jacobi, dt_x_ic, dt_x_frozen_spectral);

%% 输出性能对比
fprintf('\n================================================================================\n');
fprintf('最终性能对比\n');
fprintf('================================================================================\n');
fprintf('%-30s | %-10s | %-15s | %-12s\n', '方法', '迭代步数', '最终梯度范数', '运行时间(s)');
fprintf('--------------------------------------------------------------------------------\n');
fprintf('%-30s | %-10d | %-15.4e | %-12.3f\n', 'Standard HiSD', hist_standard.iterations, hist_standard.grad_norm(end), hist_standard.total_time);
fprintf('%-30s | %-10d | %-15.4e | %-12.3f\n', 'Block Jacobi p-HiSD', hist_block_jacobi.iterations, hist_block_jacobi.grad_norm(end), hist_block_jacobi.total_time);
fprintf('%-30s | %-10d | %-15.4e | %-12.3f\n', 'IC p-HiSD', hist_ic.iterations, hist_ic.grad_norm(end), hist_ic.total_time);
fprintf('%-30s | %-10d | %-15.4e | %-12.3f\n', 'Frozen Spectral p-HiSD', hist_frozen_spectral.iterations, hist_frozen_spectral.grad_norm(end), hist_frozen_spectral.total_time);

if hist_standard.converged
    if hist_block_jacobi.converged
        speedup_bj_iter = hist_standard.iterations / hist_block_jacobi.iterations;
        speedup_bj_time = hist_standard.total_time / hist_block_jacobi.total_time;
        fprintf('--------------------------------------------------------------------------------\n');
        fprintf('Block Jacobi 加速比: %.1f× (迭代), %.1f× (时间)\n', speedup_bj_iter, speedup_bj_time);
    end
    if hist_ic.converged
        speedup_ic_iter = hist_standard.iterations / hist_ic.iterations;
        speedup_ic_time = hist_standard.total_time / hist_ic.total_time;
        fprintf('IC 加速比: %.1f× (迭代), %.1f× (时间)\n', speedup_ic_iter, speedup_ic_time);
    end
    if hist_frozen_spectral.converged
        speedup_fs_iter = hist_standard.iterations / hist_frozen_spectral.iterations;
        speedup_fs_time = hist_standard.total_time / hist_frozen_spectral.total_time;
        fprintf('Frozen Spectral 加速比: %.1f× (迭代), %.1f× (时间)\n', speedup_fs_iter, speedup_fs_time);
    end
end
fprintf('================================================================================\n');

% 验证 Frozen Spectral 只构建了一次
fprintf('\n[验证] Frozen Spectral 预条件子构造次数: %d (应该为1)\n', hist_frozen_spectral.precond_constructions);

%% 保存数据供 Python 使用
fprintf('\n正在保存数据到 .mat 文件...\n');
save('experiment_results.mat', ...
     'hist_standard', 'hist_block_jacobi', 'hist_ic', 'hist_frozen_spectral', ...
     'dt_x_standard', 'dt_x_block_jacobi', 'dt_x_ic', 'dt_x_frozen_spectral', ...
     'dt_v_standard', 'dt_v_block_jacobi', 'dt_v_ic', 'dt_v_frozen_spectral', ...
     'v_iter', 'N', 'K');
fprintf('数据已保存到 experiment_results.mat\n');
fprintf('可以使用 Python 的 scipy.io.loadmat() 读取此文件\n');

% ============================================================================
% 主求解函数
% ============================================================================
function history = run_hisd(X0, N, K, dt_x, dt_v, v_iter, max_iter, tol, method)
    n_dof = 2 * N;
    X = X0;

    % 初始化历史记录
    history.grad_norm = zeros(1, max_iter);
    history.converged = false;
    history.iterations = 0;
    history.precond_constructions = 0;  % 记录预条件子构造次数
    iter_count = 0;

    % 构建预条件子（如果需要）
    L = [];
    M = [];
    U_spectral = [];
    Lambda_inv = [];

    if strcmp(method, 'ic')
        fprintf('构建 Shifted-IC 预条件子...\n');
        fprintf('========================================\n');

        % 调用新的 Shifted-IC 构造函数
        [L, M, prec_info] = build_shifted_ic_preconditioner(X, N, K);

        % 输出诊断信息
        fprintf('========================================\n');
        fprintf('预条件子构造完成:\n');
        fprintf('  最终 shift δ = %.4e\n', prec_info.final_delta);
        fprintf('  Shift 尝试次数 = %d\n', prec_info.shift_trials);
        fprintf('  nnz(L) = %d\n', prec_info.nnz_L);
        fprintf('  nnz(A) = %d\n', prec_info.nnz_A);
        fprintf('  Fill ratio = %.2f\n', prec_info.fill_ratio);
        fprintf('  构造耗时 = %.4f s\n', prec_info.factor_time);
        fprintf('  使用排序 = %s\n', prec_info.ordering);
        if prec_info.failed
            fprintf('  警告: IC失败，使用 fail-safe 预条件子\n');
        end
        fprintf('========================================\n\n');
        history.precond_constructions = 1;

    elseif strcmp(method, 'frozen_spectral')
        % ================================================================
        % Frozen Spectral 预条件子构造（仅一次）
        % ================================================================
        fprintf('构建 Frozen Spectral 预条件子（基于 X0 的 Hessian）...\n');

        % 构造初始 Hessian（解析公式）
        H0 = compute_hessian(X, N, K);

        % 全谱分解
        fprintf('  计算全谱分解...\n');
        [U_spectral, Lambda] = eig(full(H0));
        lambda_vals = diag(Lambda);

        % 构造 M = U (Lambda + sigma*I) U^T
        % 其中 sigma 足够大，使得所有特征值为正
        sigma = max(0, -min(lambda_vals)) + 1.0;  % shift 使得最小特征值 >= 1.0
        lambda_shifted = lambda_vals + sigma;

        % M^{-1} = U (Lambda + sigma*I)^{-1} U^T
        Lambda_inv = diag(1 ./ lambda_shifted);

        fprintf('  谱分解完成: %d 个特征值\n', n_dof);
        fprintf('  最小特征值: %.6f\n', min(lambda_vals));
        fprintf('  最大特征值: %.6f\n', max(lambda_vals));
        fprintf('  Shift σ: %.2f\n', sigma);
        fprintf('  Shifted 最小特征值: %.6f\n', min(lambda_shifted));
        fprintf('  Shifted 最大特征值: %.6f\n', max(lambda_shifted));

        history.precond_constructions = 1;  % 只构造一次
    end

    % 初始化不稳定方向
    fprintf('初始化不稳定方向...\n');
    V = initialize_frame(X, N, K, method, L, M, U_spectral, Lambda_inv);

    % 开始计时
    t_start = tic;

    % 迭代求解
    for iter = 1:max_iter
        % 计算梯度
        grad = compute_gradient(X, N, K);
        gnorm = norm(grad);

        iter_count = iter_count + 1;
        history.grad_norm(iter_count) = gnorm;

        if gnorm < tol
            history.converged = true;
            history.iterations = iter;
            fprintf('收敛! 迭代次数: %d\n', iter);
            break;
        end

        if ~isfinite(gnorm) || gnorm > 1e8
            history.converged = false;
            history.iterations = iter;
            fprintf('发散! 停止于迭代 %d\n', iter);
            break;
        end

        % 执行HiSD更新
        if strcmp(method, 'standard')
            % Standard HiSD
            v = V{1};
            c = v' * grad;
            grad_mod = grad - 2 * c * v;
            X = X - dt_x * grad_mod;
            V = update_frame_standard(X, V, N, K, dt_v, v_iter);

        elseif strcmp(method, 'block_jacobi')
            % Block Jacobi p-HiSD
            v = V{1};
            c = v' * grad;  % 修正: 使用标准内积 v^T g，而不是 v^T M^{-1} g
            direction = apply_block_jacobi_inverse(X, grad, N, K);  % M^{-1} grad
            X = X - dt_x * (direction - 2 * c * v);  % dir_x = -M^{-1}g + 2(v^T g)v
            V = update_frame_block_jacobi(X, V, N, K, dt_v, v_iter);

        elseif strcmp(method, 'ic')
            % IC p-HiSD (动态更新预条件子)
            % 每次迭代重新构建IC预条件子
            [L, M, ~] = build_shifted_ic_preconditioner(X, N, K);
            history.precond_constructions = history.precond_constructions + 1;

            v = V{1};
            t_g = L' \ (L \ grad);
            c = v' * grad;
            X = X - dt_x * (t_g - 2 * c * v);
            V = update_frame_ic(X, V, N, K, dt_v, v_iter, L, M);

        else  % frozen_spectral
            % ================================================================
            % Frozen Spectral p-HiSD（简化版：预条件梯度 + 反射修正）
            % ================================================================
            v = V{1};

            % 步骤1: 计算预条件梯度 d = M^{-1} g
            Ut_grad = U_spectral' * grad;
            Lambda_inv_Ut_grad = Lambda_inv * Ut_grad;
            d = U_spectral * Lambda_inv_Ut_grad;

            % 步骤2: 计算反射系数 c = v^T g（使用标准欧氏内积）
            c = v' * grad;

            % 步骤3: 组合更新量 direction = d - 2c*v
            direction = d - 2 * c * v;

            % 步骤4: 更新位置
            X = X - dt_x * direction;

            % 更新框架
            V = update_frame_frozen_spectral(X, V, N, K, dt_v, v_iter, U_spectral, Lambda_inv);
        end

        % 定期重新计算框架
        if mod(iter, 50) == 0 && iter > 0
            V = initialize_frame(X, N, K, method, L, M, U_spectral, Lambda_inv);
        end
    end

    if ~history.converged && iter == max_iter
        history.iterations = max_iter;
        fprintf('达到最大迭代次数\n');
    end

    history.total_time = toc(t_start);
    history.grad_norm = history.grad_norm(1:iter_count);
    history.x_final = X;  % 保存收敛点
end

% ============================================================================
% 初始化不稳定方向
% ============================================================================
function V = initialize_frame(X, N, K, method, L, M, U_spectral, Lambda_inv)
    n_dof = 2 * N;

    % 构造Hessian（解析公式）
    H = full(compute_hessian(X, N, K));

    if strcmp(method, 'standard')
        [vecs, vals] = eig(H);
        [~, idx] = sort(diag(vals));
        v = vecs(:, idx(1));
        v = v / norm(v);

    elseif strcmp(method, 'block_jacobi')
        M_bj = get_block_jacobi_matrix(X, N, K);
        [vecs, vals] = eig(H, M_bj);
        [~, idx] = sort(diag(vals));
        v = vecs(:, idx(1));
        % 正确的 M-归一化: v^T M v = 1
        % 注意: M_bj 实际上是 M^{-1}，所以需要先求逆
        % 但这里我们直接用 Block Jacobi 的结构计算 Mv
        Mv = apply_block_jacobi_to_vector(X, v, N, K);  % M * v
        norm_M = sqrt(v' * Mv);
        v = v / norm_M;

    elseif strcmp(method, 'ic')
        [v, ~] = eigs(H, M, 1, 'smallestreal');
        Mv = M * v;
        norm_M = sqrt(v' * Mv);
        v = v / norm_M;

    else  % frozen_spectral
        % 对于 Frozen Spectral，直接取 H(X0) 的最小特征向量
        % 因为 M 与 H(X0) 基本对齐
        [vecs, vals] = eig(H);
        [~, idx] = sort(diag(vals));
        v = vecs(:, idx(1));

        % M-归一化: v^T M v = 1
        % 1. From Lambda_inv restore Lambda
        lambda_shifted = 1.0 ./ diag(Lambda_inv);

        % 2. Project v to eigenspace
        u_proj = U_spectral' * v;

        % 3. Compute v^T M v = u^T Lambda u
        M_norm_sq = sum( (u_proj.^2) .* lambda_shifted );
        norm_M = sqrt(M_norm_sq);

        % 4. Normalize
        if norm_M > 1e-10
            v = v / norm_M;
        end
    end

    V = {v};
end

% ============================================================================
% 更新框架 - Frozen Spectral
% ============================================================================
function V_new = update_frame_frozen_spectral(X, V, N, K, dt_v, v_iter, U_spectral, Lambda_inv)
    n_dof = 2 * N;

    % 构造当前 Hessian（解析公式）
    H = full(compute_hessian(X, N, K));

    v = V{1};

    % v 内迭代 v_iter 次
    for iter_v = 1:v_iter
        % 计算 G(x) v
        Gv = H * v;

        % 应用 Frozen Spectral 预条件子: M^{-1} G v
        Ut_Gv = U_spectral' * Gv;
        Lambda_inv_Ut_Gv = Lambda_inv * Ut_Gv;
        Minv_Gv = U_spectral * Lambda_inv_Ut_Gv;

        % M-投影: 修正: 使用 v^T Hv，而不是 v^T M^{-1}Hv
        coeff = v' * Gv;  % v^T Hv (Rayleigh Quotient)
        projected = Minv_Gv - coeff * v;

        % 更新: dir_v = -M^{-1}Hv + (v^T Hv)v
        v = v - dt_v * projected;

        % M-归一化
        % 1. From Lambda_inv restore Lambda
        lambda_shifted = 1.0 ./ diag(Lambda_inv);

        % 2. Project v to eigenspace
        u_proj = U_spectral' * v;

        % 3. Compute v^T M v = u^T Lambda u
        M_norm_sq = sum( (u_proj.^2) .* lambda_shifted );
        norm_M = sqrt(M_norm_sq);

        % 4. Normalize
        if norm_M > 1e-10
            v = v / norm_M;
        end
    end

    V_new = {v};
end

% ============================================================================
% 更新框架 - Standard
% ============================================================================
function V_new = update_frame_standard(X, V, N, K, dt_v, v_iter)
    n_dof = 2 * N;
    eps = 1e-6;

    g0 = compute_gradient(X, N, K);
    H = zeros(n_dof, n_dof);
    for i = 1:n_dof
        ei = zeros(n_dof, 1);
        ei(i) = eps;
        H(:, i) = (compute_gradient(X + ei, N, K) - g0) / eps;
    end
    H = 0.5 * (H + H');

    v = V{1};

    % v 内迭代 v_iter 次
    for iter_v = 1:v_iter
        Gv = H * v;
        coeff = v' * Gv;
        projected = Gv - coeff * v;
        v = v - dt_v * projected;
        v = v / norm(v);
    end

    V_new = {v};
end

% ============================================================================
% 更新框架 - Block Jacobi
% ============================================================================
function V_new = update_frame_block_jacobi(X, V, N, K, dt_v, v_iter)
    n_dof = 2 * N;
    eps = 1e-6;

    g0 = compute_gradient(X, N, K);
    H = zeros(n_dof, n_dof);
    for i = 1:n_dof
        ei = zeros(n_dof, 1);
        ei(i) = eps;
        H(:, i) = (compute_gradient(X + ei, N, K) - g0) / eps;
    end
    H = 0.5 * (H + H');

    v = V{1};

    % v 内迭代 v_iter 次
    for iter_v = 1:v_iter
        Gv = H * v;
        Minv_Gv = apply_block_jacobi_inverse(X, Gv, N, K);  % M^{-1}Hv
        coeff = v' * Gv;  % 修正: 使用 v^T Hv，而不是 v^T M^{-1}Hv
        projected = Minv_Gv - coeff * v;
        v = v - dt_v * projected;  % dir_v = -M^{-1}Hv + (v^T Hv)v
        norm_M = sqrt(m_inner_product_bj(X, v, v, N, K));
        if norm_M > 1e-10
            v = v / norm_M;
        end
    end

    V_new = {v};
end

% ============================================================================
% 更新框架 - IC
% ============================================================================
function V_new = update_frame_ic(X, V, N, K, dt_v, v_iter, L, M)
    n_dof = 2 * N;
    eps = 1e-6;

    g0 = compute_gradient(X, N, K);
    H = zeros(n_dof, n_dof);
    for i = 1:n_dof
        ei = zeros(n_dof, 1);
        ei(i) = eps;
        H(:, i) = (compute_gradient(X + ei, N, K) - g0) / eps;
    end
    H = 0.5 * (H + H');

    v = V{1};

    % v 内迭代 v_iter 次
    for iter_v = 1:v_iter
        Gv = H * v;
        Minv_Gv = L' \ (L \ Gv);
        % 注意: 系数必须是 Rayleigh Quotient v^T H v，而不是 v^T M^{-1} H v
        coeff = v' * Gv;  % 正确: v^T H v (Rayleigh Quotient)
        projected = Minv_Gv - coeff * v;
        v = v - dt_v * projected;
        Mv = M * v;
        norm_M = sqrt(v' * Mv);
        v = v / norm_M;
    end

    V_new = {v};
end

% ============================================================================
% 计算解析 Hessian
% ============================================================================
function H = compute_hessian(X, N, K)
    n_dof = 2 * N;
    x = X(1:2:end);
    y = X(2:2:end);

    % 构造稀疏 Hessian 矩阵
    I_idx = [];
    J_idx = [];
    V_vals = [];

    for i = 1:N
        idx_x = 2*i - 1;
        idx_y = 2*i;

        % 对角元素: ∂²E/∂x_i²
        H_xx = K + 12*x(i)^2 - 4;
        if i > 1
            H_xx = H_xx + 1;  % 来自 (y_{i-1} - x_i)² 项
        end

        % 对角元素: ∂²E/∂y_i²
        H_yy = K + 12*y(i)^2 - 4;
        if i < N
            H_yy = H_yy + 1;  % 来自 (y_i - x_{i+1})² 项
        end

        % 交叉项: ∂²E/∂x_i∂y_i
        H_xy = -K;

        % 添加对角块
        I_idx = [I_idx; idx_x; idx_y; idx_x; idx_y];
        J_idx = [J_idx; idx_x; idx_y; idx_y; idx_x];
        V_vals = [V_vals; H_xx; H_yy; H_xy; H_xy];

        % 耦合项: ∂²E/∂y_i∂x_{i+1}
        if i < N
            idx_x_next = 2*(i+1) - 1;
            I_idx = [I_idx; idx_y; idx_x_next];
            J_idx = [J_idx; idx_x_next; idx_y];
            V_vals = [V_vals; -1; -1];
        end
    end

    H = sparse(I_idx, J_idx, V_vals, n_dof, n_dof);
end

% ============================================================================
% 计算梯度
% ============================================================================
function grad = compute_gradient(X, N, K)
    x = X(1:2:end);
    y = X(2:2:end);

    grad_x = K * (x - y) + 4 * x .* (x.^2 - 1);
    grad_y = -K * (x - y) + 4 * y .* (y.^2 - 1);

    grad_y(1:end-1) = grad_y(1:end-1) + (y(1:end-1) - x(2:end));
    grad_x(2:end) = grad_x(2:end) - (y(1:end-1) - x(2:end));

    grad = zeros(2*N, 1);
    grad(1:2:end) = grad_x;
    grad(2:2:end) = grad_y;
end

% ============================================================================
% Block Jacobi 相关函数
% ============================================================================
function [H_xx, H_yy, H_xy] = get_local_blocks(X, N, K)
    x = X(1:2:end);
    y = X(2:2:end);

    damping = 1.0;
    curvature_x = 12 * x.^2 - 4;
    curvature_y = 12 * y.^2 - 4;

    H_xx = K + abs(curvature_x) + damping;
    H_yy = K + abs(curvature_y) + damping;
    H_xy = -K * ones(N, 1);
end

function result = apply_block_jacobi_inverse(X, vector, N, K)
    vec_x = vector(1:2:end);
    vec_y = vector(2:2:end);

    [H_xx, H_yy, H_xy] = get_local_blocks(X, N, K);

    det = H_xx .* H_yy - H_xy.^2;
    det = max(det, 1e-8);
    inv_det = 1.0 ./ det;

    res_x = inv_det .* (H_yy .* vec_x - H_xy .* vec_y);
    res_y = inv_det .* (-H_xy .* vec_x + H_xx .* vec_y);

    result = zeros(2*N, 1);
    result(1:2:end) = res_x;
    result(2:2:end) = res_y;
end

% ============================================================================
% 应用Block Jacobi预条件子: M @ vector (不是逆)
% ============================================================================
function result = apply_block_jacobi_to_vector(X, vector, N, K)
    vec_x = vector(1:2:end);
    vec_y = vector(2:2:end);

    [H_xx, H_yy, H_xy] = get_local_blocks(X, N, K);

    % 直接应用 2×2 块矩阵
    res_x = H_xx .* vec_x + H_xy .* vec_y;
    res_y = H_xy .* vec_x + H_yy .* vec_y;

    result = zeros(2*N, 1);
    result(1:2:end) = res_x;
    result(2:2:end) = res_y;
end

function M = get_block_jacobi_matrix(X, N, K)
    n_dof = 2 * N;
    M = zeros(n_dof, n_dof);

    for i = 1:n_dof
        ei = zeros(n_dof, 1);
        ei(i) = 1.0;
        M(:, i) = apply_block_jacobi_inverse(X, ei, N, K);
    end
end

function result = m_inner_product_bj(X, u, v, N, K)
    Mv = apply_block_jacobi_to_vector(X, v, N, K);  % M * v (不是 M^{-1} * v)
    result = u' * Mv;
end

% ============================================================================
% IC 预条件子 - Shifted IC with Robustness
% ============================================================================
function [L, M, info] = build_shifted_ic_preconditioner(X, N, K)
    n_dof = 2 * N;

    % 初始化诊断信息
    info = struct();
    info.failed = false;
    info.shift_trials = 0;
    info.final_delta = 0;
    info.ordering = 'none';

    tic;  % 开始计时

    % ========================================================================
    % Step 0: 构造基础矩阵 A(x)（稀疏 SPD surrogate）
    % ========================================================================
    A = build_base_matrix(X, N, K);
    info.nnz_A = nnz(A);

    % ========================================================================
    % Step 1: 自适应 Shift 策略
    % ========================================================================
    % 参数设置
    delta_init = 1e-6;           % 初始 shift
    delta_max = 1e3;             % 最大 shift（防止过度正则化）
    max_shift_trials = 10;       % 最大尝试次数
    shift_growth_factor = 10;    % shift 增长因子

    delta = delta_init;

    % IC 参数
    ic_opts.type = 'ict';
    ic_opts.droptol = 1e-5;      % drop tolerance（控制 fill-in）

    % 排序策略（AMD: Approximate Minimum Degree）
    try
        perm = amd(A);
        info.ordering = 'AMD';
    catch
        perm = 1:n_dof;
        info.ordering = 'none';
    end

    % 应用排序
    A_perm = A(perm, perm);

    L = [];
    ic_success = false;

    % 自适应 shift 循环
    for trial = 1:max_shift_trials
        info.shift_trials = trial;

        % 构造 A_delta = A + delta*I
        A_delta = A_perm + delta * speye(n_dof);

        % ====================================================================
        % Step 2: Incomplete Cholesky 分解
        % ====================================================================
        try
            L_perm = ichol(A_delta, ic_opts);

            % 检查分解质量（防止数值问题）
            if any(~isfinite(L_perm(:))) || any(diag(L_perm) <= 0)
                error('IC produced invalid entries');
            end

            % 成功！
            ic_success = true;
            info.final_delta = delta;

            % 恢复原始排序
            L = L_perm(invperm(perm), :);

            break;

        catch ME
            % IC 失败，增大 shift
            if trial < max_shift_trials
                fprintf('  IC 失败 (trial %d, δ=%.2e): %s\n', trial, delta, ME.message);
                fprintf('  增大 shift...\n');
                delta = delta * shift_growth_factor;

                % 检查是否超过上限
                if delta > delta_max
                    fprintf('  警告: shift 超过上限 (%.2e)，停止尝试\n', delta_max);
                    break;
                end
            else
                fprintf('  IC 失败: 达到最大尝试次数\n');
            end
        end
    end

    % ========================================================================
    % Step 3: Fail-safe 机制
    % ========================================================================
    if ~ic_success
        fprintf('  使用 fail-safe: Block-Jacobi 预条件子\n');
        info.failed = true;

        % 构造简单的 Block-Jacobi 预条件子
        [L, M] = build_failsafe_preconditioner(X, N, K);
        info.nnz_L = nnz(L);
        info.fill_ratio = info.nnz_L / info.nnz_A;
        info.factor_time = toc;
        return;
    end

    % ========================================================================
    % Step 2: 构造最终预条件子矩阵 M（用于 M-归一化等）
    % ========================================================================
    % 注意: M = L*L' (对应 H_δ = H + δI 的 IC 分解)
    M = L * L';

    % ========================================================================
    % 诊断信息
    % ========================================================================
    info.nnz_L = nnz(L);
    info.fill_ratio = info.nnz_L / info.nnz_A;
    info.factor_time = toc;

    % 检查 over-regularization（δ 太大的警告）
    diag_A = abs(diag(A));
    relative_shift = info.final_delta / max(diag_A);
    if relative_shift > 0.1
        fprintf('  警告: 相对 shift 较大 (%.2f%%), 可能过度正则化\n', relative_shift * 100);
        fprintf('  建议: 增大 fill-in 预算或改进排序\n');
    end
end

% ============================================================================
% 构造基础矩阵 A(x)（稀疏 SPD surrogate）
% ============================================================================
function A = build_base_matrix(X, N, K)
    n_dof = 2 * N;
    x = X(1:2:end);
    y = X(2:2:end);

    % 计算局部曲率（取绝对值保证 SPD）
    damping = 1.0;
    curvature_x = 12 * x.^2 - 4;
    curvature_y = 12 * y.^2 - 4;

    H_xx = K + abs(curvature_x) + damping;
    H_yy = K + abs(curvature_y) + damping;
    H_xy = -K * ones(N, 1);

    % 确保 2×2 块是 SPD（调整 H_xy）
    for i = 1:N
        % 2×2 块的特征值必须为正
        % λ = (H_xx + H_yy ± sqrt((H_xx-H_yy)^2 + 4*H_xy^2)) / 2
        % 要求最小特征值 > 0
        discriminant = (H_xx(i) - H_yy(i))^2 + 4 * H_xy(i)^2;
        lambda_min = (H_xx(i) + H_yy(i) - sqrt(discriminant)) / 2;

        if lambda_min <= 0
            % 调整 H_xy 使块 SPD
            H_xy(i) = 0.9 * sqrt(H_xx(i) * H_yy(i));
        end
    end

    % 构建稀疏矩阵（2×2块对角结构 + 分子间耦合）
    I_idx = [];
    J_idx = [];
    V_vals = [];

    for i = 1:N
        % 块的全局索引
        idx_x = 2*i - 1;
        idx_y = 2*i;

        % (x_i, x_i)
        I_idx = [I_idx; idx_x];
        J_idx = [J_idx; idx_x];
        V_vals = [V_vals; H_xx(i)];

        % (x_i, y_i)
        I_idx = [I_idx; idx_x];
        J_idx = [J_idx; idx_y];
        V_vals = [V_vals; H_xy(i)];

        % (y_i, x_i)
        I_idx = [I_idx; idx_y];
        J_idx = [J_idx; idx_x];
        V_vals = [V_vals; H_xy(i)];

        % (y_i, y_i)
        I_idx = [I_idx; idx_y];
        J_idx = [J_idx; idx_y];
        V_vals = [V_vals; H_yy(i)];
    end

    % 添加分子间耦合（稀疏）
    coupling_strength = 0.5;  % 弱耦合
    for i = 1:N-1
        idx_y_i = 2*i;
        idx_x_ip1 = 2*(i+1) - 1;

        % (y_i, x_{i+1})
        I_idx = [I_idx; idx_y_i];
        J_idx = [J_idx; idx_x_ip1];
        V_vals = [V_vals; -coupling_strength];

        % (x_{i+1}, y_i)
        I_idx = [I_idx; idx_x_ip1];
        J_idx = [J_idx; idx_y_i];
        V_vals = [V_vals; -coupling_strength];
    end

    A = sparse(I_idx, J_idx, V_vals, n_dof, n_dof);

    % 确保对称性
    A = 0.5 * (A + A');
end

% ============================================================================
% Fail-safe 预条件子（Block-Jacobi）
% ============================================================================
function [L, M] = build_failsafe_preconditioner(X, N, K)
    n_dof = 2 * N;
    x = X(1:2:end);
    y = X(2:2:end);

    % 简单的 Block-Jacobi: 每个 2×2 块独立
    damping = 10.0;  % 更大的 damping 保证稳定性
    curvature_x = 12 * x.^2 - 4;
    curvature_y = 12 * y.^2 - 4;

    H_xx = K + abs(curvature_x) + damping;
    H_yy = K + abs(curvature_y) + damping;

    % 对角预条件子（忽略耦合）
    diag_vals = zeros(n_dof, 1);
    diag_vals(1:2:end) = H_xx;
    diag_vals(2:2:end) = H_yy;

    M = spdiags(diag_vals, 0, n_dof, n_dof);
    L = spdiags(sqrt(diag_vals), 0, n_dof, n_dof);
end

% ============================================================================
% 辅助函数: 逆排列
% ============================================================================
function iperm = invperm(perm)
    n = length(perm);
    iperm = zeros(1, n);
    iperm(perm) = 1:n;
end

% ============================================================================
% 旧的 IC 预条件子函数（保留以防其他地方调用）
% ============================================================================
function M = build_ic_preconditioner_matrix(X, N, K)
    n_dof = 2 * N;
    x = X(1:2:end);
    y = X(2:2:end);

    damping = 1.0;
    curvature_x = 12 * x.^2 - 4;
    curvature_y = 12 * y.^2 - 4;

    H_xx = K + abs(curvature_x) + damping;
    H_yy = K + abs(curvature_y) + damping;
    H_xy = -K * ones(N, 1);

    I_idx = [];
    J_idx = [];
    V_vals = [];

    for i = 1:N
        idx_x = 2*i - 1;
        idx_y = 2*i;

        I_idx = [I_idx; idx_x; idx_x; idx_y; idx_y];
        J_idx = [J_idx; idx_x; idx_y; idx_x; idx_y];
        V_vals = [V_vals; H_xx(i); H_xy(i); H_xy(i); H_yy(i)];
    end

    M = sparse(I_idx, J_idx, V_vals, n_dof, n_dof);
end

% ============================================================================
% 绘图函数 - 四条曲线
% ============================================================================
function plot_comparison_four(h1, h2, h3, h4, dt1, dt2, dt3, dt4)
    % 设置图形大小：与 Python 的 figsize=(10, 6) 一致
    figure('Position', [100, 100, 1000, 600]);

    iter1 = h1.iterations;
    iter2 = h2.iterations;
    iter3 = h3.iterations;
    iter4 = h4.iterations;

    valid1 = isfinite(h1.grad_norm);
    valid2 = isfinite(h2.grad_norm);
    valid3 = isfinite(h3.grad_norm);
    valid4 = isfinite(h4.grad_norm);

    if any(valid1)
        steps1 = find(valid1);
        norms1 = h1.grad_norm(valid1);
        semilogy(steps1, norms1, 'b-', 'LineWidth', 2, ...
            'DisplayName', sprintf('Standard HiSD (dt=%.0e): %d iters', dt1, iter1));
    end

    hold on;

    if any(valid2)
        steps2 = find(valid2);
        norms2 = h2.grad_norm(valid2);
        semilogy(steps2, norms2, 'g-', 'LineWidth', 2, ...
            'DisplayName', sprintf('Block Jacobi p-HiSD (dt=%.2f): %d iters', dt2, iter2));
    end

    if any(valid3)
        steps3 = find(valid3);
        norms3 = h3.grad_norm(valid3);
        semilogy(steps3, norms3, 'r-', 'LineWidth', 2, ...
            'DisplayName', sprintf('IC p-HiSD (dt=%.2f): %d iters', dt3, iter3));
    end

    if any(valid4)
        steps4 = find(valid4);
        norms4 = h4.grad_norm(valid4);
        semilogy(steps4, norms4, 'm-', 'LineWidth', 2, ...
            'DisplayName', sprintf('Frozen Spectral p-HiSD (dt=%.2f): %d iters', dt4, iter4));
    end

    xlabel('Iteration', 'FontSize', 17);
    ylabel('Gradient Norm ||\nabla E||', 'FontSize', 17);
    title('Stiff Diatomic Chain: Four Methods Comparison', 'FontSize', 19, 'FontWeight', 'normal');

    % 设置网格样式（与 Python 一致：alpha=0.2）
    grid on;
    set(gca, 'GridAlpha', 0.2);
    set(gca, 'MinorGridAlpha', 0.2);

    % 控制 y 轴刻度数量（减少刻度，使图更清晰）
    ax = gca;
    ax.YAxis.MinorTick = 'off';  % 关闭次要刻度

    legend('Location', 'best', 'FontSize', 16);

    set(gca, 'LooseInset', get(gca, 'TightInset'));

    exportgraphics(gcf, '3.pdf', 'ContentType', 'vector', 'BackgroundColor', 'none');
    fprintf('\n图片已保存: 3.pdf\n');
end
