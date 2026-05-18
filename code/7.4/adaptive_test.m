% ============================================================================
% Stiff Diatomic Chain: Adaptive Preconditioner Selection (Algorithm 6.1)
% ============================================================================

clear; close all; clc;

fprintf('======================================================================\n');
fprintf('Stiff Diatomic Chain: Adaptive Preconditioner Selection\n');
fprintf('======================================================================\n');

%% ========================================================================
%  Manual parameter settings (modify here to test different N values)
% =========================================================================
N = 50;           % Number of molecular pairs (options: 50, 200, 501, etc.)
                  % N=50  -> n=100  -> Frozen Spectral
                  % N=200 -> n=400  -> Block Jacobi
                  % N=501 -> n=1002 -> IC

K = 10000.0;      % Stiffness coefficient

%% ========================================================================
%  Run tests
% =========================================================================
n_dof = 2 * N;

fprintf('\n======================================================================\n');
fprintf('Test configuration: N=%d, n=%d, K=%.0e\n', N, n_dof, K);
fprintf('======================================================================\n');

% Algorithm 6.1: Select preconditioner
selected_method = select_preconditioner(n_dof, K);
fprintf('Algorithm 6.1 selection: %s\n', selected_method);

% Create initial guess
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
fprintf('Initial gradient norm |g| = %.4e\n\n', norm(g0));

% Step-size settings
dt_standard = 5e-5;
dt_precond = 0.5;
dt_v = 0.05;
v_iter = 5;

% Run Standard HiSD
fprintf('[1] Solving Standard HiSD (dt=%.0e)...\n', dt_standard);
hist_standard = run_hisd(X0, N, K, dt_standard, dt_v, v_iter, 2000, 1e-6, 'standard');

% Run the selected preconditioned method
fprintf('\n[2] Solving %s p-HiSD (dt=%.2f)...\n', selected_method, dt_precond);
hist_precond = run_hisd(X0, N, K, dt_precond, dt_v, v_iter, 2000, 1e-6, selected_method);

% Plot comparison
plot_adaptive_comparison(hist_standard, hist_precond, dt_standard, dt_precond, ...
                        selected_method, N);

% Print performance comparison
fprintf('\n================================================================================\n');
fprintf('Performance comparison (N=%d)\n', N);
fprintf('================================================================================\n');
fprintf('%-30s | %-10s | %-15s | %-12s\n', 'Method', 'Iterations', 'Final grad norm', 'Runtime(s)');
fprintf('--------------------------------------------------------------------------------\n');
fprintf('%-30s | %-10d | %-15.4e | %-12.3f\n', 'Standard HiSD', hist_standard.iterations, hist_standard.grad_norm(end), hist_standard.total_time);
fprintf('%-30s | %-10d | %-15.4e | %-12.3f\n', [selected_method ' p-HiSD'], hist_precond.iterations, hist_precond.grad_norm(end), hist_precond.total_time);

if hist_standard.converged && hist_precond.converged
    speedup_iter = hist_standard.iterations / hist_precond.iterations;
    speedup_time = hist_standard.total_time / hist_precond.total_time;
    fprintf('--------------------------------------------------------------------------------\n');
    fprintf('Speedup: %.1f× (iterations), %.1f× (time)\n', speedup_iter, speedup_time);
end
fprintf('================================================================================\n');

% ============================================================================
% Algorithm 6.1: Adaptive preconditioner selection
% ============================================================================
function method = select_preconditioner(n, K)
    % Input:
    %   n: number of degrees of freedom
    %   K: stiffness coefficient
    % Output:
    %   method: selected preconditioner method

    if n < 200
        method = 'frozen_spectral';
    elseif n >= 200 && n <= 1000
        method = 'block_jacobi';
    else  % n > 1000
        % Compute rho (simplified here as a fixed value; in practice it should be problem-dependent)
        rho = 0.005;  % Assume rho < 0.01
        if rho < 0.01
            method = 'ic';
        else
            method = 'jacobi';  % Fallback option
        end
    end
end

% ============================================================================
% Main solver
% ============================================================================
function history = run_hisd(X0, N, K, dt_x, dt_v, v_iter, max_iter, tol, method)
    n_dof = 2 * N;
    X = X0;

    % Initialize history
    history.grad_norm = zeros(1, max_iter);
    history.converged = false;
    history.iterations = 0;
    iter_count = 0;

    % Build preconditioner if needed
    L = [];
    M = [];
    U_spectral = [];
    Lambda_inv = [];
    M_jacobi_inv = [];

    if strcmp(method, 'ic')
        fprintf('Building IC preconditioner...\n');
        M = build_ic_preconditioner_matrix(X, N, K);
        opts.type = 'ict';
        opts.droptol = 1e-5;
        L = ichol(M, opts);
        fprintf('IC preconditioning: nnz(L)/nnz(M) = %.2f\n', nnz(L)/nnz(M));

    elseif strcmp(method, 'frozen_spectral')
        fprintf('Building Frozen Spectral preconditioner (based on the Hessian at X0)...\n');
        H0 = compute_hessian(X, N, K);
        fprintf('  Computing full spectral decomposition...\n');
        [U_spectral, Lambda] = eig(full(H0));
        lambda_vals = diag(Lambda);
        eps_M = 1e-2;
        lambda_metric = abs(lambda_vals) + eps_M;
        Lambda_inv = diag(1 ./ lambda_metric);
        fprintf('  Spectral decomposition complete: %d eigenvalues\n', n_dof);
        fprintf('  Minimum eigenvalue: %.6f\n', min(lambda_vals));
        fprintf('  eps_M: %.2e\n', eps_M);
        fprintf('  Minimum metric weight: %.6f\n', min(lambda_metric));
        fprintf('  Maximum metric weight: %.6f\n', max(lambda_metric));

    elseif strcmp(method, 'jacobi')
        fprintf('Building Jacobi preconditioner...\n');
        H0 = compute_hessian(X, N, K);
        d = abs(diag(H0));
        d = max(d, 0.1);  % Minimum value 0.1
        M_jacobi_inv = spdiags(1 ./ d, 0, n_dof, n_dof);
        fprintf('  Jacobi preconditioner construction complete\n');
    end

    % Initialize unstable direction
    fprintf('Initializing unstable direction...\n');
    V = initialize_frame(X, N, K, method, L, M, U_spectral, Lambda_inv, M_jacobi_inv);

    % Start timing
    t_start = tic;

    % Iteration loop
    for iter = 1:max_iter
        % Compute gradient
        grad = compute_gradient(X, N, K);
        gnorm = norm(grad);

        iter_count = iter_count + 1;
        history.grad_norm(iter_count) = gnorm;

        if gnorm < tol
            history.converged = true;
            history.iterations = iter;
            fprintf('Converged! Iterations: %d\n', iter);
            break;
        end

        if ~isfinite(gnorm) || gnorm > 1e8
            history.converged = false;
            history.iterations = iter;
            fprintf('Diverged! Stopped at iteration %d\n', iter);
            break;
        end

        % Perform HiSD update
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
            c = v' * grad;  % Use the standard inner product v^T g
            direction = apply_block_jacobi_inverse(X, grad, N, K);  % M^{-1} grad
            X = X - dt_x * (direction - 2 * c * v);  % -M^{-1}g + 2(v^T g)v
            V = update_frame_block_jacobi(X, V, N, K, dt_v, v_iter);

        elseif strcmp(method, 'ic')
            % IC p-HiSD
            M = build_ic_preconditioner_matrix(X, N, K);
            opts.type = 'ict';
            opts.droptol = 1e-5;
            L = ichol(M, opts);

            v = V{1};
            Mv = M * v;
            norm_M = sqrt(v' * Mv);
            if norm_M > 1e-10
                v = v / norm_M;
                V{1} = v;
            end
            t_g = L' \ (L \ grad);
            c = v' * grad;
            X = X - dt_x * (t_g - 2 * c * v);

            M = build_ic_preconditioner_matrix(X, N, K);
            opts.type = 'ict';
            opts.droptol = 1e-5;
            L = ichol(M, opts);
            V = update_frame_ic(X, V, N, K, dt_v, v_iter, L, M);

        elseif strcmp(method, 'frozen_spectral')
            % Frozen Spectral p-HiSD
            v = V{1};
            Ut_grad = U_spectral' * grad;
            Lambda_inv_Ut_grad = Lambda_inv * Ut_grad;
            d = U_spectral * Lambda_inv_Ut_grad;
            c = v' * grad;
            direction = d - 2 * c * v;
            X = X - dt_x * direction;
            V = update_frame_frozen_spectral(X, V, N, K, dt_v, v_iter, U_spectral, Lambda_inv);

        else  % jacobi
            % Jacobi p-HiSD
            v = V{1};
            M_inv_g = M_jacobi_inv * grad;
            c = v' * grad;
            direction = M_inv_g - 2 * c * v;
            X = X - dt_x * direction;
            V = update_frame_jacobi(X, V, N, K, dt_v, v_iter, M_jacobi_inv);
        end

        % Periodically recompute the frame
        if mod(iter, 50) == 0 && iter > 0
            V = initialize_frame(X, N, K, method, L, M, U_spectral, Lambda_inv, M_jacobi_inv);
        end
    end

    if ~history.converged && iter == max_iter
        history.iterations = max_iter;
        fprintf('Reached maximum iterations\n');
    end

    history.total_time = toc(t_start);
    history.grad_norm = history.grad_norm(1:iter_count);
end

% ============================================================================
% Initialize unstable direction
% ============================================================================
function V = initialize_frame(X, N, K, method, L, M, U_spectral, Lambda_inv, M_jacobi_inv)
    n_dof = 2 * N;

    % Construct Hessian (analytical formula)
    H = full(compute_hessian(X, N, K));

    if strcmp(method, 'standard')
        [vecs, vals] = eig(H);
        [~, idx] = sort(diag(vals));
        v = vecs(:, idx(1));
        v = v / norm(v);

    elseif strcmp(method, 'block_jacobi')
        M_bj = get_block_jacobi_M_matrix(X, N, K);
        [vecs, vals] = eig(H, M_bj);
        [~, idx] = sort(diag(vals));
        v = vecs(:, idx(1));
        Mv = apply_block_jacobi(X, v, N, K);  % Use M, not M^{-1}
        norm_M = sqrt(v' * Mv);
        v = v / norm_M;

    elseif strcmp(method, 'ic')
        [v, ~] = eigs(H, M, 1, 'smallestreal');
        Mv = M * v;
        norm_M = sqrt(v' * Mv);
        v = v / norm_M;

    elseif strcmp(method, 'frozen_spectral')
        lambda_metric = 1.0 ./ diag(Lambda_inv);
        M_frozen = U_spectral * diag(lambda_metric) * U_spectral';
        M_frozen = 0.5 * (M_frozen + M_frozen');

        [vecs, vals] = eig(H, M_frozen);
        [~, idx] = sort(diag(vals));
        v = vecs(:, idx(1));

        % M-normalization
        u_proj = U_spectral' * v;
        M_norm_sq = sum( (u_proj.^2) .* lambda_metric );
        norm_M = sqrt(M_norm_sq);

        if norm_M > 1e-10
            v = v / norm_M;
        end

    else  % jacobi
        inv_diag = full(diag(M_jacobi_inv));
        M_diag = 1.0 ./ inv_diag;
        M_jacobi = diag(M_diag);
        [vecs, vals] = eig(H, M_jacobi);
        [~, idx] = sort(diag(vals));
        v = vecs(:, idx(1));

        % M-normalization: v^T M v = 1
        Mv = M_jacobi_inv \ v;  % M * v
        norm_M = sqrt(v' * Mv);
        v = v / norm_M;
    end

    V = {v};
end

% ============================================================================
% Frame update - Standard
% ============================================================================
function V_new = update_frame_standard(X, V, N, K, dt_v, v_iter)
    n_dof = 2 * N;

    H = full(compute_hessian(X, N, K));

    v = V{1};
    for iter_v = 1:v_iter
        Gv = H * v;
        coeff = v' * Gv;
        projected = Gv - coeff * v;
        v_new = v - dt_v * projected;
        v = v_new / norm(v_new);
    end

    V_new = {v};
end

% ============================================================================
% Frame update - Block Jacobi
% ============================================================================
function V_new = update_frame_block_jacobi(X, V, N, K, dt_v, v_iter)
    n_dof = 2 * N;

    H = full(compute_hessian(X, N, K));

    v = V{1};
    for iter_v = 1:v_iter
        Gv = H * v;
        Minv_Gv = apply_block_jacobi_inverse(X, Gv, N, K);
        coeff = v' * Gv;  % Use v^T Hv, not v^T M^{-1}Hv
        projected = Minv_Gv - coeff * v;
        v_new = v - dt_v * projected;
        norm_M = sqrt(m_inner_product_bj(X, v_new, v_new, N, K));
        if norm_M > 1e-10
            v_new = v_new / norm_M;
        end
        v = v_new;
    end

    V_new = {v};
end

% ============================================================================
% Frame update - IC
% ============================================================================
function V_new = update_frame_ic(X, V, N, K, dt_v, v_iter, L, M)
    n_dof = 2 * N;

    H = full(compute_hessian(X, N, K));

    v = V{1};
    for iter_v = 1:v_iter
        Gv = H * v;
        Minv_Gv = L' \ (L \ Gv);
        coeff = v' * Gv;  % Use v^T Hv, not v^T M^{-1}Hv
        projected = Minv_Gv - coeff * v;
        v_new = v - dt_v * projected;
        Mv_new = M * v_new;
        norm_M = sqrt(v_new' * Mv_new);
        v = v_new / norm_M;
    end

    V_new = {v};
end

% ============================================================================
% Frame update - Frozen Spectral
% ============================================================================
function V_new = update_frame_frozen_spectral(X, V, N, K, dt_v, v_iter, U_spectral, Lambda_inv)
    n_dof = 2 * N;

    H = full(compute_hessian(X, N, K));

    v = V{1};
    for iter_v = 1:v_iter
        Gv = H * v;

        % Apply Frozen Spectral preconditioner
        Ut_Gv = U_spectral' * Gv;
        Lambda_inv_Ut_Gv = Lambda_inv * Ut_Gv;
        Minv_Gv = U_spectral * Lambda_inv_Ut_Gv;

        % M-projection
        coeff = v' * Gv;  % Use v^T Hv, not v^T M^{-1}Hv
        projected = Minv_Gv - coeff * v;

        % Update
        v_new = v - dt_v * projected;

        % M-normalization
        lambda_metric = 1.0 ./ diag(Lambda_inv);
        u_proj = U_spectral' * v_new;
        M_norm_sq = sum( (u_proj.^2) .* lambda_metric );
        norm_M = sqrt(M_norm_sq);

        if norm_M > 1e-10
            v_new = v_new / norm_M;
        end
        v = v_new;
    end

    V_new = {v};
end

% ============================================================================
% Frame update - Jacobi
% ============================================================================
function V_new = update_frame_jacobi(X, V, N, K, dt_v, v_iter, M_jacobi_inv)
    n_dof = 2 * N;

    H = full(compute_hessian(X, N, K));

    v = V{1};
    for iter_v = 1:v_iter
        Gv = H * v;
        Minv_Gv = M_jacobi_inv * Gv;
        coeff = v' * Gv;  % Use v^T Hv, not v^T M^{-1}Hv
        projected = Minv_Gv - coeff * v;
        v_new = v - dt_v * projected;

        % M-normalization: v^T M v = 1
        Mv_new = M_jacobi_inv \ v_new;  % M * v_new
        norm_M = sqrt(v_new' * Mv_new);
        v = v_new / norm_M;
    end

    V_new = {v};
end

% ============================================================================
% Compute analytical Hessian
% ============================================================================
function H = compute_hessian(X, N, K)
    n_dof = 2 * N;
    x = X(1:2:end);
    y = X(2:2:end);

    I_idx = [];
    J_idx = [];
    V_vals = [];

    for i = 1:N
        idx_x = 2*i - 1;
        idx_y = 2*i;

        % Diagonal entry: ∂²E/∂x_i²
        H_xx = K + 12*x(i)^2 - 4;
        if i > 1
            H_xx = H_xx + 1;
        end

        % Diagonal entry: ∂²E/∂y_i²
        H_yy = K + 12*y(i)^2 - 4;
        if i < N
            H_yy = H_yy + 1;
        end

        % Cross term: ∂²E/∂x_i∂y_i
        H_xy = -K;

        % Add diagonal block
        I_idx = [I_idx; idx_x; idx_y; idx_x; idx_y];
        J_idx = [J_idx; idx_x; idx_y; idx_y; idx_x];
        V_vals = [V_vals; H_xx; H_yy; H_xy; H_xy];

        % Coupling term: ∂²E/∂y_i∂x_{i+1}
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
% Compute gradient
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
% Block Jacobi helper functions
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

function M = get_block_jacobi_matrix(X, N, K)
    n_dof = 2 * N;
    M = zeros(n_dof, n_dof);

    for i = 1:n_dof
        ei = zeros(n_dof, 1);
        ei(i) = 1.0;
        M(:, i) = apply_block_jacobi_inverse(X, ei, N, K);
    end
end

function M = get_block_jacobi_M_matrix(X, N, K)
    n_dof = 2 * N;
    M = zeros(n_dof, n_dof);

    for i = 1:n_dof
        ei = zeros(n_dof, 1);
        ei(i) = 1.0;
        M(:, i) = apply_block_jacobi(X, ei, N, K);
    end

    M = 0.5 * (M + M');
end

function result = m_inner_product_bj(X, u, v, N, K)
    Mv = apply_block_jacobi(X, v, N, K);  % Apply M, not M^{-1}
    result = u' * Mv;
end

% Apply the Block Jacobi preconditioner matrix M (not M^{-1})
function result = apply_block_jacobi(X, vector, N, K)
    vec_x = vector(1:2:end);
    vec_y = vector(2:2:end);

    [H_xx, H_yy, H_xy] = get_local_blocks(X, N, K);

    res_x = H_xx .* vec_x + H_xy .* vec_y;
    res_y = H_xy .* vec_x + H_yy .* vec_y;

    result = zeros(2*N, 1);
    result(1:2:end) = res_x;
    result(2:2:end) = res_y;
end

% ============================================================================
% IC preconditioner
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
% Plotting function
% ============================================================================
function plot_adaptive_comparison(h_std, h_prec, dt_std, dt_prec, method_name, N)
    % Convert the method name to a display-friendly label (avoid subscript issues caused by underscores)
    if strcmp(method_name, 'frozen_spectral')
        display_name = 'Frozen Spectral';
    elseif strcmp(method_name, 'block_jacobi')
        display_name = 'Block Jacobi';
    elseif strcmp(method_name, 'ic')
        display_name = 'IC';
    elseif strcmp(method_name, 'jacobi')
        display_name = 'Jacobi';
    else
        display_name = strrep(method_name, '_', ' ');  % Replace all underscores with spaces
    end

    figure('Position', [100, 100, 800, 500]);

    iter_std = h_std.iterations;
    iter_prec = h_prec.iterations;
    time_std = h_std.total_time;
    time_prec = h_prec.total_time;

    valid_std = isfinite(h_std.grad_norm);
    valid_prec = isfinite(h_prec.grad_norm);

    if any(valid_std)
        steps_std = find(valid_std);
        norms_std = h_std.grad_norm(valid_std);
        semilogy(steps_std, norms_std, 'b-', 'LineWidth', 2, ...
            'DisplayName', sprintf('Standard HiSD: %d iters, %.2fs', iter_std, time_std));
    end

    hold on;

    if any(valid_prec)
        steps_prec = find(valid_prec);
        norms_prec = h_prec.grad_norm(valid_prec);
        semilogy(steps_prec, norms_prec, 'r-', 'LineWidth', 2.5, ...
            'DisplayName', sprintf('%s p-HiSD: %d iters, %.2fs', display_name, iter_prec, time_prec));
    end

    xlabel('Iteration', 'FontSize', 14);
    ylabel('Gradient Norm ||∇E||', 'FontSize', 14);
    title(sprintf('Adaptive Selection (N=%d, n=%d): %s', N, 2*N, display_name), 'FontSize', 15);
    grid on;
    legend('Location', 'best', 'FontSize', 11);

    set(gca, 'LooseInset', get(gca, 'TightInset'));

    filename = sprintf('adaptive_comparison_N%d.pdf', N);
    exportgraphics(gcf, filename, 'ContentType', 'vector', 'BackgroundColor', 'none');
    fprintf('\nFigure saved: %s\n', filename);
end
