% Section 7.4: compare four index-1 solvers for a stiff diatomic chain.
% Save data for Figure 5 and Table 5 to outputs/7.4/; fig.py draws the paper figure.


clear; close all; clc;

timing_repeats = 3;

fprintf('======================================================================\n');
fprintf('Stiff Diatomic Chain: Four-method comparison (including Frozen Spectral)\n');
fprintf('======================================================================\n');

N = 50;           % Number of molecular pairs
K = 10000.0;      % Stiffness coefficient
n_dof = 2 * N;    % Number of degrees of freedom

fprintf('System: %d molecular pairs, K=%.0e\n', N, K);

dt_x_standard = 5e-5;        % Standard HiSD - x step size
dt_v_standard = 5e-5;        % Standard HiSD - v step size

dt_x_block_jacobi = 0.5;     % Block Jacobi p-HiSD - x step size
dt_v_block_jacobi = 0.05;    % Block Jacobi p-HiSD - v step size

dt_x_ic = 0.5;               % IC p-HiSD - x step size
dt_v_ic = 0.05;              % IC p-HiSD - v step size

dt_x_frozen_spectral = 0.5;  % Frozen Spectral p-HiSD - x step size
dt_v_frozen_spectral = 0.05; % Frozen Spectral p-HiSD - v step size

v_iter = 5;                  % Shared by all methods: v_iter v steps per x step
delta_ind = 1e-6;            % Fixed absolute endpoint Morse-index threshold

% Perturb the alternating state once and share it across methods and repetitions.
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

timing_rng_initial = rng;
timing_raw = struct();
timing_raw.schema = 'section7.4.timing.v3';
timing_raw.methods = {'standard', 'block_jacobi', 'ic', 'frozen_spectral'};
timing_raw.repeats = timing_repeats;
timing_raw.X0 = X0;
timing_raw.rng_initial = timing_rng_initial;
timing_raw.parameters = struct('N', N, 'K', K, 'dt_x', ...
    [dt_x_standard, dt_x_block_jacobi, dt_x_ic, dt_x_frozen_spectral], ...
    'dt_v', [dt_v_standard, dt_v_block_jacobi, dt_v_ic, dt_v_frozen_spectral], ...
    'v_iter', v_iter, 'max_iter', 2000000, 'tol', 1e-6, 'seed', 42, ...
    'target_index', 1, 'delta_ind', delta_ind);
timing_raw.environment = struct('matlab_version', version, 'computer', computer, ...
    'threads', maxNumCompThreads, 'blas', version('-blas'), 'lapack', version('-lapack'));
timing_raw.run_policy = ['Sequential repetitions in one MATLAB session; identical X0, ' ...
    'parameters and initial RNG state; original method order; no solver warmup; ' ...
    'solver printing disabled; instrumented wall-clock includes timer overhead.'];
timing_raw.definitions = timing74_definitions();
timing_raw.r1_schema = 'section7.4.cost.v2';
timing_raw.r1_definitions = r1_cost_definitions();
timing_raw.T_total_includes_certification = false;
timing_raw.component_times_may_overlap = true;
timing_raw.histories = cell(timing_repeats, 4);
timing_raw.timings = cell(timing_repeats, 4);
timing_raw.rng_before_method = cell(timing_repeats, 4);
timing_raw.rng_final = cell(timing_repeats, 1);
% Repeat the same numerical experiment three times; retain the first history for plotting.
for timing_repeat = 1:timing_repeats
    rng(timing_rng_initial);
    fprintf('\nTiming repetition %d/%d\n', timing_repeat, timing_repeats);
fprintf('[1] Solving Standard HiSD (dt_x=%.0e, dt_v=%.3g, v_iter=%d)...\n', dt_x_standard, dt_v_standard, v_iter);
timing_raw.rng_before_method{timing_repeat, 1} = rng;
hist_standard = run_hisd(X0, N, K, dt_x_standard, dt_v_standard, v_iter, 2000000, 1e-6, 'standard', delta_ind);

fprintf('\n[2] Solving Block Jacobi p-HiSD (dt_x=%.2f, dt_v=%.2f, v_iter=%d)...\n', dt_x_block_jacobi, dt_v_block_jacobi, v_iter);
timing_raw.rng_before_method{timing_repeat, 2} = rng;
hist_block_jacobi = run_hisd(X0, N, K, dt_x_block_jacobi, dt_v_block_jacobi, v_iter, 2000000, 1e-6, 'block_jacobi', delta_ind);

fprintf('\n[3] Solving IC p-HiSD (dt_x=%.2f, dt_v=%.2f, v_iter=%d)...\n', dt_x_ic, dt_v_ic, v_iter);
timing_raw.rng_before_method{timing_repeat, 3} = rng;
hist_ic = run_hisd(X0, N, K, dt_x_ic, dt_v_ic, v_iter, 2000000, 1e-6, 'ic', delta_ind);

fprintf('\n[4] Solving Frozen Spectral p-HiSD (dt_x=%.2f, dt_v=%.2f, v_iter=%d)...\n', dt_x_frozen_spectral, dt_v_frozen_spectral, v_iter);
timing_raw.rng_before_method{timing_repeat, 4} = rng;
hist_frozen_spectral = run_hisd(X0, N, K, dt_x_frozen_spectral, dt_v_frozen_spectral, v_iter, 2000000, 1e-6, 'frozen_spectral', delta_ind);

    timing_histories = {hist_standard, hist_block_jacobi, hist_ic, hist_frozen_spectral};
    timing_raw.rng_final{timing_repeat} = rng;
    for timing_method = 1:4
        timing_history = timing_histories{timing_method};
        timing_raw.histories{timing_repeat, timing_method} = timing_history;
        timing_raw.timings{timing_repeat, timing_method} = timing_history.timing;
        if timing_repeat > 1
            timing_reference = timing_raw.histories{1, timing_method};
            timing_fields = setdiff(fieldnames(timing_reference), ...
                {'timing', 'total_time', 'legacy_outer_time'});
            for timing_field = 1:numel(timing_fields)
                assert(isequaln(timing_reference.(timing_fields{timing_field}), ...
                    timing_history.(timing_fields{timing_field})), ...
                    'Section 7.4 numerical repetition mismatch: %s / %s', ...
                    timing_raw.methods{timing_method}, timing_fields{timing_field});
            end
            assert(isequaln(timing_raw.rng_before_method{1, timing_method}, ...
                timing_raw.rng_before_method{timing_repeat, timing_method}));
        end
        assert(~timing_history.timing.T_total_includes_certification);
        assert(timing_history.total_time == timing_history.timing.T_total);
        if timing_method == 4
            assert(timing_history.timing.T_setup > 0);
            assert(timing_history.timing.T_update == 0);
            assert(timing_history.timing.fs_spectral_builds == 1);
        end
    end
    assert(isequaln(timing_raw.rng_final{1}, timing_raw.rng_final{timing_repeat}));
end
timing_raw.numerical_repetitions_identical = true;
hist_standard = timing_raw.histories{1, 1};
hist_block_jacobi = timing_raw.histories{1, 2};
hist_ic = timing_raw.histories{1, 3};
hist_frozen_spectral = timing_raw.histories{1, 4};
timing_summary = timing74_summarize(timing_raw);
fprintf('\nLegacy timing summary in seconds (median / min / max):\n');
disp(timing_summary.table(:, 1:7));
fprintf('%s\n', timing_raw.definitions.accounting);
fprintf('Legacy T_total excludes the final residual and Morse-index certification.\n');
fprintf('The original performance table below describes repetition 1.\n');


fprintf('\n================================================================================\n');
fprintf('Final performance comparison\n');
fprintf('================================================================================\n');
fprintf('%-30s | %-10s | %-15s | %-12s\n', 'Method', 'Iterations', 'Final gradient norm', 'Runtime (s)');
fprintf('--------------------------------------------------------------------------------\n');
fprintf('%-30s | %-10d | %-15.4e | %-12.3f\n', 'Standard HiSD', hist_standard.iterations, hist_standard.endpoint_residual, hist_standard.total_time);
fprintf('%-30s | %-10d | %-15.4e | %-12.3f\n', 'Block Jacobi p-HiSD', hist_block_jacobi.iterations, hist_block_jacobi.endpoint_residual, hist_block_jacobi.total_time);
fprintf('%-30s | %-10d | %-15.4e | %-12.3f\n', 'IC p-HiSD', hist_ic.iterations, hist_ic.endpoint_residual, hist_ic.total_time);
fprintf('%-30s | %-10d | %-15.4e | %-12.3f\n', 'Frozen Spectral p-HiSD', hist_frozen_spectral.iterations, hist_frozen_spectral.endpoint_residual, hist_frozen_spectral.total_time);

fprintf('\nEndpoint certification (absolute delta_ind = %.0e; target index = 1):\n', delta_ind);
fprintf('%-24s | %-23s | %-5s | %-12s | %-12s\n', ...
    'Method', 'Status', 'Index', 'lambda_1', 'lambda_2');
cert_histories = {hist_standard, hist_block_jacobi, hist_ic, hist_frozen_spectral};
for cert_method = 1:4
    cert_history = cert_histories{cert_method};
    fprintf('%-24s | %-23s | %-5g | %12.4e | %12.4e\n', ...
        timing_raw.methods{cert_method}, cert_history.endpoint_status, ...
        cert_history.morse_index, cert_history.lambda_1, cert_history.lambda_2);
end

if hist_standard.converged
    if hist_block_jacobi.converged
        speedup_bj_iter = hist_standard.iterations / hist_block_jacobi.iterations;
        speedup_bj_time = hist_standard.total_time / hist_block_jacobi.total_time;
        fprintf('--------------------------------------------------------------------------------\n');
        fprintf('Block Jacobi speedup: %.1f× (iterations), %.1f× (time)\n', speedup_bj_iter, speedup_bj_time);
    end
    if hist_ic.converged
        speedup_ic_iter = hist_standard.iterations / hist_ic.iterations;
        speedup_ic_time = hist_standard.total_time / hist_ic.total_time;
        fprintf('IC speedup: %.1f× (iterations), %.1f× (time)\n', speedup_ic_iter, speedup_ic_time);
    end
    if hist_frozen_spectral.converged
        speedup_fs_iter = hist_standard.iterations / hist_frozen_spectral.iterations;
        speedup_fs_time = hist_standard.total_time / hist_frozen_spectral.total_time;
        fprintf('Frozen Spectral speedup: %.1f× (iterations), %.1f× (time)\n', speedup_fs_iter, speedup_fs_time);
    end
end
fprintf('================================================================================\n');

fprintf('\n[Check] Frozen Spectral preconditioner builds: %d (expected: 1)\n', hist_frozen_spectral.precond_constructions);

script_dir = fileparts(mfilename('fullpath'));
output_dir = fullfile(script_dir, '..', '..', 'outputs', '7.4');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
mat_path = fullfile(output_dir, 'experiment_results.mat');

fprintf('\nSaving data to a .mat file...\n');
save(mat_path, ...
     'hist_standard', 'hist_block_jacobi', 'hist_ic', 'hist_frozen_spectral', ...
     'dt_x_standard', 'dt_x_block_jacobi', 'dt_x_ic', 'dt_x_frozen_spectral', ...
     'dt_v_standard', 'dt_v_block_jacobi', 'dt_v_ic', 'dt_v_frozen_spectral', ...
     'v_iter', 'N', 'K', 'delta_ind');
fprintf('Data saved to %s\n', mat_path);

save(fullfile(output_dir, 'timing74_raw.mat'), 'timing_raw');
save(fullfile(output_dir, 'timing74_summary.mat'), 'timing_summary');
writetable(timing_summary.table, fullfile(output_dir, 'timing74_summary.csv'));
fprintf('Timing data saved to %s\n', output_dir);

[timing_runs, cost_summary, cost_variability] = r1_make_cost_tables(timing_raw);
r1_write_cost_csv(timing_runs, fullfile(output_dir, 'timing_runs.csv'));
r1_write_cost_csv(cost_summary, fullfile(output_dir, 'timing_summary.csv'));
fprintf('\nExclusive cost summary in seconds (median of %d repetitions):\n', timing_repeats);
disp(cost_summary(:, {'method', 'max_iter', 'actual_outer_updates', ...
    'endpoint_residual', 'stop_reason', 'morse_index', 'T_total', ...
    'T_M_setup_upd', 'T_M_apply_solve', 'T_eig_net', 'T_certify'}));
fprintf('T_total includes final residual and Morse-index certification; T_certify reports their cost.\n');
fprintf('CSV files saved to %s\n', output_dir);

% Evolve one unstable direction with v_iter frame sweeps per state update.
function history = run_hisd(X0, N, K, dt_x, dt_v, v_iter, max_iter, tol, method, delta_ind)
    t74_total = tic;
    global S74;
    S74 = struct('setup_update', 0, 'apply_solve', 0, 'eig', 0, 'setup_ic_initial', 0, 'setup_ic_state', 0, 'setup_ic_frame', 0, 'setup_fs_initial', 0, 'fs_matrix_reconstruction_time', 0, 'fs_weight_restore_time', 0, 'setup_bj_blocks', 0, 'setup_bj_inverse_coefficients', 0, 'setup_bj_matrix', 0, 'eig_initial', 0, 'eig_update', 0, 'eig_reinitialize', 0, 'eig_metric_spectral', 0, 'eig_state_normalization', 0, 'ic_builds', 0, 'ic_initial_builds', 0, 'ic_state_builds', 0, 'ic_frame_builds', 0, 'ic_factorizations', 0, 'ic_factorization_time', 0, 'ic_failures', 0, 'bj_block_builds', 0, 'bj_matrix_builds', 0, 'fs_spectral_builds', 0, 'fs_matrix_reconstructions', 0, 'frame_initializations', 0, 'frame_updates', 0, 'apply_calls', 0, 'setup_depth', 0, ...
        'setup', 0, 'update', 0, 'apply', 0, 'solve', 0, ...
        'solve_calls', 0, 'initializing', true, 'quiet', true);
    S74.r1_M_setup_upd = 0;
    S74.r1_M_apply_solve = 0;
    S74.r1_eig_net = 0;
    S74.r1_frame_inclusive = 0;
    S74.r1_frame_setup_excluded = 0;
    S74.r1_frame_apply_excluded = 0;
    S74.r1_frame_regions = 0;
    S74.r1_min_frame_net = Inf;
    S74.r1_bj_dense_inclusive = 0;
    S74.r1_bj_dense_apply_excluded = 0;
    S74.r1_bj_dense_setup_net = 0;
    S74.r1_min_bj_dense_net = Inf;
    n_dof = 2 * N;
    X = X0;

    history.grad_norm = zeros(1, max_iter);
    history.converged = false;
    history.iterations = 0;
    history.precond_constructions = 0;  % Count preconditioner builds
    iter_count = 0;

    L = [];
    M = [];
    U_spectral = [];
    Lambda_inv = [];

    if strcmp(method, 'ic')
    if ~S74.quiet
        fprintf('Building the Shifted-IC preconditioner...\n');
    end
    if ~S74.quiet
        fprintf('========================================\n');
    end

    t74_setup_ic_initial = tic;
        [L, M, prec_info] = build_shifted_ic_preconditioner(X, N, K);
    t74_elapsed = toc(t74_setup_ic_initial);
    S74.setup_ic_initial = S74.setup_ic_initial + t74_elapsed;
    S74.r1_M_setup_upd = S74.r1_M_setup_upd + t74_elapsed;
    S74.setup = S74.setup + t74_elapsed;
    S74.ic_initial_builds = S74.ic_initial_builds + 1;

    if ~S74.quiet
        fprintf('========================================\n');
    end
    if ~S74.quiet
        fprintf('Preconditioner construction complete:\n');
    end
    if ~S74.quiet
        fprintf('  Final shift δ = %.4e\n', prec_info.final_delta);
    end
    if ~S74.quiet
        fprintf('  Shift trials = %d\n', prec_info.shift_trials);
    end
    if ~S74.quiet
        fprintf('  nnz(L) = %d\n', prec_info.nnz_L);
    end
    if ~S74.quiet
        fprintf('  nnz(A) = %d\n', prec_info.nnz_A);
    end
    if ~S74.quiet
        fprintf('  Fill ratio = %.2f\n', prec_info.fill_ratio);
    end
    if ~S74.quiet
        fprintf('  Construction time = %.4f s\n', prec_info.factor_time);
    end
    if ~S74.quiet
        fprintf('  Ordering = %s\n', prec_info.ordering);
    end
        if prec_info.failed
    if ~S74.quiet
            fprintf('  Warning: IC failed; using the fail-safe preconditioner\n');
    end
        end
    if ~S74.quiet
        fprintf('========================================\n\n');
    end
        history.precond_constructions = 1;

    elseif strcmp(method, 'frozen_spectral')
    if ~S74.quiet
        fprintf('Building the Frozen Spectral preconditioner (from the Hessian at X0)...\n');
    end

    t74_setup_fs_initial = tic;
        H0 = compute_hessian(X, N, K);

    if ~S74.quiet
        fprintf('  Computing the full eigendecomposition...\n');
    end
    t74_eig_metric_spectral = tic;
        [U_spectral, Lambda] = eig(full(H0));
    t74_elapsed = toc(t74_eig_metric_spectral);
    S74.eig_metric_spectral = S74.eig_metric_spectral + t74_elapsed;
    S74.eig = S74.eig + t74_elapsed;
        lambda_vals = diag(Lambda);

        eps_M = 1e-2;
        lambda_metric = abs(lambda_vals) + eps_M;

        Lambda_inv = diag(1 ./ lambda_metric);
    t74_elapsed = toc(t74_setup_fs_initial);
    S74.setup_fs_initial = S74.setup_fs_initial + t74_elapsed;
    S74.r1_M_setup_upd = S74.r1_M_setup_upd + t74_elapsed;
    S74.setup = S74.setup + t74_elapsed;
    S74.fs_spectral_builds = S74.fs_spectral_builds + 1;
    if ~S74.quiet

        fprintf('  Eigendecomposition complete: %d eigenvalues\n', n_dof);
    end
    if ~S74.quiet
        fprintf('  Minimum eigenvalue: %.6f\n', min(lambda_vals));
    end
    if ~S74.quiet
        fprintf('  Maximum eigenvalue: %.6f\n', max(lambda_vals));
    end
    if ~S74.quiet
        fprintf('  eps_M: %.2e\n', eps_M);
    end
    if ~S74.quiet
        fprintf('  Minimum metric weight: %.6f\n', min(lambda_metric));
    end
    if ~S74.quiet
        fprintf('  Maximum metric weight: %.6f\n', max(lambda_metric));
    end

        history.precond_constructions = 1;  % Build only once
    end

    if ~S74.quiet
    fprintf('Initialize the unstable direction...\n');
    end
    r1_frame_setup_before = S74.r1_M_setup_upd;
    r1_frame_apply_before = S74.r1_M_apply_solve;
    t74_eig_initial = tic;
    V = initialize_frame(X, N, K, method, L, M, U_spectral, Lambda_inv);
    t74_elapsed = toc(t74_eig_initial);
    S74.eig_initial = S74.eig_initial + t74_elapsed;
    r1_account_frame(t74_elapsed, r1_frame_setup_before, r1_frame_apply_before);
    S74.eig = S74.eig + t74_elapsed;
    S74.frame_initializations = S74.frame_initializations + 1;
    S74.initializing = false;

    t_start = tic;

    % Check the raw gradient before updating; stop on tolerance, divergence or budget.
    for iter = 1:max_iter
        grad = compute_gradient(X, N, K);
        gnorm = norm(grad);

        iter_count = iter_count + 1;
        history.grad_norm(iter_count) = gnorm;

        if gnorm < tol
            history.converged = true;
            history.iterations = iter;
    if ~S74.quiet
            fprintf('Gradient tolerance reached. Iterations: %d\n', iter);
    end
            break;
        end

        if ~isfinite(gnorm) || gnorm > 1e8
            history.converged = false;
            history.iterations = iter;
    if ~S74.quiet
            fprintf('Diverged! Stopped at iteration %d\n', iter);
    end
            break;
        end

        if strcmp(method, 'standard')
            v = V{1};
            c = v' * grad;
            grad_mod = grad - 2 * c * v;
            X = X - dt_x * grad_mod;
    r1_frame_setup_before = S74.r1_M_setup_upd;
    r1_frame_apply_before = S74.r1_M_apply_solve;
    t74_eig_update = tic;
            V = update_frame_standard(X, V, N, K, dt_v, v_iter);
    t74_elapsed = toc(t74_eig_update);
    S74.eig_update = S74.eig_update + t74_elapsed;
    r1_account_frame(t74_elapsed, r1_frame_setup_before, r1_frame_apply_before);
    S74.eig = S74.eig + t74_elapsed;
    S74.frame_updates = S74.frame_updates + 1;

        elseif strcmp(method, 'block_jacobi')
            v = V{1};
            c = v' * grad;  % Reflection coefficient v^T g for an M-normalized direction.
            direction = apply_block_jacobi_inverse(X, grad, N, K);  % M^{-1} grad
            X = X - dt_x * (direction - 2 * c * v);  % dir_x = -M^{-1}g + 2(v^T g)v
    r1_frame_setup_before = S74.r1_M_setup_upd;
    r1_frame_apply_before = S74.r1_M_apply_solve;
    t74_eig_update = tic;
            V = update_frame_block_jacobi(X, V, N, K, dt_v, v_iter);
    t74_elapsed = toc(t74_eig_update);
    S74.eig_update = S74.eig_update + t74_elapsed;
    r1_account_frame(t74_elapsed, r1_frame_setup_before, r1_frame_apply_before);
    S74.eig = S74.eig + t74_elapsed;
    S74.frame_updates = S74.frame_updates + 1;

        % IC uses a rebuilt metric at the current state and again at the new state.
        elseif strcmp(method, 'ic')
    t74_setup_ic_state = tic;
            [L, M, ~] = build_shifted_ic_preconditioner(X, N, K);
    t74_elapsed = toc(t74_setup_ic_state);
    S74.setup_ic_state = S74.setup_ic_state + t74_elapsed;
    S74.r1_M_setup_upd = S74.r1_M_setup_upd + t74_elapsed;
    S74.update = S74.update + t74_elapsed;
    S74.ic_state_builds = S74.ic_state_builds + 1;
            history.precond_constructions = history.precond_constructions + 1;

            v = V{1};
    r1_frame_setup_before = S74.r1_M_setup_upd;
    r1_frame_apply_before = S74.r1_M_apply_solve;
    t74_eig_state_normalization = tic;
    t74_apply = tic;
            Mv = M * v;
    r1_apply_elapsed = toc(t74_apply);
    S74.apply = S74.apply + r1_apply_elapsed;
    S74.r1_M_apply_solve = S74.r1_M_apply_solve + r1_apply_elapsed;
    S74.apply_calls = S74.apply_calls + 1;
            norm_M = sqrt(v' * Mv);
            if norm_M > 1e-10
                v = v / norm_M;
                V{1} = v;
            end
    t74_elapsed = toc(t74_eig_state_normalization);
    S74.eig_state_normalization = S74.eig_state_normalization + t74_elapsed;
    r1_account_frame(t74_elapsed, r1_frame_setup_before, r1_frame_apply_before);
    S74.eig = S74.eig + t74_elapsed;
    t74_apply = tic;
            t_g = L' \ (L \ grad);
    r1_apply_elapsed = toc(t74_apply);
    S74.solve = S74.solve + r1_apply_elapsed;
    S74.r1_M_apply_solve = S74.r1_M_apply_solve + r1_apply_elapsed;
    S74.solve_calls = S74.solve_calls + 1;
            c = v' * grad;
            X = X - dt_x * (t_g - 2 * c * v);

    t74_setup_ic_frame = tic;
            [L, M, ~] = build_shifted_ic_preconditioner(X, N, K);
    t74_elapsed = toc(t74_setup_ic_frame);
    S74.setup_ic_frame = S74.setup_ic_frame + t74_elapsed;
    S74.r1_M_setup_upd = S74.r1_M_setup_upd + t74_elapsed;
    S74.update = S74.update + t74_elapsed;
    S74.ic_frame_builds = S74.ic_frame_builds + 1;
            history.precond_constructions = history.precond_constructions + 1;
    r1_frame_setup_before = S74.r1_M_setup_upd;
    r1_frame_apply_before = S74.r1_M_apply_solve;
    t74_eig_update = tic;
            V = update_frame_ic(X, V, N, K, dt_v, v_iter, L, M);
    t74_elapsed = toc(t74_eig_update);
    S74.eig_update = S74.eig_update + t74_elapsed;
    r1_account_frame(t74_elapsed, r1_frame_setup_before, r1_frame_apply_before);
    S74.eig = S74.eig + t74_elapsed;
    S74.frame_updates = S74.frame_updates + 1;

        else  % frozen_spectral
            v = V{1};

    t74_apply = tic;
            Ut_grad = U_spectral' * grad;
            Lambda_inv_Ut_grad = Lambda_inv * Ut_grad;
            d = U_spectral * Lambda_inv_Ut_grad;
    r1_apply_elapsed = toc(t74_apply);
    S74.solve = S74.solve + r1_apply_elapsed;
    S74.r1_M_apply_solve = S74.r1_M_apply_solve + r1_apply_elapsed;
    S74.solve_calls = S74.solve_calls + 1;

            c = v' * grad;

            direction = d - 2 * c * v;

            X = X - dt_x * direction;

    r1_frame_setup_before = S74.r1_M_setup_upd;
    r1_frame_apply_before = S74.r1_M_apply_solve;
    t74_eig_update = tic;
            V = update_frame_frozen_spectral(X, V, N, K, dt_v, v_iter, U_spectral, Lambda_inv);
    t74_elapsed = toc(t74_eig_update);
    S74.eig_update = S74.eig_update + t74_elapsed;
    r1_account_frame(t74_elapsed, r1_frame_setup_before, r1_frame_apply_before);
    S74.eig = S74.eig + t74_elapsed;
    S74.frame_updates = S74.frame_updates + 1;
        end

        % Reinitialize the unstable direction every 50 outer iterations.
        if mod(iter, 50) == 0 && iter > 0
    r1_frame_setup_before = S74.r1_M_setup_upd;
    r1_frame_apply_before = S74.r1_M_apply_solve;
    t74_eig_reinitialize = tic;
            V = initialize_frame(X, N, K, method, L, M, U_spectral, Lambda_inv);
    t74_elapsed = toc(t74_eig_reinitialize);
    S74.eig_reinitialize = S74.eig_reinitialize + t74_elapsed;
    r1_account_frame(t74_elapsed, r1_frame_setup_before, r1_frame_apply_before);
    S74.eig = S74.eig + t74_elapsed;
    S74.frame_initializations = S74.frame_initializations + 1;
        end
    end

    if ~history.converged && iter == max_iter
        history.iterations = max_iter;
    if ~S74.quiet
        fprintf('Maximum iteration count reached\n');
    end
    end

    t74_solver_elapsed = toc(t74_total);
    history.total_time = toc(t_start);
    history.grad_norm = history.grad_norm(1:iter_count);
    history.x_final = X;  % Save the final point
    history.legacy_outer_time = history.total_time;
    S74.T_total = t74_solver_elapsed;
    S74.T_setup = S74.setup;
    S74.T_update = S74.update;
    S74.T_setup_update = S74.T_setup + S74.T_update;
    S74.T_apply = S74.apply;
    S74.T_solve = S74.solve;
    S74.T_apply_solve = S74.T_apply + S74.T_solve;
    S74.T_eig = S74.eig;
    S74.setup_update = S74.T_setup_update;
    S74.apply_solve = S74.T_apply_solve;
    S74.apply_solve_calls = S74.apply_calls + S74.solve_calls;
    S74.component_times_may_overlap = true;
    S74.T_total_includes_certification = false;
    t74_certify = tic;
    % Check the final point independently; r1 total includes this certification.
    % The separately stored solver-only T_total excludes it.
    certificate = certify_chain_endpoint(history.x_final, N, K, tol, delta_ind);
    cert_fields = fieldnames(certificate);
    for cert_field = 1:numel(cert_fields)
        history.(cert_fields{cert_field}) = certificate.(cert_fields{cert_field});
    end
    S74.T_certify = toc(t74_certify);
    r1_total_through_endpoint = toc(t74_total);
    S74.r1_T_total = r1_total_through_endpoint;
    S74.r1_T_total_includes_endpoint = true;
    S74.r1_components_overlap = false;
    history.timing = S74;
    history.total_time = history.timing.T_total;
end

% Require the residual tolerance and a resolved index-1 ordinary Hessian spectrum.
function certificate = certify_chain_endpoint(X, N, K, tol, delta_ind)
    certificate = struct('target_index', 1, 'delta_ind', delta_ind, ...
        'endpoint_residual', Inf, 'endpoint_residual_pass', false, ...
        'endpoint_eigenvalues', nan(2*N, 1), 'lambda_1', NaN, 'lambda_2', NaN, ...
        'morse_index', NaN, 'index_resolved', false, 'index_pass', false, ...
        'endpoint_status', 'residual_not_converged', 'converged', false, ...
        'certification_error', '');
    if all(isfinite(X(:)))
        certificate.endpoint_residual = norm(compute_gradient(X, N, K));
        H = full(compute_hessian(X, N, K));
        if all(isfinite(H(:)))
            try
                eigenvalues = sort(eig((H + H') / 2));
                certificate.endpoint_eigenvalues = eigenvalues;
                certificate.lambda_1 = eigenvalues(1);
                certificate.lambda_2 = eigenvalues(2);
                certificate.index_resolved = all(isfinite(eigenvalues)) && ...
                    all(abs(eigenvalues) > delta_ind);
                if certificate.index_resolved
                    certificate.morse_index = sum(eigenvalues < -delta_ind);
                    certificate.index_pass = eigenvalues(1) < -delta_ind && ...
                        eigenvalues(2) > delta_ind;
                end
            catch err
                certificate.certification_error = err.message;
            end
        else
            certificate.certification_error = 'Non-finite endpoint Hessian.';
        end
    else
        certificate.certification_error = 'Non-finite endpoint state.';
    end
    certificate.endpoint_residual_pass = isfinite(certificate.endpoint_residual) && ...
        certificate.endpoint_residual < tol;
    if ~certificate.endpoint_residual_pass
        certificate.endpoint_status = 'residual_not_converged';
    elseif ~certificate.index_resolved
        certificate.endpoint_status = 'index_unresolved';
    elseif ~certificate.index_pass
        certificate.endpoint_status = 'wrong_index';
    else
        certificate.endpoint_status = 'converged';
        certificate.converged = true;
    end
end

% Initialize the lowest mode and normalize it in the method's metric.
function V = initialize_frame(X, N, K, method, L, M, U_spectral, Lambda_inv)
    global S74;
    n_dof = 2 * N;

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
        Mv = apply_block_jacobi_to_vector(X, v, N, K);  % M * v
        norm_M = sqrt(v' * Mv);
        v = v / norm_M;

    elseif strcmp(method, 'ic')
        [v, ~] = eigs(H, M, 1, 'smallestreal');
    t74_apply = tic;
        Mv = M * v;
    r1_apply_elapsed = toc(t74_apply);
    S74.apply = S74.apply + r1_apply_elapsed;
    S74.r1_M_apply_solve = S74.r1_M_apply_solve + r1_apply_elapsed;
    S74.apply_calls = S74.apply_calls + 1;
        norm_M = sqrt(v' * Mv);
        v = v / norm_M;

    else  % frozen_spectral
    t74_fs_matrix_reconstruction_time = tic;
        lambda_metric = 1.0 ./ diag(Lambda_inv);
        M_frozen = U_spectral * diag(lambda_metric) * U_spectral';
        M_frozen = 0.5 * (M_frozen + M_frozen');
    t74_elapsed = toc(t74_fs_matrix_reconstruction_time);
    S74.fs_matrix_reconstruction_time = S74.fs_matrix_reconstruction_time + t74_elapsed;
    S74.r1_M_setup_upd = S74.r1_M_setup_upd + t74_elapsed;
    if S74.initializing
        S74.setup = S74.setup + t74_elapsed;
    end
    S74.fs_matrix_reconstructions = S74.fs_matrix_reconstructions + 1;

        [vecs, vals] = eig(H, M_frozen);
        [~, idx] = sort(diag(vals));
        v = vecs(:, idx(1));

    t74_apply = tic;
        u_proj = U_spectral' * v;
    r1_apply_elapsed = toc(t74_apply);
    S74.apply = S74.apply + r1_apply_elapsed;
    S74.r1_M_apply_solve = S74.r1_M_apply_solve + r1_apply_elapsed;
    S74.apply_calls = S74.apply_calls + 1;

        M_norm_sq = sum( (u_proj.^2) .* lambda_metric );
        norm_M = sqrt(M_norm_sq);

        if norm_M > 1e-10
            v = v / norm_M;
        end
    end

    V = {v};
end

% Apply the fixed spectral inverse to H v; the projection coefficient is v' H v.
function V_new = update_frame_frozen_spectral(X, V, N, K, dt_v, v_iter, U_spectral, Lambda_inv)
    global S74;
    n_dof = 2 * N;

    H = full(compute_hessian(X, N, K));

    v = V{1};

    for iter_v = 1:v_iter
        Gv = H * v;

    t74_apply = tic;
        Ut_Gv = U_spectral' * Gv;
        Lambda_inv_Ut_Gv = Lambda_inv * Ut_Gv;
        Minv_Gv = U_spectral * Lambda_inv_Ut_Gv;
    r1_apply_elapsed = toc(t74_apply);
    S74.solve = S74.solve + r1_apply_elapsed;
    S74.r1_M_apply_solve = S74.r1_M_apply_solve + r1_apply_elapsed;
    S74.solve_calls = S74.solve_calls + 1;

        coeff = v' * Gv;  % v^T Hv (Rayleigh Quotient)
        projected = Minv_Gv - coeff * v;

        v = v - dt_v * projected;

    t74_fs_weight_restore_time = tic;
        lambda_metric = 1.0 ./ diag(Lambda_inv);
    t74_elapsed = toc(t74_fs_weight_restore_time);
    S74.fs_weight_restore_time = S74.fs_weight_restore_time + t74_elapsed;
    S74.r1_M_setup_upd = S74.r1_M_setup_upd + t74_elapsed;

    t74_apply = tic;
        u_proj = U_spectral' * v;
    r1_apply_elapsed = toc(t74_apply);
    S74.apply = S74.apply + r1_apply_elapsed;
    S74.r1_M_apply_solve = S74.r1_M_apply_solve + r1_apply_elapsed;
    S74.apply_calls = S74.apply_calls + 1;

        M_norm_sq = sum( (u_proj.^2) .* lambda_metric );
        norm_M = sqrt(M_norm_sq);

        if norm_M > 1e-10
            v = v / norm_M;
        end
    end

    V_new = {v};
end

% Use finite-difference Hessian actions and Euclidean frame normalization.
function V_new = update_frame_standard(X, V, N, K, dt_v, v_iter)
    global S74;
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

    for iter_v = 1:v_iter
        Gv = H * v;
        coeff = v' * Gv;
        projected = Gv - coeff * v;
        v = v - dt_v * projected;
        v = v / norm(v);
    end

    V_new = {v};
end

% Use the state-dependent 2-by-2 block metric for frame updates and normalization.
function V_new = update_frame_block_jacobi(X, V, N, K, dt_v, v_iter)
    global S74;
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

    for iter_v = 1:v_iter
        Gv = H * v;
        Minv_Gv = apply_block_jacobi_inverse(X, Gv, N, K);  % M^{-1}Hv
        coeff = v' * Gv;  % Rayleigh coefficient v^T H v in the metric frame update.
        projected = Minv_Gv - coeff * v;
        v = v - dt_v * projected;  % dir_v = -M^{-1}Hv + (v^T Hv)v
        norm_M = sqrt(m_inner_product_bj(X, v, v, N, K));
        if norm_M > 1e-10
            v = v / norm_M;
        end
    end

    V_new = {v};
end

% Reuse the supplied IC factor during the frame sweeps at this state.
function V_new = update_frame_ic(X, V, N, K, dt_v, v_iter, L, M)
    global S74;
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

    for iter_v = 1:v_iter
        Gv = H * v;
    t74_apply = tic;
        Minv_Gv = L' \ (L \ Gv);
    r1_apply_elapsed = toc(t74_apply);
    S74.solve = S74.solve + r1_apply_elapsed;
    S74.r1_M_apply_solve = S74.r1_M_apply_solve + r1_apply_elapsed;
    S74.solve_calls = S74.solve_calls + 1;
        coeff = v' * Gv;  % Correct: v^T H v (Rayleigh quotient)
        projected = Minv_Gv - coeff * v;
        v = v - dt_v * projected;
    t74_apply = tic;
        Mv = M * v;
    r1_apply_elapsed = toc(t74_apply);
    S74.apply = S74.apply + r1_apply_elapsed;
    S74.r1_M_apply_solve = S74.r1_M_apply_solve + r1_apply_elapsed;
    S74.apply_calls = S74.apply_calls + 1;
        norm_M = sqrt(v' * Mv);
        v = v / norm_M;
    end

    V_new = {v};
end

% Assemble the analytical Hessian from local molecular blocks and neighbor coupling.
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

        H_xx = K + 12*x(i)^2 - 4;
        if i > 1
            H_xx = H_xx + 1;  % From the (y_{i-1} - x_i)² term
        end

        H_yy = K + 12*y(i)^2 - 4;
        if i < N
            H_yy = H_yy + 1;  % From the (y_i - x_{i+1})² term
        end

        H_xy = -K;

        I_idx = [I_idx; idx_x; idx_y; idx_x; idx_y];
        J_idx = [J_idx; idx_x; idx_y; idx_y; idx_x];
        V_vals = [V_vals; H_xx; H_yy; H_xy; H_xy];

        if i < N
            idx_x_next = 2*(i+1) - 1;
            I_idx = [I_idx; idx_y; idx_x_next];
            J_idx = [J_idx; idx_x_next; idx_y];
            V_vals = [V_vals; -1; -1];
        end
    end

    H = sparse(I_idx, J_idx, V_vals, n_dof, n_dof);
end

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

function [H_xx, H_yy, H_xy] = get_local_blocks(X, N, K)
    global S74;
    t74_blocks = tic;
    x = X(1:2:end);
    y = X(2:2:end);

    damping = 1.0;
    curvature_x = 12 * x.^2 - 4;
    curvature_y = 12 * y.^2 - 4;

    H_xx = K + abs(curvature_x) + damping;
    H_yy = K + abs(curvature_y) + damping;
    H_xy = -K * ones(N, 1);
    t74_elapsed = toc(t74_blocks);
    S74.setup_bj_blocks = S74.setup_bj_blocks + t74_elapsed;
    S74.bj_block_builds = S74.bj_block_builds + 1;
    if S74.setup_depth == 0
        S74.r1_M_setup_upd = S74.r1_M_setup_upd + t74_elapsed;
        if S74.initializing
            S74.setup = S74.setup + t74_elapsed;
        else
            S74.update = S74.update + t74_elapsed;
        end
    end
end

% Apply independent 2-by-2 inverse blocks without a global matrix solve.
function result = apply_block_jacobi_inverse(X, vector, N, K)
    global S74;
    vec_x = vector(1:2:end);
    vec_y = vector(2:2:end);

    [H_xx, H_yy, H_xy] = get_local_blocks(X, N, K);

    t74_setup_bj_inverse_coefficients = tic;
    det = H_xx .* H_yy - H_xy.^2;
    det = max(det, 1e-8);
    inv_det = 1.0 ./ det;
    t74_elapsed = toc(t74_setup_bj_inverse_coefficients);
    S74.setup_bj_inverse_coefficients = S74.setup_bj_inverse_coefficients + t74_elapsed;
    S74.r1_M_setup_upd = S74.r1_M_setup_upd + t74_elapsed;
    if S74.initializing
        S74.setup = S74.setup + t74_elapsed;
    else
        S74.update = S74.update + t74_elapsed;
    end

    t74_apply = tic;
    res_x = inv_det .* (H_yy .* vec_x - H_xy .* vec_y);
    res_y = inv_det .* (-H_xy .* vec_x + H_xx .* vec_y);

    result = zeros(2*N, 1);
    result(1:2:end) = res_x;
    result(2:2:end) = res_y;
    r1_apply_elapsed = toc(t74_apply);
    S74.solve = S74.solve + r1_apply_elapsed;
    S74.r1_M_apply_solve = S74.r1_M_apply_solve + r1_apply_elapsed;
    S74.solve_calls = S74.solve_calls + 1;
end

% Apply M itself for the metric inner product and frame normalization.
function result = apply_block_jacobi_to_vector(X, vector, N, K)
    global S74;
    vec_x = vector(1:2:end);
    vec_y = vector(2:2:end);

    [H_xx, H_yy, H_xy] = get_local_blocks(X, N, K);

    t74_apply = tic;
    res_x = H_xx .* vec_x + H_xy .* vec_y;
    res_y = H_xy .* vec_x + H_yy .* vec_y;

    result = zeros(2*N, 1);
    result(1:2:end) = res_x;
    result(2:2:end) = res_y;
    r1_apply_elapsed = toc(t74_apply);
    S74.apply = S74.apply + r1_apply_elapsed;
    S74.r1_M_apply_solve = S74.r1_M_apply_solve + r1_apply_elapsed;
    S74.apply_calls = S74.apply_calls + 1;
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
    global S74;
    r1_bj_apply_before = S74.r1_M_apply_solve;
    t74_bj_matrix = tic;
    S74.setup_depth = S74.setup_depth + 1;
    n_dof = 2 * N;
    M = zeros(n_dof, n_dof);

    for i = 1:n_dof
        ei = zeros(n_dof, 1);
        ei(i) = 1.0;
        M(:, i) = apply_block_jacobi_to_vector(X, ei, N, K);
    end

    M = 0.5 * (M + M');
    t74_elapsed = toc(t74_bj_matrix);
    S74.setup_bj_matrix = S74.setup_bj_matrix + t74_elapsed;
    r1_bj_apply_inside = S74.r1_M_apply_solve - r1_bj_apply_before;
    r1_bj_setup_net = t74_elapsed - r1_bj_apply_inside;
    S74.r1_M_setup_upd = S74.r1_M_setup_upd + r1_bj_setup_net;
    S74.r1_bj_dense_inclusive = S74.r1_bj_dense_inclusive + t74_elapsed;
    S74.r1_bj_dense_apply_excluded = S74.r1_bj_dense_apply_excluded + r1_bj_apply_inside;
    S74.r1_bj_dense_setup_net = S74.r1_bj_dense_setup_net + r1_bj_setup_net;
    S74.r1_min_bj_dense_net = min(S74.r1_min_bj_dense_net, r1_bj_setup_net);
    if S74.initializing
        S74.setup = S74.setup + t74_elapsed;
    else
        S74.update = S74.update + t74_elapsed;
    end
    S74.setup_depth = S74.setup_depth - 1;
    S74.bj_matrix_builds = S74.bj_matrix_builds + 1;
end

function result = m_inner_product_bj(X, u, v, N, K)
    Mv = apply_block_jacobi_to_vector(X, v, N, K);  % M * v (not M^{-1} * v)
    result = u' * Mv;
end

% Factor a shifted sparse curvature surrogate, increasing the shift on failure.
% If all IC attempts fail, use the diagonal positive fallback below.
function [L, M, info] = build_shifted_ic_preconditioner(X, N, K)
    global S74;
    S74.ic_builds = S74.ic_builds + 1;
    n_dof = 2 * N;

    info = struct();
    info.failed = false;
    info.shift_trials = 0;
    info.final_delta = 0;
    info.ordering = 'none';

    t74_legacy_ic = tic;

    A = build_base_matrix(X, N, K);
    info.nnz_A = nnz(A);

    delta_init = 1e-6;           % Initial shift
    delta_max = 1e3;             % Maximum shift (prevent over-regularization)
    max_shift_trials = 10;       % Maximum number of trials
    shift_growth_factor = 10;    % Shift growth factor

    delta = delta_init;

    ic_opts.type = 'ict';
    ic_opts.droptol = 1e-5;      % Drop tolerance (controls fill-in)

    try
        perm = amd(A);
        info.ordering = 'AMD';
    catch
        perm = 1:n_dof;
        info.ordering = 'none';
    end

    A_perm = A(perm, perm);

    L = [];
    ic_success = false;

    for trial = 1:max_shift_trials
        info.shift_trials = trial;

        A_delta = A_perm + delta * speye(n_dof);

        try
    t74_factor = tic;
    S74.ic_factorizations = S74.ic_factorizations + 1;
            L_perm = ichol(A_delta, ic_opts);
    S74.ic_factorization_time = S74.ic_factorization_time + toc(t74_factor);
    t74_factor = [];

            if any(~isfinite(L_perm(:))) || any(diag(L_perm) <= 0)
                error('IC produced invalid entries');
            end

            ic_success = true;
            info.final_delta = delta;

            L = L_perm(invperm(perm), :);

            break;

        catch ME
    if ~isempty(t74_factor)
        S74.ic_factorization_time = S74.ic_factorization_time + toc(t74_factor);
    end
    S74.ic_failures = S74.ic_failures + 1;
            if trial < max_shift_trials
    if ~S74.quiet
                fprintf('  IC failed (trial %d, δ=%.2e): %s\n', trial, delta, ME.message);
    end
    if ~S74.quiet
                fprintf('  Increasing the shift...\n');
    end
                delta = delta * shift_growth_factor;

                if delta > delta_max
    if ~S74.quiet
                    fprintf('  Warning: shift exceeded the upper limit (%.2e); stopping trials\n', delta_max);
    end
                    break;
                end
            else
    if ~S74.quiet
                fprintf('  IC failed: maximum number of trials reached\n');
    end
            end
        end
    end

    if ~ic_success
    if ~S74.quiet
        fprintf('  Using fail-safe: Block-Jacobi preconditioner\n');
    end
        info.failed = true;

        [L, M] = build_failsafe_preconditioner(X, N, K);
        info.nnz_L = nnz(L);
        info.fill_ratio = info.nnz_L / info.nnz_A;
        info.factor_time = toc(t74_legacy_ic);
        return;
    end

    % The metric is the product of the computed factors, including IC approximation.
    M = L * L';

    info.nnz_L = nnz(L);
    info.fill_ratio = info.nnz_L / info.nnz_A;
    info.factor_time = toc(t74_legacy_ic);

    diag_A = abs(diag(A));
    relative_shift = info.final_delta / max(diag_A);
    if relative_shift > 0.1
    if ~S74.quiet
        fprintf('  Warning: large relative shift (%.2f%%); possible over-regularization\n', relative_shift * 100);
    end
    if ~S74.quiet
        fprintf('  Suggestion: increase the fill-in budget or improve the ordering\n');
    end
    end
end

% Combine positive local curvature blocks with sparse neighbor coupling.
function A = build_base_matrix(X, N, K)
    n_dof = 2 * N;
    x = X(1:2:end);
    y = X(2:2:end);

    damping = 1.0;
    curvature_x = 12 * x.^2 - 4;
    curvature_y = 12 * y.^2 - 4;

    H_xx = K + abs(curvature_x) + damping;
    H_yy = K + abs(curvature_y) + damping;
    H_xy = -K * ones(N, 1);

    for i = 1:N
        discriminant = (H_xx(i) - H_yy(i))^2 + 4 * H_xy(i)^2;
        lambda_min = (H_xx(i) + H_yy(i) - sqrt(discriminant)) / 2;

        if lambda_min <= 0
            H_xy(i) = 0.9 * sqrt(H_xx(i) * H_yy(i));
        end
    end

    I_idx = [];
    J_idx = [];
    V_vals = [];

    for i = 1:N
        idx_x = 2*i - 1;
        idx_y = 2*i;

        I_idx = [I_idx; idx_x];
        J_idx = [J_idx; idx_x];
        V_vals = [V_vals; H_xx(i)];

        I_idx = [I_idx; idx_x];
        J_idx = [J_idx; idx_y];
        V_vals = [V_vals; H_xy(i)];

        I_idx = [I_idx; idx_y];
        J_idx = [J_idx; idx_x];
        V_vals = [V_vals; H_xy(i)];

        I_idx = [I_idx; idx_y];
        J_idx = [J_idx; idx_y];
        V_vals = [V_vals; H_yy(i)];
    end

    coupling_strength = 0.5;  % Weak coupling
    for i = 1:N-1
        idx_y_i = 2*i;
        idx_x_ip1 = 2*(i+1) - 1;

        I_idx = [I_idx; idx_y_i];
        J_idx = [J_idx; idx_x_ip1];
        V_vals = [V_vals; -coupling_strength];

        I_idx = [I_idx; idx_x_ip1];
        J_idx = [J_idx; idx_y_i];
        V_vals = [V_vals; -coupling_strength];
    end

    A = sparse(I_idx, J_idx, V_vals, n_dof, n_dof);

    A = 0.5 * (A + A');
end

% Use positive diagonal local curvatures when incomplete factorization is unavailable.
function [L, M] = build_failsafe_preconditioner(X, N, K)
    n_dof = 2 * N;
    x = X(1:2:end);
    y = X(2:2:end);

    damping = 10.0;  % Larger damping ensures stability
    curvature_x = 12 * x.^2 - 4;
    curvature_y = 12 * y.^2 - 4;

    H_xx = K + abs(curvature_x) + damping;
    H_yy = K + abs(curvature_y) + damping;

    diag_vals = zeros(n_dof, 1);
    diag_vals(1:2:end) = H_xx;
    diag_vals(2:2:end) = H_yy;

    M = spdiags(diag_vals, 0, n_dof, n_dof);
    L = spdiags(sqrt(diag_vals), 0, n_dof, n_dof);
end

function iperm = invperm(perm)
    n = length(perm);
    iperm = zeros(1, n);
    iperm(perm) = 1:n;
end

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

function plot_comparison_four(h1, h2, h3, h4, dt1, dt2, dt3, dt4)
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

    grid on;
    set(gca, 'GridAlpha', 0.2);
    set(gca, 'MinorGridAlpha', 0.2);

    ax = gca;
    ax.YAxis.MinorTick = 'off';  % Disable minor ticks

    legend('Location', 'best', 'FontSize', 16);

    set(gca, 'LooseInset', get(gca, 'TightInset'));

    exportgraphics(gcf, '3.pdf', 'ContentType', 'vector', 'BackgroundColor', 'none');
    fprintf('\nFigure saved: 3.pdf\n');
end

function definitions = timing74_definitions()
    definitions = struct();
    definitions.unit = 'seconds';
    definitions.T_total = ['Solver entry before initialization/setup through solver exit, ' ...
        'before history packaging and endpoint certification. Includes all original solver ' ...
        'operations and instrumentation overhead; solver printing is disabled. Excludes ' ...
        'common initial-state generation, repetition checks, summary, plotting and file I/O.'];
    definitions.T_setup = ['Initial IC construction including factorization; Frozen Spectral ' ...
        'initial Hessian/spectrum/weights and first dense metric materialization; BJ initial ' ...
        'dense metric assembly and local blocks needed by initial frame normalization.'];
    definitions.T_update = ['IC state and frame rebuilds; BJ local blocks, inverse coefficients ' ...
        'and dense assemblies after initial frame completion. Counts actual executed rebuild ' ...
        'work, including repeated construction at the same state. Standard and once-frozen ' ...
        'Frozen Spectral have zero update time.'];
    definitions.T_apply = ['Explicit M*v, BJ forward block multiplication and output assembly, ' ...
        'and Frozen Spectral U-transpose projections used for metric normalization. ' ...
        'Spectral weighted reductions/sqrt remain within T_eig; BJ block setup is excluded.'];
    definitions.T_solve = ['Original IC nested triangular backslash solves, BJ inverse block ' ...
        'multiplication and output assembly after coefficients are built, and the original ' ...
        'three-stage Frozen Spectral inverse application.'];
    definitions.T_setup_update = 'T_setup + T_update.';
    definitions.T_apply_solve = 'T_apply + T_solve.';
    definitions.T_eig = ['Inclusive initial frame, all full frame updates, every 50-step ' ...
        'reinitialization, IC state metric normalization, and initial Frozen Spectral metric ' ...
        'eigendecomposition. Includes original Hessian work and internal normalization/' ...
        'orthogonalization. Generalized eig/eigs internal metric operations are not ' ...
        'separately observable in T_apply or T_solve.'];
    definitions.T_certify = ['Post-solver gradient residual, analytical Hessian, full symmetric spectrum ' ...
        'and fixed-absolute-threshold Morse-index classification; excluded from legacy T_total.'];
    definitions.legacy_outer_time = ['Diagnostic original outer-loop timing boundary; ' ...
        'history.total_time always equals history.timing.T_total.'];
    definitions.frozen_metric = ['Frozen Spectral dense rematerialization and weight restoration ' ...
        'do not refresh the metric. They retain diagnostic counters and remain in inclusive ' ...
        'T_eig; only the first dense materialization also belongs to initial T_setup.'];
    definitions.accounting = ['Component times are diagnostic counters and may overlap when ' ...
        'preconditioner applications/solves occur inside eigenspace computations. Initial ' ...
        'metric spectral work and setup inside frame computations also overlap; BJ dense ' ...
        'matrix setup contains forward applications. BJ setup_depth excludes nested local ' ...
        'block setup from its setup aggregate. Do not sum components to obtain T_total.'];
end

function summary = timing74_summarize(raw)
    metrics = {'T_total', 'T_setup', 'T_update', 'T_setup_update', ...
        'T_apply', 'T_solve', 'T_apply_solve', 'T_eig', 'T_certify'};
    summary = struct('schema', raw.schema, 'unit', 'seconds', 'repeats', raw.repeats, ...
        'T_total_includes_certification', false, 'component_times_may_overlap', true);
    summary.definitions = raw.definitions;
    summary.run_policy = raw.run_policy;
    summary.environment = raw.environment;
    summary.numerical_repetitions_identical = raw.numerical_repetitions_identical;
    rows = cell(4 * numel(metrics), 10);
    row = 0;
    for method = 1:4
        for metric = 1:numel(metrics)
            values = cellfun(@(t) t.(metrics{metric}), raw.timings(:, method));
            stats = struct('median', median(values), 'min', min(values), 'max', max(values));
            summary.statistics.(raw.methods{method}).(metrics{metric}) = stats;
            row = row + 1;
            rows(row, :) = {raw.methods{method}, metrics{metric}, 'seconds', ...
                stats.median, stats.min, stats.max, raw.repeats, false, true, ...
                raw.definitions.accounting};
        end
    end
    summary.table = cell2table(rows, 'VariableNames', {'Method', 'Metric', 'Unit', ...
        'Median', 'Min', 'Max', 'Repeats', 'T_total_includes_certification', ...
        'Component_times_may_overlap', 'Accounting_note'});
end


% Subtract separately measured metric work from the inclusive frame interval.
function r1_account_frame(r1_inclusive, r1_setup_before, r1_apply_before)
    global S74;
    r1_setup_inside = S74.r1_M_setup_upd - r1_setup_before;
    r1_apply_inside = S74.r1_M_apply_solve - r1_apply_before;
    r1_net = r1_inclusive - r1_setup_inside - r1_apply_inside;
    S74.r1_eig_net = S74.r1_eig_net + r1_net;
    S74.r1_frame_inclusive = S74.r1_frame_inclusive + r1_inclusive;
    S74.r1_frame_setup_excluded = S74.r1_frame_setup_excluded + r1_setup_inside;
    S74.r1_frame_apply_excluded = S74.r1_frame_apply_excluded + r1_apply_inside;
    S74.r1_frame_regions = S74.r1_frame_regions + 1;
    S74.r1_min_frame_net = min(S74.r1_min_frame_net, r1_net);
end

function definitions = r1_cost_definitions()
    definitions.T_total = ['Solver setup through final residual and Morse-index certification; ' ...
        'includes instrumentation overhead; excludes startup, driver checks, file I/O and plotting.'];
    definitions.T_M_setup_upd = ['Actual metric construction, repeated construction, shift/factorization; ' ...
        'FS full eig and fixed-metric representation work; BJ dense setup excludes nested explicit applications.'];
    definitions.T_M_apply_solve = ['All observable explicit metric/inverse applications in state and frame work, ' ...
        'including forward applications inside BJ dense assembly.'];
    definitions.T_eig_net = ['Frame initialization/updates/reinitialization/normalization, including their Hessian work, ' ...
        'minus separately measured metric work inside the same region; generalized eig/eigs internal costs remain inside. ' ...
        'Endpoint certification is excluded.'];
    definitions.T_certify = ['Final residual, analytical Hessian, full symmetric eig and Morse-index classification; ' ...
        'included in this T_total, excluded from the three metric/frame components and legacy T_total.'];
    definitions.implementation = ['Standard/BJ/IC retain finite-difference frame Hessians; FS retains analytical frame Hessians. ' ...
        'IC retains dynamic surrogate construction and repeated factorization. All methods certify target index 1 ' ...
        'using the original Hessian and raw.parameters.delta_ind as a fixed absolute threshold.'];
end

function reason = r1_cost_stop_reason(history, max_iter)
    last_residual = history.grad_norm(end);
    if history.converged
        reason = 'converged';
    elseif ~isfinite(last_residual) || last_residual > 1e8 || ...
            ~isfinite(history.endpoint_residual) || history.endpoint_residual > 1e8
        reason = 'diverged';
    elseif history.endpoint_residual_pass
        reason = history.endpoint_status;
    elseif history.timing.frame_updates == max_iter
        reason = 'budget_exhausted';
    else
        reason = 'stopped';
    end
end

function [runs, summary, variability] = r1_make_cost_tables(raw)
    labels = {'HiSD', 'Block Jacobi', 'Incomplete Cholesky', 'Frozen Spectral'};
    keys = {'method', 'repeat', 'N', 'eta', 'tau', 'J', 'max_iter', ...
        'history_records', 'actual_outer_updates', 'stop_reason', 'endpoint_residual', ...
        'T_total', 'T_M_setup_upd', 'T_M_apply_solve', 'T_eig_net', ...
        'metric_build_count', 'metric_build_count_definition', 'factorization_count', ...
        'M_solve_count', 'M_apply_count', 'bj_dense_build_count', ...
        'ic_state_build_count', 'ic_frame_build_count', 'ic_failure_count', ...
        'fs_dense_rematerialization_count', 'frame_initialization_count', ...
        'target_index', 'delta_ind', 'morse_index', 'index_resolved', 'index_pass', ...
        'lambda_1', 'lambda_2', 'endpoint_status', 'T_certify'};
    data = cell(raw.repeats * 4, numel(keys));
    metric_labels = {'none', 'local block helper calls including dense assembly', ...
        'complete IC builder calls', 'frozen spectrum builds'};
    row = 0;
    for method = 1:4
        for repeat = 1:raw.repeats
            row = row + 1;
            h = raw.histories{repeat, method};
            t = h.timing;
            times = [t.r1_T_total, t.r1_M_setup_upd, t.r1_M_apply_solve, t.r1_eig_net, t.T_certify];
            assert(all(isfinite(times)) && all(times >= 0), ...
                'run74:TimingValues', 'Non-finite or negative timing component.');
            assert(sum(times(2:end)) <= times(1) + 1e-12, ...
                'run74:TimingTotal', 'Component sum exceeds the total.');
            assert(t.r1_T_total_includes_endpoint && ~t.r1_components_overlap);
            assert(t.r1_min_frame_net >= 0);
            assert(abs(t.r1_frame_inclusive - t.r1_frame_setup_excluded - ...
                t.r1_frame_apply_excluded - t.r1_eig_net) <= ...
                1e-12 * max(1, t.r1_frame_inclusive));
            if method == 2
                assert(t.r1_min_bj_dense_net >= 0);
                assert(abs(t.r1_bj_dense_inclusive - t.r1_bj_dense_apply_excluded - ...
                    t.r1_bj_dense_setup_net) <= 1e-12 * max(1, t.r1_bj_dense_inclusive));
            end
            counts = [0, t.bj_block_builds, t.ic_builds, t.fs_spectral_builds];
            data(row, :) = {labels{method}, repeat, raw.parameters.N, ...
                raw.parameters.dt_x(method), raw.parameters.dt_v(method), raw.parameters.v_iter, ...
                raw.parameters.max_iter, numel(h.grad_norm), t.frame_updates, ...
                r1_cost_stop_reason(h, raw.parameters.max_iter), h.endpoint_residual, ...
                t.r1_T_total, t.r1_M_setup_upd, t.r1_M_apply_solve, t.r1_eig_net, ...
                counts(method), metric_labels{method}, t.ic_factorizations, t.solve_calls, ...
                t.apply_calls, t.bj_matrix_builds, t.ic_state_builds, t.ic_frame_builds, ...
                t.ic_failures, t.fs_matrix_reconstructions, t.frame_initializations, ...
                h.target_index, h.delta_ind, h.morse_index, h.index_resolved, h.index_pass, ...
                h.lambda_1, h.lambda_2, h.endpoint_status, t.T_certify};
        end
    end
    runs = cell2table(data, 'VariableNames', keys);
    summary_keys = {'method', 'N', 'eta', 'tau', 'J', 'max_iter', 'history_records', ...
        'actual_outer_updates', 'stop_reason', 'endpoint_residual', 'repeats', ...
        'T_total', 'T_M_setup_upd', 'T_M_apply_solve', 'T_eig_net', ...
        'target_index', 'delta_ind', 'morse_index', 'index_resolved', 'index_pass', ...
        'lambda_1', 'lambda_2', 'endpoint_status', 'T_certify'};
    summary_data = cell(4, numel(summary_keys));
    time_keys = {'T_total', 'T_M_setup_upd', 'T_M_apply_solve', 'T_eig_net', 'T_certify'};
    variability = cell(4, numel(time_keys));
    for method = 1:4
        indices = (method-1)*raw.repeats + (1:raw.repeats);
        medians = zeros(1, numel(time_keys));
        for metric = 1:numel(time_keys)
            time_column = strcmp(keys, time_keys{metric});
            values = cell2mat(data(indices, time_column));
            medians(metric) = median(values);
            variability{method, metric} = struct('method', labels{method}, ...
                'quantity', time_keys{metric}, 'min', min(values), 'max', max(values), ...
                'median', median(values), 'mean', mean(values), 'sample_std', std(values), ...
                'CV', std(values) / max(mean(values), realmin));
        end
        h = raw.histories{1, method};
        summary_data(method, :) = {labels{method}, raw.parameters.N, ...
            raw.parameters.dt_x(method), raw.parameters.dt_v(method), raw.parameters.v_iter, ...
            raw.parameters.max_iter, numel(h.grad_norm), h.timing.frame_updates, ...
            r1_cost_stop_reason(h, raw.parameters.max_iter), h.endpoint_residual, raw.repeats, ...
            medians(1), medians(2), medians(3), medians(4), ...
            h.target_index, h.delta_ind, h.morse_index, h.index_resolved, h.index_pass, ...
            h.lambda_1, h.lambda_2, h.endpoint_status, medians(5)};
    end
    summary = cell2table(summary_data, 'VariableNames', summary_keys);
end

function r1_write_cost_csv(table_data, path)
    [fid, message] = fopen(path, 'w');
    if fid < 0
        error('run74:CSVWrite', '%s', message);
    end
    cleanup = onCleanup(@() fclose(fid));
    headers = table_data.Properties.VariableNames;
    fprintf(fid, '%s\n', strjoin(headers, ','));
    for row = 1:height(table_data)
        text = cell(1, width(table_data));
        for col = 1:width(table_data)
            value = table_data{row, col};
            if iscell(value)
                value = value{1};
            end
            if isnumeric(value)
                text{col} = sprintf('%.17g', value);
            elseif islogical(value)
                text{col} = sprintf('%d', value);
            else
                text{col} = ['"' strrep(char(value), '"', '""') '"'];
            end
        end
        fprintf(fid, '%s\n', strjoin(text, ','));
    end
end

