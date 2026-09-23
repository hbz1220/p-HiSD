% Section 7.5.1: index-3 saddles of the stiff semilinear Dirichlet problem.
% Compare HiSD with four metrics; save histories and timings to outputs/7.5/7.5.1/.
% The companion fig.py reads the histories to produce Figure 6.

semilinear_elliptic_phisd();

function semilinear_elliptic_phisd()
    p = default_params();
    script_dir = fileparts(mfilename('fullpath'));
    result_dir = fullfile(script_dir, '..', '..', '..', ...
                          'outputs', '7.5', '7.5.1');
    if ~exist(result_dir, 'dir')
        mkdir(result_dir);
    end

    [A, h, x] = build_laplacian_dirichlet(p.N);
    u0 = sin(4 * x);

    % Repeat identical initial data and parameters; retain the first numerical history.
    timing_repeats = 3;
    timing_raw = struct();
    timing_raw.schema = 'section7.5.1.timing.v2';
    timing_raw.methods = {'standard', 'spectral', 'block-jacobi', 'shifted-ic', 'h1-reaction'};
    timing_raw.repeats = timing_repeats;
    timing_raw.retained_experiment_repetition = 1;
    timing_raw.params = p;
    timing_raw.A = A; timing_raw.h = h; timing_raw.x = x; timing_raw.u0 = u0;
    timing_raw.rng_initial = rng;
    timing_raw.rng_final = cell(timing_repeats, 1);
    timing_raw.runs = cell(timing_repeats, numel(timing_raw.methods));
    timing_raw.definitions = timing751_definitions();
    timing_raw.T_total_includes_certification = false;
    timing_raw.component_times_are_inclusive = true;
    timing_raw.run_policy = ['Complete five-method repetitions in one MATLAB session; ' ...
        'same problem, initial condition, parameters and initial RNG state; original ' ...
        'method order; no warm-up solve or tuning; repetition 1 supplies original outputs.'];
    timing_raw.unsuccessful_run_policy = ['For methods without status success, T_total is ' ...
        'wall-clock time consumed until termination, not time-to-solution.'];
    timing_raw.environment = struct('MATLAB_version', version, 'MATLAB_release', version('-release'), ...
        'architecture', computer, 'max_threads', maxNumCompThreads, ...
        'BLAS', version('-blas'), 'LAPACK', version('-lapack'));
    for timing_repeat = 1:timing_repeats
        rng(timing_raw.rng_initial);
        fprintf('Section 7.5.1 timing repetition %d/%d\n', timing_repeat, timing_repeats);

    runs = {
        run_standard(A, h, u0, p)
        run_phisd('spectral', 'spectral', p.ETA_AFTER_PHISD, A, h, u0, p)
        run_phisd('block-jacobi', 'block-jacobi', p.ETA_AFTER_BLOCK_JACOBI, A, h, u0, p)
        run_phisd('shifted-ic', 'shifted-ic', p.ETA_AFTER_PHISD, A, h, u0, p)
        run_phisd('h1-reaction', 'h1-reaction', p.ETA_AFTER_PHISD, A, h, u0, p)
    };

        for timing_method = 1:numel(runs)
            assert(strcmp(runs{timing_method}.method, timing_raw.methods{timing_method}), ...
                'Section 7.5.1 method order changed.');
            runs{timing_method}.timing.T_total_includes_certification = false;
            runs{timing_method}.timing.component_times_are_inclusive = true;
            timing_raw.runs{timing_repeat, timing_method} = runs{timing_method};
            if timing_repeat > 1
                assert(isequaln(timing751_numerical_run(runs{timing_method}), ...
                    timing751_numerical_run(timing_raw.runs{1, timing_method})), ...
                    'STOP: Section 7.5.1 non-timing repetition mismatch: %s, repetition %d.', ...
                    timing_raw.methods{timing_method}, timing_repeat);
            end
        end
        timing_raw.rng_final{timing_repeat} = rng;
        assert(isequaln(timing_raw.rng_final{1}, timing_raw.rng_final{timing_repeat}), ...
            'STOP: Section 7.5.1 repetition RNG mismatch.');
    end
    timing_raw.non_timing_repetitions_exact = true;
    runs = timing_raw.runs(1, :).';

    [summary, histories] = save_outputs(runs, result_dir);

    N = p.N; K = p.K; TAU = p.TAU; J = p.J; TOL = p.TOL;
    REFREEZE_THRESHOLD = p.REFREEZE_THRESHOLD;
    MAX_ITER_STANDARD = p.MAX_ITER_STANDARD; MAX_ITER_PHISD = p.MAX_ITER_PHISD;
    ETA_STANDARD = p.ETA_STANDARD; ETA_BEFORE_REFREEZE = p.ETA_BEFORE_REFREEZE;
    ETA_AFTER_PHISD = p.ETA_AFTER_PHISD; ETA_AFTER_BLOCK_JACOBI = p.ETA_AFTER_BLOCK_JACOBI;
    EPS_SPEC = p.EPS_SPEC; EPS_REACTION = p.EPS_REACTION; SHIFT_ALPHA = p.SHIFT_ALPHA;
    BLOCK_SIZE = p.BLOCK_SIZE; BLOCK_REG_PARAM = p.BLOCK_REG_PARAM;

    history_standard = histories.standard;
    history_spectral = histories.spectral;
    history_block_jacobi = histories.block_jacobi;
    history_shifted_ic = histories.shifted_ic;
    history_h1_reaction = histories.h1_reaction;

    params = p;
    save(fullfile(result_dir, 'semilinear_results.mat'), ...
        'runs', 'summary', ...
        'history_standard', 'history_spectral', 'history_block_jacobi', 'history_shifted_ic', 'history_h1_reaction', ...
        'params', 'N', 'K', 'h', 'TAU', 'J', 'TOL', ...
        'REFREEZE_THRESHOLD', 'MAX_ITER_STANDARD', 'MAX_ITER_PHISD', ...
        'ETA_STANDARD', 'ETA_BEFORE_REFREEZE', 'ETA_AFTER_PHISD', 'ETA_AFTER_BLOCK_JACOBI', ...
        'EPS_SPEC', 'EPS_REACTION', 'SHIFT_ALPHA', 'BLOCK_SIZE', 'BLOCK_REG_PARAM');

    fprintf('method          status            iter      final_res      refreeze_iter    eta_after    index\n');
    for i = 1:numel(runs)
        r = runs{i};
        fprintf('%-14s%-18s%-10d%-14.3e%-16d%-12.3g%s\n', ...
            r.method, r.status, r.iterations, r.final_residual, r.refreeze_iter, r.eta_after_refreeze, tf_to_str(r.index_k_verified));
    end
    timing_summary = timing751_summarize(timing_raw);
    save(fullfile(result_dir, 'timing751_raw.mat'), 'timing_raw');
    save(fullfile(result_dir, 'timing751_summary.mat'), 'timing_summary');
    writetable(timing_summary.table, fullfile(result_dir, 'timing751_summary.csv'));
    fprintf('\nSection 7.5.1 timing in seconds (median / min / max):\n');
    disp(timing_summary.table(:, {'Method', 'Metric', 'Median', 'Min', 'Max'}));
    fprintf('%s\n', timing_raw.definitions.accounting);
    fprintf('%s\n', timing_raw.unsuccessful_run_policy);
    fprintf('T_cert is excluded from T_total. Original outputs retain repetition 1.\n');
    fprintf('Timing files saved to %s\n', result_dir);
end

% Fixed grid, target index, frame sweeps, step sizes and stopping safeguards.
function p = default_params()
    p = struct(...
        'N', 400, 'K', 3, 'TAU', 0.01, 'J', 5, 'TOL', 1e-6, ...
        'REFREEZE_THRESHOLD', 3e-1, 'INDEX_THRESHOLD', 1e-6, ...
        'MAX_ITER_STANDARD', 2000000, 'MAX_ITER_PHISD', 2000000, 'PLOT_MAX_ITER', 1500, ...
        'ETA_STANDARD', 1e-5, 'ETA_BEFORE_REFREEZE', 0.25, 'ETA_AFTER_PHISD', 1.5, 'ETA_AFTER_BLOCK_JACOBI', 0.9, ...
        'EPS_SPEC', 1e-2, 'EPS_REACTION', 1e-2, 'SHIFT_ALPHA', 1.0, ...
        'BLOCK_SIZE', 64, 'BLOCK_REG_PARAM', 1.0, ...
        'DIVERGENCE_MAX_ABS_U', 20.0, 'DIVERGENCE_RESIDUAL', 1e6, ...
        'ICHOL_DROPTOL', 1e-3, 'ICHOL_MAX_RETRIES', 6);
end

% Second-order difference operator for -u_xx with zero boundary values.
function [A, h, x] = build_laplacian_dirichlet(n)
    h = pi / (n + 1);
    main = (2 / h^2) * ones(n, 1);
    off = (-1 / h^2) * ones(n - 1, 1);
    A = spdiags([[off; 0], main, [0; off]], -1:1, n, n);
    x = h * (1:n)';
end

function y = f_nl(u), y = u.^4 - 10 * u.^2; end
function y = fp_nl(u), y = 4 * u.^3 - 20 * u; end
function g = grad_g(A, u), g = A * u - f_nl(u); end
function H = hess_matrix(A, u), H = A - spdiags(fp_nl(u), 0, numel(u), numel(u)); end
% Use the discrete L2 norm sqrt(h) * ||g|| for convergence and switching.
function r = residual_l2h(A, u, h), r = sqrt(h) * norm(grad_g(A, u)); end

function v0 = eigsh_v0(n)
    v0 = sin((1:n)');
    nrm = norm(v0);
    if nrm > 0, v0 = v0 / nrm; else, v0 = ones(n, 1) / sqrt(n); end
end

function opts = eigs_opts(n)
    opts = struct('v0', eigsh_v0(n));
end

function eig8 = ordinary_eigs8(A, u, p)
    H = hess_matrix(A, u);
    k = min(8, p.N);
    try
        vals = eigs(H, k, 'sa', eigs_opts(p.N));
        vals = sort(real(vals(:)));
    catch
        vals = sort(real(eig(full(H))));
        vals = vals(1:k);
    end
    eig8 = nan(8, 1);
    eig8(1:k) = vals;
end

% Certify index 3 from lambda_3 < -delta and lambda_4 > delta of the ordinary Hessian.
function ok = index_verified(eigs8, p)
    ok = numel(eigs8) >= 4 && isfinite(eigs8(3)) && isfinite(eigs8(4)) && ...
         eigs8(3) < -p.INDEX_THRESHOLD && eigs8(4) > p.INDEX_THRESHOLD;
end

function [V, ok] = orthonormalize_euclidean(V)
    ok = true;
    for i = 1:size(V, 2)
        for j = 1:i-1
            V(:, i) = V(:, i) - V(:, j) * (V(:, j)' * V(:, i));
        end
        nrm = norm(V(:, i));
        if (~isfinite(nrm)) || (nrm <= 1e-14), ok = false; return; end
        V(:, i) = V(:, i) / nrm;
    end
end

% Modified Gram-Schmidt enforces V' M V = I in the current metric.
function [V, ok] = orthonormalize_metric(V, metric)
    ok = true;
    for i = 1:size(V, 2)
        for j = 1:i-1
            V(:, i) = V(:, i) - V(:, j) * (V(:, j)' * metric_apply(metric, V(:, i)));
        end
        nrm2 = V(:, i)' * metric_apply(metric, V(:, i));
        if (~isfinite(nrm2)) || (nrm2 <= 1e-14), ok = false; return; end
        V(:, i) = V(:, i) / sqrt(nrm2);
    end
end

function metric = make_metric(kind)
    metric = struct('name', kind, 'M', [], 'info', struct(), 'Q', [], 'mu', [], 'lam', [], 'blocks', {{}}, 'MB', {{}}, 'R', {{}}, 'L', [], 'dec', []);
end

function [metric, ok] = build_metric(kind, A, u, p)
    metric = make_metric(kind);
    [metric, ok] = rebuild_metric(metric, A, u, p);
end

function [metric, ok] = rebuild_metric(metric, A, u, p)
    global S751;
    ok = true;
    n = numel(u);
    switch metric.name
        % Absolute Hessian spectrum with positive regularization.
        case 'spectral'
            H = full(hess_matrix(A, u)); H = 0.5 * (H + H');
    S751.metric_spectral_decomposition_count = S751.metric_spectral_decomposition_count + 1;
    t751_T_eig_metric = tic;
            [Q, D] = eig(H);
    t751_elapsed = toc(t751_T_eig_metric);
    S751.T_eig_metric = S751.T_eig_metric + t751_elapsed;
    S751.T_eig = S751.T_eig + t751_elapsed;
    t751_T_eig_metric = [];
            lam = real(diag(D));
            mu = abs(lam) + p.EPS_SPEC;
            M = (Q .* mu') * Q';
            metric.Q = Q; metric.lam = lam; metric.mu = mu; metric.M = sparse(0.5 * (M + M'));
            metric.info = struct('u_ref', u);

        % Apply the absolute-Hessian construction to independent diagonal blocks.
        case 'block-jacobi'
            H = full(hess_matrix(A, u));
            blocks = make_blocks(n, p.BLOCK_SIZE);
            MB = cell(numel(blocks), 1); R = cell(numel(blocks), 1);
            try
                for bi = 1:numel(blocks)
                    B = blocks{bi};
                    HB = H(B, B); HB = 0.5 * (HB + HB');
    S751.metric_spectral_decomposition_count = S751.metric_spectral_decomposition_count + 1;
    t751_T_eig_metric = tic;
                    [Q, D] = eig(HB);
    t751_elapsed = toc(t751_T_eig_metric);
    S751.T_eig_metric = S751.T_eig_metric + t751_elapsed;
    S751.T_eig = S751.T_eig + t751_elapsed;
    t751_T_eig_metric = [];
                    mu = abs(real(diag(D))) + p.BLOCK_REG_PARAM;
                    MB{bi} = 0.5 * ((Q .* mu') * Q' + ((Q .* mu') * Q')');
    S751.factorization_attempt_count = S751.factorization_attempt_count + 1;
    t751_factor = tic;
                    R{bi} = chol(MB{bi});
    S751.T_factorization = S751.T_factorization + toc(t751_factor);
    t751_factor = [];
                end
            catch
    if exist('t751_T_eig_metric', 'var') && ~isempty(t751_T_eig_metric)
        t751_elapsed = toc(t751_T_eig_metric);
        S751.T_eig_metric = S751.T_eig_metric + t751_elapsed;
        S751.T_eig = S751.T_eig + t751_elapsed;
        t751_T_eig_metric = [];
    end
    if exist('t751_factor', 'var') && ~isempty(t751_factor)
        S751.T_factorization = S751.T_factorization + toc(t751_factor);
        S751.factorization_failure_count = S751.factorization_failure_count + 1;
        t751_factor = [];
    end
                ok = false; return;
            end
            Md = zeros(n, n);
            for bi = 1:numel(blocks)
                B = blocks{bi}; Md(B, B) = MB{bi};
            end
            metric.blocks = blocks; metric.MB = MB; metric.R = R; metric.M = sparse(Md);
            sz = cellfun(@numel, blocks);
            metric.info = struct('u_ref', u, 'num_blocks', numel(blocks), 'min_block_size', min(sz), 'max_block_size', max(sz));

        % Shift beyond negative curvature and retry incomplete Cholesky if needed.
        case 'shifted-ic'
            H = hess_matrix(A, u);
    t751_T_eig_metric = tic;
            try
                lam_min = real(eigs(H, 1, 'sa', eigs_opts(n)));
            catch
    S751.metric_smallest_eigenvalue_fallback_count = S751.metric_smallest_eigenvalue_fallback_count + 1;
                lam_min = min(real(eig(full(H))));
            end
    t751_elapsed = toc(t751_T_eig_metric);
    S751.T_eig_metric = S751.T_eig_metric + t751_elapsed;
    S751.T_eig = S751.T_eig + t751_elapsed;
    S751.metric_smallest_eigenvalue_query_count = S751.metric_smallest_eigenvalue_query_count + 1;
            neg_mag = max(0, -lam_min);
            sigma = neg_mag + p.SHIFT_ALPHA * max(neg_mag, 1.0);
            retries = 0; ok = false;
            while retries < p.ICHOL_MAX_RETRIES
                M_shift = 0.5 * (H + H') + sigma * speye(n);
                try
    S751.factorization_attempt_count = S751.factorization_attempt_count + 1;
    t751_factor = tic;
                    L = ichol(M_shift, struct('type', 'ict', 'droptol', p.ICHOL_DROPTOL));
    S751.T_factorization = S751.T_factorization + toc(t751_factor);
    t751_factor = [];
                    metric.L = L; metric.M = L * L';
                    metric.info = struct('u_ref', u, 'lambda_min', lam_min, 'sigma', sigma, ...
                                         'ichol_droptol', p.ICHOL_DROPTOL, 'ichol_retries', retries, 'ichol_success', true);
                    ok = true; return;
                catch
    if exist('t751_factor', 'var') && ~isempty(t751_factor)
        S751.T_factorization = S751.T_factorization + toc(t751_factor);
        S751.factorization_failure_count = S751.factorization_failure_count + 1;
        t751_factor = [];
    end
                    sigma = 2 * sigma;
                    retries = retries + 1;
                end
            end
            metric.info = struct('sigma', sigma, 'ichol_droptol', p.ICHOL_DROPTOL, 'ichol_retries', retries, 'ichol_success', false);

        % Combine the Laplacian with the magnitude of the local reaction derivative.
        case 'h1-reaction'
            M = A + spdiags(abs(fp_nl(u)), 0, n, n) + p.EPS_REACTION * speye(n);
            M = 0.5 * (M + M');
            metric.M = M;
            try
    S751.factorization_attempt_count = S751.factorization_attempt_count + 1;
    t751_factor = tic;
                metric.dec = decomposition(M, 'lu');
    S751.T_factorization = S751.T_factorization + toc(t751_factor);
    t751_factor = [];
            catch
    if exist('t751_factor', 'var') && ~isempty(t751_factor)
        S751.T_factorization = S751.T_factorization + toc(t751_factor);
        S751.factorization_failure_count = S751.factorization_failure_count + 1;
        t751_factor = [];
    end
                metric.dec = [];
            end
            metric.info = struct('u_ref', u);

        otherwise
            ok = false;
    end
end

function y = metric_apply(metric, z)
    global S751;
    S751.metric_apply_count = S751.metric_apply_count + 1;
    t751_operation = tic;
    switch metric.name
        case 'spectral'
            y = metric.Q * (metric.mu .* (metric.Q' * z));
        case 'block-jacobi'
            y = zeros(size(z));
            for bi = 1:numel(metric.blocks)
                B = metric.blocks{bi};
                y(B) = metric.MB{bi} * z(B);
            end
        case 'shifted-ic'
            y = metric.L * (metric.L' * z);
        case 'h1-reaction'
            y = metric.M * z;
        otherwise
            error('unknown metric');
    end
    t751_elapsed = toc(t751_operation);
    S751.T_metric_apply = S751.T_metric_apply + t751_elapsed;
    S751.T_apply_solve = S751.T_apply_solve + t751_elapsed;
end

function x = metric_solve(metric, z)
    global S751;
    S751.metric_solve_count = S751.metric_solve_count + 1;
    t751_operation = tic;
    switch metric.name
        case 'spectral'
            x = metric.Q * ((metric.Q' * z) ./ metric.mu);
        case 'block-jacobi'
            x = zeros(size(z));
            for bi = 1:numel(metric.blocks)
                B = metric.blocks{bi};
                R = metric.R{bi};
                x(B) = R \ (R' \ z(B));
            end
        case 'shifted-ic'
            y = metric.L \ z;
            x = metric.L' \ y;
        case 'h1-reaction'
            if isempty(metric.dec), x = metric.M \ z; else, x = metric.dec \ z; end
        otherwise
            error('unknown metric');
    end
    t751_elapsed = toc(t751_operation);
    S751.T_metric_solve = S751.T_metric_solve + t751_elapsed;
    S751.T_apply_solve = S751.T_apply_solve + t751_elapsed;
end

function blocks = make_blocks(n, bs)
    blocks = {};
    i = 1;
    while i <= n
        j = min(i + bs - 1, n);
        blocks{end+1} = i:j;
        i = j + 1;
    end
    if numel(blocks) >= 2 && numel(blocks{end}) < bs / 2
        blocks{end-1} = [blocks{end-1}, blocks{end}];
        blocks(end) = [];
    end
end

function [V, ok] = initialize_standard_frame(A, u, p)
    global S751;
    H = hess_matrix(A, u);
    try
        [vecs, D, flag] = eigs(H, p.K, 'sa', eigs_opts(p.N));
        if flag ~= 0, error('eigs failed'); end
        vals = real(diag(D));
    catch
        [Vf, Df] = eig(full(H));
        valsf = real(diag(Df));
        [~, ord] = sort(valsf, 'ascend');
        vecs = Vf(:, ord(1:p.K)); vals = valsf(ord(1:p.K));
    end
    [~, idx] = sort(vals, 'ascend');
    V = vecs(:, idx);
    t751_T_Euclidean_orth = tic;
    [V, ok] = orthonormalize_euclidean(V);
    t751_elapsed = toc(t751_T_Euclidean_orth);
    S751.T_Euclidean_orth = S751.T_Euclidean_orth + t751_elapsed;
    S751.Euclidean_orth_count = S751.Euclidean_orth_count + 1;
end

function [V, ok] = initialize_phisd_frame(A, u, metric, p)
    global S751;
    if strcmp(metric.name, 'spectral') && ~isempty(metric.lam)
        [~, idx] = sort(metric.lam ./ metric.mu, 'ascend');
        idx = idx(1:p.K);
        V = zeros(size(metric.Q, 1), p.K);
        for j = 1:p.K
            ii = idx(j);
            V(:, j) = metric.Q(:, ii) / sqrt(metric.mu(ii));
        end
    else
        H = hess_matrix(A, u);
        try
            [vecs, D, flag] = eigs(H, metric.M, p.K, 'sa', eigs_opts(p.N));
            if flag ~= 0, error('geigs failed'); end
            vals = real(diag(D));
            [~, idx] = sort(vals, 'ascend');
            V = vecs(:, idx);
        catch
            [Vf, Df] = eig(full(H), full(metric.M));
            valsf = real(diag(Df));
            [~, ord] = sort(valsf, 'ascend');
            V = Vf(:, ord(1:p.K));
        end
    end
    t751_T_M_orth = tic;
    [V, ok] = orthonormalize_metric(V, metric);
    t751_elapsed = toc(t751_T_M_orth);
    S751.T_M_orth = S751.T_M_orth + t751_elapsed;
    S751.M_orth_count = S751.M_orth_count + 1;
end

function [V, ok] = update_frame_standard(V, u, A, p)
    global S751;
    ok = true;
    fp_u = fp_nl(u);
    for t = 1:p.J
        for i = 1:p.K
            Hv = A * V(:, i) - fp_u .* V(:, i);
            proj = Hv - V(:, i) * (V(:, i)' * Hv);
            for j = 1:i-1
                proj = proj - 2 * V(:, j) * (V(:, j)' * Hv);
            end
            V(:, i) = V(:, i) - p.TAU * proj;
        end
    t751_T_Euclidean_orth = tic;
        [V, ok_orth] = orthonormalize_euclidean(V);
    t751_elapsed = toc(t751_T_Euclidean_orth);
    S751.T_Euclidean_orth = S751.T_Euclidean_orth + t751_elapsed;
    S751.Euclidean_orth_count = S751.Euclidean_orth_count + 1;
        if (~ok_orth) || (~all(isfinite(V(:)))), ok = false; return; end
    end
end

% Perform J Rayleigh-quotient sweeps with M^{-1} H and restore metric orthogonality.
function [V, ok] = update_frame_phisd(V, u, A, metric, p)
    global S751;
    ok = true;
    fp_u = fp_nl(u);
    for t = 1:p.J
        for i = 1:p.K
            Hv = A * V(:, i) - fp_u .* V(:, i);
            w = metric_solve(metric, Hv);
            Mw = metric_apply(metric, w);
            proj = w - V(:, i) * (V(:, i)' * Mw);
            for j = 1:i-1
                proj = proj - 2 * V(:, j) * (V(:, j)' * Mw);
            end
            V(:, i) = V(:, i) - p.TAU * proj;
        end
    t751_T_M_orth = tic;
        [V, ok_orth] = orthonormalize_metric(V, metric);
    t751_elapsed = toc(t751_T_M_orth);
    S751.T_M_orth = S751.T_M_orth + t751_elapsed;
    S751.M_orth_count = S751.M_orth_count + 1;
        if (~ok_orth) || (~all(isfinite(V(:)))), ok = false; return; end
    end
end

function diag_info = generalized_spectrum_diagnostic(A, u, metric, eta_after)
    try
        H = full(hess_matrix(A, u)); H = 0.5 * (H + H');
        M = full(metric.M); M = 0.5 * (M + M');
        vals = sort(real(eig(H, M)));
        abs_vals = abs(vals); mask = abs_vals > 1e-14;
        if ~any(mask)
            diag_info = struct('mu_M', NaN, 'L_M', NaN, 'kappa_M', NaN, 'eta_stable_upper', NaN, ...
                'eta_after', eta_after, 'eta_after_is_below_bound', false, ...
                'lambda_gen_first8', vals(1:min(8, end)), 'lambda_gen_last8', vals(max(1, end-7):end));
            return;
        end
        mu_M = min(abs_vals(mask)); L_M = max(abs_vals(mask)); up = 2 / L_M;
        diag_info = struct('mu_M', mu_M, 'L_M', L_M, 'kappa_M', L_M / mu_M, 'eta_stable_upper', up, ...
            'eta_after', eta_after, 'eta_after_is_below_bound', eta_after < up, ...
            'lambda_gen_first8', vals(1:min(8, end)), 'lambda_gen_last8', vals(max(1, end-7):end));
    catch
        diag_info = struct('mu_M', NaN, 'L_M', NaN, 'kappa_M', NaN, 'eta_stable_upper', NaN, ...
            'eta_after', eta_after, 'eta_after_is_below_bound', false, 'lambda_gen_first8', [], 'lambda_gen_last8', []);
    end
end

% A small residual triggers the ordinary-Hessian index check; divergence is retained.
function run = run_standard(A, h, u0, p)
    global S751;
    S751 = struct('T_total', 0, 'T_setup_update', 0, 'T_apply_solve', 0, 'T_eig', 0, 'T_cert', 0, 'initial_setup_time', 0, 'update_time', 0, 'initial_setup_count', 0, 'initial_setup_success_count', 0, 'update_count', 0, 'update_success_count', 0, 'refreeze_attempt_iteration', 0, 'frame_initialization_count', 0, 'frame_update_count', 0, 'metric_apply_count', 0, 'metric_solve_count', 0, 'T_metric_apply', 0, 'T_metric_solve', 0, 'T_eig_initial', 0, 'T_eig_refreeze', 0, 'T_eig_update', 0, 'T_eig_metric', 0, 'T_eig_diagnostic', 0, 'T_eig_status_cert', 0, 'T_M_orth', 0, 'T_Euclidean_orth', 0, 'M_orth_count', 0, 'Euclidean_orth_count', 0, 'T_factorization', 0, 'factorization_attempt_count', 0, 'factorization_failure_count', 0, 'metric_spectral_decomposition_count', 0, 'metric_smallest_eigenvalue_query_count', 0, 'metric_smallest_eigenvalue_fallback_count', 0, 'solver_status_cert_count', 0, 'posthoc_cert_count', 0);
    S751.refreeze_attempt_iteration = -1;
    u = u0;
    S751.total_token = tic;
    t751_T_eig_initial = tic;
    [V, ok_init] = initialize_standard_frame(A, u0, p);
    t751_elapsed = toc(t751_T_eig_initial);
    S751.T_eig_initial = S751.T_eig_initial + t751_elapsed;
    S751.T_eig = S751.T_eig + t751_elapsed;
    S751.frame_initialization_count = S751.frame_initialization_count + 1;
    if ~ok_init, V = []; end

    Hmax = p.MAX_ITER_STANDARD + 1;
    hist = new_history(Hmax);
    status = 'stagnation'; iterations = p.MAX_ITER_STANDARD; min_res = Inf;
    t0 = tic;

    if isempty(V)
        r0 = residual_l2h(A, u, h);
        hist = hist_append(hist, 0, r0, 0, p.ETA_STANDARD);
        run = finalize_run('standard', 'frame_instability', 0, trim_history(hist), min(r0, min_res), ...
            false, -1, NaN, p.ETA_STANDARD, p.ETA_STANDARD, A, u, p, toc(t0), []);
        run = fill_shifted_fields(run, []);
        return;
    end

    for m = 0:p.MAX_ITER_STANDARD
        g = grad_g(A, u);
        res = residual_l2h(A, u, h);
        hist = hist_append(hist, m, res, 0, p.ETA_STANDARD);
        if isfinite(res) && (res < min_res), min_res = res; end

        if (~isfinite(res)) || (max(abs(u)) > p.DIVERGENCE_MAX_ABS_U) || (res > p.DIVERGENCE_RESIDUAL)
            status = 'divergence'; iterations = m; break;
        end
        if res <= p.TOL
    t751_T_eig_status_cert = tic;
            eig8 = ordinary_eigs8(A, u, p);
            status = ternary(index_verified(eig8, p), 'success', 'wrong_index');
    t751_elapsed = toc(t751_T_eig_status_cert);
    S751.T_eig_status_cert = S751.T_eig_status_cert + t751_elapsed;
    S751.T_eig = S751.T_eig + t751_elapsed;
    S751.solver_status_cert_count = S751.solver_status_cert_count + 1;
            iterations = m; break;
        end
        if m == p.MAX_ITER_STANDARD
            status = 'stagnation'; iterations = m; break;
        end

    t751_T_eig_update = tic;
        [V, ok] = update_frame_standard(V, u, A, p);
    t751_elapsed = toc(t751_T_eig_update);
    S751.T_eig_update = S751.T_eig_update + t751_elapsed;
    S751.T_eig = S751.T_eig + t751_elapsed;
    S751.frame_update_count = S751.frame_update_count + 1;
        if ~ok, status = 'frame_instability'; iterations = m + 1; break; end

        d = -g + 2 * V * (V' * g);
        u = u + p.ETA_STANDARD * d;
        if ~all(isfinite(u)), status = 'divergence'; iterations = m + 1; break; end
    end

    run = finalize_run('standard', status, iterations, trim_history(hist), min_res, ...
        false, -1, NaN, p.ETA_STANDARD, p.ETA_STANDARD, A, u, p, toc(t0), []);
    run = fill_shifted_fields(run, []);
end

function run = run_phisd(method_name, metric_kind, eta_after, A, h, u0, p)
    global S751;
    S751 = struct('T_total', 0, 'T_setup_update', 0, 'T_apply_solve', 0, 'T_eig', 0, 'T_cert', 0, 'initial_setup_time', 0, 'update_time', 0, 'initial_setup_count', 0, 'initial_setup_success_count', 0, 'update_count', 0, 'update_success_count', 0, 'refreeze_attempt_iteration', 0, 'frame_initialization_count', 0, 'frame_update_count', 0, 'metric_apply_count', 0, 'metric_solve_count', 0, 'T_metric_apply', 0, 'T_metric_solve', 0, 'T_eig_initial', 0, 'T_eig_refreeze', 0, 'T_eig_update', 0, 'T_eig_metric', 0, 'T_eig_diagnostic', 0, 'T_eig_status_cert', 0, 'T_M_orth', 0, 'T_Euclidean_orth', 0, 'M_orth_count', 0, 'Euclidean_orth_count', 0, 'T_factorization', 0, 'factorization_attempt_count', 0, 'factorization_failure_count', 0, 'metric_spectral_decomposition_count', 0, 'metric_smallest_eigenvalue_query_count', 0, 'metric_smallest_eigenvalue_fallback_count', 0, 'solver_status_cert_count', 0, 'posthoc_cert_count', 0);
    S751.refreeze_attempt_iteration = -1;
    u = u0;
    Hmax = p.MAX_ITER_PHISD + 1;
    hist = new_history(Hmax);
    status = 'stagnation'; iterations = p.MAX_ITER_PHISD; min_res = Inf;
    did_refreeze = false; refreeze_iter = -1; refreeze_res = NaN; post_diag = [];
    t0 = tic;

    S751.total_token = tic;
    t751_initial_setup_time = tic;
    [metric, ok_metric] = build_metric(metric_kind, A, u, p);
    t751_elapsed = toc(t751_initial_setup_time);
    S751.initial_setup_time = S751.initial_setup_time + t751_elapsed;
    S751.T_setup_update = S751.T_setup_update + t751_elapsed;
    S751.initial_setup_count = S751.initial_setup_count + 1;
    S751.initial_setup_success_count = S751.initial_setup_success_count + double(ok_metric);
    if ~ok_metric
        r0 = residual_l2h(A, u, h);
        hist = hist_append(hist, 0, r0, 0, p.ETA_BEFORE_REFREEZE);
        run = finalize_run(method_name, 'factorization_failed', 0, trim_history(hist), r0, ...
            false, -1, NaN, p.ETA_BEFORE_REFREEZE, eta_after, A, u, p, toc(t0), []);
        run = fill_shifted_fields(run, metric);
        return;
    end

    t751_T_eig_initial = tic;
    [V, ok_init] = initialize_phisd_frame(A, u, metric, p);
    t751_elapsed = toc(t751_T_eig_initial);
    S751.T_eig_initial = S751.T_eig_initial + t751_elapsed;
    S751.T_eig = S751.T_eig + t751_elapsed;
    S751.frame_initialization_count = S751.frame_initialization_count + 1;
    if ~ok_init
        r0 = residual_l2h(A, u, h);
        hist = hist_append(hist, 0, r0, 0, p.ETA_BEFORE_REFREEZE);
        run = finalize_run(method_name, 'frame_instability', 0, trim_history(hist), r0, ...
            false, -1, NaN, p.ETA_BEFORE_REFREEZE, eta_after, A, u, p, toc(t0), []);
        run = fill_shifted_fields(run, metric);
        return;
    end

    for m = 0:p.MAX_ITER_PHISD
        g = grad_g(A, u);
        res = residual_l2h(A, u, h);
        if isfinite(res) && (res < min_res), min_res = res; end

        eta_probe = ternary_num(did_refreeze, eta_after, p.ETA_BEFORE_REFREEZE);
        rid = double(did_refreeze);

        if (~isfinite(res)) || (max(abs(u)) > p.DIVERGENCE_MAX_ABS_U) || (res > p.DIVERGENCE_RESIDUAL)
            status = 'divergence'; iterations = m;
            hist = hist_append(hist, m, res, rid, eta_probe); break;
        end
        if res <= p.TOL
    t751_T_eig_status_cert = tic;
            eig8 = ordinary_eigs8(A, u, p);
            status = ternary(index_verified(eig8, p), 'success', 'wrong_index');
    t751_elapsed = toc(t751_T_eig_status_cert);
    S751.T_eig_status_cert = S751.T_eig_status_cert + t751_elapsed;
    S751.T_eig = S751.T_eig + t751_elapsed;
    S751.solver_status_cert_count = S751.solver_status_cert_count + 1;
            iterations = m;
            hist = hist_append(hist, m, res, rid, eta_probe); break;
        end
        if m == p.MAX_ITER_PHISD
            status = 'stagnation'; iterations = m;
            hist = hist_append(hist, m, res, rid, eta_probe); break;
        end

        % Rebuild the metric and reinitialize the frame once at this threshold.
        % Freeze the metric and use the prescribed larger state step afterward.
        if (~did_refreeze) && (res <= p.REFREEZE_THRESHOLD)
    t751_update_time = tic;
            [metric, ok_rb] = rebuild_metric(metric, A, u, p);
    t751_elapsed = toc(t751_update_time);
    S751.update_time = S751.update_time + t751_elapsed;
    S751.T_setup_update = S751.T_setup_update + t751_elapsed;
    S751.update_count = S751.update_count + 1;
    S751.update_success_count = S751.update_success_count + double(ok_rb);
    S751.refreeze_attempt_iteration = m;
            if ~ok_rb
                status = 'frame_instability'; iterations = m;
                hist = hist_append(hist, m, res, 0, p.ETA_BEFORE_REFREEZE); break;
            end
    t751_T_eig_refreeze = tic;
            [V, ok_rf] = initialize_phisd_frame(A, u, metric, p);
    t751_elapsed = toc(t751_T_eig_refreeze);
    S751.T_eig_refreeze = S751.T_eig_refreeze + t751_elapsed;
    S751.T_eig = S751.T_eig + t751_elapsed;
    S751.frame_initialization_count = S751.frame_initialization_count + 1;
            if ~ok_rf
                status = 'frame_instability'; iterations = m;
                hist = hist_append(hist, m, res, 0, p.ETA_BEFORE_REFREEZE); break;
            end
            did_refreeze = true;
            refreeze_iter = m;
            refreeze_res = res;
    t751_T_eig_diagnostic = tic;
            post_diag = generalized_spectrum_diagnostic(A, u, metric, eta_after);
    t751_elapsed = toc(t751_T_eig_diagnostic);
    S751.T_eig_diagnostic = S751.T_eig_diagnostic + t751_elapsed;
    S751.T_eig = S751.T_eig + t751_elapsed;
        end

        eta_cur = ternary_num(did_refreeze, eta_after, p.ETA_BEFORE_REFREEZE);
        hist = hist_append(hist, m, res, double(did_refreeze), eta_cur);

    t751_T_eig_update = tic;
        [V, ok_up] = update_frame_phisd(V, u, A, metric, p);
    t751_elapsed = toc(t751_T_eig_update);
    S751.T_eig_update = S751.T_eig_update + t751_elapsed;
    S751.T_eig = S751.T_eig + t751_elapsed;
    S751.frame_update_count = S751.frame_update_count + 1;
        if ~ok_up, status = 'frame_instability'; iterations = m + 1; break; end

        % Reflect the preconditioned gradient along the M-orthonormal unstable frame.
        d = -metric_solve(metric, g) + 2 * V * (V' * g);
        u = u + eta_cur * d;
        if ~all(isfinite(u)), status = 'divergence'; iterations = m + 1; break; end
    end

    run = finalize_run(method_name, status, iterations, trim_history(hist), min_res, ...
        did_refreeze, refreeze_iter, refreeze_res, p.ETA_BEFORE_REFREEZE, eta_after, A, u, p, toc(t0), post_diag);
    run = fill_shifted_fields(run, metric);
end

% Post-solver verification is timed separately; a status check inside the solve remains included.
function run = finalize_run(method_name, status, iterations, hist, min_res, ...
        ref_triggered, ref_iter, ref_res, eta_before, eta_after, A, u, p, run_time, post_diag)
    global S751;
    S751.T_total = toc(S751.total_token);
    t751_cert = tic;

    if isempty(hist.iteration), final_res = NaN; else, final_res = hist.residual(end); end
    if ~isfinite(min_res), min_res = final_res; end

    if all(isfinite(u)), eig8 = ordinary_eigs8(A, u, p); else, eig8 = nan(8, 1); end
    eig5 = nan(5, 1); eig5(1:min(5, numel(eig8))) = eig8(1:min(5, numel(eig8)));

    run = struct(...
        'method', method_name, 'status', status, 'iterations', int32(iterations), ...
        'final_residual', double(final_res), 'min_residual', double(min_res), ...
        'refreeze_triggered', logical(ref_triggered), 'refreeze_iter', int32(ref_iter), 'refreeze_residual', double(ref_res), ...
        'eta_before_refreeze', double(eta_before), 'eta_after_refreeze', double(eta_after), ...
        'tau', double(p.TAU), 'J', int32(p.J), ...
        'lambda_1', double(eig5(1)), 'lambda_2', double(eig5(2)), 'lambda_3', double(eig5(3)), 'lambda_4', double(eig5(4)), 'lambda_5', double(eig5(5)), ...
        'index_k_verified', logical(index_verified(eig8, p)), ...
        'post_refreeze_spectral_diag', post_diag, ...
        'max_abs_u', double(max_abs_safe(u)), 'run_time', double(run_time), 'history', hist, ...
        'sigma', NaN, 'ichol_droptol', NaN, 'ichol_retries', NaN, 'ichol_success', false);
    % Read-only endpoint verification; never used by state/frame/stopping logic.
    endpoint_residual = residual_l2h(A, u, pi / (p.N + 1));
    S751.T_cert = toc(t751_cert);
    S751.posthoc_cert_count = double(all(isfinite(u)));
    run.timing = rmfield(S751, 'total_token');
    run.timing.endpoint = u;
    run.timing.endpoint_residual = endpoint_residual;
end

function run = fill_shifted_fields(run, metric)
    if isempty(metric) || ~isstruct(metric) || ~isfield(metric, 'name') || ~strcmp(metric.name, 'shifted-ic')
        return;
    end
    if isfield(metric, 'info')
        if isfield(metric.info, 'sigma'), run.sigma = metric.info.sigma; end
        if isfield(metric.info, 'ichol_droptol'), run.ichol_droptol = metric.info.ichol_droptol; end
        if isfield(metric.info, 'ichol_retries'), run.ichol_retries = metric.info.ichol_retries; end
        if isfield(metric.info, 'ichol_success'), run.ichol_success = logical(metric.info.ichol_success); end
    end
end

function m = max_abs_safe(u)
    if all(isfinite(u)), m = max(abs(u)); else, m = NaN; end
end

function h = new_history(maxn)
    h = struct('iteration', zeros(maxn, 1), 'residual', zeros(maxn, 1), ...
               'refreeze_id', zeros(maxn, 1), 'eta_current', zeros(maxn, 1), 'n', int32(0));
end

function h = hist_append(h, it, res, rid, eta)
    k = double(h.n) + 1;
    h.iteration(k) = it;
    h.residual(k) = res;
    h.refreeze_id(k) = rid;
    h.eta_current(k) = eta;
    h.n = int32(k);
end

function h = trim_history(h)
    n = double(h.n);
    h.iteration = h.iteration(1:n);
    h.residual = h.residual(1:n);
    h.refreeze_id = h.refreeze_id(1:n);
    h.eta_current = h.eta_current(1:n);
    h = rmfield(h, 'n');
end

% Preserve complete residual histories and unsuccessful outcomes in the exported data.
function [summary, histories] = save_outputs(runs, result_dir)
    names = struct('standard', 'history_standard.csv', ...
                   'spectral', 'history_spectral.csv', ...
                   'block_jacobi', 'history_block-jacobi.csv', ...
                   'shifted_ic', 'history_shifted-ic.csv', ...
                   'h1_reaction', 'history_h1-reaction.csv');

    method_to_field = @(m) strrep(m, '-', '_');
    histories = struct();
    for i = 1:numel(runs)
        r = runs{i};
        f = method_to_field(r.method);
        histories.(f) = r.history;
        T = table(r.history.iteration, r.history.residual, r.history.refreeze_id, r.history.eta_current, ...
                  'VariableNames', {'iteration', 'residual', 'refreeze_id', 'eta_current'});
        writetable(T, fullfile(result_dir, names.(f)));
    end

    summary = runs_to_summary(runs);
    Ts = struct2table(summary);
    writetable(Ts, fullfile(result_dir, 'summary.csv'));
    save(fullfile(result_dir, 'summary.mat'), 'summary');
end

function summary = runs_to_summary(runs)
    summary = struct([]);
    for i = 1:numel(runs)
        r = runs{i};
        s = struct(...
            'method', r.method, 'status', r.status, 'iterations', r.iterations, ...
            'final_residual', r.final_residual, 'min_residual', r.min_residual, ...
            'refreeze_triggered', r.refreeze_triggered, 'refreeze_iter', r.refreeze_iter, 'refreeze_residual', r.refreeze_residual, ...
            'eta_before_refreeze', r.eta_before_refreeze, 'eta_after_refreeze', r.eta_after_refreeze, ...
            'tau', r.tau, 'J', r.J, ...
            'lambda_1', r.lambda_1, 'lambda_2', r.lambda_2, 'lambda_3', r.lambda_3, 'lambda_4', r.lambda_4, 'lambda_5', r.lambda_5, ...
            'index_k_verified', r.index_k_verified, 'max_abs_u', r.max_abs_u, 'run_time', r.run_time, ...
            'sigma', r.sigma, 'ichol_droptol', r.ichol_droptol, 'ichol_retries', r.ichol_retries, 'ichol_success', r.ichol_success);
        if i == 1, summary = s; else, summary(i, 1) = s; end
    end
end

function y = ternary(cond, a, b)
    if cond, y = a; else, y = b; end
end

function y = ternary_num(cond, a, b)
    if cond, y = a; else, y = b; end
end

function s = tf_to_str(tf)
    if tf, s = 'True'; else, s = 'False'; end
end

function value = timing751_numerical_run(run)
    value = rmfield(run, 'run_time');
    fields = fieldnames(value.timing);
    remove = startsWith(fields, 'T_') | ismember(fields, ...
        {'initial_setup_time', 'update_time', 'component_times_are_inclusive'});
    value.timing = rmfield(value.timing, fields(remove));
end

function definitions = timing751_definitions()
    definitions = struct();
    definitions.unit = 'seconds';
    definitions.T_total = ['Standard starts immediately before initialize_standard_frame; ' ...
        'p-HiSD starts immediately before build_metric. Both stop on entry to finalize_run, ' ...
        'before its posthoc certification. Includes initial frame/metric, all outer iterations, ' ...
        'original in-loop refreeze diagnostic, status-required eigenvalue certification, and ' ...
        'original history finalization before finalize_run. Excludes MATLAB startup, prior ' ...
        'common initial-data construction, plotting, file I/O, timing summary, and posthoc ' ...
        'certification; includes instrumentation overhead without correction.'];
    definitions.T_setup_update = ['All initial metric construction and later reconstruction, ' ...
        'including spectral work and factorization attempts/retries.'];
    definitions.T_apply_solve = ['Explicit metric_apply and metric_solve calls, including ' ...
        'those inside frame updates and M-orthogonalization.'];
    definitions.T_eig = ['Initial/refreeze frames, all frame updates, metric-building spectral ' ...
        'calculations, original refreeze spectral diagnostic, and status-required eigenvalue ' ...
        'certification.'];
    definitions.T_cert = ['Original finalize_run repeated eigenvalue/index diagnostics and ' ...
        'result packing, plus read-only endpoint residual; excluded from T_total.'];
    definitions.initial_setup_time = 'Original initial build_metric interval.';
    definitions.update_time = 'Original refreeze rebuild_metric interval, including failed attempts.';
    definitions.T_metric_apply = 'Original explicit metric_apply intervals.';
    definitions.T_metric_solve = ['Original explicit metric_solve intervals; implicit factorization ' ...
        'in the original H1 fallback backslash remains here.'];
    definitions.T_M_orth = 'Original M-orthogonalization intervals, inclusive of metric_apply.';
    definitions.T_Euclidean_orth = 'Original Euclidean orthogonalization intervals.';
    definitions.T_factorization = 'Original explicit chol/ichol/decomposition attempts, including failures.';
    definitions.T_eig_initial = 'Original initial frame computation intervals.';
    definitions.T_eig_refreeze = 'Original refreeze frame computation intervals.';
    definitions.T_eig_update = 'Original full frame update intervals.';
    definitions.T_eig_metric = 'Original spectral calculations during metric construction.';
    definitions.T_eig_diagnostic = 'Original in-loop post-refreeze generalized spectrum diagnostic.';
    definitions.T_eig_status_cert = 'Original eigenvalue/index checks required to classify solver status.';
    definitions.legacy_run_time = ['Original run_time is retained with its original unequal boundaries; ' ...
        'it is not redefined as T_total. fill_shifted_fields remains after finalize_run.'];
    definitions.accounting = ['Component timing categories are inclusive diagnostics and must not ' ...
        'be summed to obtain T_total. Metric-building eig/eigs overlaps setup/update; frame ' ...
        'T_eig includes metric apply/solve and orthogonalization; T_M_orth includes metric_apply. ' ...
        'Unobservable generalized-eigensolver metric operations remain in T_eig.'];
end

function summary = timing751_summarize(raw)
    metrics = {'T_total', 'T_setup_update', 'T_apply_solve', 'T_eig', 'T_cert', ...
        'initial_setup_time', 'update_time', 'T_metric_apply', 'T_metric_solve', ...
        'T_M_orth', 'T_Euclidean_orth', 'T_factorization', 'T_eig_initial', ...
        'T_eig_refreeze', 'T_eig_update', 'T_eig_metric', 'T_eig_diagnostic', 'T_eig_status_cert'};
    summary = struct('schema', raw.schema, 'unit', 'seconds', 'repeats', raw.repeats, ...
        'retained_experiment_repetition', 1, 'non_timing_repetitions_exact', true, ...
        'T_total_includes_certification', false, 'component_times_are_inclusive', true);
    summary.definitions = raw.definitions;
    summary.run_policy = raw.run_policy;
    summary.unsuccessful_run_policy = raw.unsuccessful_run_policy;
    summary.environment = raw.environment;
    summary.methods = raw.methods;
    summary.metrics = metrics;
    rows = cell(numel(raw.methods) * numel(metrics), 12 + raw.repeats);
    row = 0;
    for method = 1:numel(raw.methods)
        first = raw.runs{1, method};
        method_field = strrep(first.method, '-', '_');
        for metric = 1:numel(metrics)
            values = cellfun(@(run) run.timing.(metrics{metric}), raw.runs(:, method));
            assert(all(isfinite(values) & values >= 0), 'Invalid Section 7.5.1 timing data.');
            stats = struct('raw', values, 'median', median(values), 'min', min(values), 'max', max(values));
            summary.statistics.(method_field).(metrics{metric}) = stats;
            row = row + 1;
            rows(row, :) = [{first.method, first.status, double(first.iterations), ...
                metrics{metric}, 'seconds'}, num2cell(values.'), ...
                {stats.median, stats.min, stats.max, raw.repeats, false, true, raw.definitions.accounting}];
        end
    end
    raw_columns = arrayfun(@(r) sprintf('Run_%d', r), 1:raw.repeats, 'UniformOutput', false);
    columns = [{'Method', 'Status', 'Iterations', 'Metric', 'Unit'}, raw_columns, ...
        {'Median', 'Min', 'Max', 'Repeats', 'T_total_includes_certification', ...
        'Component_times_are_inclusive', 'Accounting_note'}];
    summary.table = cell2table(rows, 'VariableNames', columns);
end
