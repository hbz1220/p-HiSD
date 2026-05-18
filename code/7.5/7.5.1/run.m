semilinear_elliptic_phisd();

function semilinear_elliptic_phisd()
    p = default_params();
    result_dir = fileparts(mfilename('fullpath'));

    [A, h, x] = build_laplacian_dirichlet(p.N);
    u0 = sin(4 * x);

    runs = {
        run_standard(A, h, u0, p)
        run_phisd('spectral', 'spectral', p.ETA_AFTER_PHISD, A, h, u0, p)
        run_phisd('block-jacobi', 'block-jacobi', p.ETA_AFTER_BLOCK_JACOBI, A, h, u0, p)
        run_phisd('shifted-ic', 'shifted-ic', p.ETA_AFTER_PHISD, A, h, u0, p)
        run_phisd('h1-reaction', 'h1-reaction', p.ETA_AFTER_PHISD, A, h, u0, p)
    };

    [summary, histories] = save_outputs(runs, result_dir);

    N = p.N; K = p.K; TAU = p.TAU; J = p.J; TOL = p.TOL; %#ok<NASGU>
    REFREEZE_THRESHOLD = p.REFREEZE_THRESHOLD; %#ok<NASGU>
    MAX_ITER_STANDARD = p.MAX_ITER_STANDARD; MAX_ITER_PHISD = p.MAX_ITER_PHISD; %#ok<NASGU>
    ETA_STANDARD = p.ETA_STANDARD; ETA_BEFORE_REFREEZE = p.ETA_BEFORE_REFREEZE; %#ok<NASGU>
    ETA_AFTER_PHISD = p.ETA_AFTER_PHISD; ETA_AFTER_BLOCK_JACOBI = p.ETA_AFTER_BLOCK_JACOBI; %#ok<NASGU>
    EPS_SPEC = p.EPS_SPEC; EPS_REACTION = p.EPS_REACTION; SHIFT_ALPHA = p.SHIFT_ALPHA; %#ok<NASGU>
    BLOCK_SIZE = p.BLOCK_SIZE; BLOCK_REG_PARAM = p.BLOCK_REG_PARAM; %#ok<NASGU>

    history_standard = histories.standard; %#ok<NASGU>
    history_spectral = histories.spectral; %#ok<NASGU>
    history_block_jacobi = histories.block_jacobi; %#ok<NASGU>
    history_shifted_ic = histories.shifted_ic; %#ok<NASGU>
    history_h1_reaction = histories.h1_reaction; %#ok<NASGU>

    params = p; %#ok<NASGU>
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
end

function p = default_params()
    p = struct(...
        'N', 400, 'K', 3, 'TAU', 0.01, 'J', 5, 'TOL', 1e-6, ...
        'REFREEZE_THRESHOLD', 3e-1, 'INDEX_THRESHOLD', 1e-6, ...
        'MAX_ITER_STANDARD', 6000, 'MAX_ITER_PHISD', 6000, 'PLOT_MAX_ITER', 1500, ...
        'ETA_STANDARD', 1e-5, 'ETA_BEFORE_REFREEZE', 0.25, 'ETA_AFTER_PHISD', 1.5, 'ETA_AFTER_BLOCK_JACOBI', 0.9, ...
        'EPS_SPEC', 1e-2, 'EPS_REACTION', 1e-2, 'SHIFT_ALPHA', 1.0, ...
        'BLOCK_SIZE', 64, 'BLOCK_REG_PARAM', 1.0, ...
        'DIVERGENCE_MAX_ABS_U', 20.0, 'DIVERGENCE_RESIDUAL', 1e6, ...
        'ICHOL_DROPTOL', 1e-3, 'ICHOL_MAX_RETRIES', 6);
end

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
    ok = true;
    n = numel(u);
    switch metric.name
        case 'spectral'
            H = full(hess_matrix(A, u)); H = 0.5 * (H + H');
            [Q, D] = eig(H);
            lam = real(diag(D));
            mu = abs(lam) + p.EPS_SPEC;
            M = (Q .* mu') * Q';
            metric.Q = Q; metric.lam = lam; metric.mu = mu; metric.M = sparse(0.5 * (M + M'));
            metric.info = struct('u_ref', u);

        case 'block-jacobi'
            H = full(hess_matrix(A, u));
            blocks = make_blocks(n, p.BLOCK_SIZE);
            MB = cell(numel(blocks), 1); R = cell(numel(blocks), 1);
            try
                for bi = 1:numel(blocks)
                    B = blocks{bi};
                    HB = H(B, B); HB = 0.5 * (HB + HB');
                    [Q, D] = eig(HB);
                    mu = abs(real(diag(D))) + p.BLOCK_REG_PARAM;
                    MB{bi} = 0.5 * ((Q .* mu') * Q' + ((Q .* mu') * Q')');
                    R{bi} = chol(MB{bi});
                end
            catch
                ok = false; return;
            end
            Md = zeros(n, n);
            for bi = 1:numel(blocks)
                B = blocks{bi}; Md(B, B) = MB{bi};
            end
            metric.blocks = blocks; metric.MB = MB; metric.R = R; metric.M = sparse(Md);
            sz = cellfun(@numel, blocks);
            metric.info = struct('u_ref', u, 'num_blocks', numel(blocks), 'min_block_size', min(sz), 'max_block_size', max(sz));

        case 'shifted-ic'
            H = hess_matrix(A, u);
            try
                lam_min = real(eigs(H, 1, 'sa', eigs_opts(n)));
            catch
                lam_min = min(real(eig(full(H))));
            end
            neg_mag = max(0, -lam_min);
            sigma = neg_mag + p.SHIFT_ALPHA * max(neg_mag, 1.0);
            retries = 0; ok = false;
            while retries < p.ICHOL_MAX_RETRIES
                M_shift = 0.5 * (H + H') + sigma * speye(n);
                try
                    L = ichol(M_shift, struct('type', 'ict', 'droptol', p.ICHOL_DROPTOL));
                    metric.L = L; metric.M = L * L';
                    metric.info = struct('u_ref', u, 'lambda_min', lam_min, 'sigma', sigma, ...
                                         'ichol_droptol', p.ICHOL_DROPTOL, 'ichol_retries', retries, 'ichol_success', true);
                    ok = true; return;
                catch
                    sigma = 2 * sigma;
                    retries = retries + 1;
                end
            end
            metric.info = struct('sigma', sigma, 'ichol_droptol', p.ICHOL_DROPTOL, 'ichol_retries', retries, 'ichol_success', false);

        case 'h1-reaction'
            M = A + spdiags(abs(fp_nl(u)), 0, n, n) + p.EPS_REACTION * speye(n);
            M = 0.5 * (M + M');
            metric.M = M;
            try
                metric.dec = decomposition(M, 'lu');
            catch
                metric.dec = [];
            end
            metric.info = struct('u_ref', u);

        otherwise
            ok = false;
    end
end

function y = metric_apply(metric, z)
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
end

function x = metric_solve(metric, z)
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
end

function blocks = make_blocks(n, bs)
    blocks = {};
    i = 1;
    while i <= n
        j = min(i + bs - 1, n);
        blocks{end+1} = i:j; %#ok<AGROW>
        i = j + 1;
    end
    if numel(blocks) >= 2 && numel(blocks{end}) < bs / 2
        blocks{end-1} = [blocks{end-1}, blocks{end}];
        blocks(end) = [];
    end
end

function [V, ok] = initialize_standard_frame(A, u, p)
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
    [V, ok] = orthonormalize_euclidean(V);
end

function [V, ok] = initialize_phisd_frame(A, u, metric, p)
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
    [V, ok] = orthonormalize_metric(V, metric);
end

function [V, ok] = update_frame_standard(V, u, A, p)
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
        [V, ok_orth] = orthonormalize_euclidean(V);
        if (~ok_orth) || (~all(isfinite(V(:)))), ok = false; return; end
    end
end

function [V, ok] = update_frame_phisd(V, u, A, metric, p)
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
        [V, ok_orth] = orthonormalize_metric(V, metric);
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

function run = run_standard(A, h, u0, p)
    u = u0;
    [V, ok_init] = initialize_standard_frame(A, u0, p);
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
            eig8 = ordinary_eigs8(A, u, p);
            status = ternary(index_verified(eig8, p), 'success', 'wrong_index');
            iterations = m; break;
        end
        if m == p.MAX_ITER_STANDARD
            status = 'stagnation'; iterations = m; break;
        end

        [V, ok] = update_frame_standard(V, u, A, p);
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
    u = u0;
    Hmax = p.MAX_ITER_PHISD + 1;
    hist = new_history(Hmax);
    status = 'stagnation'; iterations = p.MAX_ITER_PHISD; min_res = Inf;
    did_refreeze = false; refreeze_iter = -1; refreeze_res = NaN; post_diag = [];
    t0 = tic;

    [metric, ok_metric] = build_metric(metric_kind, A, u, p);
    if ~ok_metric
        r0 = residual_l2h(A, u, h);
        hist = hist_append(hist, 0, r0, 0, p.ETA_BEFORE_REFREEZE);
        run = finalize_run(method_name, 'factorization_failed', 0, trim_history(hist), r0, ...
            false, -1, NaN, p.ETA_BEFORE_REFREEZE, eta_after, A, u, p, toc(t0), []);
        run = fill_shifted_fields(run, metric);
        return;
    end

    [V, ok_init] = initialize_phisd_frame(A, u, metric, p);
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
            eig8 = ordinary_eigs8(A, u, p);
            status = ternary(index_verified(eig8, p), 'success', 'wrong_index');
            iterations = m;
            hist = hist_append(hist, m, res, rid, eta_probe); break;
        end
        if m == p.MAX_ITER_PHISD
            status = 'stagnation'; iterations = m;
            hist = hist_append(hist, m, res, rid, eta_probe); break;
        end

        if (~did_refreeze) && (res <= p.REFREEZE_THRESHOLD)
            [metric, ok_rb] = rebuild_metric(metric, A, u, p);
            if ~ok_rb
                status = 'frame_instability'; iterations = m;
                hist = hist_append(hist, m, res, 0, p.ETA_BEFORE_REFREEZE); break;
            end
            [V, ok_rf] = initialize_phisd_frame(A, u, metric, p);
            if ~ok_rf
                status = 'frame_instability'; iterations = m;
                hist = hist_append(hist, m, res, 0, p.ETA_BEFORE_REFREEZE); break;
            end
            did_refreeze = true;
            refreeze_iter = m;
            refreeze_res = res;
            post_diag = generalized_spectrum_diagnostic(A, u, metric, eta_after);
        end

        eta_cur = ternary_num(did_refreeze, eta_after, p.ETA_BEFORE_REFREEZE);
        hist = hist_append(hist, m, res, double(did_refreeze), eta_cur);

        [V, ok_up] = update_frame_phisd(V, u, A, metric, p);
        if ~ok_up, status = 'frame_instability'; iterations = m + 1; break; end

        d = -metric_solve(metric, g) + 2 * V * (V' * g);
        u = u + eta_cur * d;
        if ~all(isfinite(u)), status = 'divergence'; iterations = m + 1; break; end
    end

    run = finalize_run(method_name, status, iterations, trim_history(hist), min_res, ...
        did_refreeze, refreeze_iter, refreeze_res, p.ETA_BEFORE_REFREEZE, eta_after, A, u, p, toc(t0), post_diag);
    run = fill_shifted_fields(run, metric);
end

function run = finalize_run(method_name, status, iterations, hist, min_res, ...
        ref_triggered, ref_iter, ref_res, eta_before, eta_after, A, u, p, run_time, post_diag)

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
