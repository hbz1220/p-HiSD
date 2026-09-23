# Section 7.6 results and timing

Three independent repetitions are summarized below. Times are in seconds.

## Numerical results

| Method | Iterations | Status | Final residual | Index | Endpoint max diff | Full-history max diff |
|---|---:|---|---:|---:|---:|---:|
| HiSD | 33939 | converged | 9.9971692599843206e-07 | 1 | 0 | 0 |
| Laplacian | 841 | converged | 9.8796390238865983e-07 | 1 | 0 | 0 |
| shifted Cholesky | 25 | converged | 6.0205425089205993e-07 | 1 | 0 | 0 |
| Spectral | 37 | converged | 7.4442636075049441e-07 | 1 | 0 | 0 |

Operation counts and factorization backend (identical across baseline, reference and all formal repetitions):

- HiSD: `{"apply_count": 0, "certification_count": 1, "factorization_attempt_count": 0, "factorization_failure_count": 0, "frame_iterations": 169695, "initial_setup_count": 0, "setup_count": 0, "solve_count": 0, "state_updates": 33939, "update_count": 0}`
- Laplacian: `{"apply_count": 4206, "certification_count": 1, "factorization_attempt_count": 1, "factorization_backend": "scipy.sparse.linalg.factorized: SuperLU", "factorization_failure_count": 0, "frame_iterations": 4205, "initial_setup_count": 1, "setup_count": 1, "solve_count": 5046, "state_updates": 841, "update_count": 0}`
- shifted Cholesky: `{"apply_count": 126, "certification_count": 1, "factorization_attempt_count": 1, "factorization_backend": "scipy.linalg.cho_factor / cho_solve", "factorization_failure_count": 0, "frame_iterations": 125, "initial_setup_count": 1, "setup_count": 1, "solve_count": 150, "state_updates": 25, "update_count": 0}`
- Spectral: `{"apply_count": 186, "certification_count": 1, "factorization_attempt_count": 1, "factorization_backend": "scipy.linalg.cho_factor / cho_solve", "factorization_failure_count": 0, "frame_iterations": 185, "initial_setup_count": 1, "setup_count": 1, "solve_count": 222, "state_updates": 37, "update_count": 0}`

## Timing

timing_repeats = 3

All values are seconds. Use median in the manuscript. Each row retains all three raw times in JSON/CSV. No solver warm-up, tuning, backend or thread changes are applied.

| Method | Quantity | Median | Min | Max |
|---|---|---:|---:|---:|
| HiSD | T_total | 80.1314782 | 78.9604714 | 81.322182 |
| HiSD | T_setup_update | 0 | 0 | 0 |
| HiSD | T_apply_solve | 0 | 0 | 0 |
| HiSD | T_eig | 69.4392764 | 68.4235564 | 70.4712086 |
| HiSD | T_cert | 0.105715375 | 0.104821917 | 0.112293875 |
| HiSD | T_shared_initial | 0.107021666 | 0.104962833 | 0.110243 |
| HiSD | T_H0_build | 0.103819917 | 0.1017035 | 0.107170333 |
| HiSD | T_shared_spectrum | 0.00315295899 | 0.00299983402 | 0.003203542 |
| HiSD | T_shared_other | 5.57909952e-05 | 4.87900397e-05 | 7.2832976e-05 |
| HiSD | T_method_initial | 0.00300154099 | 0.00299666601 | 0.00301337501 |
| HiSD | T_initial_eig | 0.00299887499 | 0.00299483398 | 0.003008584 |
| HiSD | T_outer | 80.0182218 | 78.850453 | 81.2142177 |
| HiSD | T_metric_setup | 0 | 0 | 0 |
| HiSD | T_metric_apply | 0 | 0 | 0 |
| HiSD | T_metric_solve | 0 | 0 | 0 |
| HiSD | T_frame | 69.3260976 | 68.3135886 | 70.3633027 |
| HiSD | T_cert_Hessian | 0.103770125 | 0.102761292 | 0.110266792 |
| Laplacian | T_total | 2.21007604 | 2.11914217 | 2.23050683 |
| Laplacian | T_setup_update | 7.52909982e-05 | 7.19169911e-05 | 0.000139666983 |
| Laplacian | T_apply_solve | 0.0325003132 | 0.0323556836 | 0.0334076383 |
| Laplacian | T_eig | 1.9301565 | 1.85149239 | 1.95135833 |
| Laplacian | T_cert | 0.104850833 | 0.104741 | 0.109616334 |
| Laplacian | T_shared_initial | 0.107021666 | 0.104962833 | 0.110243 |
| Laplacian | T_H0_build | 0.103819917 | 0.1017035 | 0.107170333 |
| Laplacian | T_shared_spectrum | 0.00315295899 | 0.00299983402 | 0.003203542 |
| Laplacian | T_shared_other | 5.57909952e-05 | 4.87900397e-05 | 7.2832976e-05 |
| Laplacian | T_method_initial | 0.00347966599 | 0.00346708298 | 0.003677209 |
| Laplacian | T_initial_eig | 0.00339720902 | 0.00339054101 | 0.00352845798 |
| Laplacian | T_outer | 2.10163354 | 2.00522196 | 2.12001808 |
| Laplacian | T_metric_setup | 7.52909982e-05 | 7.19169911e-05 | 0.000139666983 |
| Laplacian | T_metric_apply | 0.011400029 | 0.0112956643 | 0.011715935 |
| Laplacian | T_metric_solve | 0.0211002842 | 0.0210600193 | 0.0216917033 |
| Laplacian | T_frame | 1.82185225 | 1.73779376 | 1.84099491 |
| Laplacian | T_cert_Hessian | 0.1029265 | 0.102743 | 0.10769925 |
| shifted Cholesky | T_total | 0.179706333 | 0.17903525 | 0.181769875 |
| shifted Cholesky | T_setup_update | 0.00289458298 | 0.002744709 | 0.00317058398 |
| shifted Cholesky | T_apply_solve | 0.00459961453 | 0.00450820613 | 0.00486174284 |
| shifted Cholesky | T_eig | 0.167248546 | 0.166635621 | 0.169781376 |
| shifted Cholesky | T_cert | 0.106362042 | 0.10589775 | 0.110213917 |
| shifted Cholesky | T_shared_initial | 0.107021666 | 0.104962833 | 0.110243 |
| shifted Cholesky | T_H0_build | 0.103819917 | 0.1017035 | 0.107170333 |
| shifted Cholesky | T_shared_spectrum | 0.00315295899 | 0.00299983402 | 0.003203542 |
| shifted Cholesky | T_shared_other | 5.57909952e-05 | 4.87900397e-05 | 7.2832976e-05 |
| shifted Cholesky | T_method_initial | 0.00624745898 | 0.00618845801 | 0.00665333399 |
| shifted Cholesky | T_initial_eig | 0.00347954201 | 0.003291792 | 0.00349916599 |
| shifted Cholesky | T_outer | 0.066031333 | 0.065279416 | 0.067883959 |
| shifted Cholesky | T_metric_setup | 0.00289458298 | 0.002744709 | 0.00317058398 |
| shifted Cholesky | T_metric_apply | 0.000402370846 | 0.00039916989 | 0.000415247981 |
| shifted Cholesky | T_metric_solve | 0.00419724369 | 0.00410903624 | 0.00444649486 |
| shifted Cholesky | T_frame | 0.056796128 | 0.0561120431 | 0.0584367871 |
| shifted Cholesky | T_cert_Hessian | 0.104457708 | 0.103980917 | 0.108300458 |
| Spectral | T_total | 0.212078208 | 0.2115715 | 0.214061458 |
| Spectral | T_setup_update | 0.00297504102 | 0.00268412501 | 0.00310200002 |
| Spectral | T_apply_solve | 0.00650070515 | 0.00639762674 | 0.00663583697 |
| Spectral | T_eig | 0.194925295 | 0.194745046 | 0.197608042 |
| Spectral | T_cert | 0.105392291 | 0.104496958 | 0.107784083 |
| Spectral | T_shared_initial | 0.107021666 | 0.104962833 | 0.110243 |
| Spectral | T_H0_build | 0.103819917 | 0.1017035 | 0.107170333 |
| Spectral | T_shared_spectrum | 0.00315295899 | 0.00299983402 | 0.003203542 |
| Spectral | T_shared_other | 5.57909952e-05 | 4.87900397e-05 | 7.2832976e-05 |
| Spectral | T_method_initial | 0.00627620801 | 0.00599045801 | 0.00682270899 |
| Spectral | T_initial_eig | 0.00330374998 | 0.00329908298 | 0.003711458 |
| Spectral | T_outer | 0.097828 | 0.097727125 | 0.100839167 |
| Spectral | T_metric_setup | 0.00297504102 | 0.00268412501 | 0.00310200002 |
| Spectral | T_metric_apply | 0.000537533168 | 0.00049755181 | 0.000597714068 |
| Spectral | T_metric_solve | 0.00596317198 | 0.00590007493 | 0.0060381229 |
| Spectral | T_frame | 0.084134125 | 0.084060712 | 0.0867191701 |
| Spectral | T_cert_Hessian | 0.103448917 | 0.102562834 | 0.105836084 |

T_total is an assembled standalone method estimate. T_cert outside T_total.
component timing categories must not be summed blindly to obtain T_total.
For nonconverged methods T_total is consumed wall-clock until termination, not time-to-solution; no speedup is reported.

- `T_total`: Assembled standalone method estimate = T_shared_initial + T_method_initial + T_outer; not a contiguous stopwatch. Shared initialization is executed once in its original position and charged once to each method. Python startup, common problem/u0/g0 construction, certification, plotting, summaries and file I/O are excluded.
- `T_shared_initial`: Original interval from before H0=dense_hessian through h0_norm: initial dense Hessian, shared ordinary eigendecomposition, original diagnostics, lam_min and h0_norm.
- `T_H0_build`: Complete original shared dense_hessian call, including finite-difference HVP and state/adjoint work.
- `T_shared_spectrum`: Original shared np.linalg.eigh(H0). The subsequent make_method eigendecomposition is unchanged.
- `T_shared_other`: T_shared_initial minus T_H0_build and T_shared_spectrum, including original diagnostics and instrumentation overhead.
- `T_method_initial`: Original method preparation from t76_begin_method to t76_end_method, including metric construction/setup and initial eigenspace.
- `T_setup_update`: Equal to T_metric_setup: actual initial metric construction/factorization only. H1 sparse conversion/factorized; shifted-Cholesky delta/construction/factorization; frozen-spectral eps_spec/construction/factorization. Common PDE matrices excluded. No metric updates; update_count=0.
- `T_metric_setup`: Existing initial metric setup interval measured by t76_metric from the method preparation start. Standard identity has zero.
- `T_apply_solve`: T_metric_apply + T_metric_solve, including only actual explicit metric callable executions. No added metric operations; Standard identity has zero.
- `T_metric_apply`: Original explicit metric apply callable executions, including initial M-normalization and frame normalizations.
- `T_metric_solve`: Original explicit metric solve callable executions in state directions and frame updates.
- `T_eig`: Inclusive diagnostic = T_H0_build + T_shared_spectrum + T_initial_eig + T_frame. Includes nested metric work also recorded in T_apply_solve.
- `T_initial_eig`: Original make_method including ordinary/generalized eigensolver, selection, normalization and kappa/L_M0. Internal generalized-eigensolver work remains here.
- `T_frame`: Original direction-update blocks, including HVP/state/adjoint work, Rayleigh/projection, normalization and nested metric apply/solve.
- `T_outer`: Original run entry through the end of the unchanged outer loop, stopping before final_index. Includes residual/cost/history/state/frame work and timing overhead.
- `T_cert`: Only the original final_index call: endpoint dense Hessian, eigvalsh and Morse-index threshold/count computation. Outside T_total.
- `T_cert_Hessian`: Original dense_hessian subinterval within final_index, included in T_cert only.
- `legacy_history_time`: Original output time and per-point history clocks retained separately; excluded from numerical equality checks.
- `nonconverged_T_total`: Consumed wall-clock until termination, not time-to-solution. No speedup is reported for nonconverged methods.

## Output files

Files below are relative to this report. The default location is outputs/7.6/.

- `6.pdf`
- `timing76_raw.json`
- `timing76_raw.npz`
- `timing76_summary.json`
- `timing76_summary.csv`
- `timing76_audit.md`

6.pdf uses the first formal repetition.
Endpoint and full-history differences above compare the formal runs with the reference run.
