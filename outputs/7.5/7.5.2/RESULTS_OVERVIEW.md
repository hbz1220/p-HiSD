# Lane–Emden scalability results

Execution: **COMPLETE_WITH_LIMITATIONS**.

Default formal matrix: p-HiSD N=64,128,192,256 for p=3,5; standard HiSD N=128 for p=3,5. Three fresh workers per configuration and per selected timing mode.
Recorded formal attempts: 30; verified outcomes: 30.
Current-run metadata (elapsed values here are measured before final cleanup, not the final whole-workflow wall time): `{"elapsed_measurement_scope": "Pre-cleanup elapsed includes imports/gates/production/validation. Whole-run elapsed is printed after final output cleanup.", "historical_results_required": false, "result_directory": "outputs/7.5/7.5.2", "source": "code/7.5/7.5.2/run.py", "started_unix": 1790004878.381831, "whole_run_pre_cleanup_elapsed_s": 930.497484875}`

| Method | p | N | Mode | Verified/attempts | Solver s | Certification s | Solver+cert s | Sampled full RSS MiB |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| pHiSD_H1 | 3 | 64 | profile | 3/3 | 0.059867 [0.058589, 0.061651] | 0.017196 [0.01641, 0.018081] | 0.077064 [0.074999, 0.079732] | 75.703 [73.047, 79.109] |
| pHiSD_H1 | 3 | 128 | profile | 3/3 | 0.2555 [0.25457, 0.25659] | 0.11496 [0.11405, 0.11497] | 0.37047 [0.36953, 0.37064] | 106.06 [100.81, 106.27] |
| pHiSD_H1 | 3 | 192 | profile | 3/3 | 0.65886 [0.63966, 0.66494] | 0.54806 [0.54789, 0.55142] | 1.2103 [1.1876, 1.213] | 158.5 [150.33, 160.75] |
| pHiSD_H1 | 3 | 256 | profile | 3/3 | 1.3169 [1.2588, 1.3245] | 1.6881 [1.5139, 1.7182] | 2.9469 [2.8309, 3.0428] | 228.94 [223.78, 232.92] |
| pHiSD_H1 | 5 | 64 | profile | 3/3 | 0.070411 [0.069832, 0.070754] | 0.018593 [0.018565, 0.018755] | 0.089165 [0.088425, 0.089318] | 76.328 [75.375, 77.688] |
| pHiSD_H1 | 5 | 128 | profile | 3/3 | 0.3023 [0.29438, 0.30572] | 0.12867 [0.12862, 0.12991] | 0.43097 [0.423, 0.43563] | 104.44 [101.11, 110.19] |
| pHiSD_H1 | 5 | 192 | profile | 3/3 | 0.75225 [0.74555, 0.76748] | 0.50275 [0.49964, 0.58376] | 1.2671 [1.255, 1.3293] | 163.78 [154.12, 165.86] |
| pHiSD_H1 | 5 | 256 | profile | 3/3 | 1.4499 [1.4426, 1.4562] | 1.3624 [1.295, 1.4299] | 2.8122 [2.7376, 2.8861] | 233.72 [226.56, 235.59] |
| standard_HiSD | 3 | 128 | profile | 3/3 | 21.177 [21.175, 21.181] | 0.11871 [0.11828, 0.11949] | 21.296 [21.294, 21.3] | 106.33 [102.12, 109.36] |
| standard_HiSD | 5 | 128 | profile | 3/3 | 24.676 [24.481, 24.723] | 0.10417 [0.10366, 0.10454] | 24.78 [24.586, 24.827] | 102.39 [93.234, 102.42] |

Entries show median [min, max] among verified runs; failed/censored attempts remain in outcome denominators; temporary raw evidence is checked before cleanup. Different timing modes are separate series.

## Measurement interpretation

- T_total is the solver wall interval, including operator assembly, setup, initialization and updates. T_cert lies outside T_total. Numerical T_total_with_cert is their sum; whole workflow wall time also includes imports, gates, monitoring, saves, verification and final cleanup.
- T_eig_inclusive overlaps M apply/solve. The additive per-run view is setup/update + apply/solve + eig exclusive + other. Independent column medians need not sum to median total; these marginal medians are never plotted as an additive stack.
- Sampled RSS uses only exact numerical intervals. Missing short-interval samples remain null. Cumulative OS high-water is separately retained and is never subtracted to infer stage memory. Exported sparse-array bytes exclude unexported native-library workspaces.
- LU export, NPZ compression, file output and plotting occur after numerical measurement. Requested RSS sampling is approximately 20 ms; raw gaps and sample counts are checked before temporary traces are removed; sampling limitations are listed below.

## Scope and limitations

The measured implementation fixes k=1, J=5 and sparse LU. With n=N² and F_n=nnz(L)+nnz(U), per-update work includes O(Jn+(J+1)F_n); setup, factorization, initial eigensolve and certification add separate cost. Main persistent storage includes O(n+F_n+n*ncv), while sampled RSS covers observed process/native memory. These observations establish neither linear overall complexity nor mesh-independent convergence, high-index or parallel scaling, performance at unrun N=768/1024, or independent optimal tuning of the comparison method.

Certification independently checks saved endpoints and six ordinary full-space eigenpairs, residuals, orthogonality and resolved index-one sign margins. It is a numerical acceptance protocol, not an interval-arithmetic proof.

## Automatic acceptance

regression_pass=True; validator_tests_pass=True; gate2_dry_pass=True; overhead_pass=True; data_integrity_pass=True; all_automatic_validation_pass=True.

- Sampling qualification production_pHiSD_H1_N64_p3_profile_r1_a1: cert interval has 1 RSS samples; transient peaks are unresolved.
- Sampling qualification production_pHiSD_H1_N64_p3_profile_r2_a1: cert interval has 1 RSS samples; transient peaks are unresolved.
- Sampling qualification production_pHiSD_H1_N64_p3_profile_r3_a1: cert interval has 0 RSS samples; transient peaks are unresolved.
- Sampling qualification production_pHiSD_H1_N64_p5_profile_r1_a1: cert interval has 1 RSS samples; transient peaks are unresolved.
- Sampling qualification production_pHiSD_H1_N64_p5_profile_r2_a1: cert interval has 1 RSS samples; transient peaks are unresolved.
- Sampling qualification production_pHiSD_H1_N64_p5_profile_r3_a1: cert interval has 1 RSS samples; transient peaks are unresolved.
- Sampling qualification smoke_pHiSD_H1_N16_p3_profile_r1_a1: A smoke numerical phase shorter than the requested 20 ms RSS sampling period has no captured RSS sample. Null sampled peaks are retained; cumulative OS high-water is available and is not substituted for a sampled phase peak.
Requested formal matrix complete: True.

## Retained output files

Only scalability_summary.csv, fixed_grid_cost_summary.csv, cost_breakdown.csv, memory_structure.csv and this overview are retained after successful validation. Temporary raw records, monitoring traces and internal reports are used for validation and statistics, then removed. No evidence ZIP, plots or separate logs are kept.
If the run fails, available tables are explicitly incomplete and RUN_FAILURE.txt records the stage, configuration and reason.

