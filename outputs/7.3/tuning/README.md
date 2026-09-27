# Parameter selection for the modified Rosenbrock problem

These archived records document the five-method parameter search for Section 7.3.
The selected HiSD, A-HiSD, BB-HiSD and p-HiSD configurations use `J=1`;
PC-HiSD uses its native coupled predictor-corrector update without a separate `J`.
The parameter values match those used in the new formal results in the parent directory.

The search used development seeds 101 and 102. Screening allowed 4,000 outer
updates and 60 seconds per run; full candidate evaluation allowed 400,000 updates
and 300 seconds, whichever limit was reached first. Backend and frame screening
was followed by candidate screening, full evaluation, repeated timing of finalists,
local refinement and boundary checks. The complete candidate grids and recorded
decisions are provided. Screening included other frame settings and LOBPCG;
not every screened configuration was repeated three times.

A finalist qualified only if all three full-budget repetitions at both development
points reached the target saddle. Selection used the geometric mean of the two
per-seed median total times. The subsequent development sensitivity checks were
retained as observations and were not fed back into the selection.

The directory contains 894 unique runs: 78 HiSD, 483 A-HiSD, 176 PC-HiSD,
75 BB-HiSD and 82 p-HiSD. All five selected configurations have six certified
development runs. Both successful and unsuccessful candidate records are retained.

| File | Contents |
|---|---|
| `protocol.json` | Problem, initial-point construction, candidate grids, screening and full-evaluation budgets, numerical criteria and package versions. |
| `tuning_all.csv` | One row per archived run, including parameters, budgets, outcomes, costs and certification summaries. |
| `raw_runs.jsonl.xz` | The 894 per-run numerical records, including detailed certification, operation counts, warnings and failure information. |
| `selected_parameters.json` | Final parameters, their six validation runs and time statistics, plus decisions and run references for every search stage. |

`raw_runs.jsonl.xz` is an XZ-compressed JSON Lines file, with one run per line.
It can be decompressed with an XZ-compatible archive tool or read with Python's
standard-library `lzma.open` function.
Run and candidate labels provide cross-references between these files. Repetitions
retain the archived numbering `0,1,2`. The `stage` field identifies a run's original
stage; `requested_stages` records reuse in later stages, so a reused run is listed
only once. References beginning `#/stages/` point within `selected_parameters.json`.
`screen` and `full` identify the applicable budget. `SCREEN_BUDGET` is a truncated
screen, and `BUDGET_ITER` is an exhausted update limit; neither means that numerical
divergence was established. Times are in seconds, and `t_total` is the archived
solver time including initialization, iteration and endpoint verification.

The historical gradient tolerance was `1e-6`, with target distance below `0.01`.
Its index threshold was `max(1e-8, 1e-8 * ||H||_infinity)`; the original thresholds
and certificates are retained. The new formal comparison uses a 2,000,000-update,
300-second limit and the fixed index threshold `1e-6`, as recorded in
[`../protocol.json`](../protocol.json). The state and target-distance tolerances
are unchanged. The selected archived endpoints also satisfy that fixed threshold.

These historical timing values describe parameter selection; the new measurements
for the paper remain in the parent directory. The archived sensitivity slices are
also separate from the new [`../table4.csv`](../table4.csv).
Running `code/7.3/run.py` regenerates the formal comparison and current sensitivity
results; it does not rerun or overwrite this archived parameter search.
