# Section 7.3 manuscript results

Mode: formal; status: COMPLETE_WITH_FAILURES.
Frozen configurations; the earlier parameter search is not rerun.
Times are fresh measurements on the environment recorded in protocol.json.

## Table 3

| Method | Outer updates | Gradient evaluations | HVPs | Total time (s) | Certified seeds |
|---|---:|---:|---:|---:|---:|
| HiSD | 172843 | 172845 | 864215 | 28.4088 | 5/5 |
| A-HiSD | 1956 | 1958 | 9780 | 0.336938 | 5/5 |
| PC-HiSD | 172848 | 345698 | 1.72848e+06 | 26.0178 | 5/5 |
| BB-HiSD | 426 | 428 | 2130 | 0.081611 | 5/5 |
| p-HiSD | 25 | 27 | 176 | 0.0132739 | 5/5 |

Paired BB-HiSD/p-HiSD geometric mean time ratio: 6.02043.

## Table 4

| r | HiSD outcome | T_dev (s) | p-HiSD outcome | T_dev (s) |
|---:|---|---:|---|---:|
| 0.5 | 2/2 | 55.7825 | 2/2 | 0.0233685 |
| 0.75 | 2/2 | 37.2676 | 2/2 | 0.0167823 |
| 1 | 2/2 | 27.7483 | 2/2 | 0.0136837 |
| 1.25 | 0/2 (T) | -- | 2/2 | 0.0264242 |
| 1.5 | 0/2 (T) | -- | 0/2 (T) | -- |

T: time budget exhausted. I: iteration budget exhausted. All other failures retain the solver status.
Figure 3 uses the actual median-time repetition for seed 1; figure3_history.csv contains its plotted data.
raw_runs.jsonl retains every prescribed run, including failures, endpoint certification.
