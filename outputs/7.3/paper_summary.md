# Section 7.3 manuscript results

Mode: formal; status: COMPLETE_WITH_FAILURES.
Frozen configurations; the earlier parameter search is not rerun.
Times are fresh measurements on the environment recorded in protocol.json.

## Table 3

| Method | Outer updates | Gradient evaluations | HVPs | Total time (s) | Certified seeds |
|---|---:|---:|---:|---:|---:|
| HiSD | 172843 | 172845 | 4.32106e+06 | 88.3058 | 5/5 |
| A-HiSD | 1956 | 1958 | 48880 | 1.01622 | 5/5 |
| PC-HiSD | 172848 | 345698 | 1.72848e+06 | 25.8949 | 5/5 |
| BB-HiSD | 528 | 530 | 13180 | 0.282909 | 5/5 |
| p-HiSD | 25 | 27 | 656 | 0.0325492 | 5/5 |

Paired BB-HiSD/p-HiSD geometric mean time ratio: 7.37292.

## Table 4

| r | HiSD outcome | T_dev (s) | p-HiSD outcome | T_dev (s) |
|---:|---|---:|---|---:|
| 0.5 | 2/2 | 175.846 | 2/2 | 0.0698229 |
| 0.75 | 2/2 | 117.2 | 2/2 | 0.0455498 |
| 1 | 2/2 | 89.8095 | 2/2 | 0.0333328 |
| 1.25 | 0/2 (T) | -- | 2/2 | 0.0833049 |
| 1.5 | 0/2 (T) | -- | 0/2 (T) | -- |

T: time budget exhausted. I: iteration budget exhausted. All other failures retain the solver status.
Figure 3 uses the actual median-time repetition for seed 1; figure3_history.csv contains its plotted data.
raw_runs.jsonl retains every prescribed run, including failures, endpoint certification.
