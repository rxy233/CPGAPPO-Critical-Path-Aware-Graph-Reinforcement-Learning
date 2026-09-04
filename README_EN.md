# CPGAPPO — Constraint-Preserving Guide-Actor PPO with Dual GAT for Edge-Cloud Task Offloading

[English](README_EN.md) | [中文](README.md)

Reproducible experiment package for the paper *"CPGAPPO: PPO + dual GAT for Task Offloading in Edge-Cloud Environment"*, third major-revision round, addressing reviewer comments R1 / R3 / R4.

- Algorithm: CPGAPPO = PPO + dual GAT (forward + optional backward) + Constraint-Preserving guide score + CP-weighted app-credit allocation + softened shield.
- Environment: edge-cloud cooperative task offloading, 150 users, 8 edge nodes, 1 cloud, 60-node DAG workflow, 3 topologies (chain / default / wide).
- Baselines: 5 external RL baselines (DGMA_adapt, DGMA_paper, TransEdgeStyle, GATDQN, PPO) + 5 heuristic baselines (HybridPSOGA, Genetic, Greedy, Edge-only, Local-only) + 6 ablation variants (noguidece / noshield / noappcredit / nocp / fwdonly / alloff).
- Metrics: D_all = mean latency over all apps (timed-out apps charged at per-app deadline budget, R1-5); UtilityScore_all = `compute_utility_score(E_all, D_all, ρ_app, ρ_task, w_cost=0.25)`, used in the main table, higher is better.

---

## 0. Directory Layout

```
CPGAPPO/
├── run.py                 # Unified entry point
├── requirements.txt       # Dependencies
├── README.md              # Chinese README
├── README_EN.md           # This file
│
├── Algorithms/            # Algorithm implementations
│   ├── Benchmark.py                  # 5 heuristic baselines
│   ├── RealGATPPO/                   # CPGAPPO main algorithm (agent, core, model)
│   ├── Baselines/                    # GAT-PPO encoder (core dependency)
│   ├── StrongBaselines/              # 3 DGMA / TransEdge external RL
│   ├── GNNRL/                        # GAT-DQN external RL
│   ├── PPO/                          # Pure PPO external RL (no GAT)
│   └── Train/                        # Training wrappers + shared common.py
│
├── Environment/           # Physical env (env / Graph / components / computation / service)
├── scheduler/             # TaskScheduler / GraphScheduler / TaskSelector
├── utils/                 # constant.py / tools / generate_graph
├── Experiments_new/
│   └── exp_utils.py                  # CONFIG, compute_utility_score, get_graph_cache, init_worker
│
├── runners/               # Low-level run scripts (called by run.py, also runnable standalone)
│   ├── _run3_main_comparison_3dag.py      # Main comparison (Table IV + V)
│   ├── _run3_cpgappo_*_changenumber.py    # Sensitivity sweep (Table VI, R1-6/R4-1)
│   ├── _reeval_lib.py                     # CPGAPPO-family D_all reeval (R1-5)
│   ├── _dall_reeval_lib.py                # External RL + heuristic D_all reeval (R1-5)
│   └── _collect_revision_v2.py            # Result aggregation by reviewer comment
│
├── matrix/                # 3 DAG topology matrices
│   ├── matrix_60.txt          # default
│   ├── matrix_60_chain.txt    # chain
│   └── matrix_60_wide.txt     # wide
│
├── pretrained/            # Pretrained ckpts (12 RL × 3 DAG × 5 seed, ~375MB)
│   └── {dag}/{algo}/seed{N}/{ckpt_name}
│
├── results/               # Run outputs (auto-filled after running)
├── collected_csv/         # collect-mode aggregation output (auto-filled)
└── cache_graph/           # Graph cache (auto-generated on first run, md5-named)
```

---

## 1. Setup

### 1.1 Python & Dependencies

```bash
# Python 3.9+
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

GPU optional: the default install is the CPU build of torch; with an NVIDIA GPU install the matching CUDA build from https://pytorch.org/get-started/locally/ to speed up training.

### 1.2 Verify the Installation

```bash
python run.py --mode reeval_only --dags default --seeds 42 --algos CPGAPPO
# Outputs D_all + UtilityScore_all to
# results/main_comparison_reeval_*/default__CPGAPPO__seed42/result_reeval.json
```

---

## 2. Quick Reproduction

### 2.1 Single cell (~1-2 minutes)

```bash
python run.py --mode reeval_only --dags default --seeds 42 --algos CPGAPPO
```

### 2.2 Full main row (~30-60 minutes, CPU)

```bash
python run.py --mode reeval_only --dags all --seeds 1 3 5 7 42 --algos CPGAPPO
```

### 2.3 Full Table IV + V (~1-2 hours, CPU)

```bash
python run.py --mode reeval_only --dags all --seeds 1 3 5 7 42
```

### 2.4 Aggregate into CSVs

```bash
python run.py --mode collect
```

---

## 3. Full Retraining

### 3.1 Main comparison + ablation (Table IV + V)

```bash
python run.py --mode main_comparison --dags all --seeds 1 3 5 7 42 --episodes 100
```

### 3.2 Ablation table only (Table V, 7 variants)

```bash
python run.py --mode ablation_only --dags all --seeds 1 3 5 7 42
```

### 3.3 Hyperparameter sensitivity (Table VI, R1-6 / R4-1)

```bash
python run.py --mode sensitivity --param LAMBDA_GUIDE \
              --values 0.0 0.05 0.1 0.2 0.4 --dags default --seeds 1 3 5 7 42

for P in LAMBDA_GUIDE BONUS_TO W_E W_D SHIELD_THR_HIGH SHIELD_THR_NORM OVERFLOW_MULT CP_REM_COEFF; do
  python run.py --mode sensitivity --param $P --dags default --seeds 1 3 5 7 42
done
```

### 3.4 Variance analysis (R3-1)

```bash
python run.py --mode main_comparison --dags all --seeds 1 3 5 7 42
```

---

## 4. Reviewer Comment → Experiment Mode Mapping

| Reviewer comment | Requirement | run.py command |
|---|---|---|
| R1-1 / R1-2 / R1-3 | Ablation necessity | `--mode ablation_only` (7 variants) |
| R1-5 | Latency must include timed-out apps (D_all) | `--mode reeval_only` |
| R1-6 | 8 hyperparameter sensitivities | `--mode sensitivity --param <P>` |
| R3-1 | 5-seed variance | `--mode main_comparison --seeds 1 3 5 7 42` |
| R4-1 | Constant derivation / sensitivity | `--mode sensitivity --param CP_REM_COEFF / OVERFLOW_MULT` |
| R4-2 | Fair comparison protocol | `--mode main_comparison` (all algos share env params) |

---

## 5. Key Metric Definitions

| Metric | Definition | Implementation |
|---|---|---|
| **D_succ** | Mean latency over successful (non-timed-out) apps only | `ts.get_avg_results(only_successful=True)` |
| **D_all** | Mean latency over all apps; timed-out apps charged at per-app deadline budget (R1-5) | `ts.get_avg_results(only_successful=False, timeout_charge="deadline")` |
| **UtilityScore_all** | `compute_utility_score(E_all, D_all, ρ_app, ρ_task, w_cost=0.25, sla0=0.95)`, higher is better | `Experiments_new/exp_utils.py: compute_utility_score` |

The main table (Table IV / V) uses **D_all** and **UtilityScore_all**.

---

## 6. Experimental Setup

| Parameter | Value | Meaning |
|---|---|---|
| `deadline_slot` | 55 | Per-app relative deadline (in slots) |
| `slot_interval` | 0.01 s | Physical duration of one slot |
| `user_num` | 150 | Number of users (apps) |
| `edge_num` | 8 | Number of edge nodes |
| `MAX_STEPS` | 8000 | Max steps per episode |
| `STOP_ARRIVAL_STEP` | 2000 | All users forced to arrive before this step |
| `BURST_PROB` | 0.2 | Burst arrival probability |
| `BURST_SIZE` | 55 | Burst arrival size |
| `EPISODES` | 100 | RL training episodes |
| `seeds` | {1, 3, 5, 7, 42} | Main-table 5 seeds (mean±std) |

---

## 7. Caveats

1. **Matrix paths use backslashes on Windows**: `get_graph_cache` uses `md5(MATRIX_OVERRIDE_PATH)` as the graph-cache filename. `run.py` and all runners use `os.path.normpath` for consistency.
2. **Pretrained ckpts**: the 180 ckpts (~375MB) under `pretrained/` ship with the repo. `reeval_only` reads them directly; `main_comparison` trains fresh ckpts (does not overwrite pretrained/).
3. **Heuristic baselines have no ckpt**: `reeval_only` runs the 5 heuristics on the fly; their D_all = D_succ.
4. **First-run graph cache**: a md5-named cache is generated under `cache_graph/`.
5. **GPU fallback**: automatically uses CPU when CUDA is unavailable.

---

## 8. Running a Single Runner Standalone

`run.py` is the unified entry, but the low-level runners can also be executed directly:

```bash
python runners/_run3_main_comparison_3dag.py
python runners/_run3_cpgappo_LAMBDA_GUIDE_changenumber.py
```

Each runner has a configuration block at the top where you can edit `SEEDS` / `DAGS` / `ALGOS` / `SWEEP_PARAMS` constants.

---

## 9. Citation

If you reproduce the results in this repository for academic comparison, please cite the original paper:

> CPGAPPO: Critical-Path-Aware Graph Reinforcement Learning for Task Offloading in Consumer Wireless Edge Environments,
 Xiangyu Ren, Fagui Liu, Bin Wang, Xuhao Tang, Quan Tang, Fa Zhu, Yiqun Zhong, and Jun Jiang,
 Awaiting acceptance, 2026.

---

## 10. Contact

Experiment-related questions: renxy01@pcl.ac.cn
Code issues: open a GitHub Issue
