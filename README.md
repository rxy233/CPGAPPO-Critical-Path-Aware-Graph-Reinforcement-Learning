# CPGAPPO — 边-云任务卸载的约束保持引导 PPO + 双 GAT

[English](README_EN.md) | [中文](README.md)

论文《CPGAPPO: PPO + dual GAT for Task Offloading in Edge-Cloud Environment》第三轮大修的可复现实验代码包, 回应审稿人意见 R1 / R3 / R4。

- 算法: CPGAPPO = PPO + 双 GAT (前向 + 可选反向) + 约束保持引导分数 + CP 加权应用信用分配 + 软化 shield。
- 环境: 边-云协同任务卸载, 150 用户, 8 边缘节点, 1 云, 60 节点 DAG 工作流, 3 种拓扑 (chain / default / wide)。
- 对比: 5 个外部 RL 基线 (DGMA_adapt, DGMA_paper, TransEdgeStyle, GATDQN, PPO) + 5 个启发式基线 (HybridPSOGA, Genetic, Greedy, Edge-only, Local-only) + 6 个消融变体 (noguidece / noshield / noappcredit / nocp / fwdonly / alloff)。
- 指标: D_all = 所有应用平均时延 (超时按 per-app deadline 预算计入, R1-5); UtilityScore_all = `compute_utility_score(E_all, D_all, ρ_app, ρ_task, w_cost=0.25)`, 主表用, 越大越好。

---

## 0. 目录结构

```
CPGAPPO/
├── run.py                 # 统一运行入口
├── requirements.txt       # 依赖
├── README.md              # 本文件
│
├── Algorithms/            # 算法实现
│   ├── Benchmark.py                  # 5 个启发式基线 (Local/Cloud/Edge/Greedy/Genetic/HybridPSOGA)
│   ├── RealGATPPO/                   # CPGAPPO 主算法 (agent, core, model)
│   ├── Baselines/                    # GAT-PPO 编码器 (core 依赖)
│   ├── StrongBaselines/              # 3 个 DGMA / TransEdge 外部 RL
│   ├── GNNRL/                        # GAT-DQN 外部 RL
│   ├── PPO/                          # 纯 PPO 外部 RL (无 GAT)
│   └── Train/                        # 训练 wrapper + 共享 common.py
│
├── Environment/           # 物理环境 (env / Graph / components / computation / service)
├── scheduler/             # TaskScheduler / GraphScheduler / TaskSelector
├── utils/                 # constant.py / tools / generate_graph
├── Experiments_new/
│   └── exp_utils.py                  # CONFIG, compute_utility_score, get_graph_cache, init_worker
│
├── runners/               # 底层运行脚本 (run.py 调用, 也可单独跑)
│   ├── _run3_main_comparison_3dag.py      # 主对比 (Table IV + V)
│   ├── _run3_cpgappo_*_changenumber.py    # 敏感性扫描 (Table VI, R1-6/R4-1)
│   ├── _reeval_lib.py                     # CPGAPPO 家族 D_all reeval (R1-5)
│   ├── _dall_reeval_lib.py                # 外部 RL + 启发式 D_all reeval (R1-5)
│   └── _collect_revision_v2.py            # 结果按审稿人意见分类汇总
│
├── matrix/                # 3 个 DAG 拓扑矩阵
│   ├── matrix_60.txt          # default
│   ├── matrix_60_chain.txt    # chain
│   └── matrix_60_wide.txt     # wide
│
├── pretrained/            # 预训练 ckpt (12 RL × 3 DAG × 5 seed, ~375MB)
│   └── {dag}/{algo}/seed{N}/{ckpt_name}
│
├── results/               # 运行产物 (跑完自动填)
├── collected_csv/         # collect 模式汇总输出 (跑完自动填)
└── cache_graph/           # graph 缓存 (首次跑自动生成, md5 命名)
```

---

## 1. 环境准备

### 1.1 Python 与依赖

```bash
# Python 3.9+
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

GPU 可选: 默认装 CPU 版 torch; 有 NVIDIA GPU 按 https://pytorch.org/get-started/locally/ 装对应 CUDA 版本可加速训练。

### 1.2 验证安装

```bash
python run.py --mode reeval_only --dags default --seeds 42 --algos CPGAPPO
# 输出 D_all + UtilityScore_all 到
# results/main_comparison_reeval_*/default__CPGAPPO__seed42/result_reeval.json
```

---

## 2. 快速复现

### 2.1 单格 (~1-2 分钟)

```bash
python run.py --mode reeval_only --dags default --seeds 42 --algos CPGAPPO
```

### 2.2 主表全行 (~30-60 分钟, CPU)

```bash
python run.py --mode reeval_only --dags all --seeds 1 3 5 7 42 --algos CPGAPPO
```

### 2.3 完整 Table IV + V (~1-2 小时, CPU)

```bash
python run.py --mode reeval_only --dags all --seeds 1 3 5 7 42
```

### 2.4 汇总成 CSV

```bash
python run.py --mode collect
```

---

## 3. 完整重训

### 3.1 主对比 + 消融 (Table IV + V)

```bash
python run.py --mode main_comparison --dags all --seeds 1 3 5 7 42 --episodes 100
```

### 3.2 仅消融表 (Table V, 7 变体)

```bash
python run.py --mode ablation_only --dags all --seeds 1 3 5 7 42
```

### 3.3 超参敏感性 (Table VI, R1-6 / R4-1)

```bash
python run.py --mode sensitivity --param LAMBDA_GUIDE \
              --values 0.0 0.05 0.1 0.2 0.4 --dags default --seeds 1 3 5 7 42

for P in LAMBDA_GUIDE BONUS_TO W_E W_D SHIELD_THR_HIGH SHIELD_THR_NORM OVERFLOW_MULT CP_REM_COEFF; do
  python run.py --mode sensitivity --param $P --dags default --seeds 1 3 5 7 42
done
```

### 3.4 方差分析 (R3-1)

```bash
python run.py --mode main_comparison --dags all --seeds 1 3 5 7 42
```

---

## 4. 审稿人意见 → 实验模式对照

| 审稿人意见 | 要求 | run.py 命令 |
|---|---|---|
| R1-1 / R1-2 / R1-3 | 消融必要性 | `--mode ablation_only` (7 变体) |
| R1-5 | 时延应含超时应用 (D_all) | `--mode reeval_only` |
| R1-6 | 8 项超参敏感性 | `--mode sensitivity --param <P>` |
| R3-1 | 5-seed 方差 | `--mode main_comparison --seeds 1 3 5 7 42` |
| R4-1 | 常数推导 / 敏感性 | `--mode sensitivity --param CP_REM_COEFF / OVERFLOW_MULT` |
| R4-2 | 对比协议公平性 | `--mode main_comparison` (所有算法统一 env 参数) |

---

## 5. 关键指标定义

| 指标 | 定义 | 实现 |
|---|---|---|
| **D_succ** | 仅成功应用的平均时延 | `ts.get_avg_results(only_successful=True)` |
| **D_all** | 所有应用平均时延, 超时按 per-app deadline 预算计入 (R1-5) | `ts.get_avg_results(only_successful=False, timeout_charge="deadline")` |
| **UtilityScore_all** | `compute_utility_score(E_all, D_all, ρ_app, ρ_task, w_cost=0.25, sla0=0.95)`, 越大越好 | `Experiments_new/exp_utils.py: compute_utility_score` |

主表 (Table IV / V) 用 D_all 与 UtilityScore_all 两列。

---

## 6. 实验设置

| 参数 | 值 | 含义 |
|---|---|---|
| `deadline_slot` | 55 | 每个 app 的相对 deadline (slot 数) |
| `slot_interval` | 0.01 s | 一个 slot 的物理时长 |
| `user_num` | 150 | 用户 (应用) 数 |
| `edge_num` | 8 | 边缘节点数 |
| `MAX_STEPS` | 8000 | 单 episode 最大步数 |
| `STOP_ARRIVAL_STEP` | 2000 | 在此步前所有用户强制到达 |
| `BURST_PROB` | 0.2 | 突发到达概率 |
| `BURST_SIZE` | 55 | 突发到达规模 |
| `EPISODES` | 100 | RL 训练轮数 |
| `seeds` | {1, 3, 5, 7, 42} | 主表 5 seed (mean±std) |

---

## 7. 注意事项

1. **matrix 路径用反斜杠 (Windows)**: `get_graph_cache` 用 `md5(MATRIX_OVERRIDE_PATH)` 做 graph 缓存文件名。`run.py` 与所有 runner 用 `os.path.normpath` 统一路径。
2. **预训练 ckpt**: `pretrained/` 下 180 个 ckpt (~375MB) 随仓库提供。`reeval_only` 直接读; `main_comparison` 从头训练生成新 ckpt (不覆盖 pretrained/)。
3. **启发式基线无 ckpt**: `reeval_only` 跑 5 个启发式时会现算一遍; 其 D_all = D_succ。
4. **首次运行 graph 缓存**: 在 `cache_graph/` 生成 md5 命名缓存。
5. **GPU 回退**: 无 CUDA 时自动用 CPU。

---

## 8. 单独跑某个 runner

`run.py` 是统一入口, 底层 runner 也可单独执行:

```bash
python runners/_run3_main_comparison_3dag.py
python runners/_run3_cpgappo_LAMBDA_GUIDE_changenumber.py
```

各 runner 顶部有配置区, 可手动改 `SEEDS` / `DAGS` / `ALGOS` / `SWEEP_PARAMS` 等常量。

---

## 9. 引用

如复现结果用于学术对比, 请引用原文:

> [论文标题], [作者], Awaiting acceptance, [年份].

---

## 10. 联系

实验相关问题: renxy01@pcl.ac.cn
代码问题: 提 GitHub Issue
