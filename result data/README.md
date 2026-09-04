# 第三次大修实验数据集 (定稿版)

**输出目录**: `D:\svn\第三次大修数据upload`

---

## CSV 文件清单 (按审稿人意见分类)

| 文件 | 对应审稿人意见 | 内容 |
|---|---|---|
| `master_seedwise.csv` | 全部 | 所有 cell 的逐 seed 原始数据 (含 D_all 来源标记) |
| `R3-1_table_IV_main_comparison_5seeds.csv` | **R3-1, R4-2** (主对比 5-seed 协议) | 11 算法 × 3 DAG × {1,3,5,7,42}, mean±std |
| `R3-1_table_V_ablation_5seeds.csv` | **R3-1** (消融 5-seed) | 7 变体 × 3 DAG × {1,3,5,7,42}, mean±std |
| `R1-2_ablation_ordering_check.csv` | **R1-2** (ablation 排序自检) | 每个 cell 的 AppTO/Util 与 CPGAPPO 的差 |
| `R1-6_R4-1_sensitivity.csv` | **R1-6, R4-1** (敏感性分析) | 8 参数 × 5 取值 × 5 seeds, 逐 seed; 跨拓扑 (chain/wide) 3 参数 × 5 取值 × 5 seeds |
| `R1-6_R4-1_sensitivity_summary.csv` | **R1-6, R4-1** | 同上, 聚合为 mean±std |
| `R3-1_R4-2_seedwise_variance.csv` | **R3-1, R4-2** (5-seed 方差/对比协议) | CPGAPPO 逐 seed ({1,3,5,7,42}), 含 per-DAG mean/std |
| `data_gap_summary.csv` | **数据缺口总览** | 列出所有缺 D_all / 缺 cell / 缺 seed 的项 |
| `raw_result_json/` | 全部 | 原始 result.json (按算法分子目录), 含 `*_reeval.json` |

### raw_result_json 子目录

| 子目录 | 内容 | 文件数 |
|---|---|---|
| `{algo}/` | 各算法主对比/消融原始 result.json (algo ∈ CPGAPPO, DGMA_adapt, DGMA_paper, Edge-only, GATDQN, Genetic, Greedy, HybridPSOGA, Local-only, PPO, TransEdgeStyle, alloff, fwdonly, noappcredit, nocp, noguidece, noshield) | 按 algo 分 |
| `HybridPSOGA/` | HybridPSOGA 重跑 5-seed result_reeval.json + reeval.csv | 15 |
| `_sensitivity/` | default DAG 敏感性 8 参数 × 5 取值 × 5 seeds result.json | 200 |
| `_ablation_5seeds/` | 7 消融变体 × 3 DAG × 5 seeds result.json | 105 |
| `_sensitivity_xtopo/` | chain/wide 敏感性 3 参数 × 5 取值 × 2 新种子 (5,7) result.json | 60 |

> **`algorithm` 字段说明**: 原始 result.json 中的 `"algorithm": "train_guided_v9_cpguide_credit"` 是 CPGAPPO 主算法 (v9) 在原始训练代码里的函数名。本代码包为统一命名将其重命名为 `train_cpgappo_dual_cpgappo` (位于 `Algorithms/Train/train_cpgappo_unified.py`), 两者指同一算法; 消融变体同理 (`train_guided_v9_*` → `train_cpgappo_dual_*`)。CSV 汇总表统一用 `CPGAPPO` / `noguidece` / ... 标签, 不含 v9 字样。

---

## 统计口径

- **D_all** = `ts.get_avg_results(only_successful=False, timeout_charge="deadline")`
  - 超时应用按 per-app deadline 预算 `get_app_deadline_slot(uid) * slot_interval` 计入 (TD_i^max - TS_i)
- **D_succ** = 仅成功 (非超时) 应用的平均 delay
- **UtilityScore_all** = `compute_utility_score(e_all, d_all, rho_app, rho_task, w_cost=0.25, sla0=0.95)` (主表口径, ddof=0)
- **种子协议**:
  - 5-seed (主表 / R3-1 / R4-2): {1, 3, 5, 7, 42}
- **std**: R3-1 / R1-5 聚合 CSV 用 ddof=1; 数据整理汇总.txt 用 ddof=0

---

## 16 算法清单

| label | algo | 类别 | Table IV | Table V |
|---|---|---|---|---|
| CPGAPPO | CPGAPPO | 本文方法 | ✓ | ✓ |
| noguidece | noguidece | 消融 (w/o Guide-CE) | | ✓ |
| noshield | noshield | 消融 (w/o Shield) | | ✓ |
| noappcredit | noappcredit | 消融 (w/o App Credit) | | ✓ |
| nocp | nocp | 消融 (w/o CP Sequencing) | | ✓ |
| fwdonly | fwdonly | 消融 (w/o All = forward-only) | | ✓ |
| alloff | alloff | 消融 (w/o All = all off) | | ✓ |
| DGMA_adapt | DGMA_adapt | 外部 RL | ✓ | |
| DGMA_paper | DGMA_paper | 外部 RL | ✓ | |
| TransEdgeStyle | TransEdgeStyle | 外部 RL | ✓ | |
| GATDQN | GATDQN | 外部 RL | ✓ | |
| PPO | PPO | 外部 RL (PPO backbone) | ✓ | |
| HybridPSOGA | HybridPSOGA | 启发式 | ✓ | |
| Genetic | Genetic | 启发式 | ✓ | |
| Greedy | Greedy | 启发式 | ✓ | |
| Edge-only | Edge-only | 启发式 | ✓ | |
| Local-only | Local-only | 启发式 | ✓ | |

---

## 8 个敏感性参数 (R1-6 + R4-1)

| 参数 | 默认值 | 取值 | 审稿人 |
|---|---|---|---|
| LAMBDA_GUIDE | 0.1 | 0, 0.05, 0.1, 0.2, 0.4 | R1-6 |
| BONUS_TO | -7 | -14, -10, -7, -4, -2 | R1-6 |
| W_E | 0.05 | 0, 0.025, 0.05, 0.1, 0.2 | R1-6 |
| W_D | 0.1 | 0, 0.05, 0.1, 0.2, 0.4 | R1-6 |
| SHIELD_THR_HIGH | 1.10 | 1.00, 1.05, 1.10, 1.15, 1.20 | R1-6 (θ_risk) |
| SHIELD_THR_NORM | 1.15 | 1.05, 1.10, 1.15, 1.20, 1.30 | R1-6 (θ_risk) |
| OVERFLOW_MULT | 1.25 | 1.00, 1.10, 1.25, 1.50, 2.00 | R4-1 (Eq.25 的 1.25 系数) |
| CP_REM_COEFF | 0.45 | 0.20, 0.30, 0.45, 0.60, 0.80 | R4-1 (Ĉ_rem 的 0.45 系数) |

---

## 数据规模

| 类别 | 规模 |
|---|---|
| 主对比 (Table IV) | 11 算法 × 3 DAG × 5 seeds = 55 rows |
| 消融 (Table V) | 7 变体 × 3 DAG × 5 seeds = 35 rows |
| 敏感性 default (R1-6/R4-1) | 8 参数 × 5 取值 × 5 seeds = 200 明细 |
| 敏感性 跨拓扑 (R1-6) | 3 参数 × 5 取值 × 2 DAG × 5 seeds = 150 明细 |
