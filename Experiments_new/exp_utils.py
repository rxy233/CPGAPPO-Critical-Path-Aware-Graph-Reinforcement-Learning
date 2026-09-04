# -*- coding: utf-8 -*-
"""
Experiment utilities: global CONFIG, scoring, graph cache, worker init.

Contents:
  - CONFIG: global dict of env/scheduler/training parameters (MAX_STEPS,
    STOP_ARRIVAL_STEP, BURST_*, SEED, RESULTS_BASE_DIR, TOPK_*, EVAL_*).
  - compute_score / compute_utility_score: utility function
    U = w_eff*E_norm - w_cost*D - w_app*rho_app - w_task*rho_task,
    main-table weight w_cost=0.25 (UtilityScore_all).
  - get_graph_cache(matrix_path): md5(matrix_path)-keyed DAG cache in
    cache_graph/. The md5 key depends on the exact path string, so on
    Windows matrix paths must use backslashes (run.py uses os.path.normpath
    to guarantee this).
  - init_worker / generate_arrival_plan / apply_arrival_plan: deterministic
    per-seed env setup shared by every algorithm.
  - load_model_bundle / save_model_bundle / get_feature_dim /
    graph_state_to_vector: ckpt I/O and feature plumbing.
  - calc_timeout_rate / subtask_outcome_stats / diagnose_timeout:
    timeout accounting helpers.
"""
import os
import sys
import json
import time
import random
import pickle
import csv
import copy
import hashlib
import numpy as np
import pandas as pd
import torch
import math
from pathlib import Path
from datetime import datetime
from utils.constant import para

# ================= 全局配置 =================
CONFIG = {
    "NUM_PROCESSES": 4,
    # 降低并发数，避免多进程争抢同一块 GPU 显存和计算资源
    # 建议：单 GPU 上只跑 1 个 RL 任务（GAT_PPO/RealGATPPO 模型较大），最多 2 个轻量任务
    "MAX_RL_CONCURRENT": 1,  # RTX 3090 推荐 1 个（GAT 模型），如果只需要 DQN 等小模型可改 2
    # 大幅增加最大步数，作为保底，防止死循环，但不要让它切断任务
    "MAX_STEPS": 8000,
    # 在第 2000 步之前强制所有用户到达，留 4000 步给算法处理积压
    "STOP_ARRIVAL_STEP": 2000,
    "ARRIVAL_INTERVAL": 1,
    "BURST_MODE": True,
    "BURST_PROB": 0.2,
    "BURST_SIZE": 55,
    "SEED": 0,
    # 结果输出根目录: 仓库内 results/
    "RESULTS_BASE_DIR": os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")),
    "GPU_IDS": [0, 0, 0],
    "SCORE_WINDOW": 20,  # 修复ScoreTracker 的滑动窗口大小
    # 专家数据配置（DGATDQN 使用 Genetic/PSOGA 作为专家）
    "USE_EXPERT_DATA": True,  # 默认关闭，可改为 True 启用
    "EXPERT_RATIO": 0.3,  # 专家数据混采比例（30%）
    # Top-K任务选择配置（RL视野限制）
    "TOPK_TASKS": 30,  # RL每次最多考虑的任务数（视野限制）
    "SMART_TOPK": True,  # 是否使用智能Top-K选择（按紧迫度排序）
    "ADAPTIVE_TOPK": True,  # 是否使用动态K值（根据负载调整）
    "TOPK_MIN": 15,  # 最小K值（轻度负载）
    "TOPK_MAX": 30,  # 最大K值（重度负载）
    # 训练-评估流程配置
    "AUTO_EVAL_AFTER_TRAIN": False,  # 训练结束自动 eval（已禁用）
    "EVAL_EVERY": 10,  # 每10个episode做一次eval写曲线（0表示不做中途eval）
    "EVAL_EPISODES": 1,  # eval跑几轮取平均（建议1~5）
    "EVAL_USE_BEST": True,  # eval用训练期间best模型（否则用最后模型）
    "CURVE_RETRY": 10,  # 曲线写入重试次数（防Windows锁文件）
    # 所有算法使用同一套 SLA 和风险参数
    "SLA0": 0.99,      # 目标 SLA (95% 成功率)
    "KAPPA": 3.0,      # 风险敏感指数惩罚系数（保留指数敏感）
    "V_CAP_OVER": 2.0, # over 的上限，防止指数爆炸
}

# ================= ScoreTracker 类 =================
class ScoreTracker:
    """
    【新版评分公式】指标跟踪器

    新公式: Score(θ) = (φ1·Ê + φ2·D̂succ) · exp(κ·v(θ)) + η·[v(θ)-1]^+ + λ·ρtask(θ)

    其中:
    - ρapp(θ): 应用级超时率 (SLA指标)
    - ρtask(θ): 子任务级超时率 (过程指标)
    - v(θ) = ρapp(θ) / δ, δ = 1 - SLA0 (error budget)
    - Ê, D̂succ: 归一化的能耗和成功时延
    - [x]^+ = max(0, x)
    """
    def __init__(self, E_min=0.3, E_max=3.0, D_min=0.7, D_max=5.0,
                 phi1=5.0/6.0, phi2=1.0/6.0, kappa=3.0,
                 eta=5.0, lam=0.5, sla0=0.95, v_cap_over=2.0):
        # Min-Max 归一化参数
        self.E_min, self.E_max = E_min, E_max
        self.D_min, self.D_max = D_min, D_max

        # 评分公式权重
        self.phi1, self.phi2 = phi1, phi2
        self.kappa, self.eta, self.lam = kappa, eta, lam
        self.sla0 = sla0  # 目标 SLA (0.95)
        self.delta = 1.0 - sla0  # error budget
        self.v_cap_over = v_cap_over  # over 的上限，防止指数爆炸

        # 应用级统计（SLA指标）
        self.app_timeout_cnt = 0
        self.app_arrived_cnt = 0
        self.succ_delay_sum = 0.0
        self.succ_cnt = 0

        # 子任务级统计（过程指标）
        self.task_timeout_cnt = 0
        self.task_total_cnt = 0

    def update_app(self, is_timeout: bool, delay: float = 0.0):
        """
        更新应用完成状态

        Args:
            is_timeout: 应用是否超时
            delay: 应用端到端时延（仅用于成功的应用）
        """
        self.app_arrived_cnt += 1
        if is_timeout:
            self.app_timeout_cnt += 1
        else:
            self.succ_cnt += 1
            self.succ_delay_sum += float(delay)

    def update_task(self, is_timeout: bool):
        """
        更新子任务完成状态（过程指标）

        Args:
            is_timeout: 子任务是否超时
        """
        self.task_total_cnt += 1
        if is_timeout:
            self.task_timeout_cnt += 1

    @property
    def rho_app(self):
        """应用级超时率 (SLA指标)"""
        if self.app_arrived_cnt == 0:
            return 0.0
        return self.app_timeout_cnt / self.app_arrived_cnt

    @property
    def rho_task(self):
        """子任务级超时率 (过程指标)"""
        if self.task_total_cnt == 0:
            return 0.0
        return self.task_timeout_cnt / self.task_total_cnt

    @property
    def v(self):
        """
        归一化违约强度: v = ρapp / δ

        v ≤ 1: 还在 SLA 允许的失败预算内
        v > 1: 超过 error budget
        """
        if self.delta < 1e-12:
            return 0.0
        v = self.rho_app / self.delta
        return min(v, self.v_cap)  # 防止指数爆炸

    @property
    def D_succ(self):
        """成功集合平均时延（未归一化）"""
        if self.succ_cnt == 0:
            return self.D_max  # 如果没有成功，使用最差值惩罚
        else:
            return self.succ_delay_sum / self.succ_cnt

    @property
    def D_succ_hat(self):
        """成功任务平均延迟的归一化值"""
        D_succ = self.D_succ
        den = max(self.D_max - self.D_min, 1e-9)
        return float(max(0.0, min(1.0, (D_succ - self.D_min) / den)))

    def energy_hat(self, E):
        """能耗归一化"""
        den = max(self.E_max - self.E_min, 1e-9)
        return float(max(0.0, min(1.0, (float(E) - self.E_min) / den)))

    def base_cost(self, E_hat, D_hat_succ):
        """基础成本：φ1·Ê + φ2·D̂succ"""
        return self.phi1 * E_hat + self.phi2 * D_hat_succ

    def final_score(self, E_avg):
        """
        【新版】最终评分

        Score(θ) = (φ1·Ê + φ2·D̂succ) · exp(κ·v(θ)) + η·[v(θ)-1]^+ + λ·ρtask(θ)
        """
        # 1. 归一化
        E_hat = self.energy_hat(E_avg)
        D_hat = self.D_succ_hat

        # 2. 基础性能成本
        base = self.base_cost(E_hat, D_hat)

        # 3. 风险敏感项（指数放大 SLA 违约）
        risk = math.exp(self.kappa * self.v)

        # 4. Error budget 违约惩罚 [v-1]^+
        penalty_budget = self.eta * max(0.0, self.v - 1.0)

        # 5. 子任务级过程超时惩罚
        penalty_task = self.lam * self.rho_task

        # 6. 总评分
        score = base * risk + penalty_budget + penalty_task

        return score

    def get_info(self, E_avg):
        """返回详细统计信息（用于调试和分析）"""
        E_hat = self.energy_hat(E_avg)
        D_hat = self.D_succ_hat
        base = self.base_cost(E_hat, D_hat)
        risk = math.exp(self.kappa * self.v)
        penalty_budget = self.eta * max(0.0, self.v - 1.0)
        penalty_task = self.lam * self.rho_task
        score = base * risk + penalty_budget + penalty_task

        return {
            "E_avg": E_avg,
            "E_hat": E_hat,
            "D_succ": self.D_succ,
            "D_hat": D_hat,
            "rho_app": self.rho_app,
            "rho_task": self.rho_task,
            "v": self.v,
            "base": base,
            "risk": risk,
            "penalty_budget": penalty_budget,
            "penalty_task": penalty_task,
            "score": score,
        }

    def reset(self):
        """重置所有计数器"""
        self.app_timeout_cnt = 0
        self.app_arrived_cnt = 0
        self.succ_delay_sum = 0.0
        self.succ_cnt = 0

        # 重置子任务级统计
        self.task_timeout_cnt = 0
        self.task_total_cnt = 0

def clamp01(x):
    """限制在 [0, 1] 范围内"""
    return float(max(0.0, min(1.0, x)))

# ================= 统一 mask 工具函数 =================
NEG_INF = -1e9  # mask 中禁止动作的标记值

def mask_has_any_valid(mask):
    """
    检查 mask 是否有任何有效动作（非禁止）
    输入：mask 可以是 list/np.array/torch.Tensor
    输出：True 如果有至少一个有效动作
    """
    import numpy as np
    m = np.array(mask, dtype=np.float32)
    return bool((m > NEG_INF/2).any())

def mask_allows(mask, action: int):
    """
    检查 mask 是否允许某个动作
    输入：mask 可以是 list/np.array，action 是动作索引
    输出：True 如果该动作被允许
    """
    import numpy as np
    m = np.array(mask, dtype=np.float32)
    if action < 0 or action >= len(m):
        return False
    return bool(m[action] > NEG_INF/2)

def normalize_mask(mask, action_dim, force_fallback=None):
    """
    统一 mask 规范化函数：
    1. Binary mask (0/1): 1=合法, 0=非法 -> 转成 0/-1e9
    2. Additive mask (0/-1e9): 保持不变
    3. 修复全零 mask 解释为 additive mask（全可用）
    4. 如果全非法，强制允许 fallback_action (默认 Cloud=1)

    参数：
        mask: 原始 mask（可以是 list/np.array/torch.Tensor）
        action_dim: 期望的动作维度
        force_fallback: 如果全非法，强制允许的动作索引（默认 None，使用 Cloud=1）

    返回：
        规范化后的 mask（additive 格式，0/-1e9）
    """
    import numpy as np
    m = np.array(mask, dtype=np.float32)

    # Pad/trunc 到目标维度
    if len(m) != action_dim:
        if len(m) > action_dim:
            m = m[:action_dim]
        else:
            pad = np.zeros(action_dim - len(m))
            m = np.concatenate([m, pad])

    # ========== 修复正确判断 mask 类型 ==========
    m_min, m_max = float(m.min()), float(m.max())

    # 情况1：有 < -1e6 的值 → 明确是 additive mask
    if m_min < -1e6:
        # 已经是 additive mask，不转换
        pass

    # 情况2：有 > 0.5 的值 → 明确是 binary mask
    elif m_max > 0.5:
        # binary mask: 1=可用 -> 0, 0=禁用 -> -1e9
        m = np.where(m > 0.5, 0.0, NEG_INF).astype(np.float32)

    # 情况3：全零或接近零 → 修复解释为 additive mask（全可用）
    else:
        # 原来的 bug 在这里：把全零当作 binary mask 全禁用
        # 修复全零的 mask 应该解释为 additive mask（0=可用）
        # 不做任何转换，保持 [0, 0, 0, ...]
        pass

    # ========== 检查是否有合法动作 ==========
    has_valid = (m > NEG_INF/2).any()

    if not has_valid:
        # 真的全非法，强制允许 fallback 动作
        fallback_action = force_fallback if force_fallback is not None else 1  # 默认 Cloud
        fallback_action = max(0, min(fallback_action, action_dim - 1))

        # 【诊断】打印第一次全非法的 mask
        if not hasattr(normalize_mask, '_warned'):
            print(f"[MASK WARNING] 真正的全非法 mask，强制 fallback 到 action={fallback_action}")
            print(f"  raw mask={mask}")
            print(f"  action_dim={action_dim}")
            normalize_mask._warned = True

        m[:] = NEG_INF
        m[fallback_action] = 0.0

    return m

def compute_score(E_avg, D_succ_avg, rho_app, rho_task,
                  E_min=0.3, E_max=3.0, D_min=0.7, D_max=5.0,
                  phi1=0.5, phi2=0.5,  # 权重
                  kappa=2.0,           # 指数敏感度
                  eta=20.0,            # 违约额外罚分权重
                  lam=2.0,             # 子任务罚分权重
                  sla0=0.95,           # SLA=95%
                  v_cap=3.0,
                  v_cap_over=3.0):         # 阈值
    """
    【最终修复版】评分公式 - 二次方惩罚
    解决 "High Timeout = Low Score" 的逻辑悖论。
    策略：Quadratic Penalty (二次方惩罚) - 超时越高，罚分呈二次方爆炸式增长。
    """
    eps = 1e-12

    # 1. 基础成本 (0.0 ~ 1.0)
    e_hat = np.clip((E_avg - E_min) / max(E_max - E_min, eps), 0.0, 1.0)
    d_hat = np.clip((D_succ_avg - D_min) / max(D_max - D_min, eps), 0.0, 1.0)
    base_cost = phi1 * e_hat + phi2 * d_hat

    # 2. 违约强度 v
    delta = 1.0 - sla0  # 0.05
    # v=1 (5%), v=2 (10%), v=4 (20%), v=6 (30%)
    v = rho_app / max(delta, 1e-6)

    # 3. 风险因子 (Quadratic - 二次方)
    # 不再分段，直接用二次方，简单粗暴且有效
    # 基础风险 1.0 (未违约)
    # 违约后风险 = 1.0 + eta * (v-1)^2

    if v <= 1.0:
        # 安全区：只有基础成本
        score = base_cost
    else:
        # 危险区：二次方爆炸惩罚
        # 例如超时 20% (v=4) -> (4-1)^2 = 9 -> penalty = 20 * 9 = 180
        violation = v - 1.0
        penalty = eta * (violation ** 2)
        score = base_cost * (1.0 + violation) + penalty

    # 子任务罚分
    score += lam * rho_task

    return float(score)


# def compute_utility_score(E_avg, D_succ_avg, rho_app, rho_task,
#                          E_min=0.3, E_max=3.0, D_min=0.7, D_max=5.0,
#                          phi1=5.0/6.0, phi2=1.0/6.0,
#                          alpha=1.0, beta=2.0, sla0=0.95):
#     """
#     【新版】效用评分公式 - 越大越好 (Utility-Based Score)
#
#     设计思路：
#     1. 基础奖励: (1 - 综合成本) * 成功率
#     2. 惩罚项: 超时率的线性惩罚 (替代指数爆炸)
#     3. 范围: [-1.0, 1.0]，防止数值不稳定
#
#     公式：
#     Utility = SuccessUtility - Penalty
#
#     其中：
#     - SuccessUtility = (1 - rho_app) * (1 - cost)
#     - cost = phi1 * E_hat + phi2 * D_hat (归一化成本)
#     - Penalty = alpha * rho_app + beta * max(0, rho_app - (1 - sla0))
#
#     参数：
#         E_avg: 平均能耗
#         D_succ_avg: 成功任务平均时延
#         rho_app: 应用级超时率
#         rho_task: 子任务级超时率
#         E_min, E_max: 能耗归一化范围
#         D_min, D_max: 时延归一化范围
#         phi1, phi2: 权重（默认 5/6, 1/6）
#         alpha: 基础超时惩罚系数
#         beta: 超出SLA额外惩罚系数
#         sla0: 目标 SLA (默认 0.95)
#
#     返回：效用分数 (越大越好，范围 [-1.0, 1.0])
#     """
#     eps = 1e-12
#
#     # 强制 rho 范围，防止误用
#     rho_app = float(rho_app)
#     rho_task = float(rho_task)
#     if rho_app > 1.0:  rho_app /= 100.0
#     if rho_task > 1.0: rho_task /= 100.0
#     rho_app = float(np.clip(rho_app, 0.0, 1.0))
#     rho_task = float(np.clip(rho_task, 0.0, 1.0))
#
#     # 1) 归一化基础性能
#     e_hat = np.clip((E_avg - E_min) / max(E_max - E_min, eps), 0.0, 1.0)
#     d_hat = np.clip((D_succ_avg - D_min) / max(D_max - D_min, eps), 0.0, 1.0)
#     cost = phi1 * e_hat + phi2 * d_hat
#
#     # 2) 成功效用 (0~1)
#     # 如果 100% 成功且成本为0 -> 1.0
#     # 如果全超时 -> 0.0
#     success_utility = (1.0 - rho_app) * (1.0 - cost)
#
#     # 3) 失败惩罚 (0~2.0)
#     # 基础惩罚: 1.0 * rho_app
#     # 严重惩罚: 如果 rho_app > (1 - sla0) = 0.05，额外罚
#     over_sla_threshold = max(0.0, rho_app - (1.0 - sla0))
#     penalty = alpha * rho_app + beta * over_sla_threshold
#
#     # 4) 子任务惩罚
#     penalty += 0.5 * rho_task  # 轻度惩罚子任务超时
#
#     # 5) 总分
#     utility = success_utility - penalty
#
#     # 限制在 [-1, 1] 区间
#     utility = max(-1.0, min(1.0, utility))
#
#     return float(utility)

import numpy as np


def compute_utility_score(
    E_avg: float,
    D_succ_avg: float,
    rho_app: float,
    rho_task: float,
    *,
    # —— 归一化边界 ——
    E_min: float = 0.3,
    E_max: float = 3.0,
    D_min: float = 0.7,
    D_max: float = 5.0,
    # —— 能耗/时延权重 ——
    phi1: float = 5.0 / 6.0,   # 能耗权重
    phi2: float = 1.0 / 6.0,   # 时延权重
    # —— SLA 阈值（论文采用 5%）——
    sla0: float = 0.95,        # thr = 1 - sla0 = 0.05
    # —— SLA 违约惩罚（核心，强但可控）——
    w_app_base: float = 0.6,   # 对 rho_app 的基础惩罚
    w_over_max: float = 6.0,   # 超出 SLA 的额外惩罚上限（体现“极其重视超时”）
    tau: float = 0.02,         # Softplus 温度：越小越像 ReLU 拐点
    over_cap: float = 0.20,    # over 的绝对上限，防止惩罚爆炸/不敢探索
    # —— 次级目标（轻量）——
    w_task: float = 0.15,      # rho_task 权重（远小于 app）
    w_cost: float = 0.25,      # 能耗/时延综合成本权重（远小于 app）
    # —— 训练稳定性 / 课程 anneal ——
    progress: float = 1.0,     # 评估/最终计算恒为 1.0；训练时可 0→1 线性涨
    out_scale: float = 1.0,    # tanh 压缩尺度：越小越“硬”，越大越“软"
) -> float:
    """
    UtilityScore（v2 + SLA=5%）—— 越大越好，范围约 (-1, 1)。

    与论文 Table 1 / Table 2 口径逐位一致。公式见模块顶部 docstring。

    输入约定：
      - rho_app / rho_task 以【百分比】传入，例如 9.56 表示 9.56%，
        函数内部统一 /100 转小数。不要传 0.0956（会被再除一次 100）。
      - E_avg 为平均能耗（单位与 E_min/E_max 一致，默认 J 量级 0.3~3.0）。
      - D_succ_avg 为【所有应用】的平均完成时延（D_all 口径：超时/未完成
        应用按 per-app deadline 预算计入）。参数名保留 D_succ_avg 仅作
        向后兼容，论文主表实际传入的是 D_all。

    参数 progress：训练阶段可从 0 线性涨到 1（前期惩罚弱、允许探索）；
    做评估、出论文表格时必须保持默认 1.0。

    返回：float，范围约 (-1, 1)，越大越好。
    """
    eps = 1e-12

    # 0) rho 范围修正：CSV 中是百分比，统一 /100 转小数
    rho_app = float(rho_app) / 100.0
    rho_task = float(rho_task) / 100.0
    rho_app = float(np.clip(rho_app, 0.0, 1.0))
    rho_task = float(np.clip(rho_task, 0.0, 1.0))

    # 1) 基础性能成本（0~1）
    e_hat = np.clip((float(E_avg) - E_min) / max(E_max - E_min, eps), 0.0, 1.0)
    d_hat = np.clip((float(D_succ_avg) - D_min) / max(D_max - D_min, eps), 0.0, 1.0)
    cost = float(phi1 * e_hat + phi2 * d_hat)  # 0..1

    # 2) SLA 阈值
    delta = 1.0 - float(sla0)   # sla0=0.95 -> delta=0.05
    thr = delta

    # 3) 平滑的“超 SLA 部分” over（Softplus 近似 max(0, rho_app - thr)）
    z = (rho_app - thr) / max(tau, eps)
    over = tau * np.log1p(np.exp(z))
    over = float(min(over, over_cap))

    # 4) 课程式加权：训练初期惩罚弱（允许探索），后期惩罚强
    p = float(np.clip(progress, 0.0, 1.0))
    w_over = w_over_max * p

    # 5) utility 组成（越大越好）
    success = 1.0 - cost
    penalty_app = w_app_base * rho_app + w_over * over
    penalty_task = w_task * rho_task
    penalty_cost = w_cost * cost

    utility_raw = success - penalty_cost - penalty_task - penalty_app

    # 6) 平滑压缩（比硬 clip 更利于学习）
    utility = np.tanh(utility_raw / max(out_scale, eps))
    return float(utility)


# ================= 新版奖励函数 =================

class RewardCalculator:
    """
    【新版】奖励函数计算器

    支持两种奖励形式：
    1. 终止式 reward (episode-level): R_episode = -Score(θ)
    2. 密集 shaping reward: 每子任务即时惩罚
    """

    def __init__(self, phi1=5.0/6.0, phi2=1.0/6.0, kappa=3.0,
                 eta=5.0, lam=0.5, sla0=0.95, v_cap_over=2.0,
                 c_task=2.0, c_E=0.05, c_app=50.0):
        """
        参数：
            phi1, phi2: 权重（默认 5/6, 1/6）
            kappa: 风险敏感指数惩罚系数
            eta: Error budget 违约惩罚系数
            lam: 子任务级过程超时惩罚系数
            sla0: 目标 SLA
            v_cap: v 的上限
            c_task: 子任务超时惩罚系数（用于密集 shaping）修复从1.0提高到2.0
            c_E: 能耗即时惩罚系数修复从1.0大幅降低到0.05
            c_app: 应用级超时惩罚系数（用于终止式奖励）修复从10.0大幅提高到50.0
        """
        self.phi1 = phi1
        self.phi2 = phi2
        self.kappa = kappa
        self.eta = eta
        self.lam = lam
        self.sla0 = sla0
        self.delta = 1.0 - sla0
        self.v_cap_over = v_cap_over
        self.c_task = c_task
        self.c_E = c_E
        self.c_app = c_app

    def step_reward(self, task_info, action, energy, delay, is_timeout, bandwidth=5.0, is_app_timeout_event=False):
        """
        【密集 shaping reward】每步/每任务即时奖励

        R_step = r_task + r_E + (可选 r_app_end)

        其中：
            r_task(i,j) = -c_task · χ_i,j  (子任务超时惩罚)
            r_E(t) = -c_E · ΔE(t)  (能耗即时惩罚)
            r_app_event(i) = -c_app · I(触发应用超时事件)  (应用级超时事件惩罚)

        参数：
            is_app_timeout_event: 当前动作是否触发应用超时事件（App 从未超时变成已超时）
        """
        user_id, subtask_id = task_info

        # 1. 【方案 2：差异化超时惩罚】死在本地是重罪，死在外面是轻罪
        r_task = 0.0
        if is_timeout:
            if action == 0:  # Local Timeout（重罚）
                r_task = -self.c_task * 2.0  # 本地超时罚双倍 (-4.0)
            else:           # Edge/Cloud Timeout（轻罚）
                r_task = -self.c_task * 0.8  # 外面超时罚轻点 (-1.6)
        else:
            # 修复按时完成的任务给予正向奖励
            # 鼓励模型追求"完成任务"而不是"不做事"
            r_task = 0.5  # 基础完成奖励

        # 2. 能耗即时惩罚（保持极低）
        # 修复权重进一步降低到 0.01，让模型更敢于卸载
        r_E = -0.01 * energy

        # 3. 应用超时事件惩罚（修复真正让 c_app 生效）
        r_app = 0.0
        if is_app_timeout_event:
            r_app = -self.c_app

        # 4. 卸载引导奖励（修复加大引导）
        r_offload = 0.0
        if action > 0:  # 非本地计算（Cloud 或 Edge）
            r_offload = 2.0  # 大幅增加引导！只要肯出去就给糖吃

        # 5. 修复延迟惩罚：迫使 Agent 加快进度，降低 AppTO
        # 问题：TaskTO=0% 但 AppTO=45%，说明 Agent 只顾子任务不超时，不顾全局时间
        # 解决：重度惩罚延迟，让 Agent 时时刻刻都感到"时间就是分数"
        # 使用指数惩罚，延迟越大惩罚越狠
        r_delay = -0.1 * (delay ** 2)  # 延迟 0.1s -> -0.001, 0.5s -> -0.025, 1.0s -> -0.1

        # 总步奖励
        r_step = r_task + r_E + r_app + r_offload + r_delay

        return r_step

    def episode_reward(self, E_avg, rho_app, rho_task, D_succ_avg):
        """
        【终止式 reward】episode-level 奖励

        【新版】使用效用评分函数（越大越好）

        R_episode = Utility(θ)

        直接对齐最终优化目标，数值稳定在 [-1, 1] 范围内。

        注意：compute_utility_score 现在按论文口径 (v2 + SLA=5%) 实现，
        rho_app / rho_task 以【百分比】传入（函数内部 /100），不再接受
        alpha / beta 关键字（旧版参数已废弃）。
        """
        utility = compute_utility_score(
            E_avg=E_avg,
            D_succ_avg=D_succ_avg,
            rho_app=rho_app,
            rho_task=rho_task,
            phi1=self.phi1, phi2=self.phi2,
            sla0=self.sla0,
        )

        return utility  # 效用分数越大越好

    def app_end_reward(self, is_timeout, energy_avg):
        """
        【可选】应用结束时的额外奖励/惩罚

        R_app_end = -c_app · I(C_app > D_app) - c_E · (E_avg / E_max)

        Args:
            is_timeout: 应用是否超时
            energy_avg: 平均能耗
        """
        reward = 0.0

        # 应用超时惩罚
        if is_timeout:
            reward -= self.c_app

        # 能耗相对惩罚（归一化到 [0,1]）
        from utils.constant import para
        E_max = para.get("E_max", 3.0)  # 需要从外部传入或使用全局值
        e_norm = min(energy_avg / max(E_max, 1e-9), 1.0)
        reward -= self.c_E * e_norm

        return reward

    def hybrid_reward(self, step_rewards, E_avg, rho_app, rho_task, D_succ_avg):
        """
        【推荐】混合奖励：密集 shaping + 终止惩罚

        R_hybrid = Σ r_step(t) + R_app_end

        这样既有密集信号，又有严格的 SLA 约束。
        """
        # 1. 累加所有步奖励
        total_step_reward = sum(step_rewards)

        # 2. 添加应用级超时惩罚（如果应用超时）
        app_penalty = 0.0
        if rho_app > self.delta:
            app_penalty = -self.c_app * (rho_app / self.delta - 1.0)

        return total_step_reward + app_penalty

# ================= 常用工具函数 =================
def to_scalar(val, default=0.0):
    """安全地转换为标量，处理 inf 和 nan"""
    if isinstance(val, torch.Tensor):
        val = val.item()
    if isinstance(val, (np.ndarray, list, tuple)):
        if len(val) == 0: return default
        return to_scalar(val[0], default)
    if val is None or pd.isna(val):
        return default

    val = float(val)
    if np.isnan(val) or np.isinf(val):
        return default
    return val

def fix_task_size_units_inplace(ts):
    """
    修复强制转换任务大小单位 Bytes -> KB
    确保所有训练和评估的 RL 输入都是 KB 级别（而不是几十万级的 Bytes）
    """
    try:
        if hasattr(ts, "task_size"):
            # 采样检查是否需要转换
            sample = 0
            if isinstance(ts.task_size, list):
                if len(ts.task_size) > 0:
                    if isinstance(ts.task_size[0], list):
                        sample = ts.task_size[0][0] if len(ts.task_size[0]) > 0 else 0
                    else:
                        sample = ts.task_size[0]
            elif isinstance(ts.task_size, np.ndarray):
                sample = float(ts.task_size.flat[0])
            elif isinstance(ts.task_size, dict):
                sample = list(ts.task_size.values())[0]

            # 如果平均值 > 10000，说明是 Bytes，强制除以 1024
            if float(sample) > 10000:
                print(f"[System] 检测到 Task Size 为 Bytes 单位 (val={sample:.0f})，正在转换为 KB...")
                if isinstance(ts.task_size, list):
                    if isinstance(ts.task_size[0], list):
                        ts.task_size = [[x / 1024.0 for x in row] for row in ts.task_size]
                    else:
                        ts.task_size = [x / 1024.0 for x in ts.task_size]
                elif isinstance(ts.task_size, np.ndarray):
                    ts.task_size = ts.task_size / 1024.0
                elif isinstance(ts.task_size, dict):
                    for k in ts.task_size:
                        ts.task_size[k] = ts.task_size[k] / 1024.0

                ts._task_size_fixed = True
            elif float(sample) < 1:
                print(f"[System] ⚠️ 警告：检测到极小的 task_size (mean={sample:.4f})，可能是被重复除以1024了！")
            else:
                print(f"[System] 单位看起来正常 (KB, mean={sample:.0f})，无需修正。")
                ts._task_size_fixed = True
    except Exception as e:
        print(f"[WARN] fix_task_size_units_inplace failed: {e}")

def save_model_bundle(algo_name, agent, checkpoint_dir, **meta_kwargs):
    """
    Bundle v2: 可完整复现（模型+配置+计划+deadline+随机状态）

    Args:
        algo_name: 算法名称 ("DQN", "PPO", "GAT_PPO", "SATA", "RealGATPPO", "DynamicGAT_DQN")
        agent: 算法代理对象
        checkpoint_dir: 保存目录
        **meta_kwargs: 元数据（seed, state_dim, action_dim等）
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_dir = checkpoint_dir.parent  # RUN_DIR

    # 尝试读取 arrivals / deadlines，嵌入 bundle（保证只有.pt也能复现）
    arrival_data = None
    deadline_data = None
    try:
        arr_path = run_dir / "arrivals" / "arrival_seed0.json"
        if arr_path.exists():
            arrival_data = json.loads(arr_path.read_text(encoding="utf-8"))
    except Exception:
        arrival_data = None

    try:
        dl_path = run_dir / "configs" / "deadlines.json"
        if dl_path.exists():
            deadline_data = json.loads(dl_path.read_text(encoding="utf-8"))
    except Exception:
        deadline_data = None

    # 模型权重抽取：兼容 DQN / PPO / GAT-PPO 系列
    model_state = {}
    optim_state = {}

    # DQN 系列
    if hasattr(agent, "q_net"):
        model_state["q_net"] = agent.q_net.state_dict()
        if hasattr(agent, "target_net"):
            model_state["target_net"] = agent.target_net.state_dict()
        if hasattr(agent, "optimizer"):
            optim_state["optimizer"] = agent.optimizer.state_dict()

    # PPO 系列
    if hasattr(agent, "actor"):
        model_state["actor"] = agent.actor.state_dict()
        if hasattr(agent, "critic"):
            model_state["critic"] = agent.critic.state_dict()
        if hasattr(agent, "actor_optimizer"):
            optim_state["actor_optimizer"] = agent.actor_optimizer.state_dict()
        if hasattr(agent, "critic_optimizer"):
            optim_state["critic_optimizer"] = agent.critic_optimizer.state_dict()

    # GAT-PPO 系列
    if hasattr(agent, "policy"):
        model_state["policy"] = agent.policy.state_dict()
        model_key = "policy"
    elif hasattr(agent, "q"):  # SATA/DynamicGAT_DQN 兼容
        model_state["q"] = agent.q.state_dict()
        model_key = "q"

    # 保存训练状态
    train_state = {}
    if hasattr(agent, "step_count"):
        train_state["step_count"] = agent.step_count
    if hasattr(agent, "epsilon"):
        train_state["epsilon"] = agent._get_epsilon() if hasattr(agent, "_get_epsilon") else getattr(agent, "epsilon", 0.0)

    bundle = {
        "bundle_version": 2,
        "algo": algo_name,
        "created_at": time.time(),
        "meta": dict(meta_kwargs),

        # 快照（关键：保证 pt 单文件可复现）
        "CONFIG_snapshot": copy.deepcopy(CONFIG),
        "para_snapshot": copy.deepcopy(para),
        "arrival_data": arrival_data,
        "deadline_data": deadline_data,

        # 随机状态（可复现训练过程）
        "rng_state": _get_rng_state(),

        # 模型与优化器
        "model_state": model_state,
        "optim_state": optim_state,
        "train_state": train_state,
    }

    # 保存 bundle
    bundle_path = checkpoint_dir / f"{algo_name}.pt"
    torch.save(bundle, bundle_path)
    print(f"[{algo_name}] Bundle v2已保存到: {bundle_path}")
    print(f"[{algo_name}] Meta: {meta_kwargs}")

    # 兼容旧格式：保存简单 .pth（只放主要网络）
    try:
        if "q_net" in model_state:
            torch.save(model_state["q_net"], checkpoint_dir / f"{algo_name.lower()}_model.pth")
        elif "actor" in model_state:
            torch.save(model_state["actor"], checkpoint_dir / f"{algo_name.lower()}_model.pth")
        elif "policy" in model_state:
            torch.save(model_state["policy"], checkpoint_dir / f"{algo_name.lower()}_model.pth")
    except Exception as e:
        print(f"[{algo_name}] 兼容旧格式保存失败(忽略): {e}")

    return bundle_path


def _get_rng_state():
    """获取所有随机数生成器状态"""
    s = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().cpu(),
    }
    if torch.cuda.is_available():
        s["torch_cuda_all"] = [x.cpu() for x in torch.cuda.get_rng_state_all()]
    else:
        s["torch_cuda_all"] = None
    return s

def get_task_size_bytes(ts, user_id, subtask_id):
    """
    安全地获取任务大小（Bytes），兼容不同的 task_size 结构
    修复禁用了 fix_task_size_units_inplace，所以值始终是 Bytes，直接返回
    """
    s = getattr(ts, "task_size", None)
    if s is None:
        return 0.0
    try:
        # 直接获取原始值（已经是 Bytes）
        if isinstance(s, dict):
            v = s.get(subtask_id, list(s.values())[0])
        elif isinstance(s, (list, tuple, np.ndarray)):
            if len(s) == 0:
                return 0.0
            if isinstance(s[0], (list, tuple, np.ndarray)):  # [user][subtask]
                v = s[user_id][subtask_id]
            else:  # [subtask]
                v = s[subtask_id]
        else:
            v = s

        return float(v)  # 直接返回，已经是 Bytes
    except Exception:
        return 0.0

def subtask_outcome_stats(ts):
    """
    修复统计子任务级别的完成情况（返回总子任务数、完成数、未完成数、超时数）

    返回值：
        - total: 所有已到达用户的子任务总数
        - finished: 已完成的子任务数（finish_time != inf）
        - unfinished: 未完成的子任务数（finish_time == inf）
        - timeout: 超时的子任务数（包括超时完成的和未完成的）
    """
    total = finished = unfinished = timeout = 0
    eps = getattr(ts, "TIMEOUT_EPSILON", 1e-6)  # 添加 epsilon，避免卡线误判
    for uid in range(ts.user_num):
        if ts.enter_time[uid] == float("inf"):
            # 该用户没有到达，跳过
            continue

        # 获取子任务 deadline
        try:
            task_deadline_abs = ts.enter_time[uid] + ts.get_task_deadline_slot(uid) * para["slot_interval"]
        except:
            # 如果获取不到 deadline，跳过该用户
            continue

        for sid, ft in ts.finish_time[uid].items():
            total += 1
            if ft != float("inf"):
                finished += 1
                # 检查是否超时完成（添加 epsilon，避免卡线误判）
                if ft > task_deadline_abs + eps:
                    timeout += 1
            else:
                # 未完成的子任务
                unfinished += 1
                # 未完成的子任务也算超时（必然无法在 deadline 前完成）
                timeout += 1
    return total, finished, unfinished, timeout

def subtask_partition_stats(ts, task2action):
    """
    统计子任务最终分配到 Local/Cloud/Edge/Timeout 的分布
    根据 finish_time 判断是否完成，结合 task2action 判断分配位置

    参数：
        - ts: TaskScheduler 对象
        - task2action: dict[(uid, sid) -> action]

    返回：
        dict 包含：
            - total_subtasks: 总子任务数（已到达用户的子任务数）
            - local: 完成且分配到 Local 的子任务数
            - cloud: 完成且分配到 Cloud 的子任务数
            - edge: 完成且分配到 Edge 的子任务数
            - timeout: 未完成的子任务数（finish_time == inf）
            - timeout_scheduled: 超时但已被调度的子任务数（在 task2action 中有记录）
            - timeout_unscheduled: 从未被调度就超时的子任务数（在 task2action 中无记录）
            - unknown: 完成但未记录动作的子任务数
    """
    local = cloud = edge = timeout = timeout_scheduled = timeout_unscheduled = unknown = 0
    total = 0

    for uid in range(ts.user_num):
        if ts.enter_time[uid] == float("inf"):
            continue  # 未到达应用不计入 total

        for sid, ft in ts.finish_time[uid].items():
            total += 1

            if ft == float("inf"):
                # 这个子任务没有完成：统一算作 Timeout
                timeout += 1
                # 区分"超时但已被调度"和"从未被调度就超时"
                if (uid, sid) in task2action:
                    # 超时但曾被调度
                    timeout_scheduled += 1
                else:
                    # 从未被调度就超时
                    timeout_unscheduled += 1
                continue

            # 完成的子任务：看它当时被分配到哪里
            a = task2action.get((uid, sid), None)
            if a is None:
                unknown += 1
            elif a == 0:
                local += 1
            elif a == 1:
                cloud += 1
            else:
                edge += 1

    return {
        "total_subtasks": total,
        "local": local,
        "cloud": cloud,
        "edge": edge,
        "timeout": timeout,
        "timeout_scheduled": timeout_scheduled,
        "timeout_unscheduled": timeout_unscheduled,
        "unknown": unknown
    }

def calc_timeout_rate(ts):
    """
    修复计算超时率 - 关键：未完成 = 超时

    问题根源：
    - RL 算法通过不完成任务逃避惩罚，导致虚假的低超时率
    - 必须将仿真结束时所有未完成的应用强制计入超时
    """
    try:
        # 1. 应用级超时统计
        total_arrived = sum(1 for uid in range(ts.user_num) if ts.enter_time[uid] != float("inf"))
        total_arrived = max(1, total_arrived)

        app_finished = len(getattr(ts, 'application_finished', set()))
        app_timeout = len(getattr(ts, 'application_timeout_finished', set()))

        # 修复计算未完成的应用数（强制算作超时）
        # 修复原因：RL 算法通过拖延不完成任务来逃避超时惩罚
        unfinished_apps = total_arrived - app_finished - app_timeout
        total_timeout = app_timeout + unfinished_apps  # 未完成也算超时！

        app_timeout_rate = total_timeout / total_arrived

        # 2. 子任务级统计
        total_subtasks, finished_subtasks, unfinished_subtasks, timeout_subtasks = subtask_outcome_stats(ts)
        task_timeout_rate = timeout_subtasks / total_subtasks if total_subtasks > 0 else 0.0

        return {
            'combined': app_timeout_rate,
            'app_timeout_rate': app_timeout_rate,
            'app_finished_count': app_finished,
            'app_timeout_count': app_timeout,
            'unfinished_apps': unfinished_apps,  # 未完成应用数
            'subtask_stats': {
                'total': total_subtasks,
                'finished': finished_subtasks,
                'unfinished': unfinished_subtasks,
                'timeout': timeout_subtasks
            },
            'task_timeout_rate': task_timeout_rate
        }
    except Exception as e:
        return {
            'combined': 1.0,
            'app_timeout_rate': 1.0,
            'app_timeout_count': 0,
            'app_finished_count': 0,
            'unfinished_apps': 0,  # 
            'subtask_stats': {
                'total': 0,
                'finished': 0,
                'timeout': 0
            },
            'task_timeout_rate': 1.0
        }

def get_simple_timeout_rate(ts):
    """简化版：只返回数值"""
    result = calc_timeout_rate(ts)
    return result['combined'] if isinstance(result, dict) else result

def calc_application_timeout_rate(ts):
    """保留旧函数名用于兼容性"""
    return get_simple_timeout_rate(ts)

# NEG_INF 已在上面第 215 行统一定义为 -1e9，此处重复定义已删除

def _mask_to_valid(mask, action_dim=None):
    """
    修复兼容两种 mask：
    - additive mask: 0 可用 / -1e9 禁用
    - binary mask: 1 可用 / 0 禁用
    返回: valid(bool array)

    修复关键：全零 mask 解释为 additive mask（全可用）
    """
    m = mask
    if hasattr(m, "detach"):  # torch.Tensor
        m = m.detach().cpu().numpy()
    m = np.asarray(m, dtype=np.float32).reshape(-1)

    if action_dim is not None and m.shape[0] != action_dim:
        # 维度不匹配，保守起见：全可用
        return np.ones((action_dim,), dtype=bool)

    m_min, m_max = float(m.min()), float(m.max())

    # 情况1：有 < -1e6 的值 → additive mask
    if m_min < -1e6:
        return m > NEG_INF

    # 情况2：有 > 0.5 的值 → binary mask
    elif m_max > 0.5:
        return m > 0.5

    # 情况3：全零 → 修复解释为 additive mask（全可用）
    else:
        return np.ones_like(m, dtype=bool)

def get_expert_action(env, ts, user_id, subtask_id, task_complex, slot):
    """
    修复版专家决策：
    1. 兼容 additive mask (-1e9/0)
    2. 确保 KB 单位正确
    3. 增加对队列拥塞的简单感知
    """
    task_size_bytes = get_task_size_bytes(ts, user_id, subtask_id)
    task_size_kb = task_size_bytes / 1024.0

    # 尽量统一 slot_interval 来源（你主工程一般用 para["slot_interval"]）
    slot_interval = None
    if hasattr(env, "env_para") and isinstance(env.env_para, dict) and "slot_interval" in env.env_para:
        slot_interval = float(env.env_para["slot_interval"])
    else:
        # 保底：用 ts 或 para 的 slot_interval（按你工程可改）
        slot_interval = 0.01  # 你日志里 now=0.01 对应的就是这个量级

    now_time = slot * slot_interval
    base_mask = ts.get_action_mask(user_id, task_size_bytes, now_time)

    action_dim = len(base_mask)
    valid = _mask_to_valid(base_mask, action_dim=action_dim)

    # 正确的"没有任何可用动作"判断
    if not valid.any():
        return 1  # fallback Cloud

    # 可用集合
    local_ok = bool(valid[0])
    cloud_ok = bool(valid[1]) if action_dim > 1 else False
    edge_actions = [i for i in range(2, action_dim) if valid[i]]

    # 选负载最低的 Edge（如果有 edge_useful）
    def pick_best_edge(edge_list):
        if not edge_list:
            return None
        try:
            return min(edge_list, key=lambda a: sum(1 for x in ts.edge_useful[a - 2] if x != 0))
        except Exception:
            return edge_list[0]

    # 规则：小任务优先 Local，中任务优先 Edge，大任务优先 Cloud
    if task_size_kb < 400:
        if local_ok:
            return 0
        best_edge = pick_best_edge(edge_actions)
        if best_edge is not None:
            return int(best_edge)
        return 1 if cloud_ok else 0

    elif task_size_kb < 650:
        best_edge = pick_best_edge(edge_actions)
        if best_edge is not None:
            return int(best_edge)
        if local_ok:
            return 0
        return 1 if cloud_ok else 0

    else:
        if cloud_ok:
            return 1
        best_edge = pick_best_edge(edge_actions)
        if best_edge is not None:
            return int(best_edge)
        return 0

# ================= 图相关辅助函数 =================
def graph_state_to_vector(state_data, method='mean'):
    """
    将图状态(Data对象或Tensor)统一转换为 CPU 上的 1D Tensor
    """
    if state_data is None:
        return torch.zeros(32, dtype=torch.float32)

    if hasattr(state_data, 'x') and state_data.x is not None:
        x = state_data.x
        if x is None or x.numel() == 0:
            return torch.zeros(32, dtype=torch.float32)
        if x.dim() == 2:
            if method == 'max':
                return x.max(dim=0)[0].cpu()
            else:  # 默认 'mean'
                return x.mean(dim=0).cpu()
        elif x.dim() == 1:
            return x.view(-1).cpu()
        else:
            return x.view(-1).cpu()
    elif isinstance(state_data, torch.Tensor):
        if state_data.dim() == 2:
            if method == 'max':
                return state_data.max(dim=0)[0].cpu()
            else:
                return state_data.mean(dim=0).cpu()
        return state_data.view(-1).cpu()
    else:
        try:
            return torch.tensor(state_data, dtype=torch.float32).view(-1)
        except:
            return torch.zeros(32, dtype=torch.float32)

def get_feature_dim(env, gs, task_complex_index):
    """
    获取固定的特征维度（使用全局池化后的维度）
    修复task_complex_index 必须是 int，不能传 list
    """
    # 修复强制转为 int，防止误传 list
    if isinstance(task_complex_index, (list, tuple, np.ndarray)):
        task_complex_index = int(getattr(env, "task_complex_index", 0))
    else:
        task_complex_index = int(task_complex_index)

    try:
        dummy_state = gs.get_graph_state_new(env, (0, 0), task_complex_index)
        vec = graph_state_to_vector(dummy_state)
        return vec.shape[0]
    except Exception as e:
        print(f"[Warning] get_feature_dim failed: {e}")
        return 32  # 默认维度

def get_graph_cache(user_num, subgraph_num, basegraph_num, project_root):
    """获取图缓存，使用绝对路径确保多进程一致性"""
    cache_dir = Path(project_root) / "cache_graph"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # 【扩展】当使用 MATRIX_OVERRIDE_PATH 时，在缓存key中加入路径hash，避免缓存污染
    _override = os.environ.get("MATRIX_OVERRIDE_PATH", "")
    _suffix = f"_{hashlib.md5(_override.encode()).hexdigest()[:8]}" if _override else ""
    cache_file = cache_dir / f"graph_cache_u{user_num}_s{subgraph_num}_b{basegraph_num}{_suffix}.pkl"

    if cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass

    import networkx as nx

    # 有MATRIX_OVERRIDE_PATH时返回None，由Environment自行读取matrix文件（避免随机图覆盖）
    if _override and os.path.exists(_override):
        print(f"[get_graph_cache] 有MATRIX_OVERRIDE_PATH，跳过缓存/随机生成，使用Environment自身解析的图")
        return None

    import utils.generate_graph
    edge_list, vertex_list = utils.generate_graph.generate_graph(basegraph_num, user_num, subgraph_num)
    G = nx.Graph()
    G.add_nodes_from(vertex_list)
    G.add_edges_from(edge_list)
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(G, f)
    except Exception:
        pass
    return G

# ================= IO 函数 =================
def generate_arrival_plan(seed: int, max_steps: int, stop_step: int,
                          base_prob: float, burst_prob: float, burst_min: int, burst_max: int):
    """
    【精确控制版】生成到达计划，确保正好有 para['user_num'] 个用户到达

    Returns:
        dict: {
            "plan": List[int],  # 每个 slot 的到达数量（兼容旧格式）
            "schedule": List[List[int]],  # 每个 slot 的 uid 列表（新格式，可复现）
            "env_seed": seed  # 环境种子，用于复现
        }
    """
    from utils.constant import para  # 确保能读到 user_num
    target_users = int(para["user_num"])  # 目标：150个用户

    rng = random.Random(seed)
    plan = [0] * max_steps
    schedule = [[] for _ in range(max_steps)]  # 每个 slot 的 uid 列表

    # 1) 先把 uid 打乱，保证可复现
    uids = list(range(target_users))
    rng.shuffle(uids)

    # 2) 选择"允许到达"的 slot 范围
    horizon = max(1, min(stop_step, max_steps))

    # 3) 基础：让到达大体均匀铺开（不会不足）
    #    例如把 150 个用户均匀分到 horizon 里
    for i, uid in enumerate(uids):
        slot = int(i * horizon / target_users)  # 均匀映射到 [0, horizon)
        schedule[slot].append(uid)
        plan[slot] += 1

    # 4) 可选：加突发（把部分 uid 往前挪，制造拥塞峰）
    #    注意：这一步不会改变"总人数"，只是重排 slot
    if burst_prob > 0:
        for slot in range(horizon):
            if rng.random() < burst_prob:
                extra = rng.randint(burst_min, burst_max)
                # 从后面 slot 抽 extra 个 uid 挪到当前 slot（保证总人数不变）
                pulled = 0
                pull_slot = horizon - 1
                while pulled < extra and pull_slot > slot:
                    # 从 pull_slot 抽取 uid（非空且不是当前 slot）
                    while pull_slot > 0 and not schedule[pull_slot] and pulled < extra:
                        pull_slot -= 1

                    if pull_slot > slot and schedule[pull_slot]:
                        uid = schedule[pull_slot].pop()
                        plan[pull_slot] -= 1
                        schedule[slot].append(uid)
                        plan[slot] += 1
                        pulled += 1
                    pull_slot -= 1

    return {"plan": plan, "schedule": schedule, "env_seed": seed}


def make_offline_arrival_plan(seed: int, max_steps: int, target_users: int = None):
    """【离线 DAG 模式】生成"全部用户在 slot=0 一次性到达"的到达计划。

    用途: 差异点 1 实验"DAG 在开始前全部给定 (offline) vs DAG 只有在应用到达时
    加入系统 (online)"对比。配合 save_arrival_plan_to_run_dir() 写入 RUN_DIR
    后, 任何走 load_arrival_plan() 的 wrapper 都会自动用上 offline 计划,
    无需修改 wrapper 内部逻辑。

    Args:
        seed: 用于复现 (本身 offline 计划是确定性的, 但 env_seed 仍要透传)
        max_steps: episode 总 slot 数
        target_users: 默认读 para["user_num"]

    Returns:
        dict: {"plan": [N,0,0,...], "schedule": [[0..N-1],[],...], "env_seed": seed}
    """
    from utils.constant import para
    if target_users is None:
        target_users = int(para["user_num"])
    plan = [0] * max_steps
    schedule = [[] for _ in range(max_steps)]
    plan[0] = target_users
    schedule[0] = list(range(target_users))
    return {"plan": plan, "schedule": schedule, "env_seed": seed}


def save_arrival_plan_to_run_dir(run_dir: str, seed_offset: int, arrival_plan: dict,
                                 env_seed: int = None, max_steps: int = None,
                                 stop_step: int = 0, base_prob: float = 0.0,
                                 burst_prob: float = 0.0, burst_min: int = 0,
                                 burst_max: int = 0):
    """把 arrival_plan dict 写到 RUN_DIR/arrivals/arrival_seed{seed_offset}.json,
    供 wrapper 内部 load_arrival_plan() 自动读取。

    格式与 load_arrival_plan() 期待对齐: arrival_plan 字段保留完整 dict
    (含 plan + schedule), 这样 apply_arrival_plan 走 schedule 路径精确注入。
    """
    p = Path(run_dir) / "arrivals" / f"arrival_seed{seed_offset}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "env_seed": env_seed if env_seed is not None else arrival_plan.get("env_seed", 0),
        "arrival_plan": arrival_plan,  # dict with plan + schedule
        "max_steps": max_steps,
        "stop_step": stop_step,
        "base_prob": base_prob,
        "burst_prob": burst_prob,
        "burst_min": burst_min,
        "burst_max": burst_max,
    }
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(p)


def apply_arrival_plan(ts, slot, plan):
    """
    应用到达计划到调度器（兼容多种格式）
    支持：
      1) plan 是 {"schedule":[[uids...],...], "plan":[counts...], ...}  -> 固定uid到达（强一致）
      2) plan 是 {"arrival_plan": {...}} / {"arrival_plan_dict": {...}} -> 外层包装
      3) plan 是 List[int] -> 只按数量到达（uid随机）
      4) plan 是 {"arrival_plan": List[int]} -> 外层包装
    """
    if plan is None:
        return

    # ---- 1) 处理外层包装 dict：把里面真正的 plan 抽出来 ----
    if isinstance(plan, dict):
        if "arrival_plan_dict" in plan and isinstance(plan["arrival_plan_dict"], dict):
            plan = plan["arrival_plan_dict"]
        elif "arrival_plan" in plan:
            # arrival_plan 可能是 dict(plan+schedule) 或 list[int]
            plan = plan["arrival_plan"]

    # ---- 2) dict + schedule：固定 uid 到达（推荐）----
    if isinstance(plan, dict):
        schedule = plan.get("schedule", None)
        if isinstance(schedule, list):
            if 0 <= slot < len(schedule) and schedule[slot]:
                ts.new_arrival(1, slot, fixed_uids=schedule[slot])
            return

        # 如果 dict 没 schedule，但有 plan(list[int])，退化成按数量
        counts = plan.get("plan", None)
        if isinstance(counts, list):
            if 0 <= slot < len(counts):
                k = int(counts[slot])
                for _ in range(k):
                    ts.new_arrival(1, slot)
            return

        # dict 既没 schedule 也没 plan：直接不做
        return

    # ---- 3) list[int]：按数量到达（uid随机）----
    if isinstance(plan, list):
        if 0 <= slot < len(plan):
            k = int(plan[slot])
            for _ in range(k):
                ts.new_arrival(1, slot)
        return

def get_arrived_apps(ts):
    """获取所有已到达的应用（等待/开始/完成/超时都算）"""
    arrived = set()
    arrived |= set(getattr(ts, "application_waiting", []))
    arrived |= set(getattr(ts, "application_started", set()))
    arrived |= set(getattr(ts, "application_finished", set()))
    arrived |= set(getattr(ts, "application_timeout_finished", set()))
    return arrived

def all_arrived_done(ts):
    """检查所有已到达的应用是否都已完成（完成或超时）"""
    arrived = get_arrived_apps(ts)
    done = set(getattr(ts, "application_finished", set())) | set(getattr(ts, "application_timeout_finished", set()))
    return len(arrived) > 0 and len(arrived - done) == 0

def safe_rest_tasks_total(rest_tasks):
    """安全计算剩余任务总数（处理不同的数据结构）"""
    if isinstance(rest_tasks, (list, tuple, np.ndarray)):
        return int(np.sum(rest_tasks))
    elif isinstance(rest_tasks, dict):
        return int(sum(rest_tasks.values()))
    else:
        return 0

def _to_int_safe(x):
    """安全地将各种类型转换为 int，用于 arrival_plan 净化"""
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, float):
        return int(x)
    if isinstance(x, str):
        s = x.strip()
        if s == "":
            return 0
        try:
            return int(float(s))  # 兼容 "3" / "3.0"
        except Exception:
            return 0
    if isinstance(x, (list, tuple, np.ndarray)):
        return int(len(x))       # 兼容旧格式：slot里存的是到达对象列表
    return 0

def load_arrival_plan(run_dir: str, seed_offset: int):
    """
    加载到达计划（包含环境种子）

    Returns:
        dict: 包含 env_seed, arrival_plan, max_steps, stop_step, base_prob, burst_prob, burst_min, burst_max
        或 None 如果文件不存在
    """
    p = Path(run_dir) / "arrivals" / f"arrival_seed{seed_offset}.json"
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        # 如果是旧格式（只有列表），返回 None（需要兼容处理）
        if isinstance(data, list):
            print(f"[警告] 加载到旧格式的到达计划（仅列表），无法获取环境种子")
            return {"arrival_plan": data, "env_seed": None}

        # 强制把 arrival_plan 转成 list[int]
        plan = data.get("arrival_plan", [])

        # 处理新格式：arrival_plan 是 {"plan": [...], "schedule": [...]}
        if isinstance(plan, dict):
            # 修复保留完整 dict，包括 schedule 字段，供 apply_arrival_plan 使用
            if "plan" in plan:
                # 确保所有元素都是 int
                data["arrival_plan_dict"] = plan  # 保留完整 dict
                data["arrival_plan"] = [_to_int_safe(v) for v in plan["plan"]]  # 旧字段兼容
            else:
                data["arrival_plan_dict"] = plan
                data["arrival_plan"] = []
        elif isinstance(plan, list):
            # 旧格式，包装成 dict 以兼容
            data["arrival_plan_dict"] = {"plan": plan}
            data["arrival_plan"] = [_to_int_safe(v) for v in plan]
        else:
            print(f"[警告] arrival_plan 格式异常: {type(plan)}")
            data["arrival_plan_dict"] = {"plan": []}
            data["arrival_plan"] = []

        return data
    else:
        return None

def make_run_dir(base_dir: str, algo_name: str = None, seed: int = None) -> Path:
    """创建统一的结果目录结构

    Args:
        base_dir: 基础目录
        algo_name: 算法名称（可选，如果只运行一个RL算法时传入）
        seed: 随机种子（可选）
    """
    ts = datetime.now().strftime("%m%d%H%M%S")

    # 如果提供了算法名和seed，则使用新的命名格式：run{algo}_seed{seed}_{ts}
    if algo_name and seed is not None:
        run_dir = Path(base_dir) / f"run{algo_name}_seed{seed}_{ts}"
    else:
        run_dir = Path(base_dir) / f"run_{ts}"

    (run_dir / "curves").mkdir(parents=True, exist_ok=True)
    (run_dir / "traces").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    (run_dir / "arrivals").mkdir(parents=True, exist_ok=True)
    (run_dir / "configs").mkdir(parents=True, exist_ok=True)
    return run_dir

# ================= 曲线记录常量 =================
CURVE_FIELDS = [
    "episode", "phase",
    "reward", "energy", "delay",
    "app_timeout_rate", "task_timeout_rate",
    "score", "utility_score",
    "epsilon", "avg_loss",
    "note"
]

def safe_append_curve_row(path: Path, row: dict, retry: int = 10, sleep: float = 0.15):
    """
    安全写入CSV（Windows下防文件锁）
    支持 phase 字段区分 train/eval/eval_final
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 补齐字段，确保列稳定
    full = {k: row.get(k, "") for k in CURVE_FIELDS}
    header_needed = (not path.exists()) or (path.stat().st_size == 0)

    for i in range(retry):
        try:
            with open(path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=CURVE_FIELDS)
                if header_needed:
                    w.writeheader()
                    header_needed = False
                w.writerow(full)
            return True
        except PermissionError:
            time.sleep(sleep * (i + 1))
        except Exception as e:
            print(f"[WARN] safe_append_curve_row 失败: {path}, error: {e}")
            return False
    return False

def append_curve_row(path: Path, row: dict, max_retries=3):
    """追加写入训练曲线数据，增加重试机制和错误处理"""
    for attempt in range(max_retries):
        try:
            header = not path.exists()
            df = pd.DataFrame([row])
            df.to_csv(path, mode="a", header=header, index=False)
            return
        except PermissionError as e:
            if attempt < max_retries - 1:
                time.sleep(0.1 * (attempt + 1))  # 递增延迟
            else:
                print(f"[WARN] append_curve_row 失败: {path}")
                print(f"[WARN] 文件可能被占用（如 Excel 打开）或多个进程同时写入")
                print(f"[WARN] 丢失数据: {row}")
        except Exception as e:
            print(f"[WARN] append_curve_row 异常: {path}, error: {e}")
            break

def atomic_to_csv(df: pd.DataFrame, path: Path):
    """
    原子操作写入CSV（避免并发写入损坏）
    改进：增加错误处理，避免文件被占用时崩溃
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    # 修复使用 float_format 确保百分比等浮点数正确显示
    df.to_csv(tmp, index=False, float_format='%.4f')
    try:
        if path.exists():
            path.unlink()
        tmp.replace(path)
    except PermissionError:
        print(f"[WARN] 无法写入 {path}，文件可能被占用（如 Excel 打开）")
        print(f"[WARN] 数据已保存至临时文件: {tmp}")
    except Exception as e:
        print(f"[WARN] 写入 {path} 失败: {e}")
        print(f"[WARN] 数据已保存至临时文件: {tmp}")

# ================= 初始化和种子函数 =================
def seed_everything(seed: int):
    """eval/复现实验专用：显式设置随机种子（不是seed_offset）"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def init_worker(seed_offset, para, CONFIG):
    """Worker进程初始化，设置随机种子和para参数"""
    seed = CONFIG["SEED"] + seed_offset
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # 修复如果 para["local_power"] 已经存在且长度足够，则不重新生成
    if "local_power" not in para or len(para.get("local_power", [])) < para["user_num"]:
        # 修复强制重生成确定的 local_power，确保所有进程一致
        env_seed = CONFIG["SEED"]
        rng = random.Random(env_seed)
        if "local_power_range" in para:
            low, high = para["local_power_range"]
            para["local_power"] = [rng.uniform(low, high) * (10 ** 9) for _ in range(para["user_num"])]
        else:
            para["local_power"] = [rng.uniform(0.3e9, 0.6e9) for _ in range(para["user_num"])]

# ================= 其他辅助函数 =================
def safe_rest_tasks_total(rest_tasks) -> int:
    """
    安全地把 rest_tasks（可能是 list/np.ndarray/标量）转成总剩余任务数(int)
    """
    try:
        v = np.sum(rest_tasks)
        if isinstance(v, np.ndarray):
            v = v.item()
        return int(v)
    except Exception:
        try:
            return int(rest_tasks)
        except Exception:
            return 0

def safe_action_to_int(action) -> int:
    """
    安全地把 action（可能是 int/list/np.ndarray）转成 int
    """
    try:
        if isinstance(action, (list, tuple, np.ndarray)):
            if len(action) > 0:
                return int(action[0])
            else:
                return 0
        else:
            return int(action)
    except Exception:
        return 0

def diagnose_timeout(ts, algo_name=""):
    """诊断函数"""
    timeout_info = calc_timeout_rate(ts)
    print(f"\n=== [{algo_name}] 超时诊断 ===")
    print(f"  user_num: {ts.user_num}")
    print(f"  application_finished: {timeout_info['app_finished_count']}")
    print(f"  application_timeout_finished: {timeout_info['app_timeout_count']}")
    print(f"  rest_tasks: {sum(getattr(ts, 'rest_tasks', []))}")
    print(f"  应用超时率: {timeout_info['app_timeout_rate']:.2%}")
    print("=" * 40)
    sys.stdout.flush()

def resolve_model_run_dir(model_run_dir: str, model_root_dir: str, seed: int, auto_pick_latest: bool = True) -> Path:
    """
    解析模型目录：
    - 如果传了 model_run_dir：直接用
    - 否则从 model_root_dir 下挑最新的 seed{seed}_* 文件夹
    """
    if model_run_dir:
        p = Path(model_run_dir)
        if not p.exists():
            raise FileNotFoundError(f"MODEL_RUN_DIR 不存在: {p}")
        return p

    root = Path(model_root_dir)
    if not root.exists():
        raise FileNotFoundError(f"MODEL_ROOT_DIR 不存在: {root}")

    if not auto_pick_latest:
        raise ValueError("未指定 MODEL_RUN_DIR，且 AUTO_PICK_LATEST=False，无法确定要加载哪个目录")

    prefix = f"seed{seed}_"
    candidates = sorted(
        [d for d in root.iterdir() if d.is_dir() and d.name.startswith(prefix)],
        key=lambda x: x.name,
        reverse=True
    )
    if not candidates:
        raise FileNotFoundError(f"在 {root} 下未找到 {prefix}* 目录")
    return candidates[0]

# ================= 新增：模型加载与兼容性工具 (供 huanjing.py 使用) =================

def torch_load_compat(path, map_location=None):
    """兼容旧版/新版 PyTorch 的 torch.load"""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # 旧版 PyTorch：没有 weights_only 参数
        return torch.load(path, map_location=map_location)

def strip_prefix(state_dict):
    """去除 module. 前缀 (用于 DDP 保存的模型)"""
    if not isinstance(state_dict, dict) or not state_dict:
        return state_dict
    return {k.replace('module.', ''): v for k, v in state_dict.items()}

def weight_fingerprint(model):
    """打印模型权重指纹，用于验证加载是否成功"""
    try:
        for p in model.parameters():
            w = p.detach().float().cpu()
            return float(w.mean()), float(w.std()), float(w.abs().max())
    except:
        return 0.0, 0.0, 0.0

def load_model_bundle(bundle_path, device):
    """
    统一加载模型 Bundle (供 huanjing.py 使用)

    Args:
        bundle_path: 模型文件路径或目录
        device: torch.device

    Returns:
        dict: Bundle 对象或 None
    """
    bundle_path = Path(bundle_path)
    if bundle_path.is_dir():
        # 如果给的是目录，尝试找同名文件
        pt_files = list(bundle_path.glob("*.pt"))
        if pt_files:
            bundle_path = pt_files[0]
        else:
            return None

    if bundle_path.suffix != ".pt":
        return None

    try:
        bundle = torch_load_compat(bundle_path, map_location=device)
        if isinstance(bundle, dict) and "algo" in bundle and "model" in bundle:
            return bundle
    except Exception as e:
        print(f"[Utils] 加载 Bundle 失败 {bundle_path}: {e}")
    return None

# ================= 环境一致性工具函数 =================
def generate_tight_deadline_config(seed: int, user_num: int, tight_ratio: float = 0.3,
                                  deadline_slot: int = 130, factor_range=(0.6, 0.9)):
    """
    生成紧 deadline 配置（可复现）

    Args:
        seed: 随机种子
        user_num: 用户总数
        tight_ratio: 紧 deadline 用户比例
        deadline_slot: 默认 deadline (slot)
        factor_range: 因子范围 (min_factor, max_factor)

    Returns:
        dict: {
            "tight_user_ids": List[int],
            "app_deadline_slots": Dict[int, int],
            "deadline_slot_per_user": List[int],  # 每个用户的 deadline_slot（直接可用）
            "default_deadline_slot": int
        }
    """
    from utils.constant import para
    deadline_slot = para.get("deadline_slot", deadline_slot)

    rng = random.Random(seed)
    num_tight_users = int(user_num * tight_ratio)
    tight_deadline_users = rng.sample(range(user_num), num_tight_users)

    # 直接生成每个用户的 deadline_slot 数组
    deadline_slot_per_user = [deadline_slot] * user_num

    app_deadline_slots = {}
    for idx, uid in enumerate(tight_deadline_users):
        # 使用索引分配固定因子，确保确定性
        min_factor, max_factor = factor_range
        factor = min_factor + (idx % 4) * 0.1  # 0.6, 0.7, 0.8, 0.9 循环分配
        tight_deadline = int(round(deadline_slot * factor))
        app_deadline_slots[uid] = tight_deadline
        deadline_slot_per_user[uid] = tight_deadline

    return {
        "tight_user_ids": tight_deadline_users,
        "app_deadline_slots": app_deadline_slots,
        "deadline_slot_per_user": deadline_slot_per_user,
        "default_deadline_slot": deadline_slot
    }

def save_deadline_config(run_dir: str, seed: int, tight_ratio: float = 0.3):
    """
    生成并保存 tight deadline 配置到 JSON

    Args:
        run_dir: 运行目录
        seed: 随机种子
        tight_ratio: 紧 deadline 用户比例

    Returns:
        dict: tight deadline 配置
    """
    from utils.constant import para

    deadline_config = generate_tight_deadline_config(
        seed=seed,
        user_num=para["user_num"],
        tight_ratio=tight_ratio,
        deadline_slot=para["deadline_slot"]
    )

    # 保存到 configs/deadlines.json
    config_dir = Path(run_dir) / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    deadline_path = config_dir / "deadlines.json"

    with open(deadline_path, "w", encoding="utf-8") as f:
        json.dump(deadline_config, f, indent=2)

    print(f"[DEADLINE CONFIG] 已保存到: {deadline_path}")
    print(f"  紧 deadline 用户数: {len(deadline_config['tight_user_ids'])}")
    print(f"  因子范围: 0.6x - 0.9x")

    return deadline_config

def load_deadline_config(run_dir: str):
    """
    加载 tight deadline 配置

    Args:
        run_dir: 运行目录

    Returns:
        dict: deadline 配置或 None
    """
    from utils.constant import para
    deadline_path = Path(run_dir) / "configs" / "deadlines.json"
    if deadline_path.exists():
        with open(deadline_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        # 【强校验】验证 deadline_slot_per_user
        if "deadline_slot_per_user" in cfg:
            slots = cfg["deadline_slot_per_user"]
            # 校验长度
            if len(slots) != para["user_num"]:
                raise ValueError(f"deadline_slot_per_user 长度不匹配: {len(slots)} != {para['user_num']}")
            # 强制转换为 int
            slots = [int(x) for x in slots]
            cfg["deadline_slot_per_user"] = slots

            # 自动补全 app_deadline_slots（兼容旧格式）
            if "app_deadline_slots" not in cfg or len(cfg["app_deadline_slots"]) == 0:
                from collections import defaultdict
                default_slot = cfg.get("default_deadline_slot", para["deadline_slot"])
                app_deadline_slots = {}
                for uid, slot in enumerate(slots):
                    if slot != default_slot:
                        app_deadline_slots[uid] = slot
                cfg["app_deadline_slots"] = app_deadline_slots

        return cfg
    return None

# ================= 状态增强辅助函数（用于 BiGATPPO） =================
def estimate_costs_helper(env, ts, task_complex_index, uid, sid, slot, task_size_bytes):
    """
    估算本地、云端、各边缘节点的预期完成时间（排队+执行）
    用于 BiGATPPO 的 lookahead 特征构造
    """
    now = slot * para["slot_interval"]
    bw = float(min(para["uplink_range"]))

    # --- Local ---
    f_local = float(env.device_list[uid].local_power)
    from Environment import computation
    _, t_exec_l = computation.execute_consumption(task_size_bytes, f_local, task_complex_index, "l")
    t_queue_l = max(0.0, float(ts.devices_exe_useful[uid]) - now)
    cost_local = t_queue_l + float(t_exec_l)

    # --- Edge (List) ---
    edge_costs = []
    for eid in range(para["edge_num"]):
        f_edge = float(env.edges[eid].edge_power * env.edges[eid].calculate_parameter)
        dist = float(env.device_list[uid].edge_distances[eid])
        _, t_up_e = computation.upload_consumption([task_size_bytes, dist, bw], 1, "e")
        _, t_exec_e = computation.execute_consumption(task_size_bytes, f_edge, task_complex_index, "e")

        t_queue_e = max(0.0, float(ts.remain_times[eid]) - now)
        t_queue_up = max(0.0, float(ts.devices_upload_useful[uid]) - now)

        cost_edge = t_queue_up + float(t_up_e) + t_queue_e + float(t_exec_e)
        edge_costs.append(cost_edge)

    # --- Cloud ---
    nearest = int(np.argmin(env.device_list[uid].edge_distances))
    dist0 = float(env.device_list[uid].edge_distances[nearest])
    _, t_up1 = computation.upload_consumption([task_size_bytes, dist0, bw], 1, "e")
    _, t_up2 = computation.upload_consumption(task_size_bytes, 1, "c")
    fc = float(env.cloud.cloud_power)
    _, t_exec_c = computation.execute_consumption(task_size_bytes, fc, task_complex_index, "c")
    wan = float(para.get("cloud_wan_rtt", 0.0))

    t_queue_up = max(0.0, float(ts.devices_upload_useful[uid]) - now)
    cost_cloud = t_queue_up + float(t_up1) + float(t_up2) + float(t_exec_c) + wan

    return cost_local, cost_cloud, edge_costs

def augment_state_with_lookahead_and_current(state_data, node2idx, env, ts, task_complex_index, uid, sid, slot, task_size_bytes):
    """
    统一的状态增强函数：添加 Lookahead 特征 + is_current + Log(TaskSize)
    【关键】失败时必须补 0，保证维度永远一致！
    添加 Log(TaskSize) 特征，让网络更好理解任务大小

    Args:
        state_data: PyG Data 对象
        node2idx: 节点到索引的映射
        env: 环境对象
        ts: 任务调度器对象
        task_complex_index: 任务复杂度索引
        uid: 用户ID
        sid: 子任务ID
        slot: 当前时间槽
        task_size_bytes: 任务大小（字节）

    Returns:
        增强后的 state_data
    """
    LOOKAHEAD_DIM = 2 + para["edge_num"]  # local + cloud + edges

    # Log(TaskSize) 特征：帮助网络理解任务大小
    # Log(1MB) ≈ 13.8, Log(1KB) ≈ 6.9，映射到 [0, 1] 区间
    size_log = torch.log1p(torch.tensor(task_size_bytes, dtype=torch.float32)).view(1, 1)
    size_norm = (size_log - 10.0) / 10.0  # 归一化到 [0, 1]
    size_feat = size_norm.repeat(state_data.x.size(0), 1).to(state_data.x.device)

    # 1) lookahead 特征：默认全 0，保证维度永远一致
    cost_vec = [0.0] * LOOKAHEAD_DIM

    try:
        c_local, c_cloud, c_edges = estimate_costs_helper(env, ts, task_complex_index, uid, sid, slot, task_size_bytes)
        max_cost = 3.0
        cost_vec = [c_local, c_cloud] + c_edges
        cost_vec = [min(1.0, max(0.0, c / max_cost)) for c in cost_vec]
    except Exception as e:
        # 失败就保持全 0，不要 pass 掉导致维度不一致
        # 修复添加 debug 信息，便于诊断评估时是否一直失败
        import traceback
        print(f"[DEBUG] Augment failed: {e}")
        print(f"[DEBUG] Traceback: {traceback.format_exc()}")
        pass

    global_cost = torch.tensor(cost_vec, dtype=state_data.x.dtype).view(1, -1)
    global_cost = global_cost.repeat(state_data.x.size(0), 1)

    # 2) is_current：最后 1 维
    is_cur = torch.zeros((state_data.x.size(0), 1), dtype=state_data.x.dtype)
    if sid in node2idx:
        is_cur[node2idx[sid], 0] = 1.0

    # 3) 拼接：base + LogSize + lookahead + is_current
    state_data.x = torch.cat([state_data.x, size_feat, global_cost, is_cur], dim=1)
    return state_data


# ================= 模块别名同步 =================
# 防止同一文件被 Python 以不同模块名（exp_utils vs Experiments_new.exp_utils）
# 导入时创建两个独立实例，导致全局状态（如 CONFIG）不共享
_current_mod = sys.modules.get(__name__)
if __name__ == "Experiments_new.exp_utils":
    sys.modules.setdefault("exp_utils", _current_mod)
elif __name__ == "exp_utils":
    sys.modules.setdefault("Experiments_new.exp_utils", _current_mod)
