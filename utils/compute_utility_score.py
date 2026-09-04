#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UtilityScore 计算模块（v2 公式 + SLA=5% 阈值）。

公式要点（详见同目录 README.md）：
    UtilityScore = tanh( utility_raw )

    utility_raw = success - penalty_cost - penalty_task - penalty_app
        success      = 1 - cost
        penalty_app  = w_app_base * rho_app + w_over * over
        penalty_task = w_task     * rho_task
        penalty_cost = w_cost     * cost

        cost  = phi1 * e_hat + phi2 * d_hat          # 归一化能耗/时延的加权和
        e_hat = clip((E - E_min) / (E_max - E_min), 0, 1)
        d_hat = clip((D - D_min) / (D_max - D_min), 0, 1)

        over  = min( tau * log1p(exp((rho_app - thr)/tau)), over_cap )   # Softplus 近似 max(0, rho_app - thr)
        thr   = 1 - sla0 = 0.05                      # 5% 超时阈值
        w_over = w_over_max * progress               # 课程式加权，评估时 progress=1

输入约定（详见 README.md）：
    - rho_app / rho_task 以【百分比】传入，例如 9.56 表示 9.56%，
      函数内部会 /100 转成小数。不要传 0.0956。
    - E_avg     为平均能耗（单位与 E_min/E_max 一致，默认 J 量级，范围 0.3~3.0）。
    - D_succ_avg 为【成功子任务】的平均完成时延（不是全部任务均值）。

输出：UtilityScore，范围约 (-1, 1)，越大越好。
"""

from __future__ import annotations
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
    out_scale: float = 1.0,    # tanh 压缩尺度：越小越“硬”，越大越“软”
) -> float:
    """
    计算 UtilityScore（v2 + SLA=5% 版本）。

    参数详见模块 docstring。返回 float，范围约 (-1, 1)，越大越好。
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


# ---------------------------------------------------------------------------
# 便捷工具：批量计算 + 与论文表格对照
# ---------------------------------------------------------------------------
def aggregate_mean_std(values, ddof: int = 0):
    """返回 (mean, std)。论文表格的 ±std 使用 ddof=0（总体标准差）。"""
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=ddof))


def recalc_table(csv_path: str, out_path: str | None = None) -> "object":
    """
    给定一张含列 [Algorithm, DAG, Seed, AppTO(%), TaskTO(%), Energy, Delay] 的 CSV，
    用本模块公式重算 UtilityScore 列并写回（列名保留 AppTO(%)/TaskTO(%)）。

    返回处理后的 DataFrame。
    """
    import pandas as pd

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    col_map = {"AppTO(%)": "AppTO", "TaskTO(%)": "TaskTO"}
    rev_map = {"AppTO": "AppTO(%)", "TaskTO": "TaskTO(%)"}
    df = df.rename(columns=col_map)

    df["UtilityScore"] = df.apply(
        lambda r: compute_utility_score(r["Energy"], r["Delay"], r["AppTO"], r["TaskTO"]),
        axis=1,
    )
    df = df.rename(columns=rev_map)

    if out_path:
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  -> {out_path} ({len(df)} rows)")
    return df


# ---------------------------------------------------------------------------
# 自检 / 演示
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 78)
    print("UtilityScore(v2 + SLA=5%) 自检：复现论文 CPGAPPO Default DAG 3 个代表 seed")
    print("=" * 78)

    # CPGAPPO, default, 3 个代表 seed：来自 Table1_RAW_v2_sla5.csv 原始行
    seeds = [
        # (AppTO%, Energy, Delay, TaskTO%, 论文CSV里的UtilityScore)
        (10.67, 0.17777, 1.09709, 64.0,  0.441179188905731),
        (7.33,  0.1725,  1.04093, 64.63, 0.5850968434638428),
        (10.67, 0.2002,  1.0926,  63.7,  0.4417166401760658),
    ]
    us = []
    for appto, e, d, tto, ref in seeds:
        u = compute_utility_score(e, d, appto, tto)
        us.append(u)
        print(f"  AppTO={appto:6.2f}  E={e:.5f}  D={d:.5f}  TaskTO={tto:5.2f}"
              f"  -> calc={u:.6f}  ref={ref:.6f}  diff={u-ref:+.2e}")
    m, s = aggregate_mean_std(us, ddof=0)
    print(f"\n  聚合: Mean={m:.2f}  Std(ddof=0)={s:.2f}")
    print(f"  论文 Table 1 报告: Mean=0.49  Std=0.07  ->  {'PASS' if abs(m-0.49)<0.005 and abs(s-0.07)<0.005 else 'CHECK'}")

    print()
    print("=" * 78)
    print("边界样例：观察超时率对 UtilityScore 的影响（E=0.18, D=1.10, TaskTO=64）")
    print("=" * 78)
    for appto in [0, 2, 5, 8, 10, 20, 40, 80, 100]:
        u = compute_utility_score(0.18, 1.10, appto, 64.0)
        print(f"  AppTO={appto:6.2f}%  ->  UtilityScore={u:+.4f}")
