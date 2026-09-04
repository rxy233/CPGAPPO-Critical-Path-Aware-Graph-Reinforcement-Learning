# -*- coding: utf-8 -*-
"""
_dall_reeval_lib.py — 外部 RL 算法 D_all 重评估共享库
=========================================================================
用途: 复用已有 checkpoint, 对 5 个外部 RL 对比算法 × 3 拓扑 × 5 seeds 做
       deterministic eval, 把时延口径从 D_succ 补成 D_all (超时应用按 per-app deadline
       预算计入), 并重算 UtilityScore_all。


  _reeval_lib 只覆盖 CPGAPPO + 6 消融变体 (CPGAPPO 家族, 共用 GAT_PPO_Agent_CPGAPPO)。
  5 个外部 RL 各有不同 agent 类 / 不同 state 提取函数 / 不同 ckpt 文件名, 不在
  _reeval_lib.VARIANT_CONFIG 里, 所以这里另起一套, 但口径完全一致:
    D_all = ts.get_avg_results(only_successful=False, timeout_charge="deadline")
    UtilityScore_all = compute_utility_score(E_all, D_all, rho_app, rho_task, w_cost=0.25)

设计要点: 
  • 不重训: 仅加载 .pt → 各算法自带的 eval 函数跑 deterministic eval →
    同一 ts 状态调 ts.get_avg_results(only_successful=False, timeout_charge="deadline") 拿 D_all。
  • 不修改原算法文件: 通过 import 调用各 wrapper 的 eval 函数。
  • 输出: 在原 result.json 上原地合并 D_all / D_succ / E_all / E_succ /
    UtilityScore_all 字段 (保留所有已有字段)。
    同时写 result_dall.json 留底 (不覆盖原 result.json 的备份)。

5 个外部 RL 算法的 eval 入口
  1) DGMA_adapt   : train_dgma_adapt_wrapper._eval_once(agent, env, arrival_plan, ...)
                    ckpt: checkpoints/DGMA_adapt.pt (agent.actor.state_dict())
  2) DGMA_paper   : train_dgma_paper_wrapper._eval_once(agent, env, arrival_plan, ...)
                    ckpt: checkpoints/DGMA_paper_seed{S}_best.pt (agent.state_dict())
  3) TransEdgeStyle: train_transedge_style_wrapper.eval_transedge_style_once(agent, ts, gs, bc, ...)
                    ckpt: checkpoints/TransEdgeStyle.pt (agent.policy.state_dict())
  4) GATDQN       : train_dynamic_gat_dqn.run_eval(agent, env, ts, gs, bc, arrival_plan, topk, ...)
                    ckpt: checkpoints/dynamic_gat_dqn_model.pth ({policy_gpu: ...})
  5) PPO          : 无独立 eval 函数, 训练时已返回 best_metrics (D_succ 口径)。
                    这里仿 train_ppo.py 末尾的 dead-code eval loop 自己跑一遍
                    deterministic eval (与训练时同 env 同 arrival_plan 同 mask)。
                    ckpt: checkpoints/PPO.pt (save_model_bundle, 含 actor+critic+meta)

输出目录优先 env CPGAPPO_DALL_OUT_ROOT, 否则 results/ 下最新 main_comparison_* 批次
"""
import os
import sys
import json
import time
import traceback
import importlib

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch
from pathlib import Path

from Experiments_new.exp_utils import (
    CONFIG, init_worker, generate_arrival_plan, load_deadline_config,
    compute_score, compute_utility_score, calc_timeout_rate,
    safe_rest_tasks_total, all_arrived_done, get_graph_cache,
    apply_arrival_plan, subtask_partition_stats, load_model_bundle,
    get_feature_dim, graph_state_to_vector, get_task_size_bytes,
)
from Environment.environment import Environment
from scheduler.graph_scheduler import GraphScheduler
from scheduler.task_scheduler import TaskScheduler
from Algorithms.Benchmark import Benchmark as BenchmarkClass
from utils.constant import para


# ================= 评估环境参数 (与 _run3_main_comparison_3dag.py worker 一致) =================
DEADLINE_SLOT = 55
BURST_PROB = 0.2
BURST_SIZE = 55
MAX_STEPS = 8000
STOP_ARRIVAL_STEP = 2000

# ================= DAG 拓扑 → matrix_path (全反斜杠, 避免 get_graph_cache md5 坑) =================
DAG_MATRIX = {
    "chain":   os.path.normpath(os.path.join(_PROJECT_ROOT, "matrix", "matrix_60_chain.txt")),
    "default": os.path.normpath(os.path.join(_PROJECT_ROOT, "matrix", "matrix_60.txt")),
    "wide":    os.path.normpath(os.path.join(_PROJECT_ROOT, "matrix", "matrix_60_wide.txt")),
}

# ================= 输出根目录 (动态: 优先 env CPGAPPO_DALL_OUT_ROOT, 否则 results/ 下最新 main_comparison 批次) =================
# 固定输出根改为动态解析,
#   这样在 fresh clone (results/ 为空) 时, run.py 的 reeval_only 模式会先创建
#   results/main_comparison_reeval_{ts}/ 并 stage ckpts, 然后通过 env var 指过来。
_OUT_ROOT_DEFAULT = os.path.join(_PROJECT_ROOT, "results")
OUT_ROOT = os.environ.get("CPGAPPO_DALL_OUT_ROOT", "")
if not OUT_ROOT:
    import glob as _glob
    _cands = sorted(_glob.glob(os.path.join(_OUT_ROOT_DEFAULT, "main_comparison_*")))
    OUT_ROOT = _cands[-1] if _cands else _OUT_ROOT_DEFAULT


# ============================================================================
# 通用: 构造 env / ts / gs / bc (与各 wrapper 的训练 setup 一致)
# ============================================================================
def _build_env(matrix_path, run_dir, seed):
    """
    按 worker 的 setup 模式构造 env/ts/gs/bc。
    关键: CONFIG["SEED"]=0 + seed_offset=seed → env_seed=seed, 与原训练一致。
    返回 dict (env, ts, gs, bc, arrival_plan, task_complex_index, max_steps, device)
    """
    os.environ["MATRIX_OVERRIDE_PATH"] = matrix_path
    para["deadline_slot"] = DEADLINE_SLOT
    CONFIG["RUN_DIR"] = run_dir
    CONFIG["BURST_PROB"] = BURST_PROB
    CONFIG["BURST_SIZE"] = BURST_SIZE
    CONFIG["MAX_STEPS"] = MAX_STEPS
    CONFIG["STOP_ARRIVAL_STEP"] = STOP_ARRIVAL_STEP
    CONFIG["SEED"] = 0  # env_seed = CONFIG["SEED"] + seed_offset = seed

    init_worker(seed, para, CONFIG)
    device = torch.device('cuda:0' if torch.cuda.is_available() else torch.device('cpu'))

    user_num = para["user_num"]
    subgraph_num = 20
    basegraph_num = 60
    task_complex = para["task_complex"]
    _max_steps = CONFIG["MAX_STEPS"]

    if isinstance(task_complex, (list, tuple)):
        task_complex_index = (CONFIG["SEED"] + seed) % len(task_complex)
    else:
        task_complex_index = int(task_complex) if isinstance(task_complex, int) else 0

    env = Environment(user_num, subgraph_num, basegraph_num, task_complex_index)
    env_seed = CONFIG["SEED"] + seed

    arrival_plan = generate_arrival_plan(
        env_seed, _max_steps, CONFIG["STOP_ARRIVAL_STEP"],
        0.3, CONFIG.get("BURST_PROB", 0.15),
        max(1, CONFIG.get("BURST_SIZE", 4) // 2), CONFIG.get("BURST_SIZE", 4),
    )

    env.generate_components(seed=env_seed)
    G = get_graph_cache(user_num, subgraph_num, basegraph_num, _PROJECT_ROOT)
    if G is not None and env.basegraph:
        env.basegraph.nx_graph = G

    deadline_config = load_deadline_config(run_dir)
    ts = TaskScheduler(user_num, subgraph_num, basegraph_num, env,
                       tight_deadline_config=deadline_config, seed=env_seed)
    gs = GraphScheduler(env.basegraph, env.subgraph_list, ts)
    bc = BenchmarkClass(env, gs, ts, task_complex_index, effective=True, seed=env_seed)
    ts.env = env
    ts.using_Algorithm = -1
    bc.reset()
    ts.reset()

    return {
        "env": env, "ts": ts, "gs": gs, "bc": bc,
        "arrival_plan": arrival_plan,
        "task_complex_index": task_complex_index,
        "max_steps": _max_steps,
        "device": device,
        "user_num": user_num,
    }


def _assemble_result(algo, dag, seed, e_succ, d_succ, e_all, d_all,
                     to_info, total_energy, ckpt_used, ckpt_path, elapsed):
    """组装 D_all reeval 结果 dict (与 _reeval_lib.reeval_one 同结构)。"""
    # calc_timeout_rate 返回小数 (0~1)；compute_utility_score 按论文口径
    # 要求传入百分比并在函数内部 /100，这里乘 100 还原。Delay 用 D_all。
    rho_app_pct = float(to_info['app_timeout_rate']) * 100.0
    rho_task_pct = float(to_info.get('task_timeout_rate', 0)) * 100.0

    utility_succ = float(compute_utility_score(
        e_succ, d_succ, rho_app_pct, rho_task_pct, w_cost=0.25))
    utility_all = float(compute_utility_score(
        e_all, d_all, rho_app_pct, rho_task_pct, w_cost=0.25))

    return {
        "status": "ok",
        "algorithm": algo,
        "dag": dag,
        "seed": seed,
        "D_succ": float(d_succ),
        "D_all": float(d_all),
        "E_succ": float(e_succ),
        "E_all": float(e_all),
        "AppTO": float(rho_app_pct),
        "TaskTO": float(rho_task_pct),
        "UtilityScore_succ": utility_succ,
        "UtilityScore_all": utility_all,  # ← 主表用此列
        "energy": float(e_succ),
        "total_energy": float(total_energy),
        "ckpt_used": ckpt_used,
        "ckpt_path": ckpt_path,
        "config": {
            "timeout_charge_mode": "deadline",
            "deadline_slot": DEADLINE_SLOT,
            "matrix_path": DAG_MATRIX[dag],
            "burst_prob": BURST_PROB,
            "burst_size": BURST_SIZE,
            "max_steps": MAX_STEPS,
            "stop_arrival_step": STOP_ARRIVAL_STEP,
        },
        "action_stats": to_info.get('action_stats', {}),
        "subtask_stats": to_info.get('subtask_stats', {}),
        "elapsed_sec": round(elapsed, 1),
    }


def _merge_and_write(run_dir, dall_result, also_write_dall=True):
    """
    把 dall_result 的字段合并进原 result.json (原地更新), 并另存 result_dall.json。
    保留原 result.json 的所有已有字段, 仅覆盖/新增 D_all/D_succ/E_all/E_succ/
    UtilityScore_all/UtilityScore_succ/delay_all/delay_succ。
    """
    result_path = os.path.join(run_dir, "result.json")
    existing = {}
    if os.path.exists(result_path):
        try:
            with open(result_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception as ex:
            print(f"[_dall_reeval_lib] 读原 result.json 失败, 将新建: {ex}")

    # 合并: dall_result 覆盖 (D_all/D_succ/E_all/E_succ/UtilityScore_all 等)
    existing["status"] = dall_result.get("status", existing.get("status", "unknown"))
    existing["D_succ"] = dall_result.get("D_succ")
    existing["D_all"] = dall_result.get("D_all")
    existing["E_succ"] = dall_result.get("E_succ")
    existing["E_all"] = dall_result.get("E_all")
    existing["delay_succ"] = dall_result.get("D_succ")
    existing["delay_all"] = dall_result.get("D_all")
    existing["UtilityScore_succ"] = dall_result.get("UtilityScore_succ")
    existing["UtilityScore_all"] = dall_result.get("UtilityScore_all")
    existing["ckpt_used"] = dall_result.get("ckpt_used", existing.get("ckpt_used", ""))
    existing["dall_reeval_ts"] = time.strftime("%Y-%m-%d %H:%M:%S")

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    print(f"[_dall_reeval_lib] result.json (merged) -> {result_path}")

    if also_write_dall:
        dall_path = os.path.join(run_dir, "result_dall.json")
        with open(dall_path, "w", encoding="utf-8") as f:
            json.dump(dall_result, f, indent=2, ensure_ascii=False)
        print(f"[_dall_reeval_lib] result_dall.json -> {dall_path}")


# ============================================================================
# 1) DGMA_adapt
# ============================================================================
def reeval_dgma_adapt(dag, seed, use_heuristic_sort=True):
    """
    DGMA_adapt reeval: 加载 DGMA_adapt.pt → _eval_once(agent, env, ...) →
    同一 ts_eval 状态取 D_all。
    注: _eval_once 内部自建 ts_eval 并调 finalize_episode, 但不返回 ts_eval。
        这里通过把 _eval_once 的逻辑搬过来 (重跑一遍), 在 eval loop 后直接查 D_all。
    """
    from Algorithms.StrongBaselines.dgma_adapt_agent import DGMAAdaptAgent
    from Algorithms.StrongBaselines.dgma_adapt_core import extract_dgma_adapt_state

    t_start = time.time()
    run_dir = os.path.join(OUT_ROOT, f"{dag}__DGMA_adapt__seed{seed}")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    ckpt_path = os.path.join(ckpt_dir, "DGMA_adapt.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"DGMA_adapt ckpt not found: {ckpt_path}")

    print(f"[_dall_reeval_lib] === DGMA_adapt / {dag} / seed{seed} ===")
    ctx = _build_env(DAG_MATRIX[dag], run_dir, seed)
    env = ctx["env"]
    device = ctx["device"]

    # node_dim 探测 (与 train_dgma_adapt_wrapper 一致)
    try:
        s_data, _ = extract_dgma_adapt_state(ctx["ts"], (0, 0), slot=0,
                                              task_complex_index=ctx["task_complex_index"])
        node_dim = s_data.x.shape[1]
    except Exception as e:
        print(f"[DGMA_adapt] node_dim detect failed, fallback=27. err={e}")
        node_dim = 27
    action_dim = para["edge_num"] + 2
    # 关键hidden_dim/batch_size/buffer_capacity 必须与 train_dgma_adapt_wrapper 一致
    # (训练默认 hidden_dim=64, batch_size=64, buffer_capacity=20000), 否则 ckpt 形状对不上
    hidden_dim = 64
    batch_size = 64
    buffer_capacity = 20000

    agent = DGMAAdaptAgent(
        node_dim=node_dim, action_dim=action_dim, device=device,
        hidden_dim=hidden_dim, num_layers=2,
        lr_actor=3e-4, lr_critic=3e-4,
        gamma=0.99, tau=0.005, gumbel_tau=1.0,
        batch_size=batch_size, buffer_capacity=buffer_capacity,
        per_alpha=0.6, per_beta0=0.4, per_beta_anneal_steps=10000,
        updates_per_step=1, min_updates_after=max(256, batch_size * 4), grad_clip=1.0,
    )
    best_state = torch.load(ckpt_path, map_location=device)
    agent.actor.load_state_dict(best_state)
    agent.actor.eval()

    # ===== 复刻 _eval_once 的 loop (拿 ts_eval + D_all) =====
    user_num = para["user_num"]
    env_seed = CONFIG["SEED"] + seed
    deadline_config = load_deadline_config(run_dir)
    ts_eval = TaskScheduler(user_num, 20, 60, env,
                            tight_deadline_config=deadline_config, seed=env_seed)
    gs_eval = GraphScheduler(env.basegraph, env.subgraph_list, ts_eval)
    bc_eval = BenchmarkClass(env, gs_eval, ts_eval, ctx["task_complex_index"],
                             effective=True, seed=env_seed)
    ts_eval.env = env
    ts_eval.using_Algorithm = -1
    bc_eval.reset(); ts_eval.reset()
    ts_eval.route_stats = {"local": 0, "edge": 0, "cloud": 0, "recent": []}

    eval_task2action = {}
    for slot in range(MAX_STEPS):
        apply_arrival_plan(ts_eval, slot, ctx["arrival_plan"])
        ts_eval.check_timeouts(slot)
        tasks = gs_eval.get_tasks(slot, sort_tasks=use_heuristic_sort)
        if not tasks:
            if (slot >= STOP_ARRIVAL_STEP and
                    safe_rest_tasks_total(ts_eval.rest_tasks) == 0 and
                    all_arrived_done(ts_eval)):
                break
            continue
        for task in tasks:
            s_data, mask = extract_dgma_adapt_state(ts_eval, task, slot=slot,
                                                    task_complex_index=ctx["task_complex_index"])
            act, _ = agent.take_action(s_data, action_mask=mask, deterministic=True)
            bc_eval.step([[task, act]])
            eval_task2action[tuple(task)] = int(act)

    ts_eval.finalize_episode(MAX_STEPS - 1)
    e_succ, d_succ = ts_eval.get_avg_results(only_successful=True)
    e_all, d_all = ts_eval.get_avg_results(only_successful=False, timeout_charge="deadline")
    to_info = calc_timeout_rate(ts_eval)
    app_to = float(to_info["app_timeout_rate"]); task_to = float(to_info.get("task_timeout_rate", app_to))
    if app_to > 1.0: app_to /= 100.0
    if task_to > 1.0: task_to /= 100.0
    to_info["app_timeout_rate"] = app_to
    to_info["task_timeout_rate"] = task_to

    partition = subtask_partition_stats(ts_eval, eval_task2action)
    to_info["action_stats"] = {
        "local": partition["local"], "cloud": partition["cloud"], "edge": partition["edge"],
        "timeout": partition["timeout"], "unknown": partition["unknown"],
        "total": partition["total_subtasks"], "total_actions": len(eval_task2action),
    }
    total_e = ts_eval.get_sum_energy()

    elapsed = time.time() - t_start
    result = _assemble_result("DGMA_adapt", dag, seed, e_succ, d_succ, e_all, d_all,
                              to_info, total_e, "best", ckpt_path, elapsed)
    _merge_and_write(run_dir, result)
    print(f"[_dall_reeval_lib] OK DGMA_adapt/{dag}/seed{seed} "
          f"D_succ={d_succ:.4f} D_all={d_all:.4f} AppTO={app_to:.2%} "
          f"Util_all={result['UtilityScore_all']:.4f} ({elapsed:.1f}s)")
    return result


# ============================================================================
# 2) DGMA_paper
# ============================================================================
def reeval_dgma_paper(dag, seed, use_heuristic_sort=True):
    """
    DGMA_paper reeval: 加载 DGMA_paper_seed{S}_best.pt → 复刻 _eval_once loop →
    同一 ts_eval 取 D_all。
    注: DGMA_paper 用 K=edge_num 个 region agents, take_action_region 返回 (server_id, c_alloc, _),
        c_alloc 通过 ts_eval._dgma_c_alloc 挂钩传给 scheduler (runtime patch 已在 wrapper 训练时
        安装; 这里 reeval 需重新安装, 见 _install_c_alloc_patch)。
    """
    from Algorithms.StrongBaselines.dgma_paper_agent import DGMAPaperAgent
    from Algorithms.StrongBaselines.dgma_paper_core import (
        extract_region_state, R1_FEATURE_DIM,
    )
    # 安装 c_alloc runtime patch (与 wrapper 训练时一致)
    from Algorithms.Train.train_dgma_paper_wrapper import _install_c_alloc_patch
    _install_c_alloc_patch()

    t_start = time.time()
    run_dir = os.path.join(OUT_ROOT, f"{dag}__DGMA_paper__seed{seed}")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    ckpt_path = os.path.join(ckpt_dir, f"DGMA_paper_seed{seed}_best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"DGMA_paper ckpt not found: {ckpt_path}")

    print(f"[_dall_reeval_lib] === DGMA_paper / {dag} / seed{seed} ===")
    ctx = _build_env(DAG_MATRIX[dag], run_dir, seed)
    env = ctx["env"]
    device = ctx["device"]

    action_dim = para["edge_num"] + 2
    num_regions = int(para["edge_num"])
    try:
        dummy_data, _ = extract_region_state(
            ctx["ts"], list(range(min(2, para["user_num"]))), (0, 0), slot=0,
            task_complex_index=ctx["task_complex_index"],
        )
        node_dim = dummy_data.x.shape[1]
        print(f"[DGMA_paper] Detected node_dim={node_dim}, K={num_regions}, A={action_dim}")
    except Exception as e:
        print(f"[DGMA_paper] node_dim detect failed, fallback={R1_FEATURE_DIM}. err={e}")
        node_dim = R1_FEATURE_DIM

    # 关键hidden_dim/batch_size/buffer_capacity/sps_every 必须与 train_dgma_paper_wrapper 一致
    # (训练默认 hidden_dim=64, batch_size=32, buffer_capacity=10000, sps_every=200), 否则 ckpt 形状对不上
    hidden_dim = 64
    batch_size = 32
    buffer_capacity = 10000
    sps_every = 200
    agent = DGMAPaperAgent(
        env=env, num_regions=num_regions, node_dim=node_dim,
        action_dim=action_dim, hidden_dim=hidden_dim, num_layers=2,
        device=device,
        lr_actor=3e-4, lr_critic=3e-4,
        gamma=0.99, tau=0.005, gumbel_tau=1.0,
        batch_size=batch_size, buffer_capacity=buffer_capacity,
        per_alpha=0.6, per_beta0=0.4, per_beta_anneal_steps=10000,
        updates_per_step=1, min_updates_after=max(256, batch_size * 4), grad_clip=1.0,
        sps_every=sps_every, sps_self_weight=0.6,
    )
    state_dict = torch.load(ckpt_path, map_location=device)
    agent.load_state_dict(state_dict)

    # ===== 复刻 _eval_once loop (拿 ts_eval + D_all) =====
    user_num = para["user_num"]
    env_seed = CONFIG["SEED"] + seed
    deadline_config = load_deadline_config(run_dir)
    ts_eval = TaskScheduler(user_num, 20, 60, env,
                            tight_deadline_config=deadline_config, seed=env_seed)
    gs_eval = GraphScheduler(env.basegraph, env.subgraph_list, ts_eval)
    bc_eval = BenchmarkClass(env, gs_eval, ts_eval, ctx["task_complex_index"],
                             effective=True, seed=env_seed)
    ts_eval.env = env
    ts_eval.using_Algorithm = -1
    bc_eval.reset(); ts_eval.reset()
    ts_eval.route_stats = {"local": 0, "edge": 0, "cloud": 0, "recent": []}
    ts_eval._dgma_c_alloc = 1.0

    user_to_region = agent.region_info["user_to_region"]
    region_to_users = agent.region_info["region_to_users"]

    for slot in range(MAX_STEPS):
        apply_arrival_plan(ts_eval, slot, ctx["arrival_plan"])
        ts_eval.check_timeouts(slot)
        tasks = gs_eval.get_tasks(slot, sort_tasks=use_heuristic_sort)
        if not tasks:
            if (slot >= STOP_ARRIVAL_STEP and
                    safe_rest_tasks_total(ts_eval.rest_tasks) == 0 and
                    all_arrived_done(ts_eval)):
                break
            continue
        for task in tasks:
            uid = task[0]
            region_id = int(user_to_region[uid])
            state_data, mask_bin = extract_region_state(
                ts_eval, region_to_users[region_id], task,
                slot=slot, task_complex_index=ctx["task_complex_index"],
            )
            server_id, c_alloc, _ = agent.take_action_region(
                region_id, state_data, action_mask=mask_bin, deterministic=True,
            )
            if server_id == 0:
                ts_eval.route_stats["local"] += 1
            elif server_id == 1:
                ts_eval.route_stats["cloud"] += 1
            else:
                ts_eval.route_stats["edge"] += 1
            ts_eval._dgma_c_alloc = float(c_alloc) * 2.0
            bc_eval.step([[task, int(server_id)]])

    ts_eval.finalize_episode(MAX_STEPS - 1)
    e_succ, d_succ = ts_eval.get_avg_results(only_successful=True)
    e_all, d_all = ts_eval.get_avg_results(only_successful=False, timeout_charge="deadline")
    to_info = calc_timeout_rate(ts_eval)
    app_to = float(to_info["app_timeout_rate"]); task_to = float(to_info.get("task_timeout_rate", 0))
    if app_to > 1.0: app_to /= 100.0
    if task_to > 1.0: task_to /= 100.0
    to_info["app_timeout_rate"] = app_to
    to_info["task_timeout_rate"] = task_to
    to_info["action_stats"] = dict(ts_eval.route_stats)
    total_e = ts_eval.get_sum_energy()

    elapsed = time.time() - t_start
    result = _assemble_result("DGMA_paper", dag, seed, e_succ, d_succ, e_all, d_all,
                              to_info, total_e, "best", ckpt_path, elapsed)
    _merge_and_write(run_dir, result)
    print(f"[_dall_reeval_lib] OK DGMA_paper/{dag}/seed{seed} "
          f"D_succ={d_succ:.4f} D_all={d_all:.4f} AppTO={app_to:.2%} "
          f"Util_all={result['UtilityScore_all']:.4f} ({elapsed:.1f}s)")
    return result


# ============================================================================
# 3) TransEdgeStyle
# ============================================================================
def reeval_transedge(dag, seed, use_heuristic_sort=False):
    """
    TransEdgeStyle reeval: 加载 TransEdgeStyle.pt → eval_transedge_style_once →
    复刻其 ts_eval loop 取 D_all。
    注: eval_transedge_style_once 内部自建 ts_eval 并调 finalize_episode,
        返回 5-tuple (e_succ, d_succ, score, to_info, total_e) 但不返回 ts_eval。
        这里复刻 loop 以拿到 ts_eval 查 D_all。
    """
    from Algorithms.StrongBaselines.transedge_style_agent import TransEdgeStylePPOAgent
    from Algorithms.StrongBaselines.transedge_style_core import extract_transedge_state

    t_start = time.time()
    run_dir = os.path.join(OUT_ROOT, f"{dag}__TransEdgeStyle__seed{seed}")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    ckpt_path = os.path.join(ckpt_dir, "TransEdgeStyle.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"TransEdgeStyle ckpt not found: {ckpt_path}")

    print(f"[_dall_reeval_lib] === TransEdgeStyle / {dag} / seed{seed} ===")
    ctx = _build_env(DAG_MATRIX[dag], run_dir, seed)
    ts = ctx["ts"]; env = ctx["env"]; device = ctx["device"]

    # state_dim 探测 (与 train_transedge_style_wrapper 一致)
    try:
        s_data, _ = extract_transedge_state(ts, (0, 0), slot=0,
                                            task_complex_index=ctx["task_complex_index"])
        state_dim = s_data.x.shape[1]
        print(f"[TransEdgeStyle] Detected node_dim={state_dim}")
    except Exception as e:
        print(f"[TransEdgeStyle] node_dim detect failed, fallback=52. err={e}")
        state_dim = 52
    action_dim = para["edge_num"] + 2

    agent = TransEdgeStylePPOAgent(
        node_dim=state_dim, action_dim=action_dim, device=device,
        lr=3e-4, gamma=0.99, lmbda=0.95,
        eps_clip=0.2, K_epochs=4, entropy_coef=0.01,
    )
    best_state = torch.load(ckpt_path, map_location=device)
    agent.policy.load_state_dict(best_state)
    agent.policy.eval()

    # ===== 复刻 eval_transedge_style_once 的 loop (拿 ts_eval + D_all) =====
    user_num = para["user_num"]
    env_seed = CONFIG["SEED"] + seed
    deadline_config = load_deadline_config(run_dir)
    ts_eval = TaskScheduler(user_num, 20, 60, env,
                            tight_deadline_config=deadline_config, seed=env_seed)
    gs_eval = GraphScheduler(env.basegraph, env.subgraph_list, ts_eval)
    bc_eval = BenchmarkClass(env, gs_eval, ts_eval, ctx["task_complex_index"],
                             effective=True, seed=env_seed)
    ts_eval.env = env
    ts_eval.using_Algorithm = -1
    bc_eval.reset(); ts_eval.reset()
    ts_eval.route_stats = {"local": 0, "edge": 0, "cloud": 0, "recent": []}

    eval_task2action = {}
    for slot in range(MAX_STEPS):
        apply_arrival_plan(ts_eval, slot, ctx["arrival_plan"])
        ts_eval.check_timeouts(slot)
        tasks = gs_eval.get_tasks(slot, sort_tasks=use_heuristic_sort)
        if not tasks:
            if (slot >= STOP_ARRIVAL_STEP and
                    safe_rest_tasks_total(ts_eval.rest_tasks) == 0 and
                    all_arrived_done(ts_eval)):
                break
            continue
        for task in tasks:
            s_data, mask = extract_transedge_state(ts_eval, task, slot=slot,
                                                   task_complex_index=ctx["task_complex_index"])
            act, _ = agent.take_action(s_data, action_mask=mask, deterministic=True,
                                       route_stats=getattr(ts_eval, "route_stats", None))
            bc_eval.step([[task, act]])
            eval_task2action[tuple(task)] = int(act)
            if act == 0:
                ts_eval.route_stats["local"] += 1
            elif act == 1:
                ts_eval.route_stats["cloud"] += 1
            else:
                ts_eval.route_stats["edge"] += 1
            ts_eval.route_stats["recent"].append(int(act))
            if len(ts_eval.route_stats["recent"]) > 100:
                ts_eval.route_stats["recent"].pop(0)

    ts_eval.finalize_episode(MAX_STEPS - 1)
    e_succ, d_succ = ts_eval.get_avg_results(only_successful=True)
    e_all, d_all = ts_eval.get_avg_results(only_successful=False, timeout_charge="deadline")
    to_info = calc_timeout_rate(ts_eval)
    app_to = float(to_info["app_timeout_rate"]); task_to = float(to_info.get("task_timeout_rate", app_to))
    if app_to > 1.0: app_to /= 100.0
    if task_to > 1.0: task_to /= 100.0
    to_info["app_timeout_rate"] = app_to
    to_info["task_timeout_rate"] = task_to

    partition = subtask_partition_stats(ts_eval, eval_task2action)
    to_info["action_stats"] = {
        "local": partition["local"], "cloud": partition["cloud"], "edge": partition["edge"],
        "timeout": partition["timeout"], "unknown": partition["unknown"],
        "total": partition["total_subtasks"], "total_actions": len(eval_task2action),
    }
    total_e = ts_eval.get_sum_energy()

    elapsed = time.time() - t_start
    result = _assemble_result("TransEdgeStyle", dag, seed, e_succ, d_succ, e_all, d_all,
                              to_info, total_e, "best", ckpt_path, elapsed)
    _merge_and_write(run_dir, result)
    print(f"[_dall_reeval_lib] OK TransEdgeStyle/{dag}/seed{seed} "
          f"D_succ={d_succ:.4f} D_all={d_all:.4f} AppTO={app_to:.2%} "
          f"Util_all={result['UtilityScore_all']:.4f} ({elapsed:.1f}s)")
    return result


# ============================================================================
# 4) GATDQN
# ============================================================================
def reeval_gatdqn(dag, seed):
    """
    GATDQN reeval: 加载 dynamic_gat_dqn_model.pth → run_eval(agent, env, ts, gs, bc, ...) →
    同一 ts 取 D_all。
    注: run_eval 用 SHARED ts (传入的), reset 后跑 eval, 末尾调 ts.finalize_episode + get_avg_results,
        返回 (e, d, metrics) 但不返回 ts。这里直接复刻 run_eval 的 loop 以拿 ts 查 D_all。
    """
    from Algorithms.GNNRL.dynamic_gat_dqn import DynamicGAT_DQN_Agent as DynamicGATDQN
    # GATDQN 的 eval loop 依赖这些辅助函数 (与 train_dynamic_gat_dqn.py 同 module 级)
    from Algorithms.Train.train_dynamic_gat_dqn import (
        get_stable_active_users, get_global_graph_state_dag, construct_action_mask,
        GRAPH_CACHE,
    )

    t_start = time.time()
    run_dir = os.path.join(OUT_ROOT, f"{dag}__GATDQN__seed{seed}")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    ckpt_path = os.path.join(ckpt_dir, "dynamic_gat_dqn_model.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"GATDQN ckpt not found: {ckpt_path}")

    print(f"[_dall_reeval_lib] === GATDQN / {dag} / seed{seed} ===")
    ctx = _build_env(DAG_MATRIX[dag], run_dir, seed)
    env = ctx["env"]; ts = ctx["ts"]; gs = ctx["gs"]; bc = ctx["bc"]; device = ctx["device"]

    # node_dim 探测 (与 train_dynamic_gat_dqn_wrapper 一致)
    base_node_dim = None
    for uid in range(min(10, para["user_num"])):
        try:
            sg_data, _ = gs.get_app_dag_data(env, uid, 0, ctx["task_complex_index"])
            if sg_data is not None and hasattr(sg_data, "x") and sg_data.x is not None:
                base_node_dim = int(sg_data.x.shape[1])
                break
        except Exception:
            pass
    if base_node_dim is None:
        base_node_dim = 10
    node_dim = base_node_dim + 2 + para["edge_num"]
    action_dim = para["edge_num"] + 2
    print(f"[GATDQN] BaseDim={base_node_dim}, FinalDim={node_dim}, ActionDim={action_dim}")

    agent = DynamicGATDQN(node_dim, action_dim, device, use_expert_data=False, expert_ratio=0.0)
    checkpoint = torch.load(ckpt_path, map_location=device)
    agent.policy_gpu.load_state_dict(checkpoint['policy_gpu'])
    agent.policy_cpu.load_state_dict(checkpoint['policy_gpu'])

    # ===== 复刻 run_eval loop (拿 ts + D_all) =====
    ts.reset(); bc.reset(); GRAPH_CACHE.reset()
    agent.policy_cpu.load_state_dict(agent.policy_gpu.state_dict())
    agent.policy_cpu.eval()
    eval_action_count = [0] * action_dim
    eval_total_actions = 0
    topk = 30

    for slot in range(MAX_STEPS):
        apply_arrival_plan(ts, slot, ctx["arrival_plan"])
        ts.check_timeouts(slot)
        inner_steps = 0
        while True:
            active_users, ready_tasks = get_stable_active_users(ts, gs, slot, k=topk)
            if not active_users:
                break
            (state_x, state_edge_index, state_batch), task_to_global_idx, global_idx_to_task = \
                get_global_graph_state_dag(env, gs, active_users, ctx["task_complex_index"],
                                           slot, base_node_dim)
            if state_x is None:
                break
            ready_mask = torch.zeros(state_x.shape[0], dtype=torch.bool)
            for t in ready_tasks:
                if t in task_to_global_idx:
                    ready_mask[task_to_global_idx[t]] = True
            if not ready_mask.any():
                break
            full_action_mask = construct_action_mask(
                ts, ready_tasks, task_to_global_idx, action_dim, state_x.shape[0], slot)
            node_idx, offload_action = agent.select_action(
                (state_x, state_edge_index, state_batch), ready_mask, full_action_mask,
                training=False, custom_eps=0.0)
            if node_idx is None:
                break
            selected_task = global_idx_to_task[int(node_idx)]
            if selected_task is None:
                break
            bc.step([[selected_task, int(offload_action)]])
            eval_action_count[int(offload_action)] += 1
            eval_total_actions += 1
            inner_steps += 1
            if inner_steps >= 100:
                break
        if slot >= STOP_ARRIVAL_STEP and all_arrived_done(ts) and safe_rest_tasks_total(ts.rest_tasks) == 0:
            break

    ts.finalize_episode(slot)
    e_succ, d_succ = ts.get_avg_results(only_successful=True)
    e_all, d_all = ts.get_avg_results(only_successful=False, timeout_charge="deadline")
    to_info = calc_timeout_rate(ts)
    app_to = float(to_info["app_timeout_rate"]); task_to = float(to_info.get("task_timeout_rate", app_to))
    if app_to > 1.0: app_to /= 100.0
    if task_to > 1.0: task_to /= 100.0
    to_info["app_timeout_rate"] = app_to
    to_info["task_timeout_rate"] = task_to
    to_info["action_stats"] = {
        "local": eval_action_count[0], "cloud": eval_action_count[1],
        "edge": sum(eval_action_count[2:]), "total_actions": eval_total_actions,
    }
    total_e = ts.get_sum_energy()

    elapsed = time.time() - t_start
    result = _assemble_result("GATDQN", dag, seed, e_succ, d_succ, e_all, d_all,
                              to_info, total_e, "best", ckpt_path, elapsed)
    _merge_and_write(run_dir, result)
    print(f"[_dall_reeval_lib] OK GATDQN/{dag}/seed{seed} "
          f"D_succ={d_succ:.4f} D_all={d_all:.4f} AppTO={app_to:.2%} "
          f"Util_all={result['UtilityScore_all']:.4f} ({elapsed:.1f}s)")
    return result


# ============================================================================
# 5) PPO
# ============================================================================
def reeval_ppo(dag, seed):
    """
    PPO reeval: 加载 PPO.pt (save_model_bundle, 含 actor+critic+meta) →
    仿 train_ppo.py 末尾 dead-code eval loop 跑 deterministic eval → 同一 ts 取 D_all。
    注: PPO 是 vector state (非 GNN), 用 gs.get_graph_state_new + graph_state_to_vector。
        mask 用全 1 (与训练一致: "暂时禁用mask，使用全1 mask")。
        use_heuristic_sort=False (与 wrapper 调用一致)。
    """
    from Algorithms.PPO.ppo_gae_fixed import PPO_GAE_Fixed
    from Algorithms.Train.common import decode_action

    t_start = time.time()
    run_dir = os.path.join(OUT_ROOT, f"{dag}__PPO__seed{seed}")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    ckpt_path = os.path.join(ckpt_dir, "PPO.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"PPO ckpt not found: {ckpt_path}")

    print(f"[_dall_reeval_lib] === PPO / {dag} / seed{seed} ===")
    ctx = _build_env(DAG_MATRIX[dag], run_dir, seed)
    env = ctx["env"]; ts = ctx["ts"]; gs = ctx["gs"]; bc = ctx["bc"]; device = ctx["device"]

    # n_states 探测 (与 train_ppo_wrapper 一致)
    n_states = get_feature_dim(env, gs, ctx["task_complex_index"])
    n_actions = para["edge_num"] + 2
    print(f"[PPO] 状态维度={n_states}, 动作维度={n_actions}")

    agent = PPO_GAE_Fixed(n_states=n_states,
                          n_hiddens=min(n_states * 3, 768),
                          n_actions=n_actions,
                          actor_lr=3e-4, critic_lr=1e-3,
                          lmbda=0.95, epochs=5, eps=0.2, gamma=0.99,
                          device=device, batch_size=512, ent_coef=0.05)

    # 加载 bundle (save_model_bundle 格式)
    bundle = load_model_bundle(ckpt_path, device)
    if bundle is None or "model_state" not in bundle:
        # fallback: 直接 load ppo_model.pth (兼容旧格式, 只有 actor state_dict)
        pth_path = os.path.join(ckpt_dir, "ppo_model.pth")
        if os.path.exists(pth_path):
            actor_state = torch.load(pth_path, map_location=device)
            agent.actor.load_state_dict(actor_state)
            print(f"[PPO] Loaded actor from legacy ppo_model.pth")
        else:
            raise RuntimeError(f"Cannot load PPO ckpt from {ckpt_dir}")
    else:
        ms = bundle["model_state"]
        if "actor" in ms:
            agent.actor.load_state_dict(ms["actor"])
        if "critic" in ms:
            agent.critic.load_state_dict(ms["critic"])
        print(f"[PPO] Loaded bundle (algo={bundle.get('algo')}, meta={bundle.get('meta', {})})")
    agent.actor.eval()
    agent.critic.eval()

    # ===== 复刻 train_ppo.py 末尾 dead-code eval loop (拿 ts + D_all) =====
    ts.reset(); bc.reset()
    eval_action_count = [0] * n_actions
    eval_total_actions = 0
    task2action = {}

    for slot in range(MAX_STEPS):
        apply_arrival_plan(ts, slot, ctx["arrival_plan"])
        ts.check_timeouts(slot)
        if slot >= STOP_ARRIVAL_STEP:
            if all_arrived_done(ts) and safe_rest_tasks_total(ts.rest_tasks) == 0:
                break
        tasks = gs.get_tasks(slot, sort_tasks=False)
        if not tasks:
            continue
        for task in tasks:
            state_data = gs.get_graph_state_new(env, task, ctx["task_complex_index"], slot=slot)
            state = graph_state_to_vector(state_data, method='mean')
            if hasattr(state, "to"):
                state = state.to(agent.device)
            user_id, subtask_id = task
            task_size_bytes = get_task_size_bytes(ts, user_id, subtask_id)
            now_time = slot * para["slot_interval"]
            base_mask = ts.get_action_mask(user_id, task_size_bytes, now_time)
            # 与训练一致: 全 1 mask (train_ppo.py line 788)
            mask_bin = [1.0] * len(base_mask)
            action, *rest = agent.take_action(state, f"{user_id}_{subtask_id}",
                                              action_mask=mask_bin, deterministic=True)
            bc.step([[task, int(action)]])
            task2action[tuple(task)] = int(action)
            if 0 <= action < n_actions:
                eval_action_count[action] += 1
            eval_total_actions += 1

    ts.finalize_episode(slot)
    e_succ, d_succ = ts.get_avg_results(only_successful=True)
    e_all, d_all = ts.get_avg_results(only_successful=False, timeout_charge="deadline")
    to_info = calc_timeout_rate(ts)
    app_to = float(to_info["app_timeout_rate"]); task_to = float(to_info.get("task_timeout_rate", app_to))
    if app_to > 1.0: app_to /= 100.0
    if task_to > 1.0: task_to /= 100.0
    to_info["app_timeout_rate"] = app_to
    to_info["task_timeout_rate"] = task_to

    partition = subtask_partition_stats(ts, task2action)
    to_info["action_stats"] = {
        "local": partition["local"], "cloud": partition["cloud"], "edge": partition["edge"],
        "timeout": partition["timeout"], "unknown": partition["unknown"],
        "total": partition["total_subtasks"], "total_actions": len(task2action),
    }
    total_e = ts.get_sum_energy()

    elapsed = time.time() - t_start
    result = _assemble_result("PPO", dag, seed, e_succ, d_succ, e_all, d_all,
                              to_info, total_e, "best", ckpt_path, elapsed)
    _merge_and_write(run_dir, result)
    print(f"[_dall_reeval_lib] OK PPO/{dag}/seed{seed} "
          f"D_succ={d_succ:.4f} D_all={d_all:.4f} AppTO={app_to:.2%} "
          f"Util_all={result['UtilityScore_all']:.4f} ({elapsed:.1f}s)")
    return result


# ============================================================================
# 6) Bench (5 启发式: HybridPSOGA/Genetic/Greedy/Edge-only/Local-only)
# ============================================================================
# bench 的 mode → ts.using_Algorithm 映射 (与 Algorithms/Train/common.py algo_map 一致)
BENCH_ALGO_MAP = {
    "HybridPSOGA": 8,
    "Genetic":     6,   # GeneticFair
    "Greedy":      4,
    "Edge":        2,   # Random Edge (bench 内部 mode 名是 "Edge", 不是 "Edge-only")
    "Local":       0,
}
# 输出目录名 → bench mode (与 _run3_main_comparison_3dag.py ALGOS 列表的 mode 字段一致)
BENCH_LABEL_TO_MODE = {
    "HybridPSOGA": "HybridPSOGA",
    "Genetic":     "Genetic",
    "Greedy":      "Greedy",
    "Edge-only":   "Edge",
    "Local-only":  "Local",
}


def reeval_bench(algo_label, dag, seed):
    """
    Bench reeval: 启发式无 ckpt, 直接重跑一遍 (fast ~3 sec) →
    同一 ts 取 D_all (timeout_charge="deadline")。
    注: 原 run_benchmark_worker 末尾用 ts.get_avg_results() (默认 2x_deadline), 不是 R1-5。
        这里复刻 run_benchmark_worker 的 loop (用同一 arrival_plan + bc.get_actions + bc.step),
        末尾调 ts.get_avg_results(only_successful=False, timeout_charge="deadline") 拿真 D_all。
    """
    mode = BENCH_LABEL_TO_MODE.get(algo_label)
    if mode is None:
        raise ValueError(f"Unknown bench algo '{algo_label}', expected one of {list(BENCH_LABEL_TO_MODE.keys())}")
    using_alg = BENCH_ALGO_MAP[mode]

    t_start = time.time()
    run_dir = os.path.join(OUT_ROOT, f"{dag}__{algo_label}__seed{seed}")
    print(f"[_dall_reeval_lib] === {algo_label} (bench) / {dag} / seed{seed} ===")

    ctx = _build_env(DAG_MATRIX[dag], run_dir, seed)
    env = ctx["env"]; ts = ctx["ts"]; gs = ctx["gs"]; bc = ctx["bc"]

    # bench 的 bc 需要 task_order_mode="random" (与 _run3_main_comparison_3dag.py worker 一致)
    # _build_env 已用默认 task_order_mode 构造 bc, 这里重建一个带 random 的 bc
    bc = BenchmarkClass(env, gs, ts, ctx["task_complex_index"],
                        effective=True, seed=CONFIG["SEED"] + seed, task_order_mode="random")
    ts.env = env
    ts.using_Algorithm = using_alg
    bc.reset(); ts.reset()

    task2action = {}
    for slot in range(MAX_STEPS):
        apply_arrival_plan(ts, slot, ctx["arrival_plan"])
        ts.check_timeouts(slot)
        if slot >= STOP_ARRIVAL_STEP:
            if all_arrived_done(ts) and safe_rest_tasks_total(ts.rest_tasks) == 0:
                break
        actions, _ = bc.get_actions(ts.using_Algorithm, slot, 0)
        if actions:
            bc.step(actions)
        for entry in actions:
            if len(entry) >= 2:
                task, action_raw = entry[0], entry[1]
                # safe_action_to_int (与 common.py 一致)
                try:
                    action = int(action_raw)
                except Exception:
                    action = 0
                task2action[tuple(task)] = action

    ts.finalize_episode(slot)
    e_succ, d_succ = ts.get_avg_results(only_successful=True)
    e_all, d_all = ts.get_avg_results(only_successful=False, timeout_charge="deadline")
    to_info = calc_timeout_rate(ts)
    app_to = float(to_info["app_timeout_rate"]); task_to = float(to_info.get("task_timeout_rate", app_to))
    if app_to > 1.0: app_to /= 100.0
    if task_to > 1.0: task_to /= 100.0
    to_info["app_timeout_rate"] = app_to
    to_info["task_timeout_rate"] = task_to

    partition = subtask_partition_stats(ts, task2action)
    to_info["action_stats"] = {
        "local": partition["local"], "cloud": partition["cloud"], "edge": partition["edge"],
        "timeout": partition["timeout"], "unknown": partition["unknown"],
        "total": partition["total_subtasks"], "total_actions": len(task2action),
    }
    total_e = ts.get_sum_energy()

    elapsed = time.time() - t_start
    result = _assemble_result(algo_label, dag, seed, e_succ, d_succ, e_all, d_all,
                              to_info, total_e, "n/a (heuristic)", "n/a", elapsed)
    _merge_and_write(run_dir, result)
    print(f"[_dall_reeval_lib] OK {algo_label}/{dag}/seed{seed} "
          f"D_succ={d_succ:.4f} D_all={d_all:.4f} AppTO={app_to:.2%} "
          f"Util_all={result['UtilityScore_all']:.4f} ({elapsed:.1f}s)")
    return result


# ============================================================================
# 分派表: algo → reeval fn
# ============================================================================
REEVAL_FNS = {
    "DGMA_adapt":     reeval_dgma_adapt,
    "DGMA_paper":     reeval_dgma_paper,
    "TransEdgeStyle": reeval_transedge,
    "GATDQN":         reeval_gatdqn,
    "PPO":            reeval_ppo,
    "HybridPSOGA":    reeval_bench,
    "Genetic":        reeval_bench,
    "Greedy":         reeval_bench,
    "Edge-only":      reeval_bench,
    "Local-only":     reeval_bench,
}

# bench 算法集合 (用于 reeval_one 区分 RL vs bench)
BENCH_ALGOS = {"HybridPSOGA", "Genetic", "Greedy", "Edge-only", "Local-only"}


def reeval_one(algo, dag, seed):
    """对单格 (algo, dag, seed) 做 D_all reeval。失败返回 error dict。"""
    fn = REEVAL_FNS.get(algo)
    if fn is None:
        raise ValueError(f"Unknown algo '{algo}', expected one of {list(REEVAL_FNS.keys())}")
    try:
        if algo in BENCH_ALGOS:
            return fn(algo, dag, seed)
        else:
            return fn(dag, seed)
    except Exception as ex:
        err_msg = f"{type(ex).__name__}: {ex}"
        print(f"[_dall_reeval_lib] [ERROR] {algo}/{dag}/seed{seed}: {err_msg}")
        traceback.print_exc()
        err_result = {
            "status": "error",
            "algorithm": algo, "dag": dag, "seed": seed,
            "error": err_msg, "traceback": traceback.format_exc(),
        }
        # 也把 error 写到 result_dall.json 留底 (不动原 result.json)
        run_dir = os.path.join(OUT_ROOT, f"{dag}__{algo}__seed{seed}")
        try:
            with open(os.path.join(run_dir, "result_dall.json"), "w", encoding="utf-8") as f:
                json.dump(err_result, f, indent=2, ensure_ascii=False)
        except Exception:
            pass
        return err_result


if __name__ == "__main__":
    # 自测: 跑 default/GATDQN/seed1 一格
    print("=" * 70)
    print("_dall_reeval_lib.py self-test: default/GATDQN/seed1")
    print("=" * 70)
    reeval_one("GATDQN", "default", 1)
