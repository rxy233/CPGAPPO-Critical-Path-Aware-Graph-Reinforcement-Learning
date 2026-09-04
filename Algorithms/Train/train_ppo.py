# -*- coding: utf-8 -*-
"""
Training wrapper for the plain-PPO external baseline (no GAT, no guide).

English
-------
train_ppo.py trains the `PPO` baseline: a PPO agent (Algorithms/PPO/) over the
same edge-cloud env and arrival plan as every other algorithm, so the
comparison is apples-to-apples (R4-2). It produces checkpoints/*.pt (model
bundle: actor+critic+meta) and result.json with per-seed metrics. The R1-5
D_all reeval for this baseline is handled separately by
runners/_dall_reeval_lib.py (loads the ckpt and runs a deterministic eval
with timeout_charge="deadline"). print_lock is set by the runner entry
script via set_print_lock(), not fenxi.py.

中文
----
纯 PPO 外部基线的训练 wrapper: 在与其他算法相同的 env/arrival plan 上训练 PPO,
输出 ckpt 与 result.json; R1-5 的 D_all 重评估由 _dall_reeval_lib.py 单独完成。
"""
import os
import sys
import time
import json
import traceback
import gc
import numpy as np
import pandas as pd
import torch
import random
from pathlib import Path
from argparse import Namespace
from concurrent.futures import as_completed

# 引入项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

# 导入环境和算法
from Environment.environment import Environment
from scheduler.graph_scheduler import GraphScheduler
from scheduler.task_scheduler import TaskScheduler
from scheduler.task_selector import TaskSelector
from Algorithms import Benchmark
from utils.constant import para

# 尝试导入 RL 算法
DQN_Agent = None
PPO_GAE_Fixed = None
GAT_PPO_DQNAE = None
DynamicGATDQN = None

try:
    from Algorithms.GNNRL.dqnAgent import DQN_Agent
except Exception as e:
    print(f"[Worker Warning] DQN import failed: {e}")

try:
    from Algorithms.PPO.ppo_gae_fixed import PPO_GAE_Fixed
except Exception as e:
    print(f"[Worker Warning] PPO import failed: {e}")

except Exception as e:
    print(f"[Worker Warning] GAT_PPO import failed: {e}")

try:
    from Algorithms.GNNRL.dynamic_gat_dqn import DynamicGAT_DQN_Agent as DynamicGATDQN
except Exception as e:
    print(f"[Worker Warning] Dynamic GAT-DQN import failed: {e}")

# 导入工具函数
from Experiments_new.exp_utils import (
    CONFIG, ScoreTracker, init_worker, load_arrival_plan, generate_arrival_plan, apply_arrival_plan,
    load_deadline_config,
    get_graph_cache, get_feature_dim, graph_state_to_vector,
    get_task_size_bytes, calc_timeout_rate, to_scalar, save_model_bundle,
    safe_rest_tasks_total, safe_action_to_int, append_curve_row, diagnose_timeout,
    get_expert_action, NEG_INF, mask_has_any_valid, mask_allows, compute_score,
    get_arrived_apps, all_arrived_done, subtask_outcome_stats, subtask_partition_stats, normalize_mask,
)
from Algorithms.Train.common import decode_action, ensure_trace_dir

print_lock = None  # 将在训练入口脚本中通过 set_print_lock() 设置

def set_print_lock(lock):
    global print_lock
    print_lock = lock

# ================= Worker 函数 =================

def train_ppo_wrapper(gpu_id, seed_offset, use_heuristic_sort=False, episodes=50):
    """
    PPO 训练包装函数

    Args:
        gpu_id: GPU ID
        seed_offset: 种子偏移
        use_heuristic_sort: 是否使用启发式排序（默认=False，使用原始默认顺序）
                         - True: 启发式排序（仅GAT-PPO使用）
                         - False: 固定顺序（PPO等算法使用）
        episodes: 训练轮数（默认=50）
    """
    init_worker(seed_offset, para, CONFIG)
    # 修复从环境变量读取RUN_DIR和EVAL_MODE
    if "RUN_DIR" in os.environ:
        CONFIG["RUN_DIR"] = os.environ["RUN_DIR"]
    if "EVAL_MODE" in os.environ:
        CONFIG["EVAL_MODE"] = os.environ["EVAL_MODE"] == "True"
    if PPO_GAE_Fixed is None: raise ImportError("PPO_GAE_Fixed 未成功导入")
    device = torch.device(f"cuda:{gpu_id}") if torch.cuda.is_available() else torch.device("cpu")

    # 统一保存到checkpoints目录，不再创建算法_seed子文件夹
    if CONFIG.get("RUN_DIR"):
        checkpoint_dir = Path(CONFIG["RUN_DIR"]) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    else:
        checkpoint_dir = None

    try:
        user_num = para["user_num"]
        subgraph_num = 20
        basegraph_num = 60
        task_complex = para["task_complex"]

        # 修复在使用 MAX_STEPS 之前定义
        MAX_STEPS = CONFIG["MAX_STEPS"]  # 修复统一使用 CONFIG["MAX_STEPS"]，所有算法步数相同

        # 修复GAT_PPO使用基于seed的task_complex_index（统一使用offset=0）
        # 【安全修复】兼容 task_complex 是整数或列表的情况
        if isinstance(task_complex, (list, tuple)):
            task_complex_index = (CONFIG["SEED"] + seed_offset) % len(task_complex)
        else:
            # 如果是整数或其他类型，转换为索引（默认为0）
            task_complex_index = int(task_complex) if isinstance(task_complex, int) else 0

        env = Environment(user_num=user_num, subgraph_num=subgraph_num, basegraph_num=basegraph_num,
                          task_complex_index=task_complex_index)

        # 统一：arrival_plan 读到什么 env_seed，就用什么 env_seed 初始化环境
        run_dir = CONFIG.get("RUN_DIR", "")
        plan_data = load_arrival_plan(run_dir, seed_offset) if run_dir else None
        if plan_data is None:
            print(f"[PPO] 警告：未找到到达计划，使用随机生成（seed_offset={seed_offset}）")
            env_seed = CONFIG["SEED"] + seed_offset
            arrival_plan = generate_arrival_plan(
                env_seed,
                MAX_STEPS,
                CONFIG["STOP_ARRIVAL_STEP"],
                0.3,
                CONFIG.get("BURST_PROB", 0.15),
                max(1, CONFIG.get("BURST_SIZE", 4) // 2),
                CONFIG.get("BURST_SIZE", 4)
            )
        else:
            arrival_plan = plan_data.get("arrival_plan", [])
            env_seed = plan_data.get("env_seed", CONFIG["SEED"] + seed_offset)
            print(f"[PPO] 已加载预生成的到达计划（env_seed={env_seed}, seed_offset={seed_offset}）")

        env.generate_components(seed=env_seed)
        G = get_graph_cache(user_num, subgraph_num, basegraph_num, project_root)
        if G is not None and env.basegraph: env.basegraph.nx_graph = G

        # 修复加载 tight deadline 配置（确保环境一致性）
        deadline_config = load_deadline_config(run_dir)
        if deadline_config:
            print(f"[PPO] 【环境一致性】加载预计算的 deadline 配置: "
                  f"{len(deadline_config['tight_user_ids'])} 个紧 deadline 用户")
        else:
            print(f"[PPO] 【警告】未找到 deadline 配置，使用默认随机生成")

        ts = TaskScheduler(user_num, subgraph_num, basegraph_num, env, tight_deadline_config=deadline_config, seed=env_seed)
        gs = GraphScheduler(env.basegraph, env.subgraph_list, ts)
        bc = Benchmark.Benchmark(env, gs, ts, task_complex_index, effective=True, seed=env_seed)
        ts.env = env;
        ts.using_Algorithm = -1
        bc.reset();
        ts.reset()

        # 打印环境指纹：验证 enter_time/deadline 一致性
        slot_interval = para["slot_interval"]
        print(
            f"[PPO] 【环境指纹】enter_time[:10] = {[float(ts.enter_time[i]) if ts.enter_time[i] != float('inf') else 'inf' for i in range(10)]}")
        if hasattr(ts, 'app_deadline_slots'):
            deadline_slots = [ts.get_app_deadline_slot(i) for i in range(10)]
            print(f"[PPO] 【环境指纹】deadline_slot[:10] = {deadline_slots}")
            # 计算 deadline_abs（假设前 10 个用户的 enter_time 都不是 inf）
            deadline_abs = []
            for i in range(10):
                if ts.enter_time[i] != float('inf'):
                    dead_abs = float(ts.enter_time[i]) + int(ts.get_app_deadline_slot(i)) * slot_interval
                    deadline_abs.append(f"{dead_abs:.3f}s")
                else:
                    deadline_abs.append("N/A")
            print(f"[PPO] 【环境指纹】deadline_abs[:10] = {deadline_abs}")
        else:
            print(f"[PPO] 【环境指纹】deadline_slot = N/A (未设置)")

        # 如果 arrival_plan 是 dict，打印 schedule 前几个 slot
        if isinstance(arrival_plan, dict) and "schedule" in arrival_plan:
            schedule = arrival_plan.get("schedule", [])
            print(
                f"[PPO] 【环境指纹】arrival_schedule[:10] = {[f'slot{i}: {schedule[i]}' for i in range(min(10, len(schedule)))]}")

        # 修复使用固定的特征维度（全局池化后）
        n_states = get_feature_dim(env, gs, task_complex_index)  # 修复传 int index
        n_actions = para["edge_num"] + 2

        print(f"[PPO] 状态维度: {n_states}, 动作维度: {n_actions}")
        sys.stdout.flush()

        # 🚀🚀🚀 【PPO 增强版参数】🚀🚀🚀
        # 目标：将 AppTO 从 31.5% 降到 15-25%
        # 改进点：
        # 1. 提高学习率（更快收敛）
        # 2. 增加 entropy（增强探索）
        # 3. 增加 batch size（更稳定）
        # 4. 增加 PPO epochs（更充分更新）
        # 5. 网络宽度增加（更好特征提取）
        agent = PPO_GAE_Fixed(n_states=n_states, 
                               n_hiddens=min(n_states * 3, 768),  # 【增强】从 512 增加到 768
                               n_actions=n_actions, 
                               actor_lr=3e-4,      # 【增强】从 1e-4 提高到 3e-4
                               critic_lr=1e-3,     # 【增强】从 5e-4 提高到 1e-3
                               lmbda=0.95, 
                               epochs=5,           # 【增强】从 3 增加到 5
                               eps=0.2, 
                               gamma=0.99,
                               device=device, 
                               batch_size=512,     # 【增强】从 256 增加到 512
                               ent_coef=0.05)      # 【增强】从 0.02 提高到 0.05
        # [增加] 增加训练量
        EPISODES = episodes  # 使用传入的 episodes 参数
        BATCH_STEPS = 500  # 每 500 步更新一次

        # 终止条件参数
        best_score = float('inf')
        best_metrics = None  # 保存 best episode 的完整 metrics

        print(f"[PPO] 开始训练: 固定 {EPISODES} 个 episode")
        sys.stdout.flush()

        # 修复跳过模型文件加载，避免旧模型污染（脏数据缓存）
        # 【说明】原逻辑是评估模式下才加载 checkpoint，现在强制禁用以避免加载失败模型
        is_eval_mode = CONFIG.get("EVAL_MODE", False)
        checkpoint_path = None  # 强制禁用 checkpoint 加载

        # 【新增诊断】计算参数范数
        def get_param_norm(model):
            return sum(float(p.abs().sum().cpu().item()) for p in model.parameters())

        before_norm = get_param_norm(agent.actor)

        # 修复强制跳过 checkpoint 加载，避免脏数据污染
        # 原来的加载逻辑已全部注释，确保不会加载任何模型文件
        print(f"[PPO] 信息：已禁用 checkpoint 加载（避免脏数据污染）")
        print(f"[PPO] 将使用随机初始化的模型从头训练")

        # 【原逻辑注释】只在评估模式下才加载checkpoint并跳过训练
        # if is_eval_mode:
        #     if checkpoint_path and checkpoint_path.exists():
        #         print(f"[PPO] 评估模式：加载checkpoint: {checkpoint_path}")
        #         ... (原加载逻辑已注释)
        #     sys.stdout.flush()
        #     # [评估模式] 跳过训练，直接进入测试
        #     EPISODES = 0

        # 修复无论是训练模式还是评估模式，都强制从头训练
        # print(f"[PPO] 训练模式: 最多 {EPISODES} episodes, 固定轮次")
        print(f"[PPO] 训练模式: 最多 {EPISODES} episodes, 固定轮次（禁用warmstart）")
        sys.stdout.flush()

        # 修复初始化变量，防止评估模式下未定义
        timeout_info = {'app_timeout_rate': 1.0, 'task_timeout_rate': 1.0}
        e, d, current_score = 1000.0, 1000.0, 1000.0

        for episode in range(EPISODES):
            ts.reset();
            bc.reset()
            # 修复禁用单位转换，保持原始 Bytes 值以增加系统负载
            # fix_task_size_units_inplace(ts)

            transition_dicts = {}
            step_count = 0
            last_print_step = 0
            episode_rewards = []  # 收集 episode 的所有奖励

            # 初始化 ScoreTracker 用于对齐新版评分公式（SLA=0.99，即1%超时率）
            tracker = ScoreTracker(E_min=0.3, E_max=3.0, D_min=0.7, D_max=5.0,
                                  sla0=CONFIG.get("SLA0", 0.95),
                                  kappa=CONFIG.get("KAPPA", 3.0),
                                  v_cap_over=CONFIG.get("V_CAP_OVER", 2.0))

            # 【新增诊断】动作分布和mask统计
            action_count = [0] * n_actions
            total_actions = 0
            mask_stats = {"all_valid": 0, "only_local": 0, "cloud_edge_banned": 0}
            task2action = {}  # 记录任务到动作的映射（用于子任务分桶统计）

            # 【移除】不再使用启发式排序，改为默认顺序

            for slot in range(MAX_STEPS):
                # 修复使用预生成的到达计划，避免 random 被污染
                apply_arrival_plan(ts, slot, arrival_plan)
                ts.check_timeouts(slot)  # 修复必须检查超时

                # 修复保存当前应用状态，用于计算应用级奖励
                prev_app_to = set(getattr(ts, "application_timeout_finished", set()))
                prev_app_fin = set(getattr(ts, "application_finished", set()))

                # 使用默认顺序，不使用任何启发式排序
                tasks = gs.get_tasks(slot, sort_tasks=False)
                if not tasks:
                    # 到达停止后 + 没有待执行任务 + 所有到达任务完成 => 系统空了，直接结束本episode
                    # 不会改变最终指标，因为后面本来就没有任务/状态不再变化
                    if slot >= CONFIG["STOP_ARRIVAL_STEP"]:
                        from Experiments_new.exp_utils import safe_rest_tasks_total, all_arrived_done
                        if safe_rest_tasks_total(ts.rest_tasks) == 0 and all_arrived_done(ts):
                            break
                    continue
                for task in tasks:
                    task_id = f"{task[0]}_{task[1]}"
                    if task_id not in transition_dicts:
                        transition_dicts[task_id] = {'states': [], 'actions': [], 'rewards': [], 'next_states': [],
                                                     'dones': []}

                    # 修复使用全局池化获取固定维度状态
                    state_data = gs.get_graph_state_new(env, task, task_complex_index, slot=slot)  # 修复传 slot 参数
                    state = graph_state_to_vector(state_data, method='mean')

                    if hasattr(state, 'to'):
                        state = state.to(agent.device)

                    # 修复计算并传递 action_mask
                    user_id, subtask_id = task
                    task_size_bytes = get_task_size_bytes(ts, user_id, subtask_id)
                    now_time = slot * para["slot_interval"]
                    # 修复传递 Bytes 给 get_action_mask
                    base_mask = ts.get_action_mask(user_id, task_size_bytes, now_time)
                    mask_add = list(base_mask)

                    # 修复转换为 binary mask 给 agent (1=valid, 0=invalid)
                    # 修复使用正确的阈值：valid=0.0, invalid=-1e4/-1e9
                    # 原阈值 -1e8 太小，导致 invalid(-1e4) 也被认为 valid
                    mask_bin = [1.0 if float(m) > -1e3 else 0.0 for m in mask_add]

                    # 【防御】确保长度正确
                    mask_bin = mask_bin[:n_actions] + [0.0] * max(0, n_actions - len(mask_bin))

                    # 基于队列长度的动态 Mask (强制负载均衡) 
                    # 获取当前所有 Edge 的排队情况
                    for eid in range(para["edge_num"]):
                        # 估算排队时间：取该 Edge 最快核心的可用时间
                        core_times = [max(0, t - now_time) for t in ts.edge_useful[eid]]
                        min_wait = min(core_times) if core_times else 0.0
                        avg_wait = sum(core_times) / len(core_times) if core_times else 0.0
                        
                        # 【阈值】如果平均等待超过 1.5 秒 (Deadline 约 1.9s)，直接禁用该 Edge！
                        # 逼迫 Agent 去 Cloud (传输虽慢，但也许只要 0.5s)
                        if avg_wait > 1.5:
                            action_idx = 2 + eid  # Edge 动作索引从 2 开始
                            if action_idx < len(mask_bin):
                                mask_bin[action_idx] = 0.0
                                # if step_count < 10:  # 只在前10步打印，避免刷屏
                                #     print(f"[PPO DEBUG] Masked Edge {eid} due to congestion (avg_wait={avg_wait:.2f}s)")

                    # 再次检查：防止全0导致报错，如果全0则全1兜底
                    if sum(mask_bin) < 0.5:
                        # 如果全堵了，只开放 Cloud (动作 1)
                        mask_bin = [0.0] * n_actions
                        mask_bin[1] = 1.0
                        if step_count < 10:
                            print(f"[PPO WARN] 所有动作被 mask 掉了！fallback 到 Cloud")

                    # 【新增诊断】统计 mask 可用动作数（使用 binary mask）
                    valid_actions = int(sum(mask_bin))
                    if valid_actions == n_actions:
                        mask_stats["all_valid"] += 1
                    elif valid_actions == 1 and mask_bin[0] == 1.0:
                        mask_stats["only_local"] += 1
                    else:
                        mask_stats["cloud_edge_banned"] += 1

                    # 【新增诊断】检查 mask 是否只有一个动作可用
                    if valid_actions == 1:
                        # 修复使用 mask_bin 而不是未定义的 mask_add
                        only_cloud = (mask_bin[1] > 0.5)
                        only_local = (mask_bin[0] > 0.5)
                        only_edge = any(mask_bin[i] > 0.5 for i in range(2, len(mask_bin)))
                        if only_cloud and slot % 1000 < 5:  # 每 1000 步只打印一次
                            print(f"[PPO WARN] Slot {slot}: Mask 只有 Cloud 可用！这将导致策略被锁死！")
                        if only_local and slot % 1000 < 5:
                            print(f"[PPO WARN] Slot {slot}: Mask 只有 Local 可用！这将导致策略被锁死！")
                        if only_edge and slot % 1000 < 5:
                            print(f"[PPO WARN] Slot {slot}: Mask 只有 Edge 可用！这将导致策略被锁死！")

                    # 修复安全解包：兼容不同返回值数量
                    action, *rest = agent.take_action(state, task_id, action_mask=mask_bin, deterministic=False)

                    # 记录任务到动作的映射（用于子任务分桶统计）
                    task2action[tuple(task)] = int(action)

                    # 【新增诊断】统计动作分布
                    if action < len(action_count):
                        action_count[action] += 1
                    total_actions += 1

                    # 修复安全解包：兼容不同返回值数量
                    step_result = bc.step([[task, action]])
                    reward = step_result[0]
                    # 修复正确解包 info
                    info = step_result[1] if len(step_result) > 1 else {}
                    # 修复移除错误的 done = step_result[1]，bc.step() 只返回 (reward, info)
                    # info 是字典，不能作为 done。done 应该基于任务完成状态判断。

                    # 修复定义 app_id（应用 ID 就是用户 ID）
                    app_id = user_id

                    # 修复检查应用级事件（应用完成/超时）
                    curr_to = set(ts.application_timeout_finished)
                    curr_fin = set(ts.application_finished)
                    new_to = curr_to - prev_app_to
                    new_fin = curr_fin - prev_app_fin
                    prev_app_to, prev_app_fin = curr_to, curr_fin

                    # ================= [统一 Utility-Based Reward - 优先 AppTO] =================
                    r_val = to_scalar(reward)

                    # 【优化版】奖励逻辑 - 强制打破局部最优
                    def compute_utility_step_reward(app_timeout, app_done, energy_norm, action_val, r_subtask, finish_ratio=0.0, local_wait_time=0.0, episode_num=0):
                        """
                        【增强版】奖励逻辑 - 更温和的惩罚，更好的引导
                        1. 终局奖励：Done 大奖，Timeout 中等惩罚（降低）
                        2. 过程奖励：适度厌恶排队，鼓励 Cloud
                        """
                        # --- 1. 终局奖励 (Sparse Reward) ---
                        if app_timeout:
                            # 【增强】降低惩罚：从 -20~-25 降到 -10~-12
                            # 避免 agent 过于保守
                            return -10.0 - 2.0 * (1.0 - finish_ratio)

                        if app_done:
                            # 成功大奖：+10.0 (保持不变)
                            # 稍微加一点能耗奖励微调
                            return 10.0 + 1.0 * (1.0 - energy_norm)

                        # --- 2. 过程奖励 (Step Reward) ---
                        # 【增强】适度厌恶排队（降低惩罚系数）
                        step_reward = 0.0  # 基础分归零

                        # 【增强】每排队 1秒，扣 2分（从 5分 降到 2分）
                        if action_val >= 2:  # Edge
                            edge_id = action_val - 2
                            # 获取最快能用上的核心时间
                            core_times = [max(0, t - now_time) for t in ts.edge_useful[edge_id]]
                            min_wait = min(core_times) if core_times else 0.0
                            # 每排队 1秒，扣 2分
                            step_reward -= (min_wait * 2.0)

                        elif action_val == 0:  # Local
                            wait = max(0, ts.devices_exe_useful[user_id] - now_time)
                            # 每排队 1秒，扣 2分
                            step_reward -= (wait * 2.0)

                        # 鼓励 Cloud (补贴传输)
                        # Cloud 没有排队惩罚 (假设无限算力)，只受传输影响
                        if action_val == 1:
                            step_reward += 1.0  # 【增强】从 0.5 提高到 1.0，更鼓励 Cloud

                        # 子任务超时的小惩罚
                        if r_subtask < -2.0:
                            step_reward -= 0.5

                        return step_reward

                    # 计算能耗归一化
                    current_energy = float(info.get("step_energy", 0.0))
                    e_norm = max(0.0, min(1.0, (current_energy - 0.3) / (3.0 - 0.3)))

                    # 修复使用 devices_exe_useful 估算 Local 等待时间
                    local_wait_time = 0.0
                    if action == 0:
                        # 获取当前时间
                        now = slot * para["slot_interval"]
                        # 获取该用户本地 CPU 的释放时间
                        free_time = float(ts.devices_exe_useful[user_id])
                        # 计算排队时间（拥堵程度）
                        local_wait_time = max(0.0, free_time - now)

                    # 计算应用完成度（用于惩罚调整）
                    if hasattr(ts, 'finish_time') and app_id in ts.finish_time:
                        finished_count = len([t for t in ts.finish_time[app_id].values() if t != float('inf')])
                        total_count = max(1, len(ts.finish_time[app_id]))
                        finish_ratio = finished_count / total_count
                    else:
                        finish_ratio = 0.0  # 默认值，表示完全未完成

                    # 使用统一的 utility reward
                    train_reward = compute_utility_step_reward(
                        app_timeout=(app_id in new_to),
                        app_done=(app_id in new_fin),
                        energy_norm=e_norm,
                        action_val=action,
                        r_subtask=r_val,
                        finish_ratio=finish_ratio,  # 传入完成度
                        local_wait_time=local_wait_time,
                        episode_num=episode
                    )

                    # 【增强】值域裁剪：更温和的范围
                    # 【增强】范围：[-12.0, 11.0] - 降低惩罚上限
                    train_reward = max(-12.0, min(11.0, train_reward))
                    # ====================================================================

                    # 修复计算 done 标志（基于任务完成状态）
                    done = False
                    # 检查是否所有任务已完成
                    from Experiments_new.exp_utils import all_arrived_done, safe_rest_tasks_total
                    if all_arrived_done(ts) and safe_rest_tasks_total(ts.rest_tasks) == 0:
                        done = True

                    if hasattr(state, 'to'):
                        state_cpu = state.to('cpu')
                    else:
                        state_cpu = state
                    transition_dicts[task_id]['states'].append(state_cpu)
                    transition_dicts[task_id]['actions'].append(action)
                    transition_dicts[task_id]['rewards'].append(train_reward)
                    episode_rewards.append(train_reward)  # 收集奖励
                    transition_dicts[task_id]['dones'].append(done)

                    next_state_data = gs.get_graph_state_new(env, task, task_complex_index, slot=slot)  # 修复传 slot 参数
                    next_state = graph_state_to_vector(next_state_data, method='mean')
                    if hasattr(next_state, 'to'):
                        next_state = next_state.to('cpu')
                    transition_dicts[task_id]['next_states'].append(next_state)
                    step_count += 1
                    # 【修复3】禁止中途更新！(非常重要)
                    # 原代码：每500步更新一次，导致terminal bonus无法生效
                    # 修改为：pass (什么都不做，等到 Episode 结束统一算总账)
                    if step_count % BATCH_STEPS == 0:
                        pass  # 禁用中途更新

                    # 每 1000 步输出一次进度
                    if step_count - last_print_step >= 1000:
                        print(
                            f"[PPO]   Episode {episode + 1} 进度: 已执行 {step_count} 步 (当前 slot: {slot}/{MAX_STEPS})")
                        sys.stdout.flush()
                        last_print_step = step_count
                if step_count > MAX_STEPS: break
            # 【修复3】已移除这里的学习 - 改为 Episode 结束时统一学习

            # ==========================================
            # 【修复3】Episode 统一更新与 Global Bonus
            # ==========================================
            ts.finalize_episode(slot)

            # 获取本 episode 的真实统计
            timeout_info = calc_timeout_rate(ts)
            app_to = float(timeout_info['app_timeout_rate'])
            task_to = float(timeout_info['task_timeout_rate'])

            # 计算 score
            e_succ, d_succ = ts.get_avg_results(only_successful=True)
            current_score = compute_score(
                E_avg=e_succ,
                D_succ_avg=d_succ,
                rho_app=app_to,
                rho_task=task_to,
                sla0=CONFIG.get("SLA0", 0.95),
                kappa=CONFIG.get("KAPPA", 2.0),
                v_cap=3.0
            )

            # 全局引导：如果本局超时率高，全员连坐扣分
            # 增强惩罚力度：0%超时->0, 100%超时->-20
            terminal_bonus_total = -20.0 * app_to  # 范围: [-20.0, 0]
            if app_to < 0.05:
                terminal_bonus_total += 5.0  # 完美通关奖励

            # 分配 Bonus 并学习
            if transition_dicts:
                # 计算总步数
                total_episode_steps = sum(len(d['rewards']) for d in transition_dicts.values())
                if total_episode_steps > 0:
                    bonus_per_step = terminal_bonus_total / total_episode_steps

                    # 回填 Reward
                    for t_id in transition_dicts:
                        for i in range(len(transition_dicts[t_id]['rewards'])):
                            transition_dicts[t_id]['rewards'][i] += bonus_per_step

                    print(f"[PPO] Ep {episode + 1} 终止引导: AppTO={app_to:.2%}, "
                          f"Bonus总={terminal_bonus_total:.3f}, 每step={bonus_per_step:.4f} (共{total_episode_steps}步)")



                # 只有在这里才进行学习 
                # 修复将所有任务的 transition 合并成一个大的 transition_dict，然后 learn 一次
                # 避免：单个 task 样本少，adv 标准化抖动大，反复更新导致策略崩
                merged_transition = {
                    'states': [],
                    'actions': [],
                    'rewards': [],
                    'next_states': [],
                    'dones': []
                }
                for task_id, transition_dict in transition_dicts.items():
                    if transition_dict['states']:
                        merged_transition['states'].extend(transition_dict['states'])
                        merged_transition['actions'].extend(transition_dict['actions'])
                        merged_transition['rewards'].extend(transition_dict['rewards'])
                        merged_transition['next_states'].extend(transition_dict['next_states'])
                        merged_transition['dones'].extend(transition_dict['dones'])

                # 只 learn 一次（PPO 标准做法）
                if merged_transition['states']:
                    actor_loss, critic_loss = agent.learn(merged_transition)


            agent.clear_task_data()

            # ==========================================

            e, d = ts.get_avg_results()

            # 【新增诊断】打印子任务完成统计（用于诊断环境/超时定义是否一致）
            total_subtasks, finished_subtasks, unfinished_subtasks, timeout_subtasks = subtask_outcome_stats(ts)
            print(
                f"[PPO] 【环境一致性】Ep {episode + 1} 子任务统计: total={total_subtasks}, finished={finished_subtasks}, unfinished={unfinished_subtasks}, timeout={timeout_subtasks}")

            # 检查是否有改善
            if current_score < best_score:
                best_score = current_score
                flag = "新最佳"

                # 保存 best episode 的完整 metrics
                best_metrics = {
                    "e": float(e),
                    "d": float(d_succ),
                    "score": float(current_score),
                    "timeout_info": dict(timeout_info),
                    "action_count": list(action_count),
                    "total_actions": int(total_actions),
                    "mask_stats": dict(mask_stats),
                    "task2action": dict(task2action),
                }

                # 修复计算并添加 action_stats 到 timeout_info
                partition = subtask_partition_stats(ts, task2action)
                best_metrics["timeout_info"]["action_stats"] = {
                    "local": partition["local"],
                    "cloud": partition["cloud"],
                    "edge": partition["edge"],
                    "timeout": partition["timeout"],
                    "unknown": partition["unknown"],
                    "total": partition["total_subtasks"],
                    "total_actions": int(total_actions),
                }

                # 保存最佳模型
                if checkpoint_dir and not is_eval_mode:
                    try:
                        meta = {
                            "seed": int(CONFIG["SEED"]),
                            "seed_offset": int(seed_offset),
                            "state_dim": int(n_states),
                            "action_dim": int(n_actions),
                            "edge_num": int(para["edge_num"]),
                            "deadline_slot": int(para["deadline_slot"]),
                            "uplink_range": list(para["uplink_range"]),
                            "task_complex_index": int(task_complex_index),
                            "score": float(current_score),
                            "energy": float(e),
                            "delay": float(d_succ),
                            "app_timeout_rate": float(timeout_info['app_timeout_rate']),
                            "use_gnn": False,
                        }
                        save_model_bundle("PPO", agent, checkpoint_dir, **meta)
                        print(f"[PPO] 已保存最佳模型 (Score={current_score:.4f})")
                    except Exception as save_err:
                        print(f"[PPO] 保存模型失败: {save_err}")
            else:
                flag = ""

            # Reward统计
            if episode_rewards:
                episode_rewards_array = np.array(episode_rewards)
                total_reward = float(np.sum(episode_rewards))
                print(f"[PPO] Ep {episode + 1} Reward统计: sum={total_reward:.4f}, mean={episode_rewards_array.mean():.4f}, "
                      f"min={episode_rewards_array.min():.4f}, max={episode_rewards_array.max():.4f}, count={len(episode_rewards)}")
            else:
                total_reward = 0.0
                print(f"[PPO] Ep {episode + 1} Reward统计: 无数据 (episode_rewards 为空, step_count={step_count})")

            # 保存训练曲线（每个 episode 都保存，不管是不是 eval 模式）
            if checkpoint_dir:
                try:
                    curve_file = Path(CONFIG["RUN_DIR"]) / "curves" / f"PPO_seed{seed_offset}.csv"
                    curve_file.parent.mkdir(parents=True, exist_ok=True)
                    append_curve_row(curve_file, {
                        "episode": episode + 1,
                        "reward": total_reward,  # 修复使用实际奖励总和
                        "score": float(current_score),
                        "energy": float(e),
                        "delay": float(d_succ),
                        "app_timeout_rate": float(timeout_info['app_timeout_rate']),
                        "best_score": float(best_score),
                    })
                except Exception as curve_err:
                    print(f"[PPO] 保存曲线失败: {curve_err}")

            # 降低诊断输出频率，每 20 个 episode 打印一次
            if (episode + 1) % 20 == 0:
                diagnose_timeout(ts, "PPO")
                # 【新增诊断】打印动作分布和mask统计
                if total_actions > 0:
                    print(f"[PPO] 动作分布统计 (共{total_actions}个动作):")
                    print(
                        f"  - Local (action=0): {action_count[0]} ({100.0 * action_count[0] / total_actions if total_actions > 0 else 0:.1f}%)")
                    if n_actions > 1:
                        print(
                            f"  - Cloud (action=1): {action_count[1]} ({100.0 * action_count[1] / total_actions if total_actions > 0 else 0:.1f}%)")
                    if n_actions > 2:
                        edge_total = sum(action_count[2:])
                        print(
                            f"  - Edge (action=2..{n_actions - 1}): {edge_total} ({100.0 * edge_total / total_actions if total_actions > 0 else 0:.1f}%)")
                    print(f"[PPO] Mask 统计:")
                    print(f"  - 所有动作可用: {mask_stats['all_valid']}")
                    print(f"  - 仅 Local 可用: {mask_stats['only_local']}")
                    print(f"  - Cloud/Edge 被禁用: {mask_stats['cloud_edge_banned']}")
                    if action_count[0] > 0.95 * total_actions:
                        print(f"  [警告] PPO几乎全选Local，可能退化！")
                        if mask_stats['only_local'] > 0.5 * total_actions:
                            print(f"  [分析] 可能原因：Mask太严，大部分时间只能选Local")
                        else:
                            print(f"  [分析] 可能原因：PPO没有学会卸载策略")

            print(f"[PPO] Ep {episode + 1}/{EPISODES} 完成: steps={step_count}, "
                  f"App超时率={timeout_info['app_timeout_rate']:.2%}, Score={current_score:.4f}, "
                  f"E={e:.4f}, D={d:.4f}, Reward={total_reward:.4f} {flag}")
            sys.stdout.flush()

        # 直接使用训练过程中的最好结果，不再进行 deterministic 评估
        if best_metrics is not None:
            best_e = best_metrics["e"]
            best_d = best_metrics["d"]
            best_timeout_info = best_metrics["timeout_info"]
            print(f"[PPO] 【使用训练最佳成绩】best_score={best_score:.4f}")
            sys.stdout.flush()

            # 保存决策顺序trace（使用最佳训练结果）
            trace_dir = ensure_trace_dir(CONFIG["RUN_DIR"]) if CONFIG.get("RUN_DIR") else None
            if trace_dir and 'trace_decisions' in locals() and trace_decisions:
                pd.DataFrame(trace_decisions).to_csv(trace_dir / "PPO_decisions.csv", index=False)

            # 返回训练最佳成绩的 timeout_info（已包含子任务统计）
            return best_e, best_d, best_timeout_info
        else:
            print(f"[PPO] [警告] 未找到最佳训练结果，使用当前结果")
            sys.stdout.flush()
            return 1000.0, 1000.0, {'timeout_rate': 1.0, 'app_timeout_rate': 1.0, 'task_timeout_rate': 1.0}

        for slot in range(MAX_STEPS):
            # 统计 slot 数
            total_eval_slots += 1

            # 修复使用预生成的到达计划，避免 random 被污染
            apply_arrival_plan(ts, slot, arrival_plan)
            ts.check_timeouts(slot)

            # 修复检查退出条件 - 确保所有任务都已完成
            if slot >= CONFIG["STOP_ARRIVAL_STEP"]:
                # 所有已到达应用都完成 + 没有剩余任务
                if all_arrived_done(ts) and safe_rest_tasks_total(ts.rest_tasks) == 0:
                    break

            # 使用默认顺序，不使用任何启发式排序
            tasks = gs.get_tasks(slot, sort_tasks=False)
            if not tasks:
                # 统计空 slot
                empty_slots += 1
                continue

            for task in tasks:
                task_id = f"{task[0]}_{task[1]}"
                if task_id not in transition_dicts:
                    transition_dicts[task_id] = {'states': [], 'actions': [], 'rewards': [], 'next_states': [],
                                                 'dones': []}

                state_data = gs.get_graph_state_new(env, task, task_complex_index, slot=slot)  # 修复传 slot 参数
                state = graph_state_to_vector(state_data, method='mean')

                if hasattr(state, 'to'):
                    state = state.to(agent.device)

                # 修复使用 deterministic=True 和 action_mask
                user_id, subtask_id = task
                task_size_bytes = get_task_size_bytes(ts, user_id, subtask_id)
                now_time = slot * para["slot_interval"]
                # 修复传递 Bytes 给 get_action_mask
                base_mask = ts.get_action_mask(user_id, task_size_bytes, now_time)
                mask_add = list(base_mask)

                # 修复转换为 binary mask 给 agent (1=valid, 0=invalid)
                # 【暂时禁用mask】使用全1 mask，所有动作都可用
                mask_bin = [1.0] * len(mask_add)  # [1.0 if float(m) > -1e8 else 0.0 for m in mask_add]

                # 记录推理时间
                inference_start = time.time()
                # 修复安全解包：兼容不同返回值数量
                action, *rest = agent.take_action(state, task_id, action_mask=mask_bin, deterministic=True)
                inference_time_ms = (time.time() - inference_start) * 1000.0
                inference_times.append(inference_time_ms)

                # 记录任务到动作的映射（用于子任务分桶统计）
                task2action[tuple(task)] = int(action)

                # 记录决策顺序
                target, edge_id = decode_action(int(action))
                trace_decisions.append({
                    "decision_idx": decision_idx,
                    "slot": slot,
                    "uid": int(user_id),
                    "sid": int(subtask_id),
                    "action": int(action),
                    "target": target,
                    "edge_id": int(edge_id)
                })
                decision_idx += 1

                # 统计动作分布
                if 0 <= action < len(action_counts):
                    action_counts[action] += 1
                total_eval_actions += 1

                # 修复安全解包：兼容不同返回值数量
                step_result = bc.step([[task, action]])
                reward = step_result[0]
                # 修复移除错误的 done = step_result[1]，评估模式不需要 done 标志

                # 修复评估 loop 不需要保存 transition_dicts（只为了性能评估）
                # 删除多余的 reward 计算，避免混淆
                # next_state_data = gs.get_graph_state_new(env, task, task_complex_index)  # 修复传 int index
                # next_state = graph_state_to_vector(next_state_data, method='mean')
                # if hasattr(next_state, 'to'):
                #     next_state = next_state.to('cpu')
                # transition_dicts[task_id]['states'].append(state_cpu)
                # transition_dicts[task_id]['actions'].append(action)
                # transition_dicts[task_id]['rewards'].append(train_reward)
                # transition_dicts[task_id]['dones'].append(done)
                # transition_dicts[task_id]['next_states'].append(next_state)
                step_count += 1

            if step_count > MAX_STEPS:
                break


        ts.finalize_episode(slot)
        eval_e, eval_d = ts.get_avg_results()
        eval_timeout_info = calc_timeout_rate(ts)

        # 使用新版 compute_score，与所有算法保持一致
        eval_e_succ, eval_d_succ = ts.get_avg_results(only_successful=True)
        app_to = eval_timeout_info['app_timeout_rate']
        task_to = eval_timeout_info['task_timeout_rate']

        # 从 CONFIG 读取统一评分参数
        sla0 = CONFIG.get("SLA0", 0.95)
        kappa = CONFIG.get("KAPPA", 2.0)
        v_cap = 3.0

        # 使用统一的评分函数
        eval_score = compute_score(
            E_avg=eval_e_succ,
            D_succ_avg=eval_d_succ,
            rho_app=app_to,
            rho_task=task_to,
            sla0=sla0,
            kappa=kappa,
            v_cap=v_cap
        )

        print(f"[PPO] 评估结果: E={eval_e:.4f}, D={eval_d:.4f}, Score={eval_score:.4f}, "
              f"App超时率={eval_timeout_info['app_timeout_rate']:.2%}")
        sys.stdout.flush()

        # 最终Score计算（使用评估结果）
        eval_timeout_info['score'] = eval_score

        # 添加子任务级统计和动作选择统计
        total_subtasks, finished_subtasks, unfinished_subtasks, timeout_subtasks = subtask_outcome_stats(ts)
        eval_timeout_info["subtask_stats"] = {
            "total": int(total_subtasks),
            "finished": int(finished_subtasks),
            "unfinished": int(unfinished_subtasks),
            "timeout": int(timeout_subtasks)
        }
        partition = subtask_partition_stats(ts, task2action)
        if total_eval_actions > 0:
            eval_timeout_info["action_stats"] = {
                "local": partition["local"],
                "cloud": partition["cloud"],
                "edge": partition["edge"],
                "timeout": partition["timeout"],
                "unknown": partition["unknown"],
                "total": partition["total_subtasks"],  # 现在这个 total 就是"子任务总数"
                "total_slots": total_eval_slots,  # 修复实际运行的 slot 数
                "empty_slots": empty_slots,  # 空 slot 数（无任务）
                "total_actions": total_eval_actions  # 总动作数
            }

        # 推理时间统计
        if inference_times:
            eval_timeout_info['inference_time_ms'] = float(np.mean(inference_times))
            eval_timeout_info['inference_stats'] = {
                'median': float(np.median(inference_times)),
                'min': float(np.min(inference_times)),
                'max': float(np.max(inference_times)),
                'std': float(np.std(inference_times)),
                'count': len(inference_times)
            }
            print(f"[PPO] 推理时间统计: 平均={eval_timeout_info['inference_time_ms']:.4f}ms, "
                  f"中位数={eval_timeout_info['inference_stats']['median']:.4f}ms, "
                  f"最小={eval_timeout_info['inference_stats']['min']:.4f}ms, "
                  f"最大={eval_timeout_info['inference_stats']['max']:.4f}ms, "
                  f"共{len(inference_times)}次决策")
        else:
            eval_timeout_info['inference_time_ms'] = 0.0
            eval_timeout_info['inference_stats'] = {
                'median': 0.0, 'min': 0.0, 'max': 0.0, 'std': 0.0, 'count': 0
            }

        # 【公平对比】返回 deterministic 评估的指标，而不是训练过程中的 best episode
        # 这样可以与 Greedy/Genetic 进行公平对比（都是固定策略、固定环境）
        print(
            f"[PPO] 【使用训练最佳成绩】best_score={best_score:.4f}, AppTO={best_timeout_info['app_timeout_rate']:.2%})")
        sys.stdout.flush()

        # 保存决策顺序trace
        if trace_dir and trace_decisions:
            pd.DataFrame(trace_decisions).to_csv(trace_dir / "PPO_decisions.csv", index=False)

        # 返回训练最佳成绩的 timeout_info（已包含子任务统计）
        return best_e, best_d, best_timeout_info
    except Exception as e:
        traceback.print_exc()
        return 1000.0, 1000.0, {'timeout_rate': 1.0, 'app_timeout_rate': 1.0, 'task_timeout_rate': 1.0, 'error': str(e)}
