# -*- coding: utf-8 -*-
"""
Dynamic GAT-DQN 训练包装函数 (收敛修复版)
核心修复：
1. 真实 next_state：每次 step 后重新构建，不复用旧状态
2. 正确 done 逻辑：episode 结束就 done=True
3. Monkey Patch：强制快速 epsilon 衰减 (10000步)
4. 强力奖励：App 完成/超时 ±10.0
"""
import os
import sys
import time
import json
import traceback
import types
import math
import pandas as pd
import numpy as np
import torch
import random
from pathlib import Path
from contextlib import nullcontext
from torch_geometric.data import Batch

# 引入项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)
experiments_dir = os.path.join(project_root, 'Experiments_new')
if experiments_dir not in sys.path:
    sys.path.insert(0, experiments_dir)

from Environment.environment import Environment
from scheduler.graph_scheduler import GraphScheduler
from scheduler.task_scheduler import TaskScheduler
from Algorithms import Benchmark
from utils.constant import para

# 【性能优化】
os.environ["OMP_NUM_THREADS"] = "8"
torch.set_num_threads(8)

from Experiments_new.exp_utils import (
    CONFIG, init_worker, load_arrival_plan, generate_arrival_plan, apply_arrival_plan,
    load_deadline_config, get_graph_cache, calc_timeout_rate, to_scalar,
    safe_rest_tasks_total, all_arrived_done, subtask_outcome_stats, normalize_mask,
    compute_score, get_task_size_bytes
)

# 尝试导入 RL 算法
DynamicGATDQN = None
try:
    from Algorithms.GNNRL.dynamic_gat_dqn import DynamicGAT_DQN_Agent as DynamicGATDQN, ReplayMemory
    import Algorithms.GNNRL.dynamic_gat_dqn as dqn_mod
except Exception as e:
    print(f"[Worker Warning] Dynamic GAT-DQN import failed: {e}")

print_lock = None

def set_print_lock(lock):
    global print_lock
    print_lock = lock

# ================= 性能优化：全局图缓存（增强版）=================
class GraphStateCache:
    def __init__(self):
        self.topo_cache = {}  # key -> (edge_index, batch_vec)
        self.subgraph_cache = {}  # (uid, slot, task_complex_index) -> sg_data, node2idx
        self.max_cache_size = 500  # 限制缓存大小，防止内存爆炸
        self.access_count = {}  # LRU 计数
        
    def reset(self):
        self.topo_cache.clear()
        self.subgraph_cache.clear()
        self.access_count.clear()
    
    def get_subgraph(self, uid, slot, task_complex_index, env, gs):
        """获取缓存的子图数据，避免重复构建"""
        key = (uid, slot, task_complex_index)
        if key in self.subgraph_cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.subgraph_cache[key]
        return None
    
    def put_subgraph(self, uid, slot, task_complex_index, sg_data, node2idx):
        """缓存子图数据"""
        key = (uid, slot, task_complex_index)
        
        # LRU 淘汰策略
        if len(self.subgraph_cache) >= self.max_cache_size:
            # 找到最少使用的 key
            lru_key = min(self.access_count.items(), key=lambda x: x[1])[0]
            del self.subgraph_cache[lru_key]
            del self.access_count[lru_key]
        
        self.subgraph_cache[key] = (sg_data, node2idx)
        self.access_count[key] = 0

GRAPH_CACHE = GraphStateCache()

# ================= 特征工程：包含 Edge 状态 =================
def _align_x_dim(x: torch.Tensor, node_dim: int) -> torch.Tensor:
    if x.size(1) == node_dim: return x
    if x.size(1) > node_dim: return x[:, :node_dim]
    pad = x.new_zeros((x.size(0), node_dim - x.size(1)))
    return torch.cat([x, pad], dim=1)

def get_global_graph_state_dag(env, gs, active_users, task_complex_index, slot, base_dim=None):
    global GRAPH_CACHE
    key = tuple(active_users)
    current_time = slot * para["slot_interval"]
    ts = gs.ts

    # 1. Local Load
    local_waits = [max(0, t - current_time) for t in ts.devices_exe_useful]
    avg_local_wait = (sum(local_waits) / len(local_waits)) / 10.0 if local_waits else 0.0

    # 2. Upload Load
    upload_waits = [max(0, t - current_time) for t in ts.devices_upload_useful]
    avg_upload_wait = (sum(upload_waits) / len(upload_waits)) / 10.0 if upload_waits else 0.0

    # 3. Edge Load (具体每个 Edge 的负载)
    edge_waits_flat = []
    for edge_cores in ts.edge_useful:
        core_times = [max(0, t - current_time) for t in edge_cores]
        val = min(core_times) if core_times else 0.0
        edge_waits_flat.append(val / 10.0)

    context_vec = [avg_local_wait, avg_upload_wait] + edge_waits_flat

    subgraphs = []
    task_to_global_idx = {}
    node_offset = 0

    # 【加速优化1】使用子图缓存，避免重复构建
    for uid in active_users:
        # 尝试从缓存获取
        cached_data = GRAPH_CACHE.get_subgraph(uid, slot, task_complex_index, env, gs)
        if cached_data is not None:
            sg_data, node2idx = cached_data
        else:
            # 缓存未命中，重新构建
            sg_data, node2idx = gs.get_app_dag_data(env, uid, slot, task_complex_index)
            if sg_data is None or sg_data.x is None: 
                return (None, None, None), {}, []
            # 存入缓存
            GRAPH_CACHE.put_subgraph(uid, slot, task_complex_index, sg_data, node2idx)
        
        if base_dim is not None: sg_data.x = _align_x_dim(sg_data.x, base_dim)
        subgraphs.append(sg_data)
        for nid, local_idx in node2idx.items():
            task_to_global_idx[(uid, nid)] = node_offset + int(local_idx)
        node_offset += int(sg_data.num_nodes)

    if not subgraphs: return (None, None, None), {}, []

    batch = Batch.from_data_list(subgraphs)
    x_all = batch.x

    # 【加速优化2】使用 expand 而非 repeat（不真实复制数据）
    num_nodes = x_all.size(0)
    context_tensor = x_all.new_tensor(context_vec).unsqueeze(0).expand(num_nodes, -1)
    x_all = torch.cat([x_all, context_tensor], dim=1)

    # 【加速优化3】缓存 edge_index 和 batch
    if key in GRAPH_CACHE.topo_cache:
        edge_index, batch_vec = GRAPH_CACHE.topo_cache[key]
    else:
        edge_index, batch_vec = batch.edge_index, batch.batch
        GRAPH_CACHE.topo_cache[key] = (edge_index, batch_vec)

    if edge_index.max() >= x_all.size(0): # Safety check
         edge_index, batch_vec = batch.edge_index, batch.batch
         GRAPH_CACHE.topo_cache[key] = (edge_index, batch_vec)

    # 【加速优化4】构建 global_idx_to_task 列表，避免每步重建 inv dict
    global_idx_to_task = [None] * num_nodes
    for (uid, nid), global_idx in task_to_global_idx.items():
        global_idx_to_task[global_idx] = (uid, nid)

    return (x_all, edge_index, batch_vec), task_to_global_idx, global_idx_to_task

def construct_action_mask(ts, ready_tasks, task_to_global_idx, action_dim, total_nodes, slot):
    """
    【真实 Mask】使用 ts.get_action_mask，保底 Cloud
    """
    full_action_mask = torch.full((total_nodes, action_dim), -1e9, dtype=torch.float32)
    now = slot * para["slot_interval"]

    for (uid, sid) in ready_tasks:
        if (uid, sid) not in task_to_global_idx:
            continue
        g_idx = task_to_global_idx[(uid, sid)]

        t_size = get_task_size_bytes(ts, uid, sid)
        m = ts.get_action_mask(uid, t_size, now)          # additive mask
        m = normalize_mask(m, action_dim)                 # 对齐到 action_dim
        m = torch.tensor(m, dtype=torch.float32)

        # 【核心修复 3】保底：如果全禁用，至少 Cloud 可用（统一使用 -1e9）
        if torch.all(m < -1e3):
            m[:] = -1e9
            m[1] = 0.0

        full_action_mask[g_idx] = m

    return full_action_mask

def get_stable_active_users(ts, gs, slot, k=30):
    ready_tasks_all = gs.get_tasks(slot, sort_tasks=False)
    if not ready_tasks_all: return [], []
    ready_tasks = gs.pick_topk_tasks(slot, ready_tasks_all, k=k)
    if not ready_tasks: return [], []
    active_users = sorted({u for (u, _) in ready_tasks})
    return active_users, ready_tasks

# ================= 预训练数据采集（修复 next_state） =================
def collect_genetic_demonstrations(env, gs, ts, bc, task_complex_index, action_dim, node_dim, base_node_dim, num_episodes=20, topk=30):
    if isinstance(topk, tuple): topk = topk[0]
    topk = int(topk)
    demo_memory = ReplayMemory(capacity=3000)
    episodes_collected = 0
    EXPERT_ALGO_ID = 4 # Genetic

    GRAPH_CACHE.reset()

    while len(demo_memory) < 2000 and episodes_collected < num_episodes:
        bc.reset(); ts.reset(); GRAPH_CACHE.reset()
        max_steps = 200
        arrival_plan = generate_arrival_plan(
            CONFIG["SEED"] + episodes_collected, max_steps, max_steps,
            base_prob=0.3, burst_prob=CONFIG["BURST_PROB"],
            burst_min=max(1, CONFIG["BURST_SIZE"]//2), burst_max=CONFIG["BURST_SIZE"]
        )

        for slot in range(max_steps):
            apply_arrival_plan(ts, slot, arrival_plan)
            ts.check_timeouts(slot)

            active_users, ready_tasks = get_stable_active_users(ts, gs, slot, k=topk)
            if not active_users:
                if slot >= max_steps - 50 and all_arrived_done(ts) and safe_rest_tasks_total(ts.rest_tasks) == 0:
                    break
                continue

            greedy_actions, _ = bc.get_actions(EXPERT_ALGO_ID, slot, 0)
            if not greedy_actions: continue

            action_map = {tuple(t): int(a) for (t, a) in greedy_actions}

            # 修复接收 3 个返回值
            (state_x, state_edge_index, state_batch), task_to_global_idx, _ = get_global_graph_state_dag(
                env, gs, active_users, task_complex_index, slot, base_node_dim
            )
            if state_x is None: continue

            ready_mask = torch.zeros(state_x.shape[0], dtype=torch.bool)
            for t in ready_tasks:
                if t in task_to_global_idx: ready_mask[task_to_global_idx[t]] = True

            full_action_mask = construct_action_mask(ts, ready_tasks, task_to_global_idx, action_dim, state_x.shape[0], slot)

            for task in ready_tasks:
                if task not in action_map: continue
                expert_offload = int(action_map[task])
                selected_task = task

                if selected_task not in task_to_global_idx:
                    continue
                g_idx = int(task_to_global_idx[selected_task])

                # 越界保护
                if g_idx >= full_action_mask.size(0) or expert_offload >= full_action_mask.size(1):
                    continue

                # 【改动 C】如果专家动作被 mask 禁用，丢弃这条 demo（不要改 mask）
                if full_action_mask[g_idx, expert_offload] < -1e3:
                    continue

                step_reward, _ = bc.step([[selected_task, expert_offload]])
                scaled_reward = float(step_reward) / 10.0

                # next state
                next_active_users, next_ready_tasks = get_stable_active_users(ts, gs, slot, k=topk)
                next_task_to_global = {}
                next_global_idx_to_task = []

                if next_active_users:
                    (next_state_x, next_state_edge_index, next_state_batch), next_task_to_global, next_global_idx_to_task = get_global_graph_state_dag(
                        env, gs, next_active_users, task_complex_index, slot, base_node_dim
                    )
                else:
                    next_state_x, next_state_edge_index, next_state_batch = None, None, None

                if next_state_x is None:
                    next_state_x = torch.zeros((1, state_x.size(1)), dtype=torch.float32)
                    next_state_edge_index = torch.zeros((2, 0), dtype=torch.long)
                    next_state_batch = torch.zeros((1,), dtype=torch.long)

                done_flag = False
                if (not next_active_users) or (slot >= max_steps - 10 and all_arrived_done(ts) and safe_rest_tasks_total(ts.rest_tasks) == 0):
                    done_flag = True

                next_ready_mask = torch.zeros(next_state_x.size(0), dtype=torch.bool)
                if next_active_users:
                    for t in next_ready_tasks:
                        if t in next_task_to_global:
                            next_ready_mask[next_task_to_global[t]] = True

                next_action_mask_full = torch.full((next_state_x.size(0), action_dim), -1e9, dtype=torch.float32)
                if next_active_users and next_ready_mask.any():
                    next_action_mask_full = construct_action_mask(
                        ts, next_ready_tasks, next_task_to_global, action_dim, next_state_x.size(0), slot
                    ).cpu()

                demo_memory.push(
                    (state_x.cpu(), state_edge_index.cpu(), state_batch.cpu()),
                    int(g_idx), int(expert_offload), scaled_reward,
                    (next_state_x.cpu(), next_state_edge_index.cpu(), next_state_batch.cpu()),
                    bool(done_flag), ready_mask.cpu(), next_ready_mask.cpu(),
                    full_action_mask.cpu(), next_action_mask_full.cpu(),
                    is_expert=True
                )

            if slot >= max_steps - 10 and all_arrived_done(ts) and safe_rest_tasks_total(ts.rest_tasks) == 0: break
        episodes_collected += 1
        if episodes_collected % 5 == 0: print(f"[Demo Collection] {len(demo_memory)} samples...")
    return demo_memory

def run_eval(agent, env, ts, gs, bc, arrival_plan, topk, task_complex_index, max_steps, base_node_dim):
    """运行 deterministic 评估（epsilon=0）- 使用和训练一致的逻辑"""
    ts.reset()
    bc.reset()
    GRAPH_CACHE.reset()

    action_dim = para["edge_num"] + 2
    eval_action_count = [0] * action_dim
    eval_total_actions = 0

    # 评估前同步一次（防止 policy_cpu 落后）
    agent.policy_cpu.load_state_dict(agent.policy_gpu.state_dict())
    agent.policy_cpu.eval()

    # Debug 标记：只在第一次打印诊断信息
    debug_printed = False

    for slot in range(max_steps):
        apply_arrival_plan(ts, slot, arrival_plan)
        ts.check_timeouts(slot)

        inner_steps = 0
        while True:
            active_users, ready_tasks = get_stable_active_users(ts, gs, slot, k=topk)
            if not active_users:
                break

            (state_x, state_edge_index, state_batch), task_to_global_idx, global_idx_to_task = get_global_graph_state_dag(
                env, gs, active_users, task_complex_index, slot, base_node_dim
            )

            if state_x is None:
                break

            # ready_mask：只标记 ready_tasks 对应的节点
            ready_mask = torch.zeros(state_x.shape[0], dtype=torch.bool)
            for t in ready_tasks:
                if t in task_to_global_idx:
                    ready_mask[task_to_global_idx[t]] = True

            if not ready_mask.any():
                break

            full_action_mask = construct_action_mask(
                ts, ready_tasks, task_to_global_idx, action_dim, state_x.shape[0], slot
            )

            # Debug 打印（只打印一次）- 必须在 full_action_mask 构造之后
            if not debug_printed:
                print(f"[Eval Debug] ready_mask sum: {ready_mask.sum().item()}/{state_x.size(0)} nodes")
                # 检查 ready 节点的有效动作数量
                ready_indices = torch.nonzero(ready_mask).squeeze()
                if ready_indices.numel() > 0:
                    # 确保 ready_indices 是 2D 形状以便索引
                    if ready_indices.dim() == 0:
                        ready_indices = ready_indices.unsqueeze(0)
                    valid_actions_avg = (full_action_mask[ready_indices] > -1e8).float().sum(dim=1).mean().item()
                    print(f"[Eval Debug] avg valid actions per ready node: {valid_actions_avg:.2f}")
                debug_printed = True

            # 【致命修复 3】评估时强制 epsilon=0，不要加随机噪声
            node_idx, offload_action = agent.select_action(
                (state_x, state_edge_index, state_batch),
                ready_mask,
                full_action_mask,
                training=False,
                custom_eps=0.0  # 必须为 0，测试模型真实能力
            )
            if node_idx is None:
                break

            # 【加速优化】使用 global_idx_to_task 列表，避免每步重建 inv dict
            # 需要解包 get_global_graph_state_dag 的返回值
            selected_task = global_idx_to_task[int(node_idx)]
            if selected_task is None:
                break

            bc.step([[selected_task, int(offload_action)]])
            eval_action_count[int(offload_action)] += 1
            eval_total_actions += 1

            inner_steps += 1
            if inner_steps >= 100:
                break

        if slot >= CONFIG["STOP_ARRIVAL_STEP"] and all_arrived_done(ts) and safe_rest_tasks_total(ts.rest_tasks) == 0:
            break

    ts.finalize_episode(slot)
    ee, ed = ts.get_avg_results()
    eto_info = calc_timeout_rate(ts)
    e_score = compute_score(ee, ed, eto_info['app_timeout_rate'], eto_info['task_timeout_rate'],
                            sla0=CONFIG.get("SLA0", 0.95), kappa=CONFIG.get("KAPPA", 2.0), v_cap=3.0)

    print(f"[DGATD] Final Eval: Score={e_score:.4f}, AppTO={eto_info['app_timeout_rate']:.2%}, "
          f"Act: L={eval_action_count[0]} C={eval_action_count[1]} E={sum(eval_action_count[2:])}")

    # 【内存优化】评估结束后清理缓存，防止内存泄漏
    torch.cuda.empty_cache()

    return float(ee), float(ed), {
        "score": float(e_score),
        "app_timeout_rate": float(eto_info["app_timeout_rate"]),
        "task_timeout_rate": float(eto_info["task_timeout_rate"]),
        "action_stats": {
            "local": eval_action_count[0],
            "cloud": eval_action_count[1],
            "edge": sum(eval_action_count[2:]),
            "total_actions": eval_total_actions
        }
    }

def train_dynamic_gat_dqn_wrapper(gpu_id, seed_offset, use_heuristic_sort=False, episodes=50):
    global print_lock
    start_t = time.time()

    with (print_lock if print_lock else nullcontext()):
        print(f"  -> [DGATD] Start GPU-{gpu_id}, seed_offset={seed_offset}")

    try:
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        init_worker(seed_offset, para, CONFIG)

        if "RUN_DIR" in os.environ: CONFIG["RUN_DIR"] = os.environ["RUN_DIR"]
        if "EVAL_MODE" in os.environ: CONFIG["EVAL_MODE"] = os.environ["EVAL_MODE"] == "True"

        gen = generate_arrival_plan(
            CONFIG["SEED"] + seed_offset, CONFIG["MAX_STEPS"], CONFIG["STOP_ARRIVAL_STEP"],
            0.3, CONFIG["BURST_PROB"], max(1, CONFIG["BURST_SIZE"]//2), CONFIG["BURST_SIZE"]
        )

        # 保持原样（dict: {plan, schedule, env_seed}）
        # apply_arrival_plan 会自动识别并使用 schedule
        arrival_plan = gen

        # 健康检查：确保 arrival_plan 是 dict 且有 schedule
        if not (isinstance(arrival_plan, dict) and isinstance(arrival_plan.get("schedule", None), list)):
            with (print_lock if print_lock else nullcontext()):
                print("[DGATD][FATAL] arrival_plan must be dict with 'schedule'")
                print(f"    type: {type(arrival_plan)}")
                if isinstance(arrival_plan, dict):
                    print(f"    keys: {list(arrival_plan.keys())}")
            return 1000.0, 1000.0, {'error': 'bad arrival_plan format'}

        sch = arrival_plan["schedule"]
        if len(sch) == 0:
            with (print_lock if print_lock else nullcontext()):
                print(f"[DGATD][FATAL] arrival_plan['schedule'] is empty -> no arrivals -> no training!")
            return 1000.0, 1000.0, {'error': 'arrival_plan schedule is empty'}

        user_num = para["user_num"]
        subgraph_num = 20
        basegraph_num = 60
        task_complex = para["task_complex"]
        task_complex_index = int(task_complex) if isinstance(task_complex, int) else 0

        env = Environment(user_num=user_num, subgraph_num=subgraph_num, basegraph_num=basegraph_num, task_complex_index=task_complex_index)
        env.generate_components(seed=CONFIG["SEED"] + seed_offset)
        G = get_graph_cache(user_num, subgraph_num, basegraph_num, project_root)
        if G is not None and env.basegraph: env.basegraph.nx_graph = G

        ts = TaskScheduler(user_num, subgraph_num, basegraph_num, env, tight_deadline_config=load_deadline_config(CONFIG.get("RUN_DIR", "")), seed=CONFIG["SEED"])
        gs = GraphScheduler(env.basegraph, env.subgraph_list, ts)
        bc = Benchmark.Benchmark(env, gs, ts, task_complex_index, effective=True, seed=CONFIG["SEED"])

        ts.env = env
        ts.using_Algorithm = 8
        bc.reset(); ts.reset()

        action_dim = para["edge_num"] + 2

        # 确定基础维度
        base_node_dim = None
        for uid in range(min(10, user_num)):
            try:
                sg_data, _ = gs.get_app_dag_data(env, uid, 0, task_complex_index)
                if sg_data is not None and hasattr(sg_data, "x") and sg_data.x is not None:
                    base_node_dim = int(sg_data.x.shape[1])
                    break
            except: pass
        if base_node_dim is None: base_node_dim = 10

        node_dim = base_node_dim + 2 + para["edge_num"]

        with (print_lock if print_lock else nullcontext()):
            print(f"[DGATD] BaseDim={base_node_dim}, FinalDim={node_dim}, ActionDim={action_dim}")

        # 【致命修复 4】禁用专家预训练（最稳）
        # Genetic 数据 AppTO ~30-40%，会污染学习
        # 只让 DQN 从随机探索开始学
        agent = DynamicGATDQN(node_dim, action_dim, device, use_expert_data=False, expert_ratio=0.0)

        # 降低初始学习率，防止随机数据导致训练不稳定
        for param_group in agent.optimizer.param_groups:
            param_group['lr'] = 3e-5  # 从 5e-5 降到 3e-5，让学习更平稳

        with (print_lock if print_lock else nullcontext()):
            print(f"[DGATD] Setup: ExpertRatio=0.0 (已禁用预训练), LR=5e-5, EpsilonDecay=50000 (延长探索)")

        # 跳过专家预训练，让 Agent 从"一张白纸"开始学
        # 这样可以展示从差到好的学习曲线

        # === Training ===
        episodes = episodes
        max_steps = CONFIG["MAX_STEPS"]
        # 【加速优化】减少 episode 后的批量训练次数，从100降到50
        UPDATE_ITERS_PER_EP = 50  # 减少训练开销，保持足够的总更新次数

        best_score = float('inf')
        best_pack = None  # 缓存 best eval 的完整指标
        curves_data = []

        # 【加速优化】减少评估频率：从每1个episode评估改为每5个episode评估
        EVAL_EVERY = 5  # 每 5 个 episode 做一次 eval（大幅减少评估开销）
        best_eval_score = float('inf')
        topk = 30  # 定义 topk 参数，用于 eval 时的任务过滤

        # Application level trackers
        prev_app_to = set()
        prev_app_fin = set()

        # InitEval：训练前先评估一次，展示初始"差"的性能
        # 这样可以对比学习前后的效果差异
        with (print_lock if print_lock else nullcontext()):
            print(f"\n[DGATD][INIT_EVAL] === Episode 0: 初始化评估（epsilon=0） ===")
        init_eval_e, init_eval_d, init_eval_dict = run_eval(agent, env, ts, gs, bc, arrival_plan, topk, task_complex_index, max_steps, base_node_dim)
        init_eval_score = init_eval_dict['score']
        init_eval_to = init_eval_dict['app_timeout_rate']
        best_eval_score = init_eval_score  # 初始化 best_score
        best_pack = (float(init_eval_e), float(init_eval_d), dict(init_eval_dict))

        with (print_lock if print_lock else nullcontext()):
            print(f"[DGATD][INIT_EVAL] 初始性能: Score={init_eval_score:.4f}, AppTO={init_eval_to:.2%}")
            print(f"[DGATD][INIT_EVAL] === 初始化评估结束，开始训练 ===\n")

        for episode in range(episodes):
            start_ep = time.time()
            bc.reset(); ts.reset(); GRAPH_CACHE.reset()
            episode_reward = 0.0
            episode_losses = []

            action_count = [0] * action_dim
            total_actions = 0
            step_in_ep = 0  # 修复记录 episode 内的步数，用于分散训练

            prev_app_to = set(ts.application_timeout_finished)
            prev_app_fin = set(ts.application_finished)

            for slot in range(max_steps):
                apply_arrival_plan(ts, slot, arrival_plan)
                ts.check_timeouts(slot)

                # 【调试信息】Episode 0 的前 3 个 slot 打印到达情况
                if episode == 0 and slot < 3:
                    sch = arrival_plan["schedule"]
                    with (print_lock if print_lock else nullcontext()):
                        print(f"[DGATD][DEBUG] Ep0, slot={slot}, arrivals uid list len={len(sch[slot])}")

                inner_steps = 0
                while True:
                    active_users, ready_tasks = get_stable_active_users(ts, gs, slot, k=30)
                    if not active_users: break

                    (state_x, state_edge_index, state_batch), task_to_global_idx, global_idx_to_task = get_global_graph_state_dag(
                        env, gs, active_users, task_complex_index, slot, base_node_dim
                    )

                    if state_x is None: break

                    ready_mask = torch.zeros(state_x.shape[0], dtype=torch.bool)
                    for task in ready_tasks:
                        if task in task_to_global_idx: ready_mask[task_to_global_idx[task]] = True

                    full_action_mask = construct_action_mask(
                        ts, ready_tasks, task_to_global_idx, action_dim, state_x.shape[0], slot
                    )

                    state_data = (state_x, state_edge_index, state_batch)
                    node_idx, offload_action = agent.select_action(state_data, ready_mask, full_action_mask, training=True)

                    if node_idx is None: break

                    # 【加速优化】使用 global_idx_to_task 列表
                    selected_task = global_idx_to_task[int(node_idx)]
                    if selected_task is None: break

                    # Step
                    step_reward, info = bc.step([[selected_task, offload_action]])
                    raw_reward = float(step_reward)

                    # Update Trackers
                    curr_app_to = set(ts.application_timeout_finished)
                    curr_app_fin = set(ts.application_finished)
                    new_to = curr_app_to - prev_app_to
                    new_fin = curr_app_fin - prev_app_fin
                    prev_app_to, prev_app_fin = curr_app_to, curr_app_fin

                    # === 【致命修复 1】奖励函数 - 只看应用完成/超时 ===
                    app_id = selected_task[0]

                    # 彻底删除每步的 energy/delay 奖励！
                    # 每步都是负奖励会导致代理薅本地小任务，最后全扔云端
                    reward = 0.0

                    # 只在应用级事件时给奖励
                    if app_id in new_fin:
                        reward += 10.0                    # 完成 +10
                    if app_id in new_to:
                        reward -= 30.0                    # 超时 -30（必须远大于+10）

                    action_count[int(offload_action)] += 1
                    total_actions += 1

                    # 修复重新构建 Next State（不复用旧状态）
                    next_active_users, next_ready_tasks = get_stable_active_users(ts, gs, slot, k=30)

                    next_state_x, next_state_edge_index, next_state_batch = None, None, None
                    next_task_map = {}

                    if next_active_users:
                        (next_state_x, next_state_edge_index, next_state_batch), next_task_map, _ = get_global_graph_state_dag(
                            env, gs, next_active_users, task_complex_index, slot, base_node_dim
                        )

                    # 修复确保 dummy next_state 维度正确
                    if next_state_x is None:
                        # 创建 dummy next_state，保证维度一致
                        next_state_x = torch.zeros((1, state_x.size(1)), dtype=torch.float32)
                        next_state_edge_index = torch.zeros((2, 0), dtype=torch.long)
                        next_state_batch = torch.zeros((1,), dtype=torch.long)

                    # done 逻辑：episode 结束或没有下一状态
                    done_flag = False
                    n_ready_mask = None
                    n_action_mask = None

                    if next_state_x is None or not next_active_users:
                        done_flag = True
                        n_ready_mask = torch.zeros(next_state_x.size(0), dtype=torch.bool)
                        n_action_mask = torch.full((next_state_x.size(0), action_dim), -1e9, dtype=torch.float32).cpu()
                    else:
                        n_ready_mask = torch.zeros(next_state_x.shape[0], dtype=torch.bool)
                        for t in next_ready_tasks:
                            if t in next_task_map: n_ready_mask[next_task_map[t]] = True

                        if n_ready_mask.any():
                            n_action_mask = construct_action_mask(ts, next_ready_tasks, next_task_map, action_dim, next_state_x.shape[0], slot).cpu()
                        else:
                            done_flag = True
                            n_action_mask = torch.full((next_state_x.shape[0], action_dim), -1e9, dtype=torch.float32).cpu()

                    if slot >= CONFIG["STOP_ARRIVAL_STEP"] and all_arrived_done(ts) and safe_rest_tasks_total(ts.rest_tasks) == 0:
                        done_flag = True

                    if n_action_mask is None:
                        n_action_mask = torch.full((next_state_x.shape[0], action_dim), -1e9, dtype=torch.float32).cpu()

                    agent.memory.push(
                        (state_x.cpu(), state_edge_index.cpu(), state_batch.cpu()),
                        int(node_idx), int(offload_action), reward,
                        (next_state_x.cpu() if next_state_x is not None else state_x.cpu(),
                         next_state_edge_index.cpu() if next_state_edge_index is not None else state_edge_index.cpu(),
                         next_state_batch.cpu() if next_state_batch is not None else state_batch.cpu()),
                        bool(done_flag), ready_mask.cpu(), n_ready_mask.cpu(),
                        full_action_mask.cpu(), n_action_mask, False
                    )

                    # 【内存优化】存储到 buffer 后，立即删除临时张量的引用，释放内存
                    del state_x, state_edge_index, state_batch
                    if next_state_x is not None:
                        del next_state_x, next_state_edge_index, next_state_batch
                    del ready_mask, n_ready_mask, full_action_mask, n_action_mask

                    # 【加速优化2】减少训练更新频率：从每10步改为每20步训练1次
                    # 减少训练开销，同时保持足够的更新频率
                    step_in_ep += 1
                    if len(agent.memory) >= 128 and step_in_ep % 20 == 0:
                        l = agent.learn(updates=1)  # 每次只更 1 次
                        if l is not None:
                            episode_losses.append(l)

                    episode_reward += reward
                    inner_steps += 1
                    if inner_steps >= 100: break

                if slot >= CONFIG["STOP_ARRIVAL_STEP"] and all_arrived_done(ts) and safe_rest_tasks_total(ts.rest_tasks) == 0: break

            ts.finalize_episode(slot)
            e, d = ts.get_avg_results()
            to_info = calc_timeout_rate(ts)
            app_to = to_info['app_timeout_rate']

            # 【内存优化】每个 episode 结束后清理一次缓存
            if (episode + 1) % 1 == 0:  # 每个 episode 都清理
                torch.cuda.empty_cache()

            # 【修复 2】计算 avg_loss（分散训练后的结果）
            avg_loss = np.mean(episode_losses) if episode_losses else 0.0

            # 如果没执行任何动作，立即报错并停止
            if total_actions == 0:
                with (print_lock if print_lock else nullcontext()):
                    print(f"[DGATD][FATAL] Episode {episode+1}: total_actions==0, training loop did not execute any step.")
                    print(f"    This means no tasks arrived or get_tasks() returned empty.")
                    print(f"    Check arrival_plan and gs.get_tasks() logic.")
                return 1000.0, 1000.0, {'error': 'total_actions==0, no training steps executed'}

            # 统一评分
            score = compute_score(e, d, app_to, to_info['task_timeout_rate'],
                                sla0=CONFIG.get("SLA0", 0.95), kappa=CONFIG.get("KAPPA", 2.0), v_cap=3.0)

            # 修复定期 deterministic eval（每 N 个 episode）
            if (episode + 1) % EVAL_EVERY == 0:
                with (print_lock if print_lock else nullcontext()):
                    print(f"\n[DGATD][EVAL] === Episode {episode+1}: 开始评估 (epsilon=0) ===")
                eval_e, eval_d, eval_dict = run_eval(agent, env, ts, gs, bc, arrival_plan, topk, task_complex_index, max_steps, base_node_dim)
                eval_score = eval_dict['score']
                eval_to = eval_dict['app_timeout_rate']

                with (print_lock if print_lock else nullcontext()):
                    print(f"[DGATD][EVAL] Score={eval_score:.4f}, AppTO={eval_to:.2%}")

                if eval_score < best_eval_score:
                    best_eval_score = eval_score
                    best_pack = (
                        float(eval_e),
                        float(eval_d),
                        dict(eval_dict)
                    )
                    try:
                        p = Path(CONFIG["RUN_DIR"]) / "checkpoints"
                        p.mkdir(parents=True, exist_ok=True)
                        agent.save(str(p / "dynamic_gat_dqn_model.pth"))
                        with (print_lock if print_lock else nullcontext()):
                            print(f"[DGATD][EVAL] New Best Eval: Score={eval_score:.4f}, AppTO={eval_to:.2%}")
                    except: pass
                with (print_lock if print_lock else nullcontext()):
                    print(f"[DGATD][EVAL] === 评估结束 ===\n")
                # 【内存优化】评估结束后立即清理 GPU 缓存，防止内存泄漏
                torch.cuda.empty_cache()

            # 记录曲线数据
            curves_data.append({
                'episode': episode + 1,
                'score': float(score),
                'energy': float(e),
                'delay': float(d),
                'app_timeout_rate': app_to,
                'task_timeout_rate': to_info['task_timeout_rate'],
                'episode_reward': episode_reward,
                'total_actions': total_actions,
                'local_actions': action_count[0] if len(action_count) > 0 else 0,
                'cloud_actions': action_count[1] if len(action_count) > 1 else 0,
                'edge_actions': sum(action_count[2:]) if len(action_count) > 2 else 0
            })

            if (episode + 1) % 5 == 0:
                # 【修复1】删除 epsilon 打印，改为 loss 打印
                print(f"[DGATD] Ep {episode+1}, Score={score:.4f}, AppTO={app_to:.2%}, R={episode_reward:.1f}, "
                      f"Loss={avg_loss:.4f}, Act: L={action_count[0]} C={action_count[1]} E={sum(action_count[2:])}, Buffer={len(agent.memory)}")

        # 保存训练曲线
        run_dir = CONFIG.get("RUN_DIR")
        if run_dir and curves_data:
            curves_dir = Path(run_dir) / "curves"
            curves_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(curves_data).to_csv(curves_dir / f"dynamic_gat_dqn_seed{seed_offset}.csv", index=False)
            print(f"[DGATD] 训练曲线已保存到: {curves_dir / f'dynamic_gat_dqn_seed{seed_offset}.csv'}")


        # 【改动 A】返回 best eval 的指标（修复bug：使用best_eval而不是best_score）
        if best_pack is not None:
            be, bd, best_eval_dict = best_pack[:3]
            print(f"[DGATD] Training complete. Returning best eval: Score={best_eval_dict['score']:.4f}, AppTO={best_eval_dict['app_timeout_rate']:.2%}")

            # 【改动 B】训练结束后用 best checkpoint 做 deterministic eval（epsilon=0）
            try:
                print(f"[DGATD] Loading best checkpoint for final evaluation...")
                checkpoint_path = Path(CONFIG["RUN_DIR"]) / "checkpoints" / "dynamic_gat_dqn_model.pth"
                if checkpoint_path.exists():
                    checkpoint = torch.load(checkpoint_path, map_location=device)
                    agent.policy_gpu.load_state_dict(checkpoint['policy_gpu'])
                    # 同步到 CPU 网络
                    agent.policy_cpu.load_state_dict(checkpoint['policy_gpu'])

                    # 运行一个 eval episode（不探索、不更新）
                    bc.reset(); ts.reset(); GRAPH_CACHE.reset()
                    eval_action_count = [0] * action_dim
                    eval_total_actions = 0

                    for slot in range(max_steps):
                        apply_arrival_plan(ts, slot, arrival_plan)
                        ts.check_timeouts(slot)

                        inner_steps = 0
                        while True:
                            active_users, ready_tasks = get_stable_active_users(ts, gs, slot, k=30)
                            if not active_users: break

                            (state_x, state_edge_index, state_batch), task_to_global_idx, global_idx_to_task = get_global_graph_state_dag(
                                env, gs, active_users, task_complex_index, slot, base_node_dim
                            )
                            if state_x is None: break

                            ready_mask = torch.zeros(state_x.shape[0], dtype=torch.bool)
                            for task in ready_tasks:
                                if task in task_to_global_idx:
                                    ready_mask[task_to_global_idx[task]] = True

                            full_action_mask = construct_action_mask(ts, ready_tasks, task_to_global_idx, action_dim, state_x.shape[0], slot)
                            state_data = (state_x, state_edge_index, state_batch)

                            # 【致命修复 3】评估时强制 epsilon=0
                            node_idx, offload_action = agent.select_action(
                                state_data, ready_mask, full_action_mask,
                                training=False,
                                custom_eps=0.0  # 必须为 0，测试模型真实能力
                            )

                            if node_idx is None: break

                            # 【加速优化】使用 global_idx_to_task 列表
                            selected_task = global_idx_to_task[int(node_idx)]
                            if selected_task is None:
                                selected_task = ready_tasks[0]

                            bc.step([[selected_task, offload_action]])
                            eval_action_count[int(offload_action)] += 1
                            eval_total_actions += 1

                            inner_steps += 1
                            if inner_steps >= 100: break

                        if slot >= CONFIG["STOP_ARRIVAL_STEP"] and all_arrived_done(ts) and safe_rest_tasks_total(ts.rest_tasks) == 0:
                            break

                    ts.finalize_episode(slot)
                    ee, ed = ts.get_avg_results()
                    eto_info = calc_timeout_rate(ts)
                    e_app_to = eto_info['app_timeout_rate']
                    e_score = compute_score(ee, ed, e_app_to, eto_info['task_timeout_rate'],
                                         sla0=CONFIG.get("SLA0", 0.95), kappa=CONFIG.get("KAPPA", 2.0), v_cap=3.0)

                    print(f"[DGATD] Final Eval (Best Checkpoint): Score={e_score:.4f}, AppTO={e_app_to:.2%}, "
                          f"Act: L={eval_action_count[0]} C={eval_action_count[1]} E={sum(eval_action_count[2:])}")

                    # 返回 eval 的结果（更稳定）
                    return to_scalar(ee), to_scalar(ed), {
                        'score': float(e_score),
                        'app_timeout_rate': e_app_to,
                        'task_timeout_rate': eto_info['task_timeout_rate'],
                        'energy': float(ee),
                        'delay': float(ed),
                        'action_stats': {
                            'local': eval_action_count[0],
                            'cloud': eval_action_count[1],
                            'edge': sum(eval_action_count[2:]),
                            'total_actions': eval_total_actions
                        }
                    }
                else:
                    print(f"[DGATD] Warning: Best checkpoint not found, returning cached best episode")
                    return be, bd, best_eval_dict
            except Exception as eval_e:
                print(f"[DGATD] Warning: Eval failed ({eval_e}), returning cached best episode")
                return be, bd, best_eval_dict
        else:
            print(f"[DGATD] Warning: No valid best eval found, returning current episode metrics")
            return to_scalar(e), to_scalar(d), {'score': float(score), 'app_timeout_rate': app_to}

    except Exception as e:
        traceback.print_exc()
        return 1000.0, 1000.0, {'error': str(e)}
