# -*- coding: utf-8 -*-
"""
Heuristic baselines: Local-only, Cloud-only, Edge-only, Greedy, Genetic, HybridPSOGA.

English
-------
Benchmark.py implements the five non-RL baselines compared in Table IV:
  - Local-only / Cloud-only / Edge-only: route every subtask to one tier.
  - Greedy: pick the tier with the lowest estimated finish time for each
    subtask (with a configurable estimation error to model real-world
    prediction noise).
  - Genetic: a genetic-algorithm search over per-subtask placements.
  - HybridPSOGA: hybrid Particle-Swarm + Genetic-Algorithm search.
All baselines share the same env, arrival plan and deadline config as the
RL agents (fair-comparison protocol), so they are driven through
Algorithms/Train/common.run_benchmark_worker. They have no checkpoint;
reeval_only re-runs them on the fly (~30 s per cell) and their D_all equals
D_succ (heuristic timeout accounting matches the RL one).

中文
----
5 个启发式基线: Local/Cloud/Edge-only, Greedy, Genetic, HybridPSOGA。与 RL 共享
同一份 env/arrival plan/deadline, 无 ckpt, reeval 时现跑, D_all=D_succ。
"""
import random
import numpy as np
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 添加 Experiments_new 到路径
experiments_dir = os.path.join(project_root, 'Experiments_new')
if experiments_dir not in sys.path:
    sys.path.insert(0, experiments_dir)

from Environment import computation
from utils.constant import para
from scheduler.task_selector import TaskSelector

# 尝试导入CONFIG（如果是在训练环境中运行）
try:
    from Experiments_new.exp_utils import CONFIG
    _HAS_CONFIG = True
except:
    CONFIG = {"TOPK_TASKS": 30}  # 默认值
    _HAS_CONFIG = False

# 辅助函数：安全转换为 float
def _as_float(val, default=0.0):
    try:
        if isinstance(val, (list, tuple, np.ndarray)):
            return float(val[0]) if len(val) > 0 else default
        return float(val)
    except:
        return default


class Benchmark:
    def __init__(self, env, gs, ts, complex_index, effective=False, seed=None, task_order_mode=None):
        self.env = env
        self.gs = gs
        self.ts = ts
        self.complex_index = complex_index
        self.effective = effective
        self.seed = seed
        # 用于 Genetic 和 Greedy 噪声的随机数生成器
        self.rng = random.Random(seed) if seed is not None else random

        # 启发式基线的现实建模参数：预测误差率与可见节点数
        self.ESTIMATION_ERROR = 0.2  # 20% 的预测误差，模拟真实世界的预测不准
        self.VISIBLE_K = 3  # 启发式只能感知最近的 3 个 Edge，模拟服务发现开销

        # 任务顺序选择器
        # task_order_mode: "cp"(关键路径), "cp_rev"(关键路径反转), "random"(随机), "fifo"(先来先), "slack"(松弛时间), "none"(保持原序)
        self.task_order_mode = task_order_mode or "none"
        self.task_selector = TaskSelector(mode=self.task_order_mode, seed=seed or 0)

    def reset(self):
        pass

    # =========================================================
    # 带噪声的观测函数：模拟真实场景下的估算不准
    # =========================================================
    def _get_noisy_value(self, true_value):
        """
        为真实值添加噪声，模拟预测误差

        Args:
            true_value: 真实值（能耗或时延）

        Returns:
            带 [80%, 120%] 噪声的估算值
        """
        if true_value == 0:
            return 0
        # 生成一个 [0.8, 1.2] 之间的系数
        noise = 1.0 + self.rng.uniform(-self.ESTIMATION_ERROR, self.ESTIMATION_ERROR)
        return true_value * noise

    # =========================================================
    # 通用辅助函数：安全获取任务大小 (Bytes)
    # =========================================================
    def _get_safe_task_size(self, user_id, subtask_id):
        """
        安全地从 TaskScheduler 获取任务大小，兼容各种数据结构
        """
        s = self.ts.task_size
        try:
            # 优先尝试 TaskScheduler 可能提供的接口
            if hasattr(self.ts, 'get_task_size_bytes'):
                return self.ts.get_task_size_bytes(user_id, subtask_id)

            # 情况1: 二维数组/列表 [user_id][subtask_id]
            if isinstance(s, (list, tuple, np.ndarray)) and len(s) > user_id:
                user_data = s[user_id]
                if isinstance(user_data, (list, tuple, np.ndarray)) and len(user_data) > subtask_id:
                    return float(user_data[subtask_id])
            
            # 情况2: 字典
            if isinstance(s, dict):
                # 尝试 (user, subtask) 组合键
                if (user_id, subtask_id) in s: return float(s[(user_id, subtask_id)])
                # 尝试 subtask_id
                if subtask_id in s: return float(s[subtask_id])
                if str(subtask_id) in s: return float(s[str(subtask_id)])
                return float(list(s.values())[0])

            # 情况3: 一维数组
            if isinstance(s, (list, tuple, np.ndarray)) and len(s) > subtask_id:
                return float(s[subtask_id])

            return 200000.0 # 默认值
        except:
            return 200000.0

    def get_actions(self, type, slot, preslot):
        """
        type: 0:Local, 1:Cloud, 2:RandomEdge, 4:Greedy, 6:GeneticFair(现实建模版), 8:HybridPSOGA
        【并行调度】每个slot可以调度多个任务
        【Greedy / GeneticFair】采用带噪声的现实建模（预测误差 + 视野受限），模拟真实部署条件下的启发式
        【任务顺序】使用 task_selector 对任务进行排序（支持 random, cp, fifo, slack, none）
        """
        tasks = self.gs.get_tasks(slot, sort_tasks=False)
        if not tasks:
            return [], preslot

        # 对任务进行排序（使用 TaskSelector）
        tasks = self.task_selector.order(self.gs, slot, tasks)

        # 【并行调度】对所有 ready tasks 进行调度
        # 可以选择对所有任务调度，或者只对 top-k 任务调度
        k = CONFIG.get("TOPK_TASKS", 30)
        if hasattr(self.gs, "pick_topk_tasks"):
            try:
                tasks = self.gs.pick_topk_tasks(slot, tasks, k=min(k, len(tasks)))
            except Exception:
                pass  # 失败时使用所有 tasks

        # 0. Local Only
        if type == 0:
            return [[task, 0] for task in tasks], preslot

        # 1. Cloud Only
        elif type == 1:
            return [[task, 1] for task in tasks], preslot

        # 2. Random Edge
        elif type == 2:
            edge_num = para["edge_num"]
            return_list = []
            for task in tasks:
                # 随机选一个 Edge (2 ~ 2+edge_num-1)
                edge_id = self.rng.randint(0, edge_num - 1)
                return_list.append([task, edge_id + 2])
            return return_list, preslot

        # ==========================================
        # 4. Greedy (Task-by-Task) - 带现实约束的启发式
        # 约束1: 视野限制 (只看最近的 K 个节点，模拟服务发现开销)
        # 约束2: 估算误差 (任务大小和带宽带噪声，模拟预测不准)
        # ==========================================
        elif type == 4:
            return_list = []

            for task in tasks:
                user_id, subtask_id = task
                # 任务大小带噪声（模拟预测不准）
                real_task_size = self._get_safe_task_size(user_id, subtask_id)
                est_task_size = self._get_noisy_value(real_task_size)

                # 1. Local Time（使用估算大小）
                f_local = _as_float(self.env.device_list[user_id].local_power, 1e9)
                _, t_local = computation.execute_consumption(est_task_size, f_local, self.complex_index, "l")

                best_action = 0
                min_time = t_local

                # 2. Cloud Time（加入带宽波动）
                # 真实场景下无法知道精确的当前带宽，用历史均值估算 + 噪声
                est_bw_cloud = self._get_noisy_value(np.mean(para["uplink_range"]))
                nearest_edge_idx = int(np.argmin(self.env.device_list[user_id].edge_distances))
                dist = float(self.env.device_list[user_id].edge_distances[nearest_edge_idx])

                _, t_up1 = computation.upload_consumption([est_task_size, dist, est_bw_cloud], 1, "e")
                _, t_up2 = computation.upload_consumption(est_task_size, 1, "c")
                fc = _as_float(self.env.cloud.cloud_power, 6e9)
                _, t_exe_c = computation.execute_consumption(est_task_size, fc, self.complex_index, "c")

                t_cloud = t_up1 + t_up2 + t_exe_c

                if t_cloud < min_time:
                    min_time = t_cloud
                    best_action = 1

                # 3. Edge Time（受限视野）
                # 只允许探测最近的 K 个边缘节点（模拟服务发现开销）
                dists = self.env.device_list[user_id].edge_distances
                # 按距离排序，只取前 VISIBLE_K 个
                visible_edges = list(np.argsort(dists))[:self.VISIBLE_K]

                for e_id in visible_edges:
                    # 同样使用带噪声的带宽和任务大小
                    est_bw_edge = self._get_noisy_value(np.mean(para["uplink_range"]))
                    f_edge = self.env.edges[e_id].edge_power * self.env.edges[e_id].calculate_parameter
                    dist_e = float(dists[e_id])

                    _, t_up_e = computation.upload_consumption([est_task_size, dist_e, est_bw_edge], 1, "e")
                    _, t_exe_e = computation.execute_consumption(est_task_size, f_edge, self.complex_index, "e")

                    # Greedy 难以准确预知排队时间，这里忽略排队（模拟预测不准）
                    t_edge = t_up_e + t_exe_e

                    if t_edge < min_time:
                        min_time = t_edge
                        best_action = e_id + 2

                return_list.append([task, best_action])

            return return_list, preslot

        elif type == 6:
            now = slot * float(para.get("slot_interval", 0.01))

            # 计算"公平"的全局负载摘要（只汇总，不给每个 edge 单独排队信息）
            def _avg_wait(times):
                if not times:
                    return 0.0
                ws = []
                for t in times:
                    try:
                        ws.append(max(0.0, float(t) - now))
                    except Exception:
                        ws.append(0.0)
                return float(sum(ws) / max(1, len(ws)))

            avg_local_wait = _avg_wait(getattr(self.ts, "devices_exe_useful", []))
            avg_upload_wait = _avg_wait(getattr(self.ts, "devices_upload_useful", []))

            # avg_edge_wait：不区分 edge，只做一个全局平均等待
            avg_edge_wait = 0.0
            edge_useful = getattr(self.ts, "edge_useful", None)
            if isinstance(edge_useful, (list, tuple)) and len(edge_useful) > 0:
                per_edge_best = []
                for edge_cores in edge_useful:
                    try:
                        core_times = [max(0.0, float(t) - now) for t in edge_cores]
                        per_edge_best.append(min(core_times) if core_times else 0.0)
                    except Exception:
                        per_edge_best.append(0.0)
                if per_edge_best:
                    avg_edge_wait = float(sum(per_edge_best) / len(per_edge_best))

            edge_num = int(para["edge_num"])
            bw_choices = para.get("uplink_range", [para["uplink_range"][0]])

            rng = self.rng  # 用 Benchmark 初始化的 rng，保证可复现

            # GA 参数
            action_max = 1 + edge_num  # 完整动作空间: Local=0, Cloud=1, Edge=2..edge_num+1
            population_size = 4
            generations = 1
            mutation_rate = 0.2
            crossover_rate = 0.8

            def _app_deadline_abs(uid: int) -> float:
                slot_interval = float(para.get("slot_interval", 0.01))
                default_slots = int(para.get("deadline_slot", 180))
                enter = getattr(self.ts, "enter_time", None)
                enter_t = float(enter[uid]) if enter is not None and uid < len(enter) and enter[uid] != float("inf") else now
                if hasattr(self.ts, "get_app_deadline_slot"):
                    try:
                        dslots = int(self.ts.get_app_deadline_slot(uid))
                    except Exception:
                        dslots = default_slots
                else:
                    dslots = default_slots
                return enter_t + dslots * slot_interval

            def _estimate_energy_delay(uid: int, task_size_bytes: float, action: int):
                """
                返回 (energy, delay) 的"粗估"。
                action 2..edge_num+1 直接映射 edge_id
                """
                # Local
                if action == 0:
                    f_local = _as_float(self.env.device_list[uid].local_power, 1e9)
                    e_ex, t_ex = computation.execute_consumption(task_size_bytes, f_local, self.complex_index, "l")
                    return float(e_ex), float(avg_local_wait + t_ex)

                # Cloud
                if action == 1:
                    fc = _as_float(self.env.cloud.cloud_power, 6e9)
                    nearest_edge = int(np.argmin(self.env.device_list[uid].edge_distances))
                    dist = float(self.env.device_list[uid].edge_distances[nearest_edge])
                    bw = float(rng.choice(bw_choices))

                    e1, d1 = computation.upload_consumption([task_size_bytes, dist, bw], 1, "e")
                    e2, d2 = computation.upload_consumption(task_size_bytes, 1, "c")
                    e3, d3 = computation.execute_consumption(task_size_bytes, fc, self.complex_index, "c")

                    return float(e1 + e2 + e3), float(avg_upload_wait + d1 + d2 + d3)

                # Edge
                # action 2..edge_num+1 直接映射 edge_id
                edge_id = int(action - 2)

                edge_id = max(0, min(edge_id, edge_num - 1))
                f_edge = self.env.edges[edge_id].edge_power * self.env.edges[edge_id].calculate_parameter
                dist_e = float(self.env.device_list[uid].edge_distances[edge_id])
                bw = float(rng.choice(bw_choices))

                e_up, d_up = computation.upload_consumption([task_size_bytes, dist_e, bw], 1, "e")
                e_ex, d_ex = computation.execute_consumption(task_size_bytes, f_edge, self.complex_index, "e")

                return float(e_up + e_ex), float(avg_upload_wait + avg_edge_wait + d_up + d_ex)

            def _late_penalty(uid: int, est_delay: float) -> float:
                deadline_abs = _app_deadline_abs(uid)
                remaining = float(deadline_abs - now)
                if remaining <= 0.0:
                    return 20.0
                overflow = max(0.0, est_delay - remaining)
                return 10.0 * overflow

            def _heuristic_seed_action(uid: int, subtask_id: int) -> int:
                task_size = float(self._get_safe_task_size(uid, subtask_id))
                local_e, local_d = _estimate_energy_delay(uid, task_size, 0)
                cloud_e, cloud_d = _estimate_energy_delay(uid, task_size, 1)
                nearest_eid = int(np.argmin(self.env.device_list[uid].edge_distances))
                edge_action = 2 + nearest_eid
                edge_e, edge_d = _estimate_energy_delay(uid, task_size, edge_action)

                # 如果 local delay 可接受（小于 deadline 剩余一半），优先 local
                deadline_abs = _app_deadline_abs(uid)
                remaining = float(deadline_abs - now)
                if remaining > 0 and local_d < remaining * 0.5:
                    return 0  # local

                # 否则：delay 最短的卸载选项（cloud vs nearest_edge）
                if cloud_d <= edge_d:
                    return 1  # cloud
                else:
                    return edge_action

            # 按 app 分组任务
            tasks_by_app = {}
            for task in tasks:
                user_id, subtask_id = task
                app_id = user_id
                if app_id not in tasks_by_app:
                    tasks_by_app[app_id] = []
                tasks_by_app[app_id].append(task)

            # 对每个 app 做 GA，输出动作
            task_best_actions = {}

            for uid, app_tasks in tasks_by_app.items():
                if not app_tasks:
                    continue
                n = len(app_tasks)

                # ============ GeneticFair 配置 ============

                def fitness(individual):
                    total_energy = 0.0
                    total_delay = 0.0
                    for i, task in enumerate(app_tasks):
                        user_id, subtask_id = task
                        action = individual[i]
                        real_task_size = float(self._get_safe_task_size(user_id, subtask_id))
                        noisy_task_size = self._get_noisy_value(real_task_size)
                        e, d = _estimate_energy_delay(uid, noisy_task_size, action)
                        total_energy += e
                        total_delay += d
                    penalty = _late_penalty(uid, total_delay)
                    return 0.5 * total_energy + 0.5 * total_delay + penalty

                def selection(pop, fits):
                    selected = []
                    for _ in range(len(pop)):
                        i1, i2 = rng.sample(range(len(pop)), 2)
                        selected.append(pop[i1][:] if fits[i1] < fits[i2] else pop[i2][:])
                    return selected

                def crossover(p1, p2):
                    if rng.random() > crossover_rate or len(p1) < 2:
                        return p1[:], p2[:]
                    pt = rng.randint(1, len(p1) - 1)
                    return p1[:pt] + p2[pt:], p2[:pt] + p1[pt:]

                def mutation(ind):
                    if rng.random() < mutation_rate:
                        idx = rng.randint(0, len(ind) - 1)
                        ind[idx] = rng.randint(0, action_max)
                    return ind

                population = []

                seed = [_heuristic_seed_action(t[0], t[1]) for t in app_tasks]
                population.append(seed)

                # 随机种群
                while len(population) < population_size:
                    ind = [rng.randint(0, action_max) for _ in range(n)]
                    population.append(ind)

                # 进化
                for _gen in range(generations):
                    fits = [fitness(ind) for ind in population]
                    parents = selection(population, fits)
                    next_pop = []
                    while len(next_pop) < population_size:
                        p1, p2 = rng.sample(parents, 2) if len(parents) >= 2 else (parents[0], parents[0])
                        c1, c2 = crossover(p1, p2)
                        next_pop.append(mutation(c1))
                        if len(next_pop) < population_size:
                            next_pop.append(mutation(c2))
                    population = next_pop

                # 选择最佳个体
                final_fits = [fitness(ind) for ind in population]
                best_idx = int(np.argmin(final_fits))
                best_ind = population[best_idx]

                for i, task in enumerate(app_tasks):
                    task_best_actions[task] = int(best_ind[i])

            # 生成返回动作列表
            return_list = []
            for task in tasks:
                if task in task_best_actions:
                    return_list.append([task, task_best_actions[task]])
                else:
                    return_list.append([task, 0])

            return return_list, preslot

        # ==========================================
        # 8. Hybrid PSO-GA (排队感知模糊化 + 噪声模型 + 视野限制)
        # 针对 500 用户高压力环境优化
        # ==========================================
        elif type == 8:
            now = slot * float(para.get("slot_interval", 0.01))

            # 按 app 分组任务（PSO-GA 擅长处理多任务组合优化）
            tasks_by_app = {}
            for task in tasks:
                uid, sid = task
                if uid not in tasks_by_app: tasks_by_app[uid] = []
                tasks_by_app[uid].append(task)

            task_best_actions = {}

            # PSO-GA 核心参数
            # PSO-GA 核心参数
            pop_size = 5
            max_iter = 2
            w = 0.7            # 惯性权重
            c1, c2 = 1.2, 1.2  # 学习因子
            mutation_rate = 0.1
            edge_num = int(para["edge_num"])
            action_max = 1 + edge_num
            bw_avg = np.mean(para["uplink_range"])
            bw_choices = para.get("uplink_range", [bw_avg])

            # 排队信息模糊化：使用全局平均等待时间，而非精确的每个节点排队信息
            # 与 GeneticFair 采用同样的现实建模策略
            def _avg_wait(times):
                if not times:
                    return 0.0
                ws = []
                for t in times:
                    try:
                        ws.append(max(0.0, float(t) - now))
                    except Exception:
                        ws.append(0.0)
                return float(sum(ws) / max(1, len(ws)))

            avg_local_wait = _avg_wait(getattr(self.ts, "devices_exe_useful", []))
            avg_upload_wait = _avg_wait(getattr(self.ts, "devices_upload_useful", []))

            # edge 等待信息：wide 高并行下需要区分每个 edge 的具体等待
            # 保留 avg_edge_wait 兼容旧代码引用，新增 per_edge_wait[i] 给 fitness 按 edge 索引
            per_edge_wait = []  # len == edge_num，第 i 项 = edge i 最早空闲 core 的等待
            avg_edge_wait = 0.0
            edge_useful = getattr(self.ts, "edge_useful", None)
            if isinstance(edge_useful, (list, tuple)) and len(edge_useful) > 0:
                for edge_cores in edge_useful:
                    try:
                        core_times = [max(0.0, float(t) - now) for t in edge_cores]
                        per_edge_wait.append(min(core_times) if core_times else 0.0)
                    except Exception:
                        per_edge_wait.append(0.0)
                if per_edge_wait:
                    avg_edge_wait = float(sum(per_edge_wait) / len(per_edge_wait))

            for uid, app_tasks in tasks_by_app.items():
                if not app_tasks: continue
                n = len(app_tasks)
                deadline_abs = self.ts.enter_time[uid] + self.ts.get_app_deadline_slot(uid) * para["slot_interval"]

                # 预估函数：使用模糊化的排队信息 + 噪声模型
                def _est_cost(action, noisy_size, task_idx):
                    # 使用全局平均等待时间（而非精确的每个节点排队）
                    q_local = avg_local_wait
                    q_up = avg_upload_wait

                    if action == 0: # Local
                        f_local = _as_float(self.env.device_list[uid].local_power, 1e9)
                        e, d = computation.execute_consumption(noisy_size, f_local, self.complex_index, "l")
                        return e, q_local + d
                    elif action == 1: # Cloud
                        fc = _as_float(self.env.cloud.cloud_power, 6e9)
                        dist = float(self.env.device_list[uid].edge_distances[int(np.argmin(self.env.device_list[uid].edge_distances))])
                        # 带宽带噪声
                        bw = self._get_noisy_value(bw_avg)
                        e1, d1 = computation.upload_consumption([noisy_size, dist, bw], 1, "e")
                        e2, d2 = computation.upload_consumption(noisy_size, 1, "c")
                        e3, d3 = computation.execute_consumption(noisy_size, fc, self.complex_index, "c")
                        return (e1+e2+e3), (q_up + d1 + d2 + d3)
                    else: # Edge
                        eid = int(action - 2)
                        f_e = self.env.edges[eid].edge_power * self.env.edges[eid].calculate_parameter
                        dist = float(self.env.device_list[uid].edge_distances[eid])
                        # 用每个 edge 的具体等待（非模糊化），wide 多并发能让 fitness 区分 edge 拥塞
                        if 0 <= eid < len(per_edge_wait):
                            q_edge = per_edge_wait[eid]
                        else:
                            q_edge = avg_edge_wait
                        # 带宽带噪声
                        bw = self._get_noisy_value(bw_avg)
                        e_up, d_up = computation.upload_consumption([noisy_size, dist, bw], 1, "e")
                        e_ex, d_ex = computation.execute_consumption(noisy_size, f_e, self.complex_index, "e")
                        return (e_up + e_ex), (q_up + d_up + q_edge + d_ex)

                def fitness(particle_pos):
                    total_e, max_d, total_d = 0, 0, 0
                    edge_loads = {}  # eid -> 该 edge 被分配到的任务数（鼓励分散）
                    for i, act_raw in enumerate(particle_pos):
                        # 视野限制：只考虑最近的 VISIBLE_K 个 Edge 节点
                        act = max(0, min(action_max, int(round(act_raw))))
                        if act >= 2:  # Edge 动作，需要检查视野限制
                            eid = act - 2
                            # 获取所有 Edge 的距离
                            dists = self.env.device_list[uid].edge_distances
                            # 按距离排序，只取前 VISIBLE_K 个可见的 edge_id
                            visible_edges = list(np.argsort(dists))[:self.VISIBLE_K]
                            # 如果选中的 edge_id 不在可见范围内，强制改为最近的可见 edge
                            if eid not in visible_edges:
                                act = visible_edges[0] + 2  # 使用最近的 edge
                                eid = act - 2
                        else:
                            eid = -1

                        real_s = self._get_safe_task_size(app_tasks[i][0], app_tasks[i][1])
                        # 任务大小带噪声
                        noisy_s = self._get_noisy_value(real_s)
                        e, d = _est_cost(act, noisy_s, i)
                        total_e += e
                        total_d += d
                        max_d = max(max_d, d) # App 完成时间由最慢的任务决定
                        if eid >= 0:
                            edge_loads[eid] = edge_loads.get(eid, 0) + 1

                    # 负载均衡惩罚：wide 高并行下避免任务全堆同一 edge（提升 AppTO）
                    # load_var = 任务到 edge 分配数的方差；越大说明越集中
                    if len(edge_loads) > 1:
                        loads = list(edge_loads.values())
                        mean_load = sum(loads) / len(edge_loads)
                        load_var = sum((x - mean_load) ** 2 for x in loads) / len(edge_loads)
                    else:
                        load_var = 0.0

                    # 超时惩罚
                    overflow = max(0.0, (now + max_d) - deadline_abs)
                    penalty = 2.5 * overflow + (2.5 if overflow > 0 else 0.0)

                    # 综合 fitness（保持 A: 删除 0.125*total_d 和 0.04*load_var）
                    return 0.5 * total_e + 0.5 * max_d + penalty

                # 初始化粒子：多种启发式种子 + 随机补充，降低方差
                particles = []
                dists = self.env.device_list[uid].edge_distances
                visible_edges = list(np.argsort(dists))[:self.VISIBLE_K]
                visible_actions = [0, 1] + [e + 2 for e in visible_edges]

                # 启发式种子 1: 分散轮询（按 edge 等待时间升序）
                if n >= 2 and len(visible_edges) >= 2:
                    sorted_eids = sorted(visible_edges, key=lambda x: per_edge_wait[x] if 0 <= x < len(per_edge_wait) else 0.0)
                    seed_dispersion = [float(2 + sorted_eids[i % len(sorted_eids)]) for i in range(n)]
                else:
                    seed_dispersion = [float(self.rng.choice(visible_actions)) for _ in range(n)]

                # 启发式种子 2: 最近 edge 优先（所有任务分配到最近 edge）
                if visible_edges:
                    nearest_eid = visible_edges[0]
                    seed_nearest = [float(2 + nearest_eid) for _ in range(n)]
                else:
                    seed_nearest = [float(self.rng.choice(visible_actions)) for _ in range(n)]

                # 启发式种子 3: 本地优先（尽量本地执行）
                seed_local = [0.0 for _ in range(n)]

                for _p_idx in range(pop_size):
                    if _p_idx == 0:
                        pos = seed_dispersion
                    elif _p_idx == 1:
                        pos = seed_nearest
                    elif _p_idx == 2:
                        pos = seed_local
                    else:
                        pos = [float(self.rng.choice(visible_actions)) for _ in range(n)]
                    particles.append({
                        'pos': [float(act) for act in pos],
                        'vel': [self.rng.uniform(-1, 1) for _ in range(n)],
                        'best_pos': pos[:],
                        'best_fit': float('inf')
                    })

                g_best_pos = particles[0]['pos'][:]
                g_best_fit = float('inf')

                # PSO 迭代
                for _iter in range(max_iter):
                    for p in particles:
                        fit = fitness(p['pos'])
                        if fit < p['best_fit']:
                            p['best_fit'] = fit
                            p['best_pos'] = p['pos'][:]
                        if fit < g_best_fit:
                            g_best_fit = fit
                            g_best_pos = p['pos'][:]

                    for p in particles:
                        for i in range(n):
                            r1, r2 = self.rng.random(), self.rng.random()
                            p['vel'][i] = w * p['vel'][i] + c1*r1*(p['best_pos'][i]-p['pos'][i]) + c2*r2*(g_best_pos[i]-p['pos'][i])
                            # 速度更新后也要限制在可见范围内
                            new_pos_raw = p['pos'][i] + p['vel'][i]
                            new_pos = max(0, min(action_max, new_pos_raw))
                            p['pos'][i] = new_pos

                    # 混合 GA：小概率变异
                    if self.rng.random() < mutation_rate:
                        idx = self.rng.randint(0, pop_size-1)
                        gene_idx = self.rng.randint(0, n-1)
                        # 变异也在可见范围内
                        dists = self.env.device_list[uid].edge_distances
                        visible_edges = list(np.argsort(dists))[:self.VISIBLE_K]
                        visible_actions = [0, 1] + [e + 2 for e in visible_edges]
                        particles[idx]['pos'][gene_idx] = float(self.rng.choice(visible_actions))

                # 记录结果
                for i, task in enumerate(app_tasks):
                    # 确保最终动作也在可见范围内
                    act_raw = int(round(g_best_pos[i]))
                    if act_raw >= 2:
                        eid = act_raw - 2
                        dists = self.env.device_list[uid].edge_distances
                        visible_edges = list(np.argsort(dists))[:self.VISIBLE_K]
                        if eid not in visible_edges:
                            act_raw = visible_edges[0] + 2
                    task_best_actions[task] = act_raw

            return [[t, task_best_actions.get(t, 0)] for t in tasks], preslot

        else:
            return [], preslot

    def step(self, actions):
        """
        执行动作并计算即时 reward。
        直接使用 TaskScheduler 返回的真实能耗和延迟（包含排队和依赖）。

        Args:
            actions: List[[[uid, sid], action_int]]

        Returns:
            (reward, info): reward 是负成本
        """
        total_energy = 0.0
        total_delay = 0.0

        for task, action in actions:
            user_id, subtask_id = task

            # 执行动作到 TaskScheduler（推进状态），并获取真实结果
            # 兼容不同接口名称
            if hasattr(self.ts, 'offload_task'):
                try:
                    energy, delay = self.ts.offload_task(user_id, subtask_id, action)
                except TypeError:
                    energy, delay = self.ts.offload_task(subtask_id, user_id, action)
            elif hasattr(self.ts, 'add_task'):
                energy, delay = self.ts.add_task(user_id, subtask_id, action)
            else:
                raise AttributeError("TaskScheduler has neither offload_task nor add_task")

            # 累加真实的能耗和延迟
            total_energy += float(energy)
            total_delay += float(delay)

        # 计算 reward：负成本
        # 权重可以根据需要调整，这里用 0.5 能耗 + 0.5 延迟
        # 注意：这里的 delay 包含了排队时间，如果排队严重，reward 会显著降低（负得更多）
        reward = -(0.5 * total_energy + 0.5 * total_delay)

        info = {
            "step_energy": total_energy,
            "step_delay": total_delay
        }

        return reward, info
