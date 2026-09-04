# -*- coding: utf-8 -*-
"""
GraphScheduler: DAG-level scheduler that drives subtask selection.

English
-------
GraphScheduler sits above TaskScheduler and decides WHICH subtask to act on
next (the RL agent then decides WHERE to place it). It manages the per-user
DAG (SubGraph), computes ready subtasks (all parents finished), applies the
top-K task selection (TaskSelector) that limits the RL field of view, and
packs the selected subtask's neighbourhood into a PyG `Data` object for the
GAT encoder. It also holds the deadline config and the per-app enter/finish
times used by the slack reward and the R1-5 timeout accounting.

中文
----
DAG 级调度器: 决定下一个动作的子任务 (RL 再决定放在哪), 管理 per-user DAG,
计算就绪子任务, 做 top-K 视野限制, 把邻域打包成 PyG Data 给 GAT 编码器。
"""
import networkx as nx
from Environment import computation
from Environment.Graph import BaseGraph
from Environment.environment import Environment
from utils.constant import para
from collections import defaultdict
from torch_geometric.data import Data
import numpy as np
import torch


class GraphScheduler:
    def __init__(self, basegraph: BaseGraph, subgraph_list, task_scheduler, device=None):
        self.basegraph = basegraph
        self.subgraph_list = subgraph_list
        self.ts = task_scheduler
        # 缓存 BFS 顺序，避免重复计算
        self.bfs_order = self.get_bfs_order()

        # 【关键路径改进】缓存 rank_u 值（HEFT upward-rank）
        # 格式: {user_id: {node_id: rank_u_value}}
        self.rank_u = {}

        # 预计算一些常数
        self.num_edges = para["edge_num"]
        self.num_users = para["user_num"]
        # 修改设备选择逻辑：如果没传 device，才去自动检测，否则听从指挥
        if device is not None:
            self.device = device
        else:
            self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        # 缓存静态的边索引 (假设图结构在 Episode 中不变)
        self.static_edge_index = self._build_static_edge_index()

    def get_bfs_order(self):
        bfs_tree = nx.DiGraph()
        for source in self.basegraph.enter_nodes:
            tree = nx.bfs_tree(self.basegraph.nx_graph, source=source)
            bfs_tree.add_edges_from(tree.edges())
        return list(bfs_tree.nodes())

    def _estimate_w(self, env, uid, node_id, task_complex_index):
        """
        【关键路径】估算节点 v 的执行代价 w(v)
        取本地、边缘最优、云端的最小执行时间（不考虑排队）

        Args:
            env: 环境对象
            uid: 用户ID
            node_id: 节点ID
            task_complex_index: 任务复杂度索引

        Returns:
            最小估计执行时间（秒）
        """
        bytes_ = float(self.ts.task_size[node_id])

        # 本地执行
        f_local = float(env.device_list[uid].local_power)
        _, t_local = computation.execute_consumption(bytes_, f_local, task_complex_index, "l")

        # 边缘执行（找最优边缘节点）
        best_edge_t = 1e9
        for eid in range(para["edge_num"]):
            f_edge = env.edges[eid].edge_power * env.edges[eid].calculate_parameter
            dist = float(env.device_list[uid].edge_distances[eid])
            bw = min(para["uplink_range"])  # 保守值，不使用随机
            _, t_up = computation.upload_consumption([bytes_, dist, bw], 1, "e")
            _, t_ex = computation.execute_consumption(bytes_, f_edge, task_complex_index, "e")
            best_edge_t = min(best_edge_t, t_up + t_ex)

        # 云端执行
        fc = float(env.cloud.cloud_power)
        nearest = int(np.argmin(env.device_list[uid].edge_distances))
        dist = float(env.device_list[uid].edge_distances[nearest])
        bw = min(para["uplink_range"])
        _, t1 = computation.upload_consumption([bytes_, dist, bw], 1, "e")
        _, t2 = computation.upload_consumption(bytes_, 1, "c")
        _, t3 = computation.execute_consumption(bytes_, fc, task_complex_index, "c")
        wan = float(para.get("cloud_wan_rtt", 0.0))
        t_cloud = t1 + t2 + t3 + wan

        return min(t_local, best_edge_t, t_cloud)

    def _compute_rank_u_for_app(self, env, uid, task_complex_index):
        """
        【关键路径】为单个应用计算所有节点的 HEFT upward-rank
        rank_u(v) = w(v) + max_{succ} rank_u(succ)

        Args:
            env: 环境对象
            uid: 用户ID
            task_complex_index: 任务复杂度索引
        """
        g = self.ts.subgraph_list[uid].nx_graph
        topo = list(nx.topological_sort(g))
        rank = {}

        # 逆拓扑序计算 rank_u
        for v in reversed(topo):
            wv = self._estimate_w(env, uid, v, task_complex_index)
            succs = list(g.successors(v))
            if not succs:
                rank[v] = wv
            else:
                rank[v] = wv + max(rank[s] for s in succs)

        self.rank_u[uid] = rank

    def _build_static_edge_index(self):
        """预先构建边索引，添加安全检查防止索引越界"""
        adj = nx.to_numpy_array(self.basegraph.nx_graph)
        edge_coords = np.array(adj.nonzero())

        # 过滤自环 (src == dst)
        valid_mask = edge_coords[0] != edge_coords[1]
        edge_coords = edge_coords[:, valid_mask]

        # 修复验证边索引范围
        # 获取基础图的实际节点数
        num_graph_nodes = adj.shape[0]  # 基础图的节点数

        # 过滤掉超出节点范围的边
        if edge_coords.size > 0:
            valid_mask = (edge_coords[0] < num_graph_nodes) & (edge_coords[1] < num_graph_nodes)
            edge_coords = edge_coords[:, valid_mask]

        # 如果没有有效边，创建少量自环防止 GNN 出错
        if edge_coords.size == 0:
            num_self_loops = min(num_graph_nodes, 10)
            edge_coords = np.stack([
                np.arange(num_self_loops),
                np.arange(num_self_loops)
            ], axis=0)

        return torch.tensor(edge_coords, dtype=torch.long, device=self.device)

    def get_graph_state_new(self, env: Environment, task, complex_index: int, slot=None):
        """
        生成图状态 Feature Matrix (X) - 关键路径增强版（修复：不跨应用映射 subtask->user）

        约定：
        - complex_index 必须是 int（task_complex_index）
        - 关键路径 rank_u 只对 current_user 的 subgraph 计算与使用
        - 对不属于 current_user 子图的节点：关键路径特征置 0
        - task_size_normalized / static_edge_index 仍以 basegraph 维度构建（num_nodes = basegraph_num）
        - slot: 当前时间槽（如果传入则优先使用，否则尝试从 env.slot 获取）

        Args:
            env: Environment 对象
            task: (user_id, subtask_id) 元组
            complex_index: 任务复杂度索引
            slot: 当前时间槽（可选，默认 None）
        """
        # ---------- 安全处理 task / complex_index ----------
        if isinstance(task, (list, tuple)) and len(task) >= 2:
            current_user_id = int(task[0])
            current_subtask_id = int(task[1])
        else:
            current_user_id, current_subtask_id = 0, 0

        # complex_index 兜底：防止误传 list（para["task_complex"]）
        if isinstance(complex_index, (list, tuple, np.ndarray)):
            complex_index = int(getattr(env, "task_complex_index", 0))
        else:
            complex_index = int(complex_index)

        # ---------- 关键路径：只为当前 user lazy compute ----------
        uid = current_user_id
        if uid not in self.rank_u:
            self._compute_rank_u_for_app(env, uid, complex_index)

        rank_u_u = self.rank_u.get(uid, {})
        max_rank_u = max(rank_u_u.values()) if rank_u_u else 1.0
        max_rank_u = max(max_rank_u, 1e-9)

        # 当前应用 deadline（秒）
        deadline_u = float(self.ts.get_app_deadline_slot(uid)) * float(para["slot_interval"])
        deadline_u = max(deadline_u, 1e-9)

        # 修复统一时间基准 - 优先使用传入的 slot，否则尝试从 env.slot 获取
        if slot is not None:
            now = para["slot_interval"] * slot
        elif hasattr(env, 'slot'):
            now = para["slot_interval"] * env.slot
        else:
            # 兜底：如果都没有，假设是 0 或者抛出警告
            now = 0.0
            print(f"[Warning] get_graph_state_new: slot not provided and env.slot not found, using 0.0")

        # ---------- 全局特征（保持你原来的设计） ----------
        u_upload = self.ts.devices_upload_useful
        u_exe = self.ts.devices_exe_useful

        # 【关键修复 3.2】使用 now 计算，避免特征失真
        curr_upload_wait = max(0.0, float(u_upload[uid]) - now)
        curr_exe_wait = max(0.0, float(u_exe[uid]) - now)

        # 每个边缘节点的等待时间（归一化到 [0,1]）
        # 【关键修复 3.2】使用 now 计算
        edge_waits = [min(2.0, max(0.0, float(t) - now)) / 2.0 for t in self.ts.remain_times]

        global_features = [
            min(1.0, curr_upload_wait),
            min(1.0, curr_exe_wait),
        ] + edge_waits

        # 加入距离特征（归一化）
        edge_distances = env.device_list[uid].edge_distances
        norm_dists = [min(1.0, float(d) / 500.0) for d in edge_distances]
        global_features = global_features + norm_dists

        # ---------- 节点特征维度 ----------
        # [in_app, size_norm, is_current, rank_u_norm, urgency_norm] + global_features
        # 修复添加 in_app 特征，让GAT能够区分当前应用节点和其他节点
        feature_len = 5 + len(global_features)

        # ---------- 当前用户子图节点集合 ----------
        g_u = self.ts.subgraph_list[uid].nx_graph
        nodes_u = set(g_u.nodes())

        # ---------- 构建节点特征矩阵（num_nodes = basegraph_num） ----------
        raw_data = []
        for node_id, size_norm in enumerate(self.ts.task_size_normalized):
            in_app = 1.0 if node_id in nodes_u else 0.0

            # current_subtask_id 本身应该属于该 app；为了稳，叠加 in_app
            is_current = 1.0 if (node_id == current_subtask_id and in_app > 0.5) else 0.0

            ru = float(rank_u_u.get(node_id, 0.0))
            # 只对 in_app 节点有效
            rank_u_norm = (ru / max_rank_u) * in_app

            # 紧迫性（简化版）：rank_u / deadline，占比越大越紧迫
            urgency_norm = float(np.clip(ru / deadline_u, 0.0, 1.0)) * in_app

            node_feats = [
                in_app,  # 修复添加 in_app 特征
                float(size_norm),
                is_current,
                rank_u_norm,
                urgency_norm,
            ] + global_features

            # 长度保护
            if len(node_feats) < feature_len:
                node_feats += [0.0] * (feature_len - len(node_feats))
            elif len(node_feats) > feature_len:
                node_feats = node_feats[:feature_len]

            raw_data.append(node_feats)

        x = torch.tensor(raw_data, dtype=torch.float32, device=self.device)

        # ---------- edge_index 安全过滤 ----------
        num_nodes = x.shape[0]
        edge_index = self.static_edge_index

        if edge_index.numel() > 0:
            max_idx = int(edge_index.max().item())
            if max_idx >= num_nodes:
                valid_mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
                edge_index = edge_index[:, valid_mask]
                if edge_index.numel() == 0:
                    device = x.device
                    num_self_loops = min(num_nodes, 10)
                    edge_index = torch.stack(
                        [torch.arange(num_self_loops, device=device),
                         torch.arange(num_self_loops, device=device)],
                        dim=0
                    )

        batch = torch.zeros(x.size(0), dtype=torch.long, device=self.device)
        return Data(x=x, edge_index=edge_index, batch=batch)

    def _get_user_by_subtask(self, subtask_id, user_to_subtasks=None):
        """
        辅助函数：根据 subtask_id 判断属于哪个 user
        """
        if user_to_subtasks is None:
            # 临时构建映射
            user_to_subtasks = {}
            for user_id in range(len(self.subgraph_list)):
                user_to_subtasks[user_id] = set(self.subgraph_list[user_id].nx_graph.nodes())

        # 遍历所有用户，找到包含该 subtask_id 的用户
        for user_id, subtasks in user_to_subtasks.items():
            if subtask_id in subtasks:
                return user_id

        return -1  # 未找到

    def get_tasks(self, slot, sort_tasks: bool = False, enable_triage: bool = True, triage_margin: float = 0.2):
        """
        [关键路径改进版] 获取当前 slot 需要调度的任务列表
        - 严格过滤死尸任务
        - 可选择是否按 rank_u（关键路径优先）+ deadline 紧迫性排序
        - 提前剔除（Triage）：过滤掉"已经没救了"的应用（deadline 还没到但赶不上了）

        Args:
            slot: 当前时隙
            sort_tasks: 是否对任务排序（默认 False，仅 GAT-PPO 使用启发式排序）
            enable_triage: 是否启用提前剔除（默认 True）
            triage_margin: 提前剔除的阈值（默认 0.2 秒），slack_cp < -margin 的应用会被剔除
                          - 0.2s: 比较温和的提前剔除（约 20 slots）
                          - 0.5s: 更激进（约 50 slots）
                          - -1.0: 保守（只在 slack_cp < 1s 时才剔除）
        """
        tasks = []
        # 定义搜索范围：只看 Started 且 (未完成 且 未超时) 的用户
        # 修复排除 application_timeout_finished
        active_users = (
            self.ts.application_started
            - self.ts.application_finished
            - getattr(self.ts, "application_timeout_finished", set())
        )

        # 【关键修复 1.1】统一时间基准：使用 slot * dt 与训练循环一致
        # 加 EPS 避免浮点误差
        EPS = 1e-9
        now = para["slot_interval"] * slot

        # 预计算所有用户的 slack_cp（用于提前剔除）
        user_slack_cp = {}
        if enable_triage:
            for user in active_users:
                enter_time = self.ts.enter_time[user]
                if enter_time == float("inf"):
                    user_slack_cp[user] = float("inf")
                    continue
                user_slack_cp[user] = self.get_slack_cp(user, slot, now=now)

        # 第一次遍历：收集所有候选任务（用于 fallback 保护）
        tasks_all = []
        for user in active_users:
            # 再次检查：如果 enter_time 是 inf，说明还没真正到达，跳过
            enter_time = self.ts.enter_time[user]
            if enter_time == float("inf"):
                continue

            # 提前剔除（Triage）：过滤掉"已经没救了"的应用
            if enable_triage and user_slack_cp.get(user, float("inf")) < -triage_margin:
                # slack_cp 负得太多，说明即使只看关键路径下界，也赶不上了
                # 跳过这个用户的所有任务（不修改 ts 状态，等 check_timeouts() 正式超时）
                continue

            # 使用各自应用的 deadline（可能有更短的）
            app_deadline_abs = enter_time + self.ts.get_app_deadline_slot(user) * para["slot_interval"]

            # 遍历该用户的所有子任务
            for subtask, start_t in self.ts.start_time[user].items():
                # 1. 任务已开始 (start_time != inf)
                # 2. 任务未完成 (finish_time == inf)
                # 3. 任务开始时间 <= 当前时隙 (Ready to schedule)
                if start_t == float("inf"):
                    continue
                if self.ts.finish_time[user][subtask] != float("inf"):
                    continue
                # 【关键修复 1.2】使用 now + EPS 避免浮点误差，保持边界一致性
                if start_t > now + EPS:
                    continue

                # 修复真正的死尸过滤：看"最乐观情况下的最早开工时间"是否也已超过 deadline
                # 不要用 max(upload, exe) —— 那会误杀；要用 min(本地可开始, 卸载可开始)
                start_if_local = max(start_t, self.ts.devices_exe_useful[user])
                start_if_offld = max(start_t, self.ts.devices_upload_useful[user])
                best_possible_start = min(start_if_local, start_if_offld)

                # 【关键修复 2.1】添加 EPS 容忍度，避免刚好等于时误杀
                if best_possible_start > app_deadline_abs + EPS:
                    continue  # 真死尸，不再给 RL/传统算法喂垃圾样本

                tasks.append((user, subtask))  # 修复返回 tuple，避免 dict key 报错

        # Fallback 保护：如果 triage 过滤后没有任务了，退回最接近 deadline 的应用
        if enable_triage and not tasks:
            # 找到 slack_cp 最接近 0 的（即最不太可能赶不上的）应用
            triage_users = [(u, s) for u, s in user_slack_cp.items() if s < -triage_margin]
            if triage_users:
                # 挑选 slack_cp 最大的（最接近 0 的，即"最不太可能赶不上的"）
                triage_users.sort(key=lambda x: x[1], reverse=True)
                best_user = triage_users[0][0]
                # 重新遍历这个用户的任务
                enter_time = self.ts.enter_time[best_user]
                if enter_time != float("inf"):
                    app_deadline_abs = enter_time + self.ts.get_app_deadline_slot(best_user) * para["slot_interval"]
                    for subtask, start_t in self.ts.start_time[best_user].items():
                        if start_t == float("inf"):
                            continue
                        if self.ts.finish_time[best_user][subtask] != float("inf"):
                            continue
                        if start_t > now + EPS:
                            continue
                        start_if_local = max(start_t, self.ts.devices_exe_useful[best_user])
                        start_if_offld = max(start_t, self.ts.devices_upload_useful[best_user])
                        best_possible_start = min(start_if_local, start_if_offld)
                        if best_possible_start > app_deadline_abs + EPS:
                            continue
                        tasks.append((best_user, subtask))

        # 【关键路径改进】按关键路径 + deadline 紧迫性排序（可选）
        if tasks and sort_tasks:
            # 确保 rank_u 已计算（lazy compute）
            task_complex_index = self.ts.env.task_complex_index
            for (u, _) in tasks:
                if u not in self.rank_u:
                    self._compute_rank_u_for_app(self.ts.env, u, task_complex_index)

            # 计算 deadline 紧迫性 (slack)
            def get_slack_cp(user_id):
                """获取应用的关键路径 slack（越小越急）"""
                enter_time = self.ts.enter_time[user_id]
                if enter_time == float("inf"):
                    return float("inf")

                app_deadline_abs = enter_time + self.ts.get_app_deadline_slot(user_id) * para["slot_interval"]

                # cp_remain: 该应用 ready 任务中最大的 rank_u（下界估计）
                user_tasks = [t for t in tasks if t[0] == user_id]
                if not user_tasks:
                    return float("inf")

                cp_remain = max(self.rank_u[user_id].get(t[1], 0.0) for t in user_tasks)
                slack_cp = app_deadline_abs - now - cp_remain
                return float(slack_cp)  # 修复保留负值，负得越多越急

            # 排序优先级：1) slack 越小越急  2) rank_u 越大越关键
            tasks.sort(key=lambda pair: (get_slack_cp(pair[0]), -self.rank_u[pair[0]].get(pair[1], 0.0)))

        return tasks

    def get_task_priority_info(self, task, slot, all_tasks=None):
        """
        获取任务的优先级信息，用于自定义排序策略

        Args:
            task: (user_id, subtask_id) 任务元组
            slot: 当前时隙
            all_tasks: 当前所有任务的列表（用于计算 cp_remain，可选）

        Returns:
            dict: 包含 rank_u, slack, task_size, enter_time 等信息
        """
        uid, sid = task
        # 【关键修复 1.3】统一时间基准
        now = para["slot_interval"] * slot

        # 确保 Rank_u 已计算 (Lazy Compute)
        if uid not in self.rank_u:
            task_complex_index = self.ts.env.task_complex_index
            self._compute_rank_u_for_app(self.ts.env, uid, task_complex_index)

        # 1. 关键路径优先级 (Rank_u)
        rank_u = self.rank_u.get(uid, {}).get(sid, 0.0)

        # 2. 松弛时间 (Slack)
        enter_time = self.ts.enter_time[uid]
        if enter_time == float("inf"):
            slack = float("inf")
        else:
            app_deadline_abs = enter_time + self.ts.get_app_deadline_slot(uid) * para["slot_interval"]

            # 修复计算 cp_remain（关键路径剩余时间）
            # cp_remain = 该应用 ready 任务中最大的 rank_u
            if all_tasks is not None:
                user_tasks = [t for t in all_tasks if t[0] == uid]
                if user_tasks:
                    cp_remain = max(self.rank_u[uid].get(t[1], 0.0) for t in user_tasks)
                else:
                    cp_remain = rank_u  # 如果没有其他任务，就用当前任务的 rank_u
            else:
                # 兜底：如果没有提供 all_tasks，就用当前任务的 rank_u
                cp_remain = rank_u

            deadline_remain = app_deadline_abs - now
            slack = deadline_remain - cp_remain

        # 3. 任务大小
        s = self.ts.task_size
        if isinstance(s, (list, tuple, np.ndarray)) and len(s) > 0 and isinstance(s[0], (list, tuple, np.ndarray)):
            task_size = float(s[uid][sid])
        elif isinstance(s, dict):
            task_size = float(s.get(sid, 0))
        else:
            task_size = float(s[sid])

        return {
            "rank_u": rank_u,
            "slack": slack,
            "task_size": task_size,
            "enter_time": enter_time
        }

    def get_tasks_priority_info_batch(self, tasks, slot):
        """
        【批量计算】获取多个任务的优先级信息（避免重复遍历）

        Args:
            tasks: 任务列表 [(user_id, subtask_id), ...]
            slot: 当前时隙

        Returns:
            dict: {task: priority_info_dict}
        """
        # 【关键修复 1.4】统一时间基准
        now = para["slot_interval"] * slot

        # 预计算所有应用的 rank_u
        task_complex_index = self.ts.env.task_complex_index
        for (uid, _) in tasks:
            if uid not in self.rank_u:
                self._compute_rank_u_for_app(self.ts.env, uid, task_complex_index)

        # 按应用分组任务
        tasks_by_app = {}
        for task in tasks:
            uid = task[0]
            if uid not in tasks_by_app:
                tasks_by_app[uid] = []
            tasks_by_app[uid].append(task)

        # 计算每个应用的 cp_remain（该应用 ready 任务中最大的 rank_u）
        cp_remain_by_app = {}
        for uid, app_tasks in tasks_by_app.items():
            cp_remain_by_app[uid] = max(self.rank_u[uid].get(t[1], 0.0) for t in app_tasks)

        # 批量获取任务大小
        s = self.ts.task_size

        # 批量计算优先级信息
        result = {}
        for task in tasks:
            uid, sid = task

            rank_u = self.rank_u[uid].get(sid, 0.0)
            cp_remain = cp_remain_by_app[uid]

            enter_time = self.ts.enter_time[uid]
            if enter_time == float("inf"):
                slack = float("inf")
            else:
                app_deadline_abs = enter_time + self.ts.get_app_deadline_slot(uid) * para["slot_interval"]
                deadline_remain = app_deadline_abs - now
                slack = deadline_remain - cp_remain

            # 获取任务大小
            if isinstance(s, (list, tuple, np.ndarray)) and len(s) > 0 and isinstance(s[0], (list, tuple, np.ndarray)):
                task_size = float(s[uid][sid])
            elif isinstance(s, dict):
                task_size = float(s.get(sid, 0))
            else:
                task_size = float(s[sid])

            result[task] = {
                "rank_u": rank_u,
                "slack": slack,
                "task_size": task_size,
                "enter_time": enter_time
            }

        return result

    def pick_topk_tasks(self, slot, ready_tasks, k=30, adaptive=True):
        """
        【Top-K 候选】从 ready_tasks 中选择 Top-K 最紧急的任务

        排序策略（按优先级）：
        1. slack_cp 越小越急（应用级紧迫性：deadline - now - cp_remain）
        2. 任务大小调整的紧迫度（大任务需要更多时间，同等剩余时间下更紧急）
        3. rank_u 越大越关键（应用内关键性）
        4. start_time 越小越早 ready

        【动态K】如果adaptive=True，根据ready任务数量动态调整K值
        - 轻度负载（<=30任务）：K=min(20, len)
        - 中度负载（30-50任务）：K=min(30, len)
        - 重度负载（>50任务）：K=min(50, len)

        Args:
            slot: 当前时隙
            ready_tasks: 所有 ready 任务列表 [(user_id, subtask_id), ...]
            k: 返回的 Top-K 数量（基础值）
            adaptive: 是否使用动态K值调整

        Returns:
            Top-K 任务列表
        """
        if not ready_tasks:
            return []

        # 【动态K值调整】
        num_ready = len(ready_tasks)
        if adaptive:
            # 尝试从全局配置读取TOPK_MIN/TOPK_MAX
            topk_min = 20
            topk_max = 50
            try:
                from Experiments_new.exp_utils import CONFIG
                topk_min = CONFIG.get("TOPK_MIN", 20)
                topk_max = CONFIG.get("TOPK_MAX", 50)
            except:
                pass

            # 根据负载动态调整K
            if num_ready <= 30:
                k = min(topk_min, num_ready)
            elif num_ready <= 50:
                k = min(30, num_ready)
            else:
                k = min(topk_max, num_ready)
        else:
            k = min(k, num_ready)

        # 【关键修复 1.5】统一时间基准
        now = para["slot_interval"] * slot

        # 确保 rank_u 已计算
        task_complex_index = self.ts.env.task_complex_index
        for (u, _) in ready_tasks:
            if u not in self.rank_u:
                self._compute_rank_u_for_app(self.ts.env, u, task_complex_index)

        # 按应用分组任务
        tasks_by_user = {}
        for (u, s) in ready_tasks:
            tasks_by_user.setdefault(u, []).append(s)

        # 计算每个应用的 slack_cp 和任务大小
        slack_cp_by_user = {}
        task_size_map = {}  # (uid, sid) -> task_size

        for u, subtasks in tasks_by_user.items():
            enter_time = self.ts.enter_time[u]
            if enter_time == float("inf"):
                slack_cp_by_user[u] = float("inf")
                continue

            app_deadline_abs = enter_time + self.ts.get_app_deadline_slot(u) * para["slot_interval"]
            # cp_remain: 该应用 ready 任务 中最大的 rank_u
            cp_remain = max(self.rank_u[u].get(s, 0.0) for s in subtasks) if subtasks else 0.0
            # 修复保留负值，负得越多越急（不要max(0.0, ...)截断）
            slack_cp_by_user[u] = float(app_deadline_abs - now - cp_remain)

            # 获取每个任务的大小
            for s in subtasks:
                try:
                    t_size = float(self.ts.task_size[u][s]) if isinstance(self.ts.task_size, (list, np.ndarray)) else float(self.ts.task_size.get((u, s), 200000))
                    task_size_map[(u, s)] = t_size
                except:
                    task_size_map[(u, s)] = 200000.0  # 默认200KB

        # 排序键：(slack_adjusted, -rank_u, start_time)
        # slack_adjusted = slack_cp - (task_size / 1e6) * 0.5
        # 解释：任务越大，相当于"有效剩余时间"越少（需要更多执行时间）
        #     例如：slack_cp=5s，任务大小1MB -> slack_adjusted=5-0.5=4.5s
        #          slack_cp=5s，任务大小10MB -> slack_adjusted=5-5=0s（非常紧急！）
        def sort_key(t):
            u, s = t
            slack = slack_cp_by_user.get(u, float("inf"))
            t_size = task_size_map.get((u, s), 200000.0)
            # 任务大小调整：每MB减少0.5s的"有效slack"
            slack_adjusted = slack - (t_size / 1e6) * 0.5
            ru = float(self.rank_u[u].get(s, 0.0))
            st = float(self.ts.start_time[u].get(s, float("inf")))
            return (slack_adjusted, -ru, st)

        # 返回 Top-K
        return sorted(ready_tasks, key=sort_key)[:k]

    def get_slack_cp(self, uid, slot, now=None):
        """
        计算应用的关键路径 Slack（越小越急，负数表示已超时）
        Slack_CP = Deadline_Abs - Now - CP_Remain
        CP_Remain = max(rank_u of all ready tasks)

        Args:
            uid: 用户ID
            slot: 当前时隙
            now: 绝对时间（可选，默认为 slot * slot_interval）

        Returns:
            float: 关键路径松弛时间（负值表示已超时）
        """
        # 【关键修复 1.6】统一时间基准
        if now is None:
            now = para["slot_interval"] * slot

        enter = self.ts.enter_time[uid]
        if enter == float("inf"):
            return float("inf")

        app_deadline_abs = enter + self.ts.get_app_deadline_slot(uid) * para["slot_interval"]

        # 确保 rank_u 已计算
        if uid not in self.rank_u:
            self._compute_rank_u_for_app(self.ts.env, uid, self.ts.env.task_complex_index)

        # 计算 CP_Remain：当前该用户所有 ready 任务中 rank_u 的最大值
        # 注意：这里需要获取该用户当前的 ready tasks，为了效率，可以传入或简略估计
        # 简略估计：取该用户所有未完成任务的最大 rank_u
        unfinished_tasks = [n for n in self.ts.subgraph_list[uid].nx_graph.nodes
                           if self.ts.finish_time[uid][n] == float("inf")]

        if not unfinished_tasks:
            return float("inf")  # 已完成

        cp_remain = max(self.rank_u[uid].get(n, 0.0) for n in unfinished_tasks)

        return app_deadline_abs - now - cp_remain

    def get_app_dag_data(self, env, uid: int, slot: int, complex_index: int, current_subtask_id=None):
        """
        【DAG 边】获取单个应用的 DAG 图数据（使用应用依赖边，而非 basegraph 静态边）

        Args:
            env: 环境对象
            uid: 用户ID
            slot: 当前时隙
            complex_index: 任务复杂度索引
            current_subtask_id: 当前要决策的子任务 ID（用于计算前瞻特征）

        Returns:
            Data: PyG 图数据对象（包含 x, edge_index, batch）
            node2idx: {node_id: local_idx} 节点到局部索引的映射
        """
        g = self.ts.subgraph_list[uid].nx_graph
        nodes = list(g.nodes())
        node2idx = {nid: i for i, nid in enumerate(nodes)}
        
        # 构建 DAG 边（应用依赖边）
        if g.number_of_edges() > 0:
            edges = list(g.edges())
            src = [node2idx[a] for (a, b) in edges]
            dst = [node2idx[b] for (a, b) in edges]
            edge_index = torch.tensor([src, dst], dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        # --- 全局环境特征 (Global Features) ---
        # 【关键修复 3.1】统一时间基准 + 使用 now 而非 enter_time
        now = para["slot_interval"] * slot
        enter = self.ts.enter_time[uid]

        # (1) 本地等待时间 - 【关键修复 3.1】使用 now 计算，避免特征失真
        local_upload_wait = max(0.0, self.ts.devices_upload_useful[uid] - now)
        local_exe_wait = max(0.0, self.ts.devices_exe_useful[uid] - now)

        # (2) 边缘节点排队时间 - 【关键修复 3.1】使用 now 计算
        edge_waits = [max(0.0, float(t) - now) for t in self.ts.remain_times]

        # (3) 边缘节点负载特征 - 用于感知全局竞争
        # 计算每个 Edge 节点当前正在执行的任务数
        edge_loads = []
        for eid in range(para["edge_num"]):
            # 统计该 Edge 节点的核心占用情况
            # edge_useful[eid] 是一个列表，表示每个核心的完成时间
            # 非零值表示该核心正在执行任务
            eid_load = sum(1 for t in self.ts.edge_useful[eid] if t != 0)
            # 归一化（假设最大并发任务数为 edge_core，通常为 5）
            edge_loads.append(min(1.0, eid_load / max(1, self.ts.edge_core)))

        # (4) 边缘距离特征 (Distance)
        dists = [min(1.0, float(d)/1000.0) for d in env.device_list[uid].edge_distances]

        # 归一化 (根据你的环境参数调整，假设最大等待 2.0s)
        global_feats = [
            min(1.0, local_upload_wait / 2.0),
            min(1.0, local_exe_wait / 2.0)
        ] + [min(1.0, w / 2.0) for w in edge_waits] + edge_loads + dists
        # ------------------------------------------------

        # 构建动态节点特征
        # (now 已经在上面定义了)

        # 确保 rank_u 已计算
        if uid not in self.rank_u:
            self._compute_rank_u_for_app(env, uid, complex_index)
        rank_u = self.rank_u.get(uid, {})
        max_ru = max(rank_u.values()) if rank_u else 1.0
        max_ru = max(max_ru, 1e-9)

        # 计算应用级信息
        if enter != float("inf"):
            app_deadline_abs = enter + self.ts.get_app_deadline_slot(uid) * para["slot_interval"]
            app_deadline_slots = self.ts.get_app_deadline_slot(uid)
        else:
            app_deadline_abs = 0.0
            app_deadline_slots = 100.0

        # 计算ready节点的cp_remain（用于cp_slack）
        ready_nodes = []
        for nid in nodes:
            st = self.ts.start_time[uid].get(nid, float("inf"))
            ft = self.ts.finish_time[uid].get(nid, float("inf"))
            if st != float("inf") and ft == float("inf") and st <= now:
                ready_nodes.append(nid)

        cp_remain = max(rank_u.get(n, 0.0) for n in ready_nodes) if ready_nodes else 0.0

        # 【核武器三】前瞻特征 (Potential Costs)
        potential_costs = []
        if current_subtask_id is not None:
            try:
                # 获取当前任务大小
                curr_task_size = self.ts.get_task_size_bytes(uid, current_subtask_id)

                # 计算边缘节点预估完成时间 = 队列时间 + 执行时间
                # 【关键修复 4.1】使用队列时间而非绝对时间，避免特征过大
                # 归一化因子：假设最大耗时 2.0s
                for eid in range(para["edge_num"]):
                    e_power = env.edges[eid].edge_power * env.edges[eid].calculate_parameter
                    exec_time = curr_task_size / e_power
                    # 【关键修复 4.1】使用 max(0.0, remain_time - now) 计算实际队列时间
                    queue_time = max(0.0, self.ts.remain_times[eid] - now)
                    total = queue_time + exec_time
                    potential_costs.append(min(1.0, total / 2.0))
            except:
                # 出错兜底：全 0
                potential_costs = [0.0] * para["edge_num"]
        else:
            potential_costs = [0.0] * para["edge_num"]

        # 将前瞻特征加入全局特征
        global_feats = global_feats + potential_costs
        # ----------------------------------------------

        x_rows = []
        for nid in nodes:
            # 1. 任务是否 ready
            st = self.ts.start_time[uid].get(nid, float("inf"))
            ft = self.ts.finish_time[uid].get(nid, float("inf"))
            is_ready = (st != float("inf") and ft == float("inf") and st <= now)
            is_done = (ft != float("inf"))

            # 2. rank_u normalized
            ru = float(rank_u.get(nid, 0.0))
            ru_norm = ru / max_ru

            # 3. size_norm
            size_norm = float(self.ts.task_size_normalized[nid])

            # 4. slack（应用级）
            # 修复保留负值，负得越多越急（不要max(0.0, ...)截断）
            if app_deadline_slots > 0:
                slack = app_deadline_abs - now  # 保留负值
                slack_norm = min(1.0, slack / max(1e-6, app_deadline_slots * para["slot_interval"]))
            else:
                slack_norm = 1.0

            # 5. cp_slack（关键路径紧迫度）
            # cp_slack = deadline_remain - cp_remain（考虑关键路径剩余工作量的真实紧迫度）
            if app_deadline_slots > 0:
                cp_slack = (app_deadline_abs - now) - cp_remain  # 保留负值
                cp_slack_norm = min(1.0, cp_slack / max(1e-6, app_deadline_slots * para["slot_interval"]))
            else:
                cp_slack_norm = 1.0

            x_rows.append([size_norm, float(is_ready), float(is_done), ru_norm, slack_norm, cp_slack_norm] + global_feats)

        x = torch.tensor(x_rows, dtype=torch.float32)
        batch = torch.zeros(x.size(0), dtype=torch.long)

        return Data(x=x, edge_index=edge_index, batch=batch), node2idx
