# -*- coding: utf-8 -*-
"""
TaskScheduler: the RL-facing task-state manager.

English
-------
TaskScheduler is the central per-app/per-subtask state machine that every
algorithm (RL agents and heuristics) interacts with:
  - Tracks arrival, queueing, upload, execution and completion of subtasks
    across 150 users, 8 edges and 1 cloud.
  - Maintains the per-edge remaining-work estimate `core_remaining_work`
    used by the guide-score functions, and the per-device queue occupancy
    `devices_exe_useful` / `devices_upload_useful` used in state features.
  - Enforces per-app deadline slots (`get_app_deadline_slot`) and computes
    the R1-5 latency metric `get_avg_results(only_successful=False,
    timeout_charge="deadline")` (D_all) and the legacy D_succ
    (only_successful=True).
  - Exposes the action mask (valid local/cloud/edge placements) and
    subtask size / complexity helpers consumed by the guide scores and the
    GAT state encoder.

This is the single source of truth for task state; the RL agent never
mutates env state directly, only through TaskScheduler methods.

中文
----
任务状态机: 负责任务到达/排队/上传/执行/完成的记录, 维护 edge 剩余工作量、设备
队列占用、app deadline、D_all/D_succ 指标, 提供 action mask 与子任务大小等接口。
所有算法 (RL + 启发式) 都通过它读写任务状态, RL 不直接改 env。
"""
import random
import os

import numpy as np
import heapq

import torch
from sympy.physics.units import length

from utils.constant import para
from typing import List
from Environment.Graph import SubGraph
from Environment.environment import Environment
from Environment import computation  # [新增] 用于计算本地执行时间
import networkx as nx
from utils.tools import *

# 【性能优化】添加VERBOSE开关，默认关闭海量print输出
VERBOSE = os.environ.get("VERBOSE", "0") == "1"

def vprint(*args, **kwargs):
    """条件打印：仅在VERBOSE模式才输出"""
    if VERBOSE:
        print(*args, **kwargs)




class TaskScheduler:
    # 修复超时判定的 epsilon，防止浮点误差导致误判
    # 【再次修复】增大到 1e-5，比 1e-6 更抗浮点误差，避免 Finish==Deadline 被误判
    TIMEOUT_EPSILON = 1e-5

    def __init__(self, user_num, subgraph_num, basegraph_num, env: Environment,
                 tasksize=para["task_size_range"], tight_deadline_config=None, seed=None):
        """
        Args:
            tight_deadline_config: 可选，预计算的 tight deadline 配置
                {"tight_user_ids": [uid1, ...], "app_deadline_slots": {uid: slot}, "deadline_slot_per_user": [...]}
                如果提供，则使用预计算的值（可复现）；否则在 __init__ 中随机生成
            seed: 可选，用于生成 task_size 的随机种子（确保可复现）
        """
        # 记录大小
        self.user_num = user_num
        self.subgraph_num = subgraph_num
        self.basegraph_num = basegraph_num
        self.edge_core = para["edgecore_limit"]
        # 任务属性
        self.env = env
        self.subgraph_list = self.env.subgraph_list

        # 修复使用局部 RNG 替代全局 random，确保可复现
        self.rng = random.Random(seed) if seed is not None else random
        self.task_size = self.rng.choices(tasksize, k=basegraph_num)

        # 【诊断】打印 task_size 的样本值以确认单位
        if len(self.task_size) > 0:
            sample_sizes = self.task_size[:min(5, len(self.task_size))]
            sample_kb = np.array(sample_sizes) / 1024
            print(f"[TASKSCHEDULER INIT] task_size 样本值: {sample_sizes} "
                  f"(单位: Bytes, 相当于 {sample_kb} KB)")

        self.task_size_normalized = minmax_normalize(self.task_size)
        self.overtime = 0

        # 【TotalEnergy修复】初始化总能耗累加器
        self.total_energy = 0.0

        self.using_Algorithm = 0

        # 计算并保存固定的总子任务数（用于稳定的超时率计算）
        self.total_subtasks = sum(len(sub.nx_graph.nodes) for sub in self.subgraph_list)

        # 应用属性
        self.application_waiting = set([i for i in range(self.user_num)])
        self.application_started = set() # 整个应用是否已经开始
        self.application_finished = set() # 整个应用是否已经完成
        self.application_timeout_finished = set() # 超时未完成的应用
        self.application_task_timeout = set() # 发生过子任务超时的应用（用于更严格的 AppTO 统计）

        # 环境属性：设备，边缘，任务情况
        self.devices_exe_useful = [0 for i in range(self.user_num)]#终端设备本地执行时间
        self.devices_upload_useful = [0 for i in range(self.user_num)]#终端设备上传时间
        # self.edge_useful = [[] for _ in range(para["edge_num"])]
        self.edge_useful = [] # 每个边缘节点对应一个小顶堆, 表示其上的运行任务与等待任务完成时间
        self.core_remaining_work =[]#每个边缘节点的剩余工作量，edge_useful由这个生成
        for _ in range(para["edge_num"]):
            # self.edge_useful.append([])
            self.core_remaining_work.append([])
            self.edge_useful.append([0] * self.edge_core)

            # self.core_remaining_work.append([0] * self.edge_core)

        #这些变量是用来记录每个边缘节点的负载情况的，应该放component.py不应放这里
        # # 可用性指标
        # self.current_remainTime = 0  # 记录节点当前最快可用时间（最快任一核心空闲时间）
        # self.task_count = 0  # 记录节点当前任务数量
        # self.time_usefuls = [0] * para["edgecore_limit"]  # 记录节点各核心可用时间
        # # 存储指标
        #
        # self.task_size = 0  # 记录节点当前任务量(根据这个算存储需要空间和计算量)
        # self.used_storage = 0  # 已用存储空间
        # self.total_difficulty = 0  # 累计任务难度（任务大小*复杂度系数）
        #
        # self.edge_task_amount = 0#每个边缘节点上的运行任务量，关注任务量
        # self.edge_task_difficulty = 0  # 每个边缘节点上的运行任务总难度，关注计算量
        # self.edge_task_storage = 0 # 每个边缘节点上的运行任务总存储量，关注存储量



        self.rest_tasks = [len(sub.nx_graph.nodes) for sub in self.subgraph_list]#总任务数吧
        # 当前的总能耗
        self.energy = [0 for i in range(self.user_num)]
        self.de_wait_energy = [0 for i in range(self.user_num)]

        self.upload_energy = [0 for _ in range(self.user_num)]  # 每个用户的上传能耗

        # 时间相关的记录
        self.enter_time = [float("inf") for i in range(self.user_num)]#应用开始时间
        self.exit_time = [float("inf") for i in range(self.user_num)]#应用完成时间
        self.local_time = [0 for i in range(self.user_num)]  # 每个用户的本地执行时间
        self.upload_time = [0 for i in range(self.user_num)]  # 每个用户的总上传时间

        # 为部分应用设置更短的截止时间（增加难度）
        # 在初始化时就确定固定的用户集合，这些用户有更紧的 deadline
        # 这样每次运行都使用相同的用户集合（用 seed 固定）

        # ==================== 压力调整配置 ====================
        # 可以根据需要调整这些参数来改变超时压力：
        #
        # 【配置1：轻松模式】（默认）
        #   TIGHT_DEADLINE_RATIO = 0.05   # 5% 用户有紧 deadline
        #   TIGHT_DEADLINE_FACTOR = 0.5   # 紧 deadline = 50% 默认值
        #   基本不会超时
        #
        # 【配置2：中等压力】
        #   TIGHT_DEADLINE_RATIO = 0.1    # 10% 用户有紧 deadline
        #   TIGHT_DEADLINE_FACTOR = 0.4   # 紧 deadline = 40% 默认值
        #   预期超时率 5-10%
        #
        # 【配置3：高压力】
        #   TIGHT_DEADLINE_RATIO = 0.2    # 20% 用户有紧 deadline
        #   TIGHT_DEADLINE_FACTOR = 0.3   # 紧 deadline = 30% 默认值
        #   预期超时率 15-20%
        #
        # 【配置4：极限压力】
        #   TIGHT_DEADLINE_RATIO = 0.3    # 30% 用户有紧 deadline
        #   TIGHT_DEADLINE_FACTOR = 0.2   # 紧 deadline = 20% 默认值
        #   预期超时率 25-30%
        # =================================================

        # 修复支持预计算的 tight deadline 配置（可复现）
        if tight_deadline_config is not None:
            # 使用预计算的配置（可复现模式）
            # 优先使用 deadline_slot_per_user（最直接的方式）
            if "deadline_slot_per_user" in tight_deadline_config:
                self.deadline_slot_per_user = list(tight_deadline_config["deadline_slot_per_user"])
                self.app_deadline_slots = tight_deadline_config.get("app_deadline_slots", {})

                # 统计紧 deadline 用户数和范围
                default_slot = para["deadline_slot"]
                tight_users = [uid for uid, slot in enumerate(self.deadline_slot_per_user)
                              if slot != default_slot]
                num_tight_users = len(tight_users)
                if tight_users:
                    tight_slots = [self.deadline_slot_per_user[uid] for uid in tight_users]
                    min_slot = min(tight_slots)
                    max_slot = max(tight_slots)
                else:
                    min_slot = max_slot = default_slot

                print(f"[Deadline Setup (PRE-CALCULATED)] 总用户={self.user_num}, 紧deadline用户={num_tight_users}, "
                      f"紧deadline范围={min_slot}-{max_slot} slot, 默认={default_slot} slot")
            else:
                # 兼容旧格式：只保存了 tight_user_ids + app_deadline_slots
                tight_deadline_users = tight_deadline_config.get("tight_user_ids", [])
                self.app_deadline_slots = tight_deadline_config.get("app_deadline_slots", {})

                # 从 app_deadline_slots 生成 deadline_slot_per_user
                self.deadline_slot_per_user = [para["deadline_slot"]] * self.user_num
                for uid, slot in self.app_deadline_slots.items():
                    self.deadline_slot_per_user[uid] = slot

                num_tight_users = len(tight_deadline_users)
                min_factor = min([self.app_deadline_slots.get(uid, para["deadline_slot"]) / para["deadline_slot"]
                                for uid in tight_deadline_users]) if tight_deadline_users else 1.0
                max_factor = max([self.app_deadline_slots.get(uid, para["deadline_slot"]) / para["deadline_slot"]
                                for uid in tight_deadline_users]) if tight_deadline_users else 1.0
                print(f"[Deadline Setup (PRE-CALCULATED - OLD FORMAT)] 总用户={self.user_num}, 紧deadline用户={num_tight_users}, "
                      f"紧deadline范围={int(para['deadline_slot']*min_factor)}-{int(para['deadline_slot']*max_factor)} slot")
        else:
            # 随机生成（兼容旧用法）
            TIGHT_DEADLINE_RATIO = 0.3   # 30% 用户有紧 deadline

            # 使用固定的 seed 确保每次运行都选择相同的用户集合
            random.seed(42)
            num_tight_users = int(self.user_num * TIGHT_DEADLINE_RATIO)
            tight_deadline_users = random.sample(range(self.user_num), num_tight_users)

            self.app_deadline_slots = {}  # 存储每个应用的 deadline_slot（如果有特殊设置）
            # 直接生成 deadline_slot_per_user 数组
            self.deadline_slot_per_user = [para["deadline_slot"]] * self.user_num

            # 为每个紧deadline用户分配不同的因子（0.6到0.9阶梯式分布）
            # 确保每个用户每次reset都得到相同的因子
            for idx, uid in enumerate(tight_deadline_users):
                # 使用索引分配固定因子，确保确定性
                factor = 0.6 + (idx % 4) * 0.1  # 0.6, 0.7, 0.8, 0.9 循环分配
                tight_deadline = int(round(para["deadline_slot"] * factor))
                self.app_deadline_slots[uid] = tight_deadline
                self.deadline_slot_per_user[uid] = tight_deadline

            # 恢复随机 seed（不影响其他随机性）
            random.seed(None)

            # 显示紧deadline的分布范围
            min_factor = 0.6
            max_factor = 0.9
            print(f"[Deadline Setup (RANDOM)] 总用户={self.user_num}, 紧deadline用户={num_tight_users} ({TIGHT_DEADLINE_RATIO*100}%), "
                  f"紧deadline范围={int(para['deadline_slot']*min_factor)}-{int(para['deadline_slot']*max_factor)} slot "
                  f"({min_factor}-{max_factor}倍), "
                  f"默认={para['deadline_slot']} slot")

            # 保存配置供 reset() 复用（确保环境一致性）
            self._tight_deadline_config = {
                "tight_user_ids": tight_deadline_users,
                "app_deadline_slots": self.app_deadline_slots.copy(),
                "deadline_slot_per_user": self.deadline_slot_per_user.copy()
            }

        # 计算基于关键路径的应用级deadline
        self.app_deadline_slots_critical_path = self._compute_critical_path_deadlines()

        self.finish_time = None#各应用中各个任务的完成时间
        self.start_time = None#各应用中各个任务的开始时间
        self.est_time = None#各应用中各个任务的最早开始时间（仅依赖）
        self.ast_time = None#各应用中各个任务的实际开始时间（考虑设备占用）
        self.cp_tasks = None#各应用的关键路径任务集合
        self.init_times()

        self.remain_times = [0] * para["edge_num"]#初始化每个边缘节点的负载为0

    def init_times(self):
        # 遍历并生成用户的开始时间和结束时间
        self.start_time = [{} for _ in range(self.user_num)]
        self.finish_time = [{} for _ in range(self.user_num)]
        self.est_time = [{} for _ in range(self.user_num)]  # EST: 仅依赖的最早开始时间
        self.ast_time = [{} for _ in range(self.user_num)]  # AST: 实际开始时间
        self.cp_tasks = [set() for _ in range(self.user_num)]  # 关键路径任务集合
        for i, sub in enumerate(self.subgraph_list):
            # print(sub.nx_graph.nodes)
            for node_id in sub.nx_graph.nodes:
                self.start_time[i][node_id] = float('inf')
                self.finish_time[i][node_id] = float('inf')
                self.est_time[i][node_id] = None
                self.ast_time[i][node_id] = None

        # 初始化关键路径标记
        self._init_critical_path_marks()

    def _compute_critical_path_deadlines(self):
        """
        计算基于关键路径的应用级deadline

        应用级deadline = 关键路径长度 × 子任务deadline × (1 + 余量系数)

        Returns:
            dict: {user_id: app_deadline_slot}
        """
        app_deadline_slots = {}
        slack_factor = para.get("app_deadline_slack_factor", 0.25)
        app_deadline_alpha = para.get("app_deadline_alpha", 0.35)
        tight_app_slack_factor = para.get("tight_app_slack_factor", 0.0)
        base_deadline_slot = para["deadline_slot"]

        for user_id in range(self.user_num):
            subgraph = self.subgraph_list[user_id]

            # 计算关键路径长度（最长路径的节点数）
            try:
                # nx.dag_longest_path_length 返回的是最长路径的节点数 - 1
                # 所以需要 + 1 得到节点数
                cp_length = nx.dag_longest_path_length(subgraph.nx_graph) + 1
            except:
                # 如果计算失败（例如图中有环），使用总节点数作为保守估计
                cp_length = len(subgraph.nx_graph.nodes)

            # 计算应用级deadline_slot (带 alpha 缩放)
            app_deadline_slot = int(cp_length * base_deadline_slot * (1.0 + slack_factor) * app_deadline_alpha)
            app_deadline_slots[user_id] = app_deadline_slot

        # 如果存在紧deadline配置，取较小值 (带 tight_app_slack_factor + alpha)
        if hasattr(self, 'deadline_slot_per_user') and self.deadline_slot_per_user is not None:
            for user_id in range(self.user_num):
                tight_slot = self.deadline_slot_per_user[user_id]
                subgraph = self.subgraph_list[user_id]
                try:
                    cp_length = nx.dag_longest_path_length(subgraph.nx_graph) + 1
                except:
                    cp_length = len(subgraph.nx_graph.nodes)
                tight_app_deadline = int(cp_length * tight_slot * (1.0 + tight_app_slack_factor) * app_deadline_alpha)
                app_deadline_slots[user_id] = min(app_deadline_slots[user_id], tight_app_deadline)
        elif hasattr(self, 'app_deadline_slots') and self.app_deadline_slots:
            for user_id, tight_slot in self.app_deadline_slots.items():
                subgraph = self.subgraph_list[user_id]
                try:
                    cp_length = nx.dag_longest_path_length(subgraph.nx_graph) + 1
                except:
                    cp_length = len(subgraph.nx_graph.nodes)
                tight_app_deadline = int(cp_length * tight_slot * (1.0 + tight_app_slack_factor) * app_deadline_alpha)
                app_deadline_slots[user_id] = min(app_deadline_slots[user_id], tight_app_deadline)

        # 打印统计信息
        if user_id >= 0:  # 至少有一个用户
            cp_lengths = []
            app_deadlines = []
            for uid in range(self.user_num):
                subgraph = self.subgraph_list[uid]
                try:
                    cp_len = nx.dag_longest_path_length(subgraph.nx_graph) + 1
                except:
                    cp_len = len(subgraph.nx_graph.nodes)
                cp_lengths.append(cp_len)
                app_deadlines.append(app_deadline_slots[uid])

            # 【静默】不输出 deadline 配置信息
            # print(f"[App Deadline Setup (CRITICAL PATH)] 总用户={self.user_num}")
            # print(f"  关键路径长度: 最小={min(cp_lengths)}, 最大={max(cp_lengths)}, 平均={np.mean(cp_lengths):.1f}")
            # print(f"  应用级deadline_slot: 最小={min(app_deadlines)}, 最大={max(app_deadlines)}, 平均={np.mean(app_deadlines):.1f}")
            # print(f"  子任务deadline_slot: {base_deadline_slot} ({base_deadline_slot * para['slot_interval']:.2f}s)")
            # print(f"  余量系数: {slack_factor * 100}%")

        return app_deadline_slots

    def get_app_deadline_slot(self, user_id):
        """
        获取应用的 deadline_slot（应用级，基于关键路径）
        如果有特殊设置（紧张的 deadline），则使用特殊值
        否则使用基于关键路径计算的值

        修复应用级deadline与子任务级deadline分离
        - 子任务级deadline: para["deadline_slot"] (170 slot = 1.7s)
        - 应用级deadline: 基于关键路径长度计算，考虑余量
        """
        # 优先使用基于关键路径计算的应用级deadline
        if hasattr(self, 'app_deadline_slots_critical_path') and self.app_deadline_slots_critical_path is not None:
            return self.app_deadline_slots_critical_path[user_id]
        # 兼容旧代码
        return self.app_deadline_slots.get(user_id, para["deadline_slot"])

    def get_task_deadline_slot(self, user_id):
        """
        获取子任务的 deadline_slot（子任务级）

        修复子任务 deadline 应该是应用级 deadline 的分摊值
        而不是完整的应用级 deadline，否则会导致 TaskTO=0% 但 AppTO=45% 的怪象

        子任务级deadline用于判断单个子任务是否按时完成（过程指标）
        应用级deadline用于判断整个应用是否按时完成（SLA指标）

        Returns:
            子任务级deadline_slot（分摊后的）
        """
        # 获取应用级总 deadline
        total_app_slot = 0
        if hasattr(self, 'deadline_slot_per_user') and self.deadline_slot_per_user is not None:
            total_app_slot = self.deadline_slot_per_user[user_id]
        else:
            total_app_slot = self.app_deadline_slots.get(user_id, para["deadline_slot"])

        # 修复估算关键路径长度，分摊 deadline
        # 这样每个子任务都有自己的紧迫度，迫使 Agent 加快进度
        try:
            subgraph = self.subgraph_list[user_id]
            cp_length = len(subgraph.nx_graph.nodes)
            # 防御：至少为 1，避免除零
            cp_length = max(cp_length, 1)
        except:
            cp_length = 8  # 保守估计平均深度

        # 【V12 修复】直接返回子任务级 deadline，不再除以关键路径长度
        # 根本原因：deadline_slot 本身就是子任务级 deadline（如 180 slot = 1.8s）
        # 错误逻辑：task_deadline = total_app_slot / cp_length
        #            导致 180/10 = 18 slot（0.18s），TaskTO=96%+
        # 修复：直接使用子任务级 deadline
        task_deadline_slot = total_app_slot  # 直接使用，不除以 cp_length

        return task_deadline_slot

    def _init_critical_path_marks(self):
        """
        初始化关键路径任务标记

        使用 longest path 来识别关键路径任务：
        - 找到 DAG 的最长路径（关键路径）
        - 标记关键路径上的所有任务
        """
        for user_id in range(self.user_num):
            subgraph = self.subgraph_list[user_id]
            try:
                # 获取最长路径（关键路径）
                longest_path = nx.dag_longest_path(subgraph.nx_graph)
                # 标记关键路径任务
                self.cp_tasks[user_id] = set(longest_path)
            except Exception as e:
                # 如果无法计算，默认为空
                self.cp_tasks[user_id] = set()

    def _calculate_est(self, user_id, subtask_id):
        """
        计算任务的 EST（Earliest Start Time，仅考虑依赖）

        Args:
            user_id: 用户 ID
            subtask_id: 子任务 ID

        Returns:
            float: 最早开始时间（仅依赖）
        """
        # 对于入口节点，EST = 应用到达时间
        subgraph = self.subgraph_list[user_id]
        if subtask_id in subgraph.enter_nodes:
            return self.enter_time[user_id]

        # 对于非入口节点，EST = 所有前驱任务的最大完成时间
        est = 0.0
        for pred in subgraph.nx_graph.predecessors(subtask_id):
            if pred in self.finish_time[user_id] and self.finish_time[user_id][pred] != float('inf'):
                est = max(est, self.finish_time[user_id][pred])

        return est

    def compute_cp_metrics(self, eps=1e-9):
        """
        计算关键路径等待和阻塞指标

        Args:
            eps: 防止除零的小数值

        Returns:
            tuple: (cp_wait_avg, cp_stall_ratio_avg)
                - cp_wait_avg: 成功应用的关键路径等待时间均值
                - cp_stall_ratio_avg: 成功应用的关键路径阻塞率均值
        """
        cp_wait_list = []
        cp_ratio_list = []

        for user_id in range(self.user_num):
            # 只统计成功应用
            if user_id not in self.application_finished:
                continue

            # 获取关键路径任务
            cp_task_set = self.cp_tasks[user_id]
            if not cp_task_set:
                continue

            # 计算关键路径等待时间
            cp_wait = 0.0
            for task_id in cp_task_set:
                # 检查任务是否完成
                if task_id not in self.finish_time[user_id] or \
                   task_id not in self.ast_time[user_id] or \
                   task_id not in self.est_time[user_id]:
                    continue

                ast = self.ast_time[user_id][task_id]
                est = self.est_time[user_id][task_id]

                if ast is not None and est is not None:
                    cp_wait += max(0.0, ast - est)

            # 计算应用的总时长（makespan）
            if task_id not in self.finish_time[user_id]:
                continue
            makespan = self.finish_time[user_id][task_id] - self.enter_time[user_id]

            # 计算阻塞率
            cp_ratio = cp_wait / max(makespan, eps)

            cp_wait_list.append(cp_wait)
            cp_ratio_list.append(cp_ratio)

        # 计算均值
        cp_wait_avg = float(np.mean(cp_wait_list)) if cp_wait_list else 0.0
        cp_ratio_avg = float(np.mean(cp_ratio_list)) if cp_ratio_list else 0.0

        return cp_wait_avg, cp_ratio_avg

    def get_gantt_data(self, user_id):
        """
        获取指定应用的甘特图数据

        Args:
            user_id: 用户 ID

        Returns:
            list: [{"task_id": int, "channel": str, "ast": float, "ft": float, "is_cp": bool, "action": int}, ...]
        """
        if user_id >= len(self.subgraph_list):
            return []

        gantt_data = []
        cp_set = self.cp_tasks[user_id]

        # 需要一个额外的数据结构来记录每个任务的执行通道
        # 暂时使用简单的逻辑：根据任务开始时的设备状态来判断
        for task_id in self.subgraph_list[user_id].nx_graph.nodes:
            if task_id not in self.finish_time[user_id] or \
               task_id not in self.ast_time[user_id]:
                continue

            ft = self.finish_time[user_id][task_id]
            ast = self.ast_time[user_id][task_id]

            if ast is not None and ft != float('inf'):
                # 简化：根据任务大小和执行时间推断通道
                # 本地：通常执行时间较短（高算力）
                # 上传+边缘/云：需要传输时间
                duration = ft - ast
                task_size = self.get_task_size_bytes(user_id, task_id)

                # 估算本地执行时间
                f_local = float(self.env.device_list[user_id].local_power)
                _, local_exec_time = computation.execute_consumption(task_size, f_local, self.env.task_complex_index, "l")

                # 判断通道
                if duration <= local_exec_time * 1.1:  # 允许 10% 误差
                    channel = "local"
                else:
                    channel = "up"

                gantt_data.append({
                    "task_id": task_id,
                    "channel": channel,
                    "ast": ast,
                    "ft": ft,
                    "is_cp": task_id in cp_set,
                    "action": 0 if channel == "local" else 1  # 0=local, 1=up
                })

        return gantt_data

    def new_arrival(self, num_to_select, slot, fixed_uids=None):
        """
        随slot产生新应用机制，num_to_select表示一个slot能产生多少个应用（不是任务）

        Args:
            num_to_select: 要选择的用户数量（如果是固定 uid 模式，此参数会被忽略）
            slot: 当前 slot
            fixed_uids: 可选，指定的 uid 列表 [uid1, uid2, ...]。
                        如果提供，则使用这些 uid（可复现模式）；
                        如果为 None，则随机选择（兼容旧用法）

        Returns:
            chosen: 选择的 uid 集合
        """
        # 这个函数是用来处理应用的到达的，随机选用户表示其开始任务
        # 随机生成不在set里的，然后放进set里
        if not self.application_waiting:
            return []

        # 模式判断：固定 uid 模式 vs 随机模式
        if fixed_uids is not None:
            # 【固定 uid 模式】可复现
            # fixed_uids 是一个 uid 列表，按顺序应用
            chosen = set()
            for uid in fixed_uids:
                if uid in self.application_waiting:
                    chosen.add(uid)
                else:
                    # 如果指定的 uid 已经不在 waiting 里（可能已经到达），跳过
                    continue
            vprint(f"[FIXED UID MODE] slot={slot}, fixed_uids={list(fixed_uids)}, chosen={list(chosen)}")
        else:
            # 【随机模式】兼容旧用法
            if num_to_select >= len(self.application_waiting):
                chosen = self.application_waiting.copy()
            else:
                # 修复使用 random.sample（不放回抽样）替代 random.choices（有放回抽样）
                # random.choices 会导致重复抽样，set 去重后实际到达数 < num_to_select
                k = min(num_to_select, len(self.application_waiting))
                chosen = set(random.sample(list(self.application_waiting), k=k))

        # 更新时间，加入到started
        vprint("application_waiting:{}   -    {}  = ".format(self.application_waiting, chosen))
        self.application_waiting -= chosen
        vprint("new application_waiting:{}".format(self.application_waiting))
        self.application_started = self.application_started | chosen #开始任务的应用
        for c in chosen:
            self.enter_time[c] = (slot * para["slot_interval"])
            for en in self.subgraph_list[c].enter_nodes:
                self.start_time[c][en] = (slot * para["slot_interval"])

            # 【已移除】紧 deadline 设置已在初始化时完成，固定的用户集合
            # 不再每次到达时随机设置
        return chosen
    """
    修改哪些任务
    1. 计算传输时间
    2. 计算执行时间
    3. 更新ft和st
    """
    def check_prev_all_finished(self, user_id, subtask_id):
        """
        检查当前任务的所有前驱节点是否已经执行完毕
        :param user_id:
        :param subtask_id:
        :return:
        """
        predecessors = self.subgraph_list[user_id].nx_graph.predecessors(subtask_id)
        for prev in predecessors:
            if prev in self.finish_time[user_id] and self.finish_time[user_id][prev] == float('inf'):
                return False
        return True

    # def edge_update(self, task_info, upload_consumption, execute_consumption):
    #     """
    #     当前任务卸载至edge
    #         1. edge有空位，可直接插入任务的完成时间
    #         2. edge无空位，如果当前时间>edge中最小完成时间，则插入，并正常计算时间
    #         3. edge无空位，且当前时间<=edge中最小完成时间，把当前时间修改为edge中的最小完成时间，并插入
    #     :return:
    #     """
    #     user_id, subtask_id = task_info
    #     upload_e, upload_d = upload_consumption
    #     execute_e, execute_d = execute_consumption
    #     if len(self.edge_useful) < self.edge_core:
    #         self.update(task_info, upload_e + execute_e, upload_d + execute_d, local=False)
    #         finish_time = self.finish_time[user_id][subtask_id]
    #         heapq.heappush(self.edge_useful, finish_time)
    #     else:
    #         # 计算开始时间
    #         predecessors = self.subgraph_list[user_id].nx_graph.predecessors(subtask_id)
    #         start_time = max(self.devices_exe_useful[user_id], self.start_time[user_id][subtask_id])
    #         # 当前节点的开始时间是前驱节点的最大值和设备可用时间的最大值
    #         for pred in predecessors:
    #             if pred in self.finish_time[user_id]:
    #                 start_time = max(start_time, self.finish_time[user_id][pred])
    #
    #         self.rest_tasks[user_id] -= 1
    #         # 根据edge的情况修改结束时间和edge的可用情况
    #         if self.edge_useful[0] <= start_time + upload_d:  # 到达时间比上一个完成时间长，可以直接放到edge上
    #             heapq.heappushpop(self.edge_useful, start_time + upload_d + execute_d)
    #             self.update(task_info, upload_e + execute_e, upload_d + execute_d, local=False)
    #         else:
    #             top = self.edge_useful[0]
    #             # 计算等待时间有多久
    #             gap = top - (start_time + upload_d)
    #             heapq.heappushpop(self.edge_useful, start_time + upload_d + gap + execute_d)
    #             self.update(task_info, upload_e + execute_e, upload_d + gap + execute_d, local=False)
    #     # print(len(self.edge_useful))
    #     energy = upload_e + execute_e
    #     delay = self.finish_time[user_id][subtask_id] - self.start_time[user_id][subtask_id]
    #     return energy, delay

    def calculate_start_time(self, user_id, subtask_id, local):
        # 1. 计算并记录 EST（仅考虑依赖）
        est = self._calculate_est(user_id, subtask_id)
        self.est_time[user_id][subtask_id] = est

        # 2. 考虑设备当前状态
        device_available_time = (self.devices_exe_useful[user_id] if local
                                 else self.devices_upload_useful[user_id])

        # 3. 计算实际开始时间（AST = max(EST, device_available_time)）
        ast = max(est, device_available_time)

        # 记录 AST
        self.ast_time[user_id][subtask_id] = ast

        return ast

    def edge_update(self, task_info, task_size, upload_consumption, execute_consumption, edge_id):#多边缘环境中的
        """
        当前任务卸载至edge
            1. edge有空位，可直接插入任务的完成时间
            2. edge无空位，如果当前时间>edge中最小完成时间，则插入，并正常计算时间
            3. edge无空位，且当前时间<=edge中最小完成时间，把当前时间修改为edge中的最小完成时间，并插入
        :return:
        """

        user_id, subtask_id = task_info
        upload_e, upload_d = upload_consumption
        vprint("edge upload_e, upload_d:", upload_e, upload_d)
        execute_e, execute_d = execute_consumption
        target_edge = self.env.edges[edge_id]

        # 检查任务是否会超时，如果会超时则不执行
        # 使用子任务级deadline来判断单个任务是否能按时完成
        task_deadline_abs = self.enter_time[user_id] + self.get_task_deadline_slot(user_id) * para["slot_interval"]
        app_deadline_abs = self.enter_time[user_id] + self.get_app_deadline_slot(user_id) * para["slot_interval"]
        task_start_time = self.calculate_start_time(user_id, subtask_id, False)
        edge_useful_num = sum(1 for x in self.edge_useful[edge_id] if x != 0)

        # 计算预估完成时间
        if edge_useful_num < self.edge_core:
            estimated_finish = task_start_time + upload_d + execute_d
        else:
            # 需要排队
            top = self.edge_useful[edge_id][0]
            if top <= task_start_time + upload_d:
                estimated_finish = task_start_time + upload_d + execute_d
            else:
                gap = top - (task_start_time + upload_d)
                estimated_finish = task_start_time + upload_d + gap + execute_d

        # 修复移除预判超时终止逻辑，让任务照常执行（即使会超时）
        # 由 check_timeouts/finalize_episode 来统计超时，由 RL 的 reward/timeout penalty 来学习
        # 这样 episode 不会变得极短，reward 信号也不会被掐断
        # 原逻辑会导致：一旦预判超时就 rest_tasks=0，整个应用被终止，后续没有任务可调度
        if estimated_finish > app_deadline_abs + self.TIMEOUT_EPSILON:
            vprint(f"[Timeout Warning] Task ({user_id}, {subtask_id}) on Edge {edge_id}: "
                   f"est_finish={estimated_finish:.3f}s > app_deadline={app_deadline_abs:.3f}s")
            # 可以选择标记应用超时（可选），但不终止任务
            # 让任务照常执行，由 check_timeouts 统计超时

        # 更新目标边缘节点的负载
        target_edge.task_count += 1
        target_edge.total_task_size += task_size

        task_difficulty = task_size * para["task_complex"][target_edge.task_complex_index] #暂任务难度由边缘节点定义，待改正
        target_edge.total_difficulty += task_difficulty
        target_edge.used_storage += task_size * para["task_storage_index"] #暂任务存储由canstant直接定义，待改正

        # # 当前边缘端状态useful
        # print("卸载至目标边缘节点ID:", edge_id)
        # print("边缘节点可用时间:", self.edge_useful[edge_id])
        # print("当前任务总数量:", target_edge.task_count)
        # print("任务总规模:", target_edge.task_size)
        # print("总计算难度:", target_edge.total_difficulty)
        # print("当前使用存储空间:", target_edge.used_storage)
        # print("当前边缘每时计算能力", target_edge.edge_power * target_edge.calculate_parameter * para["slot_interval"])
        # print("当前边缘节点的负载情况:", self.core_remaining_work[edge_id])

        # print("卸载至目标边缘 ", edge_id," :   ", self.edge_useful[edge_id],"\n", target_edge.task_count, "\n", target_edge.task_size,"\n", target_edge.total_difficulty,"\n", target_edge.used_storage)

        # print("处理前core_remaining_work", self.core_remaining_work)

        # print("edge_useful_num",edge_useful_num)
        # if len(self.edge_useful[edge_id]) < self.edge_core: #当前并行任务数少于总核心数，有空闲核心直接运行
        edge_useful_num = sum(1 for x in self.edge_useful[edge_id] if x != 0)
        # 【移除】不再计算空闲等待能耗，只计算实际产生的能耗
        # self.de_wait_energy[user_id] += upload_d * para["local_wait"]

        # 计算开始时间#后续可能需要修改为现算，现在是通过上一个slot产生的用时useful和旧的时间系统计算
        task_start_time = self.calculate_start_time(user_id, subtask_id, False)

        if edge_useful_num < self.edge_core:  #当前并行任务数少于总核心数，有空闲核心直接运行
            # [修复] 传入 upload_duration=upload_d
            self.update(task_info, upload_e + execute_e, upload_d + execute_d, local=False,
                      upload_duration=upload_d, exec_duration=execute_d)
            subtask_finish_time = self.finish_time[user_id][subtask_id]
            # print("heappushpop前edge_useful:", self.edge_useful[edge_id])
            heapq.heappushpop(self.edge_useful[edge_id], subtask_finish_time)
            # print("heappushpop后edge_useful:", self.edge_useful[edge_id])
            if len(self.core_remaining_work[edge_id]) < self.edge_core: #当前并行任务数少于总核心数，有空闲核心直接运行
                heapq.heappush(self.core_remaining_work[edge_id], task_difficulty)
            else:
                difficulty_together = task_difficulty + self.core_remaining_work[edge_id][0]
                popped = heapq.heappushpop(self.core_remaining_work[edge_id], difficulty_together)
            #     print("弹出的元素:", popped)



        else: #任务数超过可并行任务数，需要等待

            #需要修改，现在来任务就直接将上传时间设为任务时间，应该排队
            # if start_time<= self.devices_upload_useful[user_id]: #设备上传无需空置

            # self.rest_tasks[user_id] -= 1 #不知道为什么，但是这里多余，可能edge_update总是和update函数一起用
            # print("to edge rest_tasks[", user_id, "] : ", self.rest_tasks[user_id])
            # 根据edge的情况修改结束时间和edge的可用情况
            if self.edge_useful[edge_id][0] <= task_start_time + upload_d:  # 到达时间比上一个完成时间长，可以直接放到指定edge上
                # print("heappushpop前edge_useful:", self.edge_useful[edge_id])
                heapq.heappushpop(self.edge_useful[edge_id], task_start_time + upload_d + execute_d)
                # print("heappushpop后edge_useful:", self.edge_useful[edge_id])
                # [修复] 传入 upload_duration=upload_d
                self.update(task_info, upload_e + execute_e, upload_d + execute_d, local=False,
                          upload_duration=upload_d, exec_duration=execute_d)
                if len(self.core_remaining_work[edge_id]) < self.edge_core: #当前并行任务数少于总核心数，有空闲核心直接插入
                    heapq.heappush(self.core_remaining_work[edge_id], task_difficulty)#从按时间修改为按量
                else:
                    popped = heapq.heappushpop(self.core_remaining_work[edge_id], task_difficulty)#从按时间修改为按量
                #     print("弹出的元素:", popped)
                # self.update(task_info, upload_e + execute_e, upload_d + execute_d, local=False)
            else:
                top = self.edge_useful[edge_id][0]  #这里默认用最顶端的来计算#虽然改为按任务量计算但这个用于update函数使用
                # 计算等待时间有多久
                gap = top - (task_start_time + upload_d)
                # [建议添加] 如果 gap 太大，打印警告，帮助调试
                if gap > 1000:
                    vprint(f"[Warning] Huge congestion on Edge {edge_id}, gap={gap}")
                # print("heappushpop前edge_useful:", self.edge_useful[edge_id])
                heapq.heappushpop(self.edge_useful[edge_id], task_start_time + upload_d + gap + execute_d)
                # print("heappushpop后edge_useful:", self.edge_useful[edge_id])
                old_task_difficulty = self.core_remaining_work[edge_id][0]  # 这里默认用最顶端的来计算，这是因为当前设计会,完成edge中的最早完成的任务再插入新的
                if len(self.core_remaining_work[edge_id]) < self.edge_core: #当前并行任务数少于总核心数，有空闲核心直接插入
                    heapq.heappush(self.core_remaining_work[edge_id], old_task_difficulty + task_difficulty)#从按时间修改为按量
                else:
                    popped = heapq.heappushpop(self.core_remaining_work[edge_id],
                                      old_task_difficulty + task_difficulty)  # 从按时间修改为按量
                #     print("弹出的元素:", popped)
                # [修复] 传入 upload_duration=upload_d (设备只上传了 upload_d 这么久，gap是边缘排队时间)
                self.update(task_info, upload_e + execute_e, upload_d + gap + execute_d, local=False,
                          upload_duration=upload_d, exec_duration=execute_d)
                # print(len(self.edge_useful))

                # self.core_remaining_work = all_works
                # for x in self.core_remaining_work:
                #     self.time_usefuls = x / (target_edge.edge_power * target_edge.calculate_parameter)#由剩余任务量计算可用时间
        # print("处理后core_remaining_work", self.core_remaining_work)

        """
                    更新指定边缘节点的可用性（时间）（remain_time）。
                    :param edge_id: 边缘节点的 ID
                """
        # print("处理前edge_useful", self.edge_useful[edge_id])
        # for i in range(len(self.edge_useful[edge_id])):
        #     if i < len(self.core_remaining_work[edge_id]):
        #         self.edge_useful[edge_id][i] = self.core_remaining_work[edge_id][i] / (
        #                     target_edge.edge_power * target_edge.calculate_parameter)  # 由剩余任务量计算可用时间
        #     else:
        #         self.edge_useful[edge_id][i] = 0
        # print("处理后edge_useful", self.edge_useful[edge_id])

        # 当前边缘端状态
        # print("卸载至目标边缘节点ID:", edge_id)
        # print("边缘节点可用时间:", self.edge_useful[edge_id])
        # print("当前任务总数量:", target_edge.task_count)
        # print("任务总规模:", target_edge.task_size)
        # print("总计算难度:", target_edge.total_difficulty)
        # print("当前使用存储空间:", target_edge.used_storage)
        # print("当前边缘每时计算能力",
        #       target_edge.edge_power * target_edge.calculate_parameter * para["slot_interval"])
        # print("本次任务计算量", task_difficulty)
        # print("当前边缘节点的负载情况:", self.core_remaining_work[edge_id])
        # print("各边缘节点的负载情况:", self.core_remaining_work)

        # 【关键修正】更新 remain_times
        # edge_useful 是最小堆，[0] 就是最早一个核心变空闲的时间
        # 如果还有空位(0)，堆顶应该是0（初始化是[0]*core）。
        # 所以直接取堆顶即可，不需要复杂的 if/else
        self.remain_times[edge_id] = self.edge_useful[edge_id][0]

        # print("当前边缘节点[",edge_id,"]核心情况:", self.edge_useful[edge_id])
        if len(self.edge_useful[edge_id]) > self.edge_core:
            print("当前边缘节点[",edge_id,"]核心情况不对！:", self.edge_useful[edge_id])
            print("len(self.edge_useful)", len(self.edge_useful[edge_id]))
            print("self.edge_core", self.edge_core)
            input()
        #暂时未完成按量计算，先把这个减速注释掉，等完成后再加上
        # target_edge.update_core_speeds(edge_useful_num/self.edge_core)#更新核心速度

        energy = upload_e + execute_e
        delay = self.finish_time[user_id][subtask_id] - self.start_time[user_id][subtask_id]
        remain_time = self.remain_times[edge_id]

        # [REVISION FIX] 移除双重累计 — update() 内部已做 self.energy[user_id] += energy
        # self.energy[user_id] += energy

        return energy, delay, remain_time #添加核心最短等待时间


    def is_core_remaining_empty(self, edge_id):
        """
        判断指定边缘节点的核心任务堆是否为空
        :param edge_id: 边缘节点ID
        :return: 如果核心任务堆为空返回True,否则返回False
        """
        return len(self.core_remaining_work[edge_id]) == 0

    #每个slot进行edge计算，变更状态
    def edge_doTask_inslot(self):
        slot_time = para["slot_interval"]
        # print("处理前core_remaining_work", self.core_remaining_work)

        for edge_id in range(para["edge_num"]):
            target_edge = self.env.edges[edge_id]
            edge_speed = target_edge.edge_power * target_edge.calculate_parameter
            edge_calculation = slot_time * edge_speed#一个slot的计算量
            real_reduction = 0

            # 处理核心任务堆（保持最小堆结构）
            remaining_work = []
            original_heap = self.core_remaining_work[edge_id].copy()



            # 清空原堆
            self.core_remaining_work[edge_id] = []


            # 处理原堆中的每个元素
            for work in original_heap:
                # 计算本 slot 可完成的最大工作量
                actual_reduction = min(work, edge_calculation)
                real_reduction += actual_reduction  # 记录实际完成量

                if edge_calculation < work:#任务做不完
                    remaining = work - actual_reduction
                    # remaining_work.append(remaining)
                    heapq.heappush(remaining_work, remaining)
                elif work==0:
                    pass
                else:#做完的任务不能作为0插入，否则会放最前面
                    target_edge.task_count -= 1 #标志一个任务完成



            # 将处理后的元素重新插入堆中
            # 修复heappushpop 在空堆上会 push 又 pop 掉，导致堆可能永远为空
            # 改用 heappush 正确插入
            for work in remaining_work:
                heapq.heappush(self.core_remaining_work[edge_id], work)



            target_edge.total_difficulty -= real_reduction

            did_task_size = real_reduction / para["task_complex"][target_edge.task_complex_index]#计算完成的任务量
            target_edge.total_task_size -= did_task_size

            finished_task_storage = did_task_size * para["task_storage_index"]#计算完成的任务存储量
            target_edge.used_storage -= finished_task_storage
        # print("处理后core_remaining_work",self.core_remaining_work)



    # 更新当前执行的任务相关的信息
    def update(self, task_info, energy, delay, local=False, upload_duration=0, exec_duration=0):#device_upload_delay专门用来表示上传至云时的第一段上传时延
        """
        [修复版] 增加 upload_duration 和 exec_duration 参数，精确控制设备占用时间

        Args:
            task_info: (user_id, subtask_id)
            energy: 能耗
            delay: 总延迟（可能包含排队时间）
            local: 是否本地执行
            upload_duration: 上传持续时间（仅用于卸载任务）
            exec_duration: 执行持续时间（仅用于本地任务）
        """
        # 本地设备本来一次只能跑一个任务，所以时间还要取决于当前设备是否可用
        # 修改为本地设备可同时进行本地计算和上传任务各一个
        user_id, subtask_id = task_info

        # self.de_wait_energy[user_id] = delay * para["local_wait"]#后面计算空闲能耗时直接加上整个应用时间的等待能耗，这里相应的要减去对应的时间的等待能耗，对于本地与上传任务并行的情况，多出来的部分算是并行的代价
        # self.upload_energy[user_id] -= self.de_wait_energy[user_id]#消去多加的等待能耗

        # if local:#修改，将终端设备的本地计算时间记录和上传时间记录从ts放到benchmark中的step了，防止添加等待时间与边缘→云的时间混在一起
        #     self.local_time[user_id] += delay#本地计算时间#这里的delay不对，这里是整个任务考虑了等待时间的最后完成时间
        # else:
        #     self.upload_time[user_id] += delay#上传时间#上传至云的情况下，没有把边缘→云的时间分离出来，所以这里错误把边缘→云的时间也加上了
        #     print("upload_time[{}] += {} = {}".format(user_id, delay, self.upload_time[user_id]))
        # self. #本来想把任务量写这，后面觉得写这函数外面就行


        # 计算开始时间
        task_start_time = self.calculate_start_time(user_id, subtask_id, local)

        # 检查任务是否会超时，如果会超时则不执行
        # 区分子任务级和应用级deadline
        task_deadline_abs = self.enter_time[user_id] + self.get_task_deadline_slot(user_id) * para["slot_interval"]
        app_deadline_abs = self.enter_time[user_id] + self.get_app_deadline_slot(user_id) * para["slot_interval"]
        estimated_finish = task_start_time + delay

        # 判断子任务是否会超过子任务级deadline（过程指标）
        if estimated_finish > task_deadline_abs + self.TIMEOUT_EPSILON:
            # 子任务预计会超时，标记该应用发生过子任务超时（过程指标）
            if not hasattr(self, 'application_task_timeout'):
                self.application_task_timeout = set()
            self.application_task_timeout.add(user_id)

        # 修复移除预判超时终止逻辑，让任务照常执行（即使会超时）
        # 由 check_timeouts/finalize_episode 来统计超时，由 RL 的 reward/timeout penalty 来学习
        # 这样 episode 不会变得极短，reward 信号也不会被掐断
        # 原逻辑会导致：一旦预判超时就 rest_tasks=0，整个应用被终止，后续没有任务可调度
        if estimated_finish > app_deadline_abs + self.TIMEOUT_EPSILON:
            vprint(f"[Timeout Warning] Task ({user_id}, {subtask_id}) Local/Cloud: "
                   f"est_finish={estimated_finish:.3f}s > app_deadline={app_deadline_abs:.3f}s")
            # 可以选择标记应用超时（可选），但不终止任务
            # 让任务照常执行，由 check_timeouts 统计超时

        # 修复能耗在确定任务真的执行后才增加（避免超时任务也加能耗）
        self.energy[user_id] += energy#本地计算能耗以及上传能耗是在这里加的

        # 【TotalEnergy修复】累加到总能耗
        self.total_energy += energy

        # [关键修复] 更新设备占用时间
        if not local:
            # 卸载任务：设备被占用直到【开始时间 + 上传耗时】
            # 注意：这里不能用 total_delay，因为 total_delay 可能包含了在边缘的排队和计算时间
            # 我们只关心设备上传通道何时变为空闲
            self.devices_upload_useful[user_id] = task_start_time + upload_duration
        else:
            # 本地任务：设备被占用直到【开始时间 + 执行耗时】
            # 本地执行没有排队gap(除非本地排队逻辑)，通常 total_delay 就是 exec_duration
            self.devices_exe_useful[user_id] = task_start_time + exec_duration

        self.finish_time[user_id][subtask_id] = task_start_time + delay
        self.rest_tasks[user_id] -= 1
        # 修复防止 rest_tasks 变成负数（重复减法导致）
        if self.rest_tasks[user_id] < 0:
            self.rest_tasks[user_id] = 0

        # 修复统一记录"子任务超时"（与get_reward()逻辑一致）
        # 使用子任务级deadline来判断单个子任务是否超时
        task_deadline_abs = self.enter_time[user_id] + self.get_task_deadline_slot(user_id) * para["slot_interval"]
        finish_abs = self.finish_time[user_id][subtask_id]
        # 修复：使用 epsilon，Finish==Deadline 不应算超时
        EPS = 1e-6
        if self.enter_time[user_id] != float("inf") and finish_abs > task_deadline_abs + EPS:
            self.overtime += 1
            if hasattr(self, "application_task_timeout"):
                self.application_task_timeout.add(user_id)

        # 修改后继节点的开始时间为当前的结束时间
        successors = self.subgraph_list[user_id].nx_graph.successors(subtask_id)
        for succ in successors:
            # print(user_id, succ, self.check_prev_all_finished(user_id, succ))
            if succ in self.start_time[user_id] and self.check_prev_all_finished(user_id, succ):
                # 修复后继节点的 start_time 必须考虑所有前驱完成时间 + 设备实际可用时间
                # 计算所有前驱的最大完成时间
                max_pred_finish = 0
                for pred in self.subgraph_list[user_id].nx_graph.predecessors(succ):
                    if pred in self.finish_time[user_id]:
                        max_pred_finish = max(max_pred_finish, self.finish_time[user_id][pred])

                # 修复只记录拓扑依赖就绪时间（DAG Ready Time）
                # 不包含设备资源状态，避免"双通道能力打折"
                # 设备忙闲检查由 calculate_start_time 在真正执行时动态查询
                self.start_time[user_id][succ] = max_pred_finish
                # print("start_time[{}][{}] = {}".format(user_id, succ, self.start_time[user_id][succ]))

        if self.is_application_done(user_id):
            self.exit_time[user_id] = max(self.finish_time[user_id].values())#应用完成时间是所有子任务完成时间的最大值
            # print("finish_time[{}]: {}".format(user_id,self.finish_time[user_id].values()))
            # print("exit_time[{}]: {}".format(user_id, self.exit_time[user_id]))
            # input()
            self.application_finished.add(user_id)

            # 修复应用级超时统计（完成但晚于deadline也算超时）
            deadline_abs = self.enter_time[user_id] + self.get_app_deadline_slot(user_id) * para["slot_interval"]
            # 修复：使用 epsilon，Finish==Deadline 不应算超时
            EPS = 1e-6
            if self.enter_time[user_id] != float("inf") and self.exit_time[user_id] > deadline_abs + EPS:
                self.application_timeout_finished.add(user_id)

        delay = self.finish_time[user_id][subtask_id] - self.start_time[user_id][subtask_id]#该子任务的总延迟时间
        return energy, delay


    def is_application_done(self, user_id):
        for v in self.finish_time[user_id].values():
            if v == float('inf'):
                return False
        return True

    def is_done(self):
        # finishtime全都有值的时候就结束了
        # 这个函数不判断"是否还有新任务会到达"，只判断"当前所有应用是否完成"
        # 正确的 done 判断应该在主循环中：all_arrived_done and no_more_arrivals
        return len(self.application_finished) == self.user_num

    def finalize_episode(self, end_slot):
        """统计前调用：统一按 exit_time 与 deadline 判断应用超时，避免 inf"""
        end_time = end_slot * para["slot_interval"]

        # 【诊断】打印每个用户的 deadline 和完成情况
        if not hasattr(self, 'deadline_debug_count'):
            self.deadline_debug_count = 0

        for uid in range(self.user_num):
            if self.enter_time[uid] == float("inf"):
                continue  # 没到达的应用不计入

            # 使用各自应用的 deadline（可能有更短的）
            app_deadline_slot = self.get_app_deadline_slot(uid)
            deadline_abs = self.enter_time[uid] + app_deadline_slot * para["slot_interval"]

            # 【诊断】前 10 个用户的详细信息
            if self.deadline_debug_count < 10:
                finish_time = self.exit_time[uid] if self.exit_time[uid] != float('inf') else 0
                # 修复使用 epsilon 防止浮点误差导致误判
                is_timeout = finish_time > deadline_abs + self.TIMEOUT_EPSILON if self.is_application_done(uid) else True
                print(f"[Deadline Debug] UID={uid}, DeadSlot={app_deadline_slot}, DeadTime={deadline_abs:.3f}s, "
                      f"Finish={finish_time:.3f}s, Timeout={'YES' if is_timeout else 'NO'}")
                self.deadline_debug_count += 1

            # 1) 还没完成：一定超时，并给一个有限 exit_time
            if not self.is_application_done(uid):
                self.application_timeout_finished.add(uid)
                if self.exit_time[uid] == float("inf"):
                    self.exit_time[uid] = min(end_time, deadline_abs)
                continue

            # 2) 已完成：如果完成时间晚于 deadline，也算超时
            if self.exit_time[uid] == float("inf"):
                # 正常情况下完成时会写 exit_time，这里兜底
                self.exit_time[uid] = max(self.finish_time[uid].values())

            # 修复使用 epsilon 防止浮点误差导致误判
            if self.exit_time[uid] > deadline_abs + self.TIMEOUT_EPSILON:
                self.application_timeout_finished.add(uid)

    def check_timeouts(self, current_slot):
        """检查并标记超时未完成的应用，并清理僵尸任务"""
        current_time = current_slot * para["slot_interval"]

        for uid in range(self.user_num):
            if uid in self.application_finished or uid in self.application_timeout_finished:
                continue
            if self.enter_time[uid] == float("inf"):
                continue  # 还没到达就跳过

            # 应用级deadline（用于判断整个应用是否超时）
            app_deadline_abs = self.enter_time[uid] + self.get_app_deadline_slot(uid) * para["slot_interval"]

            # 子任务级deadline（用于判断单个子任务是否超时）
            task_deadline_abs = self.enter_time[uid] + self.get_task_deadline_slot(uid) * para["slot_interval"]

            # 检查子任务级超时（正在执行的任务）
            # 对于已分配但未完成的任务（finish_time != inf），检查是否超过子任务deadline
            for subtask_id in self.finish_time[uid]:
                if self.finish_time[uid][subtask_id] != float('inf'):
                    # 任务已分配，检查完成时间是否超过子任务deadline
                    if self.finish_time[uid][subtask_id] > task_deadline_abs + self.TIMEOUT_EPSILON:
                        # 子任务超时，标记该应用发生过子任务超时（过程指标）
                        if not hasattr(self, 'application_task_timeout'):
                            self.application_task_timeout = set()
                        self.application_task_timeout.add(uid)
                        # 可以在这里统计 overtime，但避免重复统计
                        # self.overtime += 1

            # 修复使用 epsilon 防止浮点误差导致误判
            # 应用级超时：整个应用未在应用级deadline内完成
            if current_time > app_deadline_abs + self.TIMEOUT_EPSILON and not self.is_application_done(uid):
                self.application_timeout_finished.add(uid)
                # 固定退出时间为应用级deadline
                self.exit_time[uid] = app_deadline_abs

                # 修复直接清空该用户的剩余任务，防止僵尸任务继续占用资源
                # 这能显著降低计算量，并让"超时"的概念更清晰
                self.rest_tasks[uid] = 0

                # 可选：统计子任务超时数（一次性加上剩余任务数）
                # task_unfinished = len([t for t in self.subgraph_list[uid].nx_graph.nodes
                #                        if self.finish_time[uid][t] == float('inf')])
                # if task_unfinished > 0:
                #     self.overtime += task_unfinished

    def reset(self):
        self.application_waiting = set([i for i in range(self.user_num)])
        self.application_started = set()  # 整个应用是否已经开始
        self.application_finished = set()  # 整个应用是否已经完成
        self.application_timeout_finished = set()  # 超时未完成的应用
        self.application_task_timeout = set()  # 发生过子任务超时的应用（清空）
        # 【额外修复】关闭超时次数打印，避免淹没日志
        # if self.overtime > 0:
        #     print("任务超时次数：", self.overtime)
        self.overtime = 0

        # 修复不再重新生成 tight deadline，复用 __init__ 中的配置
        # 这样 reset() 不会改变 deadline 配置，确保环境一致性
        # 【减少日志】只在第一次 reset 时打印（避免淹没其他日志）
        if not hasattr(self, '_reset_count'):
            self._reset_count = 0

        if self._reset_count == 0:
            # 第一次 reset：显示配置
            if hasattr(self, 'deadline_slot_per_user') and self.deadline_slot_per_user is not None:
                # 使用 deadline_slot_per_user 统计
                default_slot = para["deadline_slot"]
                tight_users = [uid for uid, slot in enumerate(self.deadline_slot_per_user)
                              if slot != default_slot]
                num_tight_users = len(tight_users)
                if tight_users:
                    tight_slots = [self.deadline_slot_per_user[uid] for uid in tight_users]
                    min_slot = min(tight_slots)
                    max_slot = max(tight_slots)
                else:
                    min_slot = max_slot = default_slot
                # 【静默】不输出 deadline 复用信息
                # print(f"[Deadline Reset] 复用初始配置: 总用户={self.user_num}, 紧deadline用户={num_tight_users}, "
                #       f"紧deadline范围={min_slot}-{max_slot} slot, 默认={default_slot} slot "
                #       f"(不再重新生成)")
            else:
                # 兼容旧代码：使用 app_deadline_slots 统计
                num_tight_users = len([uid for uid in range(self.user_num)
                                     if uid in self.app_deadline_slots])
                factors = [self.app_deadline_slots.get(uid, para["deadline_slot"]) / para["deadline_slot"]
                         for uid in range(self.user_num) if uid in self.app_deadline_slots]
                min_factor = min(factors) if factors else 1.0
                max_factor = max(factors) if factors else 1.0
                # 【静默】不输出 deadline 复用信息（旧格式）
                # print(f"[Deadline Reset] 复用初始配置(旧格式): 总用户={self.user_num}, 紧deadline用户={num_tight_users}, "
                #       f"紧deadline范围={int(para['deadline_slot']*min_factor)}-{int(para['deadline_slot']*max_factor)} slot "
                #       f"(不再重新生成)")
        else:
            # 后续 reset：不再打印
            pass

        self._reset_count += 1

        self.devices_exe_useful = [0 for i in range(self.user_num)]
        self.devices_upload_useful = [0 for i in range(self.user_num)]
        self.local_time = [0 for i in range(self.user_num)]  # 每个用户的本地执行时间记录
        self.upload_time = [0 for i in range(self.user_num)]  # 每个用户的上传时间记录
        self.rest_tasks = [len(sub.nx_graph.nodes) for sub in self.subgraph_list]
        self.edge_useful = []  # 小顶堆，长度始终为5 → #多边缘化后变为二维数组，表示每个边缘节点上的所有运行任务完成时间

        self.core_remaining_work = []
        for _ in range(para["edge_num"]):
            # self.edge_useful.append([])
            self.core_remaining_work.append([])
            self.edge_useful.append([0] * para["edgecore_limit"])
            # self.core_remaining_work.append([0] * para["edgecore_limit"])

        # 边缘端存储指标，考虑到每个边缘节点的存储情况不同，这些东西还是不放本文件了，但仍需要重置
        for edge_id in range(para["edge_num"]):
            target_edge = self.env.edges[edge_id]
            target_edge.task_count = 0  # 记录节点当前任务数量
            target_edge.total_task_size = 0  # 记录节点当前任务量(根据这个算存储需要空间和计算量)
            target_edge.total_difficulty = 0  # 累计任务难度（任务大小*复杂度系数）
            target_edge.used_storage = 0  # 已用存储空间
            target_edge.current_remainTime = 0  # 记录节点当前最快可用时间（最快任一核心空闲时间）
            target_edge.calculate_parameter = 1


        # 当前的总能耗
        self.energy = [0 for i in range(self.user_num)]
        self.de_wait_energy = [0 for i in range(self.user_num)]

        # 【TotalEnergy修复】重置总能耗累加器
        self.total_energy = 0.0

        self.upload_energy = [0 for _ in range(self.user_num)]  # 每个用户的上传能耗

        # 时间相关的记录
        self.enter_time = [float("inf") for i in range(self.user_num)]
        self.exit_time = [float("inf") for i in range(self.user_num)]
        self.init_times()



    def get_avg_results(self, only_successful=False, timeout_charge="2x_deadline"):#居然计算的时候加能耗，可能待改进
        """
        计算平均能耗和平均时延

        Args:
            only_successful: 如果为True，只计算成功（未超时）任务的时延
            timeout_charge: 超时/未完成应用的时延计入方式（仅 only_successful=False 时生效）
                - "2x_deadline"（默认，向后兼容）：算作 deadline_limit * 2
                - "deadline"（审稿人 R1-5 口径）：算作该应用的 (TD_i^max - TS_i)
                  = get_app_deadline_slot(uid) * slot_interval（即 per-app deadline 预算，相对值）
                  对应论文公式 D_i = min(D_i, TD_i^max - TS_i)
        """
        # 计算所有任务的空转时间
        for uid in range(self.user_num):
            #总时间需要减去上传时间和本地运行时间 #修改，因为可以同时本地计算和上传，取其最大值减
            # wait_time = self.exit_time[uid] - self.enter_time[uid] - max(self.local_time[uid], self.upload_time[uid])
            application_time = self.exit_time[uid] - self.enter_time[uid]


            # if wait_time < 0:
            #     print("wait_time < 0!")
            #     print("wait_time: ", wait_time)
            #     print("exit_time[",uid,"]: ", self.exit_time[uid])
            #     print("enter_time[",uid,"]: ", self.enter_time[uid])
            #     print("local_time[",uid,"]: ", self.local_time[uid])
            #     print("upload_time[",uid,"]: ", self.upload_time[uid])
            #     if wait_time < -0.1:
            #         input()

            #下面的代码是为了计算等待能耗，但是等待时间太难算了，要把所有终端设备进行本地计算和卸载的时间都记录下来然后对比，所以先不算等待能耗了
            # self.energy[uid] += (para["local_wait"] * application_time)
            # self.energy[uid] -= self.de_wait_energy[uid]  # 消去多加的等待能耗（三种计算的时间）
            #
            # running_time = self.local_time[uid] + self.upload_time[uid] #在终端设备同时只能运行一项任务时，应用总时间减去本地运行时间和上传时间就剩下等待时间了
            #                                                             #但若终端设备可以同时运行本地计算和上传任务，则不是这样
            # real_nowait = running_time * para["local_wait"]#实际等待能耗
            # if abs(self.de_wait_energy[uid] - real_nowait) > 0.001:
            #     print("wrong wait time!-------------------------------- ")
            #     print("de_wait_energy[uid] - real_nowait", self.de_wait_energy[uid] - real_nowait)
            #     print("local_time: ", self.local_time[uid])
            #     print("upload_time: ", self.upload_time[uid])
            #
            #     print("de_wait_energy: ", self.de_wait_energy[uid])
            #     print("real_dewait_energy: ", real_nowait)
            #     input()
            #
            # if self.energy[uid] < 0:
            #     print("self.energy[uid] < 0!")
            #     print("application_time: ", application_time)
            #     print("exit_time[",uid,"]: ", self.exit_time[uid])
            #     print("enter_time[",uid,"]: ", self.enter_time[uid])
            #     print("local_time[",uid,"]: ", self.local_time[uid])
            #     print("upload_time[",uid,"]: ", self.upload_time[uid])
            #     input()

            #这里好像是直接将设备运作的全部时间都算进来了 #修改，减去了本地运行时间和上传时间
        avg_energy = sum(self.energy)/len(self.energy)
        #上传至边缘端的能量在别的地方算了

        # [修改] 计算平均延迟
        valid_delays = []
        deadline_limit = para["deadline_slot"] * para["slot_interval"]

        for uid in range(self.user_num):
            # 获取该应用的 deadline
            app_deadline = self.enter_time[uid] + self.get_app_deadline_slot(uid) * para["slot_interval"]
            # per-app deadline 预算（相对值，秒）= (TD_i^max - TS_i)
            # 审稿人 R1-5 口径：超时应用按此值计入 D_all，而非 deadline_limit*2
            app_deadline_budget = self.get_app_deadline_slot(uid) * para["slot_interval"]

            if only_successful:
                # 只计算成功（未超时）任务的时延
                # 修复必须排除超时应用，否则会被误判为成功
                if (uid not in self.application_timeout_finished) and self.exit_time[uid] != float('inf'):
                    delay = self.exit_time[uid] - self.enter_time[uid]
                    valid_delays.append(delay)
            else:
                # 计算所有任务的时延，包括超时任务的惩罚
                # 1. 正常完成（未超时）
                if uid not in self.application_timeout_finished and self.exit_time[uid] != float('inf'):
                    delay = self.exit_time[uid] - self.enter_time[uid]
                # 2. 超时或未完成（给惩罚）
                else:
                    # 惩罚口径：
                    # - "2x_deadline"（默认，向后兼容）：算作 Deadline * 2
                    # - "deadline"（审稿人 R1-5 口径）：算作该应用 (TD_i^max - TS_i) = per-app deadline 预算
                    if timeout_charge == "deadline":
                        delay = app_deadline_budget
                    else:
                        delay = deadline_limit * 2
                valid_delays.append(delay)

        if len(valid_delays) > 0:
            avg_delay = sum(valid_delays) / len(valid_delays)
        else:
            # 全部超时时，返回最差值
            if timeout_charge == "deadline":
                avg_delay = deadline_limit
            else:
                avg_delay = deadline_limit * 2

        # print("Avg - Energy: {}, Delay: {}".format(avg_energy, avg_delay))
        return avg_energy, avg_delay

    def get_sum_energy(self):
        """
        计算总能耗（只计算实际产生的能耗）

        能耗组成：
        1. 本地计算能耗：κ × f² × cycle（只在本地计算时产生）
        2. 上传能耗：P_upload × T_upload（只在上传时产生）

        注意：
        - 不计算空闲等待能耗（设备空闲时的基础能耗）
        - 因为本地计算和上传可以并行，计算空闲时间很复杂
        - 只计算实际工作时的能耗，更简单准确
        """
        # 能耗已经在 update() 中累加：
        # - self.energy[user_id] += energy (本地计算能耗 + 上传能耗)
        # 这里不需要额外添加任何能耗
        return sum(self.energy)


    def get_delays(self):
        return [exi - ent for exi, ent in zip(self.exit_time, self.enter_time)]

    def get_finish_times(self):
        flattened_list = [element for sublist in self.finish_time for element in sublist.values()]
        return flattened_list

    def get_system_state(self):
        """
        返回系统的情况：[用户f+下一个用户设备可使用的时间+剩余任务]+边缘f+云f
        :return:
        """
        freq = [d.local_power for d in self.env.device_list]
        useful = self.devices_exe_useful
        rest = self.rest_tasks

        freq.extend([self.env.edge.edge_power * self.env.edge.calculate_parameter, self.env.cloud.cloud_power])
        useful.extend([self.edge_useful[0] if self.edge_useful else 0, len(self.edge_useful)])
        # useful.extend([self.edge_useful[edge_id][0] if self.edge_useful else 0, len(self.edge_useful[edge_id])]) #未修复edge_id，只是该函数未被调用
        rest.extend([0, 0])

        freq = normalize_list(freq)
        useful = normalize_list(useful)
        rest = normalize_list(rest)

        res = [[d, u, r] for d, u, r in zip(freq, useful, rest)]
        res = [item for sublist in res for item in sublist]

        return torch.Tensor([res])

    def get_action_mask(self, user_id, task_size, current_time=None):
        """
        [物理约束版本] 只屏蔽物理上不可能的动作，不屏蔽"可能超时"的

        Mask 的正确用法：
        - 只屏蔽"物理上绝对不可能"的动作（比如没电、存储满了、不可用）
        - 不屏蔽"可能会超时"的动作
        - RL 通过 Reward（负分）学会不去选那些会超时的动作

        物理约束包括：
        1. 本地设备 power <= 0（没电了）
        2. 云端不可用
        3. 边缘节点存储已满

        Args:
            user_id: 用户 ID
            task_size: 任务大小（Bytes）
            current_time: 当前时间（秒），可选

        Returns:
            mask: [num_actions] 1=有效, 0=无效（物理上不可能）
        """
        num_actions = para["edge_num"] + 2
        mask = torch.ones(num_actions)  # 默认全部有效

        # 1. 检查本地资源是否足够
        if self.env.device_list[user_id].local_power <= 0:
            mask[0] = 0  # 禁用本地

        # 2. 检查云端连接是否可用
        if not self.env.cloud.is_available():
            mask[1] = 0  # 禁用云端

        # 3. 检查各边缘节点是否可用
        for eid in range(para["edge_num"]):
            # 存储满了，禁用该边缘节点
            if self.env.edges[eid].used_storage >= self.env.edges[eid].max_storage:
                mask[eid + 2] = 0

        # 确保至少有一个动作可用
        if mask.sum() == 0:
            mask[0] = 1  # fallback 到本地（即使没电也要用）

        return mask

    def get_complete_infos(self, task_info, local_delay):
        """
        返回当前用户的完成率和距离deadline的距离
        :param task_info:
        :return:
        """
        user_id, subtask_id = task_info
        # 使用各自应用的 deadline（可能有更短的）
        deadline_time = self.get_app_deadline_slot(user_id) * para["slot_interval"]
        deadline = deadline_time + self.enter_time[user_id]  # 绝对 deadline
        # print("subtask_id:", subtask_id)
        # print("deadline_time:", deadline_time)
        # print("deadline:", deadline)
        # print("starttime:", self.enter_time[user_id])
        # complete_rate = self.rest_tasks[user_id]/len(self.subgraph_list[user_id].nx_graph.nodes)#不对吧，这是剩余任务比例了
        # complete_rate = 1 - self.rest_tasks[user_id] / len(self.subgraph_list[user_id].nx_graph.nodes)
        remain_rate = self.rest_tasks[user_id] / len(self.subgraph_list[user_id].nx_graph.nodes)
        complete_rate = 1 - remain_rate
        # print("complete_rate:", complete_rate)
        if remain_rate < 0:
            print("remain_rate < 0 !")
            input()

        current_time = self.finish_time[user_id][subtask_id]
        # print("current_time:", current_time)

        # time_remaining = max(0, deadline - current_time)  # 计算剩余时间
        time_remaining = deadline - current_time  # 计算剩余时间
        if deadline < current_time:
            self.overtime+=1
            # 记录该应用发生过子任务超时（用于更严格的 AppTO 统计）
            if hasattr(self, 'application_task_timeout'):
                self.application_task_timeout.add(user_id)
        # print("time_remaining:", time_remaining)
        # time_ratio = time_remaining / deadline  # 直接拿deadline做分母这不是任务越晚自动更小吗
        time_ratio = time_remaining / deadline_time  # 计算剩余时间比例
        # print("time_ratio:", time_ratio)

        # 使用指数函数作为示例，提前完成任务给予额外奖励，超时给予更大的惩罚
        # 这里只是一种示例，可以根据具体需求调整函数形式和参数
        if time_ratio > 0.2:  # 设置一个阈值，距离截止时间20%以上时给予额外奖励
            deadline_rate = time_ratio  # 提前完成任务给予额外奖励
        elif time_ratio < 0:  # 任务超时的情况给于大额惩罚
            print("任务超时！")
            deadline_rate = 10 * time_ratio - 10 # 超时给予更大的惩罚
        else:
            deadline_rate = 1 - (1 - time_ratio) * 2  # 接近超时给予更大的惩罚
        return complete_rate, deadline_rate

    # -----------------------------
    # Compatibility layer for Benchmark.py
    # -----------------------------

    def get_task_size_bytes(self, user_id: int, subtask_id: int) -> float:
        """
        安全获取任务大小（Bytes）。
        兼容 task_size 为 2D(list[list]), 1D(list), dict 等情况，
        并对越界做兜底，避免 Benchmark.get_actions 直接炸掉。
        """
        s = getattr(self, "task_size", None)
        if s is None:
            return 0.0

        try:
            # 2D: s[user][subtask]
            if isinstance(s, (list, tuple, np.ndarray)) and len(s) > 0 and isinstance(s[0], (list, tuple, np.ndarray)):
                uid = int(user_id)
                if uid < 0 or uid >= len(s):
                    uid = max(0, min(uid, len(s) - 1))

                row = s[uid]
                if row is None or len(row) == 0:
                    return 0.0

                sid = int(subtask_id)
                if 0 <= sid < len(row):
                    return float(row[sid])

                # 越界兜底：夹紧到合法范围
                sid = max(0, min(sid, len(row) - 1))
                return float(row[sid])

            # 1D: s[subtask]
            if isinstance(s, (list, tuple, np.ndarray)):
                sid = int(subtask_id)
                if 0 <= sid < len(s):
                    return float(s[sid])
                sid = max(0, min(sid, len(s) - 1))
                return float(s[sid])

            # dict: s[node_id] or s[subtask_id]
            if isinstance(s, dict):
                if subtask_id in s:
                    return float(s[subtask_id])
                # fallback: first value
                return float(next(iter(s.values())))
        except Exception:
            pass

        return 0.0

    def add_task(self, user_id, subtask_id, node_id=None, *args, **kwargs):
        """
        兼容旧版 Benchmark.step() 调用的接口：ts.add_task(uid, sid, node_id)
        这里把调用转发到实际的任务执行逻辑上。

        根据 node_id 的值，任务会被分配到：
        - 0: Local (本地执行)
        - 1: Cloud (云端执行)
        - >=2: Edge (边缘节点执行，node_id-2 是边缘节点索引)

        这个方法实际上执行任务分配和计算逻辑，不直接调用其他方法。
        """
        # 获取任务复杂度索引（从 env 中获取）
        if hasattr(self.env, 'task_complex_index'):
            task_complex_index = self.env.task_complex_index
        else:
            # 默认使用第一个复杂度系数
            task_complex_index = 0

        # 获取任务大小
        task_size = self.get_task_size_bytes(user_id, subtask_id)
        if task_size == 0:
            # 如果获取失败，使用默认值
            task_size = 200000.0

        task_info = (user_id, subtask_id)

        if node_id == 0:
            # Local: 本地执行
            from Environment import computation
            f_local = float(self.env.device_list[user_id].local_power)
            upload_e, upload_d = 0.0, 0.0
            exec_e, exec_d = computation.execute_consumption(task_size, f_local, task_complex_index, "l")
            e, d = self.update(task_info, exec_e, exec_d, local=True, upload_duration=0, exec_duration=exec_d)
            return e, d

        elif node_id == 1:
            # Cloud: 云端执行
            from Environment import computation
            # 先上传到最近的边缘
            nearest_edge = int(np.argmin(self.env.device_list[user_id].edge_distances))
            distance = float(self.env.device_list[user_id].edge_distances[nearest_edge])
            bandwidth = para["uplink_range"][0]  # 使用乐观估计
            upload_e1, upload_d1 = computation.upload_consumption([task_size, distance, bandwidth], 1, "e")
            upload_e2, upload_d2 = computation.upload_consumption(task_size, 1, "c")
            # 云端执行
            fc = float(self.env.cloud.cloud_power)
            exec_e, exec_d = computation.execute_consumption(task_size, fc, task_complex_index, "c")
            
            # 修复终端设备真正占用无线传输的时间只应该是 upload_d1
            # upload_d2 是 edge->cloud 的有线传输，不应该阻塞终端的 devices_upload_useful
            total_energy = upload_e1 + upload_e2 + exec_e
            total_delay = upload_d1 + upload_d2 + exec_d
            self.update(task_info, total_energy, total_delay, local=False,
                      upload_duration=upload_d1, exec_duration=exec_d)
            
            # 返回真实的能耗和延迟
            return total_energy, total_delay

        elif node_id >= 2:
            # Edge: 边缘节点执行
            edge_id = node_id - 2
            from Environment import computation
            distance = float(self.env.device_list[user_id].edge_distances[edge_id])
            bandwidth = para["uplink_range"][0]  # 使用乐观估计
            upload_e, upload_d = computation.upload_consumption([task_size, distance, bandwidth], 1, "e")
            f_edge = self.env.edges[edge_id].edge_power * self.env.edges[edge_id].calculate_parameter
            exec_e, exec_d = computation.execute_consumption(task_size, f_edge, task_complex_index, "e")
            e, d, remain = self.edge_update(task_info, task_size, (upload_e, upload_d), (exec_e, exec_d), edge_id)
            return e, d

        else:
            # 无效的 node_id，默认执行到本地
            from Environment import computation
            f_local = float(self.env.device_list[user_id].local_power)
            exec_e, exec_d = computation.execute_consumption(task_size, f_local, task_complex_index, "l")
            e, d = self.update(task_info, exec_e, exec_d, local=True, upload_duration=0, exec_duration=exec_d)
            return e, d
