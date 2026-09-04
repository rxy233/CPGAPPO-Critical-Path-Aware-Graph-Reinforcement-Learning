"""
GAT-PPO R1 状态编码器 — 最小表示修补版

修补目标：
1. 清除 B0 中 47 维零填充
2. 加入 edge/cloud/local runtime placement 特征
3. 让模型显式看到 Local vs Best-Edge vs Cloud 的延迟/能耗差异

特征维度: 17
  [0]  size_norm          - 任务大小归一化
  [1]  comp_norm          - 计算复杂度归一化
  [2]  time_left          - 距deadline剩余时间
  [3]  local_queue        - 本地CPU排队时间
  [4]  local_upload_queue - 本地上传通道排队时间
  [5]  est_local_delay    - 本地执行预估延迟
  [6]  est_cloud_delay    - 云端执行预估延迟（含传输）
  [7]  best_edge_delay    - 最佳边缘节点预估延迟（含传输）
  [8]  edge_cloud_delay_gap - (best_edge - cloud) 延迟差，>0说明edge更差
  [9]  edge_local_delay_gap  - (best_edge - local) 延迟差，>0说明edge更差
  [10] best_edge_queue_load  - 最佳edge排队负载
  [11] best_edge_storage_frac - 最佳edge存储使用比例
  [12] reachable_edge_count   - 可达(范围内)edge数量
  [13] best_edge_idx_norm     - 最佳edge编号归一化
  [14] local_power_norm       - 本地算力归一化
  [15] cloud_local_delay_gap  - (cloud - local) 延迟差，<0说明cloud更快
  [16] is_cur                 - 是否当前候选任务
"""

import torch
from torch_geometric.data import Data
import numpy as np
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from utils.constant import para
from Environment import computation

R1_FEATURE_DIM = 27


def _estimate_placement_costs(ts, uid, task_size_bytes, task_complex_index, now):
    """估算 local / cloud / best_edge 的延迟和能耗"""
    env = ts.env
    device = env.device_list[uid]

    # --- Local ---
    local_freq = device.local_power
    _, t_local = computation.execute_consumption(task_size_bytes, local_freq, task_complex_index, "l",
                                                 local_wait=device.local_wait)
    local_queue = max(0.0, float(ts.devices_exe_useful[uid]) - now)
    est_local_delay = local_queue + t_local

    # Get upload bandwidth for this device
    upload_bw = device.local_bw if hasattr(device, 'local_bw') else np.mean(para["uplink_range"])

    # --- Cloud (with UE→edge first hop) ---
    cloud_freq = env.cloud.cloud_power
    _, t_cloud_exec = computation.execute_consumption(task_size_bytes, cloud_freq, task_complex_index, "c")

    nearest_edge = int(np.argmin(device.edge_distances))
    nearest_dist = float(device.edge_distances[nearest_edge])

    # UE -> nearest edge (wireless uplink)
    _, t_cloud_upload_ue = computation.upload_consumption(
        [task_size_bytes, nearest_dist, upload_bw], 0, "e",
        local_trans=device.local_trans if hasattr(device, 'local_trans') else para["local_trans"],
        local_wait=device.local_wait
    )

    # edge -> cloud (wired)
    _, t_cloud_upload_ec = computation.upload_consumption(
        task_size_bytes, 0, "c",
        local_trans=device.local_trans if hasattr(device, 'local_trans') else para["local_trans"],
        local_wait=device.local_wait
    )

    upload_queue = max(0.0, float(ts.devices_upload_useful[uid]) - now)
    est_cloud_delay = upload_queue + t_cloud_upload_ue + t_cloud_upload_ec + t_cloud_exec

    # --- Edge (iterate all, find best) ---
    edge_num = para["edge_num"]
    best_edge_delay = float('inf')
    best_edge_eid = -1
    best_edge_queue = float('inf')
    best_edge_storage_frac = 1.0

    for eid in range(edge_num):
        dist = device.edge_distances[eid]
        if dist == float('inf') or dist > para["edge_radius"]:
            continue

        edge_freq = float(env.edges[eid].edge_power * env.edges[eid].calculate_parameter)
        _, t_edge_exec = computation.execute_consumption(task_size_bytes, edge_freq, task_complex_index, "e")

        # Upload delay to edge
        _, t_edge_upload = computation.upload_consumption(
            [task_size_bytes, dist, upload_bw], 0, "e",
            local_trans=device.local_trans if hasattr(device, 'local_trans') else para["local_trans"],
            local_wait=device.local_wait
        )

        # Edge queue time
        edge_queue = 0.0
        if eid < len(ts.core_remaining_work) and ts.core_remaining_work[eid]:
            # Estimate queue time from remaining work
            total_work = sum(ts.core_remaining_work[eid])
            edge_queue = total_work / (edge_freq * para["slot_interval"]) * para["slot_interval"]
            # Simpler: just use the min heap top as earliest available
            if ts.core_remaining_work[eid]:
                earliest_work = ts.core_remaining_work[eid][0]
                edge_queue = earliest_work / (edge_freq * para["slot_interval"]) * para["slot_interval"]

        total_edge_delay = t_edge_upload + edge_queue + t_edge_exec

        if total_edge_delay < best_edge_delay:
            best_edge_delay = total_edge_delay
            best_edge_eid = eid
            # Queue load: number of items in heap / core limit
            edge_core = para.get("edgecore_limit", 4)
            if eid < len(ts.core_remaining_work):
                best_edge_queue = len(ts.core_remaining_work[eid]) / max(1, edge_core)
            # Storage fraction
            ms = env.edges[eid].max_storage if hasattr(env.edges[eid], 'max_storage') else 1.0
            us = env.edges[eid].used_storage if hasattr(env.edges[eid], 'used_storage') else 0.0
            best_edge_storage_frac = min(1.0, float(us) / max(1.0, float(ms)))

    # Count reachable edges
    reachable = 0
    for eid in range(edge_num):
        dist = device.edge_distances[eid]
        if dist != float('inf') and dist <= para["edge_radius"]:
            reachable += 1

    return {
        "est_local_delay": est_local_delay,
        "est_cloud_delay": est_cloud_delay,
        "best_edge_delay": best_edge_delay if best_edge_eid >= 0 else est_cloud_delay,
        "best_edge_eid": best_edge_eid,
        "best_edge_queue": best_edge_queue if best_edge_eid >= 0 else 1.0,
        "best_edge_storage_frac": best_edge_storage_frac,
        "reachable_edge_count": reachable,
        "edge_num": edge_num,
        "local_power": local_freq,
    }


def _get_route_structure_features(ts, uid, now):
    """
    v2: 全局路由结构特征（共享给所有节点）
    返回 10 维:
      [0] route_ratio_local
      [1] route_ratio_edge
      [2] route_ratio_cloud
      [3] recent_ratio_local
      [4] recent_ratio_edge
      [5] recent_ratio_cloud
      [6] avg_edge_queue_pressure
      [7] cloud_pressure_proxy
      [8] ready_local_feasible_ratio
      [9] ready_edge_feasible_ratio
    """
    stats = getattr(ts, "route_stats", None)
    if stats is None:
        return [0.0] * 10

    total = max(1, stats.get("local", 0) + stats.get("edge", 0) + stats.get("cloud", 0))
    route_ratio_local = stats.get("local", 0) / total
    route_ratio_edge = stats.get("edge", 0) / total
    route_ratio_cloud = stats.get("cloud", 0) / total

    recent = stats.get("recent", [])
    if len(recent) == 0:
        recent_ratio_local = 0.0
        recent_ratio_edge = 0.0
        recent_ratio_cloud = 0.0
    else:
        rtot = len(recent)
        recent_ratio_local = sum(1 for a in recent if a == 0) / rtot
        recent_ratio_cloud = sum(1 for a in recent if a == 1) / rtot
        recent_ratio_edge = sum(1 for a in recent if a >= 2) / rtot

    edge_pressures = []
    if hasattr(ts, "core_remaining_work"):
        for eid in range(min(len(ts.core_remaining_work), para["edge_num"])):
            try:
                edge_pressures.append(len(ts.core_remaining_work[eid]))
            except Exception:
                edge_pressures.append(0.0)
    avg_edge_queue_pressure = float(np.mean(edge_pressures)) / 10.0 if edge_pressures else 0.0

    cloud_pressure_proxy = route_ratio_cloud

    ready_local_feasible_ratio = max(0.0, 1.0 - route_ratio_cloud)
    ready_edge_feasible_ratio = max(0.0, 1.0 - route_ratio_local)

    return [
        route_ratio_local,
        route_ratio_edge,
        route_ratio_cloud,
        recent_ratio_local,
        recent_ratio_edge,
        recent_ratio_cloud,
        avg_edge_queue_pressure,
        cloud_pressure_proxy,
        ready_local_feasible_ratio,
        ready_edge_feasible_ratio,
    ]


def encode_dag_gat_ppo_state_r1(ts, task_tuple, slot=None, now_time=None, task_complex_index=0):
    """
    R1 编码器: 16维有信息特征

    Args:
        ts: TaskScheduler
        task_tuple: (uid, sid)
        slot: 当前slot
        now_time: 当前绝对时间

    Returns:
        (Data, mask_bin): PyG Data (node_dim=16) and action mask
    """
    uid, sid = task_tuple
    env = ts.env

    if now_time is None:
        if slot is None:
            now = 0.0
        else:
            now = float(slot) * para["slot_interval"]
    else:
        now = float(now_time)

    # Get DAG structure
    g = ts.subgraph_list[uid].nx_graph
    nodes = list(g.nodes())
    num_nodes = len(nodes)

    if not nodes:
        x = torch.zeros((0, R1_FEATURE_DIM), dtype=torch.float32)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        mask_bin = []
        return Data(x=x, edge_index=edge_index), mask_bin

    node_map = {nid: i for i, nid in enumerate(nodes)}

    edge_index_list = []
    for src, dst in g.edges():
        if src in node_map and dst in node_map:
            edge_index_list.append([node_map[src], node_map[dst]])

    if not edge_index_list:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()

    # Task features
    task_complex_val = para["task_complex"]
    if isinstance(task_complex_val, (list, tuple)):
        safe_idx = max(0, min(int(task_complex_index), len(task_complex_val) - 1))
        comp_val = float(task_complex_val[safe_idx])
    else:
        safe_idx = 0
        comp_val = float(task_complex_val)

    task_size_bytes = float(ts.get_task_size_bytes(uid, sid))
    task_size = task_size_bytes
    task_complex_index = safe_idx

    # Deadline
    deadline_abs = float(ts.enter_time[uid]) + ts.get_app_deadline_slot(uid) * para["slot_interval"]
    time_left = max(0, deadline_abs - now)

    # Queue info
    local_queue = max(0, float(ts.devices_exe_useful[uid]) - now)
    upload_queue = max(0, float(ts.devices_upload_useful[uid]) - now)

    # Placement cost estimation
    costs = _estimate_placement_costs(ts, uid, task_size_bytes, task_complex_index, now)

    # Normalize features
    size_norm = task_size_bytes / 1e6
    comp_norm = comp_val / 100.0
    time_left_norm = time_left / 10.0
    local_queue_norm = local_queue / 5.0
    upload_queue_norm = upload_queue / 5.0
    est_local_norm = costs["est_local_delay"] / 5.0
    est_cloud_norm = costs["est_cloud_delay"] / 5.0
    best_edge_delay_norm = min(costs["best_edge_delay"], 10.0) / 5.0
    edge_cloud_gap = (costs["best_edge_delay"] - costs["est_cloud_delay"]) / 5.0
    edge_local_gap = (costs["best_edge_delay"] - costs["est_local_delay"]) / 5.0
    cloud_local_gap = (costs["est_cloud_delay"] - costs["est_local_delay"]) / 5.0
    best_edge_queue = costs["best_edge_queue"]
    best_edge_stor = costs["best_edge_storage_frac"]
    reachable_count = costs["reachable_edge_count"] / max(1, costs["edge_num"])
    best_edge_idx = (costs["best_edge_eid"] / max(1, costs["edge_num"] - 1)) if costs["best_edge_eid"] >= 0 else 0.5
    local_power_norm = costs["local_power"] / 1e9

    route_struct_feat = _get_route_structure_features(ts, uid, now)

    base_feat = [
        size_norm, comp_norm, time_left_norm, local_queue_norm, upload_queue_norm,
        est_local_norm, est_cloud_norm, best_edge_delay_norm,
        edge_cloud_gap, edge_local_gap,
        best_edge_queue, best_edge_stor,
        reachable_count, best_edge_idx,
        local_power_norm, cloud_local_gap
    ] + route_struct_feat

    node_feats = []
    for i, node_id in enumerate(nodes):
        is_cur = 1.0 if node_id == sid else 0.0
        feat = list(base_feat) + [is_cur]
        assert len(feat) == R1_FEATURE_DIM, f"feat dim {len(feat)} != {R1_FEATURE_DIM}"
        node_feats.append(feat)

    x = torch.tensor(node_feats, dtype=torch.float32)

    # Mask (same as B0)
    mask = ts.get_action_mask(uid, task_size, now)
    mask_bin = [1 if m > 0.5 else 0 for m in mask]

    return Data(x=x, edge_index=edge_index), mask_bin
