# -*- coding: utf-8 -*-
"""
状态编码器 - 用于 DQN baseline
- DAG-DQN: DAG统计特征 + 当前节点特征 + 队列统计
- SATA-DRL: 粗全局向量状态
"""
import numpy as np
import torch


def _safe_get_node_index(node2idx, sid):
    """安全获取节点索引"""
    if node2idx is None:
        return None
    return node2idx.get(sid, None)


def encode_dag_dqn_state(state_data, node2idx, uid, sid, ts, env, para, now_time):
    """
    DAG-DQN 状态编码（适配当前环境）
    - DAG统计特征
    - 当前节点特征
    - 资源队列统计
    - 边缘节点距离

    Returns:
        np.ndarray [state_dim]
    """
    x = state_data.x  # torch.Tensor [N, F]
    N = x.size(0)
    E = state_data.edge_index.size(1) if hasattr(state_data, "edge_index") else 0

    # 当前节点索引
    cur_idx = _safe_get_node_index(node2idx, sid)
    if cur_idx is None:
        # 如果找不到，用零向量
        cur_feat = torch.zeros(min(8, x.size(1)))
        indeg = outdeg = 0.0
    else:
        # 取前8维特征（基础特征）
        cur_feat = x[cur_idx, :min(8, x.size(1))].detach().cpu()
        # 统计入度/出度
        ei = state_data.edge_index
        src, dst = ei[0], ei[1]
        indeg = float((dst == cur_idx).sum().item())
        outdeg = float((src == cur_idx).sum().item())

    # DAG结构统计特征
    dag_stats = np.array([
        float(N),                           # 节点数
        float(E),                           # 边数
        float(E) / max(1.0, float(N)),     # 稀疏度
        indeg,                             # 当前节点入度
        outdeg,                            # 当前节点出度
    ], dtype=np.float32)

    # 资源队列统计（截断+归一化到[0,1]）
    # Edge队列：修复remain_times 是绝对时间戳，必须减去 now_time
    edge_q = np.array([float(t) for t in ts.remain_times], dtype=np.float32)
    edge_q = np.maximum(0.0, edge_q - now_time)  # 剩余等待时间
    edge_q = np.clip(edge_q, 0.0, 5.0) / 5.0     # 归一化到[0,1]

    # Local队列：devices_exe_useful[uid] - now
    local_q = max(0.0, float(ts.devices_exe_useful[uid]) - now_time)
    # Upload队列：devices_upload_useful[uid] - now
    uplink_q = max(0.0, float(ts.devices_upload_useful[uid]) - now_time)

    local_q = np.clip(local_q, 0.0, 5.0) / 5.0
    uplink_q = np.clip(uplink_q, 0.0, 5.0) / 5.0

    # 边缘节点距离（归一化）
    dists = np.array([float(d) for d in env.device_list[uid].edge_distances], dtype=np.float32)
    dists = np.clip(dists / (dists.max() + 1e-6), 0.0, 1.0)

    # 拼接所有特征
    vec = np.concatenate([
        dag_stats,                 # 5维
        cur_feat.numpy().astype(np.float32),  # 8维（基础特征）
        np.array([local_q, uplink_q], dtype=np.float32),  # 2维
        edge_q,                    # edge_num维
        dists,                    # edge_num维
    ], axis=0)

    return vec


def encode_sata_state(uid, task_size_bytes, tasks_in_slot, ts, env, para, now_time):
    """
    SATA-DRL 状态编码 - 修正版
    修复必须包含每个 Edge 的独立状态，而不是求和。
    
    特征组成：
    - 任务特征（大小、当前slot任务数）
    - App 级特征（slack、剩余任务比例）
    - 设备队列（本地执行、上传）
    - 每个 Edge 的状态（等待时间、算力、距离、存储余量）
    
    Returns:
        np.ndarray [6 + 4*edge_num]
    """
    edge_num = para["edge_num"]

    # ==========================================
    # 1. 任务特征
    # ==========================================
    # 任务大小：用 log，避免尺度问题
    size_log = np.log1p(task_size_bytes) / np.log1p(5e6)  # 归一到[0,1]

    # 当前 slot 任务数归一化
    num_ready = np.clip(float(len(tasks_in_slot)) / 50.0, 0.0, 1.0)

    # ==========================================
    # 2. App 级特征（Slack + 剩余任务）
    # ==========================================
    if ts.enter_time[uid] == float("inf"):
        slack_norm = 1.0
        remain_rate = 1.0
    else:
        # 计算应用的绝对 deadline
        deadline_abs = ts.enter_time[uid] + ts.get_app_deadline_slot(uid) * para["slot_interval"]
        slack = deadline_abs - now_time  # 剩余时间（秒）
        
        # 归一化：用"典型deadline秒数"归一化
        denom = para["deadline_slot"] * para["slot_interval"]
        slack_norm = np.clip(slack / (denom + 1e-6), -1.0, 1.0)

        # 剩余任务比例
        total_tasks = len(ts.subgraph_list[uid].nx_graph.nodes)
        remain_rate = np.clip(ts.rest_tasks[uid] / max(1, total_tasks), 0.0, 1.0)

    # ==========================================
    # 3. 设备队列（本地执行、上传）
    # ==========================================
    local_q = np.clip(max(0.0, ts.devices_exe_useful[uid] - now_time) / 1.0, 0.0, 1.0)
    uplink_q = np.clip(max(0.0, ts.devices_upload_useful[uid] - now_time) / 1.0, 0.0, 1.0)

    # ==========================================
    # 4. 【关键修改】展开所有 Edge 的状态
    # ==========================================
    # 4.1 边缘等待时间：必须减去 now_time（remain_times 是绝对时间戳）
    edge_free_abs = np.array(ts.remain_times, dtype=np.float32)  # 绝对时间
    edge_wait = np.maximum(0.0, edge_free_abs - now_time)         # 剩余等待(秒)
    edge_wait = np.clip(edge_wait / 1.0, 0.0, 1.0)                # 1秒做尺度，可按实际调

    # 4.2 边缘算力（如果所有 Edge 算力一样，这个可以省略，但在异构网络中很重要）
    edge_cap = np.array([e.edge_power * e.calculate_parameter for e in env.edges], dtype=np.float32)
    edge_cap = edge_cap / (edge_cap.max() + 1e-6)

    # 4.3 边缘距离
    edge_dist = np.array(env.device_list[uid].edge_distances, dtype=np.float32)
    edge_dist = edge_dist / (edge_dist.max() + 1e-6)

    # 4.4 边缘存储余量
    edge_storage_free = np.array(
        [1.0 - (e.used_storage / (e.max_storage + 1e-6)) for e in env.edges],
        dtype=np.float32
    )
    edge_storage_free = np.clip(edge_storage_free, 0.0, 1.0)

    # ==========================================
    # 5. 拼接所有特征
    # ==========================================
    # 维度: 6 + 4*edge_num
    #   - 6: size_log, slack_norm, remain_rate, local_q, uplink_q, num_ready
    #   - 4*edge_num: edge_wait, edge_cap, edge_dist, edge_storage_free
    state_vec = np.concatenate([
        np.array([size_log, slack_norm, remain_rate, local_q, uplink_q, num_ready], dtype=np.float32),
        edge_wait,
        edge_cap,
        edge_dist,
        edge_storage_free
    ], axis=0).astype(np.float32)

    return state_vec
