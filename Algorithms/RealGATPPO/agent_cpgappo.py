# -*- coding: utf-8 -*-
"""
CPGAPPO main agent: PPO with dual-GAT encoder + CP-aware guided cross-entropy.

English
-------
This module implements the core agent of CPGAPPO (Constraint-Preserving Guide-Actor
PPO with dual GAT) for edge-cloud task offloading. It contains:

  - compute_guide_scores(): baseline per-action cost estimate (local/cloud/edge),
    used by the ablation variant `noguidece`.
  - compute_guide_scores_cp(): the CP-aware guide score used by the main model
    `CPGAPPO`. Builds on `compute_guide_scores` and adds a downstream-risk penalty
    proportional to the critical-path (CP) depth of the current subtask. A subtask
    deep on the CP gets a larger penalty on high-latency actions, steering the
    policy toward safer (lower-latency) placements. The penalty uses a 1.25x
    overflow coefficient (paper Eq.25) and a 0.45 CP-remain coefficient.
  - GAT_PPO_Agent_CPGAPPO: PPO agent whose loss augments the standard clipped
    surrogate + value loss + entropy bonus with a guided cross-entropy term
    `lambda_guide * CE(policy_logits, guide_action)`. Optional auxiliary AppTimeout
    prediction head (BCE) is enabled via use_appto_aux (used by `noappcredit` to
    study the auxiliary head in isolation).

Action space: 0=Local, 1=Cloud, 2..edge_num+1=Edge node e_id (1-based offset).
State: PyG `Data` (DAG node features, forward+backward GAT) + 10-d global
queue vector. The guide action = argmin(guide_scores) within the valid mask.

中文
----
本模块是 CPGAPPO 主算法的 agent 实现, 包含:
  - compute_guide_scores(): 基础 per-action 代价估计 (本地/云/边), 供消融变体 noguidece 使用。
  - compute_guide_scores_cp(): CPGAPPO 使用的 CP 感知 guide 分数, 在基础代价上叠加与
    关键路径深度成正比的下游超时风险惩罚, 1.25x 溢出系数 (论文 Eq.25), 0.45 CP-remain 系数。
  - GAT_PPO_Agent_CPGAPPO: PPO agent, 损失 = clipped surrogate + value + entropy
    + lambda_guide * CE(logits, guide_action); 可选 AppTimeout 辅助头 (BCE, use_appto_aux)。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from torch_geometric.data import Batch
from torch.distributions import Categorical
from .cpgappo_core import CPGAPPOLateFusionActorCritic


def compute_guide_scores(ts, uid, task_tuple, task_complex_index, now):
    """Baseline per-action cost estimate (local/cloud/edge), no CP penalty.

    Used by the `noguidece` ablation variant. Returns a cost vector over the
    action space (lower = better). Action layout: 0=Local, 1=Cloud,
    2..edge_num+1=Edge. Each entry is (queue_wait + upload_time + exec_time).
    基础 guide 分数, 不含 CP 惩罚, 供 noguidece 消融使用。
    """
    from utils.constant import para
    from Environment import computation
    import numpy as np

    edge_num = para["edge_num"]
    action_dim = edge_num + 2
    guide_scores = np.full(action_dim, 1e6, dtype=np.float32)
    valid_mask = np.ones(action_dim, dtype=bool)

    env = ts.env
    device = env.device_list[uid]
    task_size_bytes = float(ts.get_task_size_bytes(uid, task_tuple[1]))
    local_freq = device.local_power
    upload_bw = device.local_bw if hasattr(device, 'local_bw') else np.mean(para["uplink_range"])

    _, t_local = computation.execute_consumption(
        task_size_bytes, local_freq, task_complex_index, "l",
        local_wait=device.local_wait)
    local_queue = max(0.0, float(ts.devices_exe_useful[uid]) - now)
    guide_scores[0] = local_queue + t_local

    cloud_freq = env.cloud.cloud_power
    _, t_cloud_exec = computation.execute_consumption(
        task_size_bytes, cloud_freq, task_complex_index, "c")
    nearest_edge = int(np.argmin(device.edge_distances))
    nearest_dist = float(device.edge_distances[nearest_edge])
    _, t_cloud_upload_ue = computation.upload_consumption(
        [task_size_bytes, nearest_dist, upload_bw], 0, "e",
        local_trans=device.local_trans if hasattr(device, 'local_trans') else para["local_trans"],
        local_wait=device.local_wait)
    _, t_cloud_upload_ec = computation.upload_consumption(
        task_size_bytes, 0, "c",
        local_trans=device.local_trans if hasattr(device, 'local_trans') else para["local_trans"],
        local_wait=device.local_wait)
    upload_queue = max(0.0, float(ts.devices_upload_useful[uid]) - now)
    guide_scores[1] = upload_queue + t_cloud_upload_ue + t_cloud_upload_ec + t_cloud_exec

    for eid in range(edge_num):
        action_idx = eid + 2
        dist = device.edge_distances[eid]
        if dist == float('inf') or dist > para["edge_radius"]:
            guide_scores[action_idx] = 1e6
            valid_mask[action_idx] = False
            continue
        edge_freq = float(env.edges[eid].edge_power * env.edges[eid].calculate_parameter)
        _, t_edge_exec = computation.execute_consumption(
            task_size_bytes, edge_freq, task_complex_index, "e")
        _, t_edge_upload = computation.upload_consumption(
            [task_size_bytes, dist, upload_bw], 0, "e",
            local_trans=device.local_trans if hasattr(device, 'local_trans') else para["local_trans"],
            local_wait=device.local_wait)
        edge_queue = 0.0
        if eid < len(ts.core_remaining_work) and ts.core_remaining_work[eid]:
            total_work = sum(ts.core_remaining_work[eid])
            edge_queue = total_work / max(1e-9, edge_freq) if edge_freq > 0 else 1e6
        guide_scores[action_idx] = t_edge_upload + edge_queue + t_edge_exec

    return guide_scores, valid_mask


def compute_guide_scores_cp(ts, uid, task_tuple, task_complex_index, now):
    """CP-aware guide scores: original guide + downstream risk penalty based on critical path depth."""
    from utils.constant import para
    from Environment import computation
    import numpy as np

    edge_num = para["edge_num"]
    action_dim = edge_num + 2  # 0=Local, 1=Cloud, 2..9=Edge
    guide_scores = np.full(action_dim, 1e6, dtype=np.float32)
    valid_mask = np.ones(action_dim, dtype=bool)

    env = ts.env
    device = env.device_list[uid]
    task_size_bytes = float(ts.get_task_size_bytes(uid, task_tuple[1]))
    local_freq = device.local_power
    upload_bw = device.local_bw if hasattr(device, 'local_bw') else np.mean(para["uplink_range"])

    # --- Action 0: Local ---
    _, t_local = computation.execute_consumption(
        task_size_bytes, local_freq, task_complex_index, "l",
        local_wait=device.local_wait)
    local_queue = max(0.0, float(ts.devices_exe_useful[uid]) - now)
    guide_scores[0] = local_queue + t_local

    # --- Action 1: Cloud ---
    cloud_freq = env.cloud.cloud_power
    _, t_cloud_exec = computation.execute_consumption(
        task_size_bytes, cloud_freq, task_complex_index, "c")
    nearest_edge = int(np.argmin(device.edge_distances))
    nearest_dist = float(device.edge_distances[nearest_edge])
    _, t_cloud_upload_ue = computation.upload_consumption(
        [task_size_bytes, nearest_dist, upload_bw], 0, "e",
        local_trans=device.local_trans if hasattr(device, 'local_trans') else para["local_trans"],
        local_wait=device.local_wait)
    _, t_cloud_upload_ec = computation.upload_consumption(
        task_size_bytes, 0, "c",
        local_trans=device.local_trans if hasattr(device, 'local_trans') else para["local_trans"],
        local_wait=device.local_wait)
    upload_queue = max(0.0, float(ts.devices_upload_useful[uid]) - now)
    guide_scores[1] = upload_queue + t_cloud_upload_ue + t_cloud_upload_ec + t_cloud_exec

    # --- Actions 2..edge_num+1: Edge nodes ---
    for eid in range(edge_num):
        action_idx = eid + 2
        dist = device.edge_distances[eid]
        if dist == float('inf') or dist > para["edge_radius"]:
            guide_scores[action_idx] = 1e6
            valid_mask[action_idx] = False
            continue
        edge_freq = float(env.edges[eid].edge_power * env.edges[eid].calculate_parameter)
        _, t_edge_exec = computation.execute_consumption(
            task_size_bytes, edge_freq, task_complex_index, "e")
        _, t_edge_upload = computation.upload_consumption(
            [task_size_bytes, dist, upload_bw], 0, "e",
            local_trans=device.local_trans if hasattr(device, 'local_trans') else para["local_trans"],
            local_wait=device.local_wait)
        edge_queue = 0.0
        if eid < len(ts.core_remaining_work) and ts.core_remaining_work[eid]:
            total_work = sum(ts.core_remaining_work[eid])
            edge_queue = total_work / max(1e-9, edge_freq) if edge_freq > 0 else 1e6
        guide_scores[action_idx] = t_edge_upload + edge_queue + t_edge_exec

    # === CP-aware downstream risk penalty ===
    sid = int(task_tuple[1])
    g = ts.subgraph_list[uid].nx_graph

    try:
        succs = list(g.successors(sid))
    except Exception:
        succs = []

    if not succs:
        cp_depth = 0
    else:
        def _cp_depth_from(node, memo):
            if node in memo:
                return memo[node]
            try:
                node_succs = list(g.successors(node))
            except Exception:
                memo[node] = 0
                return 0
            if not node_succs:
                memo[node] = 0
                return 0
            depth = 1 + max(_cp_depth_from(s, memo) for s in node_succs)
            memo[node] = depth
            return depth

        cp_depth = _cp_depth_from(sid, {})

    app_deadline_slot = ts.get_app_deadline_slot(uid) if hasattr(ts, "get_app_deadline_slot") else para["deadline_slot"]
    if uid < len(ts.enter_time) and ts.enter_time[uid] != float("inf"):
        enter_time = float(ts.enter_time[uid])
    else:
        enter_time = float(now)
    app_deadline_abs = enter_time + app_deadline_slot * para["slot_interval"]

    cp_factor = min(1.0, cp_depth / 4.0)

    for a in range(action_dim):
        if not valid_mask[a]:
            continue
        immediate_cost = float(guide_scores[a])
        pred_finish = now + immediate_cost
        time_left_after = app_deadline_abs - pred_finish
        cp_remain_est = cp_factor * app_deadline_slot * para["slot_interval"] * 0.45
        downstream_overflow = max(0.0, cp_remain_est - time_left_after)
        guide_scores[a] = immediate_cost + 1.25 * downstream_overflow

    return guide_scores, valid_mask


class GAT_PPO_Agent_CPGAPPO:
    def __init__(self, node_dim=27, global_dim=10, action_dim=10, device='cpu',
                 lr=3e-4, gamma=0.99, lmbda=0.95, eps_clip=0.2, K_epochs=4,
                 entropy_coef=0.01, lambda_guide=0.2, use_backward=True,
                 use_appto_aux=False, aux_loss_coef=0.1):
        self.device = device
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.entropy_coef = entropy_coef
        self.lambda_guide = lambda_guide
        self.use_appto_aux = use_appto_aux  # [v2 main] 辅助 AppTimeout 预测头
        self.aux_loss_coef = aux_loss_coef  # [v2 main] BCE 辅助 loss 权重

        self.policy = CPGAPPOLateFusionActorCritic(node_dim, global_dim, 64, action_dim,
                                              use_backward=use_backward,
                                              use_appto_aux=use_appto_aux).to(device)
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=lr, weight_decay=1e-5)
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')
        self.memory = []
        self.inference_times = []

    def put_data(self, item):
        self.memory.append(item)

    def clear_memory(self):
        self.memory = []

    def take_action(self, state_data, global_feat, action_mask=None, deterministic=False):
        start_time = time.time()
        self.policy.eval()
        with torch.no_grad():
            batch = Batch.from_data_list([state_data]).to(self.device)
            g_feat = global_feat.to(self.device)
            logits, _ = self.policy(batch, g_feat)
            logits = logits.squeeze(0)

            if action_mask is not None:
                mask_t = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device)
                neg_inf = torch.finfo(logits.dtype).min
                logits = logits.masked_fill(~mask_t, neg_inf)

            probs = torch.softmax(logits, dim=-1)
            dist = Categorical(probs)

            if deterministic:
                action = torch.argmax(probs).item()
            else:
                action = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action, device=self.device)).item()

        self.inference_times.append((time.time() - start_time) * 1000)
        return action, log_prob

    def update(self):
        if not self.memory:
            return 0.0
        states = [x[0] for x in self.memory]
        global_feats = torch.stack([x[1] for x in self.memory]).to(self.device)
        actions = torch.tensor([x[2] for x in self.memory], dtype=torch.long).to(self.device)
        rewards = [x[3] for x in self.memory]
        old_log_probs = torch.tensor([x[4] for x in self.memory], dtype=torch.float32).to(self.device)
        masks = [x[6] for x in self.memory]
        guide_actions = [x[7] for x in self.memory]

        batch_size = 64
        values = []
        self.policy.eval()
        with torch.no_grad():
            for i in range(0, len(states), batch_size):
                b_states = Batch.from_data_list(states[i:i + batch_size]).to(self.device)
                b_gfeats = global_feats[i:i + batch_size]
                _, v = self.policy(b_states, b_gfeats)
                values.append(v.squeeze(-1).cpu())
        values = torch.cat(values)

        returns, advantages, gae = [], [], 0
        for i in reversed(range(len(rewards))):
            next_val = 0 if i == len(rewards) - 1 else values[i + 1]
            if self.memory[i][5]:
                gae = 0
                next_val = 0
            delta = rewards[i] + self.gamma * next_val - values[i]
            gae = delta + self.gamma * self.lmbda * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        if advantages.std() > 1e-6:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        guide_actions_t = torch.tensor(guide_actions, dtype=torch.long).to(self.device)

        self.policy.train()
        total_loss = 0
        T = len(states)
        perm = np.random.permutation(T)

        for _ in range(self.K_epochs):
            for i in range(0, T, batch_size):
                idx = perm[i: i + batch_size]
                b_states = Batch.from_data_list([states[j] for j in idx]).to(self.device)
                b_gfeats = global_feats[idx]
                b_actions = actions[idx]
                b_returns = returns[idx]
                b_adv = advantages[idx]
                b_old_lp = old_log_probs[idx]
                b_masks = torch.as_tensor(np.stack([masks[j] for j in idx]), dtype=torch.bool, device=self.device)
                b_guide = guide_actions_t[idx]

                logits, v_pred = self.policy(b_states, b_gfeats) if not self.use_appto_aux else (None, None)
                if self.use_appto_aux:
                    logits, v_pred, appto_logit = self.policy(b_states, b_gfeats, return_aux=True)
                v_pred = v_pred.squeeze(-1)

                neg_inf = torch.finfo(logits.dtype).min
                masked_logits = logits.masked_fill(~b_masks, neg_inf)
                probs = torch.softmax(masked_logits, dim=-1)
                dist = Categorical(probs)

                new_log_probs = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - b_old_lp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * b_adv

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = self.mse_loss(v_pred, b_returns)
                
                guide_ce = self.ce_loss(masked_logits, b_guide)
                
                loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy + self.lambda_guide * guide_ce
                # [v2 main] 辅助 AppTimeout 预测 BCE loss
                if self.use_appto_aux:
                    b_appto_labels = torch.tensor(
                        [float(self.memory[j][8]) for j in idx],
                        dtype=torch.float32, device=self.device
                    )
                    aux_loss = F.binary_cross_entropy_with_logits(appto_logit, b_appto_labels)
                    loss = loss + self.aux_loss_coef * aux_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                total_loss += loss.item()
        return total_loss

    def print_inference_stats(self):
        if not self.inference_times:
            print("[Agent] No inference records")
            return
        arr = np.array(self.inference_times)
        print(f"[Agent] Inference stats ({len(arr)} calls):")
        print(f"  - avg: {arr.mean():.4f} ms")
        print(f"  - median: {np.median(arr):.4f} ms")
        print(f"  - P95: {np.percentile(arr, 95):.4f} ms")
