# -*- coding: utf-8 -*-
"""
GAT_PPO_Agent: baseline PPO with a single forward-GAT encoder.

English
-------
This is the plain PPO baseline used as the `PPO` external baseline (no guide
score, no backward GAT, no CP penalty). It shares the DualGATEncoder with the
main model but drops the global-resource branch and the guided CE term. Used
both as a comparison point and as the historical agent class from which
GAT_PPO_Agent_CPGAPPO (agent_cpgappo.py) evolved. Inference-time
statistics and route-aware logit regularization (v2.1) are kept for ablation
diagnostics; the main model does not use route regularization.

中文
----
基础 GAT-PPO agent (正向 GAT + PPO), 作为外部基线 PPO 使用; 无 guide 分数、
无反向 GAT、无 CP 惩罚。保留推理耗时统计与 v2.1 路径正则化用于消融诊断。
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from torch_geometric.data import Batch, Data
from torch.distributions import Categorical
from .model import SharedActorCritic


class GAT_PPO_Agent:
    def __init__(self, node_dim, action_dim=3, device='cpu',
                 lr=3e-4, gamma=0.99, lmbda=0.95, eps_clip=0.2, K_epochs=4, entropy_coef=0.01,
                 cloud_reg_coef=1.2, edge_reg_coef=0.8, local_reg_coef=0.15,
                 target_cloud_max=0.60, target_edge_min=0.20, target_local_min=0.05,
                 reg_warmup_episodes=10):
        self.device = device
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.entropy_coef = entropy_coef

        self.policy = SharedActorCritic(node_dim, 64, action_dim).to(device)
        self.optimizer = optim.AdamW(self.policy.parameters(), lr=lr, weight_decay=1e-5)
        self.mse_loss = nn.MSELoss()
        self.memory = []

        # 推理耗时统计
        self.inference_times = []

        # v2.1 route regularization
        self.cloud_reg_coef = cloud_reg_coef
        self.edge_reg_coef = edge_reg_coef
        self.local_reg_coef = local_reg_coef
        self.target_cloud_max = target_cloud_max
        self.target_edge_min = target_edge_min
        self.target_local_min = target_local_min
        self.reg_warmup_episodes = reg_warmup_episodes
        self.current_episode = 0

    def put_data(self, item):
        # item: (state_cpu, action, reward, log_prob, done, mask, route_stats)
        self.memory.append(item)

    def clear_memory(self):
        self.memory = []

    def print_inference_stats(self):
        """打印推理耗时统计"""
        if not self.inference_times:
            print("[RealGATPPO] 推理耗时统计: 无数据")
            return

        times = np.array(self.inference_times)
        print(f"[RealGATPPO] 推理耗时统计 (共 {len(times)} 次推理):")
        print(f"  - 平均耗时: {times.mean():.4f} ms")
        print(f"  - 中位数: {np.median(times):.4f} ms")
        print(f"  - 最小耗时: {times.min():.4f} ms")
        print(f"  - 最大耗时: {times.max():.4f} ms")
        print(f"  - P95 耗时: {np.percentile(times, 95):.4f} ms")
        print(f"  - P99 耗时: {np.percentile(times, 99):.4f} ms")

    def reset_inference_stats(self):
        """重置推理耗时统计"""
        self.inference_times = []

    def set_episode(self, episode_num: int):
        self.current_episode = int(episode_num)

    def _apply_route_regularization(self, logits, route_stats):
        """
        v2.1: route-aware logit regularization
        action: 0=Local, 1=Cloud, 2..=Edge
        """
        if route_stats is None:
            return logits

        total = max(1, route_stats.get("local", 0) + route_stats.get("edge", 0) + route_stats.get("cloud", 0))
        local_ratio = route_stats.get("local", 0) / total
        edge_ratio = route_stats.get("edge", 0) / total
        cloud_ratio = route_stats.get("cloud", 0) / total

        warmup = min(1.0, float(self.current_episode) / max(1, self.reg_warmup_episodes))
        cloud_reg = self.cloud_reg_coef * warmup
        edge_reg = self.edge_reg_coef * warmup
        local_reg = self.local_reg_coef * warmup

        logits = logits.clone()

        # cloud overuse penalty
        if logits.shape[-1] > 1 and cloud_ratio > self.target_cloud_max:
            penalty = cloud_reg * ((cloud_ratio - self.target_cloud_max) / max(1e-6, (1.0 - self.target_cloud_max)))
            logits[..., 1] -= penalty

        # edge under-use bonus
        if logits.shape[-1] > 2 and edge_ratio < self.target_edge_min:
            bonus = edge_reg * ((self.target_edge_min - edge_ratio) / max(1e-6, self.target_edge_min))
            logits[..., 2:] += bonus

        # local weak encouragement
        if local_ratio < self.target_local_min:
            bonus = local_reg * ((self.target_local_min - local_ratio) / max(1e-6, self.target_local_min))
            logits[..., 0] += bonus

        return logits

    def take_action(self, state_data, action_mask=None, deterministic=False, route_stats=None):
        start_time = time.time()
        self.policy.eval()
        with torch.no_grad():
            if isinstance(state_data, Data):
                batch = Batch.from_data_list([state_data]).to(self.device)
            else:
                batch = state_data.to(self.device)

            logits, _ = self.policy(batch)
            logits = logits.squeeze(0)

            # v2.1 route-aware logit regularization
            logits = self._apply_route_regularization(logits, route_stats)

            # Strict Masking
            if action_mask is not None:
                mask_t = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device)
                # 修复使用 masked_fill 和 dtype 相关的 neg_inf
                neg_inf = torch.finfo(logits.dtype).min
                logits = logits.masked_fill(~mask_t, neg_inf)

            probs = torch.softmax(logits, dim=-1)
            dist = Categorical(probs)

            if deterministic:
                action = torch.argmax(probs).item()
            else:
                action = dist.sample().item()

            log_prob = dist.log_prob(torch.tensor(action, device=self.device)).item()

        inference_time = (time.time() - start_time) * 1000  # 转换为毫秒
        self.inference_times.append(inference_time)

        return action, log_prob

    def update(self):
        if not self.memory: return 0.0

        # 1. 准备数据
        states = [x[0] for x in self.memory]
        actions = torch.tensor([x[1] for x in self.memory], dtype=torch.long).to(self.device)
        rewards = [x[2] for x in self.memory]
        old_log_probs = torch.tensor([x[3] for x in self.memory], dtype=torch.float32).to(self.device)
        dones = [x[4] for x in self.memory]
        masks = [x[5] for x in self.memory]
        route_stats_list = [x[6] for x in self.memory]

        # 2. GAE 计算
        # 需要一次前向传播获取所有 Values
        # 为防显存溢出，分 Batch 处理
        values = []
        batch_size = 64
        self.policy.eval()
        with torch.no_grad():
            for i in range(0, len(states), batch_size):
                batch_states = Batch.from_data_list(states[i:i + batch_size]).to(self.device)
                _, v = self.policy(batch_states)
                values.append(v.squeeze(-1).cpu())
        values = torch.cat(values)  # [T]

        returns = []
        advantages = []
        gae = 0

        for i in reversed(range(len(rewards))):
            next_val = 0 if i == len(rewards) - 1 else values[i + 1]
            # 如果 done，则 next_val 视为 0 (或根据具体的 done 逻辑调整)
            # 在任务调度中，done 通常意味着 Episode 结束，后续无价值
            if dones[i]: gae = 0; next_val = 0

            delta = rewards[i] + self.gamma * next_val - values[i]
            gae = delta + self.gamma * self.lmbda * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)

        # Normalize Advantages
        if advantages.std() > 1e-6:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        # 3. PPO Update
        self.policy.train()
        total_loss = 0
        T = len(states)
        perm = np.random.permutation(T)

        for _ in range(self.K_epochs):
            for i in range(0, T, batch_size):
                idx = perm[i: i + batch_size]

                b_states = Batch.from_data_list([states[j] for j in idx]).to(self.device)
                b_actions = actions[idx]
                b_returns = returns[idx]
                b_adv = advantages[idx]
                b_old_lp = old_log_probs[idx]
                # 修复使用 stack 确保 mask shape 正确为 [B, action_dim]
                b_masks = torch.as_tensor(np.stack([masks[j] for j in idx]), dtype=torch.bool, device=self.device)
                b_route_stats = [route_stats_list[j] for j in idx]

                logits, v_pred = self.policy(b_states)
                v_pred = v_pred.squeeze(-1)

                # v2.1 route-aware logit regularization (per-sample)
                reg_logits = []
                for bi in range(logits.shape[0]):
                    reg_logits.append(self._apply_route_regularization(logits[bi], b_route_stats[bi]))
                logits = torch.stack(reg_logits, dim=0)

                # 【关键断言】确保 mask 形状正确
                assert logits.dim() == 2, f"logits dim should be 2, got {logits.dim()}, shape: {logits.shape}"
                assert b_masks.shape == logits.shape, f"mask shape {b_masks.shape} != logits shape {logits.shape}"
                assert b_masks.dtype == torch.bool, f"mask dtype should be bool, got {b_masks.dtype}"
                assert (b_masks.sum(dim=1) >= 1).all(), f"each sample must have at least one valid action, got {b_masks.sum(dim=1)}"

                # Masking in update - 使用 masked_fill 更安全
                neg_inf = torch.finfo(logits.dtype).min
                logits = logits.masked_fill(~b_masks, neg_inf)
                probs = torch.softmax(logits, dim=-1)
                dist = Categorical(probs)

                new_log_probs = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - b_old_lp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * b_adv

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = self.mse_loss(v_pred, b_returns)

                loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

                total_loss += loss.item()

        return total_loss
