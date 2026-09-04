# -*- coding: utf-8 -*-
"""
PPO算法核心 - GAE修复版 (Final Stable Version)
修复 Logits/Probs 混淆导致的 NaN 崩溃问题
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


class OptimizedPolicyNet(nn.Module):
    """优化的策略网络 - 输出 Logits"""
    def __init__(self, n_states, n_hiddens, n_actions, feature_dim):
        super().__init__()
        # 使用正交初始化有助于训练初期稳定
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.Tanh(),  # Tanh 通常比 ReLU 在 PPO 中更稳定
            nn.LayerNorm(256)
        )

        self.policy_head = nn.Sequential(
            nn.Linear(256, n_hiddens),
            nn.Tanh(),
            nn.LayerNorm(n_hiddens),
            nn.Linear(n_hiddens, n_actions)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = self.feature_extractor(x)
        logits = self.policy_head(x)
        return logits


class OptimizedValueNet(nn.Module):
    """优化的价值网络"""
    def __init__(self, n_states, n_hiddens, feature_dim):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.Tanh(),
            nn.LayerNorm(256)
        )

        self.value_head = nn.Sequential(
            nn.Linear(256, n_hiddens),
            nn.Tanh(),
            nn.LayerNorm(n_hiddens),
            nn.Linear(n_hiddens, 1)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = self.feature_extractor(x)
        value = self.value_head(x)
        return value


class PPO_GAE_Fixed:
    """GAE修复版PPO算法"""

    def __init__(self, n_states, n_hiddens, n_actions,
                 actor_lr, critic_lr, lmbda, epochs, eps, gamma, device,
                 batch_size=256, ent_coef=0.01):
        self.feature_dim = n_states
        self.n_states = n_states
        self.n_actions = n_actions
        self.device = device
        self.batch_size = batch_size
        self.ent_coef = ent_coef
        self.initial_ent_coef = ent_coef  # 保存初始值用于衰减
        self.epochs = epochs
        self.eps = eps
        self.gamma = gamma
        self.lmbda = lmbda

        self.actor = OptimizedPolicyNet(n_states, n_hiddens, n_actions, self.feature_dim).to(device)
        self.critic = OptimizedValueNet(n_states, n_hiddens, self.feature_dim).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, eps=1e-5)

        # 学习率调度器（加快衰减，防止后期震荡）
        # gamma=0.99 意味着每 episode 衰减 1%，比 0.995 的 0.5% 快
        self.actor_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.actor_optimizer, gamma=0.99
        )
        self.critic_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.critic_optimizer, gamma=0.99
        )

        self.state_cache = {}
        self.update_count = 0
        self.max_updates = 200  # 用于 Entropy 和 Clip 衰减（对应 200 episodes）

    def preprocess_state(self, state):
        """高效的状态预处理"""
        # 简单缓存 key 生成
        if hasattr(state, 'x') and state.x is not None:
            x_in = state.x
        elif isinstance(state, torch.Tensor):
            x_in = state
        else:
            x_in = torch.tensor(state, dtype=torch.float32)

        # 转为 Tensor 并移动到 Device
        if not isinstance(x_in, torch.Tensor):
            x = torch.tensor(x_in, dtype=torch.float32)
        else:
            x = x_in.clone().detach().float()

        if x.device != self.device:
            x = x.to(self.device)

        # 处理 NaN (输入清洗)
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)

        # 添加 Batch 维度
        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() == 2 and x.shape[0] != 1:
            # 如果是图特征平均后的 [1, dim] 保持原样
            # 如果是 [N, dim] 需要做 mean pooling
            if x.shape[0] > 1:
                x = x.mean(dim=0, keepdim=True)

        return x

    def take_action(self, state, task_id=None, action_mask=None, deterministic=False):
        """
        选择动作
        """
        with torch.no_grad():
            x = self.preprocess_state(state)
            logits = self.actor(x).squeeze(0)  # [n_actions]

            # 【防崩检查】如果网络输出了 NaN，说明权重已坏
            if torch.isnan(logits).any():
                print("[PPO CRITICAL] Network output NaN logits in take_action! Resetting weights.")
                self._reinit_network()
                logits = self.actor(x).squeeze(0)

            # 应用 mask
            if action_mask is not None:
                mask = torch.tensor(action_mask, device=logits.device, dtype=torch.bool)
                # 检查 mask 是否全为 False
                if not mask.any():
                    mask = torch.ones_like(mask)  # 兜底全开

                # 使用 -1e8 代替 -inf，防止 softmax 计算出 nan
                logits = logits.masked_fill(~mask, -1e8)

            dist = Categorical(logits=logits)

            if deterministic:
                action = int(logits.argmax().item())
            else:
                action = int(dist.sample().item())

        return action, None

    def learn(self, transition_dict):
        """
        PPO 更新逻辑 (修复了 Logits/Probs 混淆问题)
        """
        # 1. 数据准备
        if not transition_dict['states']:
            return 0.0, 0.0

        # 将 list 转为 tensor
        state_list = [self.preprocess_state(s) for s in transition_dict['states']]
        states = torch.cat(state_list, dim=0).to(self.device)

        actions = torch.tensor(transition_dict['actions'], dtype=torch.long, device=self.device).view(-1, 1)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float32, device=self.device).view(-1, 1)
        # dones 暂时没用到，因为我们是基于 episode 结束后的完整轨迹计算 GAE

        # 2. 计算 GAE (No Grad)
        with torch.no_grad():
            values = self.critic(states)

            # 扩展 next_value，最后一步通常假设 value=0 (因为 episode 结束了)
            # 或者如果有 next_states 也可以算，这里简化处理，假设最后一步 done
            values_np = values.cpu().numpy().flatten()
            rewards_np = rewards.cpu().numpy().flatten()

            advantages_np = np.zeros_like(rewards_np)
            gae = 0.0
            next_value = 0.0  # 假设最后一个状态后 value 为 0

            for t in reversed(range(len(rewards_np))):
                delta = rewards_np[t] + self.gamma * next_value - values_np[t]
                gae = delta + self.gamma * self.lmbda * gae
                advantages_np[t] = gae
                next_value = values_np[t]  # 更新 next_value 为当前 value (近似)

            advantages = torch.tensor(advantages_np, dtype=torch.float32, device=self.device).view(-1, 1)
            returns = advantages + values  # Target Value

            # 计算旧策略的 log_prob
            # 注意：这里我们重新计算一遍 logits，因为 buffer 里没存 logits
            # 如果动作空间很大，建议 buffer 里存 old_log_probs
            old_logits = self.actor(states)
            # 这里必须应用 mask 吗？Buffer 里没存 mask。
            # 假设 old_logits 在选中 action 上的概率是合理的
            old_logits = torch.nan_to_num(old_logits, nan=-1e9, posinf=1e9, neginf=-1e9)
            dist_old = Categorical(logits=old_logits)
            old_log_probs = dist_old.log_prob(actions.squeeze(1)).unsqueeze(1).detach()

        # 3. PPO Update (Grad)
        dataset_size = states.size(0)
        indices = np.arange(dataset_size)

        # 归一化 Advantage
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_actor_loss = 0
        total_critic_loss = 0

        for _ in range(self.epochs):
            np.random.shuffle(indices)

            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                idx = indices[start:end]

                mb_states = states[idx]
                mb_actions = actions[idx]
                mb_old_log_probs = old_log_probs[idx]
                mb_advantages = advantages[idx]
                mb_returns = returns[idx]

                # --- Actor Loss ---
                curr_logits = self.actor(mb_states)
                curr_logits = torch.nan_to_num(curr_logits, nan=-1e9, posinf=1e9, neginf=-1e9)
                dist = Categorical(logits=curr_logits)
                curr_log_probs = dist.log_prob(mb_actions.squeeze(1)).unsqueeze(1)
                curr_entropy = dist.entropy().mean()

                ratio = torch.exp(curr_log_probs - mb_old_log_probs)

                # 动态计算当前的 clip 值
                # 随着 update_count 增加，从 eps (0.2) 降到 0.1
                # 防止后期策略大改，保持稳定
                progress = self.update_count / self.max_updates
                current_eps = max(0.1, self.eps * (1.0 - progress))

                surr1 = ratio * mb_advantages.squeeze(1)
                surr2 = torch.clamp(ratio, 1 - current_eps, 1 + current_eps) * mb_advantages.squeeze(1)

                # 动态调整 Entropy 系数
                # 随着训练进行，线性衰减 ent_coef（减少后期随机探索）
                current_ent_coef = self.initial_ent_coef * (1.0 - progress)
                current_ent_coef = max(0.001, current_ent_coef)  # 保留一点点底噪

                actor_loss = -torch.min(surr1, surr2).mean() - current_ent_coef * curr_entropy

                # --- Critic Loss ---
                curr_values = self.critic(mb_states)
                critic_loss = F.mse_loss(curr_values, mb_returns)

                # --- Backward ---
                # Actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)  # 梯度裁剪
                self.actor_optimizer.step()

                # Critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()

        self.update_count += 1

        # 学习率调度器更新
        self.actor_scheduler.step()
        self.critic_scheduler.step()

        # 强制限制最小 LR（防止 LR 变成 0）
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'], 1e-6)
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'], 1e-6)

        # 定期检查权重是否变坏
        if self.update_count % 10 == 0:
            self._check_weights()

        return total_actor_loss, total_critic_loss

    def clear_cache(self):
        """清理缓存"""
        self.state_cache.clear()

    def clear_task_data(self):
        """兼容接口 - 清理跨 episode 累积的数据"""
        self.clear_cache()

    @property
    def training(self):
        """检查是否在训练模式"""
        return self.actor.training

    def _reinit_network(self):
        """当网络出现 NaN 时重置"""
        self.actor = OptimizedPolicyNet(self.n_states, 256, self.n_actions, self.feature_dim).to(self.device)
        self.critic = OptimizedValueNet(self.n_states, 256, self.feature_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=5e-4)

    def _check_weights(self):
        """检查权重是否出现 NaN"""
        for param in self.actor.parameters():
            if torch.isnan(param).any():
                print("[PPO ALERT] NaN detected in Actor weights! Re-initializing...")
                self._reinit_network()
                break
