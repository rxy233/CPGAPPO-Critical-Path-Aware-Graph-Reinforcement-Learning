# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import heapq
from collections import deque

class DuelingDQNNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256): # 减小网络规模，防止过拟合
        super(DuelingDQNNetwork, self).__init__()
        # 简化网络结构，提高泛化能力
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=0.01) #以此初始化，初始输出接近0
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values

class PrioritizedReplayBuffer:
    """基于 Reward 的分层 Buffer，保证稳定性"""
    def __init__(self, capacity, state_dim, action_dim, device):
        self.capacity = capacity
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.high_buffer = deque(maxlen=int(capacity * 0.5)) # 50% 容量给好样本
        self.norm_buffer = deque(maxlen=int(capacity * 0.5)) # 50% 给普通样本

    def push(self, state, action, reward, next_state, done, valid_mask=None, next_valid_mask=None):
        transition = (state, action, reward, next_state, done, valid_mask, next_valid_mask)
        # 只有真正成功的（奖励 > 0）才进 High Buffer
        if reward > 0.0:
            self.high_buffer.append(transition)
        else:
            self.norm_buffer.append(transition)

    def sample(self, batch_size):
        # 强制 30% 样本来自 High Buffer (如果够的话)，保证正反馈
        # 不要太多，否则容易过拟合到少数成功样本
        high_ratio = 0.3
        high_count = min(len(self.high_buffer), int(batch_size * high_ratio))
        norm_count = batch_size - high_count

        # 补齐
        if len(self.norm_buffer) < norm_count:
            norm_count = len(self.norm_buffer)
            high_count = min(len(self.high_buffer), batch_size - norm_count)

        if high_count + norm_count == 0: return None

        batch_high = random.sample(self.high_buffer, high_count)
        batch_norm = random.sample(self.norm_buffer, norm_count)
        batch = batch_high + batch_norm
        random.shuffle(batch)

        # 解包 (同前)
        states = np.array([t[0] for t in batch], dtype=np.float32)
        actions = np.array([t[1] for t in batch], dtype=np.int64).reshape(-1, 1)
        rewards = np.array([t[2] for t in batch], dtype=np.float32).reshape(-1, 1)
        next_states = np.array([t[3] for t in batch], dtype=np.float32)
        dones = np.array([t[4] for t in batch], dtype=np.float32).reshape(-1, 1)

        valid_mask_list = [t[5] if t[5] is not None else np.ones(self.action_dim) for t in batch]
        valid_masks = np.array(valid_mask_list, dtype=np.uint8)

        next_valid_mask_list = [t[6] if t[6] is not None else np.ones(self.action_dim) for t in batch]
        next_valid_masks = np.array(next_valid_mask_list, dtype=np.uint8)

        return (
            torch.from_numpy(states).to(self.device),
            torch.from_numpy(actions).to(self.device),
            torch.from_numpy(rewards).to(self.device),
            torch.from_numpy(next_states).to(self.device),
            torch.from_numpy(dones).to(self.device),
            torch.from_numpy(valid_masks).to(self.device).bool(),
            torch.from_numpy(next_valid_masks).to(self.device).bool()
        )

    def __len__(self):
        return len(self.high_buffer) + len(self.norm_buffer)


class DQN_Agent:
    NEG_INF = -1e9

    def __init__(self, args, basegraph_num, state_dim, out_dim):
        self.args = args
        self.state_dim = state_dim
        self.action_dim = out_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DQN] Init Stable DQN. Device: {self.device}")

        self.gamma = getattr(args, 'gamma', 0.95) # 降低 Gamma，关注短期拥塞，避免远期估计误差累积
        self.lr = getattr(args, 'lr', 5e-5)    # 更小的 LR
        self.batch_size = getattr(args, 'batch_size', 256)
        self.capacity = getattr(args, 'memory_size', 50000)
        self.tau = getattr(args, 'tau', 0.005)  # 稍快一点的软更新，适应变化
        self.warmup_steps = getattr(args, 'warmup', 2000)

        # Epsilon: 永远保持 10% 随机性
        self.epsilon_start = getattr(args, "eps", 1.0)
        self.epsilon_end = getattr(args, "eps_min", 0.1)
        self.epsilon_decay_steps = getattr(args, "eps_decay", 50000)

        self.step_count = 0
        self.update_count = 0
        self.loss_history = []

        self.q_net = DuelingDQNNetwork(state_dim, out_dim).to(self.device)
        self.target_net = DuelingDQNNetwork(state_dim, out_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        # 增加 Weight Decay (L2 正则化)
        self.optimizer = optim.AdamW(self.q_net.parameters(), lr=self.lr, weight_decay=1e-3)

        self.memory = PrioritizedReplayBuffer(self.capacity, self.state_dim, self.action_dim, self.device)

        # 专家记忆（永久保存，防止灾难性遗忘）
        self.expert_memory = PrioritizedReplayBuffer(50000, self.state_dim, self.action_dim, self.device)

    def _get_epsilon(self):
        if self.step_count >= self.epsilon_decay_steps:
            return self.epsilon_end
        progress = self.step_count / self.epsilon_decay_steps
        return self.epsilon_start - (self.epsilon_start - self.epsilon_end) * progress

    def _process_state(self, state):
        if isinstance(state, np.ndarray): data = state.reshape(-1)
        elif isinstance(state, torch.Tensor): data = state.detach().cpu().numpy().reshape(-1)
        else: data = np.array(state, dtype=np.float32).reshape(-1)

        # 简单裁剪，不做 Log，保持原始分布
        data = np.clip(data, -10.0, 10.0)
        return data

    def select_action(self, state, action_mask=None, random=False, test_only=False):
        state_np = self._process_state(state)
        if test_only: eps = 0.0
        else:
            eps = self._get_epsilon()
            self.step_count += 1

        valid = None
        if action_mask is not None:
            m = np.asarray(action_mask).reshape(-1)
            valid = (m > 0.5)

        # 随机探索
        if random or (not test_only and np.random.random() < eps):
            if valid is None: return int(np.random.randint(0, self.action_dim))
            idx = np.where(valid)[0]
            if idx.size == 0: return 0
            return int(np.random.choice(idx))

        # 贪婪选择
        with torch.no_grad():
            state_tensor = torch.from_numpy(state_np).float().to(self.device).unsqueeze(0)
            q_values = self.q_net(state_tensor).squeeze(0)
            if valid is not None:
                invalid = torch.from_numpy(~valid).to(self.device)
                q_values = q_values.masked_fill(invalid, self.NEG_INF)

            # Softmax 采样代替 Argmax (训练时)
            # 在训练时使用 Softmax 采样可以增加策略的随机性，避免死锁
            if not test_only:
                # 温度系数，越大越随机
                temperature = 1.0
                probs = F.softmax(q_values / temperature, dim=0)
                # 再次 mask 掉非法动作的概率
                if valid is not None:
                    probs = probs.masked_fill(invalid, 0.0)
                    probs_sum = probs.sum()
                    if probs_sum > 0:
                        probs /= probs_sum
                    else:
                        # 全非法，随机
                        idx = np.where(valid)[0]
                        if idx.size > 0: return int(np.random.choice(idx))
                        return 0

                action = torch.multinomial(probs, 1).item()
                return int(action)
            else:
                return int(q_values.argmax().item())

    def add_transition(self, transition, is_expert=False):
        """支持专家数据存储"""
        if len(transition) == 5:
            state, action, reward, next_state, done = transition
            mask = None; next_mask = None
        else:
            state, action, reward, next_state, done, mask, next_mask = transition

        s_np = self._process_state(state)
        ns_np = self._process_state(next_state)

        # 奖励缩放：除以 10，让数值更小，梯度更稳
        reward = float(reward) / 10.0

        valid = (np.array(mask) > 0.5).astype(np.uint8) if mask is not None else None
        next_valid = (np.array(next_mask) > 0.5).astype(np.uint8) if next_mask is not None else None

        # 根据是否专家数据选择 buffer
        if is_expert:
            self.expert_memory.push(s_np, int(action), reward, ns_np, float(done), valid, next_valid)
        else:
            self.memory.push(s_np, int(action), reward, ns_np, float(done), valid, next_valid)

    def update(self):
        if len(self.memory) < max(self.batch_size, self.warmup_steps):
            return None

        sample = self.memory.sample(self.batch_size)
        if sample is None: return None

        states, actions, rewards, next_states, dones, valid_masks, next_valid_masks = sample

        q_values = self.q_net(states).gather(1, actions)

        with torch.no_grad():
            next_q_online = self.q_net(next_states)
            next_q_online = next_q_online.masked_fill(~next_valid_masks, self.NEG_INF)
            next_actions = next_q_online.argmax(dim=1, keepdim=True)

            next_q_target = self.target_net(next_states)
            next_q = next_q_target.gather(1, next_actions)

            is_invalid = (next_q < -1e8)
            next_q = next_q.masked_fill(is_invalid, 0.0)

            target_q_values = rewards + (1 - dones) * self.gamma * next_q

        # 使用 Huber Loss 增强稳定性
        loss = F.huber_loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        # 极强的梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=0.1)
        self.optimizer.step()

        self._soft_update()
        self.update_count += 1
        loss_val = loss.item()
        self.loss_history.append(loss_val)
        return loss_val

    def _soft_update(self):
        for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def get_stats(self):
        return {
            'step_count': self.step_count,
            'epsilon': self._get_epsilon(),
            'buffer_size': len(self.memory),
            'expert_buffer_size': len(self.expert_memory),
            'avg_loss': np.mean(self.loss_history[-100:]) if self.loss_history else 0
        }

    def save(self, path):
        torch.save({'q_net': self.q_net.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint['q_net'])
