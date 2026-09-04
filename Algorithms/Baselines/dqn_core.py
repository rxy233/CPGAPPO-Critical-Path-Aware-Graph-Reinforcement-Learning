# -*- coding: utf-8 -*-
"""
DQN Core (Vanilla DQN with Experience Replay + Target Network)
|- 用于 FlexDO (DAG-DQN) 和 GBPT (SATA-DRL) baseline
"""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque


class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, capacity=20000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done, mask=None, mask2=None):
        """
        存储经验
        Args:
            s: 当前状态向量
            a: 动作索引
            r: 奖励
            s2: 下一状态向量
            done: 是否结束
            mask: 当前动作掩码 (可选)
            mask2: 下一状态动作掩码 (可选)
        """
        self.buf.append((s, a, r, s2, done, mask, mask2))

    def sample(self, batch_size=64, device="cpu"):
        """采样一个批次"""
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, done, mask, mask2 = zip(*batch)

        s = torch.tensor(np.array(s), dtype=torch.float32, device=device)
        a = torch.tensor(a, dtype=torch.int64, device=device).unsqueeze(-1)
        r = torch.tensor(r, dtype=torch.float32, device=device).unsqueeze(-1)
        s2 = torch.tensor(np.array(s2), dtype=torch.float32, device=device)
        done = torch.tensor(done, dtype=torch.float32, device=device).unsqueeze(-1)

        if mask[0] is None:
            mask_t = None
            mask2_t = None
        else:
            mask_t = torch.tensor(np.array(mask), dtype=torch.bool, device=device)
            mask2_t = torch.tensor(np.array(mask2), dtype=torch.bool, device=device)

        return s, a, r, s2, done, mask_t, mask2_t

    def __len__(self):
        return len(self.buf)


def to_bool_mask(m, action_dim):
    """
    修复统一的 mask 转换函数，正确处理"全 0"情况
    
    Args:
        m: mask 数组（可能是 additive 或 binary）
        action_dim: 动作空间维度
    
    Returns:
        list[bool]: True=可用, False=不可用
    """
    m = np.asarray(m, dtype=np.float32).reshape(-1)
    
    # 维度不匹配：保守处理为全可用
    if m.shape[0] != action_dim:
        return [True] * action_dim
    
    # 【关键】全 0 => additive mask 表示"全可用"
    if np.all(m == 0):
        return [True] * action_dim
    
    # additive: 0 valid, -1e9 invalid
    if m.min() < -1e6:
        return (m > -1e8).tolist()
    
    # binary: 1 valid, 0 invalid
    if m.max() > 0.5:
        return (m > 0.5).tolist()
    
    # 兜底：不认识的情况，别锁死
    return [True] * action_dim


class QNet(nn.Module):
    """
    改进版 QNet: Dueling DQN 架构
    - Value Stream: 评估状态好坏
    - Advantage Stream: 评估动作优势
    - Q = V + (A - mean(A))
    """
    def __init__(self, state_dim, action_dim, hidden1=512, hidden2=512):
        super().__init__()
        
        # 公共特征层
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden1),
            nn.LayerNorm(hidden1),  # 抗数值波动
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.LayerNorm(hidden2),
            nn.ReLU()
        )
        
        # Value Stream (评估状态好坏)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden2, hidden2),
            nn.LayerNorm(hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1)
        )
        
        # Advantage Stream (评估动作优势)
        self.adv_stream = nn.Sequential(
            nn.Linear(hidden2, hidden2),
            nn.LayerNorm(hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, action_dim)
        )

    def forward(self, x):
        feat = self.feature(x)
        val = self.value_stream(feat)
        adv = self.adv_stream(feat)
        
        # Dueling 合并: Q = V + (A - mean(A))
        # 这样可以让网络更好地学习"哪些动作比其他动作更好"
        q_val = val + (adv - adv.mean(dim=1, keepdim=True))
        return q_val


class DQNAgent:
    def __init__(self, state_dim, action_dim, device="cpu",
                 lr=1e-4, gamma=0.99,
                 eps_start=1.0, eps_end=0.05, eps_decay=0.998,
                 target_update=200, tau=0.005):

        """
        Args:
            state_dim: 状态维度
            action_dim: 动作空间大小
            device: 计算设备
            lr: 学习率
            gamma: 折扣因子
            eps_start: 初始探索率
            eps_end: 最小探索率
            eps_decay: 探索率衰减
            target_update: 目标网络更新频率（如果为None，使用soft update）
            tau: 软更新系数（仅当target_update=None时使用）
        """
        self.device = device
        self.action_dim = action_dim
        self.gamma = gamma

        # 加宽网络 + 增加层数
        self.q = QNet(state_dim, action_dim, hidden1=512, hidden2=512).to(device)
        self.q_tgt = QNet(state_dim, action_dim, hidden1=512, hidden2=512).to(device)
        self.q_tgt.load_state_dict(self.q.state_dict())

        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)  # 建议 lr=5e-5
        self.steps = 0

        # ====== 兼容旧保存逻辑（关键补丁）======
        self.q_net = self.q
        self.target_q_net = self.q_tgt
        self.q_optimizer = self.opt
        # ======================================

        # 探索率参数
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update = target_update
        self.tau = tau  # 软更新系数

    @torch.no_grad()
    def act(self, s_vec, action_mask=None, deterministic=False):
        """
        修复版：标准 DQN epsilon-greedy 策略

        Args:
            s_vec: np.ndarray [state_dim] - 状态向量
            action_mask: list/np [A], True=valid, False=invalid
            deterministic: 是否确定性模式（评估时使用纯 argmax）
        """
        s = torch.tensor(s_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.q(s).squeeze(0)

        if action_mask is not None:
            m = torch.tensor(action_mask, dtype=torch.bool, device=self.device)
            q = torch.where(m, q, torch.tensor(-1e9, device=self.device))

        # Epsilon 探索（仅训练模式）
        if (not deterministic) and (random.random() < self.eps):
            if action_mask is None:
                return random.randrange(self.action_dim)
            valid = np.where(np.array(action_mask, dtype=bool))[0]
            return int(np.random.choice(valid)) if len(valid) > 0 else 0

        # Greedy exploitation：argmax（带平局随机打破）
        best = q.max()
        cand = torch.where(q >= best - 1e-6)[0]  # 找出所有接近最大值的动作
        action = int(cand[torch.randint(len(cand), (1,))].item())

        return int(action)

    def update_eps(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

    def train_step(self, replay: ReplayBuffer, batch_size=64):
        """
        训练一步
        Returns:
            loss: 当前批次损失
        """
        if len(replay) < batch_size:
            return 0.0

        s, a, r, s2, done, mask, mask2 = replay.sample(batch_size, self.device)

        # Q(s,a)
        q_sa = self.q(s).gather(1, a)

        # TD Target: r + γ * max_a' Q'(s2,a')
        with torch.no_grad():
            q2 = self.q_tgt(s2)
            if mask2 is not None:
                huge_neg = torch.tensor(-1e9, device=self.device)
                q2 = torch.where(mask2, q2, huge_neg)

            # 修复：防止 mask2 全 False 导致 max_q2 = -1e9
            if mask2 is not None:
                any_valid = mask2.any(dim=1, keepdim=True)  # [B,1]
                max_q2 = q2.max(dim=1, keepdim=True)[0]
                max_q2 = torch.where(any_valid, max_q2, torch.zeros_like(max_q2))
            else:
                max_q2 = q2.max(dim=1, keepdim=True)[0]

            target = r + self.gamma * (1 - done) * max_q2

        # 使用 SmoothL1 更稳定
        loss = F.smooth_l1_loss(q_sa, target)

        # 反向传播
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
        self.opt.step()

        # 更新目标网络
        self.steps += 1
        if self.target_update is None:
            # Soft Update（每步都更新）
            for target_param, param in zip(self.q_tgt.parameters(), self.q.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        elif self.steps % self.target_update == 0:
            # Hard Update（按频率更新）
            self.q_tgt.load_state_dict(self.q.state_dict())

        return float(loss.item())

    def state_dict(self):
        """返回模型状态字典"""
        return self.q.state_dict()

    def load_state_dict(self, state_dict):
        """加载模型状态字典"""
        self.q.load_state_dict(state_dict)
        self.q_tgt.load_state_dict(state_dict)
