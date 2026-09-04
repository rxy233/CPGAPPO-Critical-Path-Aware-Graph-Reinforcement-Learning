# -*- coding: utf-8 -*-
"""
Dual-GAT encoder + shared Actor-Critic (baseline network, forward-only encoder owner).

English
-------
SharedActorCritic wraps DualGATEncoder (num_layers stacked forward+backward
GATv2Conv) and a late-fusion Actor/Critic head. Context =
[target_embed, g_mean, g_max]. This is the network used by GAT_PPO_Agent
(agent.py). The main CPGAPPO model (CPGAPPOLateFusionActorCritic in cpgappo_core.py)
adds a global-resource MLP branch and the backward-gate; this file keeps the
minimal encoder for the PPO baseline. Orthogonal weight init (gain=sqrt(2)).

中文
----
DualGATEncoder (正向+反向 GAT 堆叠) + SharedActorCritic (late-fusion 头),
供 agent.py 的 PPO 基线使用。主模型在 cpgappo_core.py, 多了全局资源 MLP 分支。
"""
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool, global_add_pool

class DualGATEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, heads=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ELU()
        self.fwd_convs = nn.ModuleList()
        self.bwd_convs = nn.ModuleList()
        
        for _ in range(num_layers):
            self.fwd_convs.append(GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=False, dropout=0.1))
            self.bwd_convs.append(GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=False, dropout=0.1))

    def forward(self, x, edge_index):
        x = self.act(self.input_proj(x))
        rev_edge_index = edge_index[[1, 0], :] # 反向边
        
        h_fwd, h_bwd = x, x
        for f_conv, b_conv in zip(self.fwd_convs, self.bwd_convs):
            h_fwd = self.act(f_conv(h_fwd, edge_index) + h_fwd)
            h_bwd = self.act(b_conv(h_bwd, rev_edge_index) + h_bwd)
            
        return torch.cat([h_fwd, h_bwd], dim=-1)

class SharedActorCritic(nn.Module):
    def __init__(self, node_dim, hidden_dim=64, action_dim=3):
        super().__init__()
        self.encoder = DualGATEncoder(node_dim, hidden_dim)
        enc_dim = hidden_dim * 2
        # Context: Target + GlobalMean + GlobalMax
        context_dim = enc_dim * 3
        
        self.actor = nn.Sequential(
            nn.Linear(context_dim, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(),
            nn.Linear(128, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(context_dim, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None: nn.init.constant_(m.bias, 0.0)

    def forward(self, data):
        # 1. 编码
        node_embeds = self.encoder(data.x, data.edge_index) # [TotalN, EncDim]
        
        # 2. 提取 Target 节点 (假设 is_current 在最后一维)
        is_target = data.x[:, -1] > 0.5
        mask_float = is_target.float().unsqueeze(1) # [TotalN, 1]
        
        # 修复使用 masked mean pool 替代 scatter_add
        # 避免多个 target 节点时特征累加导致的策略不稳定
        masked = node_embeds * mask_float
        sum_pool = global_add_pool(masked, data.batch)
        cnt = global_add_pool(mask_float, data.batch).clamp_min(1.0)
        target_embeds = sum_pool / cnt
        
        # 3. 全局特征
        g_mean = global_mean_pool(node_embeds, data.batch)
        g_max = global_max_pool(node_embeds, data.batch)
        
        context = torch.cat([target_embeds, g_mean, g_max], dim=1)
        return self.actor(context), self.critic(context)
