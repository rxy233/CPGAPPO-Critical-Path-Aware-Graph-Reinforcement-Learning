"""
TransEdge-style GCN-Actor-Critic Core
移植版：forward-only 2-layer GCN + Actor-Critic

关键设计：
- 复用R1 graph state构造（保证公平性）
- 只做forward graph aggregation（无backward branch）
- 无global resource MLP late-fusion
- 无guide机制
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  # 仓库根 (本文件在 Algorithms/StrongBaselines/ 下, 上两级)
sys.path.insert(0, project_root)

from Algorithms.Baselines.gat_ppo_encoder_r1 import encode_dag_gat_ppo_state_r1, R1_FEATURE_DIM


class ForwardGCNEncoder(nn.Module):
    """Forward-only 2-layer GCN encoder"""
    
    def __init__(self, input_dim=R1_FEATURE_DIM, hidden_dim=64):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
    
    def forward(self, x, edge_index):
        """
        Args:
            x: [N, input_dim] node features
            edge_index: [2, E] edge indices
        
        Returns:
            h: [N, hidden_dim] node embeddings
        """
        h = self.act(self.input_proj(x))
        h1 = self.act(self.conv1(h, edge_index))
        h2 = self.act(self.conv2(h1, edge_index))
        return h2


class TransEdgeStyleActorCritic(nn.Module):
    """
    TransEdge-style Actor-Critic:
    - Forward GCN encoder
    - Candidate + global pooling context
    - No guide mechanism
    """
    
    def __init__(self, node_dim=R1_FEATURE_DIM, hidden_dim=64, action_dim=10):
        super().__init__()
        self.encoder = ForwardGCNEncoder(node_dim, hidden_dim)
        
        # Context: candidate + mean + max
        context_dim = hidden_dim * 3
        
        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(context_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim)
        )
        
        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(context_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
    
    def forward(self, data):
        """
        Args:
            data: PyG Data object with:
                - x: [N, node_dim] node features
                - edge_index: [2, E] edges
                - batch: [N] batch indices
        
        Returns:
            logits: [B, action_dim]
            value: [B, 1]
        """
        # Encode nodes
        node_embeds = self.encoder(data.x, data.edge_index)  # [N, hidden]
        
        # Extract candidate embeddings
        # R1: last dimension is 'is_cur' (1 for candidate, 0 otherwise)
        is_target = data.x[:, -1] > 0.5  # [N]
        mask_float = is_target.float().unsqueeze(1)  # [N, 1]
        
        # Candidate embedding: weighted sum by batch
        candidate_embeds = global_add_pool(node_embeds * mask_float, data.batch) / \
                          global_add_pool(mask_float, data.batch).clamp_min(1.0)  # [B, hidden]
        
        # Global pooling
        g_mean = global_mean_pool(node_embeds, data.batch)  # [B, hidden]
        g_max = global_max_pool(node_embeds, data.batch)    # [B, hidden]
        
        # Concatenate context
        context = torch.cat([candidate_embeds, g_mean, g_max], dim=1)  # [B, hidden*3]
        
        # Actor & Critic
        logits = self.actor(context)  # [B, action_dim]
        value = self.critic(context)  # [B, 1]
        
        return logits, value


def extract_transedge_state(ts, task_tuple, slot, task_complex_index=0):
    """
    Extract TransEdge-style state
    直接复用R1 graph state构造，保证state口径统一
    
    Returns:
        data: PyG Data object
        mask: action mask (list of 0/1)
    """
    data, mask = encode_dag_gat_ppo_state_r1(
        ts, task_tuple, slot=slot, now_time=None, task_complex_index=task_complex_index
    )
    return data, mask
