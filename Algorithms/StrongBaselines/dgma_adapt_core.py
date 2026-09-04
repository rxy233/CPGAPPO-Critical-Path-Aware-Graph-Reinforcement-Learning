"""
DGMA_adapt Core — FEDG Encoder + Discrete Actor / Critic Heads

Adapted from Chen et al., IEEE TMC 2026 "DGMA".
This repo's system model uses FCFS + non-preemptive multi-core queues (no continuous
resource slicing is part of the control plane). Therefore we only port the
*dependency-aware graph embedding* (FEDG) + *off-policy actor-critic* part of DGMA,
and keep the action space = this repo's discrete {Local, Cloud, Edge_k}.

FEDG components implemented here (in order):
  A) Reverse-DAG message passing  (edges are flipped: child -> parent)
  B) Attention scoring alpha_uv = softmax_u( w1^T h_v + w2^T h_u + w3 * e_uv )
  C) GRU merge at each layer:     h_v^{l+1} = GRU(agg_v^{l+1}, h_v^l)
  D) Readout: end-node pool (leaves of reverse DAG == roots of original DAG)
             + global mean pool + candidate embedding  =>  state embedding

All node features are reused from R1 encoder (27-dim). The very last channel of x
(`is_cur`) marks the candidate task, which is also used for candidate pooling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data

import os
import sys

_cur_dir = os.path.dirname(os.path.abspath(__file__))
_proj_root = os.path.dirname(os.path.dirname(_cur_dir))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

from Algorithms.Baselines.gat_ppo_encoder_r1 import (
    encode_dag_gat_ppo_state_r1,
    R1_FEATURE_DIM,
)


# ---------------------------------------------------------------------------
# Scatter helpers (avoid torch_scatter dependency)
# ---------------------------------------------------------------------------
def _scatter_softmax(score: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Per-group (by `index`) softmax. Groups with no elements => empty output.
    `score` and `index` are 1-D tensors of shape [E]. `num_nodes` is dst count.
    """
    if score.numel() == 0:
        return score
    # Max per group for numerical stability (fallback to 0 for empty groups)
    score_max = torch.full((num_nodes,), float("-inf"), device=score.device, dtype=score.dtype)
    score_max.scatter_reduce_(0, index, score, reduce="amax", include_self=True)
    score_max = torch.where(torch.isfinite(score_max), score_max, torch.zeros_like(score_max))
    exp = torch.exp(score - score_max[index])
    sum_exp = torch.zeros(num_nodes, device=score.device, dtype=score.dtype)
    sum_exp.index_add_(0, index, exp)
    sum_exp = sum_exp.clamp_min(1e-12)
    return exp / sum_exp[index]


# ---------------------------------------------------------------------------
# FEDG Layer
# ---------------------------------------------------------------------------
class FEDGLayer(nn.Module):
    """One round of dependency-aware aggregation + GRU merge.

    `edge_index` passed in is the ORIGINAL DAG direction (u->v means u is
    a prerequisite of v). We reverse it inside: `rev = (v->u)` so messages
    flow child->parent during aggregation (matches DGMA spec).
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        # attention scoring params
        self.w1 = nn.Linear(hidden_dim, 1, bias=False)       # contribution of h_v (target / parent)
        self.w2 = nn.Linear(hidden_dim, 1, bias=False)       # contribution of h_u (neighbor / child)
        self.edge_bias = nn.Parameter(torch.zeros(1))        # scalar edge bias  (no edge features)
        # message projection
        self.msg_proj = nn.Linear(hidden_dim, hidden_dim)
        # GRU merge
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """h: [N, H],  edge_index: [2, E] (original DAG direction).  Returns [N, H]."""
        N, H = h.shape
        if edge_index.numel() == 0:
            # No edges: GRU merge with zero message
            zero_msg = torch.zeros_like(h)
            return self.gru(zero_msg, h)

        # Flip to reverse DAG: src_rev = v, dst_rev = u   (msgs flow child->parent)
        src_orig, dst_orig = edge_index[0], edge_index[1]
        src_rev, dst_rev = dst_orig, src_orig  # reversed

        # Attention scores (pre-softmax), LeakyReLU as in GATv2-style
        score = (
            self.w1(h[dst_rev]).squeeze(-1)
            + self.w2(h[src_rev]).squeeze(-1)
            + self.edge_bias
        )
        score = F.leaky_relu(score, 0.2)
        alpha = _scatter_softmax(score, dst_rev, num_nodes=N)

        # Message  m_uv = alpha * W h_u
        msg = self.msg_proj(h[src_rev]) * alpha.unsqueeze(-1)

        # Aggregate into dst nodes
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst_rev, msg)

        # GRU merge
        h_new = self.gru(agg, h)
        return h_new


# ---------------------------------------------------------------------------
# FEDG Encoder (multi-layer)
# ---------------------------------------------------------------------------
class FEDGEncoder(nn.Module):
    """Input-proj -> stacked FEDGLayer -> node embeddings.

    The readout is handled by the Actor / Critic heads below, since they need
    slightly different views (candidate vs end-node pool + mean pool).
    """

    def __init__(self, input_dim: int = R1_FEATURE_DIM, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([FEDGLayer(hidden_dim) for _ in range(num_layers)])
        self.act = nn.ReLU()
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.act(self.input_proj(x))
        for layer in self.layers:
            h = layer(h, edge_index)
        return h  # [N, hidden_dim]


# ---------------------------------------------------------------------------
# Readout utilities
# ---------------------------------------------------------------------------
def _end_node_mask(edge_index: torch.Tensor, num_nodes: int, device) -> torch.Tensor:
    """Return a [N] float mask: 1 if node is a leaf in reverse DAG (== root in original DAG).

    A leaf in reverse DAG has no outgoing reverse edge, i.e. the original node is
    never a destination of any original edge.
    """
    if edge_index.numel() == 0:
        return torch.ones(num_nodes, device=device)
    dst_orig = edge_index[1]
    is_dst = torch.zeros(num_nodes, device=device, dtype=torch.bool)
    is_dst[dst_orig] = True
    return (~is_dst).float()


def fedg_readout(node_embeds: torch.Tensor, data: Data) -> torch.Tensor:
    """Compose the state embedding from a batch of graphs.

    Returns [B, 4H]  =  [candidate, end_pool, g_mean, g_max]
    """
    H = node_embeds.size(1)
    # candidate mask comes from last R1 feature column (is_cur)
    is_cand = (data.x[:, -1] > 0.5).float().unsqueeze(1)  # [N, 1]
    # end-node mask (leaves of reverse DAG) -- works per graph since edge_index
    # does not cross graphs (PyG Batch re-indexes)
    end_mask = _end_node_mask(data.edge_index, data.x.size(0), data.x.device).unsqueeze(1)  # [N, 1]

    cand = global_add_pool(node_embeds * is_cand, data.batch) / \
        global_add_pool(is_cand, data.batch).clamp_min(1.0)
    end_pool = global_add_pool(node_embeds * end_mask, data.batch) / \
        global_add_pool(end_mask, data.batch).clamp_min(1.0)
    g_mean = global_mean_pool(node_embeds, data.batch)
    g_max = global_max_pool(node_embeds, data.batch)

    return torch.cat([cand, end_pool, g_mean, g_max], dim=1)  # [B, 4H]


# ---------------------------------------------------------------------------
# Actor / Critic networks for discrete DGMA_adapt
# ---------------------------------------------------------------------------
class DGMAAdaptActor(nn.Module):
    """Outputs logits over action_dim. Shares design language with TransEdgeStyle actor."""

    def __init__(self, node_dim: int = R1_FEATURE_DIM, hidden_dim: int = 64,
                 num_layers: int = 2, action_dim: int = 10):
        super().__init__()
        self.encoder = FEDGEncoder(node_dim, hidden_dim, num_layers)
        context_dim = hidden_dim * 4
        self.head = nn.Sequential(
            nn.Linear(context_dim, 256), nn.Tanh(),
            nn.Linear(256, 128),         nn.Tanh(),
            nn.Linear(128, action_dim),
        )

    def forward(self, data: Data) -> torch.Tensor:
        h = self.encoder(data.x, data.edge_index)
        ctx = fedg_readout(h, data)
        return self.head(ctx)  # [B, action_dim]


class DGMAAdaptCritic(nn.Module):
    """Discrete-action Q network: Q(s, .) of shape [B, action_dim].

    Like dueling: Q = V(s) + A(s) - mean(A(s)) so that baseline is stable.
    """

    def __init__(self, node_dim: int = R1_FEATURE_DIM, hidden_dim: int = 64,
                 num_layers: int = 2, action_dim: int = 10):
        super().__init__()
        self.encoder = FEDGEncoder(node_dim, hidden_dim, num_layers)
        context_dim = hidden_dim * 4
        self.v_head = nn.Sequential(
            nn.Linear(context_dim, 256), nn.Tanh(),
            nn.Linear(256, 128),         nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.a_head = nn.Sequential(
            nn.Linear(context_dim, 256), nn.Tanh(),
            nn.Linear(256, 128),         nn.Tanh(),
            nn.Linear(128, action_dim),
        )

    def forward(self, data: Data) -> torch.Tensor:
        h = self.encoder(data.x, data.edge_index)
        ctx = fedg_readout(h, data)
        v = self.v_head(ctx)                         # [B, 1]
        a = self.a_head(ctx)                         # [B, A]
        q = v + (a - a.mean(dim=-1, keepdim=True))   # dueling combine
        return q


def extract_dgma_adapt_state(ts, task_tuple, slot, task_complex_index=0):
    """Wrap R1 encoder so we can swap easily later."""
    data, mask = encode_dag_gat_ppo_state_r1(
        ts, task_tuple, slot=slot, now_time=None, task_complex_index=task_complex_index
    )
    return data, mask
