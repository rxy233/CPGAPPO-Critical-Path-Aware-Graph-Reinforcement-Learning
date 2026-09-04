"""
DGMA_paper Core — Multi-agent FEDG Actor + Centralized Critic + SPS

Faithful adaptation of Chen et al., IEEE TMC 2026 "DGMA":
  * K (= edge_num) distributed actors. Each agent k observes ONLY user-DAGs
    whose nearest edge is k (POMDP per region).
  * Each actor outputs a DISCRETE server choice over {Local, Cloud, Edge_0..K-1}
    plus a CONTINUOUS resource-allocation ratio c ∈ [0, 1] (scales the edge
    compute frequency applied to that task).
  * A centralized Critic Q(s_global, a_global) uses pooled FEDG embeddings from
    ALL regions and the joint action (one-hot server + c per task decision).
  * Selective Parameter Sharing (SPS): every N training steps, each actor's
    parameters are replaced by a dependency-weighted average of its peers:
        θ_k ← Σ_j w_{k,j} * θ_j ,    Σ_j w_{k,j} = 1
    where w_{k,j} ∝ inter-region dependency strength (cross-region DAG edges).

Environment is UNMODIFIED. All region construction / per-user improvement
estimates / c_alloc scheduling hooks are implemented in sister modules
(agent.py & train wrapper).

Implemented blocks:
  1. RegionActor                : FEDGEncoder + readout + discrete-head + c-head
  2. CentralCritic              : FEDGEncoder over the full graph + MLP(Q)
  3. build_region_mapping()     : user_id -> region_id (= nearest edge)
  4. compute_sps_weights()      : inter-region dependency weights from DAGs
  5. sps_apply()                : θ_k ← Σ w_kj θ_j  (in-place soft blend)
"""

import os
import sys
import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch

_cur_dir = os.path.dirname(os.path.abspath(__file__))
_proj_root = os.path.dirname(os.path.dirname(_cur_dir))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

# Re-use the FEDG implementation already validated in DGMA_adapt ———————————
from Algorithms.StrongBaselines.dgma_adapt_core import (
    FEDGEncoder, fedg_readout, R1_FEATURE_DIM,
)


# ===========================================================================
# RegionActor : per-region actor head, discrete server + continuous c_alloc
# ===========================================================================
class RegionActor(nn.Module):
    """One FEDG actor per edge-server region.

    Input : PyG Batch of DAG observations restricted to THIS region's users.
    Output:
        server_logits [B, A]   (A = edge_num + 2, i.e. Local/Cloud/Edge_*)
        c_alloc       [B, 1]   (squashed to [0, 1] via sigmoid)
    """

    def __init__(self, node_dim: int = R1_FEATURE_DIM, hidden_dim: int = 64,
                 num_layers: int = 2, action_dim: int = 6):
        super().__init__()
        self.encoder = FEDGEncoder(node_dim, hidden_dim, num_layers)
        ctx_dim = hidden_dim * 4
        self.trunk = nn.Sequential(
            nn.Linear(ctx_dim, 256), nn.Tanh(),
            nn.Linear(256, 128),     nn.Tanh(),
        )
        self.server_head = nn.Linear(128, action_dim)
        self.c_head = nn.Linear(128, 1)
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

    def forward(self, data: Data):
        """Returns (server_logits, c_alloc, ctx)."""
        h = self.encoder(data.x, data.edge_index)
        ctx = fedg_readout(h, data)          # [B, 4H]
        z = self.trunk(ctx)                   # [B, 128]
        server_logits = self.server_head(z)   # [B, A]
        c_raw = self.c_head(z)                # [B, 1]
        c_alloc = torch.sigmoid(c_raw)        # [B, 1] in (0, 1)
        return server_logits, c_alloc, ctx


# ===========================================================================
# CentralCritic : Q(s_global, a_global) via MADDPG-style centralized training
# ===========================================================================
class CentralCritic(nn.Module):
    """Centralized critic for MADDPG.

    Input representation at training time (per transition in a batch):
      * s_global : concatenation of fedg_readout over each region's subgraph.
                   We pre-compute it in the agent (not inside critic) so that
                   this module is a plain MLP; K regions × 4H each  →  [B, K*4H].
      * a_global : per-region action vector  [B, K * (A + 1)]
                   (one-hot server over A dims, ‖ c_alloc scalar)
    Output:
      * Q(s, a)  : [B, 1]
    """

    def __init__(self, num_regions: int, hidden_dim: int = 64,
                 action_dim: int = 6, mlp_hidden: int = 256):
        super().__init__()
        self.num_regions = num_regions
        self.ctx_dim = hidden_dim * 4
        self.action_dim = action_dim
        act_block = num_regions * (action_dim + 1)
        obs_block = num_regions * self.ctx_dim
        self.net = nn.Sequential(
            nn.Linear(obs_block + act_block, mlp_hidden), nn.Tanh(),
            nn.Linear(mlp_hidden, mlp_hidden // 2),       nn.Tanh(),
            nn.Linear(mlp_hidden // 2, 1),
        )

    def forward(self, obs_flat: torch.Tensor, act_flat: torch.Tensor):
        """obs_flat: [B, K*4H],  act_flat: [B, K*(A+1)]  ->  Q: [B, 1]"""
        x = torch.cat([obs_flat, act_flat], dim=-1)
        return self.net(x)


# ===========================================================================
# Region mapping : user_id -> region_id  (nearest edge by Euclidean distance)
# ===========================================================================
def build_region_mapping(env, num_regions: int) -> dict:
    """Partition users into K regions by nearest-edge rule.

    Returns:
        dict with keys:
          'user_to_region'   : list[int]   length = user_num
          'region_to_users'  : list[list[int]]  length = K
          'in_range'         : list[list[int]]  K rows of reachable user ids
                               (radius-based membership -- a user can be in
                                multiple edges' coverage, but only ONE agent
                                owns her for action selection)
    """
    user_num = len(env.device_list)
    user_to_region = [0] * user_num
    region_to_users = [[] for _ in range(num_regions)]
    in_range = [[] for _ in range(num_regions)]

    for uid, dev in enumerate(env.device_list):
        dists = np.asarray(dev.edge_distances, dtype=np.float64)
        # nearest edge is the owning region
        owner = int(np.argmin(dists))
        user_to_region[uid] = owner
        region_to_users[owner].append(uid)
        for eid in range(num_regions):
            if np.isfinite(dists[eid]):
                try:
                    from utils.constant import para
                    rad = para.get("edge_radius", 200)
                except Exception:
                    rad = 200
                if dists[eid] <= rad:
                    in_range[eid].append(uid)
    return {
        "user_to_region": user_to_region,
        "region_to_users": region_to_users,
        "in_range": in_range,
    }


# ===========================================================================
# Selective Parameter Sharing (SPS)
# ===========================================================================
def compute_sps_weights(env, num_regions: int, region_info: dict,
                        self_weight: float = 0.5) -> np.ndarray:
    """Inter-region dependency weights.

    Heuristic:  w_{k,j} ∝ (# DAG edges that cross region k -> region j)
                 normalized to sum to 1 across j; a dominant self-weight
                 (default 0.5) is injected so every agent keeps most of
                 its own identity and only blends minority mass from peers.

    Here there is no inherent cross-region DAG wiring (each user's DAG is
    independent), so the "dependency" signal we use is *user overlap in
    coverage*:  if users of region k are also in edge j's coverage radius,
    that edge "depends on" region k. This mirrors the DGMA spec (agents that
    share demand patterns share parameters).
    """
    W = np.zeros((num_regions, num_regions), dtype=np.float64)
    in_range = region_info["in_range"]
    for k in range(num_regions):
        users_k = set(in_range[k])
        for j in range(num_regions):
            if j == k:
                continue
            users_j = set(in_range[j])
            overlap = len(users_k & users_j)
            W[k, j] = float(overlap)
    # Normalize off-diagonal rows
    for k in range(num_regions):
        row_sum = W[k].sum()
        if row_sum > 0:
            W[k] = (1.0 - self_weight) * W[k] / row_sum
        W[k, k] = self_weight
    return W


@torch.no_grad()
def sps_apply(actors, weight_matrix: np.ndarray):
    """θ_k ← Σ_j w_{k,j} * θ_j   (in-place, on a SNAPSHOT of all actors)."""
    K = len(actors)
    assert weight_matrix.shape == (K, K)
    # Snapshot current params (as tensors) so updates don't feed into each other
    snapshots = []
    for a in actors:
        snapshots.append([p.detach().clone() for p in a.parameters()])

    for k, actor in enumerate(actors):
        for p_idx, p in enumerate(actor.parameters()):
            blended = torch.zeros_like(p.data)
            for j in range(K):
                w = float(weight_matrix[k, j])
                if w == 0.0:
                    continue
                blended.add_(snapshots[j][p_idx], alpha=w)
            p.data.copy_(blended)


# ===========================================================================
# Convenience: extract FULL-graph fedg readout for the centralized critic
# ===========================================================================
@torch.no_grad()
def global_state_repr(actors, region_states: list, device):
    """For each region's state (PyG Data), run the corresponding actor's
    encoder in eval mode to get a 4H context vector. Concatenate across K.

    Returns: [1, K * 4H]  — batch dim is 1 here, caller tiles as needed.
    """
    parts = []
    for k, state in enumerate(region_states):
        if state is None:
            parts.append(torch.zeros(1, actors[k].hidden_dim * 4, device=device))
            continue
        batch = Batch.from_data_list([state]).to(device)
        h = actors[k].encoder(batch.x, batch.edge_index)
        ctx = fedg_readout(h, batch)    # [1, 4H]
        parts.append(ctx)
    return torch.cat(parts, dim=-1)      # [1, K*4H]


# ===========================================================================
# Action encoding helpers (discrete server + continuous c) -> flat vector
# ===========================================================================
def encode_joint_action(server_ids, c_allocs, action_dim: int, num_regions: int,
                        device) -> torch.Tensor:
    """server_ids  : list[int or None]  length = K
       c_allocs    : list[float or None]  length = K
       Returns     : [1, K * (A + 1)]   one-hot + scalar, zeros for empty regions
    """
    A = action_dim
    vec = torch.zeros(1, num_regions * (A + 1), device=device)
    for k in range(num_regions):
        sid = server_ids[k]
        c = c_allocs[k]
        base = k * (A + 1)
        if sid is None or c is None:
            continue
        if 0 <= int(sid) < A:
            vec[0, base + int(sid)] = 1.0
        vec[0, base + A] = float(c)
    return vec


# ===========================================================================
# State extractor: build a per-region aggregated PyG Data
# ===========================================================================
def extract_region_state(ts, region_users: list, candidate_task, slot: int,
                         task_complex_index: int = 0):
    """Compose a single PyG Data whose node set is the UNION of R1-encoded
    DAGs for all users in `region_users`. The candidate task (task_tuple) is
    the one about to be decided; its `is_cur` flag is set inside the encoder.

    To keep things cheap, we reuse the existing R1 single-user encoder and
    batch-concat the user DAGs via PyG Batch (which re-indexes nodes).
    """
    from Algorithms.StrongBaselines.dgma_adapt_core import extract_dgma_adapt_state
    if not region_users:
        # degenerate empty region -> single zero node
        x = torch.zeros((1, R1_FEATURE_DIM), dtype=torch.float32)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        return Data(x=x, edge_index=edge_index), [1] * 6

    datas = []
    mask_bin = None
    for uid in region_users:
        dummy_task = (uid, 0) if candidate_task[0] != uid else candidate_task
        try:
            d, m = extract_dgma_adapt_state(
                ts, dummy_task, slot=slot, task_complex_index=task_complex_index
            )
            datas.append(d)
            if candidate_task[0] == uid:
                mask_bin = m
        except Exception:
            continue
    if not datas:
        x = torch.zeros((1, R1_FEATURE_DIM), dtype=torch.float32)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        return Data(x=x, edge_index=edge_index), [1] * 6
    if mask_bin is None:
        # candidate user not in this region — fall back to the first user's mask
        d0, mask_bin = extract_dgma_adapt_state(
            ts, (region_users[0], 0), slot=slot,
            task_complex_index=task_complex_index
        )
    batched = Batch.from_data_list(datas)
    data = Data(x=batched.x, edge_index=batched.edge_index)
    return data, mask_bin
