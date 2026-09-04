"""
DGMA_paper Agent — Multi-agent MADDPG with SPS.

Roles:
  * K distributed actors (one per edge region).
  * 1 centralized critic (MADDPG-style).
  * Joint experience replay with PER.
  * Slot-level arbitrator that turns per-region decisions into the single-task
    single-step interface required by the repo's scheduler.

Training signal:
  * Continuous c_alloc head is trained by the policy-gradient through the
    centralized critic (reparameterized Gumbel-Softmax for the discrete server
    choice + raw sigmoid c).
  * Discrete server head trained the same way as DGMA_adapt (Gumbel-Softmax
    soft one-hot feeds the critic).
  * Per-region reward r_k is used as each actor's TD target. Target Q is the
    min-of-two-target-critics trick disabled for simplicity (single critic,
    soft target update).
"""

import os
import sys
import math
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
from torch_geometric.data import Batch

_cur_dir = os.path.dirname(os.path.abspath(__file__))
_proj_root = os.path.dirname(os.path.dirname(_cur_dir))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

from Algorithms.StrongBaselines.dgma_paper_core import (
    RegionActor, CentralCritic, R1_FEATURE_DIM,
    build_region_mapping, compute_sps_weights, sps_apply,
    global_state_repr, encode_joint_action,
)
from Algorithms.StrongBaselines.dgma_adapt_core import fedg_readout
from Algorithms.StrongBaselines.dgma_adapt_agent import PERBuffer


# ===========================================================================
# Joint transition: stores per-region (s, a_discrete, a_cont, r, s'), done
# ===========================================================================
class DGMAPaperAgent:
    def __init__(self,
                 env,
                 num_regions: int,
                 node_dim: int = R1_FEATURE_DIM,
                 action_dim: int = 6,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 device: torch.device = None,
                 lr_actor: float = 3e-4,
                 lr_critic: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 gumbel_tau: float = 1.0,
                 batch_size: int = 32,
                 buffer_capacity: int = 10000,
                 per_alpha: float = 0.6,
                 per_beta0: float = 0.4,
                 per_beta_anneal_steps: int = 10000,
                 updates_per_step: int = 1,
                 min_updates_after: int = 256,
                 grad_clip: float = 1.0,
                 sps_every: int = 200,
                 sps_self_weight: float = 0.6):
        self.device = device or (torch.device("cuda:0") if torch.cuda.is_available()
                                 else torch.device("cpu"))
        self.num_regions = num_regions
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.tau = tau
        self.gumbel_tau = gumbel_tau
        self.batch_size = batch_size
        self.per_beta0 = per_beta0
        self.per_beta_anneal_steps = per_beta_anneal_steps
        self.updates_per_step = updates_per_step
        self.min_updates_after = min_updates_after
        self.grad_clip = grad_clip
        self.sps_every = sps_every
        self.global_step = 0
        self.inference_times = []

        # === Actors (K online + K target) =======================================
        self.actors = nn.ModuleList([
            RegionActor(node_dim, hidden_dim, num_layers, action_dim).to(self.device)
            for _ in range(num_regions)
        ])
        self.actors_target = nn.ModuleList([
            RegionActor(node_dim, hidden_dim, num_layers, action_dim).to(self.device)
            for _ in range(num_regions)
        ])
        for a, a_t in zip(self.actors, self.actors_target):
            a_t.load_state_dict(a.state_dict())
            for p in a_t.parameters():
                p.requires_grad = False

        # === Critic (1 online + 1 target) ======================================
        self.critic = CentralCritic(num_regions, hidden_dim, action_dim).to(self.device)
        self.critic_target = CentralCritic(num_regions, hidden_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # === Optimizers ========================================================
        self.opt_actors = [torch.optim.Adam(a.parameters(), lr=lr_actor)
                           for a in self.actors]
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # === Replay buffer (joint transitions) =================================
        self.buffer = PERBuffer(capacity=buffer_capacity, alpha=per_alpha)

        # === Region info & SPS weights =========================================
        self.region_info = build_region_mapping(env, num_regions)
        self.sps_weights = compute_sps_weights(env, num_regions, self.region_info,
                                               self_weight=sps_self_weight)

        # === Book-keeping ======================================================
        self.last_actor_loss = 0.0
        self.last_critic_loss = 0.0

    # --------------------------------------------------------------------- set
    def set_episode(self, ep: int):
        self.current_episode = ep

    # --------------------------------------------------------------- inference
    @torch.no_grad()
    def take_action_region(self, region_id: int, state_data,
                           action_mask: Optional[list] = None,
                           deterministic: bool = False) -> Tuple[int, float, float]:
        """Pick (server_id, c_alloc, log_prob) for this region's candidate task.

        If `state_data` is None (empty region), returns a no-op tuple.
        """
        t0 = time.time()
        if state_data is None:
            return 0, 0.5, 0.0
        actor = self.actors[region_id]
        actor.eval()
        batch = Batch.from_data_list([state_data]).to(self.device)
        server_logits, c_alloc, _ = actor(batch)
        server_logits = server_logits.squeeze(0)          # [A]
        c = float(c_alloc.squeeze().item())

        if action_mask is not None:
            mask_t = torch.as_tensor(action_mask, dtype=torch.bool,
                                     device=self.device)
            neg_inf = torch.finfo(server_logits.dtype).min
            if mask_t.shape[0] == server_logits.shape[0]:
                server_logits = server_logits.masked_fill(~mask_t, neg_inf)

        if deterministic:
            server_id = int(torch.argmax(server_logits).item())
        else:
            probs = torch.softmax(server_logits, dim=-1)
            if torch.isnan(probs).any() or probs.sum().item() <= 0:
                server_id = int(torch.argmax(server_logits).item())
            else:
                server_id = int(torch.distributions.Categorical(probs=probs).sample().item())
        log_prob = float(F.log_softmax(server_logits, dim=-1)[server_id].item())
        self.inference_times.append((time.time() - t0) * 1000.0)
        return server_id, c, log_prob

    # ------------------------------------------------------------- arbitration
    @staticmethod
    def arbitrate(task_decisions: list) -> list:
        """Given a list of (task_tuple, server_id, c_alloc, region_id, urgency),
        return them in a stable dispatch order: decreasing urgency, stable.
        """
        return sorted(task_decisions, key=lambda e: (-float(e[4]), e[3], e[0]))

    # -------------------------------------------------------------- experience
    def store(self, transition: tuple):
        """transition = (region_states_t, server_ids, c_allocs, r_per_region,
                         region_states_tp1, done, masks_t, masks_tp1)
           All states are single-graph PyG Data objects (or None for empty).
        """
        self.buffer.push(transition)

    # ------------------------------------------------------------- soft update
    def _soft_update(self, target: nn.Module, source: nn.Module):
        with torch.no_grad():
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(self.tau * sp.data)

    def _beta(self):
        frac = min(1.0, self.global_step / max(1, self.per_beta_anneal_steps))
        return self.per_beta0 + frac * (1.0 - self.per_beta0)

    # ---------------------------------------------------------------- helpers
    def _encode_region_state_batch(self, states_per_region: List[list]) -> torch.Tensor:
        """states_per_region: K rows × B entries of (PyG Data | None).

        We run each region's encoder over that region's batch, then concat the
        4H contexts from all K regions. Returns [B, K*4H].
        """
        K = self.num_regions
        B = len(states_per_region[0])
        ctx_list = []
        for k in range(K):
            datas = states_per_region[k]
            # Replace None with a tiny zero graph
            filled = []
            mask = torch.zeros(B, 1, device=self.device)
            for i, d in enumerate(datas):
                if d is None:
                    x = torch.zeros((1, R1_FEATURE_DIM), dtype=torch.float32)
                    ei = torch.zeros((2, 0), dtype=torch.long)
                    from torch_geometric.data import Data
                    filled.append(Data(x=x, edge_index=ei))
                else:
                    filled.append(d)
                    mask[i, 0] = 1.0
            batch = Batch.from_data_list(filled).to(self.device)
            h = self.actors[k].encoder(batch.x, batch.edge_index)
            ctx = fedg_readout(h, batch)     # [B, 4H]
            # NaN protection from empty graphs
            ctx = torch.where(torch.isnan(ctx), torch.zeros_like(ctx), ctx)
            ctx = ctx * mask                  # zero-out empty regions
            ctx_list.append(ctx)
        return torch.cat(ctx_list, dim=-1)    # [B, K*4H]

    def _encode_joint_action_batch(self, server_ids_per_region,
                                   c_allocs_per_region) -> torch.Tensor:
        """server_ids_per_region : K rows × B ints (or None)
           c_allocs_per_region   : K rows × B floats (or None)
           Returns [B, K*(A+1)].
        """
        K = self.num_regions
        A = self.action_dim
        B = len(server_ids_per_region[0])
        vec = torch.zeros(B, K * (A + 1), device=self.device)
        for k in range(K):
            base = k * (A + 1)
            for i in range(B):
                sid = server_ids_per_region[k][i]
                c = c_allocs_per_region[k][i]
                if sid is None or c is None:
                    continue
                s = int(sid)
                if 0 <= s < A:
                    vec[i, base + s] = 1.0
                vec[i, base + A] = float(c)
        return vec

    def _compute_target_actions(self, next_states_per_region):
        """For each region & each sample in batch, run target actor to produce
        (argmax server, c_alloc). Returns flat joint action [B, K*(A+1)]."""
        K = self.num_regions
        A = self.action_dim
        B = len(next_states_per_region[0])
        vec = torch.zeros(B, K * (A + 1), device=self.device)
        for k in range(K):
            datas = next_states_per_region[k]
            filled = []
            present = []
            for d in datas:
                if d is None:
                    from torch_geometric.data import Data
                    filled.append(Data(
                        x=torch.zeros((1, R1_FEATURE_DIM), dtype=torch.float32),
                        edge_index=torch.zeros((2, 0), dtype=torch.long),
                    ))
                    present.append(0.0)
                else:
                    filled.append(d)
                    present.append(1.0)
            batch = Batch.from_data_list(filled).to(self.device)
            with torch.no_grad():
                s_logits, c_alloc, _ = self.actors_target[k](batch)
                # Greedy discrete + continuous c
                a = torch.argmax(s_logits, dim=-1)     # [B]
            base = k * (A + 1)
            for i in range(B):
                if present[i] < 0.5:
                    continue
                vec[i, base + int(a[i].item())] = 1.0
                vec[i, base + A] = float(c_alloc[i, 0].item())
        return vec

    # ----------------------------------------------------------------- update
    def update(self):
        if len(self.buffer) < max(self.batch_size, self.min_updates_after):
            return 0.0, 0.0
        self.global_step += 1
        samples, indices, is_weights = self.buffer.sample(
            self.batch_size, beta=self._beta())
        if len(samples) == 0:
            return 0.0, 0.0

        K = self.num_regions
        A = self.action_dim
        B = len(samples)
        # Unpack
        states_t = [[None] * B for _ in range(K)]
        states_tp1 = [[None] * B for _ in range(K)]
        server_ids = [[None] * B for _ in range(K)]
        c_allocs = [[None] * B for _ in range(K)]
        rewards = np.zeros((B, K), dtype=np.float32)
        dones = np.zeros(B, dtype=np.float32)
        for i, trans in enumerate(samples):
            st, sid, c, r, stp1, done, _mask_t, _mask_tp1 = trans
            for k in range(K):
                states_t[k][i] = st[k]
                states_tp1[k][i] = stp1[k] if stp1 is not None else None
                server_ids[k][i] = sid[k]
                c_allocs[k][i] = c[k]
                rewards[i, k] = r[k]
            dones[i] = float(done)

        is_w = torch.tensor(is_weights, dtype=torch.float32, device=self.device)
        r_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        done_t = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # =========== Critic update ==========================================
        self.critic.train()
        obs_flat = self._encode_region_state_batch(states_t).detach()
        act_flat = self._encode_joint_action_batch(server_ids, c_allocs)
        # NaN/Inf protection on critic inputs (mirrors actor-side guard at L400-403)
        obs_flat = torch.nan_to_num(obs_flat, nan=0.0, posinf=0.0, neginf=0.0)
        act_flat = torch.nan_to_num(act_flat, nan=0.0, posinf=0.0, neginf=0.0)
        q_sa = self.critic(obs_flat, act_flat).squeeze(-1)          # [B]
        q_sa = torch.nan_to_num(q_sa, nan=0.0, posinf=0.0, neginf=0.0)
        with torch.no_grad():
            next_obs = self._encode_region_state_batch(states_tp1)
            next_act = self._compute_target_actions(states_tp1)
            next_obs = torch.nan_to_num(next_obs, nan=0.0, posinf=0.0, neginf=0.0)
            next_act = torch.nan_to_num(next_act, nan=0.0, posinf=0.0, neginf=0.0)
            q_next = self.critic_target(next_obs, next_act).squeeze(-1)
            q_next = torch.nan_to_num(q_next, nan=0.0, posinf=0.0, neginf=0.0)
            mean_r = r_t.mean(dim=1)      # team mean for target
            mean_r = torch.nan_to_num(mean_r, nan=0.0, posinf=0.0, neginf=0.0)
            y = mean_r + self.gamma * (1.0 - done_t) * q_next
        td_err = (y - q_sa).detach()
        # NaN protection: replace any NaN in td_err with 0
        td_err = torch.where(torch.isnan(td_err), torch.zeros_like(td_err), td_err)
        # Compute critic_loss with explicit nan_to_num on the residual
        residual = q_sa - y
        residual = torch.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)
        critic_loss = (is_w * residual.pow(2)).mean()
        # Diagnostic: warn (rate-limited) when critic_loss is non-finite
        if torch.isnan(critic_loss) or torch.isinf(critic_loss):
            if not hasattr(self, "_critic_nan_count"):
                self._critic_nan_count = 0
            self._critic_nan_count += 1
            if self._critic_nan_count % 200 == 1:
                print(f"[DGMA_paper][WARN] critic_loss non-finite "
                      f"(count={self._critic_nan_count}), substituting 0.0")
            critic_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        self.opt_critic.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.opt_critic.step()

        # =========== Actor update (per region, through critic) ==============
        actor_total = 0.0
        # Re-encode obs with gradient (for grad to flow into each actor)
        for k_train in range(K):
            # Fresh encodes for each region's training pass:
            # actor k_train is differentiable; others detached.
            K_local = K
            A_local = A
            ctx_list = []
            act_parts = []
            for k in range(K_local):
                datas = states_t[k]
                filled = []
                present_mask = torch.zeros(B, 1, device=self.device)
                for i, d in enumerate(datas):
                    if d is None:
                        from torch_geometric.data import Data
                        filled.append(Data(
                            x=torch.zeros((1, R1_FEATURE_DIM), dtype=torch.float32),
                            edge_index=torch.zeros((2, 0), dtype=torch.long),
                        ))
                    else:
                        filled.append(d); present_mask[i, 0] = 1.0
                batch = Batch.from_data_list(filled).to(self.device)
                if k == k_train:
                    s_logits, c_alloc, _ = self.actors[k](batch)
                    # Clamp logits to prevent extreme values → NaN in gumbel
                    s_logits = s_logits.clamp(-20.0, 20.0)
                    soft_a = F.gumbel_softmax(s_logits, tau=self.gumbel_tau,
                                              hard=False, dim=-1)   # [B, A]
                    ctx = fedg_readout(
                        self.actors[k].encoder(batch.x, batch.edge_index), batch)
                    ctx = torch.where(torch.isnan(ctx), torch.zeros_like(ctx), ctx)
                    ctx = ctx * present_mask
                    a_vec = torch.cat([soft_a, c_alloc], dim=-1)     # [B, A+1]
                else:
                    with torch.no_grad():
                        s_logits_o, c_alloc_o, _ = self.actors[k](batch)
                        s_logits_o = s_logits_o.clamp(-20.0, 20.0)
                        a_greedy = torch.argmax(s_logits_o, dim=-1)
                        one_hot = F.one_hot(a_greedy, num_classes=A_local).float()
                        ctx_o = fedg_readout(
                            self.actors[k].encoder(batch.x, batch.edge_index), batch)
                        ctx_o = torch.where(torch.isnan(ctx_o), torch.zeros_like(ctx_o), ctx_o)
                        ctx_o = ctx_o * present_mask
                    a_vec = torch.cat([one_hot, c_alloc_o], dim=-1)
                    ctx = ctx_o
                ctx_list.append(ctx)
                act_parts.append(a_vec)

            obs_flat_k = torch.cat(ctx_list, dim=-1)
            act_flat_k = torch.cat(act_parts, dim=-1)
            # NaN protection on inputs
            obs_flat_k = torch.where(torch.isnan(obs_flat_k),
                                     torch.zeros_like(obs_flat_k), obs_flat_k)
            act_flat_k = torch.where(torch.isnan(act_flat_k),
                                     torch.zeros_like(act_flat_k), act_flat_k)
            q_pi = self.critic(obs_flat_k, act_flat_k).squeeze(-1)   # [B]
            actor_loss = -(is_w * q_pi).mean()

            # Skip actor update if loss is NaN/inf
            if torch.isnan(actor_loss) or torch.isinf(actor_loss):
                actor_total += 0.0
                continue

            self.opt_actors[k_train].zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actors[k_train].parameters(),
                                     self.grad_clip)
            self.opt_actors[k_train].step()
            actor_total += float(actor_loss.item())

        # === PER priorities =====================================================
        td_np = td_err.detach().cpu().numpy()
        td_np = np.where(np.isfinite(td_np), td_np, 0.0)
        self.buffer.update_priorities(indices, td_np)

        # === Soft target update =================================================
        self._soft_update(self.critic_target, self.critic)
        for a, a_t in zip(self.actors, self.actors_target):
            self._soft_update(a_t, a)

        # === SPS: periodic parameter blending ===================================
        if self.sps_every > 0 and self.global_step % self.sps_every == 0:
            sps_apply(self.actors, self.sps_weights)

        self.last_actor_loss = actor_total / max(1, K)
        self.last_critic_loss = float(critic_loss.item())
        return self.last_actor_loss, self.last_critic_loss

    # ----------------------------------------------------------- save / load
    def state_dict(self) -> dict:
        return {
            "actors": [a.state_dict() for a in self.actors],
            "actors_target": [a.state_dict() for a in self.actors_target],
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
        }

    def load_state_dict(self, sd: dict):
        for a, s in zip(self.actors, sd["actors"]):
            a.load_state_dict(s)
        for a, s in zip(self.actors_target, sd["actors_target"]):
            a.load_state_dict(s)
        self.critic.load_state_dict(sd["critic"])
        self.critic_target.load_state_dict(sd["critic_target"])
