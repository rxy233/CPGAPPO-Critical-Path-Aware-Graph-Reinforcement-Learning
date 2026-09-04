"""
TransEdge-style PPO Agent
复用现有稳定PPO训练框架，但使用forward-only GCN backbone
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch_geometric.data import Batch
from .transedge_style_core import TransEdgeStyleActorCritic, R1_FEATURE_DIM


class TransEdgeStylePPOAgent:
    """
    PPO agent with TransEdge-style GCN backbone
    关键特性：
    - 无global_feat（与V3区别）
    - Forward-only GCN（与Dual-GAT区别）
    - 无guide机制
    """
    
    def __init__(
        self,
        node_dim=R1_FEATURE_DIM,
        action_dim=10,
        device='cpu',
        lr=3e-4,
        gamma=0.99,
        lmbda=0.95,
        eps_clip=0.2,
        K_epochs=4,
        entropy_coef=0.01
    ):
        self.device = device
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.entropy_coef = entropy_coef
        
        # Policy network
        self.policy = TransEdgeStyleActorCritic(
            node_dim=node_dim,
            action_dim=action_dim
        ).to(device)
        
        self.optimizer = optim.AdamW(self.policy.parameters(), lr=lr, weight_decay=1e-5)
        self.mse_loss = nn.MSELoss()
        
        self.memory = []
        self.inference_times = []
    
    def put_data(self, item):
        """Store transition: (state_data, action, reward, log_prob, done, mask)"""
        self.memory.append(item)
    
    def clear_memory(self):
        self.memory = []
    
    def set_episode(self, ep):
        """No-op compat for dividelong wrapper"""
        pass

    def take_action(self, state_data, action_mask=None, deterministic=False, route_stats=None):
        """
        Select action with invalid-action masking

        Args:
            state_data: PyG Data object
            action_mask: list of 0/1 (invalid/valid)
            deterministic: if True, use argmax; else sample
            route_stats: ignored (compat param, no guide mechanism)

        Returns:
            action: int
            log_prob: float
        """
        start_time = time.time()
        self.policy.eval()
        
        with torch.no_grad():
            batch = Batch.from_data_list([state_data]).to(self.device)
            logits, _ = self.policy(batch)
            logits = logits.squeeze(0)  # [action_dim]
            
            # Masking
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
        """
        PPO update with GAE advantage estimation
        """
        if not self.memory:
            return 0.0
        
        # Unpack memory
        states = [x[0] for x in self.memory]
        actions = torch.tensor([x[1] for x in self.memory], dtype=torch.long).to(self.device)
        rewards = [x[2] for x in self.memory]
        old_log_probs = torch.tensor([x[3] for x in self.memory], dtype=torch.float32).to(self.device)
        dones = [x[4] for x in self.memory]
        masks = [x[5] for x in self.memory]
        
        # Compute values
        batch_size = 64
        values = []
        self.policy.eval()
        with torch.no_grad():
            for i in range(0, len(states), batch_size):
                b_states = Batch.from_data_list(states[i:i+batch_size]).to(self.device)
                _, v = self.policy(b_states)
                values.append(v.squeeze(-1).cpu())
        values = torch.cat(values)
        
        # GAE computation
        returns, advantages = [], []
        gae = 0
        for i in reversed(range(len(rewards))):
            next_val = 0 if i == len(rewards)-1 else values[i+1]
            if dones[i]:
                gae = 0
                next_val = 0
            delta = rewards[i] + self.gamma * next_val - values[i]
            gae = delta + self.gamma * self.lmbda * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])
        
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        
        # Normalize advantages
        if advantages.std() > 1e-6:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
        
        # PPO epochs
        total_loss = 0
        T = len(states)
        perm = np.random.permutation(T)
        
        self.policy.train()
        for _ in range(self.K_epochs):
            for i in range(0, T, batch_size):
                idx = perm[i:i+batch_size]
                
                b_states = Batch.from_data_list([states[j] for j in idx]).to(self.device)
                b_actions = actions[idx]
                b_returns = returns[idx]
                b_adv = advantages[idx]
                b_old_lp = old_log_probs[idx]
                b_masks = torch.as_tensor(np.stack([masks[j] for j in idx]), dtype=torch.bool, device=self.device)
                
                # Forward
                logits, v_pred = self.policy(b_states)
                v_pred = v_pred.squeeze(-1)
                
                # Masking
                neg_inf = torch.finfo(logits.dtype).min
                logits = logits.masked_fill(~b_masks, neg_inf)
                probs = torch.softmax(logits, dim=-1)
                dist = Categorical(probs)
                
                # Loss
                new_log_probs = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(new_log_probs - b_old_lp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * b_adv
                
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = self.mse_loss(v_pred, b_returns)
                loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                
                total_loss += loss.item()
        
        return total_loss
    
    def print_inference_stats(self):
        """Print inference time statistics"""
        if not self.inference_times:
            print("[TransEdge Agent] No inference records")
            return
        
        arr = np.array(self.inference_times)
        print(f"[TransEdge Agent] Inference stats ({len(arr)} calls):")
        print(f"  - avg: {arr.mean():.4f} ms")
        print(f"  - median: {np.median(arr):.4f} ms")
        print(f"  - min: {arr.min():.4f} ms")
        print(f"  - max: {arr.max():.4f} ms")
        print(f"  - P95: {np.percentile(arr, 95):.4f} ms")
        print(f"  - P99: {np.percentile(arr, 99):.4f} ms")
