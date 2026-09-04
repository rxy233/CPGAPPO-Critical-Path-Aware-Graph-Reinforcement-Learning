# -*- coding: utf-8 -*-
"""
Dynamic GAT-DQN 算法核心实现（修复版）
修复：
1. 移除 50% Local 偏见，恢复纯随机探索。
2. 优化 Mask 处理逻辑。
3. 增强 Learn 函数的鲁棒性。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch

# ================= 配置 =================
BATCH_SIZE = 256  # 【加速优化】增大 Batch Size 到 256，减少更新次数，提升梯度稳定性
LR = 5e-5       # 降低 LR 防止震荡
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 50000  # 延长 epsilon 衰减到 50000 步，让 Agent 前期多探索
MASK_INVALID = -1e9
TARGET_UPDATE_INTERVAL = 500 # 加快 Target 更新

class DynamicGATNetworkAllNodes(nn.Module):
    def __init__(self, node_input_dim: int, action_dim: int, hidden_dim: int = 64, heads: int = 2):
        super().__init__()
        self.action_dim = action_dim
        # GAT 层：输入 -> Hidden
        self.gat1 = GATv2Conv(node_input_dim, hidden_dim, heads=heads, concat=False, add_self_loops=False)
        self.gat2 = GATv2Conv(hidden_dim, hidden_dim, heads=1, concat=False, add_self_loops=False)

        # Dueling Network 结构
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.adv_stream = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x, edge_index, batch):
        # 显存优化：使用 ELU 激活函数
        h = F.elu(self.gat1(x, edge_index))
        h = F.elu(self.gat2(h, edge_index))

        # 图级 Embedding
        g = global_mean_pool(h, batch)
        v = self.value_stream(g) # [Batch, 1]

        # 节点级 Embedding + 图级 Embedding 拼接
        # h: [Total_Nodes, Hidden], g[batch]: [Total_Nodes, Hidden]
        # 这样每个节点都知道所属图的全局信息
        cat_feat = torch.cat([h, g[batch]], dim=-1)
        a = self.adv_stream(cat_feat) # [Total_Nodes, Action]

        # Dueling 合并: Q = V + (A - mean(A))
        # v[batch] 将图价值广播到每个节点
        return v[batch] + (a - a.mean(dim=1, keepdim=True))

class ReplayMemory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.expert_flags = []

    def push(self, *args, is_expert=False, **kwargs):
        self.buffer.append(args)
        self.expert_flags.append(bool(is_expert))
        while len(self.expert_flags) > len(self.buffer):
            self.expert_flags.pop(0)

    def sample(self, batch_size):
        indices = random.sample(range(len(self.buffer)), min(batch_size, len(self.buffer)))
        batch = [self.buffer[i] for i in indices]
        return batch

    def __len__(self): return len(self.buffer)

class DynamicGAT_DQN_Agent:
    def __init__(self, node_feature_dim, action_dim, device, use_expert_data=False, expert_ratio=0.3):
        self.action_dim = action_dim
        self.device_train = device
        self.device_infer = torch.device('cpu')

        from utils.constant import para
        self.max_apps = int(para["user_num"]) + 1

        self.policy_gpu = DynamicGATNetworkAllNodes(node_feature_dim, action_dim, heads=2).to(self.device_train)
        self.target_gpu = DynamicGATNetworkAllNodes(node_feature_dim, action_dim, heads=2).to(self.device_train)
        self.target_gpu.load_state_dict(self.policy_gpu.state_dict())
        self.target_gpu.eval()

        self.policy_cpu = DynamicGATNetworkAllNodes(node_feature_dim, action_dim, heads=2).to(self.device_infer)
        self.policy_cpu.load_state_dict(self.policy_gpu.state_dict())
        self.policy_cpu.eval()

        self.optimizer = optim.AdamW(self.policy_gpu.parameters(), lr=LR)
        # 增大 Buffer 容量到 50k，减少分布漂移
        self.memory = ReplayMemory(50000)
        self.learn_step_counter = 0
        self.steps_done = 0
        # 【加速优化】减少 warmup_steps 从 5000 到 3000，更快开始训练
        self.warmup_steps = 3000
        self.has_warmed_up = False

        # 【加速优化】减少 CPU-GPU 同步频率
        self.sync_interval = 20  # 从 10 增加到 20
        self.target_sync_interval = 200  # 从 100 增加到 200

        # 专家相关（暂时保留接口，建议禁用预训练）
        self.use_expert_data = use_expert_data
        self.expert_ratio = expert_ratio
        self.expert_memory = ReplayMemory(5000) if use_expert_data else None

    def select_action(self, state_data, ready_mask, action_mask=None, training=True, custom_eps=None):
        """
        选择动作

        Args:
            state_data: (x, edge_index, batch)
            ready_mask: bool tensor, 标记哪些节点是 ready 的
            action_mask: action mask, 标记哪些动作不可用
            training: bool, 是否在训练模式
            custom_eps: float or None, 如果不为 None，强制使用该 epsilon 值
                               （用于评估时保留微小的随机性，避免 Argmax 陷阱）
        """
        # 诊断：打印实际输入的维度
        x, edge_index, batch = state_data

        # 修复添加维度检查，避免重复拼接 context 导致维度不匹配
        expected_dim = self.policy_cpu.gat1.in_channels
        if x.size(1) != expected_dim:
            raise RuntimeError(
                f"[select_action] x_dim mismatch: got {x.size(1)}, expected {expected_dim}\n"
                f"  可能原因：get_global_graph_state_dag 中 context 被拼接了多次"
            )

        # 支持自定义 epsilon（用于评估时保留微小的随机性）
        # 如果 custom_eps 不为 None，强制使用该值，否则按原逻辑计算
        if custom_eps is not None:
            eps_threshold = custom_eps
        elif training:
            eps_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                            np.exp(-1.0 * self.steps_done / EPSILON_DECAY)
            self.steps_done += 1
        else:
            eps_threshold = 0.0

        # 1. 随机探索（只要 eps_threshold > 0，就有随机性）
        if random.random() < eps_threshold:
            ready_indices = torch.nonzero(ready_mask).squeeze()
            if ready_indices.numel() == 0: return None, None

            # 随机选一个 Ready 节点
            if ready_indices.dim() == 0: task_node_idx = ready_indices.item()
            else: task_node_idx = ready_indices[random.randint(0, len(ready_indices) - 1)].item()

            # 随机选一个 Valid 动作
            if action_mask is not None:
                if action_mask.dim() == 2: task_mask = action_mask[task_node_idx]
                else: task_mask = action_mask

                # 修复纯随机选择，移除 Local 偏见
                # valid_actions = indices where mask > large_negative
                valid_actions = torch.nonzero(task_mask > -1e5).squeeze()

                if valid_actions.numel() == 0: offload_action = 0 # 保底
                elif valid_actions.numel() == 1:
                    if valid_actions.dim() == 0: offload_action = valid_actions.item()
                    else: offload_action = valid_actions[0].item()
                else:
                    # 均匀采样
                    rand_idx = random.randint(0, len(valid_actions) - 1)
                    offload_action = valid_actions[rand_idx].item()
            else:
                offload_action = random.randint(0, self.action_dim - 1)

            return task_node_idx, offload_action

        # 2. 模型决策
        with torch.no_grad():
            x, edge_index, batch = state_data
            # 【内存优化】只在必要时移动数据到 device_infer，避免不必要的拷贝
            if x.device != self.device_infer:
                x = x.to(self.device_infer)
            if edge_index.device != self.device_infer:
                edge_index = edge_index.to(self.device_infer)
            if batch.device != self.device_infer:
                batch = batch.to(self.device_infer)
            if ready_mask.device != self.device_infer:
                ready_mask = ready_mask.to(self.device_infer)

            q_values = self.policy_cpu(x, edge_index, batch)

            if ready_mask.dim() != 1: ready_mask = ready_mask.bool().flatten()
            ready_q_values = q_values[ready_mask] # 只看 Ready 节点的 Q 值

            if ready_q_values.shape[0] == 0: return None, None

            # 应用 Mask
            if action_mask is not None:
                action_mask = action_mask.to(self.device_infer)
                if action_mask.dim() == 2: ready_action_mask = action_mask[ready_mask]
                else: ready_action_mask = action_mask.unsqueeze(0).expand(ready_q_values.shape[0], -1)
                ready_q_values = ready_q_values + ready_action_mask

            # 全局 Argmax：选择 (哪个节点, 哪个动作) Q 值最大
            # flatten argmax
            best_flat_idx = ready_q_values.argmax().item()
            best_ready_idx = best_flat_idx // self.action_dim
            offload_action = best_flat_idx % self.action_dim

            # 映射回 全局节点索引
            all_ready_nodes = torch.nonzero(ready_mask).squeeze()
            if all_ready_nodes.dim() == 0: task_node_idx = all_ready_nodes.item()
            else: task_node_idx = all_ready_nodes[best_ready_idx].item()

            return task_node_idx, offload_action

    def learn(self, updates=1):
        # 【加速优化】添加 warmup 机制，收集足够数据后才开始学习
        # 减少warmup_steps从5000到3000，更快开始训练
        if not self.has_warmed_up:
            if len(self.memory) < self.warmup_steps:
                return None
            else:
                self.has_warmed_up = True
                print(f"[DynamicGATDQN] Warmup complete! Starting training with {len(self.memory)} samples.")
        
        if len(self.memory) < BATCH_SIZE: return None
        device = self.device_train

        # 简单的采样（暂时禁用 Expert 混合，先让它自己学明白）
        transitions = self.memory.sample(BATCH_SIZE)

        # 数据解包与预处理
        data_list, next_data_list = [], []
        ready_list, next_ready_list = [], []
        mask_list, next_mask_list = [], []
        node_idx_list, offload_list, reward_list, done_list = [], [], [], []

        for t in transitions:
            # t: (state, node_idx, action, reward, next_state, done, ready, next_ready, mask, next_mask)
            state, n_idx, off, r, n_state, done, ready, n_ready, mask, n_mask = t[:10]

            x, ei, ab = state
            nx, nei, nab = n_state

            data_list.append(Data(x=x, edge_index=ei, app_batch=ab.to(torch.long)))
            next_data_list.append(Data(x=nx, edge_index=nei, app_batch=nab.to(torch.long)))

            ready_list.append(ready)
            next_ready_list.append(n_ready)
            mask_list.append(mask)
            next_mask_list.append(n_mask)

            node_idx_list.append(n_idx)
            offload_list.append(off)
            reward_list.append(r)
            done_list.append(done)

        total_loss = 0

        for _ in range(updates):
            try:
                batch_s = Batch.from_data_list(data_list).to(device)
                batch_ns = Batch.from_data_list(next_data_list).to(device)

                # 重构 Batch 索引 (PyG Batch 会重排 batch 属性)
                # 我们需要 pool_batch 来做 global_mean_pool
                pool_batch_s = batch_s.batch
                pool_batch_ns = batch_ns.batch

                # 处理 Mask 对齐 (核心痛点)
                # mask_list[i] 是第 i 个图的 mask [Nodes_i, Actions]
                # 我们需要把它们拼成 [Total_Nodes_Batch, Actions]
                # 【加速优化】先在 CPU 上 cat，再一次性搬到 GPU（减少 PCI-E 通信次数）
                batch_mask = torch.cat(mask_list, dim=0).to(device)
                batch_next_mask = torch.cat(next_mask_list, dim=0).to(device)

                batch_next_ready = torch.cat(next_ready_list, dim=0).to(device)

                # 准备其他 Tensor
                # node_idx_list 存的是【图内偏移】，在 Batch 中需要加上 ptr 偏移
                # batch_s.ptr = [0, N1, N1+N2, ...]
                batch_ptr = batch_s.ptr[:-1].to(device) # 去掉最后一个总数
                # 这里的 node_idx_list 是 list of int
                batch_node_idx = torch.tensor(node_idx_list, device=device) + batch_ptr

                batch_act = torch.tensor(offload_list, device=device)
                batch_rew = torch.tensor(reward_list, device=device)
                # 修复确保 batch_done 是 float 类型，避免后续 1 - bool_tensor 错误
                batch_done = torch.tensor(done_list, device=device).float()

                # ===1. Current Q ===
                self.policy_gpu.train()
                # Q_all: [Total_Nodes_Batch, Actions]
                q_all = self.policy_gpu(batch_s.x, batch_s.edge_index, pool_batch_s)

                # Gather Q(s, node, action)
                # batch_node_idx 指向了具体的那个被操作的节点
                current_q = q_all[batch_node_idx, batch_act]

                # ===2. Target Q (修复维度问题：使用global_max_pool聚合节点级value) ===
                with torch.no_grad():
                    # Step 1: 计算 Next State 所有节点的 Q 值
                    q_next_all = self.policy_gpu(batch_ns.x, batch_ns.edge_index, pool_batch_ns)
                    # q_next_all: [Total_Next_Nodes, Actions]

                    # Step 2.1: 找到每个节点的最优 Action Value
                    # node_max_val: [Total_Next_Nodes]
                    node_max_val, _ = q_next_all.max(dim=1)

                    # Mask 掉非 Ready 节点的 Value
                    node_max_val[~batch_next_ready] = -1e9

                    # Step 2.2: 修复将节点级 Value 聚合为图级 Value
                    # 我们需要在每个图中找到 Value 最大的那个节点（最适合卸载任务的节点）
                    # global_max_pool 需要输入 [N, Channels]，所以先 unsqueeze
                    node_max_val_2d = node_max_val.view(-1, 1)  # [Total_Nodes, 1]

                    # graph_max_val_2d: [Batch_Size, 1]
                    graph_max_val_2d = global_max_pool(node_max_val_2d, pool_batch_ns)

                    # 变回 1D: [Batch_Size]
                    graph_max_val = graph_max_val_2d.view(-1)

                    # 处理空图/全Mask情况 (即 max 为 -1e9 的情况)
                    # 如果全是负无穷，说明没动作可选，Value=0
                    graph_max_val[graph_max_val < -1e5] = 0.0

                    # Step 2.3: Bellman Update
                    # target_q: [Batch_Size]
                    target_q_batch = batch_rew + (1.0 - batch_done) * GAMMA * graph_max_val


                # ===3. Loss ===
                # 现在 current_q (128) 和 target_q_batch (128) 维度一致
                loss = F.smooth_l1_loss(current_q, target_q_batch)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_gpu.parameters(), 1.0)
                self.optimizer.step()

                total_loss += loss.item()

            except RuntimeError as e:
                if 'out of memory' in str(e).lower() or 'defaultcpuallocator' in str(e).lower():
                    # 【内存优化】更详细的错误日志和更积极的清理
                    print(f"[DynamicGATDQN Warning] OOM detected, clearing cache...")
                    torch.cuda.empty_cache()
                    # 尝试强制 Python GC
                    import gc
                    gc.collect()
                    return None
                print(f"[DynamicGATDQN Error] {e}")
                return None

        self.learn_step_counter += 1

        # 【加速优化】减少 CPU 模型同步频率（从 10 增加到 20）
        if self.learn_step_counter % self.sync_interval == 0:
            self.policy_cpu.load_state_dict(self.policy_gpu.state_dict())
            self.policy_cpu.eval()

        # 【加速优化】减少 Target 更新频率（从 100 增加到 200）
        if self.learn_step_counter % self.target_sync_interval == 0:  
             self.target_gpu.load_state_dict(self.policy_gpu.state_dict())

        return total_loss / updates

    def save(self, path):
        torch.save({'policy_gpu': self.policy_gpu.state_dict()}, path)
