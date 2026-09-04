# -*- coding: utf-8 -*-
"""
V3 late-fusion Actor-Critic with dual GAT encoder (CPGAPPO network).

English
-------
CPGAPPOLateFusionActorCritic is the policy/critic network of the main CPGAPPO model.
Architecture:
  - Forward GAT (GATv2Conv, 4 heads, concat=False) over the DAG edges.
  - Backward GAT over reversed edges (disabled for the `fwdonly` ablation).
  - A separate global-resource MLP branch ingests the 10-d queue vector
    (local_q, upload_q, edge_q[0..7]) to break the Edge blindness of pure
    message passing.
  - Late fusion: context = [target_node_embed, g_mean, g_max, global_embed].
  - Actor/critic heads are 3-layer MLPs (context->256->128->out).
  - Optional AppTimeout auxiliary head (sigmoid logit, multi-task critic),
    enabled by use_appto_aux (used by the `noappcredit` ablation).

extract_cpgappo_state(): builds (PyG Data, 10-d global_feat, action_mask) for one
subtask. compute_cpgappo_slack_reward(): physical step reward = -0.05*E - 0.1*D
with extra penalty when slack < 0 (severe deadline miss) or slack < 20% of
the deadline budget (danger zone).

中文
----
CPGAPPOLateFusionActorCritic 是 CPGAPPO 主模型的策略/价值网络:
  - 正向 GAT + 反向 GAT (fwdonly 消融禁用反向), 独立全局资源 MLP 分支,
  - Late Fusion: context = [target, g_mean, g_max, global_embed],
  - 可选 AppTimeout 辅助头 (noappcredit 消融使用)。
extract_cpgappo_state 构造状态, compute_cpgappo_slack_reward 计算基于松弛度的物理奖励。
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool, global_add_pool
from utils.constant import para
# 引入刚刚修复后的 R1 图特征编码器
from Algorithms.Baselines.gat_ppo_encoder_r1 import encode_dag_gat_ppo_state_r1

class CPGAPPOLateFusionActorCritic(nn.Module):
    def __init__(self, node_dim=27, global_dim=10, hidden_dim=64, action_dim=10, use_backward=True,
                 use_appto_aux=False):
        super().__init__()
        self.use_backward = use_backward
        self.use_appto_aux = use_appto_aux  # [v2 main] 辅助 AppTimeout 预测头
        self.input_proj = nn.Linear(node_dim, hidden_dim)
        self.act = nn.ELU()
        
        # Forward GAT (always present)
        self.fwd_conv = GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=False)
        # Backward GAT (only when use_backward=True)
        if self.use_backward:
            self.bwd_conv = GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=False)
        
        # 新增：全局资源(排队)独立 MLP 分支 (破解 Edge 盲区)
        self.global_mlp = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU()
        )
        
        # node_embed_dim: h*2 if dual GAT, h*1 if fwd-only
        self.node_embed_dim = hidden_dim * 2 if self.use_backward else hidden_dim
        # Late Fusion 上下文维度: Target节点(node_embed_dim) + 图全局均值(node_embed_dim) + 图全局最大值(node_embed_dim) + 全局排队特征(h)
        context_dim = self.node_embed_dim * 3 + hidden_dim
        
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
        # [v2 main] 辅助 AppTimeout 预测头（DGMA-inspired multi-task critic）
        if self.use_appto_aux:
            self.appto_head = nn.Sequential(
                nn.Linear(context_dim, 128), nn.Tanh(),
                nn.Linear(128, 1)  # logit, 经 sigmoid 得超时概率
            )

    def forward(self, data, global_feat, return_aux=False):
        # 1. 走图分支处理拓扑
        x = self.act(self.input_proj(data.x))
        h_fwd = self.act(self.fwd_conv(x, data.edge_index) + x)
        if self.use_backward:
            rev_edge_index = data.edge_index[[1, 0], :]
            h_bwd = self.act(self.bwd_conv(x, rev_edge_index) + x)
            node_embeds = torch.cat([h_fwd, h_bwd], dim=-1)
        else:
            # [fwdonly ablation] 只用forward GAT，不创建backward分支
            node_embeds = h_fwd
        
        is_target = data.x[:, -1] > 0.5 # 最后一维是 is_cur
        mask_float = is_target.float().unsqueeze(1)
        target_embeds = global_add_pool(node_embeds * mask_float, data.batch) / global_add_pool(mask_float, data.batch).clamp_min(1.0)
        g_mean = global_mean_pool(node_embeds, data.batch)
        g_max = global_max_pool(node_embeds, data.batch)
        
        # 2. 走资源分支处理具体排队时间
        if global_feat.dim() == 1:
            global_feat = global_feat.unsqueeze(0)
        g_embed = self.global_mlp(global_feat)
        
        # 3. 强融合 (Late Fusion)
        context = torch.cat([target_embeds, g_mean, g_max, g_embed], dim=1)
        actor_out = self.actor(context)
        critic_out = self.critic(context)
        # [v2 main] 仅当 return_aux=True 且启用辅助头时返回 3 元组，向后兼容
        if return_aux and self.use_appto_aux:
            appto_logit = self.appto_head(context).squeeze(-1)
            return actor_out, critic_out, appto_logit
        return actor_out, critic_out

def load_state_dict_compat(model, state_dict, strict=False, verbose=True):
    """兼容加载器: 把旧 forward-only ckpt 部分加载到新 dual-GAT 模型。

    场景
    ----
    主模型 / 消融变体已从 forward-only GAT 改为 dual GAT (use_backward=True),
    但要复用旧的 forward-only ckpt (context_dim=4H=256, 无 bwd_conv.* 键).
    新 dual 模型 context_dim=7H=448.

    旧 context 布局 (4H=256):
        [h_fwd_target(H) | h_fwd_g_mean(H) | h_fwd_g_max(H) | g_embed(H)]
    新 context 布局 (7H=448):
        [h_fwd_target(H) | h_bwd_target(H) | h_fwd_g_mean(H) | h_bwd_g_mean(H) |
         h_fwd_g_max(H)  | h_bwd_g_max(H)  | g_embed(H)]

    策略
    ----
    - 若 state_dict 已含 bwd_conv.* 键 (已是 dual ckpt) → 直接 load_state_dict.
    - 若 model 是 dual 但 state_dict 是 forward-only:
        * fwd_conv / input_proj / global_mlp / actor.2+ / critic.2+ → 直接复用旧权重 (形状相同).
        * bwd_conv → 保留模型构造时的随机初始化 (不加载), 训练中学习.
        * actor.0 / critic.0 第一层: 旧权重 [out, 4H] 按列交错放入新 [out, 7H]:
            - h_fwd / g_embed 对应列 ← 旧权重对应块
            - h_bwd  对应列 ← 零初始化
            - bias 直接复制
          这样初始输出 = 旧模型输出 (h_bwd 列权重为 0, 贡献为 0), 反向信号在训练中学习.
    - 若 model 是 forward-only → 直接 load_state_dict (兼容旧 forward-only ckpt).
    """
    sd = state_dict
    model_is_dual = getattr(model, 'use_backward', False)
    sd_has_bwd = any(k.startswith('bwd_conv.') for k in sd.keys())

    # Case 1: state_dict 已是 dual, 或 model 是 forward-only → 直接加载
    if sd_has_bwd or not model_is_dual:
        if verbose:
            mode = "dual-ckpt→dual-model" if sd_has_bwd else "fwd-ckpt→fwd-model"
            print(f"[load_state_dict_compat] {mode}: 直接 load_state_dict(strict={strict})")
        model.load_state_dict(sd, strict=strict)
        return model

    # Case 2: model 是 dual, state_dict 是 forward-only → 兼容加载
    # 从 actor.0.weight 形状推断 H (hidden_dim)
    old_actor0 = sd.get('actor.0.weight', None)
    new_actor0 = dict(model.named_parameters()).get('actor.0.weight', None)
    if old_actor0 is None or new_actor0 is None:
        if verbose:
            print("[load_state_dict_compat] WARNING: 找不到 actor.0.weight, 回退 strict=False 直接加载")
        model.load_state_dict(sd, strict=False)
        return model

    H_old = old_actor0.shape[1] // 4   # 旧 context = 4H
    H_new = new_actor0.shape[1] // 7   # 新 context = 7H
    if H_old != H_new or old_actor0.shape[1] != 4 * H_old or new_actor0.shape[1] != 7 * H_new:
        if verbose:
            print(f"[load_state_dict_compat] WARNING: 维度不匹配 old_ctx={old_actor0.shape[1]} "
                  f"new_ctx={new_actor0.shape[1]} H_old={H_old} H_new={H_new}, 回退 strict=False 直接加载")
        model.load_state_dict(sd, strict=False)
        return model
    H = H_new

    # 旧 4 个 H-块 → 新中 h_fwd / g_embed 对应的 4 个 H-块 (跳过 h_bwd 块)
    old_blocks = [(0, H), (H, 2 * H), (2 * H, 3 * H), (3 * H, 4 * H)]
    new_fwd_blocks = [(0, H), (2 * H, 3 * H), (4 * H, 5 * H), (6 * H, 7 * H)]

    new_sd = {}
    for k, v in sd.items():
        if k in ('actor.0.weight', 'critic.0.weight'):
            out_dim = v.shape[0]
            new_w = torch.zeros(out_dim, 7 * H, dtype=v.dtype)
            for (o_s, o_e), (n_s, n_e) in zip(old_blocks, new_fwd_blocks):
                new_w[:, n_s:n_e] = v[:, o_s:o_e]
            new_sd[k] = new_w
        elif k in ('actor.0.bias', 'critic.0.bias'):
            new_sd[k] = v  # bias 直接复制
        elif k.startswith('bwd_conv.'):
            continue  # 防御性跳过 (forward-only ckpt 不应有)
        elif k.startswith('appto_head.'):
            if getattr(model, 'use_appto_aux', False):
                new_sd[k] = v
            # else: model 无 appto_head, 跳过
        else:
            # fwd_conv / input_proj / global_mlp / actor.2+ / critic.2+ → 直接复用
            new_sd[k] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    # bwd_conv.* 应在 missing 中 (预期, 保留随机初始化); appto_head 若 model 无也在 missing
    unexpected_real = [k for k in unexpected]
    missing_real = [k for k in missing
                    if not k.startswith('bwd_conv.') and not k.startswith('appto_head.')]
    if verbose:
        print(f"[load_state_dict_compat] fwd-ckpt→dual-model 兼容加载完成. "
              f"H={H}, old_ctx={4*H}, new_ctx={7*H}")
        print(f"  复用旧权重 keys: {len(new_sd)} 个 (含 actor.0/critic.0 交错重建)")
        print(f"  bwd_conv 保留随机初始化 (missing 中 bwd_conv.* 键 = "
              f"{[k for k in missing if k.startswith('bwd_conv.')][:3]}... )")
        if unexpected_real:
            print(f"  WARNING unexpected keys: {unexpected_real}")
        if missing_real:
            print(f"  WARNING unexpected missing keys: {missing_real}")
    return model


def extract_cpgappo_state(ts, task_tuple, slot, task_complex_index=0):
    """提取状态：分离图特征和全局排队特征（原始10维版本）"""
    data, mask = encode_dag_gat_ppo_state_r1(ts, task_tuple, slot=slot, now_time=None)
    uid = task_tuple[0]
    now = slot * para["slot_interval"]
    local_q = max(0.0, float(ts.devices_exe_useful[uid]) - now)
    up_q = max(0.0, float(ts.devices_upload_useful[uid]) - now)
    edge_qs = [max(0.0, float(ts.remain_times[eid]) - now) for eid in range(para["edge_num"])]
    global_features = torch.tensor([local_q, up_q] + edge_qs, dtype=torch.float32) / 2.0
    return data, global_features, mask

def compute_cpgappo_slack_reward(ts, uid, sid, energy, delay, enter_time, deadline_slot):
    """计算基于松弛度 (Slack) 消耗的物理奖励"""
    TF = ts.finish_time[uid][sid]
    app_deadline_abs = enter_time + deadline_slot * para["slot_interval"]
    slack = app_deadline_abs - TF
    
    r_step = - (0.05 * energy) - (0.1 * delay) # 基础惩罚
    
    if slack < 0:
        r_step -= (5.0 + abs(slack) * 2.0) # 严重超时：大额扣分
    elif slack < 0.2 * (app_deadline_abs - enter_time):
        r_step -= 1.0 # 逼近危险区：警告扣分
        
    return r_step, float(slack)
