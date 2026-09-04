# -*- coding: utf-8 -*-
"""
训练模块 - 导出所有算法的训练包装函数
【可移植化】所有子模块 import 都用 try/except, 缺失模块不阻塞整个包加载,
只让真正被调用的 wrapper 在缺失时报错。这样仓库可以只保留论文实验需要的子集。
"""

# 共享函数 (common.py 是核心, 必须能 import)
from .common import set_print_lock, run_benchmark_worker

# 各算法训练函数 — 全部延迟/容错导入, 缺失时设为 None
def _try_import(mod_name, attr_name):
    try:
        mod = __import__(f".{mod_name}", package=__name__, fromlist=[attr_name])
        return getattr(mod, attr_name, None)
    except Exception:
        return None


train_dqn_wrapper = _try_import("train_dqn", "train_dqn_wrapper")
train_gat_ppo_wrapper = _try_import("train_gat_ppo", "train_gat_ppo_wrapper")
train_ppo_wrapper = _try_import("train_ppo", "train_ppo_wrapper")
train_dynamic_gat_dqn_wrapper = _try_import("train_dynamic_gat_dqn", "train_dynamic_gat_dqn_wrapper")
train_actor_learner_wrapper = _try_import("train_dynamic_gat_dqn_actor_learner", "train_actor_learner_wrapper")


__all__ = [
    'set_print_lock',
    'run_benchmark_worker',
    'train_dqn_wrapper',
    'train_gat_ppo_wrapper',
    'train_ppo_wrapper',
    'train_dynamic_gat_dqn_wrapper',
    'train_actor_learner_wrapper',
]
