import glob as _glob
# -*- coding: utf-8 -*-
"""
_reeval_lib.py ?Table V 消融?D_all 重评估共享库
=========================================================================
用途：复用已有 checkpoint，对 7 变体 × 3 拓扑 × 5 seeds ?re-eval?
     把时延口径从 D_succ 改为 D_all（超时应用按 per-app deadline 预算计入），
     UtilityScore 输入改用 D_all?
设计要点?
- 不重训：仅加?.pt ?调原 variant eval 函数（拿 D_succ）→ 同一 ts 状态下?
  ts.get_avg_results(only_successful=False, timeout_charge="deadline") ?D_all?
- 不修?variant 训练模块：仅?task_scheduler.get_avg_results 加了向后兼容参数?
- 输出 result_reeval.json 到原 run dir，不覆盖?result.json?
- ?_reeval_seed{1,3,42,5,7}.py import，每 seed 一个进程?
时延口径：D_i = min(D_i, TD_i^max - TS_i)
  ?超时/未完成应?delay = get_app_deadline_slot(uid) * slot_interval（相对预算，秒）
"""
import os
import sys
import json
import time
import traceback
# 确保项目根在 sys.path
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
import numpy as np
import torch
from pathlib import Path
from Experiments_new.exp_utils import (
    CONFIG, init_worker, generate_arrival_plan, load_deadline_config,
    compute_score, compute_utility_score, calc_timeout_rate,
    safe_rest_tasks_total, all_arrived_done, get_graph_cache,
    apply_arrival_plan, subtask_partition_stats,
)
from Environment.environment import Environment
from scheduler.graph_scheduler import GraphScheduler
from scheduler.task_scheduler import TaskScheduler
from Algorithms.Benchmark import Benchmark as BenchmarkClass
from utils.constant import para
# 按需导入 variant eval 函数（延迟到 reeval_one 内按 variant 选择，避免一次?import 全部?
from Algorithms.RealGATPPO.agent_cpgappo import GAT_PPO_Agent_CPGAPPO
from Algorithms.RealGATPPO.cpgappo_core import extract_cpgappo_state, load_state_dict_compat
# ================= 拓扑 ?(run_dir_base, matrix_path) =================
TOPOLOGIES = ["default", "chain", "wide"]
# 动态找 results/ 下 ablation_classic_{topo}_* 批次目录;
# 找不到 (fresh clone) 则回退 pretrained/{topo} (reeval_one 会从 pretrained 复制 ckpt)。
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def _resolve_run_dir_base_portable(topology):
    """从 results/ 下找最新的 ablation_classic_{topology}_* 批次; 找不到回退 pretrained/{topo}"""
    base = os.path.join(_REPO_ROOT, "results")
    cands = sorted(_glob.glob(os.path.join(base, f"ablation_classic_{topology}_*")))
    if cands:
        return cands[-1]
    return os.path.join(_REPO_ROOT, "pretrained", topology)
TOPOLOGY_CONFIG_FALLBACK = {
    "default": {
        "run_dir_base": _resolve_run_dir_base_portable("default"),
        "matrix_path": os.path.normpath(os.path.join(_REPO_ROOT, "matrix", "matrix_60.txt")),
    },
    "chain": {
        "run_dir_base": _resolve_run_dir_base_portable("chain"),
        "matrix_path": os.path.normpath(os.path.join(_REPO_ROOT, "matrix", "matrix_60_chain.txt")),
    },
    "wide": {
        "run_dir_base": _resolve_run_dir_base_portable("wide"),
        "matrix_path": os.path.normpath(os.path.join(_REPO_ROOT, "matrix", "matrix_60_wide.txt")),
    },
}
def _load_topology_config_for_seed(seed):
    """
    优先从 _retrain_manifest_seed{N}.json（最新一轮 retrain 的输出目录）查找。
    找不到则回退到原 20260608 批次。
    Returns:
        dict: 与原 TOPOLOGY_CONFIG 相同结构 {topology: {run_dir_base, matrix_path}}
    """
    manifest_path = os.path.join(_PROJECT_ROOT, f"_retrain_manifest_seed{seed}.json")
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                m = json.load(f)
            cfg = {}
            for t in TOPOLOGIES:
                if t in m.get("run_dir_bases", {}):
                    cfg[t] = {
                        "run_dir_base": m["run_dir_bases"][t],
                        "matrix_path": m.get("matrix_paths", {}).get(t, TOPOLOGY_CONFIG_FALLBACK[t]["matrix_path"]),
                    }
                else:
                    cfg[t] = dict(TOPOLOGY_CONFIG_FALLBACK[t])
            print(f"[_reeval_lib] 使用 retrain manifest: {manifest_path}  (ts={m.get('ts_tag')})")
            return cfg
        except Exception as ex:
            print(f"[_reeval_lib] ?manifest 失败，回退原批? {ex}")
    # 回退
    print(f"[_reeval_lib] 未找?retrain manifest，使用原 20260608 批次")
    return {t: dict(TOPOLOGY_CONFIG_FALLBACK[t]) for t in TOPOLOGIES}
# ================= 变体分派?=================
# variant ?(eval_module, eval_fn_name, ckpt_filename_best, ckpt_filename_last,
#            use_backward, use_shield, use_cp)
# 【CPGAPPO 统一消融】全?7 个变体统一指向 Algorithms.Train.train_cpgappo_unified
#   ?eval_cpgappo_once, 每个变体只翻一个开?
#   CPGAPPO 基准: 双向 GAT + CP 加权信用 + 软化 shield + lambda=0.1 + entropy=0.02.
# ? "main" 是外?RL 对比基线 (无消融开?, ?CPGAPPO 共享 eval 函数.
VARIANTS = ["main", "CPGAPPO", "noguidece", "noshield", "noappcredit", "nocp", "fwdonly", "alloff"]
VARIANT_CONFIG = {
    "main": {
        "eval_module": "Algorithms.Train.train_cpgappo_unified",
        "eval_fn_name": "eval_cpgappo_once",
        "ckpt_best": "CPGAPPO.pt",
        "ckpt_last": "CPGAPPO_last.pt",
        "use_backward": True,
        "use_shield": True,
        "use_cp": True,
    },
    "CPGAPPO": {
        "eval_module": "Algorithms.Train.train_cpgappo_unified",
        "eval_fn_name": "eval_cpgappo_once",
        "ckpt_best": "CPGAPPO.pt",
        "ckpt_last": "CPGAPPO_last.pt",
        "use_backward": True,
        "use_shield": True,
        "use_cp": True,
    },
    "noguidece": {
        # CPGAPPO 关闭 Guide CE (lambda_guide=0.0), 其余同基?
        "eval_module": "Algorithms.Train.train_cpgappo_unified",
        "eval_fn_name": "eval_cpgappo_once",
        "ckpt_best": "CPGAPPO_noguidece.pt",
        "ckpt_last": "CPGAPPO_noguidece_last.pt",
        "use_backward": True,
        "use_shield": True,
        "use_cp": True,
    },
    "noshield": {
        # CPGAPPO 关闭 Safety Shield (?mask_bin), 其余同基?
        "eval_module": "Algorithms.Train.train_cpgappo_unified",
        "eval_fn_name": "eval_cpgappo_once",
        "ckpt_best": "CPGAPPO_noshield.pt",
        "ckpt_last": "CPGAPPO_noshield_last.pt",
        "use_backward": True,
        "use_shield": False,   # 该变体开? eval ?mask_bin
        "use_cp": True,
    },
    "noappcredit": {
        # CPGAPPO 关闭 App Credit (eval 不发 bonus; eval 本就不发, 训练时关), 其余同基?
        "eval_module": "Algorithms.Train.train_cpgappo_unified",
        "eval_fn_name": "eval_cpgappo_once",
        "ckpt_best": "CPGAPPO_noappcredit.pt",
        "ckpt_last": "CPGAPPO_noappcredit_last.pt",
        "use_backward": True,
        "use_shield": True,
        "use_cp": True,
    },
    "nocp": {
        # CPGAPPO 关闭 CP 排序 (sort_tasks=False), 其余同基?
        "eval_module": "Algorithms.Train.train_cpgappo_unified",
        "eval_fn_name": "eval_cpgappo_once",
        "ckpt_best": "CPGAPPO_nocp.pt",
        "ckpt_last": "CPGAPPO_nocp_last.pt",
        "use_backward": True,
        "use_shield": True,
        "use_cp": False,   # 该变体开? sort_tasks=False
    },
    "fwdonly": {
        # CPGAPPO 关闭 Backward GAT (use_backward=False, 前向 only), 其余同基?
        # 这是【唯一】use_backward=False 的变? 消融维度 = GAT 方向.
        "eval_module": "Algorithms.Train.train_cpgappo_unified",
        "eval_fn_name": "eval_cpgappo_once",
        "ckpt_best": "CPGAPPO_fwdonly.pt",
        "ckpt_last": "CPGAPPO_fwdonly_last.pt",
        "use_backward": False,   # 该变体开? 前向 only
        "use_shield": True,
        "use_cp": True,
    },
    "alloff": {
        # 4 个机制开关全?(Guide CE / Shield / App Credit / CP 排序),
        # GAT 方向仍保留双?(= 基准, 不动这个开?.
        "eval_module": "Algorithms.Train.train_cpgappo_unified",
        "eval_fn_name": "eval_cpgappo_once",
        "ckpt_best": "CPGAPPO_alloff.pt",
        "ckpt_last": "CPGAPPO_alloff_last.pt",
        "use_backward": True,
        "use_shield": False,
        "use_cp": False,
    },
}
# ================= 评估环境参数（与 _run_ablation_default.py worker 一致） =================
DEADLINE_SLOT = 55
BURST_PROB = 0.2
BURST_SIZE = 55
MAX_STEPS = 8000
STOP_ARRIVAL_STEP = 2000
def _build_env_for_reeval(matrix_path, run_dir, seed_offset):
    """
    ?_run_ablation_default.py worker ?setup 模式构?env/ts/gs/bc/agent?
    关键：CONFIG["SEED"]=0（默认）+ seed_offset=seed ?env_seed=seed，与原训练一致?
    返回 (agent, ts, gs, bc, arrival_plan, task_complex_index, ckpt_used, vc)
    """
    # ---- 1) 设置全局环境（与 worker 一致） ----
    os.environ["MATRIX_OVERRIDE_PATH"] = matrix_path
    para["deadline_slot"] = DEADLINE_SLOT
    CONFIG["RUN_DIR"] = run_dir
    CONFIG["BURST_PROB"] = BURST_PROB
    CONFIG["BURST_SIZE"] = BURST_SIZE
    CONFIG["MAX_STEPS"] = MAX_STEPS
    CONFIG["STOP_ARRIVAL_STEP"] = STOP_ARRIVAL_STEP
    CONFIG["SEED"] = 0  # ?worker 一致：env_seed = CONFIG["SEED"] + seed_offset = seed
    # ---- 2) init_worker（设置随机种子、local_power 等） ----
    init_worker(seed_offset, para, CONFIG)
    device = torch.device('cuda:0' if torch.cuda.is_available() else torch.device('cpu'))
    # ---- 3) 构?env/ts/gs/bc ----
    user_num = para["user_num"]
    subgraph_num = 20
    basegraph_num = 60
    task_complex = para["task_complex"]
    _max_steps = CONFIG["MAX_STEPS"]
    if isinstance(task_complex, (list, tuple)):
        task_complex_index = (CONFIG["SEED"] + seed_offset) % len(task_complex)
    else:
        task_complex_index = int(task_complex) if isinstance(task_complex, int) else 0
    env = Environment(user_num, subgraph_num, basegraph_num, task_complex_index)
    env_seed = CONFIG["SEED"] + seed_offset
    # ?run dir 没有 arrivals/ 目录（worker ?generate_arrival_plan），故这里也?generate
    arrival_plan = generate_arrival_plan(
        env_seed, _max_steps, CONFIG["STOP_ARRIVAL_STEP"],
        0.3, CONFIG.get("BURST_PROB", 0.15),
        max(1, CONFIG.get("BURST_SIZE", 4) // 2), CONFIG.get("BURST_SIZE", 4)
    )
    env.generate_components(seed=env_seed)
    G = get_graph_cache(user_num, subgraph_num, basegraph_num, _PROJECT_ROOT)
    if G is not None and env.basegraph:
        env.basegraph.nx_graph = G
    deadline_config = load_deadline_config("")  # ?run dir ?configs/deadlines.json ?None
    ts = TaskScheduler(user_num, subgraph_num, basegraph_num, env,
                       tight_deadline_config=deadline_config, seed=env_seed)
    gs = GraphScheduler(env.basegraph, env.subgraph_list, ts)
    bc = BenchmarkClass(env, gs, ts, task_complex_index, effective=True, seed=env_seed)
    ts.env = env
    ts.using_Algorithm = -1
    bc.reset()
    ts.reset()
    # ---- 4) 自动探测 state_dim/global_dim，构?agent ----
    try:
        dummy_task = (0, 0)
        dummy_data, dummy_gfeat, _ = extract_cpgappo_state(ts, dummy_task, slot=0,
                                                       task_complex_index=task_complex_index)
        state_dim = dummy_data.x.shape[1]
        global_dim = dummy_gfeat.shape[0]
    except Exception as e:
        print(f"[_reeval_lib] Dim detect failed, using defaults. Error: {e}")
        state_dim, global_dim = 27, 10
    action_dim = para["edge_num"] + 2
    return {
        "env": env, "ts": ts, "gs": gs, "bc": bc,
        "arrival_plan": arrival_plan,
        "task_complex_index": task_complex_index,
        "max_steps": _max_steps,
        "state_dim": state_dim, "global_dim": global_dim, "action_dim": action_dim,
        "device": device,
        "user_num": user_num,
    }
def _load_ckpt_and_build_agent(ctx, variant_cfg, ckpt_dir):
    """构建 agent 并加载 ckpt。优先 best，找不到 fallback _last"""
    ckpt_best_path = Path(ckpt_dir) / variant_cfg["ckpt_best"]
    ckpt_last_path = Path(ckpt_dir) / variant_cfg["ckpt_last"]
    ckpt_used = None
    ckpt_path = None
    if ckpt_best_path.exists():
        ckpt_path = ckpt_best_path
        ckpt_used = "best"
    elif ckpt_last_path.exists():
        ckpt_path = ckpt_last_path
        ckpt_used = "last"
    else:
        raise FileNotFoundError(
            f"Neither best nor last ckpt exists:\n  {ckpt_best_path}\n  {ckpt_last_path}"
        )
    use_backward_cfg = variant_cfg["use_backward"]
    agent = GAT_PPO_Agent_CPGAPPO(
        node_dim=ctx["state_dim"], global_dim=ctx["global_dim"],
        action_dim=ctx["action_dim"], device=ctx["device"],
        lr=3e-4, entropy_coef=0.01, lambda_guide=0.2,
        use_backward=use_backward_cfg,
    )
    state_dict = torch.load(ckpt_path, map_location=ctx["device"])
    # 兼容加载: 若旧 forward-only ckpt 加载?dual 模型,
    #   则走 load_state_dict_compat; 否则直接 load_state_dict.
    load_state_dict_compat(agent.policy, state_dict, strict=False, verbose=True)
    agent.policy.eval()  # 【稳健性】确?eval 模式（dropout/BN 关闭?
    return agent, ckpt_used, str(ckpt_path)
def reeval_one(variant, topology, seed):
    """
    对单?(variant, topology, seed) ?D_all re-eval?
    Returns:
        dict: 写入 result_reeval.json 并返回。失败时返回 {"status":"error", ...}
    """
    t_start = time.time()
    TOPOLOGY_CONFIG = _load_topology_config_for_seed(seed)
    topo_cfg = TOPOLOGY_CONFIG[topology]
    variant_cfg = VARIANT_CONFIG[variant]
    run_dir = os.path.join(topo_cfg["run_dir_base"], f"{topology}__{variant}__seed{seed}")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    result_path = os.path.join(run_dir, "result_reeval.json")
    print(f"[_reeval_lib] === variant={variant} topology={topology} seed={seed} ===")
    print(f"[_reeval_lib] run_dir={run_dir}")
    try:
        # ---- 1) 构?env ----
        ctx = _build_env_for_reeval(topo_cfg["matrix_path"], run_dir, seed)
        ts = ctx["ts"]
        # ---- 2) 加载 ckpt + 构?agent ----
        agent, ckpt_used, ckpt_path = _load_ckpt_and_build_agent(ctx, variant_cfg, ckpt_dir)
        print(f"[_reeval_lib] ckpt_used={ckpt_used} path={ckpt_path}")
        # ---- 3) ?variant eval 函数（拿 D_succ 口径?----
        # 延迟 import 对应模块，拿?eval 函数
        import importlib
        mod = importlib.import_module(variant_cfg["eval_module"])
        eval_fn = getattr(mod, variant_cfg["eval_fn_name"])
        # alloff ?eval_once 需?use_shield/use_cp 参数
        if variant == "alloff":
            with torch.no_grad():
                e_succ, d_succ, score_succ, to_info, total_energy = eval_fn(
                    agent, ts, ctx["gs"], ctx["bc"], ctx["arrival_plan"],
                    ctx["task_complex_index"], ctx["max_steps"],
                    use_shield=variant_cfg["use_shield"],
                    use_cp=variant_cfg["use_cp"],
                )
        elif variant_cfg["eval_module"] == "Algorithms.Train.train_cpgappo_unified":
            # CPGAPPO 统一消融: eval_cpgappo_once 需 use_shield/use_cp 参数
            with torch.no_grad():
                e_succ, d_succ, score_succ, to_info, total_energy = eval_fn(
                    agent, ts, ctx["gs"], ctx["bc"], ctx["arrival_plan"],
                    ctx["task_complex_index"], ctx["max_steps"],
                    use_shield=variant_cfg["use_shield"],
                    use_cp=variant_cfg["use_cp"],
                )
        else:
            with torch.no_grad():
                e_succ, d_succ, score_succ, to_info, total_energy = eval_fn(
                    agent, ts, ctx["gs"], ctx["bc"], ctx["arrival_plan"],
                    ctx["task_complex_index"], ctx["max_steps"],
                )
        # ---- 4) 同一 ts 状态下计算 D_all（deadline 口径，审稿人 R1-5）----
        # eval 函数已调 ts.finalize_episode，exit_time 已就绪，直接 get_avg_results。
        e_all, d_all = ts.get_avg_results(only_successful=False, timeout_charge="deadline")
        # ---- 5) 计算 UtilityScore（主表口径 w_cost=0.25, sla0=0.95）----
        # calc_timeout_rate 返回小数 (0~1)，compute_utility_score 内部按百分比口径
        # 处理；这里统一 *100 还原为百分比再传入。
        rho_app_pct = float(to_info['app_timeout_rate']) * 100.0
        rho_task_pct = float(to_info.get('task_timeout_rate', 0)) * 100.0
        utility_succ = float(compute_utility_score(
            e_succ, d_succ, rho_app_pct, rho_task_pct, w_cost=0.25))
        utility_all = float(compute_utility_score(
            e_all, d_all, rho_app_pct, rho_task_pct, w_cost=0.25))
        # ---- 6) 组装结果 ----
        elapsed = time.time() - t_start
        result = {
            "status": "ok",
            "variant": variant,
            "topology": topology,
            "seed": seed,
            # 时延口径
            "D_succ": float(d_succ),
            "D_all": float(d_all),
            "E_succ": float(e_succ),
            "E_all": float(e_all),
            # 超时率 (%, 与论文表格 AppTO(%) 列一致)
            "AppTO": float(rho_app_pct),
            "TaskTO": float(rho_task_pct),
            # 旧口径 score (compute_score, 越小越好)
            "Score_succ": float(score_succ),
            # UtilityScore (主表口径 w_cost=0.25, sla0=0.95)
            "UtilityScore_succ": utility_succ,
            "UtilityScore_all": utility_all,
            # 能耗
            "energy": float(e_succ),
            "total_energy": float(total_energy),
            # ckpt 信息
            "ckpt_used": ckpt_used,
            "ckpt_path": ckpt_path,
            # 配置 (便于核对 R1-3)
            "config": {
                "timeout_charge_mode": "deadline",
                "deadline_charge_formula": "get_app_deadline_slot(uid) * slot_interval (= TD_i^max - TS_i)",
                "deadline_slot": DEADLINE_SLOT,
                "matrix_path": topo_cfg["matrix_path"],
                "burst_prob": BURST_PROB,
                "burst_size": BURST_SIZE,
                "max_steps": MAX_STEPS,
                "stop_arrival_step": STOP_ARRIVAL_STEP,
                "use_backward": variant_cfg["use_backward"],
                "use_shield": variant_cfg["use_shield"],
                "use_cp": variant_cfg["use_cp"],
            },
            # 动作统计
            "action_stats": to_info.get('action_stats', {}),
            "subtask_stats": to_info.get('subtask_stats', {}),
            "elapsed_sec": round(elapsed, 1),
        }
        # ---- 7) ?result_reeval.json（不覆盖?result.json?----
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"[_reeval_lib] OK  D_succ={d_succ:.4f}  D_all={d_all:.4f}  "
              f"AppTO={rho_app_pct:.2f}%  Utility_all={utility_all:.4f}  "
              f"({elapsed:.1f}s)")
        print(f"[_reeval_lib] result_reeval.json -> {result_path}")
        return result
    except Exception as ex:
        elapsed = time.time() - t_start
        err_msg = f"{type(ex).__name__}: {ex}"
        print(f"[_reeval_lib] [ERROR] {err_msg}")
        traceback.print_exc()
        err_result = {
            "status": "error",
            "variant": variant,
            "topology": topology,
            "seed": seed,
            "error": err_msg,
            "traceback": traceback.format_exc(),
            "elapsed_sec": round(elapsed, 1),
        }
        try:
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(err_result, f, indent=2, ensure_ascii=False)
        except Exception:
            pass
        return err_result
if __name__ == "__main__":
    # 自测：跑 default/main/seed1 一?
    print("=" * 70)
    print("_reeval_lib.py self-test: default/main/seed1")
    print("=" * 70)
    reeval_one("main", "default", 1)

