"""
Training / evaluation wrapper for DGMA_adapt.

Close to train_transedge_style_wrapper.py skeleton, but:
  * off-policy: store transitions into PER buffer, call agent.update() once per
    decision after warmup;
  * next-state is filled by a one-step-lagged pending-transition pointer.
"""

import os
import sys
import time
import traceback
import numpy as np
import torch
from pathlib import Path

_cur = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(_cur))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Experiments_new.exp_utils import (
    CONFIG, init_worker, seed_everything,
    generate_arrival_plan, save_deadline_config, load_deadline_config, load_arrival_plan,
    compute_score, compute_utility_score, calc_timeout_rate, safe_rest_tasks_total, all_arrived_done,
    get_graph_cache, apply_arrival_plan, subtask_partition_stats, append_curve_row,
)
from Environment.environment import Environment
from scheduler.graph_scheduler import GraphScheduler
from scheduler.task_scheduler import TaskScheduler
from Algorithms.Benchmark import Benchmark as BenchmarkClass
from utils.constant import para

from Algorithms.StrongBaselines.dgma_adapt_agent import DGMAAdaptAgent
from Algorithms.StrongBaselines.dgma_adapt_core import extract_dgma_adapt_state

_ALGO_TAG = "DGMA_adapt"


# ---------------------------------------------------------------------------
# Simple structured reward (re-used from TransEdgeStyle wrapper style — ensures
# the reward signal / reward scale is IDENTICAL across dispatch baselines so
# that DGMA_adapt is compared fairly on policy quality, not on reward design).
# ---------------------------------------------------------------------------
def _structured_reward(app_timeout, app_done, task_timeout,
                       energy_norm, delay_norm, action_val, route_stats,
                       edge_feasible=True, local_feasible=True):
    if app_timeout:
        return -3.0
    if app_done:
        return 2.0 - 0.5 * energy_norm - 0.5 * delay_norm
    r = 0.0
    if task_timeout:
        r -= 0.8
    else:
        r += 0.15
    r += 0.08 * (1.0 - energy_norm)
    r += 0.08 * (1.0 - delay_norm)
    if route_stats is not None:
        total = max(1, route_stats.get("local", 0) + route_stats.get("edge", 0) + route_stats.get("cloud", 0))
        local_ratio = route_stats.get("local", 0) / total
        edge_ratio = route_stats.get("edge", 0) / total
        cloud_ratio = route_stats.get("cloud", 0) / total
        if action_val == 1 and cloud_ratio > 0.70 and edge_feasible:
            r -= 0.20 * ((cloud_ratio - 0.70) / 0.30)
        if action_val >= 2 and edge_ratio < 0.25 and edge_feasible:
            r += 0.10 * ((0.25 - edge_ratio) / 0.25)
        if action_val == 0 and local_ratio < 0.10 and local_feasible:
            r += 0.04
    return r


# ---------------------------------------------------------------------------
# Evaluation pass (deterministic, no buffer push, no update)
# ---------------------------------------------------------------------------
def _eval_once(agent, env, arrival_plan, task_complex_index, MAX_STEPS, use_heuristic_sort,
               env_seed, deadline_config):
    user_num = para["user_num"]
    ts_eval = TaskScheduler(user_num, 20, 60, env,
                            tight_deadline_config=deadline_config, seed=env_seed)
    gs_eval = GraphScheduler(env.basegraph, env.subgraph_list, ts_eval)
    bc_eval = BenchmarkClass(env, gs_eval, ts_eval, task_complex_index, effective=True, seed=env_seed)
    ts_eval.env = env
    ts_eval.using_Algorithm = -1
    bc_eval.reset(); ts_eval.reset()
    ts_eval.route_stats = {"local": 0, "edge": 0, "cloud": 0, "recent": []}

    eval_task2action = {}
    for slot in range(MAX_STEPS):
        apply_arrival_plan(ts_eval, slot, arrival_plan)
        ts_eval.check_timeouts(slot)
        tasks = gs_eval.get_tasks(slot, sort_tasks=use_heuristic_sort)
        if not tasks:
            if (slot >= CONFIG["STOP_ARRIVAL_STEP"] and
                    safe_rest_tasks_total(ts_eval.rest_tasks) == 0 and
                    all_arrived_done(ts_eval)):
                break
            continue
        for task in tasks:
            s_data, mask = extract_dgma_adapt_state(ts_eval, task, slot=slot,
                                                    task_complex_index=task_complex_index)
            act, _ = agent.take_action(s_data, action_mask=mask, deterministic=True)
            bc_eval.step([[task, act]])
            eval_task2action[tuple(task)] = int(act)
            if act == 0:
                ts_eval.route_stats["local"] += 1
            elif act == 1:
                ts_eval.route_stats["cloud"] += 1
            else:
                ts_eval.route_stats["edge"] += 1

    ts_eval.finalize_episode(MAX_STEPS - 1)
    e_succ, d_succ = ts_eval.get_avg_results(only_successful=True)
    e_all, d_all = ts_eval.get_avg_results(only_successful=False, timeout_charge="deadline")
    to_info = calc_timeout_rate(ts_eval)
    # calc_timeout_rate 返回小数 (0~1)；compute_score 是旧版接口 (接受小数 rho)
    app_to = float(to_info["app_timeout_rate"])
    task_to = float(to_info.get("task_timeout_rate", app_to))
    if app_to > 1.0: app_to /= 100.0
    if task_to > 1.0: task_to /= 100.0

    score = compute_score(
        E_avg=e_succ, D_succ_avg=d_succ, rho_app=app_to, rho_task=task_to,
        sla0=CONFIG.get("SLA0", 0.95), kappa=CONFIG.get("KAPPA", 2.0), v_cap=3.0,
    )

    partition = subtask_partition_stats(ts_eval, eval_task2action)
    to_info["action_stats"] = {
        "local": partition["local"], "cloud": partition["cloud"], "edge": partition["edge"],
        "timeout": partition["timeout"], "unknown": partition["unknown"],
        "total": partition["total_subtasks"], "total_actions": len(eval_task2action),
    }
    total_e = ts_eval.get_sum_energy()
    return float(e_succ), float(d_succ), float(score), to_info, float(total_e)


# ---------------------------------------------------------------------------
# Main training wrapper
# ---------------------------------------------------------------------------
def train_dgma_adapt_wrapper(gpu_id, seed_offset, use_heuristic_sort=True, episodes=50,
                             hidden_dim=64, batch_size=64, buffer_capacity=20000):
    init_worker(seed_offset, para, CONFIG)
    if "RUN_DIR" in os.environ: CONFIG["RUN_DIR"] = os.environ["RUN_DIR"]
    if "EVAL_MODE" in os.environ: CONFIG["EVAL_MODE"] = (os.environ["EVAL_MODE"] == "True")

    device = torch.device(f"cuda:{gpu_id}") if torch.cuda.is_available() else torch.device("cpu")
    checkpoint_dir = None
    if CONFIG.get("RUN_DIR"):
        checkpoint_dir = Path(CONFIG["RUN_DIR"]) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    try:
        user_num = para["user_num"]
        subgraph_num = 20
        basegraph_num = 60
        task_complex = para["task_complex"]
        MAX_STEPS = CONFIG["MAX_STEPS"]
        if isinstance(task_complex, (list, tuple)):
            task_complex_index = (CONFIG["SEED"] + seed_offset) % len(task_complex)
        else:
            task_complex_index = int(task_complex) if isinstance(task_complex, int) else 0

        env = Environment(user_num, subgraph_num, basegraph_num, task_complex_index)
        run_dir = CONFIG.get("RUN_DIR", "")
        plan_data = load_arrival_plan(run_dir, seed_offset) if run_dir else None
        if plan_data is None:
            env_seed = CONFIG["SEED"] + seed_offset
            arrival_plan = generate_arrival_plan(
                env_seed, MAX_STEPS, CONFIG["STOP_ARRIVAL_STEP"],
                0.3, CONFIG.get("BURST_PROB", 0.15),
                max(1, CONFIG.get("BURST_SIZE", 4) // 2), CONFIG.get("BURST_SIZE", 4),
            )
        else:
            arrival_plan = plan_data.get("arrival_plan", [])
            env_seed = plan_data.get("env_seed", CONFIG["SEED"] + seed_offset)

        env.generate_components(seed=env_seed)
        G = get_graph_cache(user_num, subgraph_num, basegraph_num, project_root)
        if G is not None and env.basegraph: env.basegraph.nx_graph = G
        deadline_config = load_deadline_config(run_dir)

        ts = TaskScheduler(user_num, subgraph_num, basegraph_num, env,
                           tight_deadline_config=deadline_config, seed=env_seed)
        gs = GraphScheduler(env.basegraph, env.subgraph_list, ts)
        bc = BenchmarkClass(env, gs, ts, task_complex_index, effective=True, seed=env_seed)
        ts.env = env; ts.using_Algorithm = -1; bc.reset(); ts.reset()

        # feature dim auto-detect
        try:
            dummy_task = (0, 0)
            dummy_data, _ = extract_dgma_adapt_state(ts, dummy_task, slot=0,
                                                    task_complex_index=task_complex_index)
            node_dim = dummy_data.x.shape[1]
            print(f"[{_ALGO_TAG}] Detected node_dim={node_dim}")
        except Exception as e:
            print(f"[{_ALGO_TAG}] node_dim detection failed, fallback=27. err={e}")
            node_dim = 27

        action_dim = para["edge_num"] + 2

        agent = DGMAAdaptAgent(
            node_dim=node_dim, action_dim=action_dim, device=device,
            hidden_dim=hidden_dim, num_layers=2,
            lr_actor=3e-4, lr_critic=3e-4,
            gamma=0.99, tau=0.005, gumbel_tau=1.0,
            batch_size=batch_size, buffer_capacity=buffer_capacity,
            per_alpha=0.6, per_beta0=0.4, per_beta_anneal_steps=10000,
            updates_per_step=1, min_updates_after=max(256, batch_size * 4),
            grad_clip=1.0,
        )

        best_eval_score = float("inf")
        best_eval_appTO = 1.0
        best_eval_metrics = None
        EVAL_EVERY = 5
        eval_curve_data = []
        eval_run_count = 0

        # Pre-training evaluation
        print(f"\n[{_ALGO_TAG}] === Pre-training eval ===")
        ee0, dd0, sc0, to0, te0 = _eval_once(
            agent, env, arrival_plan, task_complex_index, MAX_STEPS,
            use_heuristic_sort, env_seed, deadline_config,
        )
        best_eval_score = sc0
        best_eval_appTO = to0["app_timeout_rate"]
        best_eval_metrics = {"e": ee0, "d": dd0, "score": sc0,
                             "total_energy": te0, "timeout_info": dict(to0)}
        print(f"[{_ALGO_TAG}][InitEval] Score={sc0:.3f} AppTO={to0['app_timeout_rate']:.2%} TotalE={te0:.4f}")

        # ---------------- Training loop ----------------
        for episode in range(episodes):
            bc.reset(); ts.reset()
            ts.route_stats = {"local": 0, "edge": 0, "cloud": 0, "recent": []}
            agent.set_episode(episode)

            ep_rewards = []
            ep_updates = 0
            action_count = [0] * action_dim
            total_actions = 0
            task2action = {}

            prev_app_to = set(ts.application_timeout_finished)
            prev_app_fin = set(ts.application_finished)

            # pending transition for one-step-lagged next_state filling
            # dict with keys: s, a, r, mask
            pending = None

            for slot in range(MAX_STEPS):
                apply_arrival_plan(ts, slot, arrival_plan)
                ts.check_timeouts(slot)
                tasks = gs.get_tasks(slot, sort_tasks=use_heuristic_sort)
                if not tasks:
                    if (slot >= CONFIG["STOP_ARRIVAL_STEP"] and
                            safe_rest_tasks_total(ts.rest_tasks) == 0 and
                            all_arrived_done(ts)):
                        break
                    continue

                for task in tasks:
                    s_data, mask_bin = extract_dgma_adapt_state(
                        ts, task, slot=slot, task_complex_index=task_complex_index
                    )

                    # Flush pending transition using current s_data as next_state
                    if pending is not None:
                        agent.store(pending["s"].cpu(), pending["a"], pending["r"],
                                    s_data.clone().cpu(), False, pending["mask"], mask_bin)
                        pending = None

                    action, _ = agent.take_action(s_data, action_mask=mask_bin,
                                                  deterministic=False,
                                                  route_stats=getattr(ts, "route_stats", None))
                    if action < action_dim:
                        action_count[action] += 1
                    total_actions += 1
                    task2action[tuple(task)] = int(action)

                    if action == 0:
                        ts.route_stats["local"] += 1
                    elif action == 1:
                        ts.route_stats["cloud"] += 1
                    else:
                        ts.route_stats["edge"] += 1

                    reward_env, info = bc.step([[task, action]])

                    curr_app_to = set(ts.application_timeout_finished)
                    curr_app_fin = set(ts.application_finished)
                    new_app_to = curr_app_to - prev_app_to
                    new_app_fin = curr_app_fin - prev_app_fin
                    prev_app_to, prev_app_fin = curr_app_to, curr_app_fin

                    energy = float(info.get("step_energy", 0.0))
                    e_norm = max(0.0, min(1.0, (energy - 0.3) / (3.0 - 0.3)))
                    d_step = float(info.get("step_delay", 0.0))
                    d_norm = max(0.0, min(1.0, d_step / 5.0))
                    task_to_flag = float(reward_env) < -0.1
                    app_id = task[0]

                    r_struct = _structured_reward(
                        app_timeout=(app_id in new_app_to),
                        app_done=(app_id in new_app_fin),
                        task_timeout=task_to_flag,
                        energy_norm=e_norm, delay_norm=d_norm,
                        action_val=int(action),
                        route_stats=getattr(ts, "route_stats", None),
                    )

                    # set pending (will be flushed when next decision arrives)
                    pending = {
                        "s": s_data.clone(),
                        "a": int(action),
                        "r": float(r_struct),
                        "mask": list(mask_bin),
                    }
                    ep_rewards.append(float(r_struct))

                    # off-policy update
                    al, cl = agent.update()
                    if al != 0.0 or cl != 0.0:
                        ep_updates += 1

            # End of episode: flush pending as terminal
            if pending is not None:
                zero_mask = [1] * action_dim  # placeholder, won't be used thanks to done=True
                agent.store(pending["s"].cpu(), pending["a"], pending["r"],
                            None, True, pending["mask"], zero_mask)
                pending = None
                al, cl = agent.update()
                if al != 0.0 or cl != 0.0:
                    ep_updates += 1

            # ---------------- Logging per episode ----------------
            train_to_info = calc_timeout_rate(ts)
            e_succ, d_succ = ts.get_avg_results(only_successful=True)
            e_all, d_all = ts.get_avg_results(only_successful=False, timeout_charge="deadline")
            total_reward = float(np.sum(ep_rewards)) if ep_rewards else 0.0
            # calc_timeout_rate 返回小数 (0~1)，compute_utility_score 要百分比；
            # 论文口径 delay 用 D_all。
            utility_score = compute_utility_score(
                E_avg=e_all, D_succ_avg=d_all,
                rho_app=float(train_to_info["app_timeout_rate"]) * 100.0,
                rho_task=float(train_to_info.get("task_timeout_rate", 0)) * 100.0,
            )
            if total_actions > 0:
                lp = 100.0 * action_count[0] / total_actions
                cp = 100.0 * action_count[1] / total_actions if action_dim > 1 else 0.0
                ec = sum(action_count[2:]) if action_dim > 2 else 0
                ep_pc = 100.0 * ec / total_actions if action_dim > 2 else 0.0
                print(f"[{_ALGO_TAG}] Ep {episode+1}/{episodes} "
                      f"AppTO={train_to_info['app_timeout_rate']:.2%} "
                      f"Score={utility_score:.3f} E={e_succ:.3f} D={d_all:.3f} "
                      f"R={total_reward:.2f} updates={ep_updates} "
                      f"L={action_count[0]}({lp:.0f}%) C={action_count[1]}({cp:.0f}%) Edge={ec}({ep_pc:.0f}%) "
                      f"actor_loss={agent.last_actor_loss:.4f} critic_loss={agent.last_critic_loss:.4f}")

            # ---------------- Periodic evaluation ----------------
            if (episode + 1) % EVAL_EVERY == 0 or (episode + 1) == episodes:
                ee, dd, sc, to, te = _eval_once(
                    agent, env, arrival_plan, task_complex_index, MAX_STEPS,
                    use_heuristic_sort, env_seed, deadline_config,
                )
                curr_appTO = to["app_timeout_rate"]
                eval_run_count += 1
                eval_curve_data.append({
                    "episode": episode + 1, "eval_run": eval_run_count,
                    "score": float(sc), "energy": float(ee), "total_energy": float(te),
                    "delay": float(dd), "app_timeout_rate": float(curr_appTO),
                    "task_timeout_rate": float(to.get("task_timeout_rate", curr_appTO)),
                })
                print(f"[{_ALGO_TAG}][Eval Ep{episode+1}] Score={sc:.3f} AppTO={curr_appTO:.2%} TotalE={te:.4f}")

                if (curr_appTO < best_eval_appTO - 1e-9) or \
                   (abs(curr_appTO - best_eval_appTO) < 1e-9 and sc < best_eval_score):
                    best_eval_score = sc
                    best_eval_appTO = curr_appTO
                    best_eval_metrics = {"e": float(ee), "d": float(dd), "score": float(sc),
                                         "total_energy": float(te), "timeout_info": dict(to)}
                    if checkpoint_dir:
                        torch.save(agent.actor.state_dict(), checkpoint_dir / "DGMA_adapt.pt")
                    print(f"[{_ALGO_TAG}][Eval] [BEST] AppTO={curr_appTO:.2%} Score={sc:.3f}")

            # ---------------- Curve logging ----------------
            if checkpoint_dir:
                try:
                    train_curve_file = Path(CONFIG["RUN_DIR"]) / "curves" / f"{_ALGO_TAG}_seed{seed_offset}.csv"
                    train_curve_file.parent.mkdir(parents=True, exist_ok=True)
                    append_curve_row(train_curve_file, {
                        "episode": episode + 1,
                        "reward": total_reward,
                        "score": float(utility_score),
                        "utility_score": float(utility_score),
                        "energy": float(e_succ),
                        "delay": float(d_succ),
                        "app_timeout_rate": float(train_to_info["app_timeout_rate"]),
                        "task_timeout_rate": float(train_to_info.get("task_timeout_rate", 0)),
                        "best_eval_score": float(best_eval_score),
                        "curve_type": "training",
                    })
                except Exception as cerr:
                    print(f"[{_ALGO_TAG}] save curve failed: {cerr}")

        # Restore best eval weights before returning
        try:
            if checkpoint_dir and (checkpoint_dir / "DGMA_adapt.pt").exists():
                best_state = torch.load(checkpoint_dir / "DGMA_adapt.pt", map_location=device)
                agent.actor.load_state_dict(best_state)
        except Exception as e:
            print(f"[{_ALGO_TAG}] restore best failed: {e}")

        if best_eval_metrics is None:
            best_eval_metrics = {"e": 1000.0, "d": 1000.0, "score": float("inf"),
                                 "total_energy": 0.0,
                                 "timeout_info": {"app_timeout_rate": 1.0, "task_timeout_rate": 1.0}}

        best_eval_metrics["timeout_info"]["inference_time_ms"] = 0.0
        # 统一口径: 对外 app_timeout_rate / task_timeout_rate 存【百分比】(0~100)，
        # 与 compute_utility_score 的输入约定一致。to_info 内部是小数，乘 100 转换。
        _best_to = best_eval_metrics["timeout_info"]
        _best_app_pct = float(_best_to.get("app_timeout_rate", 1.0)) * 100.0
        _best_task_pct = float(_best_to.get("task_timeout_rate", 1.0)) * 100.0
        flattened = {
            "app_timeout_rate": _best_app_pct,
            "task_timeout_rate": _best_task_pct,
            "score": best_eval_metrics.get("score", 0.0),
            "inference_time_ms": 0.0,
            "action_stats": _best_to.get("action_stats", {}),
            "timeout_rate": _best_to,
            "total_energy": best_eval_metrics.get("total_energy", best_eval_metrics["e"]),
        }
        agent.print_inference_stats()
        print(f"[{_ALGO_TAG}][FINAL] E={best_eval_metrics['e']:.4f} D={best_eval_metrics['d']:.4f} "
              f"Score={best_eval_metrics['score']:.4f} "
              f"AppTO={_best_app_pct:.2f}%")
        return best_eval_metrics["e"], best_eval_metrics["d"], flattened

    except Exception as e:
        print(f"[{_ALGO_TAG}] Training crashed: {e}")
        traceback.print_exc()
        return 1000.0, 1000.0, {"app_timeout_rate": 100.0, "task_timeout_rate": 100.0, "score": float("inf")}
