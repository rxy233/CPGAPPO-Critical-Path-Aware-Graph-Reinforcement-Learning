"""
Training / evaluation wrapper for DGMA_paper — the faithful multi-agent
re-implementation of Chen et al. TMC 2026 "DGMA".

Design highlights that differ from DGMA_adapt:
  * K=edge_num region agents (was: 1 centralized).
  * MADDPG update with centralized critic (was: Double-Q dueling).
  * Reward per region = mean utility improvement vs. local baseline (was:
    multi-tier scalar with app_timeout / app_done tiers).
  * Continuous c_alloc applied via runtime monkey-patch of TaskScheduler
    so we DO NOT modify the core scheduler source.
  * Selective Parameter Sharing every N training steps.

The environment is UNMODIFIED.
"""

import os
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import torch

_cur = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(_cur))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Experiments_new.exp_utils import (
    CONFIG, init_worker, seed_everything,
    generate_arrival_plan, save_deadline_config, load_deadline_config, load_arrival_plan,
    compute_utility_score, calc_timeout_rate, safe_rest_tasks_total, all_arrived_done,
    get_graph_cache, apply_arrival_plan, subtask_partition_stats, append_curve_row,
)
from Environment.environment import Environment
from Environment import computation
from scheduler.graph_scheduler import GraphScheduler
from scheduler.task_scheduler import TaskScheduler
from Algorithms.Benchmark import Benchmark as BenchmarkClass
from utils.constant import para

from Algorithms.StrongBaselines.dgma_paper_agent import DGMAPaperAgent
from Algorithms.StrongBaselines.dgma_paper_core import (
    extract_region_state, R1_FEATURE_DIM,
)

_ALGO_TAG = "DGMA_paper"


# ===========================================================================
# c_alloc runtime hook: scale edge.calculate_parameter for the duration of one
# add_task call. Applied via monkey-patch on TaskScheduler.add_task ONCE per
# worker and restored in finally. Keeps the scheduler source untouched.
# ===========================================================================
_ORIG_ADD_TASK = None


def _install_c_alloc_patch():
    """Idempotent monkey-patch: reads ts._dgma_c_alloc (default 1.0) and
    temporarily scales the target edge's calculate_parameter before delegating
    to the original add_task. Local/Cloud paths are untouched."""
    global _ORIG_ADD_TASK
    if getattr(TaskScheduler, "_dgma_paper_patched", False):
        return
    _ORIG_ADD_TASK = TaskScheduler.add_task

    def _patched(self, user_id, subtask_id, node_id=None, *args, **kwargs):
        c = float(getattr(self, "_dgma_c_alloc", 1.0))
        if node_id is not None and int(node_id) >= 2 and abs(c - 1.0) > 1e-6:
            eid = int(node_id) - 2
            try:
                edge = self.env.edges[eid]
            except Exception:
                return _ORIG_ADD_TASK(self, user_id, subtask_id, node_id,
                                      *args, **kwargs)
            orig_cp = getattr(edge, "calculate_parameter", 1.0)
            edge.calculate_parameter = max(c, 0.1) * orig_cp
            try:
                return _ORIG_ADD_TASK(self, user_id, subtask_id, node_id,
                                      *args, **kwargs)
            finally:
                edge.calculate_parameter = orig_cp
        return _ORIG_ADD_TASK(self, user_id, subtask_id, node_id,
                              *args, **kwargs)

    TaskScheduler.add_task = _patched
    TaskScheduler._dgma_paper_patched = True


def _restore_c_alloc_patch():
    global _ORIG_ADD_TASK
    if _ORIG_ADD_TASK is not None and getattr(TaskScheduler, "_dgma_paper_patched", False):
        TaskScheduler.add_task = _ORIG_ADD_TASK
        TaskScheduler._dgma_paper_patched = False
        _ORIG_ADD_TASK = None


# ===========================================================================
# Local-baseline estimator (for per-task utility improvement)
# ===========================================================================
def _estimate_local_cost(env, uid: int, task_size: float, task_complex_index: int):
    """Return (t_local, e_local) as rough estimates for the local baseline.

    Reused from the same formula as TaskScheduler.add_task(node_id==0):
        t_local = work / f_local
        e_local = (wait + kappa * f_local^3) * t_local
    """
    try:
        dev = env.device_list[uid]
        f_local = float(dev.local_power)
        work = float(task_size) * float(para["task_complex"][task_complex_index])
        t_local = work / max(f_local, 1.0)
        kappa = float(para.get("kappa", 2.2e-27))
        wait = float(getattr(dev, "local_wait", para.get("local_wait", 0.1)))
        e_local = (wait + kappa * f_local ** 3) * t_local
        return max(t_local, 1e-6), max(e_local, 1e-6)
    except Exception:
        return 1.0, 1.0


def _utility_improvement(step_delay, step_energy, t_local, e_local):
    """Per-task utility in [-1, 1]: mean of tanh-squashed delay/energy gains."""
    d_gain = (t_local - float(step_delay)) / t_local        # >0 means better
    e_gain = (e_local - float(step_energy)) / e_local
    return 0.5 * np.tanh(d_gain) + 0.5 * np.tanh(e_gain)


# ===========================================================================
# Evaluation pass
# ===========================================================================
def _eval_once(agent, env, arrival_plan, task_complex_index, MAX_STEPS,
               use_heuristic_sort, env_seed, deadline_config, num_regions):
    user_num = para["user_num"]
    ts_eval = TaskScheduler(user_num, 20, 60, env,
                            tight_deadline_config=deadline_config, seed=env_seed)
    gs_eval = GraphScheduler(env.basegraph, env.subgraph_list, ts_eval)
    bc_eval = BenchmarkClass(env, gs_eval, ts_eval, task_complex_index,
                             effective=True, seed=env_seed)
    ts_eval.env = env
    ts_eval.using_Algorithm = -1
    bc_eval.reset(); ts_eval.reset()
    ts_eval.route_stats = {"local": 0, "edge": 0, "cloud": 0, "recent": []}
    ts_eval._dgma_c_alloc = 1.0

    user_to_region = agent.region_info["user_to_region"]
    region_to_users = agent.region_info["region_to_users"]

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

        # Dispatch each ready task through its owning region agent
        for task in tasks:
            uid = task[0]
            region_id = int(user_to_region[uid])
            state_data, mask_bin = extract_region_state(
                ts_eval, region_to_users[region_id], task,
                slot=slot, task_complex_index=task_complex_index,
            )
            server_id, c_alloc, _ = agent.take_action_region(
                region_id, state_data, action_mask=mask_bin, deterministic=True,
            )
            if server_id == 0:
                ts_eval.route_stats["local"] += 1
            elif server_id == 1:
                ts_eval.route_stats["cloud"] += 1
            else:
                ts_eval.route_stats["edge"] += 1
            ts_eval._dgma_c_alloc = float(c_alloc) * 2.0    # c∈[0,1] → [0,2] scale
            bc_eval.step([[task, int(server_id)]])

    # Metrics
    to_info = calc_timeout_rate(ts_eval)
    # calc_timeout_rate 返回小数 (0~1)，compute_utility_score 要百分比
    app_to = float(to_info["app_timeout_rate"]) * 100.0
    task_to = float(to_info.get("task_timeout_rate", 0)) * 100.0
    # 论文口径 delay 用 D_all (超时应用按 per-app deadline 预算计入)
    e_succ, d_succ = ts_eval.get_avg_results(only_successful=True)
    e_all, d_all = ts_eval.get_avg_results(only_successful=False, timeout_charge="deadline")
    util = compute_utility_score(E_avg=e_all, D_succ_avg=d_all,
                                 rho_app=app_to, rho_task=task_to)
    total_e = ts_eval.get_sum_energy()
    return {
        "score": float(util),
        "app_to": app_to,
        "task_to": task_to,
        "delay": float(d_all) if d_all == d_all else 0.0,
        "delay_succ": float(d_succ) if d_succ == d_succ else 0.0,
        "delay_all": float(d_all) if d_all == d_all else 0.0,
        "energy": float(e_all) if e_all == e_all else 0.0,
        "energy_succ": float(e_succ) if e_succ == e_succ else 0.0,
        "total_energy": float(total_e),
        "utility": float(util),
        "route_stats": dict(ts_eval.route_stats),
    }


# ===========================================================================
# Training wrapper
# ===========================================================================
def train_dgma_paper_wrapper(gpu_id, seed_offset, use_heuristic_sort=False,
                             episodes=50, hidden_dim=64, batch_size=32,
                             buffer_capacity=10000, sps_every=200):
    # NOTE [paper-faithful]: default use_heuristic_sort=False to avoid borrowing
    # CPGAPPO's ready-task sequencing advantage. DGMA paper does not assume any
    # sequencing oracle. Set this True only if you explicitly want to study the
    # interaction with sequencing (mark it clearly in the log/paper).
    init_worker(seed_offset, para, CONFIG)
    if "RUN_DIR" in os.environ: CONFIG["RUN_DIR"] = os.environ["RUN_DIR"]
    if "EVAL_MODE" in os.environ: CONFIG["EVAL_MODE"] = (os.environ["EVAL_MODE"] == "True")

    device = torch.device(f"cuda:{gpu_id}") if torch.cuda.is_available() else torch.device("cpu")
    checkpoint_dir = None
    if CONFIG.get("RUN_DIR"):
        checkpoint_dir = Path(CONFIG["RUN_DIR"]) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Install the runtime c_alloc scheduler hook (no source edit).
    _install_c_alloc_patch()

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

        action_dim = para["edge_num"] + 2
        num_regions = int(para["edge_num"])

        # Node-dim detection
        try:
            dummy_data, _ = extract_region_state(
                ts, list(range(min(2, user_num))), (0, 0), slot=0,
                task_complex_index=task_complex_index,
            )
            node_dim = dummy_data.x.shape[1]
            print(f"[{_ALGO_TAG}] Detected node_dim={node_dim}, K={num_regions}, A={action_dim}")
        except Exception as e:
            print(f"[{_ALGO_TAG}] node_dim detection failed, fallback=27. err={e}")
            node_dim = R1_FEATURE_DIM

        agent = DGMAPaperAgent(
            env=env, num_regions=num_regions, node_dim=node_dim,
            action_dim=action_dim, hidden_dim=hidden_dim, num_layers=2,
            device=device,
            lr_actor=3e-4, lr_critic=3e-4,
            gamma=0.99, tau=0.005, gumbel_tau=1.0,
            batch_size=batch_size, buffer_capacity=buffer_capacity,
            per_alpha=0.6, per_beta0=0.4, per_beta_anneal_steps=10000,
            updates_per_step=1, min_updates_after=max(256, batch_size * 4),
            grad_clip=1.0, sps_every=sps_every, sps_self_weight=0.6,
        )

        user_to_region = agent.region_info["user_to_region"]
        region_to_users = agent.region_info["region_to_users"]

        print(f"[{_ALGO_TAG}] Region sizes: "
              f"{[len(r) for r in region_to_users]}")
        print(f"[{_ALGO_TAG}] SPS weight matrix diag: "
              f"{[float(agent.sps_weights[k,k]) for k in range(num_regions)]}")

        best_eval_score = -float("inf")
        all_metrics = []

        # ---------------- Training loop ------------------------------------
        for episode in range(episodes):
            bc.reset(); ts.reset()
            ts.route_stats = {"local": 0, "edge": 0, "cloud": 0, "recent": []}
            ts._dgma_c_alloc = 1.0
            agent.set_episode(episode)

            ep_rewards = []
            ep_updates = 0
            action_count = [0] * action_dim
            total_actions = 0

            prev_app_to = set(ts.application_timeout_finished)
            prev_app_fin = set(ts.application_finished)

            # For joint transition: accumulate per-region rewards over a slot
            # and push one transition per slot boundary (not per single-task).
            pending = None
            region_reward_accum = [[] for _ in range(num_regions)]
            last_region_states = [None] * num_regions
            last_region_actions = [(None, None)] * num_regions  # (sid, c)
            last_masks = [[1] * action_dim for _ in range(num_regions)]

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

                # ---- 1. Per-task region-agent decision (matrix style) -----
                region_slot_states = [None] * num_regions
                region_slot_actions = [(None, None)] * num_regions
                region_slot_masks = [[1] * action_dim for _ in range(num_regions)]

                scheduled = []  # list of (task, server_id, c, region_id, urgency)
                for task in tasks:
                    uid = task[0]
                    region_id = int(user_to_region[uid])
                    state_data, mask_bin = extract_region_state(
                        ts, region_to_users[region_id], task, slot=slot,
                        task_complex_index=task_complex_index,
                    )
                    server_id, c_alloc, _ = agent.take_action_region(
                        region_id, state_data, action_mask=mask_bin,
                        deterministic=False,
                    )
                    # Urgency heuristic: closer to deadline -> higher urgency
                    try:
                        ddl = ts.enter_time[uid] + \
                              ts.get_app_deadline_slot(uid) * para["slot_interval"]
                        now = slot * para["slot_interval"]
                        urgency = 1.0 / max(1e-3, ddl - now)
                    except Exception:
                        urgency = 0.0
                    scheduled.append((task, int(server_id), float(c_alloc),
                                      region_id, urgency))
                    # Track per-region "latest" decision (overwrites if multiple
                    # tasks in same region this slot — fine, we treat slot
                    # as the MADDPG step).
                    region_slot_states[region_id] = state_data
                    region_slot_actions[region_id] = (int(server_id), float(c_alloc))
                    region_slot_masks[region_id] = list(mask_bin)
                    if int(server_id) < action_dim:
                        action_count[int(server_id)] += 1
                    total_actions += 1

                # ---- 2. Arbitrate (batch → ordered single-step dispatch) --
                ordered = agent.arbitrate(scheduled)

                # ---- 3. Dispatch, collect per-task utility ---------------
                for task, server_id, c_alloc, region_id, _u in ordered:
                    uid, sid = task
                    # Local baseline estimate for utility improvement
                    try:
                        tsize = float(ts.get_task_size_bytes(uid, sid) or 0.0)
                        if tsize <= 0: tsize = 200000.0
                    except Exception:
                        tsize = 200000.0
                    t_local, e_local = _estimate_local_cost(env, uid, tsize,
                                                            task_complex_index)
                    ts._dgma_c_alloc = float(c_alloc) * 2.0
                    if server_id == 0:
                        ts.route_stats["local"] += 1
                    elif server_id == 1:
                        ts.route_stats["cloud"] += 1
                    else:
                        ts.route_stats["edge"] += 1
                    _rwd, info = bc.step([[task, int(server_id)]])
                    step_e = float(info.get("step_energy", 0.0))
                    step_d = float(info.get("step_delay", 0.0))
                    util = _utility_improvement(step_d, step_e, t_local, e_local)
                    # Bonus/penalty on app-level completion
                    app_to_new = (uid in ts.application_timeout_finished) and \
                                 (uid not in prev_app_to)
                    app_fin_new = (uid in ts.application_finished) and \
                                  (uid not in prev_app_fin)
                    if app_to_new:
                        util -= 1.0
                    if app_fin_new:
                        util += 1.0
                    region_reward_accum[region_id].append(float(util))
                    ep_rewards.append(float(util))
                    prev_app_to = set(ts.application_timeout_finished)
                    prev_app_fin = set(ts.application_finished)

                # ---- 4. Compute per-region slot reward ------------------
                r_per_region = []
                for k in range(num_regions):
                    if region_reward_accum[k]:
                        r_per_region.append(float(np.mean(region_reward_accum[k])))
                    else:
                        r_per_region.append(0.0)
                region_reward_accum = [[] for _ in range(num_regions)]

                # ---- 5. Push previous pending transition --------------
                if pending is not None:
                    next_states = [
                        region_slot_states[k].clone() if region_slot_states[k] is not None else None
                        for k in range(num_regions)
                    ]
                    agent.store((
                        pending["s"], pending["sid"], pending["c"], pending["r"],
                        next_states, False, pending["mask"], region_slot_masks,
                    ))
                    pending = None

                pending = {
                    "s": [s.clone() if s is not None else None
                          for s in region_slot_states],
                    "sid": [a[0] for a in region_slot_actions],
                    "c":   [a[1] for a in region_slot_actions],
                    "r":   r_per_region,
                    "mask": region_slot_masks,
                }

                # ---- 6. Update ----------------------------------------
                al, cl = agent.update()
                if al != 0.0 or cl != 0.0:
                    ep_updates += 1

            # End of episode: flush pending as terminal
            if pending is not None:
                agent.store((
                    pending["s"], pending["sid"], pending["c"], pending["r"],
                    None, True, pending["mask"],
                    [[1] * action_dim for _ in range(num_regions)],
                ))
                al, cl = agent.update()
                if al != 0.0 or cl != 0.0:
                    ep_updates += 1

            # Metrics
            to_info_train = calc_timeout_rate(ts)
            # calc_timeout_rate 返回小数 (0~1)，compute_utility_score 要百分比
            appto = float(to_info_train["app_timeout_rate"]) * 100.0
            task_to_train = float(to_info_train.get("task_timeout_rate", 0)) * 100.0
            # 论文口径 delay 用 D_all
            e_succ_train, d_succ_train = ts.get_avg_results(only_successful=True)
            e_all_train, d_all_train = ts.get_avg_results(only_successful=False, timeout_charge="deadline")
            util = float(compute_utility_score(
                E_avg=e_all_train, D_succ_avg=d_all_train,
                rho_app=appto, rho_task=task_to_train,
            ))
            totalE = ts.get_sum_energy()
            mean_r = float(np.mean(ep_rewards)) if ep_rewards else 0.0

            # Evaluation pass — every EVAL_EVERY episodes (and last episode).
            # Full _eval_once is a complete rollout, so doing it every episode
            # roughly doubles wall time. With dividelong subprocess wait this
            # made DGMA_paper miss its 100-ep budget. Match DGMA_adapt cadence.
            EVAL_EVERY = 5
            do_eval = ((episode + 1) % EVAL_EVERY == 0) or (episode + 1 == episodes) or (episode == 0)
            if do_eval:
                eval_stats = _eval_once(
                    agent, env, arrival_plan, task_complex_index, MAX_STEPS,
                    use_heuristic_sort, env_seed, deadline_config, num_regions,
                )
            else:
                eval_stats = {
                    "score": float("nan"),
                    "app_to": float("nan"),
                    "task_to": float("nan"),
                    "delay": float("nan"),
                    "energy": float("nan"),
                    "total_energy": float("nan"),
                    "utility": float("nan"),
                    "route_stats": dict(ts.route_stats),
                }

            print(f"[{_ALGO_TAG}][Ep{episode+1:3d}] "
                  f"MeanR={mean_r:+.3f} Upd={ep_updates} "
                  f"ActorL={agent.last_actor_loss:.4f} CriticL={agent.last_critic_loss:.4f} "
                  f"| Train: Score={util:.3f} AppTO={appto:.2f}% E={totalE:.3f} "
                  f"| Eval: Score={eval_stats['score']:.1f} "
                  f"AppTO={eval_stats['app_to']:.2f}% "
                  f"E={eval_stats['total_energy']:.3f} "
                  f"Route={eval_stats['route_stats']}")

            row = {
                "episode": episode + 1,
                "score": util, "app_to": appto, "total_energy": totalE,
                "utility": util, "mean_reward": mean_r,
                "eval_score": eval_stats["score"],
                "eval_app_to": eval_stats["app_to"],
                "eval_task_to": eval_stats.get("task_to", float("nan")),
                "eval_delay": eval_stats.get("delay", float("nan")),
                "eval_energy": eval_stats.get("energy", float("nan")),
                "eval_total_energy": eval_stats["total_energy"],
                "eval_utility": eval_stats["utility"],
                "actor_loss": agent.last_actor_loss,
                "critic_loss": agent.last_critic_loss,
                "updates": ep_updates,
            }
            all_metrics.append(row)
            if CONFIG.get("RUN_DIR"):
                from pathlib import Path as _Path
                curve_file = _Path(CONFIG["RUN_DIR"]) / "curves" / f"{_ALGO_TAG}_seed{seed_offset}.csv"
                curve_file.parent.mkdir(parents=True, exist_ok=True)
                append_curve_row(curve_file, row)

            # Checkpoint best (only when we actually evaluated)
            if do_eval and eval_stats["score"] > best_eval_score:
                best_eval_score = eval_stats["score"]
                if checkpoint_dir is not None:
                    try:
                        torch.save(
                            agent.state_dict(),
                            str(checkpoint_dir / f"{_ALGO_TAG}_seed{seed_offset}_best.pt"),
                        )
                    except Exception as e:
                        print(f"[{_ALGO_TAG}] ckpt save failed: {e}")

        # Build dividelong-compatible (energy, delay, metrics) tuple.
        # Use the LAST eval-episode as the reported final point (or last train
        # row if no eval ran due to short episodes).
        if all_metrics:
            last = all_metrics[-1]
            # Prefer eval values if they are finite; otherwise fall back to train.
            _final_e = float(last.get("eval_total_energy", float("nan")))
            if not (_final_e == _final_e):  # NaN check
                _final_e = float(last.get("total_energy", 1000.0))
            _final_appto = float(last.get("eval_app_to", float("nan")))
            if not (_final_appto == _final_appto):
                _final_appto = float(last.get("app_to", 1.0))
            _final_score = float(last.get("eval_score", float("nan")))
            if not (_final_score == _final_score):
                _final_score = float(last.get("score", -1000.0))
            # 真实 delay 和 task_to
            _final_delay = float(last.get("eval_delay", float("nan")))
            if not (_final_delay == _final_delay):
                _final_delay = float(last.get("delay", 0.0))
            _final_task_to = float(last.get("eval_task_to", float("nan")))
            if not (_final_task_to == _final_task_to):
                _final_task_to = float(last.get("task_to", _final_appto))
        else:
            _final_e, _final_appto, _final_score, _final_delay, _final_task_to = 1000.0, 1.0, -1000.0, 0.0, 1.0

        _metrics_out = {
            "app_timeout_rate": _final_appto,   # 百分比 0~100 (compute_utility_score 口径)
            "task_timeout_rate": _final_task_to, # 百分比 0~100
            "timeout_rate": _final_appto,
            "score": _final_score,
            "energy": float(last.get("eval_energy", last.get("energy", 0.0))) if all_metrics else 0.0,
            "delay": _final_delay,
            "best_eval_score": float(best_eval_score) if best_eval_score != -float("inf") else _final_score,
            "episodes": int(len(all_metrics)),
            "algorithm": _ALGO_TAG,
        }
        return float(_final_e), float(_final_delay), _metrics_out

    except Exception as e:
        print(f"[{_ALGO_TAG}] ERROR: {e}")
        traceback.print_exc()
        return 1000.0, 1000.0, {
            "app_timeout_rate": 100.0,   # 百分比口径
            "task_timeout_rate": 100.0,
            "timeout_rate": 100.0,
            "error": str(e),
            "algorithm": _ALGO_TAG,
        }
    finally:
        # R2 isolation guarantee: always restore TaskScheduler.add_task on exit
        # so the c_alloc monkey-patch never leaks into other algorithms that
        # share the same Python process (e.g. dividelong subprocess re-use).
        try:
            _restore_c_alloc_patch()
            print(f"[{_ALGO_TAG}] c_alloc patch restored on exit")
        except Exception as _re:
            print(f"[{_ALGO_TAG}] WARN: failed to restore patch: {_re}")
