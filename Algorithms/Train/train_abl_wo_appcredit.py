# -*- coding: utf-8 -*-
"""CPGAPPO + CP-aware guide + shield + app 连坐奖励。
基于 train_cpgappo_unified.py，唯一改动：使用 compute_guide_scores_cp 替代 compute_guide_scores。
"""
import os, sys, time, traceback
import numpy as np
import torch
from pathlib import Path

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Experiments_new.exp_utils import (
    CONFIG, init_worker, generate_arrival_plan, load_deadline_config,
    compute_score, calc_timeout_rate, safe_rest_tasks_total, all_arrived_done,
    get_graph_cache, apply_arrival_plan, subtask_partition_stats, append_curve_row
)
from Environment.environment import Environment
from scheduler.graph_scheduler import GraphScheduler
from scheduler.task_scheduler import TaskScheduler
from Algorithms.Benchmark import Benchmark as BenchmarkClass
from utils.constant import para

from Algorithms.RealGATPPO.agent_cpgappo import GAT_PPO_Agent_CPGAPPO, compute_guide_scores_cp
from Algorithms.RealGATPPO.cpgappo_core import extract_cpgappo_state, compute_cpgappo_slack_reward

print_lock = None

def build_cpgappo_safe_mask(ts, uid, now, mask_bin, guide_scores):
    """
    基于 CPGAPPO 当前可得信息，构造高风险物理护盾 safe_mask
    """
    app_deadline_slot = ts.get_app_deadline_slot(uid) if hasattr(ts, "get_app_deadline_slot") else para["deadline_slot"]

    if uid < len(ts.enter_time) and ts.enter_time[uid] != float("inf"):
        enter_time = float(ts.enter_time[uid])
    else:
        enter_time = float(now)

    time_left = max(0.0, enter_time + app_deadline_slot * para["slot_interval"] - now)

    budget = max(1e-6, app_deadline_slot * para["slot_interval"])
    min_slack = time_left / budget
    pressure = 1.0 - min(1.0, time_left / max(1e-6, budget * 1.5))

    is_high_risk = (pressure > 0.5) or (min_slack < 0.5)
    threshold = 1.00 if is_high_risk else 1.05

    safe_mask = np.array(mask_bin, dtype=bool)
    for a in range(len(safe_mask)):
        if safe_mask[a] and guide_scores[a] > time_left * threshold:
            safe_mask[a] = False

    if not safe_mask.any():
        safe_mask = np.array(mask_bin, dtype=bool)

    return safe_mask


def eval_cpgappo_noappcredit_once(agent, ts, gs, bc, arrival_plan, task_complex_index, MAX_STEPS):
    """Evaluate once with the trained agent + shield + CP guide."""
    ts.reset()
    eval_task2action = {}
    action_count = [0] * (para["edge_num"] + 2)
    total_actions = 0

    for slot in range(MAX_STEPS):
        apply_arrival_plan(ts, slot, arrival_plan)
        ts.check_timeouts(slot)
        tasks = gs.get_tasks(slot, sort_tasks=True)
        if not tasks:
            if (slot >= CONFIG['STOP_ARRIVAL_STEP'] and
                    safe_rest_tasks_total(ts.rest_tasks) == 0 and
                    all_arrived_done(ts)):
                break
            continue
        for task in tasks:
            uid = task[0]
            now = slot * para["slot_interval"]

            s_data, g_feat, mask_bin = extract_cpgappo_state(ts, task, slot, task_complex_index)

            guide_scores, guide_valid = compute_guide_scores_cp(
                ts, uid, task, task_complex_index, now
            )
            masked_scores = np.where(guide_valid, guide_scores, 1e6)
            safe_mask = build_cpgappo_safe_mask(ts, uid, now, mask_bin, masked_scores)

            action, _ = agent.take_action(
                s_data, g_feat, action_mask=safe_mask, deterministic=True
            )
            bc.step([[task, action]])
            eval_task2action[tuple(task)] = int(action)
            if action < len(action_count):
                action_count[action] += 1
            total_actions += 1

    ts.finalize_episode(MAX_STEPS - 1)
    try:
        e, d = ts.get_avg_results(only_successful=True)
    except Exception:
        e, d = 0.0, 0.0
    to_info = calc_timeout_rate(ts)
    rho_app = float(to_info['app_timeout_rate'])
    rho_task = float(to_info.get('task_timeout_rate', 0))
    if rho_app > 1.0:
        rho_app /= 100.0
    if rho_task > 1.0:
        rho_task /= 100.0
    score = compute_score(e, d, rho_app, rho_task)
    total_energy = ts.total_energy if hasattr(ts, "total_energy") else 0.0

    partition = subtask_partition_stats(ts, eval_task2action)
    to_info['action_stats'] = {
        "local": partition["local"],
        "cloud": partition["cloud"],
        "edge": partition["edge"],
        "timeout": partition["timeout"],
        "unknown": partition["unknown"],
        "total": partition["total_subtasks"],
        "total_actions": len(eval_task2action),
    }
    total_sub = partition["total_subtasks"]
    finished_sub = partition["local"] + partition["cloud"] + partition["edge"]
    to_info['subtask_stats'] = {
        "total": total_sub,
        "finished": finished_sub,
        "unfinished": total_sub - finished_sub,
    }

    print(f"[CPGAPPO_noappcredit][EVAL] AppTO={rho_app:.2%}, TaskTO={rho_task:.2%}, Score={score:.3f}, "
          f"E={e:.4f}, D={d:.4f}")
    print(f"[CPGAPPO_noappcredit][EVAL] action_stats: Local={partition['local']}, Cloud={partition['cloud']}, "
          f"Edge={partition['edge']}, Timeout={partition['timeout']}, Unknown={partition['unknown']}")
    print(f"[CPGAPPO_noappcredit][EVAL] TotalEnergy={total_energy:.4f}, Subtasks: {finished_sub}/{total_sub}")
    sys.stdout.flush()

    return e, d, score, to_info, total_energy


def train_cpgappo_noappcredit(gpu_id, seed_offset, episodes=20, lr=3e-4, entropy_coef=0.01, lambda_guide=0.2):
    """V3-guided + CP guide + shield + app credit training entry."""
    init_worker(seed_offset, para, CONFIG)

    device = torch.device(f'cuda:{gpu_id}') if torch.cuda.is_available() else torch.device('cpu')

    checkpoint_dir = None
    curves_dir = None
    if CONFIG.get("RUN_DIR"):
        checkpoint_dir = Path(CONFIG["RUN_DIR"]) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        curves_dir = Path(CONFIG["RUN_DIR"]) / "curves"
        curves_dir.mkdir(parents=True, exist_ok=True)

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
    env_seed = CONFIG["SEED"] + seed_offset
    arrival_plan = generate_arrival_plan(
        env_seed, MAX_STEPS, CONFIG["STOP_ARRIVAL_STEP"],
        0.3, CONFIG.get("BURST_PROB", 0.15),
        max(1, CONFIG.get("BURST_SIZE", 4) // 2), CONFIG.get("BURST_SIZE", 4)
    )

    env.generate_components(seed=env_seed)
    G = get_graph_cache(user_num, subgraph_num, basegraph_num, project_root)
    if G is not None and env.basegraph: env.basegraph.nx_graph = G

    deadline_config = load_deadline_config("")
    ts = TaskScheduler(user_num, subgraph_num, basegraph_num, env,
                       tight_deadline_config=deadline_config, seed=env_seed)
    gs = GraphScheduler(env.basegraph, env.subgraph_list, ts)
    bc = BenchmarkClass(env, gs, ts, task_complex_index, effective=True, seed=env_seed)
    ts.env = env
    ts.using_Algorithm = -1
    bc.reset()
    ts.reset()

    try:
        dummy_task = (0, 0)
        dummy_data, dummy_gfeat, _ = extract_cpgappo_state(ts, dummy_task, slot=0, task_complex_index=task_complex_index)
        state_dim = dummy_data.x.shape[1]
        global_dim = dummy_gfeat.shape[0]
        print(f"[CPGAPPO_noappcredit] Auto-detect: node_dim={state_dim}, global_dim={global_dim}")
    except Exception as e:
        print(f"[CPGAPPO_noappcredit] Dim detect failed, using defaults. Error: {e}")
        state_dim = 27
        global_dim = 10

    action_dim = para["edge_num"] + 2

    agent = GAT_PPO_Agent_CPGAPPO(
        node_dim=state_dim, global_dim=global_dim, action_dim=action_dim,
        device=device, lr=lr, entropy_coef=entropy_coef, lambda_guide=lambda_guide,
    )

    best_eval_score = float('inf')
    best_eval_appTO = 1.0
    best_eval_metrics = None

    print("[CPGAPPO_noappcredit] === Init eval ===")
    ee0, dd0, sc0, to0, te0 = eval_cpgappo_noappcredit_once(
        agent, ts, gs, bc, arrival_plan, task_complex_index, MAX_STEPS)
    best_eval_score = sc0
    best_eval_appTO = to0['app_timeout_rate']

    print(f"\n[CPGAPPO_noappcredit] === Training {episodes} episodes (lambda_guide={lambda_guide}) ===\n")
    for episode in range(episodes):
        bc.reset()
        ts.reset()
        action_count = [0] * action_dim
        total_actions = 0
        episode_rewards = []

        prev_app_to = set(ts.application_timeout_finished)
        prev_app_fin = set(ts.application_finished)
        app_buffer = {}

        for slot in range(MAX_STEPS):
            apply_arrival_plan(ts, slot, arrival_plan)
            ts.check_timeouts(slot)
            tasks = gs.get_tasks(slot, sort_tasks=True)
            if not tasks:
                if (slot >= CONFIG['STOP_ARRIVAL_STEP'] and
                        safe_rest_tasks_total(ts.rest_tasks) == 0 and
                        all_arrived_done(ts)):
                    break
                continue

            now = slot * para["slot_interval"]

            for task in tasks:
                uid = task[0]
                sid = task[1]

                s_data, g_feat, mask_bin = extract_cpgappo_state(ts, task, slot, task_complex_index)

                guide_scores, guide_valid = compute_guide_scores_cp(
                    ts, uid, task, task_complex_index, now)
                masked_scores = np.where(guide_valid, guide_scores, 1e6)
                guide_best_action = int(np.argmin(masked_scores))

                safe_mask = build_cpgappo_safe_mask(ts, uid, now, mask_bin, masked_scores)

                action, log_prob = agent.take_action(s_data, g_feat, action_mask=safe_mask)

                if action < len(action_count):
                    action_count[action] += 1
                total_actions += 1

                reward, info = bc.step([[task, action]])

                current_energy = float(info.get("step_energy", 0.0))
                step_delay = float(info.get("step_delay", 0.0))
                r_scaled, slack = compute_cpgappo_slack_reward(
                    ts, uid, sid, current_energy, step_delay,
                    ts.enter_time[uid], ts.get_app_deadline_slot(uid))

                state_cpu = s_data.clone().cpu()
                app_buffer.setdefault(uid, []).append([
                    state_cpu,
                    g_feat.clone().cpu(),
                    action,
                    r_scaled,
                    log_prob,
                    False,
                    safe_mask,
                    guide_best_action
                ])

                episode_rewards.append(r_scaled)

                curr_app_to = set(ts.application_timeout_finished)
                curr_app_fin = set(ts.application_finished)
                new_app_to = curr_app_to - prev_app_to
                new_app_fin = curr_app_fin - prev_app_fin
                prev_app_to, prev_app_fin = curr_app_to, curr_app_fin

                for f_uid in (new_app_fin | new_app_to):
                    if f_uid in app_buffer:
                        for t in app_buffer[f_uid]:
                            agent.put_data(tuple(t))
                        del app_buffer[f_uid]

        for t_list in app_buffer.values():
            for t in t_list:
                agent.put_data(tuple(t))
        app_buffer.clear()

        ts.finalize_episode(MAX_STEPS - 1)
        temp_to = calc_timeout_rate(ts)
        temp_app_to = float(temp_to['app_timeout_rate'])
        if temp_app_to > 1.0:
            temp_app_to /= 100.0

        terminal_bonus = 1.0 if temp_app_to <= 1e-6 else -5.0 * temp_app_to
        n_t = len(agent.memory)
        if n_t > 0:
            bonus_per = terminal_bonus / n_t
            updated = []
            for t in agent.memory:
                updated.append((t[0], t[1], t[2], t[3] + bonus_per, t[4], t[5], t[6], t[7]))
            agent.memory = updated
            agent.update()
            agent.clear_memory()

        ep_reward_sum = sum(episode_rewards)
        ep_mean = np.mean(episode_rewards) if episode_rewards else 0
        print(f"[CPGAPPO_noappcredit] Ep {episode+1}/{episodes} train: AppTO={temp_app_to:.2%}, "
              f"Reward sum={ep_reward_sum:.2f}, mean={ep_mean:.4f}")
        sys.stdout.flush()

        if curves_dir is not None:
            try:
                train_curve_file = curves_dir / f"CPGAPPO_noappcredit_seed{seed_offset}_train.csv"
                append_curve_row(train_curve_file, {
                    "episode": episode + 1,
                    "reward": float(ep_reward_sum),
                    "score": float(-temp_app_to),
                    "utility_score": float(-temp_app_to),
                    "energy": float(getattr(ts, "total_energy", 0.0) / max(1, user_num)),
                    "delay": 0.0,
                    "app_timeout_rate": float(temp_app_to),
                    "task_timeout_rate": float(temp_to.get("task_timeout_rate", 0.0)),
                    "curve_type": "training",
                })
            except Exception as curve_err:
                print(f"[CPGAPPO_noappcredit] 保存训练曲线失败: {curve_err}")

        if (episode + 1) % 5 == 0 or (episode + 1) == episodes:
            print(f"\n[CPGAPPO_noappcredit] --- Eval at Ep {episode+1} ---")
            ee, dd, sc, to_info, te = eval_cpgappo_noappcredit_once(
                agent, ts, gs, bc, arrival_plan, task_complex_index, MAX_STEPS)

            eval_appTO = to_info['app_timeout_rate']
            if sc < best_eval_score:
                best_eval_score = sc
                best_eval_appTO = eval_appTO
                best_eval_metrics = {
                    "e": ee, "d": dd, "score": sc, "total_energy": te,
                    "timeout_info": dict(to_info),
                }
                print(f"[CPGAPPO_noappcredit][KEEP] New best score={sc:.3f}, AppTO={eval_appTO:.2%}")
                if checkpoint_dir is not None:
                    try:
                        torch.save(agent.policy.state_dict(), checkpoint_dir / "CPGAPPO_noappcredit.pt")
                    except Exception as ckpt_err:
                        print(f"[CPGAPPO_noappcredit] 保存checkpoint失败: {ckpt_err}")
            else:
                print(f"[CPGAPPO_noappcredit][Monitor] Ep {episode+1}, AppTO={eval_appTO:.2%}, "
                      f"best_AppTO_so_far={best_eval_appTO:.2%}")
            sys.stdout.flush()

            if curves_dir is not None:
                try:
                    eval_curve_file = curves_dir / f"CPGAPPO_noappcredit_seed{seed_offset}_eval.csv"
                    append_curve_row(eval_curve_file, {
                        "episode": episode + 1,
                        "score": float(sc),
                        "energy": float(ee),
                        "delay": float(dd),
                        "app_timeout_rate": float(eval_appTO),
                        "task_timeout_rate": float(to_info.get('task_timeout_rate', 0.0)),
                        "curve_type": "evaluation",
                    })
                except Exception as eval_curve_err:
                    print(f"[CPGAPPO_noappcredit] 保存评估曲线失败: {eval_curve_err}")

    if best_eval_metrics is None:
        ee, dd, sc, to_info, te = eval_cpgappo_noappcredit_once(
            agent, ts, gs, bc, arrival_plan, task_complex_index, MAX_STEPS)
        best_eval_metrics = {
            "e": ee, "d": dd, "score": sc, "total_energy": te,
            "timeout_info": dict(to_info),
        }

    # 【可复现性修复】无论 best 是否被刷新，都强制保存 _last ckpt
    # 避免出现"训练成功但 init eval score 从未被超越 → best ckpt 从不落盘"的情况
    if checkpoint_dir is not None:
        try:
            torch.save(agent.policy.state_dict(), checkpoint_dir / "CPGAPPO_noappcredit_last.pt")
        except Exception as ckpt_err:
            print(f"[CPGAPPO_noappcredit] 保存_last checkpoint失败: {ckpt_err}")

    print(f"\n[CPGAPPO_noappcredit][FINAL] Training complete!")
    print(f"[CPGAPPO_noappcredit][FINAL] Best eval AppTO: {best_eval_metrics['timeout_info']['app_timeout_rate']:.2%}")
    print(f"[CPGAPPO_noappcredit][FINAL] Best eval Score: {best_eval_metrics['score']:.3f}")
    sys.stdout.flush()

    ti = best_eval_metrics['timeout_info']
    return (best_eval_metrics["e"], best_eval_metrics["d"], {
        "app_timeout_rate": float(ti.get('app_timeout_rate', 1.0)),
        "task_timeout_rate": float(ti.get('task_timeout_rate', 1.0)),
        "score": float(best_eval_metrics['score']),
        "total_energy": float(best_eval_metrics['total_energy']),
        "inference_time_ms": float(np.mean(agent.inference_times)) if len(agent.inference_times) > 0 else 0.0,
        "action_stats": ti.get('action_stats', {}),
        "timeout_rate": {
            "app_timeout_rate": float(ti.get('app_timeout_rate', 1.0)),
            "task_timeout_rate": float(ti.get('task_timeout_rate', 1.0)),
            "action_stats": ti.get('action_stats', {}),
            "subtask_stats": ti.get('subtask_stats', {}),
        },
        "subtask_stats": ti.get('subtask_stats', {}),
    })


def train_wo_appcredit(gpu_id, seed_offset, use_heuristic_sort=True, episodes=20, lr=3e-4, entropy_coef=0.01, lambda_guide=0.2):
    """供 dividelong.py 统一调用的包装器"""
    e, d, metrics = train_cpgappo_noappcredit(
        gpu_id=gpu_id,
        seed_offset=seed_offset,
        episodes=episodes,
        lr=lr,
        entropy_coef=entropy_coef,
        lambda_guide=lambda_guide
    )
    return float(e), float(d), metrics
