import os
import sys
import time
import traceback
import numpy as np
import torch
from pathlib import Path

# ==========================================
# 1. 路径与环境设置
# ==========================================
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入工具函数
from Experiments_new.exp_utils import (
    CONFIG, init_worker, make_run_dir, seed_everything,
    generate_arrival_plan, save_deadline_config, load_deadline_config, load_arrival_plan,
    compute_score, compute_utility_score, calc_timeout_rate, safe_rest_tasks_total, all_arrived_done,
    get_graph_cache, subtask_outcome_stats, apply_arrival_plan, subtask_partition_stats
)

# 导入环境和算法
from Environment.environment import Environment
from scheduler.graph_scheduler import GraphScheduler
from scheduler.task_scheduler import TaskScheduler
from Algorithms.Benchmark import Benchmark as BenchmarkClass
from utils.constant import para

# [替换1] 导入 TransEdgeStyle
from Algorithms.StrongBaselines.transedge_style_agent import TransEdgeStylePPOAgent
from Algorithms.StrongBaselines.transedge_style_core import extract_transedge_state

print_lock = None


def compute_structured_step_reward(
    app_timeout,
    app_done,
    task_timeout,
    energy_norm,
    delay_norm,
    action_val,
    route_stats,
    edge_feasible=True,
    local_feasible=True,
):
    """
    v2 structured reward:
    A. 应用级结果优先
    B. task-level cost shaping
    C. routing structure regularization
    """

    if app_timeout:
        return -3.0

    if app_done:
        return 2.0 - 0.5 * energy_norm - 0.5 * delay_norm

    reward = 0.0

    if task_timeout:
        reward -= 0.8
    else:
        reward += 0.15

    reward += 0.08 * (1.0 - energy_norm)
    reward += 0.08 * (1.0 - delay_norm)

    if route_stats is not None:
        total = max(1, route_stats.get("local", 0) + route_stats.get("edge", 0) + route_stats.get("cloud", 0))
        local_ratio = route_stats.get("local", 0) / total
        edge_ratio = route_stats.get("edge", 0) / total
        cloud_ratio = route_stats.get("cloud", 0) / total

        if action_val == 1 and cloud_ratio > 0.70 and edge_feasible:
            reward -= 0.20 * ((cloud_ratio - 0.70) / 0.30)

        if action_val >= 2 and edge_ratio < 0.25 and edge_feasible:
            reward += 0.10 * ((0.25 - edge_ratio) / 0.25)

        if action_val == 0 and local_ratio < 0.10 and local_feasible:
            reward += 0.04

    return reward


def set_print_lock(lock):
    global print_lock
    print_lock = lock


def eval_transedge_style_once(agent, ts, gs, bc, arrival_plan, task_complex_index, MAX_STEPS, use_heuristic_sort=False):
    """
    TransEdgeStyle 单次确定性评估 (deterministic=True)
    """
    user_num = para["user_num"]
    subgraph_num = 20
    basegraph_num = 60
    env = ts.env

    env_seed = (CONFIG["SEED"] if "SEED" in CONFIG else 0)
    deadline_config = load_deadline_config(CONFIG.get("RUN_DIR", ""))

    ts_eval = TaskScheduler(user_num, subgraph_num, basegraph_num, env,
                            tight_deadline_config=deadline_config, seed=env_seed)
    gs_eval = GraphScheduler(env.basegraph, env.subgraph_list, ts_eval)
    bc_eval = BenchmarkClass(env, gs_eval, ts_eval, task_complex_index, effective=True, seed=env_seed)
    ts_eval.env = env
    ts_eval.using_Algorithm = -1
    bc_eval.reset()
    ts_eval.reset()

    ts_eval.route_stats = {
        "local": 0,
        "edge": 0,
        "cloud": 0,
        "recent": []
    }

    agent.policy.eval()

    eval_task2action = {}
    eval_decision_count = 0

    for slot in range(MAX_STEPS):
        apply_arrival_plan(ts_eval, slot, arrival_plan)
        ts_eval.check_timeouts(slot)
        tasks = gs_eval.get_tasks(slot, sort_tasks=use_heuristic_sort)

        if not tasks:
            if (slot >= CONFIG['STOP_ARRIVAL_STEP'] and
                    safe_rest_tasks_total(ts_eval.rest_tasks) == 0 and
                    all_arrived_done(ts_eval)):
                break
            continue

        for task in tasks:
            # [替换2] 使用 extract_transedge_state
            s_data, mask = extract_transedge_state(
                ts_eval, task, slot=slot, task_complex_index=task_complex_index
            )
            act, _ = agent.take_action(
                s_data,
                action_mask=mask,
                deterministic=True,
                route_stats=getattr(ts_eval, "route_stats", None)
            )
            bc_eval.step([[task, act]])

            eval_task2action[tuple(task)] = int(act)
            eval_decision_count += 1

            if act == 0:
                ts_eval.route_stats["local"] += 1
            elif act == 1:
                ts_eval.route_stats["cloud"] += 1
            else:
                ts_eval.route_stats["edge"] += 1
            ts_eval.route_stats["recent"].append(int(act))
            if len(ts_eval.route_stats["recent"]) > 100:
                ts_eval.route_stats["recent"].pop(0)

    ts_eval.finalize_episode(MAX_STEPS - 1)
    e_succ, d_succ = ts_eval.get_avg_results(only_successful=True)
    to_info = calc_timeout_rate(ts_eval)

    app_to = float(to_info['app_timeout_rate'])
    task_to = float(to_info.get('task_timeout_rate', to_info['app_timeout_rate']))

    if app_to > 1.0: app_to /= 100.0
    if task_to > 1.0: task_to /= 100.0

    assert 0.0 <= app_to <= 1.0, f"app_to={app_to} 超出范围"
    assert 0.0 <= task_to <= 1.0, f"task_to={task_to} 超出范围"

    sla0 = CONFIG.get("SLA0", 0.95)
    kappa = CONFIG.get("KAPPA", 2.0)
    v_cap = 3.0

    score = compute_score(
        E_avg=e_succ,
        D_succ_avg=d_succ,
        rho_app=app_to,
        rho_task=task_to,
        sla0=sla0,
        kappa=kappa,
        v_cap=v_cap
    )

    eval_missed_tasks = 0
    eval_total_timeout_tasks = 0
    eval_timeout_scheduled = 0
    eval_timeout_unscheduled = 0

    for uid in range(ts_eval.user_num):
        if ts_eval.enter_time[uid] == float("inf"):
            continue

        for sid, ft in ts_eval.finish_time[uid].items():
            if ft == float("inf"):
                eval_total_timeout_tasks += 1
                if (uid, sid) in eval_task2action:
                    eval_timeout_scheduled += 1
                else:
                    eval_timeout_unscheduled += 1
                    eval_missed_tasks += 1

    if eval_missed_tasks > 0:
        print(f"[TransEdgeStyle][EVAL] 发现 {eval_missed_tasks} 个未调度任务！")
        print(
            f"[TransEdgeStyle][EVAL]  - 总超时: {eval_total_timeout_tasks} (已调度:{eval_timeout_scheduled}, 未调度:{eval_timeout_unscheduled})")
        print(f"[TransEdgeStyle][EVAL]  - 决策次数: {eval_decision_count}")

    partition = subtask_partition_stats(ts_eval, eval_task2action)
    to_info['action_stats'] = {
        "local": partition["local"],
        "cloud": partition["cloud"],
        "edge": partition["edge"],
        "timeout": partition["timeout"],
        "unknown": partition["unknown"],
        "total": partition["total_subtasks"],
        "total_actions": len(eval_task2action),
    }

    print(f"[TransEdgeStyle][EVAL] AppTO={app_to:.2%}, Score={score:.3f}, "
          f"E={e_succ:.4f}, D={d_succ:.4f}")
    print(f"[TransEdgeStyle][EVAL] action_stats: Local={partition['local']}, "
          f"Cloud={partition['cloud']}, Edge={partition['edge']}, Timeout={partition['timeout']}")

    total_e = ts_eval.get_sum_energy()
    print(f"[TransEdgeStyle][EVAL] TotalEnergy={total_e:.4f}")

    return float(e_succ), float(d_succ), float(score), to_info, float(total_e)


def train_transedge_style_wrapper(gpu_id, seed_offset, use_heuristic_sort=False, episodes=50):
    """TransEdgeStyle 训练入口"""
    init_worker(seed_offset, para, CONFIG)

    if "RUN_DIR" in os.environ: CONFIG["RUN_DIR"] = os.environ["RUN_DIR"]
    if "EVAL_MODE" in os.environ: CONFIG["EVAL_MODE"] = (os.environ["EVAL_MODE"] == "True")

    device = torch.device(f'cuda:{gpu_id}') if torch.cuda.is_available() else torch.device('cpu')

    checkpoint_dir = None
    if CONFIG.get("RUN_DIR"):
        checkpoint_dir = Path(CONFIG["RUN_DIR"]) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    try:
        # [替换3] 算法名 TransEdgeStyle
        if checkpoint_dir:
            old_checkpoint = checkpoint_dir / "TransEdgeStyle.pt"
            if old_checkpoint.exists():
                old_checkpoint.unlink()
                print(f"[TransEdgeStyle] 已删除旧 checkpoint: {old_checkpoint}")

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
            print(f"[TransEdgeStyle] 生成随机到达计划 (seed_offset={seed_offset})")
            env_seed = CONFIG["SEED"] + seed_offset
            arrival_plan = generate_arrival_plan(
                env_seed, MAX_STEPS, CONFIG["STOP_ARRIVAL_STEP"],
                0.3, CONFIG.get("BURST_PROB", 0.15),
                max(1, CONFIG.get("BURST_SIZE", 4) // 2), CONFIG.get("BURST_SIZE", 4)
            )
        else:
            arrival_plan = plan_data.get("arrival_plan", [])
            env_seed = plan_data.get("env_seed", CONFIG["SEED"] + seed_offset)
            print(f"[TransEdgeStyle] 加载到达计划 (env_seed={env_seed})")

        env.generate_components(seed=env_seed)
        G = get_graph_cache(user_num, subgraph_num, basegraph_num, project_root)
        if G is not None and env.basegraph: env.basegraph.nx_graph = G

        deadline_config = load_deadline_config(run_dir)
        if deadline_config is None:
            print(f"[TransEdgeStyle] 【警告】未找到 deadline 配置，使用默认随机生成")
        else:
            print(f"[TransEdgeStyle] 【环境一致性】加载预计算的 deadline 配置: "
                  f"{len(deadline_config['tight_user_ids'])} 个紧 deadline 用户")

        ts = TaskScheduler(user_num, subgraph_num, basegraph_num, env,
                           tight_deadline_config=deadline_config, seed=env_seed)
        gs = GraphScheduler(env.basegraph, env.subgraph_list, ts)
        bc = BenchmarkClass(env, gs, ts, task_complex_index, effective=True, seed=env_seed)

        ts.env = env
        ts.using_Algorithm = -1
        bc.reset();
        ts.reset()

        # 自动检测特征维度
        try:
            dummy_task = (0, 0)
            dummy_data, _ = extract_transedge_state(
                ts, dummy_task, slot=0, task_complex_index=task_complex_index
            )
            state_dim = dummy_data.x.shape[1]
            print(f"[TransEdgeStyle] 自动检测节点维度: {state_dim}")
        except Exception as e:
            print(f"[TransEdgeStyle] 维度检测失败，使用默认值 52. Error: {e}")
            state_dim = 52

        action_dim = para["edge_num"] + 2

        # [替换1] 使用 TransEdgeStylePPOAgent（无 cloud_reg_coef 等参数）
        agent = TransEdgeStylePPOAgent(
            node_dim=state_dim,
            action_dim=action_dim,
            device=device,
            lr=3e-4, gamma=0.99, lmbda=0.95,
            eps_clip=0.2, K_epochs=4, entropy_coef=0.01
        )

        best_score = float('inf')
        best_metrics = None

        best_eval_score = float('inf')
        best_eval_metrics = None
        best_eval_appTO = 1.0
        EVAL_EVERY = 5

        eval_curve_data = []
        eval_run_count = 0

        # 训练前评估
        print("\n[TransEdgeStyle] === 初始化：训练前先评估一次 ===")
        ee0, dd0, sc0, to0, te0 = eval_transedge_style_once(
            agent, ts, gs, bc, arrival_plan, task_complex_index, MAX_STEPS, use_heuristic_sort
        )
        best_eval_score = sc0
        best_eval_appTO = to0['app_timeout_rate']
        best_eval_metrics = {
            "e": ee0, "d": dd0, "score": sc0, "total_energy": te0,
            "timeout_info": dict(to0)
        }
        print(f"[InitEval] Score={sc0:.3f}, AppTO={to0['app_timeout_rate']:.2%}, TotalEnergy={te0:.4f}")
        print(f"[InitEval] 初始化完成，后续训练将以这个水平为基准\n")

        # 5. 训练循环
        for episode in range(episodes):
            bc.reset();
            ts.reset()

            ts.route_stats = {
                "local": 0,
                "edge": 0,
                "cloud": 0,
                "recent": []
            }

            agent.set_episode(episode)

            episode_rewards = []
            decision_count = 0

            action_count = [0] * action_dim
            total_actions = 0
            task2action = {}

            prev_app_to = set(ts.application_timeout_finished)
            prev_app_fin = set(ts.application_finished)

            arrived_tasks_total = 0

            for slot in range(MAX_STEPS):
                apply_arrival_plan(ts, slot, arrival_plan)
                ts.check_timeouts(slot)

                tasks = gs.get_tasks(slot, sort_tasks=use_heuristic_sort)
                if not tasks:
                    if (slot >= CONFIG['STOP_ARRIVAL_STEP'] and
                            safe_rest_tasks_total(ts.rest_tasks) == 0 and
                            all_arrived_done(ts)):
                        break
                    continue

                for task in tasks:
                    arrived_tasks_total += 1

                    # [替换2] 使用 extract_transedge_state
                    state_data, mask_bin = extract_transedge_state(
                        ts, task, slot=slot, task_complex_index=task_complex_index
                    )

                    action, log_prob = agent.take_action(
                        state_data,
                        action_mask=mask_bin,
                        route_stats=getattr(ts, "route_stats", None)
                    )

                    if action < len(action_count):
                        action_count[action] += 1
                    total_actions += 1
                    task2action[tuple(task)] = int(action)

                    if action == 0:
                        ts.route_stats["local"] += 1
                    elif action == 1:
                        ts.route_stats["cloud"] += 1
                    else:
                        ts.route_stats["edge"] += 1
                    ts.route_stats["recent"].append(int(action))
                    if len(ts.route_stats["recent"]) > 100:
                        ts.route_stats["recent"].pop(0)

                    reward, info = bc.step([[task, action]])

                    curr_app_to = set(ts.application_timeout_finished)
                    curr_app_fin = set(ts.application_finished)
                    new_app_to = curr_app_to - prev_app_to
                    new_app_fin = curr_app_fin - prev_app_fin
                    prev_app_to, prev_app_fin = curr_app_to, curr_app_fin

                    current_energy = float(info.get("step_energy", 0.0))
                    e_norm = max(0.0, min(1.0, (current_energy - 0.3) / (3.0 - 0.3)))

                    app_id = task[0]

                    step_delay = float(info.get("step_delay", 0.0))
                    delay_norm = max(0.0, min(1.0, step_delay / 5.0))

                    task_timeout = float(reward) < -0.1
                    route_stats = getattr(ts, "route_stats", None)

                    r_utility = compute_structured_step_reward(
                        app_timeout=(app_id in new_app_to),
                        app_done=(app_id in new_app_fin),
                        task_timeout=task_timeout,
                        energy_norm=e_norm,
                        delay_norm=delay_norm,
                        action_val=action,
                        route_stats=route_stats,
                        edge_feasible=True,
                        local_feasible=True,
                    )

                    r_scaled = r_utility

                    done = False
                    state_cpu = state_data.clone().cpu()

                    route_stats_snapshot = None
                    if hasattr(ts, "route_stats"):
                        route_stats_snapshot = {
                            "local": ts.route_stats.get("local", 0),
                            "edge": ts.route_stats.get("edge", 0),
                            "cloud": ts.route_stats.get("cloud", 0),
                            "recent": list(ts.route_stats.get("recent", []))
                        }

                    agent.put_data((state_cpu, action, r_scaled, log_prob, done, mask_bin, route_stats_snapshot))

                    episode_rewards.append(r_scaled)
                    decision_count += 1

            terminal_bonus_total = 0.0

            if len(agent.memory) > 0:
                temp_to_info = calc_timeout_rate(ts)
                temp_app_to = float(temp_to_info['app_timeout_rate'])
                if temp_app_to > 1.0:
                    temp_app_to /= 100.0

                if temp_app_to <= 1e-6:
                    terminal_bonus_total = 1.0
                else:
                    terminal_bonus_total = -5.0 * temp_app_to

            missed_tasks = 0
            total_timeout_tasks = 0
            timeout_scheduled = 0
            timeout_unscheduled = 0

            for uid in range(ts.user_num):
                if ts.enter_time[uid] == float("inf"):
                    continue

                for sid, ft in ts.finish_time[uid].items():
                    if ft == float("inf"):
                        total_timeout_tasks += 1
                        if (uid, sid) in task2action:
                            timeout_scheduled += 1
                        else:
                            timeout_unscheduled += 1
                            missed_tasks += 1

            if missed_tasks > 0:
                print(f"[TransEdgeStyle] 警报：Episode {episode + 1} 发现 {missed_tasks} 个任务从未被调度就超时！")
                print(
                    f"[TransEdgeStyle]  - 总超时任务: {total_timeout_tasks} (已被调度:{timeout_scheduled}, 未被调度:{timeout_unscheduled})")
                print(f"[TransEdgeStyle]  - 决策次数: {decision_count}")
                print(f"[TransEdgeStyle]  - [WARN] 暂不添加 missed_penalty（疑似 triage/死尸过滤导致）")

            num_transitions = len(agent.memory)
            if num_transitions > 0:
                terminal_bonus_per_step = terminal_bonus_total / num_transitions if terminal_bonus_total != 0 else 0.0

                updated_memory = []
                for transition in agent.memory:
                    updated_reward = transition[2] + terminal_bonus_per_step
                    updated_transition = (
                        transition[0],
                        transition[1],
                        updated_reward,
                        transition[3],
                        transition[4],
                        transition[5],
                        transition[6]
                    )
                    updated_memory.append(updated_transition)
                agent.memory = updated_memory

                if terminal_bonus_total != 0:
                    print(f"[TransEdgeStyle] Episode {episode + 1} 终止引导: AppTO={temp_app_to:.2%}, "
                          f"Bonus总={terminal_bonus_total:.3f}, 每step={terminal_bonus_per_step:.4f} (共{num_transitions}步)")

                episode_rewards.append(terminal_bonus_total)

            if len(agent.memory) > 0:
                agent.update()
                agent.clear_memory()

            if missed_tasks == 0 and total_timeout_tasks > 0:
                print(f"[TransEdgeStyle] Episode {episode + 1} 未调度任务检测: 无遗漏 (所有超时任务都曾被调度)")

            # 论文口径: delay 用 D_all, rho 传百分比 (calc_timeout_rate 返回小数 → ×100)
            _te_succ = ts.get_avg_results(only_successful=True)
            _te_all = ts.get_avg_results(only_successful=False, timeout_charge="deadline")
            _te_to = calc_timeout_rate(ts)
            utility_score = compute_utility_score(
                E_avg=_te_all[0],
                D_succ_avg=_te_all[1],
                rho_app=float(_te_to['app_timeout_rate']) * 100.0,
                rho_task=float(_te_to.get('task_timeout_rate', 0)) * 100.0,
            )

            if (episode + 1) % 20 == 0 and total_actions > 0:
                print(f"[TransEdgeStyle] 动作分布统计 (共{total_actions}个动作):")
                print(f"  - Local (action=0): {action_count[0]} ({100.0 * action_count[0] / total_actions:.1f}%)")
                if action_dim > 1:
                    print(f"  - Cloud (action=1): {action_count[1]} ({100.0 * action_count[1] / total_actions:.1f}%)")
                if action_dim > 2:
                    edge_total = sum(action_count[2:])
                    print(
                        f"  - Edge (action=2..{action_dim - 1}): {edge_total} ({100.0 * edge_total / total_actions:.1f}%)")

            if total_actions > 0:
                local_pct = 100.0 * action_count[0] / total_actions
                cloud_pct = 100.0 * action_count[1] / total_actions if action_dim > 1 else 0.0
                edge_count = sum(action_count[2:]) if action_dim > 2 else 0
                edge_pct = 100.0 * edge_count / total_actions if action_dim > 2 else 0.0
                print(f"[TransEdgeStyle] Ep {episode + 1} 动作选择: Local={action_count[0]}({local_pct:.1f}%), "
                      f"Cloud={action_count[1]}({cloud_pct:.1f}%), Edge={edge_count}({edge_pct:.1f}%), Total={total_actions}")

            # 定期评估
            if (episode + 1) % EVAL_EVERY == 0:
                print(f"\n[TransEdgeStyle] === Episode {episode + 1}: 开始评估 ===")

                ee, dd, sc, to, te = eval_transedge_style_once(
                    agent, ts, gs, bc, arrival_plan, task_complex_index, MAX_STEPS, use_heuristic_sort
                )
                curr_appTO = to['app_timeout_rate']
                eval_run_count += 1

                eval_curve_data.append({
                    "episode": episode + 1,
                    "eval_run": eval_run_count,
                    "score": float(sc),
                    "energy": float(ee),
                    "total_energy": float(te),
                    "delay": float(dd),
                    "app_timeout_rate": float(curr_appTO),
                    "task_timeout_rate": float(to.get('task_timeout_rate', curr_appTO)),
                })

                print(f"[Monitor] Ep {episode + 1}, "
                      f"curr_eval_AppTO={curr_appTO:.2%}, "
                      f"best_eval_AppTO_so_far={best_eval_appTO:.2%}")

                if (curr_appTO < best_eval_appTO - 1e-9) or \
                        (abs(curr_appTO - best_eval_appTO) < 1e-9 and sc < best_eval_score):
                    best_eval_score = sc
                    best_eval_appTO = curr_appTO
                    best_eval_metrics = {
                        "e": float(ee),
                        "d": float(dd),
                        "score": float(sc),
                        "total_energy": float(te),
                        "timeout_info": dict(to),
                    }
                    print(f"[ACCEPT] [BEST] 新最佳评估: AppTO={curr_appTO:.2%}, Score={sc:.3f}, TotalEnergy={te:.4f}")

                    if checkpoint_dir:
                        torch.save(agent.policy.state_dict(), checkpoint_dir / "TransEdgeStyle.pt")
                else:
                    print(f"[KEEP] 本轮未创新高 (Score={sc:.3f} >= Best={best_eval_score:.3f})")

            if episode_rewards:
                episode_rewards_array = np.array(episode_rewards)
                total_reward = float(np.sum(episode_rewards))
                print(
                    f"[TransEdgeStyle] Ep {episode + 1} Reward统计: sum={total_reward:.4f}, mean={episode_rewards_array.mean():.4f}, "
                    f"min={episode_rewards_array.min():.4f}, max={episode_rewards_array.max():.4f}, count={len(episode_rewards)}")
            else:
                total_reward = 0.0
                print(
                    f"[TransEdgeStyle] Ep {episode + 1} Reward统计: 无数据 (episode_rewards 为空, total_actions={total_actions}, decision_count={decision_count})")

            train_to_info = calc_timeout_rate(ts)
            e_succ, d_succ = ts.get_avg_results(only_successful=True)

            from contextlib import nullcontext
            with (print_lock if print_lock else nullcontext()):
                print(f"[TransEdgeStyle] Ep {episode + 1}/{episodes} 完成 (训练): "
                      f"App超时率={train_to_info['app_timeout_rate']:.2%}, Score={utility_score:.4f}, "
                      f"E={e_succ:.4f}, D={d_succ:.4f}, Reward={total_reward:.4f}")

            # 保存训练曲线
            if checkpoint_dir:
                try:
                    train_curve_file = Path(CONFIG["RUN_DIR"]) / "curves" / f"TransEdgeStyle_seed{seed_offset}_train.csv"
                    train_curve_file.parent.mkdir(parents=True, exist_ok=True)
                    from Experiments_new.exp_utils import append_curve_row
                    append_curve_row(train_curve_file, {
                        "episode": episode + 1,
                        "reward": total_reward,
                        "score": float(utility_score),
                        "utility_score": float(utility_score),
                        "energy": float(e_succ),
                        "delay": float(d_succ),
                        "app_timeout_rate": float(train_to_info['app_timeout_rate']),
                        "task_timeout_rate": float(train_to_info.get('task_timeout_rate', 0)),
                        "best_eval_score": float(best_eval_score),
                        "curve_type": "training",
                    })
                except Exception as curve_err:
                    print(f"[TransEdgeStyle] 保存训练曲线失败: {curve_err}")

        # 保存评估曲线
        if checkpoint_dir and len(eval_curve_data) > 0:
            try:
                eval_curve_file = Path(CONFIG["RUN_DIR"]) / "curves" / f"TransEdgeStyle_seed{seed_offset}_eval.csv"
                eval_curve_file.parent.mkdir(parents=True, exist_ok=True)
                from Experiments_new.exp_utils import append_curve_row
                for eval_data in eval_curve_data:
                    append_curve_row(eval_curve_file, {
                        "episode": eval_data["episode"],
                        "eval_run": eval_data["eval_run"],
                        "score": eval_data["score"],
                        "energy": eval_data["energy"],
                        "delay": eval_data["delay"],
                        "app_timeout_rate": eval_data["app_timeout_rate"],
                        "task_timeout_rate": eval_data["task_timeout_rate"],
                        "best_score": float(best_eval_score),
                        "curve_type": "evaluation",
                    })
                print(f"[TransEdgeStyle] 评估曲线已保存: {len(eval_curve_data)} 条记录")
                print(f"[TransEdgeStyle]  - 训练曲线: {train_curve_file.name}")
                print(f"[TransEdgeStyle]  - 评估曲线: {eval_curve_file.name}")
            except Exception as eval_curve_err:
                print(f"[TransEdgeStyle] 保存评估曲线失败: {eval_curve_err}")

        # 返回结果
        if best_eval_metrics is not None:
            try:
                if checkpoint_dir:
                    checkpoint_path = checkpoint_dir / "TransEdgeStyle.pt"
                    if checkpoint_path.exists():
                        best_model_state = torch.load(checkpoint_path, map_location=device)
                        agent.policy.load_state_dict(best_model_state)
                        print(f"[TransEdgeStyle] 已恢复最佳评估模型")
            except Exception as e:
                print(f"[TransEdgeStyle] 恢复模型失败: {e}")

            print(f"\n[FINAL] [DONE] 训练完成!")
            print(f"[FINAL] 最佳评估 AppTO: {best_eval_appTO:.2%}")
            print(f"[FINAL] 最佳评估 Score: {best_eval_score:.3f}")
            print(f"[FINAL] 使用最佳评估模型返回结果\n")

            best_eval_metrics['timeout_info']['inference_time_ms'] = 0.0
            best_eval_metrics['timeout_info']['inference_stats'] = {
                'median': 0.0, 'min': 0.0, 'max': 0.0, 'std': 0.0, 'count': 0
            }

            # 对外 app_timeout_rate / task_timeout_rate 存【百分比】(0~100)
            _best_to = best_eval_metrics['timeout_info']
            _best_app_pct = float(_best_to.get('app_timeout_rate', 0)) * 100.0
            _best_task_pct = float(_best_to.get('task_timeout_rate', 0)) * 100.0

            print(f"[TransEdgeStyle] 最终结果 (评估最佳): "
                  f"E={best_eval_metrics['e']:.4f}, "
                  f"D={best_eval_metrics['d']:.4f}, "
                  f"Score={best_eval_metrics['score']:.4f}, "
                  f"AppTO={_best_app_pct:.2f}%")

            if 'action_stats' in _best_to:
                stats = _best_to['action_stats']
                print(f"[TransEdgeStyle] 【子任务分布统计】")
                print(f"  - Local 完成: {stats['local']}")
                print(f"  - Cloud 完成: {stats['cloud']}")
                print(f"  - Edge 完成: {stats['edge']}")
                print(f"  - 超时: {stats['timeout']}")
                print(f"  - 未知: {stats['unknown']}")
                print(f"  - 总计: {stats['total']}")

            flattened_metrics = {
                "app_timeout_rate": _best_app_pct,
                "task_timeout_rate": _best_task_pct,
                "score": best_eval_metrics.get('score', 0),
                "inference_time_ms": _best_to.get('inference_time_ms', 0.0),
                "action_stats": _best_to.get('action_stats', {}),
                "timeout_rate": _best_to,
                "total_energy": best_eval_metrics.get('total_energy', best_eval_metrics['e'])
            }

            sys.stdout.flush()
            agent.print_inference_stats()

            return best_eval_metrics["e"], best_eval_metrics["d"], flattened_metrics

        elif best_metrics:
            # 对外 app_timeout_rate / task_timeout_rate 存【百分比】(0~100)
            _app_pct = float(to_info['app_timeout_rate']) * 100.0
            print(f"[TransEdgeStyle] 最终结果: E={e_succ:.4f}, D={d_succ:.4f}, Score={score:.4f}, "
                  f"App超时率={_app_pct:.2f}%")

            to_info_with_inference = dict(to_info)
            to_info_with_inference['inference_time_ms'] = 0.0
            to_info_with_inference['inference_stats'] = {
                'median': 0.0, 'min': 0.0, 'max': 0.0, 'std': 0.0, 'count': 0
            }

            best_metrics = {
                "e": float(e_succ),
                "d": float(d_succ),
                "score": float(score),
                "timeout_info": to_info_with_inference,
                "action_count": [0] * (para["edge_num"] + 2),
                "total_actions": 0,
                "task2action": {},
            }

            try:
                total_e = ts.get_sum_energy()
            except:
                total_e = float(e_succ)
            best_metrics["total_energy"] = float(total_e)

            try:
                partition = subtask_partition_stats(ts, {})
                best_metrics['timeout_info']['action_stats'] = {
                    "local": partition.get("local", 0),
                    "cloud": partition.get("cloud", 0),
                    "edge": partition.get("edge", 0),
                    "timeout": partition.get("timeout", 0),
                    "unknown": partition.get("unknown", 0),
                    "total": partition.get("total_subtasks", 0),
                    "total_actions": 0,
                }
            except:
                pass

        if 'timeout_info' in best_metrics:
            to_info = best_metrics['timeout_info']
            # 对外存百分比 (0~100)
            flattened_metrics = {
                "app_timeout_rate": float(to_info.get('app_timeout_rate', 0)) * 100.0,
                "task_timeout_rate": float(to_info.get('task_timeout_rate', 0)) * 100.0,
                "score": best_metrics.get('score', 0),
                "inference_time_ms": to_info.get('inference_time_ms', 0.0),
                "action_stats": to_info.get('action_stats', {}),
                "timeout_rate": to_info,
                "total_energy": best_metrics.get('total_energy', best_metrics['e'])
            }
        else:
            flattened_metrics = {
                "app_timeout_rate": 0.0,
                "task_timeout_rate": 0.0,
                "score": best_metrics.get('score', 0),
                "inference_time_ms": 0.0,
                "action_stats": {},
                "timeout_rate": {},
                "total_energy": best_metrics.get('total_energy', best_metrics['e'])
            }

        sys.stdout.flush()
        agent.print_inference_stats()

        return best_metrics["e"], best_metrics["d"], flattened_metrics

    except Exception as e:
        print(f"[TransEdgeStyle] 训练崩溃: {e}")
        traceback.print_exc()
        return 1000.0, 1000.0, {'app_timeout_rate': 100.0, 'task_timeout_rate': 100.0, 'score': float('inf')}
