# -*- coding: utf-8 -*-
"""
Shared training utilities for all algorithms.

English
-------
common.py holds the cross-algorithm helpers used by every training wrapper in
Algorithms/Train/:
  - print_lock / set_print_lock: optional lock for synchronized multi-process
    stdout (set by the runner entry script, not fenxi.py).
  - decode_action(action): maps an integer action to (target_name, edge_id).
    0=Local(-1), 1=Cloud(-1), 2..=Edge(0-based id).
  - ensure_trace_dir(run_dir): creates the trace/curve output directory.
  - run_benchmark_worker(...): the heuristic-baseline worker loop
    (Local/Cloud/Edge/Greedy/Genetic/HybridPSOGA) shared by all runners, so
    that RL agents and heuristics run under the identical env/arrival plan
    (fair-comparison protocol).

中文
----
所有训练 wrapper 共享的工具: 打印锁, 动作解码, 输出目录, 启发式基线 worker 循环
(公平对比协议要求 RL 与启发式跑同一份 env/arrival plan)。
"""
import os
import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path

# 引入项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  # 仓库根 (本文件在 Algorithms/Train/ 下, 上两级)
sys.path.insert(0, project_root)

# 添加 Experiments_new 到路径（因为 exp_utils 在那里）
experiments_dir = os.path.join(project_root, 'Experiments_new')
if experiments_dir not in sys.path:
    sys.path.insert(0, experiments_dir)

from Environment.environment import Environment
from scheduler.graph_scheduler import GraphScheduler
from scheduler.task_scheduler import TaskScheduler
from Algorithms import Benchmark
from utils.constant import para

from Experiments_new.exp_utils import (
    CONFIG, init_worker, load_arrival_plan, generate_arrival_plan, apply_arrival_plan,
    load_deadline_config,
    get_graph_cache, get_task_size_bytes, calc_timeout_rate, to_scalar,
    safe_rest_tasks_total, safe_action_to_int, diagnose_timeout,
    get_arrived_apps, all_arrived_done, subtask_outcome_stats, subtask_partition_stats,
)

print_lock = None  # 将在训练入口脚本中通过 set_print_lock() 设置

def set_print_lock(lock):
    """设置打印锁，用于多进程输出同步"""
    global print_lock
    print_lock = lock


def decode_action(action: int) -> tuple:
    """返回 (target_name, edge_id). edge_id=-1 表示非Edge"""
    if action == 0:
        return "Local", -1
    if action == 1:
        return "Cloud", -1
    return "Edge", action - 2


def ensure_trace_dir(run_dir: str) -> Path:
    """确保 trace 目录存在"""
    p = Path(run_dir) / "traces"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ================= Worker 函数 =================

def run_benchmark_worker(args):
    """
    运行对比算法（Local/Cloud/Edge/Greedy/Genetic/Random）的 Worker 函数

    Args:
        args: (algo_name, run_dir, seed_offset, para_dict, task_order_mode)

    Returns:
        (algo_name, energy, delay, time_taken, timeout_info)
    """
    # 修复接收 para_dict 并同步到子进程
    # 接收 task_order_mode 参数
    algo_name, run_dir, seed_offset, para_dict, task_order_mode = args

    # 【关键】用传入的字典覆盖当前进程的 para，确保参数一致性
    if para_dict is not None:
        para.update(para_dict)

    init_worker(seed_offset, para, CONFIG)
    # 修复从环境变量读取RUN_DIR和EVAL_MODE，确保Pool子进程能获取到
    if "RUN_DIR" in os.environ:
        CONFIG["RUN_DIR"] = os.environ["RUN_DIR"]
    if "EVAL_MODE" in os.environ:
        CONFIG["EVAL_MODE"] = os.environ["EVAL_MODE"] == "True"
    start_t = time.time()
    try:
        # 修复跳过到达计划缓存，强制重新生成以避免脏数据污染
        # plan_data = load_arrival_plan(run_dir, seed_offset) if run_dir else None
        plan_data = None  # 强制禁用到达计划缓存

        # 修复兼容所有 arrival_plan 格式（dict/list）
        if plan_data is None:
            print(f"[信息] 强制重新生成到达计划（跳过缓存以避免脏数据污染，seed_offset={seed_offset}）")
            gen = generate_arrival_plan(
                CONFIG["SEED"] + seed_offset,
                CONFIG["MAX_STEPS"],
                CONFIG["STOP_ARRIVAL_STEP"],
                base_prob=0.3,
                burst_prob=CONFIG["BURST_PROB"],
                burst_min=max(1, CONFIG["BURST_SIZE"]//2),
                burst_max=CONFIG["BURST_SIZE"]
            )
            # 兼容 generate_arrival_plan 返回 dict 或 list
            if isinstance(gen, dict):
                arrival_plan = gen.get("arrival_plan") or gen.get("plan") or []
                env_seed = int(gen.get("env_seed", CONFIG["SEED"] + seed_offset))
            else:
                arrival_plan = gen
                env_seed = CONFIG["SEED"] + seed_offset
        else:
            # 兼容 load_arrival_plan 返回 dict 或 list
            if isinstance(plan_data, list):
                arrival_plan = plan_data
                env_seed = CONFIG["SEED"] + seed_offset
            else:
                arrival_plan = plan_data.get("arrival_plan", [])
                env_seed = int(plan_data.get("env_seed", CONFIG["SEED"] + seed_offset))

            if env_seed is not None:
                print(f"[INFO] 已生成新的到达计划和环境种子（env_seed={env_seed}）")

        # 强制转成 list[int]，防止 sum/apply 出错
        arrival_plan = [int(x) for x in arrival_plan]

        user_num = para["user_num"]
        subgraph_num = 20
        basegraph_num = 60
        task_complex = para["task_complex"]

        # 修复同一轮对比，所有算法用同一个 task_complex_index
        # 【安全修复】兼容 task_complex 是整数或列表的情况
        if isinstance(task_complex, (list, tuple)):
            task_complex_index = (CONFIG["SEED"] + seed_offset) % len(task_complex)
        else:
            # 如果是整数或其他类型，转换为索引（默认为0）
            task_complex_index = int(task_complex) if isinstance(task_complex, int) else 0

        env = Environment(user_num=user_num, subgraph_num=subgraph_num,
                          basegraph_num=basegraph_num, task_complex_index=task_complex_index)
        env.generate_components(seed=env_seed)

        G = get_graph_cache(user_num, subgraph_num, basegraph_num, project_root)
        if G is not None and env.basegraph: env.basegraph.nx_graph = G

        # 修复加载 tight deadline 配置（确保环境一致性）
        deadline_config = load_deadline_config(run_dir)
        if deadline_config:
            print(f"[{algo_name}] 【环境一致性】加载预计算的 deadline 配置: "
                  f"{len(deadline_config['tight_user_ids'])} 个紧 deadline 用户")
        else:
            print(f"[{algo_name}] 【警告】未找到 deadline 配置，使用默认随机生成")

        ts = TaskScheduler(user_num, subgraph_num, basegraph_num, env, tight_deadline_config=deadline_config, seed=CONFIG["SEED"])
        gs = GraphScheduler(env.basegraph, env.subgraph_list, ts)
        # 添加 task_order_mode 参数
        bc = Benchmark.Benchmark(env, gs, ts, task_complex_index, effective=True, seed=CONFIG["SEED"], task_order_mode=task_order_mode)

        algo_map = {
            "Local": 0,
            "Cloud": 1,
            "Edge": 2,
            "Greedy": 4,
            "Genetic": 6,       # GeneticFair (type=6)
            "HybridPSOGA": 8    # 混合粒子群-遗传算法 (排队感知 + 噪声模型)
        }
        ts.env = env;
        ts.using_Algorithm = algo_map.get(algo_name, 0)
        bc.reset();
        ts.reset()

        # 【新增诊断】打印到达计划的总到达数
        total_arrivals = sum(arrival_plan) if arrival_plan else 0
        print(f"[{algo_name}] 【环境一致性】到达计划总到达数: {total_arrivals} (目标: {para['user_num']})")

        # 修复打印环境指纹：enter_time、deadline_slot、deadline_abs
        slot_interval = para["slot_interval"]
        print(f"[{algo_name}] 【环境指纹】enter_time[:10] = {[float(ts.enter_time[i]) if ts.enter_time[i] != float('inf') else 'inf' for i in range(10)]}")
        if hasattr(ts, 'app_deadline_slots'):
            deadline_slots = [ts.get_app_deadline_slot(i) for i in range(10)]
            print(f"[{algo_name}] 【环境指纹】deadline_slot[:10] = {deadline_slots}")
            # 计算 deadline_abs（假设前 10 个用户的 enter_time 都不是 inf）
            deadline_abs = []
            for i in range(10):
                if ts.enter_time[i] != float('inf'):
                    dead_abs = float(ts.enter_time[i]) + int(ts.get_app_deadline_slot(i)) * slot_interval
                    deadline_abs.append(f"{dead_abs:.3f}s")
                else:
                    deadline_abs.append("N/A")
            print(f"[{algo_name}] 【环境指纹】deadline_abs[:10] = {deadline_abs}")
        else:
            print(f"[{algo_name}] 【环境指纹】deadline_slot = N/A (未设置)")

        # 如果 arrival_plan 是 dict，打印 schedule 前几个 slot
        if isinstance(arrival_plan, dict) and "schedule" in arrival_plan:
            schedule = arrival_plan.get("schedule", [])
            print(f"[{algo_name}] 【环境指纹】arrival_schedule[:10] = {[f'slot{i}: {schedule[i]}' for i in range(min(10, len(schedule)))]}")

        # 【新增诊断】打印 ts.task_size 的前几个值
        s = ts.task_size
        vals = []
        if isinstance(s, (list, tuple, np.ndarray)) and len(s)>0 and isinstance(s[0], (list, tuple, np.ndarray)):
            for uid in range(min(3, len(s))):
                for sid in range(min(5, len(s[uid]))):
                    vals.append(float(s[uid][sid]))
        else:
            for sid in range(min(10, len(s))):
                vals.append(float(s[sid]))
        print(f"[{algo_name}] 【环境一致性】task_size 样本: {vals[:5]} (单位: {'Bytes' if max(vals) > 10000 else 'KB'})")
        print(f"[{algo_name}] 【环境一致性】task_size 范围: {min(vals):.2f} - {max(vals):.2f}")

        # 【新增诊断】初始化trace记录
        trace_timeseries = []
        trace_actions = []
        trace_decisions = []
        decision_idx = 0
        trace_dir = ensure_trace_dir(CONFIG["RUN_DIR"]) if CONFIG.get("RUN_DIR") else None

        # 动作选择统计（total_slots: slot总数, total_actions: 动作总数）
        action_stats = {"local": 0, "cloud": 0, "edge": 0, "total_slots": 0, "total_actions": 0}

        # 记录每个子任务最终被分配到哪里
        task2action = {}  # key: (uid, sid) -> action

        # 推理时间统计
        inference_times = []

        # 修复使用新的退出条件：所有已到达应用都完成
        for slot in range(CONFIG["MAX_STEPS"]):
            # 修复使用 apply_arrival_plan 应用预生成的到达计划
            apply_arrival_plan(ts, slot, arrival_plan)
            ts.check_timeouts(slot)

            # 检查退出条件
            if slot >= CONFIG["STOP_ARRIVAL_STEP"]:
                # 所有已到达应用都完成 + 没有剩余任务
                if all_arrived_done(ts) and safe_rest_tasks_total(ts.rest_tasks) == 0:
                    break

            # 修复始终尝试获取动作，即使所有任务完成
            inference_start = time.time()
            actions, _ = bc.get_actions(ts.using_Algorithm, slot, 0)
            inference_time_ms = (time.time() - inference_start) * 1000.0
            inference_times.append(inference_time_ms)

            # 【致命修复】必须执行 bc.step()，否则状态不推进导致全超时、能耗为0
            if actions:
                bc.step(actions)

            # 修复记录时序trace（队列状态）- 使用安全函数
            trace_timeseries.append({
                "slot": slot,
                "algo": algo_name,
                "rest_tasks": safe_rest_tasks_total(ts.rest_tasks),
                "waiting_apps": len(ts.application_waiting),
                "active_apps": len(ts.application_started - ts.application_finished)
            })

            # 修复记录动作trace - 使用安全函数转换 action
            # 修复无论是否有任务，都统计动作总数（确保公平对比）
            if len(actions) > 0:
                for entry in actions:
                    if len(entry) >= 2:
                        task, action_raw = entry[0], entry[1]

                        # 修复安全转换 action 为 int
                        action = safe_action_to_int(action_raw)

                        # 记录任务到动作的映射（用于子任务分桶统计）
                        task2action[tuple(task)] = int(action)

                        # 解码动作到目标
                        target, edge_id = decode_action(action)

                        trace_actions.append({
                            "slot": slot,
                            "algo": algo_name,
                            "task_id": str(task),
                            "action": action
                        })

                        # 记录决策顺序（task 分离为 uid, sid）
                        uid, sid = task[0], task[1]
                        trace_decisions.append({
                            "decision_idx": decision_idx,
                            "slot": slot,
                            "uid": int(uid),
                            "sid": int(sid),
                            "action": int(action),
                            "target": target,
                            "edge_id": int(edge_id)
                        })
                        decision_idx += 1

                        # 统计动作选择
                        if action == 0:
                            action_stats["local"] += 1
                        elif action == 1:
                            action_stats["cloud"] += 1
                        elif action >= 2:
                            action_stats["edge"] += 1

            # 修复无论是否有动作，都累加total_slots（确保所有算法的总slot数相同）
            action_stats["total_slots"] += 1
            action_stats["total_actions"] += len(actions)

        # 修复结束时调用 finalize_episode，确保超时统计准确
        ts.finalize_episode(slot)
        e, d = ts.get_avg_results()
        timeout_info = calc_timeout_rate(ts)  # 返回完整的 dict，包含 AppTO 和 TaskTO

        # 子任务分桶统计（Local/Cloud/Edge/Timeout）
        partition = subtask_partition_stats(ts, task2action)

        # 子任务级统计和动作选择统计
        total_subtasks, finished_subtasks, unfinished_subtasks, timeout_subtasks = subtask_outcome_stats(ts)
        timeout_info["subtask_stats"] = {
            "total": int(total_subtasks),
            "finished": int(finished_subtasks),
            "unfinished": int(unfinished_subtasks),
            "timeout": int(timeout_subtasks)  # 必须写进去，不然主程序 TaskTO 永远算不出来
        }
        # 使用分桶统计的 action_stats（保证 local+cloud+edge+timeout+unknown = total_subtasks）
        timeout_info["action_stats"] = {
            "local": partition["local"],
            "cloud": partition["cloud"],
            "edge": partition["edge"],
            "timeout": partition["timeout"],
            "unknown": partition["unknown"],
            "total": partition["total_subtasks"],  # 现在这个 total 就是"子任务总数"
            "total_slots": action_stats["total_slots"],
            "total_actions": action_stats["total_actions"]
        }

        # 推理时间统计
        if inference_times:
            timeout_info['inference_time_ms'] = float(np.mean(inference_times))
            timeout_info['inference_stats'] = {
                'median': float(np.median(inference_times)),
                'min': float(np.min(inference_times)),
                'max': float(np.max(inference_times)),
                'std': float(np.std(inference_times)),
                'count': len(inference_times)
            }
            print(f"[{algo_name}] 推理时间统计: 平均={timeout_info['inference_time_ms']:.4f}ms, "
                  f"中位数={timeout_info['inference_stats']['median']:.4f}ms, "
                  f"最小={timeout_info['inference_stats']['min']:.4f}ms, "
                  f"最大={timeout_info['inference_stats']['max']:.4f}ms, "
                  f"共{len(inference_times)}次决策")
        else:
            timeout_info['inference_time_ms'] = 0.0
            timeout_info['inference_stats'] = {
                'median': 0.0, 'min': 0.0, 'max': 0.0, 'std': 0.0, 'count': 0
            }

        # 【新增诊断】打印子任务完成统计（用于诊断环境/超时定义是否一致）
        total_subtasks, finished_subtasks, unfinished_subtasks, timeout_subtasks = subtask_outcome_stats(ts)
        print(f"[{algo_name}] 【环境一致性】子任务统计: total={total_subtasks}, finished={finished_subtasks}, unfinished={unfinished_subtasks}, timeout={timeout_subtasks}")

        # 【验证】确保 partition 统计正确
        sum_check = partition["local"] + partition["cloud"] + partition["edge"] + partition["timeout"] + partition["unknown"]
        check_mark = "OK" if sum_check == partition["total_subtasks"] else "NG"
        print(f"[{algo_name}] 【统计验证】Local={partition['local']} + Cloud={partition['cloud']} + Edge={partition['edge']} + "
              f"Timeout={partition['timeout']} + Unknown={partition['unknown']} = {sum_check} (Total={partition['total_subtasks']}) {check_mark}")

        # 保存trace文件（使用安全写法，防止KeyError）
        run_dir = CONFIG.get("RUN_DIR", "") or os.environ.get("RUN_DIR", "")
        if run_dir:
            trace_dir = ensure_trace_dir(run_dir)
            if trace_dir.exists():
                # 保存时序trace
                if trace_timeseries:
                    ts_df = pd.DataFrame(trace_timeseries)
                    ts_file = trace_dir / f"{algo_name}_timeseries.csv"
                    ts_df.to_csv(ts_file, index=False)

                # 保存动作trace
                if trace_actions:
                    act_df = pd.DataFrame(trace_actions)
                    act_file = trace_dir / f"{algo_name}_actions.csv"
                    act_df.to_csv(act_file, index=False)

                # 保存决策顺序trace
                if trace_decisions:
                    dec_df = pd.DataFrame(trace_decisions)
                    dec_file = trace_dir / f"{algo_name}_decisions.csv"
                    dec_df.to_csv(dec_file, index=False)

        # 获取总能耗
        try:
            total_e = ts.get_sum_energy()
        except Exception:
            total_e = to_scalar(e)  # 获取失败，使用平均能耗

        return algo_name, to_scalar(e), to_scalar(d), time.time() - start_t, {'timeout_rate': timeout_info, 'total_energy': total_e}
    except Exception as e:
        # 修复打印 traceback，便于调试
        import traceback
        print(f"[{algo_name}] [ERROR] Benchmark worker 发生异常:")
        traceback.print_exc()
        return algo_name, 1000.0, 1000.0, 0.0, {'timeout_rate': 1.0, 'app_timeout_rate': 1.0, 'task_timeout_rate': 1.0, 'error': str(e)}
