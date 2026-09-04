import sys
# -*- coding: utf-8 -*-
"""主对比实验运行器: CPGAPPO + 消融变体 + 外部对比算法 + 启发式基线
=========================================================================
【用途】在一个文件里统一跑:
  • CPGAPPO (前向 GAT + CP 加权信用) —— 主算法
  • 6 个消融变体 (noguidece / noshield / noappcredit / nocp / fwdonly / alloff)
  • 外部 RL 对比算法 (DGMA_adapt / DGMA_paper / TransEdgeStyle / GATDQN / PPO)
  • 启发式基线 (HybridPSOGA / Genetic / Greedy / Edge-only / Local-only)

【用法】
  1. 在下面 ALGOS 列表里注释掉不想跑的行 (在行首加 #)；
  2. 在下面 DAGS 列表里注释掉不想跑的 DAG 拓扑；
  3. 在下面 SEEDS 列表里改成想跑的种子；
  4. 在 PyCharm 里点 Run (或命令行 python run_main_comparison_3dag.py)。

  每个算法 × seed × DAG 会在子进程里独立训练/评估, 输出到
  results/main_comparison_{ts}/{dag}__{algo}__seed{seed}/result.json,
  汇总写到 results/main_comparison_{ts}/summary_{dag}.csv (及 .json)。

【设计要点】
  • CPGAPPO + 消融变体: 训练后基于 ckpt 做重评估 (D_all + UtilityScore),
    与 _run3_ablation_*_seed.py 完全一致, 结果可直接对比。
  • 外部 RL 对比: 直接用 wrapper 返回的 (e, d, m), 用 m 里的 AppTO/TaskTO 按
    主表口径算 UtilityScore (用 D_succ, 不做 reeval —— 这些算法没有
    _reeval_lib.VARIANT_CONFIG 条目, 也无统一 eval_fn)。
  • 启发式基线: 通过 Algorithms.Train.common.run_benchmark_worker 分发, 一次跑完
    到达计划, 无 ckpt, 记录 wrapper 返回的 (e, d, AppTO, TaskTO) + UtilityScore。
  • 所有算法统一: deadline_slot=55, BURST_PROB=0.2, BURST_SIZE=55, MAX_STEPS=8000,
    STOP_ARRIVAL_STEP=2000, user_num=150 (constant.py 默认), MATRIX_OVERRIDE_PATH
    按 DAG 切换。
"""
import os, sys, time, json, subprocess, traceback, threading
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根
PY = sys.executable  # 运行时取当前 python (原 conda env)

DEFAULT_OUT_BASE = os.path.join(PROJECT_ROOT, "results")

# ============================================================================
# 【手动设置 1】要跑哪些 DAG 拓扑？注释掉一行就不跑该 DAG
# ============================================================================
DAGS = [
    ("chain",   os.path.normpath(os.path.join(PROJECT_ROOT, "matrix", "matrix_60_chain.txt"))),
    ("default", os.path.normpath(os.path.join(PROJECT_ROOT, "matrix", "matrix_60.txt"))),
    ("wide",    os.path.normpath(os.path.join(PROJECT_ROOT, "matrix", "matrix_60_wide.txt"))),
]

# ============================================================================
# 【手动设置 2】要跑哪些种子？改成你想跑的种子列表
# ============================================================================
SEEDS = [1, 3, 5, 7, 42]

# ============================================================================
# 【手动设置 3】训练轮数
# ============================================================================
EPISODES = 100          # RL 算法训练轮数 (与 _run3_ablation_*_seed.py 一致)
GPU = 0                 # GPU ID

# ============================================================================
# 【手动设置 4】ALGOS 列表 —— 注释掉一行, 运行时就不跑该算法
# ----------------------------------------------------------------------------
# 元组格式: (label, kind, module, fn_name, mode, need_reeval)
#   label       : 显示名 / 输出目录名
#   kind        : "rl"    -> 训练 wrapper (调 fn(gpu_id, seed_offset, episodes=..))
#                 "bench" -> 启发式 (调 run_benchmark_worker((mode, run_dir, seed, para, "random")))
#   module      : import 路径 (bench kind 可填任意, 不会用到)
#   fn_name     : 函数名 (bench kind 可填任意, 不会用到)
#   mode        : rl kind 且是 train_baseline_wrapper 时 -> "SATA"/"FlexDO" 等;
#                 其它 rl kind -> "" (普通调用);
#                 bench kind -> algo_name (Local/Edge/Greedy/Genetic/HybridPSOGA)
#   need_reeval : True -> 训练后基于 ckpt 做重评估 (D_all + UtilityScore),
#                 仅对在 _reeval_lib.VARIANT_CONFIG 里有条目的 variant 生效
#                 (CPGAPPO / noguidece / noshield / noappcredit / nocp / fwdonly / alloff)
# ============================================================================
ALGOS = [
    # ===== 临时 filter: 只跑 HybridPSOGA (改版后), Genetic 用已有 raw_result_json 对照 =====
    ("HybridPSOGA", "bench", "Algorithms.Train.common", "run_benchmark_worker", "HybridPSOGA", False),
    # ===== 内部对比: CPGAPPO + 6 消融变体 (走 reeval) =====
    ("CPGAPPO",     "rl", "Algorithms.Train.train_cpgappo_unified", "train_cpgappo_dual_cpgappo",       "", True),
    ("noguidece",   "rl", "Algorithms.Train.train_cpgappo_unified", "train_cpgappo_dual_wo_guidece",   "", True),
    ("noshield",    "rl", "Algorithms.Train.train_cpgappo_unified", "train_cpgappo_dual_wo_shield",    "", True),
    ("noappcredit", "rl", "Algorithms.Train.train_cpgappo_unified", "train_cpgappo_dual_wo_appcredit", "", True),
    ("nocp",        "rl", "Algorithms.Train.train_cpgappo_unified", "train_cpgappo_dual_wo_cpseq",     "", True),
    ("fwdonly",     "rl", "Algorithms.Train.train_cpgappo_unified", "train_cpgappo_dual_forward_only", "", True),
    ("alloff",      "rl", "Algorithms.Train.train_cpgappo_unified", "train_cpgappo_dual_all_off",      "", True),
    # ===== 外部 RL 对比算法 (不走 reeval, 直接用 wrapper 返回值 + 主表口径 UtilityScore) =====
    ("DGMA_adapt",     "rl", "Algorithms.Train.train_dgma_adapt_wrapper",      "train_dgma_adapt_wrapper",      "", False),
    ("DGMA_paper",     "rl", "Algorithms.Train.train_dgma_paper_wrapper",      "train_dgma_paper_wrapper",      "", False),
    ("TransEdgeStyle", "rl", "Algorithms.Train.train_transedge_style_wrapper", "train_transedge_style_wrapper", "", False),
    ("GATDQN",         "rl", "Algorithms.Train.train_dynamic_gat_dqn",         "train_dynamic_gat_dqn_wrapper",  "", False),
    ("PPO",            "rl", "Algorithms.Train.train_ppo",                     "train_ppo_wrapper",             "", False),
    # ===== 启发式基线 (通过 run_benchmark_worker 分发, mode=algo_name) =====
    ("Genetic",     "bench", "Algorithms.Train.common", "run_benchmark_worker", "Genetic",     False),
    ("Greedy",      "bench", "Algorithms.Train.common", "run_benchmark_worker", "Greedy",      False),
    ("Edge-only",   "bench", "Algorithms.Train.common", "run_benchmark_worker", "Edge",        False),
    ("Local-only",  "bench", "Algorithms.Train.common", "run_benchmark_worker", "Local",       False),
]


def build_worker_script(run_dir: Path) -> Path:
    """子进程入口: 读环境变量跑训练/评估, dump result.json + latest.csv。
    支持 3 种 kind:
      • rl + need_reeval -> 训练 + reeval (D_all + UtilityScore)
      • rl + !need_reeval -> 训练, 直接用 wrapper 返回值 + 主表口径 UtilityScore
      • bench -> run_benchmark_worker 一次跑完, 用返回值 + 主表口径 UtilityScore
    """
    script = run_dir / "_worker.py"
    script.write_text(
        """# -*- coding: utf-8 -*-
import os, sys, json, time, traceback, importlib
PROJECT_ROOT = r\"""" + PROJECT_ROOT + """\"
sys.path.insert(0, PROJECT_ROOT)
# runners/ 也要在 sys.path, 否则 worker 子进程 import _reeval_lib / _dall_reeval_lib 找不到
_RUNNERS_DIR = os.path.join(PROJECT_ROOT, "runners")
if _RUNNERS_DIR not in sys.path:
    sys.path.insert(0, _RUNNERS_DIR)

# 读环境变量
DAG_PATH  = os.environ["ABL_MATRIX_PATH"]
DAG_NAME  = os.environ.get("ABL_DAG_NAME", "unknown")
ALGO_MOD  = os.environ.get("ABL_ALGO_MOD", "")
ALGO_FN   = os.environ.get("ABL_ALGO_FN", "")
VARIANT   = os.environ["ABL_VARIANT"]
KIND      = os.environ.get("ABL_KIND", "rl")
MODE      = os.environ.get("ABL_MODE", "")
SEED      = int(os.environ["ABL_SEED"])
EPS       = int(os.environ["ABL_EPISODES"])
GPU       = int(os.environ.get("ABL_GPU", "-1"))
RUN_DIR   = os.environ["ABL_RUN_DIR"]
NEED_REEVAL = os.environ.get("ABL_NEED_REEVAL", "0") == "1"

# 设置 MATRIX_OVERRIDE_PATH (关键! 确保使用正确的 matrix 文件)
os.environ["MATRIX_OVERRIDE_PATH"] = DAG_PATH

# 统一实验参数: deadline_slot=55, BURST_PROB=0.2, BURST_SIZE=55, MAX_STEPS=8000
from utils.constant import para as _para
_para["deadline_slot"] = 55
if os.environ.get("ABL_USER_NUM"):
    _para["user_num"] = int(os.environ["ABL_USER_NUM"])

from Experiments_new.exp_utils import CONFIG, compute_utility_score
CONFIG["RUN_DIR"] = RUN_DIR
CONFIG["BURST_PROB"] = 0.2
CONFIG["BURST_SIZE"] = 55
CONFIG["BURST_MODE"] = True
CONFIG["MAX_STEPS"] = 8000
CONFIG["STOP_ARRIVAL_STEP"] = 2000
os.environ["RUN_DIR"] = RUN_DIR

print(f"[worker] DAG={DAG_NAME}({DAG_PATH}) kind={KIND} variant={VARIANT} "
      f"mode={MODE} seed={SEED} eps={EPS} gpu={GPU} need_reeval={NEED_REEVAL}")
print(f"[worker] RUN_DIR={RUN_DIR}")
print(f"[worker] deadline_slot=55, BURST_PROB=0.2, BURST_SIZE=55, MAX_STEPS=8000")
sys.stdout.flush()


def _utility(e, d, rho_app, rho_task):
    \"\"\"主表口径 UtilityScore (越大越好). 与 _reeval_lib.compute_utility_score 一致.
    rho_app / rho_task 以【百分比】传入 (0~100)，compute_utility_score 内部 /100。\"\"\"
    try:
        return float(compute_utility_score(e, d, rho_app, rho_task, w_cost=0.25))
    except Exception as ex:
        print(f"[worker] compute_utility_score FAILED: {ex}")
        return None


t0 = time.time()
out = {"status": "unknown"}
try:
    if KIND == "bench":
        # ===== 启发式基线: run_benchmark_worker =====
        from Algorithms.Train.common import run_benchmark_worker
        current_para = _para.copy()
        bench_args = (MODE, RUN_DIR, SEED, current_para, "random")
        name, e, d, t_taken, metrics = run_benchmark_worker(bench_args)
        elapsed = time.time() - t0

        # 从 metrics 提取 AppTO / TaskTO (run_benchmark_worker 返回的 metrics 里有 timeout_rate dict)
        app_to = 0.0
        task_to = 0.0
        total_energy = float(e)
        if isinstance(metrics, dict):
            tr = metrics.get("timeout_rate", metrics)
            if isinstance(tr, dict):
                # calc_timeout_rate 返回小数 (0~1)，compute_utility_score 要百分比
                app_to = float(tr.get("app_timeout_rate", 0)) * 100.0
                task_to = float(tr.get("task_timeout_rate", app_to)) * 100.0
            else:
                app_to = float(tr) * 100.0
                task_to = app_to
            total_energy = float(metrics.get("total_energy", e))

        utility = _utility(e, d, app_to, task_to)
        out = {
            "status": "ok",
            "algorithm": MODE,
            "variant": VARIANT,
            "kind": "bench",
            "dag": DAG_NAME,
            "seed": SEED,
            "energy": float(e),
            "delay": float(d),
            "delay_succ": float(d),
            "app_timeout_rate": app_to,
            "task_timeout_rate": task_to,
            "score": utility if utility is not None else -1,
            "total_energy": total_energy,
            "UtilityScore_succ": utility,
            "UtilityScore_all": utility,  # bench 无 D_all, 用 D_succ 近似
            "runtime_sec": round(elapsed, 1),
            "episodes": EPS,
            "deadline_slot": 55,
            "matrix": DAG_PATH,
        }
        print(f"[worker][bench] {MODE}: E={e:.4f} D={d:.4f} AppTO={app_to:.2f}% "
              f"UtilityScore={utility}")
        sys.stdout.flush()

    else:
        # ===== RL 算法: 调训练 wrapper =====
        mod = importlib.import_module(ALGO_MOD)
        fn = getattr(mod, ALGO_FN)

        if MODE:
            # train_baseline_wrapper 走 mode 分发 (SATA/FlexDO 等)
            e, d, m = fn(gpu_id=(GPU if GPU >= 0 else 0), seed_offset=SEED,
                         mode=MODE, episodes=EPS, use_heuristic_sort=True)
        elif NEED_REEVAL:
            # 内部消融变体: 与 _run3_ablation_*_seed.py 一致, 不传 use_heuristic_sort
            # (让每个 variant 用自己的默认值, 保证结果可对比)
            e, d, m = fn(gpu_id=(GPU if GPU >= 0 else 0), seed_offset=SEED, episodes=EPS)
        else:
            # 外部 RL 对比: 传 use_heuristic_sort=True (与 dividelong_wide_main_comparison.py 一致)
            e, d, m = fn(gpu_id=(GPU if GPU >= 0 else 0), seed_offset=SEED,
                         episodes=EPS, use_heuristic_sort=True)

        elapsed = time.time() - t0
        # calc_timeout_rate 返回小数 (0~1)，compute_utility_score 要百分比
        app_to = float(m.get("app_timeout_rate", -1)) * 100.0
        task_to = float(m.get("task_timeout_rate", -1)) * 100.0
        total_energy = float(m.get("total_energy", -1))

        utility = _utility(e, d, app_to, task_to) if app_to >= 0 else None
        out = {
            "status": "ok",
            "algorithm": ALGO_FN,
            "variant": VARIANT,
            "kind": "rl",
            "dag": DAG_NAME,
            "seed": SEED,
            "energy": float(e),
            "delay": float(d),
            "delay_succ": float(d),
            "app_timeout_rate": app_to,
            "task_timeout_rate": task_to,
            "score": float(m.get("score", -1)),
            "total_energy": total_energy,
            "inference_time_ms": float(m.get("inference_time_ms", 0.0)),
            "UtilityScore_succ": utility,
            "UtilityScore_all": utility,  # 默认用 D_succ; reeval 会覆盖为 D_all 口径
            "runtime_sec": round(elapsed, 1),
            "runtime_min": round(elapsed / 60, 2),
            "episodes": EPS,
            "deadline_slot": 55,
            "matrix": DAG_PATH,
        }
        sys.stdout.flush()

        # ===== reeval: 仅对内部消融变体 (VARIANT_CONFIG 里有条目的) =====
        if NEED_REEVAL:
            try:
                import _reeval_lib as _rl
                if VARIANT not in _rl.VARIANT_CONFIG:
                    print(f"[worker][reeval] SKIP: variant '{VARIANT}' 不在 VARIANT_CONFIG, "
                          f"仅保留训练返回值")
                else:
                    variant_cfg = _rl.VARIANT_CONFIG[VARIANT]
                    ctx = _rl._build_env_for_reeval(DAG_PATH, RUN_DIR, SEED)
                    agent, ckpt_used, ckpt_path = _rl._load_ckpt_and_build_agent(
                        ctx, variant_cfg, os.path.join(RUN_DIR, "checkpoints"))
                    _eval_mod = importlib.import_module(variant_cfg["eval_module"])
                    _eval_fn = getattr(_eval_mod, variant_cfg["eval_fn_name"])
                    import torch
                    with torch.no_grad():
                        if VARIANT == "alloff":
                            e_succ, d_succ, sc_succ, to_info, te_succ = _eval_fn(
                                agent, ctx["ts"], ctx["gs"], ctx["bc"], ctx["arrival_plan"],
                                ctx["task_complex_index"], ctx["max_steps"],
                                use_shield=variant_cfg["use_shield"],
                                use_cp=variant_cfg["use_cp"],
                            )
                        else:
                            e_succ, d_succ, sc_succ, to_info, te_succ = _eval_fn(
                                agent, ctx["ts"], ctx["gs"], ctx["bc"], ctx["arrival_plan"],
                                ctx["task_complex_index"], ctx["max_steps"],
                            )
                    # 同一 ts 状态取 D_all (超时按 per-app deadline 预算计入)
                    e_all, d_all = ctx["ts"].get_avg_results(
                        only_successful=False, timeout_charge="deadline")

                    rho_app = float(to_info['app_timeout_rate']) * 100.0
                    rho_task = float(to_info.get('task_timeout_rate', 0)) * 100.0

                    utility_all = float(_rl.compute_utility_score(
                        e_all, d_all, rho_app, rho_task, w_cost=0.25))
                    utility_succ = float(_rl.compute_utility_score(
                        e_succ, d_succ, rho_app, rho_task, w_cost=0.25))

                    out.update({
                        "delay_succ": float(d_succ),
                        "delay_all": float(d_all),
                        "D_succ": float(d_succ),
                        "D_all": float(d_all),
                        "E_succ": float(e_succ),
                        "E_all": float(e_all),
                        "UtilityScore_succ": utility_succ,
                        "UtilityScore_all": utility_all,
                        "ckpt_used": ckpt_used,
                        "ckpt_path": ckpt_path,
                    })
                    print(f"[worker][reeval] D_succ={d_succ:.4f} D_all={d_all:.4f} "
                          f"AppTO={rho_app:.2f}% UtilityScore_all={utility_all:.4f} "
                          f"ckpt={ckpt_used}")
                    sys.stdout.flush()
            except Exception as ex_eval:
                print(f"[worker][reeval] FAILED (训练已完成, 主结果保留): {ex_eval}")
                traceback.print_exc()
                out["delay_all"] = None
                out["D_all"] = None
                out["r1_5_eval_error"] = str(ex_eval)
                sys.stdout.flush()

except Exception as ex:
    traceback.print_exc()
    out = {"status": "error", "error": str(ex),
           "variant": VARIANT, "dag": DAG_NAME, "seed": SEED,
           "runtime_sec": round(time.time() - t0, 1)}

with open(os.path.join(RUN_DIR, "result.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"[worker] result.json -> {RUN_DIR}/result.json")

try:
    import csv
    _latest_path = os.path.join(RUN_DIR, "latest.csv")
    _row = {
        "algo": VARIANT,
        "kind": KIND,
        "dag": DAG_NAME,
        "seed": SEED,
        "episodes": EPS,
        "status": out.get("status", "unknown"),
        "energy": out.get("energy", -1),
        "delay": out.get("delay", -1),
        "delay_succ": out.get("delay_succ", -1),
        "delay_all": out.get("delay_all", ""),
        "D_succ": out.get("D_succ", ""),
        "D_all": out.get("D_all", ""),
        "score": out.get("score", -1),
        "app_timeout_rate": out.get("app_timeout_rate", -1),
        "task_timeout_rate": out.get("task_timeout_rate", -1),
        "total_energy": out.get("total_energy", -1),
        "inference_time_ms": out.get("inference_time_ms", 0.0),
        "UtilityScore_succ": out.get("UtilityScore_succ", ""),
        "UtilityScore_all": out.get("UtilityScore_all", ""),
        "ckpt_used": out.get("ckpt_used", ""),
        "runtime_sec": out.get("runtime_sec", -1),
    }
    with open(_latest_path, "w", encoding="utf-8", newline="") as _lf:
        _w = csv.DictWriter(_lf, fieldnames=list(_row.keys()))
        _w.writeheader()
        _w.writerow(_row)
    print(f"[worker] latest.csv -> {_latest_path}")
except Exception as _le:
    print(f"[worker] write latest.csv FAILED: {_le}")
print(f"[worker] DONE {out.get('status', 'unknown')} in {out.get('runtime_sec', -1)}s")
sys.stdout.flush()
""",
        encoding="utf-8",
    )
    return script


def run_one(algo_label: str, kind: str, algo_mod: str, algo_fn: str,
            mode: str, need_reeval: bool,
            dag_name: str, dag_path: str,
            seed: int, gpu: int, out_root: Path) -> dict:
    run_dir = out_root / f"{dag_name}__{algo_label}__seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    worker = build_worker_script(run_dir)
    log_out = run_dir / "stdout.log"
    log_err = run_dir / "stderr.log"

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PUSHPLUS_DISABLE"] = "1"
    env["ABL_MATRIX_PATH"] = dag_path
    env["ABL_DAG_NAME"] = dag_name
    env["ABL_KIND"] = kind
    env["ABL_ALGO_MOD"] = algo_mod
    env["ABL_ALGO_FN"] = algo_fn
    env["ABL_VARIANT"] = algo_label
    env["ABL_MODE"] = mode
    env["ABL_NEED_REEVAL"] = "1" if need_reeval else "0"
    env["ABL_SEED"] = str(seed)
    env["ABL_EPISODES"] = str(EPISODES)
    env["ABL_GPU"] = str(gpu)
    env["ABL_RUN_DIR"] = str(run_dir)
    if os.environ.get("ABL_USER_NUM"):
        env["ABL_USER_NUM"] = os.environ["ABL_USER_NUM"]

    t0 = time.time()
    label_prefix = f"[{dag_name}/{algo_label}/seed{seed}] "
    with open(log_out, "w", encoding="utf-8", buffering=1) as fo, \
         open(log_err, "w", encoding="utf-8", buffering=1) as fe:
        proc = subprocess.Popen(
            [PY, "-u", str(worker)],
            cwd=PROJECT_ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
            bufsize=1,
        )

        def _pump(stream, file_handle, sink):
            for line in iter(stream.readline, ''):
                if not line:
                    break
                file_handle.write(line)
                file_handle.flush()
                try:
                    sink.write(label_prefix + line)
                    sink.flush()
                except Exception:
                    pass
            stream.close()

        t_out = threading.Thread(target=_pump, args=(proc.stdout, fo, sys.stdout), daemon=True)
        t_err = threading.Thread(target=_pump, args=(proc.stderr, fe, sys.stderr), daemon=True)
        t_out.start()
        t_err.start()
        rc = proc.wait()
        t_out.join(timeout=2)
        t_err.join(timeout=2)
    elapsed = time.time() - t0

    res_path = run_dir / "result.json"
    if res_path.exists():
        try:
            res = json.loads(res_path.read_text(encoding="utf-8"))
        except Exception as ex:
            res = {"status": "parse_error", "error": str(ex)}
    else:
        res = {"status": "no_result_file", "returncode": rc}
    res["dag"] = dag_name
    res["algo"] = algo_label
    res["kind"] = kind
    res["seed"] = seed
    res["wall_sec"] = round(elapsed, 1)
    res["wall_min"] = round(elapsed / 60, 2)
    res["run_dir"] = str(run_dir)
    return res


def _write_csv(path: Path, results: list) -> None:
    cols = ["dag", "algo", "kind", "seed", "status",
            "app_timeout_rate", "task_timeout_rate", "score",
            "energy", "delay", "delay_succ", "delay_all",
            "D_succ", "D_all",
            "UtilityScore_succ", "UtilityScore_all",
            "ckpt_used", "total_energy", "inference_time_ms",
            "runtime_sec", "wall_sec", "run_dir"]
    lines = [",".join(cols)]
    for r in results:
        lines.append(",".join(str(r.get(c, "")) for c in cols))
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(DEFAULT_OUT_BASE) / f"main_comparison_{ts_tag}"
    out_root.mkdir(parents=True, exist_ok=True)

    # 展平任务列表: (dag, algo, seed)
    tasks = []
    for dag_name, dag_path in DAGS:
        for algo_label, kind, algo_mod, algo_fn, mode, need_reeval in ALGOS:
            for seed in SEEDS:
                tasks.append((dag_name, dag_path,
                              algo_label, kind, algo_mod, algo_fn, mode, need_reeval,
                              seed))
    total = len(tasks)

    print(f"\n{'='*80}")
    print(f"主对比实验: CPGAPPO + 消融变体 + 外部对比算法 + 启发式基线")
    print(f"{'='*80}")
    print(f"输出目录:     {out_root}  (新时间戳目录, 不覆盖原批次)")
    print(f"DAG 拓扑数:   {len(DAGS)}  {[d[0] for d in DAGS]}")
    print(f"算法数:       {len(ALGOS)}  {[a[0] for a in ALGOS]}")
    print(f"种子数:       {len(SEEDS)}  {SEEDS}")
    print(f"总实验数:     {total}")
    print(f"每 RL 实验:   {EPISODES} episodes")
    print(f"GPU:          {GPU}")
    print(f"{'='*80}\n")

    results_all = []  # 所有 DAG 合并
    t_global = time.time()

    for idx, (dag_name, dag_path,
              algo_label, kind, algo_mod, algo_fn, mode, need_reeval,
              seed) in enumerate(tasks, 1):
        print(f"\n{'='*70}")
        print(f"[{idx}/{total}] DAG={dag_name}  algo={algo_label}  kind={kind}  "
              f"seed={seed}  episodes={EPISODES}")
        print(f"{'='*70}")
        sys.stdout.flush()
        res = run_one(algo_label, kind, algo_mod, algo_fn, mode, need_reeval,
                      dag_name, dag_path, seed, GPU, out_root)
        results_all.append(res)

        # 每个 DAG 维护独立 summary 文件 (增量写, 防中途崩溃丢数据)
        dag_results = [r for r in results_all if r.get("dag") == dag_name]
        summary_json_path = out_root / f"summary_{dag_name}.json"
        summary_csv_path = out_root / f"summary_{dag_name}.csv"
        summary_json_path.write_text(
            json.dumps(dag_results, indent=2, ensure_ascii=False), encoding="utf-8")
        _write_csv(summary_csv_path, dag_results)

        print(f"\n[{idx}/{total}] 完成!")
        print(f"  status:        {res.get('status')}")
        print(f"  AppTO:         {res.get('app_timeout_rate', '-')}")
        print(f"  TaskTO:        {res.get('task_timeout_rate', '-')}")
        print(f"  Score:         {res.get('score', '-')}")
        print(f"  UtilityScore:  {res.get('UtilityScore_all', '-')}")
        print(f"  Energy:        {res.get('energy', '-')}")
        print(f"  Delay:         {res.get('delay', '-')}")
        if res.get("D_all") is not None:
            print(f"  D_all:  {res.get('D_all', '-')}")
        print(f"  wall时间:      {res.get('wall_min', '-')} 分钟")
        print(f"  结果目录:      {res.get('run_dir', '-')}")
        sys.stdout.flush()

    elapsed = time.time() - t_global
    print(f"\n{'='*80}")
    print(f"[ALL DONE] {total} runs in {elapsed/60:.1f} 分钟")
    print(f"{'='*80}")

    # 最终汇总: 按 DAG 打印结果表
    for dag_name, _ in DAGS:
        dag_results = [r for r in results_all if r.get("dag") == dag_name]
        if not dag_results:
            continue
        print(f"\n{'='*80}")
        print(f"最终结果汇总: DAG = {dag_name}")
        print(f"{'='*80}")
        print(f"{'算法':<18} {'seed':<6} {'AppTO':<10} {'TaskTO':<10} "
              f"{'Score':<12} {'UtilityScore':<14} {'Energy':<10} {'Delay':<10} "
              f"{'D_all':<10}")
        print(f"{'-'*100}")
        # 按 ALGOS 顺序输出
        algo_order = [a[0] for a in ALGOS]
        dag_results_sorted = sorted(dag_results,
                                    key=lambda r: (algo_order.index(r.get('algo', '')) if r.get('algo', '') in algo_order else 999,
                                                   r.get('seed', 0)))
        for r in dag_results_sorted:
            algo = str(r.get('algo', ''))[:17]
            seed = str(r.get('seed', ''))
            def _fmt(v, pct=False):
                if v is None or v == '' or v == '-':
                    return '-'
                if isinstance(v, (int, float)):
                    # app_timeout_rate 现在存的是百分比 (0~100)，直接显示
                    return f"{v:.4f}" if not pct else f"{v:.2f}%"
                return str(v)
            appto = _fmt(r.get('app_timeout_rate'), pct=True)
            taskto = _fmt(r.get('task_timeout_rate'), pct=True)
            score = _fmt(r.get('score'))
            utility = _fmt(r.get('UtilityScore_all'))
            energy = _fmt(r.get('energy'))
            delay = _fmt(r.get('delay'))
            d_all = _fmt(r.get('D_all'))
            print(f"{algo:<18} {seed:<6} {appto:<10} {taskto:<10} "
                  f"{score:<12} {utility:<14} {energy:<10} {delay:<10} "
                  f"{d_all:<10}")
        print(f"{'='*80}")
        print(f"汇总JSON: {out_root / f'summary_{dag_name}.json'}")
        print(f"汇总CSV:  {out_root / f'summary_{dag_name}.csv'}")

    # 全局汇总 (所有 DAG 合并)
    global_summary_json = out_root / "summary_all.json"
    global_summary_csv = out_root / "summary_all.csv"
    global_summary_json.write_text(
        json.dumps(results_all, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(global_summary_csv, results_all)
    print(f"\n全局汇总JSON: {global_summary_json}")
    print(f"全局汇总CSV:  {global_summary_csv}")


if __name__ == "__main__":
    main()
