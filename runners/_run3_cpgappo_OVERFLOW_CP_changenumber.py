import sys

# -*- coding: utf-8 -*-

"""CPGAPPO 敏感性扫描 runner（可任意调参, 不影响原算法文件）

=========================================================================

对 CPGAPPO 做超参敏感性扫描。主脚本顶部配置区可任意改要扫的参数、

  取值、seeds、DAG; 运行时通过环境变量把 override 值传给 worker, worker 在 import

  train 模块后用 monkey-patch 临时覆盖 train_cpgappo / agent_cpgappo / cpgappo_core 里的

  硬编码常量, 不修改任何原算法文件 (原 train_cpgappo 在没传 override 时仍走自己的默认值)。



【输出】

  results/{out_tag}_{ts}/{dag}__{algo}__p{param}__v{value}__seed{N}/result.json

  results/{out_tag}_{ts}/summary_sweep.csv  (含 param/value/seed/各指标, 一张大表)

  results/{out_tag}_{ts}/summary_sweep.json



【能扫的参数 (见 SWEEP_PARAMS)】

  训练超参:

    LAMBDA_GUIDE      Guide-CE 权重 λ_g               默认 0.1   (R1-C6 点名)

    ENTROPY_COEF      PPO entropy_coef                默认 0.02

    LR                学习率                           默认 3e-4

  Reward:

    BONUS_FIN         app success bonus               默认 +4.0  (R1-C6 "dominant reward")

    BONUS_TO          app timeout  penalty            默认 -7.0  (R1-C6 "dominant reward")

    TERMINAL_COEF     terminal appTO 系数             默认 5.0

    W_E               step reward 能量权重 w_E        默认 0.05  (R1-C6 "dominant reward")

    W_D               step reward 时延权重 w_D        默认 0.1   (R1-C6 "dominant reward")

  Shield:

    SHIELD_THR_HIGH   高压力阈值 θ_high               默认 1.10  (R1-C6 点名)

    SHIELD_THR_NORM   正常阈值   θ_norm               默认 1.15  (R1-C6 点名)

  Guide score 常数 (R4-C1 点名):

    CP_REM_COEFF      C?_rem 系数 c_rem                默认 0.45

    OVERFLOW_MULT     下游溢出乘数 k_Δ                默认 1.25



【用法】改 SWEEP_PARAMS / SEEDS / DAGS / EPISODES 后点 Run 即可。

  ? 单参数扫描: SWEEP_PARAMS = [("LAMBDA_GUIDE", [0, 0.05, 0.1, 0.2, 0.4])]

  ? 多参数扫描: 列表里多放几个 tuple, 会按"参数分组"顺序跑 (不做全组合, 每组内逐值扫,

                其余参数保持默认). 这样 OFAT 单因子扫描, 图最好画。

  ? 想做二维网格 (如 w_E × w_D): 用 GRID_PARAMS (见下方说明)。



【顺序】参数组顺序 → 组内取值顺序 → DAG 顺序 → seed 顺序。

【注意】本脚本只创建代码, 不会自动启动; 你自己点 Run。

"""

import os, sys, time, json, subprocess, traceback, threading, itertools

from pathlib import Path

from datetime import datetime



PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根

PY = sys.executable  # 运行时取当前 python (原 conda env)



# ===========================================================================

#  配置区

# ===========================================================================



# --- A. 种子 ---

# R4-C1 点名 OVERFLOW_MULT (1.25) / CP_REM_COEFF (0.45), 建议补做敏感性

# 与已完成的 BONUS_TO / W_E 保持一致: 5 个种子

SEEDS = [1, 3, 5, 7, 42]



# --- B. DAG (注释掉不想跑的) ---

DAGS = [

    ("default", os.path.normpath(os.path.join(PROJECT_ROOT, "matrix", "matrix_60.txt"))),

    # ("chain",   os.path.normpath(os.path.join(PROJECT_ROOT, "matrix", "matrix_60_chain.txt"))),

    # ("wide",    os.path.normpath(os.path.join(PROJECT_ROOT, "matrix", "matrix_60_wide.txt"))),

]



# --- C. 训练设置 ---

EPISODES = 100

GPU      = 0

ALGO_LABEL = "CPGAPPO"

ALGO_MOD   = "Algorithms.Train.train_cpgappo_unified"

ALGO_FN    = "train_cpgappo_dual_cpgappo"

VARIANT    = "CPGAPPO"   # _reeval_lib.VARIANT_CONFIG key (reeval 用)



# --- D. 实验设置 (与主实验一致) ---

DEADLINE_SLOT     = 55

BURST_PROB        = 0.2

BURST_SIZE        = 55

MAX_STEPS         = 8000

STOP_ARRIVAL_STEP = 2000

USER_NUM          = None

USE_HEURISTIC_SORT = True



# --- E. 输出 ---

DEFAULT_OUT_BASE = os.path.join(PROJECT_ROOT, "results")

EXP_TAG          = "sens_OVERFLOW_CP"   # 输出目录 CPGAPPO_sens_OVERFLOW_CP_{ts}



# ===========================================================================

#  【要扫的参数】  OFAT 单因子扫描 (每组内逐值扫, 其余保持默认)

#  格式: (PARAM_KEY, [取值列表])

#  PARAM_KEY 必须是上面【能扫的参数】里列出的名字

# ===========================================================================

SWEEP_PARAMS = [

    # 只扫 OVERFLOW_MULT / CP_REM_COEFF (R4-C1 点名的 guide score 常数)

    ("OVERFLOW_MULT", [1.0, 1.1, 1.25, 1.5, 2.0]),

    ("CP_REM_COEFF",  [0.2, 0.3, 0.45, 0.6, 0.8]),

]



# ===========================================================================

#  【可选: 二维网格】  想做 w_E × w_D 这种小网格时用; 留空 [] 则跳过

#  格式: (PARAM_KEY_A, [values_A], PARAM_KEY_B, [values_B])

#  会跑全组合 (len_A * len_B * seeds * dags), 注意算力

# ===========================================================================

GRID_PARAMS = []   # 例: ("W_E", [0.025, 0.05, 0.1], "W_D", [0.05, 0.1, 0.2])



# ===========================================================================

#  【配置区结束】

# ===========================================================================



# 默认值表 (用于: (1) 跳过等于默认的取值; (2) 输出记录里标注是否为 default)

PARAM_DEFAULTS = {

    "LAMBDA_GUIDE":   0.1,

    "ENTROPY_COEF":   0.02,

    "LR":             3e-4,

    "BONUS_FIN":      4.0,

    "BONUS_TO":       -7.0,

    "TERMINAL_COEF":  5.0,

    "W_E":            0.05,

    "W_D":            0.1,

    "SHIELD_THR_HIGH": 1.10,

    "SHIELD_THR_NORM": 1.15,

    "CP_REM_COEFF":   0.45,

    "OVERFLOW_MULT":  1.25,

}





def _fmt_val(v):

    """把数值格式化成文件名友好的字符串 (0.05→0p05, -7.0→-7, 1e-4→1e-04)."""

    if isinstance(v, float) and v.is_integer():

        return str(int(v))

    s = f"{v:.4g}".replace(".", "p").replace("-", "m").replace("+", "")

    return s





def build_task_list():

    """构造 (param_key, value, dag_name, dag_path, seed) 任务列表."""

    tasks = []

    # OFAT 单因子

    for param_key, values in SWEEP_PARAMS:

        for val in values:

            for dag_name, dag_path in DAGS:

                for seed in SEEDS:

                    tasks.append((param_key, val, dag_name, dag_path, seed))

    # 二维网格

    for grid in GRID_PARAMS:

        pa, va_list, pb, vb_list = grid

        for va in va_list:

            for vb in vb_list:

                for dag_name, dag_path in DAGS:

                    for seed in SEEDS:

                        # 网格任务 param_key 记成 "W_E×W_D", value 记成 "0.05x0.1"

                        tasks.append((f"{pa}__x__{pb}", f"{va}__x__{vb}", dag_name, dag_path, seed,

                                      {pa: va, pb: vb}))

    return tasks





WORKER_TEMPLATE = r'''# -*- coding: utf-8 -*-

"""CPGAPPO sensitivity worker (auto-generated): train + reeval with param override."""

import os, sys, json, time, traceback, importlib, inspect

PROJECT_ROOT = r"__PROJECT_ROOT__"

if PROJECT_ROOT not in sys.path:

    sys.path.insert(0, PROJECT_ROOT)

# runners/ 也要在 sys.path, 否则 worker 子进程 import _reeval_lib 找不到

_RUNNERS_DIR = os.path.join(PROJECT_ROOT, "runners")

if _RUNNERS_DIR not in sys.path:

    sys.path.insert(0, _RUNNERS_DIR)



# ---- 读环境变量 ----

DAG_PATH   = os.environ["ABL_MATRIX_PATH"]

DAG_NAME   = os.environ.get("ABL_DAG_NAME", "unknown")

ALGO_MOD   = os.environ["ABL_ALGO_MOD"]

ALGO_FN    = os.environ["ABL_ALGO_FN"]

VARIANT    = os.environ.get("ABL_VARIANT", "")

SEED       = int(os.environ["ABL_SEED"])

EPS        = int(os.environ["ABL_EPISODES"])

GPU        = int(os.environ.get("ABL_GPU", "-1"))

RUN_DIR    = os.environ["ABL_RUN_DIR"]

PARAM_KEY  = os.environ.get("ABL_PARAM_KEY", "")

PARAM_VAL  = os.environ.get("ABL_PARAM_VAL", "")

OVERRIDE_JSON = os.environ.get("ABL_OVERRIDE", "{}")  # JSON dict: {param_key: value, ...}

LR                = float(os.environ.get("ABL_LR", "3e-4"))

ENTROPY_COEF      = float(os.environ.get("ABL_ENTROPY_COEF", "0.02"))

LAMBDA_GUIDE      = float(os.environ.get("ABL_LAMBDA_GUIDE", "0.1"))

USE_HEURISTIC_SORT = os.environ.get("ABL_USE_HEURISTIC_SORT", "1") == "1"

DEADLINE_SLOT     = int(os.environ.get("ABL_DEADLINE_SLOT", "55"))

BURST_PROB        = float(os.environ.get("ABL_BURST_PROB", "0.2"))

BURST_SIZE        = int(os.environ.get("ABL_BURST_SIZE", "55"))

MAX_STEPS         = int(os.environ.get("ABL_MAX_STEPS", "8000"))

STOP_ARRIVAL_STEP = int(os.environ.get("ABL_STOP_ARRIVAL_STEP", "2000"))

USER_NUM          = os.environ.get("ABL_USER_NUM", "")



try:

    OVERRIDE = json.loads(OVERRIDE_JSON) if OVERRIDE_JSON else {}

except Exception:

    OVERRIDE = {}



print(f"[worker] DAG={DAG_NAME}({DAG_PATH})")

print(f"[worker] algo={ALGO_MOD}.{ALGO_FN} variant={VARIANT} seed={SEED} eps={EPS} gpu={GPU}")

print(f"[worker] PARAM_KEY={PARAM_KEY} PARAM_VAL={PARAM_VAL}")

print(f"[worker] OVERRIDE={OVERRIDE}")

print(f"[worker] lr={LR} entropy={ENTROPY_COEF} lambda_guide={LAMBDA_GUIDE} heuristic_sort={USE_HEURISTIC_SORT}")

print(f"[worker] deadline_slot={DEADLINE_SLOT} burst_prob={BURST_PROB} burst_size={BURST_SIZE} max_steps={MAX_STEPS} stop_arrival={STOP_ARRIVAL_STEP} user_num={USER_NUM or '(default)'}")

print(f"[worker] RUN_DIR={RUN_DIR}")

sys.stdout.flush()



# ---- 设置全局环境 ----

os.environ["MATRIX_OVERRIDE_PATH"] = DAG_PATH

os.environ["RUN_DIR"] = RUN_DIR



from utils.constant import para as _para

_para["deadline_slot"] = DEADLINE_SLOT

if USER_NUM:

    _para["user_num"] = int(USER_NUM)



from Experiments_new.exp_utils import CONFIG

CONFIG["RUN_DIR"] = RUN_DIR

CONFIG["BURST_PROB"] = BURST_PROB

CONFIG["BURST_SIZE"] = BURST_SIZE

CONFIG["BURST_MODE"] = True

CONFIG["MAX_STEPS"] = MAX_STEPS

CONFIG["STOP_ARRIVAL_STEP"] = STOP_ARRIVAL_STEP

CONFIG["SEED"] = 0



# ============================================================

#  【关键】monkey-patch: 用 OVERRIDE 覆盖训练模块里的硬编码常量

#  只覆盖 OVERRIDE 里出现的 key, 不碰其余; 原文件一行不改

# ============================================================

def _apply_overrides():

    """根据 OVERRIDE dict, patch train_cpgappo / agent_cpgappo / cpgappo_core 里的常量."""

    if not OVERRIDE:

        return

    import Algorithms.Train.train_cpgappo_unified as _train_mod

    import Algorithms.RealGATPPO.agent_cpgappo as _agent_mod

    import Algorithms.RealGATPPO.cpgappo_core as _core



    # --- train_cpgappo 模块级常量 ---

    if "BONUS_FIN" in OVERRIDE:       _train_mod.BONUS_FIN = float(OVERRIDE["BONUS_FIN"])

    if "BONUS_TO" in OVERRIDE:        _train_mod.BONUS_TO  = float(OVERRIDE["BONUS_TO"])

    if "TERMINAL_COEF" in OVERRIDE:   _train_mod.TERMINAL_COEF = float(OVERRIDE["TERMINAL_COEF"])

    if "SHIELD_THR_HIGH" in OVERRIDE: _train_mod.SHIELD_THR_HIGH = float(OVERRIDE["SHIELD_THR_HIGH"])

    if "SHIELD_THR_NORM" in OVERRIDE: _train_mod.SHIELD_THR_NORM = float(OVERRIDE["SHIELD_THR_NORM"])



    # --- cpgappo_core.compute_cpgappo_slack_reward 里的 w_E / w_D (硬编码 0.05 / 0.1) ---

    # 用闭包替换整个函数, 把 w_E/w_D 从 OVERRIDE 读

    if "W_E" in OVERRIDE or "W_D" in OVERRIDE:

        _wE = float(OVERRIDE.get("W_E", 0.05))

        _wD = float(OVERRIDE.get("W_D", 0.1))

        _orig_slack = _core.compute_cpgappo_slack_reward

        def _patched_slack(ts, uid, sid, energy, delay, enter_time, deadline_slot,

                           _wE=_wE, _wD=_wD, _orig=_orig_slack):

            # 复刻原逻辑, 只把 0.05/0.1 换成 _wE/_wD

            TF = ts.finish_time[uid][sid]

            app_deadline_abs = enter_time + deadline_slot * _para["slot_interval"]

            slack = app_deadline_abs - TF

            r_step = - (_wE * energy) - (_wD * delay)

            if slack < 0:

                r_step -= (5.0 + abs(slack) * 2.0)

            elif slack < 0.2 * (app_deadline_abs - enter_time):

                r_step -= 1.0

            return r_step, float(slack)

        _core.compute_cpgappo_slack_reward = _patched_slack

        # train_cpgappo 是 from ... import, 所以也要替换 train_cpgappo 命名空间里的引用

        _train_mod.compute_cpgappo_slack_reward = _patched_slack

        print(f"[worker][patch] cpgappo_core.compute_cpgappo_slack_reward → w_E={_wE}, w_D={_wD}")



    # --- agent_cpgappo.compute_guide_scores_cp 里的 0.45 / 1.25 (硬编码) ---

    # 原函数把这俩常量直接写死在函数体里, 无法用参数注入. 这里用 inspect 拿原函数源码,

    # 把 "* 0.45" 替换为 "* c_rem"、把 "1.25 * downstream_overflow" 替换为

    # "k_delta * downstream_overflow", 然后 exec 重建一个同名函数. 只替换这两处,

    # 其余逻辑 (CP 深度计算 / valid_mask / immediate_cost 等) 完全不动.

    if "CP_REM_COEFF" in OVERRIDE or "OVERFLOW_MULT" in OVERRIDE:

        _c_rem = float(OVERRIDE.get("CP_REM_COEFF", 0.45))

        _k_delta = float(OVERRIDE.get("OVERFLOW_MULT", 1.25))

        _orig_guide = _agent_mod.compute_guide_scores_cp

        _new_guide = _rebuild_guide_fn(_orig_guide, _c_rem, _k_delta)

        _agent_mod.compute_guide_scores_cp = _new_guide

        # train_cpgappo 是 from ... import compute_guide_scores_cp, 同步替换其命名空间引用

        _train_mod.compute_guide_scores_cp = _new_guide

        print(f"[worker][patch] agent_v3.compute_guide_scores_cp → c_rem={_c_rem}, k_delta={_k_delta}")





def _rebuild_guide_fn(orig_fn, c_rem, k_delta):

    """从原 compute_guide_scores_cp 源码重建一个把 0.45→c_rem, 1.25→k_delta 的版本.

    只做文本替换: "* 0.45" → "* {c_rem}", "1.25 * downstream_overflow" → "{k_delta} * downstream_overflow".

    其余源码不动, 保留原函数的闭包/全局依赖.

    """

    import inspect, textwrap

    src = textwrap.dedent(inspect.getsource(orig_fn))

    src_new = src.replace("* 0.45", f"* {c_rem}").replace("1.25 * downstream_overflow", f"{k_delta} * downstream_overflow")

    lines = src_new.splitlines()

    def_idx = next(i for i, ln in enumerate(lines) if ln.strip().startswith("def "))

    func_src = "\n".join(lines[def_idx:])

    ns = dict(orig_fn.__globals__)

    ns["__builtins__"] = __builtins__

    exec(func_src, ns)

    return ns[orig_fn.__name__]



_apply_overrides()

# 回显本次 sweep 生效的 train() 关键超参

print(f"[worker][eff-train-params] lr={LR} entropy_coef={ENTROPY_COEF} lambda_guide={LAMBDA_GUIDE} "

      f"(sweep_param={PARAM_KEY} sweep_val={PARAM_VAL})")

print("[worker][patch] overrides applied")

sys.stdout.flush()



# ---- 调 train 函数 ----

mod = importlib.import_module(ALGO_MOD)

fn = getattr(mod, ALGO_FN)



t0 = time.time()

out = {"status": "unknown"}

try:

    _kwargs = dict(

        gpu_id=(GPU if GPU >= 0 else 0),

        seed_offset=SEED,

        use_heuristic_sort=USE_HEURISTIC_SORT,

        episodes=EPS,

    )

    _sig = set(inspect.signature(fn).parameters.keys())

    if "lr" in _sig:             _kwargs["lr"] = LR

    if "entropy_coef" in _sig:   _kwargs["entropy_coef"] = ENTROPY_COEF

    if "lambda_guide" in _sig:   _kwargs["lambda_guide"] = LAMBDA_GUIDE

    e, d, m = fn(**_kwargs)

    elapsed = time.time() - t0

    out = {

        "status": "ok",

        "algorithm": ALGO_FN,

        "variant": VARIANT,

        "dag": DAG_NAME,

        "seed": SEED,

        "param_key": PARAM_KEY,

        "param_val_raw": PARAM_VAL,

        "override": OVERRIDE,

        "energy": float(e),

        "delay": float(d),

        "delay_succ": float(d),

        "app_timeout_rate": float(m.get("app_timeout_rate", -1)),

        "task_timeout_rate": float(m.get("task_timeout_rate", -1)),

        "score": float(m.get("score", -1)),

        "total_energy": float(m.get("total_energy", -1)),

        "inference_time_ms": float(m.get("inference_time_ms", 0.0)),

        "runtime_sec": round(elapsed, 1),

        "runtime_min": round(elapsed / 60, 2),

        "episodes": EPS,

        "deadline_slot": DEADLINE_SLOT,

        "lr": LR,

        "entropy_coef": ENTROPY_COEF,

        "lambda_guide": LAMBDA_GUIDE,

        "matrix": DAG_PATH,

    }

    sys.stdout.flush()



    # ===== reeval =====

    if VARIANT:

        try:

            import _reeval_lib as _rl

            _rl.DEADLINE_SLOT = DEADLINE_SLOT

            _rl.BURST_PROB = BURST_PROB

            _rl.BURST_SIZE = BURST_SIZE

            _rl.MAX_STEPS = MAX_STEPS

            _rl.STOP_ARRIVAL_STEP = STOP_ARRIVAL_STEP



            variant_cfg = _rl.VARIANT_CONFIG[VARIANT]

            ctx = _rl._build_env_for_reeval(DAG_PATH, RUN_DIR, SEED)

            agent, ckpt_used, ckpt_path = _rl._load_ckpt_and_build_agent(

                ctx, variant_cfg, os.path.join(RUN_DIR, "checkpoints"))

            _eval_mod = importlib.import_module(variant_cfg["eval_module"])

            _eval_fn = getattr(_eval_mod, variant_cfg["eval_fn_name"])

            import torch

            with torch.no_grad():

                e_succ, d_succ, sc_succ, to_info, te_succ = _eval_fn(

                    agent, ctx["ts"], ctx["gs"], ctx["bc"], ctx["arrival_plan"],

                    ctx["task_complex_index"], ctx["max_steps"],

                )

            e_all, d_all = ctx["ts"].get_avg_results(only_successful=False, timeout_charge="deadline")



            # calc_timeout_rate 返回小数 (0~1)；compute_utility_score 按论文口径

            # 要求传入【百分比】并在函数内部 /100，这里乘 100 还原避免双重除。

            rho_app = float(to_info["app_timeout_rate"]) * 100.0

            rho_task = float(to_info.get("task_timeout_rate", 0)) * 100.0



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

                  f"AppTO={rho_app:.2f}% UtilityScore_all={utility_all:.4f} ckpt={ckpt_used}")

            sys.stdout.flush()

        except Exception as ex_eval:

            print(f"[worker][reeval] FAILED (训练已完成, 主结果保留): {ex_eval}")

            traceback.print_exc()

            out["delay_all"] = None

            out["D_all"] = None

            out["UtilityScore_all"] = None

            out["r1_5_eval_error"] = str(ex_eval)

            sys.stdout.flush()

except Exception as ex:

    traceback.print_exc()

    out = {"status": "error", "error": str(ex),

           "variant": VARIANT, "dag": DAG_NAME, "seed": SEED,

           "param_key": PARAM_KEY, "param_val_raw": PARAM_VAL, "override": OVERRIDE,

           "runtime_sec": round(time.time() - t0, 1)}



with open(os.path.join(RUN_DIR, "result.json"), "w", encoding="utf-8") as f:

    json.dump(out, f, indent=2, ensure_ascii=False)

print(f"[worker] result.json -> {RUN_DIR}/result.json")

print(f"[worker] DONE {out.get('status', 'unknown')} in {out.get('runtime_sec', -1)}s")

sys.stdout.flush()

'''





def build_worker_script(run_dir: Path) -> Path:

    script = run_dir / "_worker.py"

    code = WORKER_TEMPLATE.replace("__PROJECT_ROOT__", PROJECT_ROOT)

    script.write_text(code, encoding="utf-8")

    return script





def run_one(task, out_root: Path) -> dict:

    """task = (param_key, val, dag_name, dag_path, seed) 或 (..., override_dict)."""

    if len(task) == 5:

        param_key, val, dag_name, dag_path, seed = task

        override = {param_key: val}

    else:

        param_key, val, dag_name, dag_path, seed, override = task



    val_tag = _fmt_val(val)

    run_dir = out_root / f"{dag_name}__{ALGO_LABEL}__p{param_key}__v{val_tag}__seed{seed}"

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

    env["ABL_ALGO_MOD"] = ALGO_MOD

    env["ABL_ALGO_FN"] = ALGO_FN

    env["ABL_VARIANT"] = VARIANT

    env["ABL_SEED"] = str(seed)

    env["ABL_EPISODES"] = str(EPISODES)

    env["ABL_GPU"] = str(GPU)

    env["ABL_RUN_DIR"] = str(run_dir)

    env["ABL_PARAM_KEY"] = str(param_key)

    env["ABL_PARAM_VAL"] = str(val)

    env["ABL_OVERRIDE"] = json.dumps(override, ensure_ascii=False)

    # LR / ENTROPY_COEF / LAMBDA_GUIDE 既是 train() 参数, 又是需要扫的超参.

    # 默认用 PARAM_DEFAULTS; 若本次 sweep 正在扫这三个之一, 用 sweep 的 val,

    # 否则 lambda_guide=0 这种"关闭 guide"的实验实际仍以 0.1 训练 (banner 会露馅).

    _eff_lr       = override["LR"]             if "LR"             in override else PARAM_DEFAULTS["LR"]

    _eff_entropy  = override["ENTROPY_COEF"]   if "ENTROPY_COEF"   in override else PARAM_DEFAULTS["ENTROPY_COEF"]

    _eff_lambda   = override["LAMBDA_GUIDE"]   if "LAMBDA_GUIDE"   in override else PARAM_DEFAULTS["LAMBDA_GUIDE"]

    env["ABL_LR"] = str(_eff_lr)

    env["ABL_ENTROPY_COEF"] = str(_eff_entropy)

    env["ABL_LAMBDA_GUIDE"] = str(_eff_lambda)

    env["ABL_USE_HEURISTIC_SORT"] = "1" if USE_HEURISTIC_SORT else "0"

    env["ABL_DEADLINE_SLOT"] = str(DEADLINE_SLOT)

    env["ABL_BURST_PROB"] = str(BURST_PROB)

    env["ABL_BURST_SIZE"] = str(BURST_SIZE)

    env["ABL_MAX_STEPS"] = str(MAX_STEPS)

    env["ABL_STOP_ARRIVAL_STEP"] = str(STOP_ARRIVAL_STEP)

    if USER_NUM is not None:

        env["ABL_USER_NUM"] = str(USER_NUM)



    t0 = time.time()

    label_prefix = f"[{dag_name}/{ALGO_LABEL}/{param_key}={val}/seed{seed}] "

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

    res["algo"] = ALGO_LABEL

    res["seed"] = seed

    res["param_key"] = param_key

    res["param_val"] = val

    res["is_default"] = (PARAM_DEFAULTS.get(param_key) == val) if "__x__" not in str(param_key) else False

    res["wall_sec"] = round(elapsed, 1)

    res["wall_min"] = round(elapsed / 60, 2)

    res["run_dir"] = str(run_dir)

    return res





def _write_sweep_csv(path: Path, results: list) -> None:

    cols = ["dag", "algo", "param_key", "param_val", "is_default", "seed", "status",

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

    out_tag = f"{ALGO_LABEL}_{EXP_TAG}_{ts_tag}" if EXP_TAG else f"{ALGO_LABEL}_sens_{ts_tag}"

    out_root = Path(DEFAULT_OUT_BASE) / out_tag

    out_root.mkdir(parents=True, exist_ok=True)



    tasks = build_task_list()

    total = len(tasks)



    print(f"\n{'='*80}")

    print(f"{ALGO_LABEL} 敏感性扫描 runner")

    print(f"{'='*80}")

    print(f"输出目录:   {out_root}  (新时间戳, 不覆盖旧批次)")

    print(f"算法:       {ALGO_MOD}.{ALGO_FN}  (variant={VARIANT or 'N/A'})")

    print(f"DAG 拓扑:   {[d[0] for d in DAGS]}")

    print(f"种子:       {SEEDS}")

    print(f"扫描参数:   {[p[0] for p in SWEEP_PARAMS]} + {len(GRID_PARAMS)} 个网格")

    print(f"总实验数:   {total}  (参数×取值×DAG×seed)")

    print(f"每实验:     {EPISODES} episodes + reeval")

    print(f"GPU:        {GPU}")

    print(f"{'='*80}\n")



    results_all = []

    t_global = time.time()



    for idx, task in enumerate(tasks, 1):

        param_key = task[0]; val = task[1]; dag_name = task[2]; seed = task[4]

        print(f"\n{'='*70}")

        print(f"[{idx}/{total}] DAG={dag_name}  {param_key}={val}  seed={seed}  eps={EPISODES}")

        print(f"{'='*70}")

        sys.stdout.flush()

        res = run_one(task, out_root)

        results_all.append(res)



        # 增量写大表 (防中途崩溃)

        _write_sweep_csv(out_root / "summary_sweep.csv", results_all)

        (out_root / "summary_sweep.json").write_text(

            json.dumps(results_all, indent=2, ensure_ascii=False), encoding="utf-8")



        print(f"\n[{idx}/{total}] 完成!")

        print(f"  status:        {res.get('status')}")

        print(f"  {param_key}={val} (default={res.get('is_default')})")

        print(f"  AppTO:         {res.get('app_timeout_rate', '-')}")

        print(f"  Score:         {res.get('score', '-')}")

        print(f"  UtilityScore:  {res.get('UtilityScore_all', '-')}")

        print(f"  wall时间:      {res.get('wall_min', '-')} 分钟")

        sys.stdout.flush()



    elapsed = time.time() - t_global

    print(f"\n{'='*80}")

    print(f"[ALL DONE] {total} runs in {elapsed/60:.1f} 分钟")

    print(f"{'='*80}")



    # 按 (param_key, dag) 分组打印小结

    seen = set()

    for r in results_all:

        key = (r.get("param_key"), r.get("dag"))

        if key in seen:

            continue

        seen.add(key)

        pk, dn = key

        grp = [x for x in results_all if x.get("param_key") == pk and x.get("dag") == dn]

        if not grp:

            continue

        print(f"\n{'='*70}")

        print(f"小结: param={pk}  DAG={dn}")

        print(f"{'='*70}")

        print(f"{'val':<12} {'is_def':<8} {'AppTO':<10} {'Score':<12} {'UtilityScore':<14} {'D_all':<10}")

        print(f"{'-'*70}")

        for x in sorted(grp, key=lambda z: (str(z.get('param_val')), z.get('seed', 0))):

            def _fmt(v, pct=False):

                if v is None or v == '' or v == '-': return '-'

                if isinstance(v, (int, float)):

                    return f"{v:.4f}" if not pct else f"{v:.2%}"

                return str(v)

            print(f"{str(x.get('param_val')):<12} {str(x.get('is_default')):<8} "

                  f"{_fmt(x.get('app_timeout_rate'), True):<10} {_fmt(x.get('score')):<12} "

                  f"{_fmt(x.get('UtilityScore_all')):<14} {_fmt(x.get('D_all')):<10}  "

                  f"(seed{x.get('seed')})")

        print(f"{'='*70}")



    print(f"\n全局扫描CSV:  {out_root / 'summary_sweep.csv'}")

    print(f"全局扫描JSON: {out_root / 'summary_sweep.json'}")





if __name__ == "__main__":

    main()

