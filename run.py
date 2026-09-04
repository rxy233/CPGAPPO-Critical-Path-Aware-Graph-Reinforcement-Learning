# -*- coding: utf-8 -*-
"""CPGAPPO 论文实验统一运行入口 (run.py)。

CPGAPPO = Constraint-Preserving Guide-Actor PPO with dual GAT。

通过 --mode / --dags / --seeds / --algos / --param / --value 选择实验，
无需手改源文件常量。详细说明见 README.md。

用法:
  python run.py --mode reeval_only --dags default --seeds 42 --algos CPGAPPO
  python run.py --mode reeval_only --dags all --seeds 1 3 5 7 42
  python run.py --mode main_comparison --dags all --seeds 1 3 5 7 42 --episodes 100
  python run.py --mode ablation_only --dags all --seeds 1 3 5 7 42
  python run.py --mode sensitivity --param LAMBDA_GUIDE --values 0.0 0.05 0.1 0.2 0.4
  python run.py --mode collect
"""
import os
import sys
import argparse
import importlib
import shutil
from pathlib import Path
from datetime import datetime

# ============================================================================
# 项目根 = 本文件所在目录 (run.py 在 CPGAPPO/ 根)
# ============================================================================
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# matrix 路径 (反斜杠, get_graph_cache md5 要求)
MATRIX_PATHS = {
    "default": os.path.normpath(os.path.join(REPO_ROOT, "matrix", "matrix_60.txt")),
    "chain":   os.path.normpath(os.path.join(REPO_ROOT, "matrix", "matrix_60_chain.txt")),
    "wide":    os.path.normpath(os.path.join(REPO_ROOT, "matrix", "matrix_60_wide.txt")),
}
# normpath 在 win32 上用反斜杠; linux 上用正斜杠, md5 仍一致 (内部所有调用都走这个 dict)

ALL_DAGS = ["chain", "default", "wide"]

# 7 个 CPGAPPO 家族消融变体 (走 _reeval_lib, 共享 GAT_PPO_Agent_CPGAPPO)
CP_VARIANTS = ["CPGAPPO", "noguidece", "noshield", "noappcredit", "nocp", "fwdonly", "alloff"]

# 5 个外部 RL 对比算法 (走 _dall_reeval_lib)
EXTERNAL_RL = ["DGMA_adapt", "DGMA_paper", "TransEdgeStyle", "GATDQN", "PPO"]

# 5 个启发式基线 (无 ckpt, 现跑)
BENCH_ALGOS = ["HybridPSOGA", "Genetic", "Greedy", "Edge-only", "Local-only"]

# 全部 17 算法 (main_comparison 默认)
ALL_ALGOS = CP_VARIANTS + EXTERNAL_RL + BENCH_ALGOS

# sensitivity 可扫的参数白名单 (与 _run3_cpgappo_changenumber.py 一致)
SENS_PARAMS = {
    "LAMBDA_GUIDE", "ENTROPY_COEF", "LR",
    "BONUS_FIN", "BONUS_TO", "TERMINAL_COEF",
    "W_E", "W_D",
    "SHIELD_THR_HIGH", "SHIELD_THR_NORM",
    "CP_REM_COEFF", "OVERFLOW_MULT",
    "USER_NUM",
}

# 预训练 ckpt 文件名表 (CP 家族 + 外部 RL; 启发式无 ckpt)
# 注: .pt 文件名与算法名一一对应, 关联 pretrained/ 下 180 个 ckpt.
CKPT_NAMES = {
    "CPGAPPO":      "CPGAPPO.pt",
    "noguidece":    "CPGAPPO_noguidece.pt",
    "noshield":     "CPGAPPO_noshield.pt",
    "noappcredit":  "CPGAPPO_noappcredit.pt",
    "nocp":         "CPGAPPO_nocp.pt",
    "fwdonly":      "CPGAPPO_fwdonly.pt",
    "alloff":       "CPGAPPO_alloff.pt",
    "DGMA_adapt":     "DGMA_adapt.pt",
    "DGMA_paper":     "DGMA_paper_seed{S}_best.pt",  # {S} = seed
    "TransEdgeStyle": "TransEdgeStyle.pt",
    "GATDQN":         "dynamic_gat_dqn_model.pth",
    "PPO":            "PPO.pt",
}


def _resolve_dags(dags_arg):
    """把 --dags 参数解析成 [(name, path), ...]. all / 默认 → 三拓扑都跑."""
    if not dags_arg or "all" in dags_arg:
        names = ALL_DAGS
    else:
        names = []
        for d in dags_arg:
            d = d.strip().lower()
            if d in MATRIX_PATHS and d not in names:
                names.append(d)
            else:
                print(f"[warn] 未知 DAG '{d}', 跳过")
    return [(n, MATRIX_PATHS[n]) for n in names]


def _resolve_algos(algos_arg, mode):
    """把 --algos 参数解析成算法列表. 不传则按模式给默认全集."""
    if algos_arg:
        # 校验名字
        out = []
        for a in algos_arg:
            if a in ALL_ALGOS:
                out.append(a)
            else:
                print(f"[warn] 未知算法 '{a}', 跳过 (合法: {ALL_ALGOS})")
        return out
    # 默认
    if mode == "ablation_only":
        return list(CP_VARIANTS)
    return list(ALL_ALGOS)


# ============================================================================
# mode=reeval_only: 从 pretrained/ 复制 ckpt 到临时 run_dir, 再调 reeval 库
# ============================================================================
def _stage_ckpt(algo, dag, seed, ckpt_dir):
    """把 pretrained/{dag}/{algo}/seed{seed}/{ckpt_name} 复制到 ckpt_dir/."""
    src_dir = os.path.join(REPO_ROOT, "pretrained", dag, algo, f"seed{seed}")
    name_tpl = CKPT_NAMES[algo]
    fname = name_tpl.replace("{S}", str(seed))
    src = os.path.join(src_dir, fname)
    if not os.path.exists(src):
        raise FileNotFoundError(
            f"预训练 ckpt 缺失: {src}\n"
            f"  请确认 pretrained/{dag}/{algo}/seed{seed}/ 下有 {fname};\n"
            f"  或用 --mode main_comparison 从头训练生成 ckpt。"
        )
    os.makedirs(ckpt_dir, exist_ok=True)
    dst = os.path.join(ckpt_dir, fname)
    if not os.path.exists(dst):
        shutil.copy2(src, dst)
    return dst


def _stage_run_dirs(algos, dags, seeds, out_root):
    """为 reeval_only 预备 run_dir 树: out_root/{dag}__{algo}__seed{N}/checkpoints/.
    返回 {(algo, dag, seed): run_dir}. 启发式基线不需要 ckpt, 只建空 run_dir。
    """
    plan = {}
    for dag_name, _ in dags:
        for algo in algos:
            for seed in seeds:
                run_dir = os.path.join(out_root, f"{dag_name}__{algo}__seed{seed}")
                ckpt_dir = os.path.join(run_dir, "checkpoints")
                os.makedirs(run_dir, exist_ok=True)
                os.makedirs(ckpt_dir, exist_ok=True)
                if algo not in BENCH_ALGOS:
                    _stage_ckpt(algo, dag_name, seed, ckpt_dir)
                plan[(algo, dag_name, seed)] = run_dir
    return plan


def mode_reeval_only(args):
    """加载预训练 ckpt, 在标准 env 上跑 deterministic eval, 直接出 D_all."""
    dags = _resolve_dags(args.dags)
    algos = _resolve_algos(args.algos, "reeval_only")
    seeds = [int(s) for s in args.seeds]

    ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.tag}" if args.tag else ""
    out_root = os.path.join(REPO_ROOT, "results", f"main_comparison_reeval{suffix}_{ts_tag}")
    os.makedirs(out_root, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"mode=reeval_only  用预训练 ckpt 直接重算 D_all (deadline 口径)")
    print(f"{'='*80}")
    print(f"输出根目录:   {out_root}")
    print(f"DAG 拓扑:     {[d[0] for d in dags]}")
    print(f"算法:         {algos}")
    print(f"种子:         {seeds}")
    print(f"总格数:       {len(algos) * len(dags) * len(seeds)}")
    print(f"{'='*80}\n")

    # 预备 run_dir + 复制 ckpt
    plan = _stage_run_dirs(algos, dags, seeds, out_root)

    # 关键: 让 _dall_reeval_lib / _reeval_lib 把 OUT_ROOT / run_dir_base 指向我们的 out_root
    os.environ["CPGAPPO_DALL_OUT_ROOT"] = out_root

    # 把 runners/ 加到 sys.path, 才能 import _reeval_lib / _dall_reeval_lib
    runners_dir = os.path.join(REPO_ROOT, "runners")
    if runners_dir not in sys.path:
        sys.path.insert(0, runners_dir)

    results_all = []
    t_global = __import__("time").time()

    for algo in algos:
        # CP 家族走 _reeval_lib.reeval_one(variant, topology, seed)
        # 外部 RL + 启发式走 _dall_reeval_lib.reeval_one(algo, dag, seed)
        is_cp = algo in CP_VARIANTS
        if is_cp:
            import _reeval_lib as rl
        else:
            import _dall_reeval_lib as drl

        for dag_name, dag_path in dags:
            for seed in seeds:
                run_dir = plan[(algo, dag_name, seed)]
                t0 = __import__("time").time()
                print(f"\n[reeval] {algo} / {dag_name} / seed{seed}  -> {run_dir}")
                try:
                    if is_cp:
                        # _reeval_lib 用 topo_cfg["run_dir_base"] + f"{topo}__{variant}__seed{N}"
                        # 我们已通过 _stage_run_dirs 在 out_root 下建好了对应目录并把 ckpt 放好,
                        # 但 _reeval_lib 的 run_dir_base 来自 _resolve_run_dir_base_portable
                        # (扫 results/ 下 ablation_classic_*). 为让它指到 out_root, 我们
                        # 直接 monkey-patch 其 TOPOLOGY_CONFIG_FALLBACK.
                        rl.TOPOLOGY_CONFIG_FALLBACK[dag_name]["run_dir_base"] = out_root
                        rl.TOPOLOGY_CONFIG_FALLBACK[dag_name]["matrix_path"] = dag_path
                        res = rl.reeval_one(algo, dag_name, seed)
                    else:
                        # _dall_reeval_lib 的 OUT_ROOT 已通过 env var 指过来
                        # 它内部会自己拼 OUT_ROOT/{dag}__{algo}__seed{seed}
                        drl.DAG_MATRIX[dag_name] = dag_path
                        res = drl.reeval_one(algo, dag_name, seed)
                    res["wall_sec"] = round(__import__("time").time() - t0, 1)
                except Exception as ex:
                    import traceback
                    traceback.print_exc()
                    res = {
                        "status": "error", "algorithm": algo,
                        "dag": dag_name, "seed": seed,
                        "error": str(ex),
                        "wall_sec": round(__import__("time").time() - t0, 1),
                    }
                # 统一字段名: _reeval_lib 用 variant/topology, _dall_reeval_lib 用 algorithm/dag
                res.setdefault("algorithm", res.get("variant", algo))
                res.setdefault("dag", res.get("topology", dag_name))
                res.setdefault("variant", algo)
                res.setdefault("topology", dag_name)
                results_all.append(res)
                print(f"  -> status={res.get('status')}  "
                      f"D_all={res.get('D_all', '-')}  "
                      f"Util={res.get('UtilityScore_all', '-')}  "
                      f"({res.get('wall_sec')}s)")

    elapsed = __import__("time").time() - t_global
    _write_summary_csv(out_root, results_all, prefix="reeval")
    _print_summary_table(results_all, algos, dags, seeds)
    print(f"\n[ALL DONE] {len(results_all)} cells in {elapsed/60:.1f} 分钟")
    print(f"汇总CSV:  {out_root}/summary_reeval.csv")
    print(f"汇总JSON: {out_root}/summary_reeval.json")


# ============================================================================
# mode=main_comparison / ablation_only: 调用 _run3_main_comparison_3dag.py
# 通过 import + monkey-patch 它的 DAGS/SEEDS/ALGOS, 再调 main()
# ============================================================================
def _run_main_comparison_like(args, algos_filter):
    """通用: 跑 _run3_main_comparison_3dag, 但限制 ALGOS = algos_filter."""
    runners_dir = os.path.join(REPO_ROOT, "runners")
    if runners_dir not in sys.path:
        sys.path.insert(0, runners_dir)

    import _run3_main_comparison_3dag as mc

    # 覆盖 DAGS
    dags = _resolve_dags(args.dags)
    mc.DAGS = dags

    # 覆盖 SEEDS
    mc.SEEDS = [int(s) for s in args.seeds]

    # 覆盖 EPISODES / GPU
    mc.EPISODES = int(args.episodes)
    mc.GPU = int(args.gpu)

    # 覆盖 ALGOS (按 algos_filter 过滤原表)
    if algos_filter is not None:
        mc.ALGOS = [row for row in mc.ALGOS if row[0] in algos_filter]

    # 覆盖输出根 (runners 副本里 DEFAULT_OUT_BASE 已指 results/, 这里不强制改)
    print(f"\n[main_comparison] DAGS={[d[0] for d in mc.DAGS]}  SEEDS={mc.SEEDS}  "
          f"ALGOS={[r[0] for r in mc.ALGOS]}  EPISODES={mc.EPISODES}")
    mc.main()


def mode_main_comparison(args):
    algos_filter = set(args.algos) if args.algos else None
    _run_main_comparison_like(args, algos_filter=algos_filter)


def mode_ablation_only(args):
    _run_main_comparison_like(args, algos_filter=set(CP_VARIANTS))


# ============================================================================
# mode=sensitivity: 调用对应 _run3_cpgappo_{PARAM}_changenumber.py
# ============================================================================
# 敏感性参数 → 对应 runner 文件名 (在 runners/ 下)
SENS_RUNNER_FILES = {
    "LAMBDA_GUIDE":    "_run3_cpgappo_LAMBDA_GUIDE_changenumber.py",
    "BONUS_TO":        "_run3_cpgappo_BONUS_TO_changenumber.py",
    "W_E":             "_run3_cpgappo_W_E_changenumber.py",
    "W_D":             "_run3_cpgappo_W_D_changenumber.py",
    "SHIELD_THR_HIGH": "_run3_cpgappo_SHIELD_THR_changenumber.py",
    "SHIELD_THR_NORM": "_run3_cpgappo_SHIELD_THR_changenumber.py",
    "OVERFLOW_MULT":   "_run3_cpgappo_OVERFLOW_CP_changenumber.py",
    "CP_REM_COEFF":    "_run3_cpgappo_changenumber.py",
    "BONUS_FIN":       "_run3_cpgappo_changenumber.py",
    "TERMINAL_COEF":   "_run3_cpgappo_changenumber.py",
    "ENTROPY_COEF":    "_run3_cpgappo_changenumber.py",
    "LR":              "_run3_cpgappo_changenumber.py",
    "USER_NUM":        "_run3_cpgappo_changenumber.py",
}


def mode_sensitivity(args):
    """跑 CPGAPPO 单参数 OFAT 扫描. 一次一个 param."""
    if not args.param:
        print("[error] sensitivity 模式必须用 --param 指定参数 (见 SENS_PARAMS)")
        sys.exit(2)
    param = args.param
    if param not in SENS_PARAMS:
        print(f"[error] --param '{param}' 不在可扫白名单: {sorted(SENS_PARAMS)}")
        sys.exit(2)

    values = args.values
    if not values:
        print(f"[error] 必须用 --values 给出 {param} 的取值列表, 如 --values 0 0.05 0.1 0.2 0.4")
        sys.exit(2)
    # 转数值
    parsed_values = []
    for v in values:
        try:
            parsed_values.append(float(v) if ("." in v or "e" in v.lower()) else int(v))
        except ValueError:
            parsed_values.append(v)  # 保留字符串

    runner_file = SENS_RUNNER_FILES.get(param, "_run3_cpgappo_changenumber.py")
    runners_dir = os.path.join(REPO_ROOT, "runners")
    if runners_dir not in sys.path:
        sys.path.insert(0, runners_dir)
    # 仓库根也要在 sys.path, 让子进程/worker 能 import Algorithms / Environment / utils
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)

    mod_name = runner_file[:-3]  # 去掉 .py
    print(f"\n[sensitivity] 加载 runner: {runner_file}  (param={param} values={parsed_values})")
    sens = importlib.import_module(mod_name)
    # 确保 runner 内的 PROJECT_ROOT 指向仓库根 (而非 runners/)
    if hasattr(sens, "PROJECT_ROOT"):
        sens.PROJECT_ROOT = REPO_ROOT

    # 覆盖 SWEEP_PARAMS = [(param, values)]  (单因子)
    sens.SWEEP_PARAMS = [(param, parsed_values)]
    # 覆盖 SEEDS / DAGS / EPISODES / GPU
    sens.SEEDS = [int(s) for s in args.seeds]
    sens.DAGS = _resolve_dags(args.dags)
    sens.EPISODES = int(args.episodes)
    sens.GPU = int(args.gpu)
    if args.tag:
        sens.EXP_TAG = args.tag

    print(f"[sensitivity] SWEEP_PARAMS={sens.SWEEP_PARAMS}")
    print(f"[sensitivity] SEEDS={sens.SEEDS}  DAGS={[d[0] for d in sens.DAGS]}  "
          f"EPISODES={sens.EPISODES}  GPU={sens.GPU}")
    sens.main()


# ============================================================================
# mode=collect: 调用 _collect_revision_v2.py 扫 results/ 汇总成 CSV
# ============================================================================
def mode_collect(args):
    runners_dir = os.path.join(REPO_ROOT, "runners")
    if runners_dir not in sys.path:
        sys.path.insert(0, runners_dir)
    # _collect_revision_v2.py 在 import 时即执行全部汇总逻辑 (无 main() 函数),
    # 顶部 ROOT_FENXI 已被改成 results/, OUT_DIR 已改成 collected_csv/.
    print(f"\n[collect] 扫 {os.path.join(REPO_ROOT, 'results')} 汇总到 {os.path.join(REPO_ROOT, 'collected_csv')}")
    import _collect_revision_v2  # noqa: F401  (import 即执行)


# ============================================================================
# 输出工具
# ============================================================================
def _write_summary_csv(out_root, results, prefix="summary"):
    import json
    cols = ["dag", "algorithm", "variant", "seed", "status",
            "D_succ", "D_all", "E_succ", "E_all",
            "AppTO", "TaskTO",
            "UtilityScore_succ", "UtilityScore_all",
            "ckpt_used", "elapsed_sec", "wall_sec"]
    lines = [",".join(cols)]
    for r in results:
        lines.append(",".join(str(r.get(c, "")) for c in cols))
    csv_path = os.path.join(out_root, f"{prefix}.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    json_path = os.path.join(out_root, f"{prefix}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        import json
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"summary CSV  -> {csv_path}")
    print(f"summary JSON -> {json_path}")


def _print_summary_table(results, algos, dags, seeds):
    print(f"\n{'='*100}")
    print("结果汇总 (按 algo / dag / seed 排序)")
    print(f"{'='*100}")
    print(f"{'algo':<18} {'dag':<10} {'seed':<6} {'status':<10} "
          f"{'D_all':<10} {'Util_all':<12} {'AppTO':<8}")
    print("-" * 100)
    for algo in algos:
        for dag_name, _ in dags:
            for seed in seeds:
                r = next((x for x in results
                          if (x.get("algorithm") == algo or x.get("variant") == algo)
                          and (x.get("dag") == dag_name or x.get("topology") == dag_name)
                          and int(x.get("seed", -1)) == seed), None)
                if not r:
                    continue
                d_all = r.get("D_all")
                util = r.get("UtilityScore_all")
                appto = r.get("AppTO") or r.get("app_timeout_rate")
                def _f(v, pct=False):
                    if v is None or v == "" or v == "-": return "-"
                    if isinstance(v, (int, float)):
                        return f"{v:.4f}" if not pct else f"{v:.2%}"
                    return str(v)
                print(f"{algo:<18} {dag_name:<10} {seed:<6} {str(r.get('status','')):<10} "
                      f"{_f(d_all):<10} {_f(util):<12} {_f(appto, True):<8}")
    print("=" * 100)


# ============================================================================
# CLI
# ============================================================================
def build_parser():
    p = argparse.ArgumentParser(
        prog="run.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--mode", required=True,
                   choices=["main_comparison", "ablation_only", "sensitivity",
                            "reeval_only", "collect"],
                   help="运行模式 (见上方文档)")
    p.add_argument("--dags", nargs="+", default=["all"],
                   help="DAG 拓扑: chain / default / wide / all (默认 all)")
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 3, 42],
                   help="随机种子列表 (默认 1 3 42, 与论文主表一致)")
    p.add_argument("--algos", nargs="+", default=None,
                   help="算法白名单 (不传 = 模式默认全集). 合法: "
                        + ", ".join(ALL_ALGOS))
    p.add_argument("--param", default=None,
                   help="sensitivity 模式要扫的参数 (见 SENS_PARAMS)")
    p.add_argument("--values", nargs="+", default=None,
                   help="sensitivity 模式该参数的取值列表")
    p.add_argument("--episodes", type=int, default=100,
                   help="RL 训练轮数 (默认 100, 与论文一致)")
    p.add_argument("--gpu", type=int, default=0,
                   help="GPU id (默认 0; 无 GPU 自动回退 CPU)")
    p.add_argument("--tag", default=None,
                   help="输出目录后缀, 便于区分多次跑 (默认空)")
    return p


def main():
    args = build_parser().parse_args()
    # 把 REPO_ROOT 放到 sys.path 最前, 让子进程 import Algorithms / Environment 等能找到
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)

    mode = args.mode
    print(f"\n[run.py] mode={mode}  dags={args.dags}  seeds={args.seeds}  "
          f"algos={args.algos}  episodes={args.episodes}  gpu={args.gpu}  tag={args.tag}")

    if mode == "main_comparison":
        mode_main_comparison(args)
    elif mode == "ablation_only":
        mode_ablation_only(args)
    elif mode == "sensitivity":
        mode_sensitivity(args)
    elif mode == "reeval_only":
        mode_reeval_only(args)
    elif mode == "collect":
        mode_collect(args)
    else:
        print(f"[error] 未知 mode: {mode}")
        sys.exit(2)


if __name__ == "__main__":
    main()
