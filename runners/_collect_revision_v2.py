# -*- coding: utf-8 -*-
"""
Collect ALREADY-RUN experiment data into categorized CSVs for the 3rd major revision.
Output -> <repo>/collected_csv

Logic:
  - Scan results/{main_comparison_*, ablation_*} for result.json (newest mtime wins per cell).
  - For 'main' variant: result.json from old batches has NO D_all.
    Merge result_reeval.json (reeval产物) when it exists -> provides true D_all + Util_all.
  - Scan results/main3_sens_* for sensitivity result.json (all have D_all in-place).
  - Emit categorized CSVs + a gap summary showing which cells still lack D_all.

NO re-running. NO skip-if-exists. Pure collection of what's already on disk.
"""
import os, json, glob, csv, shutil, math
from collections import defaultdict
from datetime import datetime

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_FENXI = os.path.join(_REPO_ROOT, "results")
OUT_DIR = os.path.join(_REPO_ROOT, "collected_csv")
RAW_DIR    = os.path.join(OUT_DIR, 'raw_result_json')

# 旧版 CPGAPPO 中间变体, 论文不使用, 全部排除
EXCLUDE_ALGOS = {'main', 'main_v2', 'main_v3', 'main_v4', 'main_v5'}

ALGO_LABEL = {
    'main3': 'CPGAPPO',
    'noguidece': 'noguidece', 'noshield': 'noshield',
    'noappcredit': 'noappcredit', 'nocp': 'nocp',
    'fwdonly': 'fwdonly', 'alloff': 'alloff',
    'DGMA_adapt': 'DGMA_adapt', 'DGMA_paper': 'DGMA_paper',
    'TransEdgeStyle': 'TransEdgeStyle', 'GATDQN': 'GATDQN', 'PPO': 'PPO',
    'HybridPSOGA': 'HybridPSOGA', 'Genetic': 'Genetic', 'Greedy': 'Greedy',
    'Edge-only': 'Edge-only', 'Local-only': 'Local-only',
}
ALGO_KIND = {
    'main3': 'research',
    'noguidece': 'ablation', 'noshield': 'ablation', 'noappcredit': 'ablation',
    'nocp': 'ablation', 'fwdonly': 'ablation', 'alloff': 'ablation',
    'DGMA_adapt': 'ext_rl', 'DGMA_paper': 'ext_rl', 'TransEdgeStyle': 'ext_rl',
    'GATDQN': 'ext_rl', 'PPO': 'ext_rl',
    'HybridPSOGA': 'heuristic', 'Genetic': 'heuristic', 'Greedy': 'heuristic',
    'Edge-only': 'heuristic', 'Local-only': 'heuristic',
}
TABLE_IV_ALGOS = ['main3', 'DGMA_adapt', 'DGMA_paper', 'TransEdgeStyle', 'GATDQN', 'PPO',
                  'HybridPSOGA', 'Genetic', 'Greedy', 'Edge-only', 'Local-only']
TABLE_V_ALGOS  = ['main3', 'noguidece', 'noshield', 'noappcredit', 'nocp', 'fwdonly', 'alloff']
DAGS    = ['chain', 'default', 'wide']
# 主表协议种子: 论文 Table IV/V 用 5 seed {1,3,5,7,42}, mean±std
SEEDS   = ['1', '3', '5', '7', '42']

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)
def _sub(label):
    d = os.path.join(RAW_DIR, label)
    os.makedirs(d, exist_ok=True)
    return d
_sub('_sensitivity')

# 清理上一轮输出里残留的旧变体目录, 论文不使用
for stale in ['CPGAPPO_v3', 'main', 'main_v2', 'main_v3', 'main_v4', 'main_v5']:
    stale_dir = os.path.join(RAW_DIR, stale)
    if os.path.isdir(stale_dir):
        shutil.rmtree(stale_dir, ignore_errors=True)
        print(f'[cleanup] removed stale raw dir: {stale_dir}')


def _load(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# =====================================================================
# 1. Collect main_comparison + ablation -> per (dag, algo, seed) newest result
#    Merge result_reeval.json for 'main' variant (provides true D_all/Util_all)
# =====================================================================
# key=(dag, algo, seed) -> dict(record)
best = {}
for d in sorted(glob.glob(os.path.join(ROOT_FENXI, 'main_comparison_*')) +
                glob.glob(os.path.join(ROOT_FENXI, 'ablation_*'))):
    for p in glob.glob(os.path.join(d, '*', 'result.json')):
        try:
            bn = os.path.basename(os.path.dirname(p))
            parts = bn.split('__')
            if len(parts) < 3:
                continue
            dag, algo, seed = parts[0], parts[1], parts[2].replace('seed', '')
            if algo in EXCLUDE_ALGOS:
                continue
            mt = os.path.getmtime(p)
            key = (dag, algo, seed)
            if key in best and mt <= best[key]['_mt']:
                continue
            r = _load(p)
            rr_path = os.path.join(os.path.dirname(p), 'result_reeval.json')
            rr = _load(rr_path) if os.path.exists(rr_path) else None
            # Prefer in-place D_all; fall back to reeval D_all
            dall  = r.get('delay_all')
            dsucc = r.get('delay_succ') or r.get('delay')
            eall  = r.get('E_all') or r.get('energy')
            esucc = r.get('E_succ') or r.get('energy')
            util_all  = r.get('UtilityScore_all')
            util_succ = r.get('UtilityScore_succ')
            dall_src  = 'result.json'
            if dall is None and rr is not None:
                dall  = rr.get('D_all') or rr.get('delay_all')
                dsucc = rr.get('D_succ') or dsucc
                eall  = rr.get('E_all') or eall
                esucc = rr.get('E_succ') or esucc
                util_all  = rr.get('UtilityScore_all')
                util_succ = rr.get('UtilityScore_succ')
                dall_src  = 'result_reeval.json'
            best[key] = {
                'dag': dag, 'algo': algo, 'label': ALGO_LABEL.get(algo, algo),
                'kind': ALGO_KIND.get(algo, 'other'), 'seed': seed,
                'app_timeout_rate': r.get('app_timeout_rate'),
                'task_timeout_rate': r.get('task_timeout_rate'),
                'energy': r.get('energy'),
                'delay_succ': dsucc, 'delay_all': dall,
                'E_succ': esucc, 'E_all': eall,
                'UtilityScore_succ': util_succ,
                'UtilityScore_all': util_all,
                'score': r.get('score'),
                'runtime_min': r.get('runtime_min'),
                'inference_time_ms': r.get('inference_time_ms'),
                'ckpt_used': r.get('ckpt_used', ''),
                'src_dir': os.path.basename(d),
                'src_result': p,
                'has_reeval': rr is not None,
                'D_all_src': dall_src,
                '_mt': mt,
                '_mtime_str': datetime.fromtimestamp(mt).strftime('%Y-%m-%d %H:%M'),
            }
        except Exception as e:
            print(f'  skip {p}: {e}')

print(f'[main] collected {len(best)} unique (dag,algo,seed) cells')


# =====================================================================
# 1b. Pre-scan ALL result_reeval.json across batches -> reeval_map
#     (a cell's newest result.json may live in a batch WITHOUT reeval,
#      while an older batch has the reeval with true D_all. Fall back here.)
# =====================================================================
reeval_map = {}  # key=(dag,algo,seed) -> (mtime, rr_dict, src_dir)
for d in sorted(glob.glob(os.path.join(ROOT_FENXI, 'main_comparison_*')) +
                glob.glob(os.path.join(ROOT_FENXI, 'ablation_*'))):
    for p in glob.glob(os.path.join(d, '*', 'result_reeval.json')):
        try:
            bn = os.path.basename(os.path.dirname(p))
            parts = bn.split('__')
            if len(parts) < 3:
                continue
            dag, algo, seed = parts[0], parts[1], parts[2].replace('seed', '')
            if algo in EXCLUDE_ALGOS:
                continue
            mt = os.path.getmtime(p)
            key = (dag, algo, seed)
            if key in reeval_map and mt <= reeval_map[key][0]:
                continue
            rr = _load(p)
            reeval_map[key] = (mt, rr, os.path.basename(d))
        except Exception:
            pass

# Merge reeval_map into best: fill D_all when best[key] still has None
_filled = 0
for key, (rr_mt, rr, src) in reeval_map.items():
    if key not in best:
        continue
    rec = best[key]
    if rec['delay_all'] is not None:
        continue  # already has D_all from result.json
    dall  = rr.get('D_all') or rr.get('delay_all')
    if dall is None:
        continue
    rec['delay_all'] = dall
    rec['delay_succ'] = rr.get('D_succ') or rec['delay_succ']
    rec['E_all']   = rr.get('E_all') or rec['E_all']
    rec['E_succ']  = rr.get('E_succ') or rec['E_succ']
    rec['UtilityScore_all']  = rr.get('UtilityScore_all')
    rec['UtilityScore_succ'] = rr.get('UtilityScore_succ')
    rec['has_reeval'] = True
    rec['D_all_src']  = f'result_reeval.json ({src})'
    _filled += 1
print(f'[reeval] filled D_all for {_filled} cells from result_reeval.json (cross-batch fallback)')


# =====================================================================
# 2. Copy raw result.json (newest per cell) to raw_result_json/<label>/
# =====================================================================
for key, rec in best.items():
    label = rec['label']
    out_name = f"{rec['dag']}_{rec['algo']}_seed{rec['seed']}.json"
    dst = os.path.join(_sub(label), out_name)
    try:
        shutil.copy2(rec['src_result'], dst)
    except Exception:
        pass
    # also copy result_reeval.json if exists (for traceability)
    rr_path = rec['src_result'].replace('result.json', 'result_reeval.json')
    if os.path.exists(rr_path):
        shutil.copy2(rr_path, dst.replace('.json', '_reeval.json'))


# =====================================================================
# 3. Collect sensitivity (main3_sens_*)
# =====================================================================
sens_best = {}
for d in sorted(glob.glob(os.path.join(ROOT_FENXI, 'main3_sens_*'))):
    for p in glob.glob(os.path.join(d, '*', 'result.json')):
        try:
            bn = os.path.basename(os.path.dirname(p))
            parts = bn.split('__')
            if len(parts) < 5:
                continue
            dag = parts[0]
            pk  = parts[2][1:]            # strip leading 'p'
            val = parts[3][1:]             # strip leading 'v'
            seed = parts[4].replace('seed', '')
            mt = os.path.getmtime(p)
            key = (dag, pk, val, seed)     # 含 DAG 维度, 支持跨拓扑对比
            if key in sens_best and mt <= sens_best[key]['_mt']:
                continue
            r = _load(p)
            val_str = val.replace('p', '.')
            sens_best[key] = {
                'dag': dag,
                'param_key': pk, 'param_val': val_str, 'seed': seed,
                'app_timeout_rate': r.get('app_timeout_rate'),
                'task_timeout_rate': r.get('task_timeout_rate'),
                'energy': r.get('energy'),
                'delay_succ': r.get('delay_succ') or r.get('delay'),
                'delay_all': r.get('delay_all'),
                'E_succ': r.get('E_succ'), 'E_all': r.get('E_all'),
                'UtilityScore_succ': r.get('UtilityScore_succ'),
                'UtilityScore_all': r.get('UtilityScore_all'),
                'score': r.get('score'),
                'runtime_min': r.get('runtime_min'),
                'src_dir': os.path.basename(d),
                'src_result': p,
                '_mt': mt,
            }
        except Exception as e:
            print(f'  skip {p}: {e}')

# copy raw sens result.json
for key, rec in sens_best.items():
    out_name = f"{rec['dag']}_{rec['param_key']}_{rec['param_val'].replace('.','p')}_seed{rec['seed']}.json"
    dst = os.path.join(_sub('_sensitivity'), out_name)
    try:
        shutil.copy2(rec['src_result'], dst)
    except Exception:
        pass

print(f'[sens] collected {len(sens_best)} unique (dag,param,val,seed) cells')


# =====================================================================
# 4. Helper: mean/std (sample std, ddof=1) over a list
# =====================================================================
def _stats(vals):
    vals = [v for v in vals if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return (None, None, 0)
    mean = sum(vals) / len(vals)
    if len(vals) == 1:
        return (mean, 0.0, 1)
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return (mean, math.sqrt(var), len(vals))


def _fmt(v, p=4):
    if v is None:
        return ''
    if isinstance(v, float) and math.isnan(v):
        return ''
    return f'{v:.{p}f}'


# =====================================================================
# 5. CSV: master_seedwise.csv  (all cells, per-seed, with D_all + source)
#    Rows = dag, algo, kind, label, seed, AppTO, TaskTO, E, D_succ, D_all,
#           Util_succ, Util_all, D_all_src, src_dir
# =====================================================================
master_cols = ['dag', 'algo', 'kind', 'label', 'seed',
               'app_timeout_rate', 'task_timeout_rate', 'energy',
               'delay_succ', 'delay_all', 'E_succ', 'E_all',
               'UtilityScore_succ', 'UtilityScore_all',
               'score', 'runtime_min', 'inference_time_ms',
               'D_all_src', 'has_reeval', 'src_dir', 'mtime']
rows_sorted = sorted(best.values(),
                     key=lambda r: (r['dag'],
                                    0 if r['algo'] == 'main3' else
                                    (1 if r['kind'] == 'ablation' else
                                     (2 if r['kind'] == 'ext_rl' else 3)),
                                    r['algo'],
                                    int(r['seed']) if r['seed'].isdigit() else 99))
with open(os.path.join(OUT_DIR, 'master_seedwise.csv'), 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=master_cols, extrasaction='ignore')
    w.writeheader()
    for r in rows_sorted:
        w.writerow({**{k: r.get(k) for k in master_cols}, 'mtime': r['_mtime_str']})
print(f'[write] master_seedwise.csv  {len(rows_sorted)} rows')


# =====================================================================
# 6. CSV: table_IV_main_comparison.csv
#    11 algos x 3 DAG, mean+/-std over seeds {1,3,5,7,42}
#    Cols: algo, kind, dag, AppTO_mean, AppTO_std, n,
#          D_succ_mean, D_succ_std, D_all_mean, D_all_std,
#          Util_all_mean, Util_all_std, Util_succ_mean, Util_succ_std
# =====================================================================
def _agg_table(algo_list, seeds, out_name, comment_tag):
    rows = []
    for algo in algo_list:
        for dag in DAGS:
            cells = [best[(dag, algo, s)] for s in seeds if (dag, algo, s) in best]
            if not cells:
                rows.append({'algo': algo, 'label': ALGO_LABEL.get(algo, algo),
                             'kind': ALGO_KIND.get(algo, ''),
                             'dag': dag, 'n': 0})
                continue
            apto_m, apto_s, n = _stats([c['app_timeout_rate'] for c in cells])
            dsucc_m, dsucc_s, _ = _stats([c['delay_succ'] for c in cells])
            dall_m,  dall_s,  _ = _stats([c['delay_all']  for c in cells])
            usucc_m, usucc_s, _ = _stats([c['UtilityScore_succ'] for c in cells])
            uall_m,  uall_s,  _ = _stats([c['UtilityScore_all']  for c in cells])
            e_m, e_s, _ = _stats([c['energy'] for c in cells])
            rows.append({
                'algo': algo, 'label': ALGO_LABEL.get(algo, algo),
                'kind': ALGO_KIND.get(algo, ''), 'dag': dag, 'n': n,
                'AppTO_mean_pct': apto_m * 100 if apto_m is not None else None,
                'AppTO_std_pct':  apto_s * 100 if apto_s is not None else None,
                'D_succ_mean': dsucc_m, 'D_succ_std': dsucc_s,
                'D_all_mean':  dall_m,  'D_all_std':  dall_s,
                'E_mean': e_m, 'E_std': e_s,
                'Util_succ_mean': usucc_m, 'Util_succ_std': usucc_s,
                'Util_all_mean':  uall_m,  'Util_all_std':  uall_s,
            })
    cols = ['algo', 'label', 'kind', 'dag', 'n',
            'AppTO_mean_pct', 'AppTO_std_pct',
            'D_succ_mean', 'D_succ_std', 'D_all_mean', 'D_all_std',
            'E_mean', 'E_std',
            'Util_succ_mean', 'Util_succ_std', 'Util_all_mean', 'Util_all_std']
    with open(os.path.join(OUT_DIR, out_name), 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print(f'[write] {out_name}  {len(rows)} rows  ({comment_tag})')

_agg_table(TABLE_IV_ALGOS, SEEDS, 'table_IV_main_comparison_5seeds.csv',
           '主对比 5-seed (11 algos x 3 DAG, seeds 1,3,5,7,42)')
_agg_table(TABLE_V_ALGOS,  SEEDS, 'table_V_ablation_5seeds.csv',
           '消融 5-seed (7 algos x 3 DAG, seeds 1,3,5,7,42)')


# =====================================================================
# 7. CSV: sensitivity.csv
#    8 params x 5 vals x 5 seeds, per-seed rows + flag default
# =====================================================================
SENS_DEFAULTS = {
    'LAMBDA_GUIDE': 0.1, 'BONUS_TO': -7.0, 'W_E': 0.05, 'W_D': 0.1,
    'SHIELD_THR_HIGH': 1.10, 'SHIELD_THR_NORM': 1.15,
    'CP_REM_COEFF': 0.45, 'OVERFLOW_MULT': 1.25,
}
SENS_PARAM_ORDER = ['LAMBDA_GUIDE', 'BONUS_TO', 'W_E', 'W_D',
                    'SHIELD_THR_HIGH', 'SHIELD_THR_NORM',
                    'OVERFLOW_MULT', 'CP_REM_COEFF']

sens_rows = []
for (dag, pk, val, seed), r in sens_best.items():
    try:
        val_f = float(val)
        is_def = (abs(SENS_DEFAULTS.get(pk, -999) - val_f) < 1e-9)
    except Exception:
        val_f, is_def = None, False
    sens_rows.append({
        'dag': dag,
        'param_key': pk, 'param_val': val, 'is_default': is_def, 'seed': seed,
        'app_timeout_rate': r['app_timeout_rate'],
        'delay_succ': r['delay_succ'], 'delay_all': r['delay_all'],
        'UtilityScore_all': r['UtilityScore_all'],
        'UtilityScore_succ': r['UtilityScore_succ'],
        'energy': r['energy'],
    })
sens_rows.sort(key=lambda r: (SENS_PARAM_ORDER.index(r['param_key']) if r['param_key'] in SENS_PARAM_ORDER else 99,
                              {'chain':0,'default':1,'wide':2}.get(r['dag'], 99),
                              float(r['param_val']) if r['param_val'].lstrip('-').replace('.', '').isdigit() else 0,
                              int(r['seed']) if r['seed'].isdigit() else 99))
sens_cols = ['dag', 'param_key', 'param_val', 'is_default', 'seed',
             'app_timeout_rate', 'delay_succ', 'delay_all',
             'UtilityScore_all', 'UtilityScore_succ', 'energy']
with open(os.path.join(OUT_DIR, 'sensitivity.csv'), 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=sens_cols, extrasaction='ignore')
    w.writeheader()
    w.writerows(sens_rows)
print(f'[write] sensitivity.csv  {len(sens_rows)} rows')


# =====================================================================
# 8. CSV: sensitivity_summary.csv
#    8 params x 5 vals x {DAGs}, mean+/-std over 5 seeds, flag default
#    按 DAG 分组聚合: 每个 (dag, param, val) 一行, 便于画跨拓扑 OFAT 对比图
# =====================================================================
sens_agg = defaultdict(lambda: defaultdict(list))
for r in sens_rows:
    sens_agg[(r['dag'], r['param_key'], r['param_val'])]['util_all'].append(r['UtilityScore_all'])
    sens_agg[(r['dag'], r['param_key'], r['param_val'])]['util_succ'].append(r['UtilityScore_succ'])
    sens_agg[(r['dag'], r['param_key'], r['param_val'])]['apto'].append(r['app_timeout_rate'])
    sens_agg[(r['dag'], r['param_key'], r['param_val'])]['dall'].append(r['delay_all'])
    sens_agg[(r['dag'], r['param_key'], r['param_val'])]['dsucc'].append(r['delay_succ'])

sens_sum_rows = []
for pk in SENS_PARAM_ORDER:
    keys = sorted([k for k in sens_agg if k[1] == pk],
                  key=lambda k: ({'chain':0,'default':1,'wide':2}.get(k[0], 99),
                                 float(k[2]) if k[2].lstrip('-').replace('.', '').isdigit() else 0))
    for (dag, p, v) in keys:
        d = sens_agg[(dag, p, v)]
        um, us, n = _stats(d['util_all'])
        usm, uss, _ = _stats(d['util_succ'])
        am, as_, _ = _stats(d['apto'])
        dam, das, _ = _stats(d['dall'])
        dsm, dss, _ = _stats(d['dsucc'])
        try:
            is_def = abs(SENS_DEFAULTS.get(pk, -999) - float(v)) < 1e-9
        except Exception:
            is_def = False
        sens_sum_rows.append({
            'dag': dag,
            'reviewer': SENS_REVIEWER.get(pk, ''), 'param_key': pk, 'param_val': v,
            'is_default': is_def, 'n_seeds': n,
            'AppTO_mean_pct': am * 100 if am is not None else None,
            'AppTO_std_pct':  as_ * 100 if as_ is not None else None,
            'D_all_mean': dam, 'D_all_std': das,
            'D_succ_mean': dsm, 'D_succ_std': dss,
            'Util_all_mean': um, 'Util_all_std': us,
            'Util_succ_mean': usm, 'Util_succ_std': uss,
        })
sens_sum_cols = ['dag', 'reviewer', 'param_key', 'param_val', 'is_default', 'n_seeds',
                 'AppTO_mean_pct', 'AppTO_std_pct',
                 'D_all_mean', 'D_all_std', 'D_succ_mean', 'D_succ_std',
                 'Util_all_mean', 'Util_all_std', 'Util_succ_mean', 'Util_succ_std']
with open(os.path.join(OUT_DIR, 'sensitivity_summary.csv'), 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=sens_sum_cols, extrasaction='ignore')
    w.writeheader()
    w.writerows(sens_sum_rows)
print(f'[write] sensitivity_summary.csv  {len(sens_sum_rows)} rows')


# =====================================================================
# 9. CSV: seedwise_variance.csv
#    main3 per-seed across 3 DAG x {1,3,5,7,42} (5-seed 主表协议)
#    Show per-seed Util_all + AppTO + D_all, plus per-DAG mean/std/max-min
# =====================================================================
var_rows = []
for dag in DAGS:
    for s in SEEDS:
        key = (dag, 'main3', s)
        if key not in best:
            continue
        r = best[key]
        var_rows.append({
            'algo': 'main3', 'label': 'CPGAPPO', 'dag': dag, 'seed': s,
            'in_paper_seeds': 'True',
            'app_timeout_rate': r['app_timeout_rate'],
            'delay_all': r['delay_all'], 'delay_succ': r['delay_succ'],
            'UtilityScore_all': r['UtilityScore_all'],
            'UtilityScore_succ': r['UtilityScore_succ'],
        })

# add per-DAG aggregate rows (5-seed 主表)
for dag in DAGS:
    paper_cells = [r for r in var_rows if r['dag'] == dag and r['in_paper_seeds'] in ('True', True)]
    if not paper_cells:
        continue
    um, us, _ = _stats([r['UtilityScore_all'] for r in paper_cells])
    am, as_, _ = _stats([r['app_timeout_rate'] for r in paper_cells])
    dm, ds, _ = _stats([r['delay_all'] for r in paper_cells])
    uvals = [r['UtilityScore_all'] for r in paper_cells]
    var_rows.append({
        'algo': 'main3', 'label': 'CPGAPPO', 'dag': dag, 'seed': 'MEAN(1,3,5,7,42)',
        'in_paper_seeds': '',
        'app_timeout_rate': am, 'delay_all': dm, 'delay_succ': None,
        'UtilityScore_all': um, 'UtilityScore_succ': None,
        '_Util_std': us, '_Util_maxmin': (max(uvals) - min(uvals)) if len(uvals) >= 2 else None,
        '_AppTO_std': as_, '_D_all_std': ds,
    })

var_cols = ['algo', 'label', 'dag', 'seed', 'in_paper_seeds',
            'app_timeout_rate', 'delay_all', 'delay_succ',
            'UtilityScore_all', 'UtilityScore_succ',
            '_Util_std', '_Util_maxmin', '_AppTO_std', '_D_all_std']
with open(os.path.join(OUT_DIR, 'seedwise_variance.csv'), 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=var_cols, extrasaction='ignore')
    w.writeheader()
    w.writerows(var_rows)
print(f'[write] seedwise_variance.csv  {len(var_rows)} rows')


# =====================================================================
# 10. CSV: data_gap_summary.csv  (cells MISSING D_all among main-table seeds {1,3,5,7,42})
# =====================================================================
gap_rows = []
for algo in sorted(set(ALGO_LABEL) | set(ALGO_KIND)):
    for dag in DAGS:
        for s in SEEDS:
            key = (dag, algo, s)
            if key not in best:
                gap_rows.append({
                    'dag': dag, 'algo': algo, 'label': ALGO_LABEL.get(algo, algo),
                    'seed': s, 'gap_type': 'NO_RESULT',
                    'has_reeval': False, 'D_all_src': '',
                    'detail': 'cell从未跑过 (no result.json)',
                })
                continue
            r = best[key]
            if r['delay_all'] is None:
                gap_rows.append({
                    'dag': dag, 'algo': algo, 'label': ALGO_LABEL.get(algo, algo),
                    'seed': s, 'gap_type': 'NO_D_ALL',
                    'has_reeval': r['has_reeval'],
                    'D_all_src': r['D_all_src'],
                    'detail': 'result.json无D_all, 且无result_reeval.json补齐',
                })

# also: sensitivity gaps (any param/val/dag missing any of 5 seeds)
sens_have = defaultdict(set)  # (dag, pk) -> set of vals
for (dag, pk, val, seed) in sens_best:
    sens_have[(dag, pk)].add(val)
# expected vals per param (from observed, union across dags)
SENS_EXPECTED = {}
for pk in SENS_PARAM_ORDER:
    all_vals = set()
    for (d, p) in sens_have:
        if p == pk:
            all_vals |= sens_have[(d, p)]
    SENS_EXPECTED[pk] = sorted(all_vals, key=lambda v: float(v) if v.lstrip('-').replace('.', '').isdigit() else 0)
# default DAG 已经全部跑过, 跨拓扑补跑只关心 chain+wide
_XTOPO_DAGS = ['chain', 'wide']
for pk in SENS_PARAM_ORDER:
    # 只对 R1-6 跨拓扑的 3 个参数报缺口 (lambda_g/shield_high/shield_norm)
    if pk not in ('LAMBDA_GUIDE', 'SHIELD_THR_HIGH', 'SHIELD_THR_NORM'):
        continue
    for dag in _XTOPO_DAGS:
        for val in SENS_EXPECTED[pk]:
            for s in SEEDS:
                if (dag, pk, val, s) not in sens_best:
                    gap_rows.append({
                        'dag': dag, 'algo': f'sens_{pk}', 'label': pk,
                        'seed': s, 'gap_type': 'SENS_MISSING_SEED',
                        'has_reeval': False, 'D_all_src': '',
                        'detail': f'跨拓扑敏感性 {pk}={val} 在 {dag} 上缺 seed {s}',
                    })

gap_cols = ['dag', 'algo', 'label', 'seed', 'gap_type', 'has_reeval', 'D_all_src', 'detail']
with open(os.path.join(OUT_DIR, 'data_gap_summary.csv'), 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=gap_cols, extrasaction='ignore')
    w.writeheader()
    w.writerows(gap_rows)
print(f'[write] data_gap_summary.csv  {len(gap_rows)} rows')


# =====================================================================
# 11. CSV: ablation_ordering_check.csv  (消融排序检查)
#    For Table V: per-DAG, list AppTO + Util_all for main3 vs each ablation,
#    flag if any ablation unexpectedly beats main3 on AppTO (R1-2 concern)
# =====================================================================
order_rows = []
for dag in DAGS:
    m3_cells = [best[(dag, 'main3', s)] for s in SEEDS if (dag, 'main3', s) in best]
    m3_apto_m, _, _ = _stats([c['app_timeout_rate'] for c in m3_cells])
    m3_uall_m, _, _ = _stats([c['UtilityScore_all'] for c in m3_cells])
    for algo in TABLE_V_ALGOS:
        cells = [best[(dag, algo, s)] for s in SEEDS if (dag, algo, s) in best]
        if not cells:
            continue
        am, _, n = _stats([c['app_timeout_rate'] for c in cells])
        um, _, _ = _stats([c['UtilityScore_all'] for c in cells])
        flag = ''
        if am is not None and m3_apto_m is not None and am < m3_apto_m - 0.005 and algo != 'main3':
            flag = '⚠ 消融AppTO低于main3 (R1-2关注点)'
        if um is not None and m3_uall_m is not None and um > m3_uall_m + 0.005 and algo != 'main3':
            flag = (flag + '; ' if flag else '') + '⚠ 消融Util高于main3'
        order_rows.append({
            'dag': dag, 'algo': algo, 'label': ALGO_LABEL.get(algo, algo), 'n': n,
            'AppTO_mean_pct': am * 100 if am is not None else None,
            'Util_all_mean': um,
            'main3_AppTO_mean_pct': m3_apto_m * 100 if m3_apto_m is not None else None,
            'main3_Util_all_mean': m3_uall_m,
            'delta_AppTO_pct': (am - m3_apto_m) * 100 if am is not None and m3_apto_m is not None else None,
            'delta_Util_all': (um - m3_uall_m) if um is not None and m3_uall_m is not None else None,
            'flag': flag,
        })
order_cols = ['dag', 'algo', 'label', 'n', 'AppTO_mean_pct', 'Util_all_mean',
              'main3_AppTO_mean_pct', 'main3_Util_all_mean',
              'delta_AppTO_pct', 'delta_Util_all', 'flag']
with open(os.path.join(OUT_DIR, 'ablation_ordering_check.csv'), 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=order_cols, extrasaction='ignore')
    w.writeheader()
    w.writerows(order_rows)
print(f'[write] ablation_ordering_check.csv  {len(order_rows)} rows')


# =====================================================================
# 12. README.md
# =====================================================================
n_main_total = sum(1 for a in TABLE_IV_ALGOS for d in DAGS for s in SEEDS)
n_main_have  = sum(1 for a in TABLE_IV_ALGOS for d in DAGS for s in SEEDS if (d, a, s) in best and best[(d, a, s)]['delay_all'] is not None)
n_abl_total  = sum(1 for a in TABLE_V_ALGOS  for d in DAGS for s in SEEDS)
n_abl_have   = sum(1 for a in TABLE_V_ALGOS  for d in DAGS for s in SEEDS if (d, a, s) in best and best[(d, a, s)]['delay_all'] is not None)
# 敏感性: default DAG 8 参数全覆盖, 跨拓扑只补 R1-6 点名的 3 参数 (lambda_g/shield_high/shield_norm)
n_sens_default_total = 8 * 5 * len(SEEDS)  # 8 参数 × 5 取值 × 5 seeds, 只算 default
n_sens_default_have  = sum(1 for (dag,pk,val,seed) in sens_best if dag == 'default')
n_sens_xtopo_total   = 3 * 5 * len(SEEDS)   # 3 参数 × 5 取值 × 2 DAG(chain+wide) × 5 seeds
n_sens_xtopo_have    = sum(1 for (dag,pk,val,seed) in sens_best if dag in ('chain','wide')
                           and pk in ('LAMBDA_GUIDE','SHIELD_THR_HIGH','SHIELD_THR_NORM'))

gap_main = [(d, a, s) for a in TABLE_IV_ALGOS for d in DAGS for s in SEEDS
            if (d, a, s) not in best or best[(d, a, s)]['delay_all'] is None]
gap_abl  = [(d, a, s) for a in TABLE_V_ALGOS  for d in DAGS for s in SEEDS
            if (d, a, s) not in best or best[(d, a, s)]['delay_all'] is None]

readme = f'''# 第三次大修实验数据集 (修正版)

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**输出目录**: `collected_csv/` (仓库内)

---

## 数据来源 (已跑完的实验, 仅收集不重跑)

| 来源 | 路径 | 内容 |
|---|---|---|
| 主对比 + 消融 (新) | `results/main_comparison_20260824_203442/` | CPGAPPO + 5 外部RL + 5 启发 + 7 消融, 3 DAG × 5 seeds, **result.json 自带真 D_all** |
| 敏感性 | `results/main3_sens_*/` | 8 参数 × 5 取值 × 5 seeds, result.json 自带 D_all |

> **注**: `results/ablation_classic_*` 等旧批次中的 `main` / `main_v2..v5` 是
> CPGAPPO 早期变体, 已从本数据集排除 (论文不使用)。原文件保留在
> `results/` 不删, 只是本输出目录不收录。

---

## CSV 文件清单

| 文件 | 内容 |
|---|---|
| `master_seedwise.csv` | 所有 cell 的逐 seed 原始数据 (含 D_all 来源标记) |
| `table_IV_main_comparison_5seeds.csv` | 11 算法 × 3 DAG × {{1,3,5,7,42}}, mean±std, 含 **D_all** 列 (超时按 deadline 计费) |
| `table_V_ablation_5seeds.csv` | 7 算法 (main3 + 6 消融, 已剔除 alloff 与 main3 重复) × 3 DAG × {{1,3,5,7,42}}, mean±std, 含 D_all |
| `ablation_ordering_check.csv` | 每个 cell 的 AppTO/Util 与 main3 的差, 自动 flag 消融反超 main3 的情况 |
| `sensitivity.csv` | 8 参数 × 5 取值 × 5 seeds, 逐 seed |
| `sensitivity_summary.csv` | 同上, 聚合为 mean±std |
| `seedwise_variance.csv` | main3 逐 seed (5 seeds {{1,3,5,7,42}}), 含 per-DAG mean/std |
| `data_gap_summary.csv` | **数据缺口总览** | 列出所有缺 D_all / 缺 cell / 缺 seed 的项 |
| `raw_result_json/` | 全部 | 原始 result.json (按算法分子目录), 含 `*_reeval.json` 备份 |

---

## 统计口径 (R1-5)

- **D_all** = `ts.get_avg_results(only_successful=False, timeout_charge="deadline")`
  - 超时应用按 per-app deadline 预算 `get_app_deadline_slot(uid) * slot_interval` 计入 (即 TD_i^max - TS_i)
  - 旧版用 `timeout_charge="2x_deadline"` (超时按 deadline×2 计), **已废弃**
- **D_succ** = 仅成功 (非超时) 应用的平均 delay (旧 Table IV 口径)
- **UtilityScore_all** = `compute_utility_score(e_all, d_all, rho_app, rho_task, w_cost=0.25)`
  - 用 D_all (R1-5 口径) 算, 是论文 Table IV/V 应用的效用值
- **主表种子 = {{1, 3, 5, 7, 42}}** (5-seed 协议): 所有 17 算法都有这 5 个种子的真 D_all

---

## 数据完成度 (一眼看还差什么)

### 主对比 (Table IV, R1-5): {n_main_have}/{n_main_total} cells 有 D_all
{'完整' if n_main_have == n_main_total else '⚠ 缺 ' + str(n_main_total - n_main_have) + ' 个 cell'}
{('缺: ' + ', '.join(f"{d}/{a}/s{s}" for d,a,s in gap_main)) if gap_main else ''}

### 消融 (Table V, R1-1/2/3): {n_abl_have}/{n_abl_total} cells 有 D_all
{'完整' if n_abl_have == n_abl_total else '⚠ 缺 ' + str(n_abl_total - n_abl_have) + ' 个 cell (见 data_gap_summary.csv)'}
{('缺: ' + ', '.join(f"{d}/{a}/s{s}" for d,a,s in gap_abl)) if gap_abl else ''}

### 敏感性 default (R1-6/R4-1): {n_sens_default_have}/{n_sens_default_total} cells (8 参数 × 5 取值 × 5 seeds, default DAG)
{'完整' if n_sens_default_have >= n_sens_default_total else '⚠ 缺 ' + str(n_sens_default_total - n_sens_default_have) + ' 个 cell'}

### 敏感性 跨拓扑 (R1-6): {n_sens_xtopo_have}/{n_sens_xtopo_total} cells (3 参数 × 5 取值 × 2 DAG × 5 seeds, chain+wide)
{'完整' if n_sens_xtopo_have >= n_sens_xtopo_total else '⚠ 缺 ' + str(n_sens_xtopo_total - n_sens_xtopo_have) + ' 个 cell (跑 _supplement1/2/3 后补齐)'}

### 方差 (R3-1/R4-2): main3 在 3 DAG × {{1,3,5,7,42}} 全有 D_all
{'完整' if all((d,'main3',s) in best and best[(d,'main3',s)]['delay_all'] is not None for d in DAGS for s in SEEDS) else '⚠ 有缺'}

---

## 当前已知缺口 (需补跑才能填, ckpt 都在)

{('以下 cell 的 D_all 为空, 需用 `_reeval_lib.reeval_one(variant, dag, seed)` 补 R1-5 reeval:' if gap_abl else '无缺口')}
'''

for d, a, s in gap_abl:
    rr_exists = best.get((d, a, s), {}).get('has_reeval', False)
    readme += f'\n- `{d}/{a}/seed{s}` (label={ALGO_LABEL.get(a,a)}) — result_reeval.json={"有但D_all仍空" if rr_exists else "无"}, ckpt 在 `ablation_classic_{d}_20260608_1141xx/{d}__{a}__seed{s}/checkpoints/`'

readme += f'''

---

## 17 算法清单 (已剔除 CPGAPPO 早期变体 main / main_v2..v5)

| label | algo | 类别 | Table IV | Table V |
|---|---|---|---|---|
| CPGAPPO | main3 | 本文方法 | ✓ | ✓ |
| noguidece | noguidece | 消融 (w/o Guide-CE) | | ✓ |
| noshield | noshield | 消融 (w/o Shield) | | ✓ |
| noappcredit | noappcredit | 消融 (w/o App Credit) | | ✓ |
| nocp | nocp | 消融 (w/o CP Sequencing) | | ✓ |
| fwdonly | fwdonly | 消融 (w/o All = forward-only) | | ✓ |
| alloff | alloff | 消融 (w/o All = all off) | | ✓ |
| DGMA_adapt | DGMA_adapt | 外部 RL | ✓ | |
| DGMA_paper | DGMA_paper | 外部 RL | ✓ | |
| TransEdgeStyle | TransEdgeStyle | 外部 RL | ✓ | |
| GATDQN | GATDQN | 外部 RL | ✓ | |
| PPO | PPO | 外部 RL (PPO backbone) | ✓ | |
| HybridPSOGA | HybridPSOGA | 启发式 | ✓ | |
| Genetic | Genetic | 启发式 | ✓ | |
| Greedy | Greedy | 启发式 | ✓ | |
| Edge-only | Edge-only | 启发式 | ✓ | |
| Local-only | Local-only | 启发式 | ✓ | |

---

## 8 个敏感性参数

| 参数 | 默认值 | 取值 |
|---|---|---|
| LAMBDA_GUIDE | 0.1 | 0, 0.05, 0.1, 0.2, 0.4 |
| BONUS_TO | -7 | -14, -10, -7, -4, -2 |
| W_E | 0.05 | 0, 0.025, 0.05, 0.1, 0.2 |
| W_D | 0.1 | 0, 0.05, 0.1, 0.2, 0.4 |
| SHIELD_THR_HIGH | 1.10 | 1.00, 1.05, 1.10, 1.15, 1.20 |
| SHIELD_THR_NORM | 1.15 | 1.05, 1.10, 1.15, 1.20, 1.30 |
| OVERFLOW_MULT | 1.25 | 1.00, 1.10, 1.25, 1.50, 2.00 |
| CP_REM_COEFF | 0.45 | 0.20, 0.30, 0.45, 0.60, 0.80 |

---

## 说明
- CPGAPPO (内部代号 main3) 已完成 3 DAG × 5 seeds (1,3,5,7,42), 论文用 {{1,3,5,7,42}} 5 种子做主对比 (R4-2 协议)
- 早期变体 main / main_v2 / main_v3 / main_v4 / main_v5 已从本数据集剔除, 原始结果仍保留在 `results/` 不删
- 敏感性 8 参数全覆盖 5 seeds, 可直接做 OFAT 图 (R1-6/R4-1)
- DGMA_paper 在所有 DAG × seed 上 AppTO≈50-100% (训练坍塌到全本地路由), 是该 baseline 在本环境下的真实表现
- D_all 可能小于 D_succ: 当超时 app 的 deadline 预算 < 成功 app 的平均 delay 时, R1-5 口径下 D_all < D_succ (正确行为)
'''

with open(os.path.join(OUT_DIR, 'README.md'), 'w', encoding='utf-8') as f:
    f.write(readme)
print(f'[write] README.md')

print(f'\n[DONE] all files -> {OUT_DIR}')
print(f'  主对比 D_all: {n_main_have}/{n_main_total}')
print(f'  消融   D_all: {n_abl_have}/{n_abl_total}')
print(f'  敏感性 default: {n_sens_default_have}/{n_sens_default_total}')
print(f'  敏感性 跨拓扑: {n_sens_xtopo_have}/{n_sens_xtopo_total}  (chain+wide)')
