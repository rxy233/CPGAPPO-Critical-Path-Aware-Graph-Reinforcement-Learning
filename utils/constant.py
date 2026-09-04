# -*- coding: utf-8 -*-
"""
Global parameter dictionary `para` and energy-model constants.

English
-------
constant.py holds the single source of truth for every physical / topology /
energy parameter in the repo, in the dict `para`:
  - Topology: user_num=150, edge_num=8, cloud, slot_interval=0.01s,
    edge_radius, uplink/downlink ranges.
  - Workload: task_complex distribution, deadline_slot=55, burst params.
  - Energy model: three independent units (base platform local_wait,
    CPU module, radio module) — see the energy-model block below.
  - Scheduler: top-K, scoring weights.
All algorithms import `para` from here, so changing a value once updates
every algorithm identically (R4-2 fair-comparison protocol). The Python
`random.seed(42)` here is cosmetic; the real per-seed RNG is driven by
CONFIG["SEED"] in Experiments_new/exp_utils.py.

中文
----
全局参数字典 `para`: 拓扑/工作负载/能耗模型/调度参数的唯一来源, 所有算法都从这里
import, 改一处全算法同步 (R4-2)。random.seed(42) 仅装饰用, 真正的 per-seed RNG
由 exp_utils.py 的 CONFIG["SEED"] 控制。
"""
import random

# 为了可重复性，设置固定seed（可选）
random.seed(42)#没用，要改改 exp_utils.py L32 的 CONFIG["SEED"]。

# ==============================================================================
# 【能耗模型说明】三独立耗能单元模型
# ==============================================================================
# 设备能耗分解为三个独立的耗能单元：
# 1. 基础平台能耗（Base Platform Energy）
#    - 功能：维持设备待机运行
#    - 参数：local_wait = 0.05 W
#    - 特点：始终存在（无论设备是计算、传输还是等待）
#
# 2. CPU模块能耗（CPU Module Energy）
#    - 功能：执行计算任务
#    - 参数：动态功率 = 0.04 * (f * 0.1^8)^1.5 W
#    - 特点：仅在计算时产生，与计算频率和时间成正比
#
# 3. 射频模块能耗（RF Module Energy）
#    - 功能：无线数据上传
#    - 参数：local_trans = 1.5 W
#    - 特点：仅在传输时产生，与传输时间成正比
#
# 【线性叠加假设】
# CPU和射频模块供电独立，同时工作时互不干扰
# 总能耗 = 基础平台能耗 + CPU模块能耗 + 射频模块能耗
#
# 【计算公式】
# - 计算能耗 = local_wait * t_compute + cpu_power * t_compute
# - 传输能耗 = local_wait * t_transmit + local_trans * t_transmit
# - 等待能耗 = local_wait * t_wait
#
# ==============================================================================

para = {
    "matrix_root": r"./matrix/",
    "matrix_file_name": ["matrix_file05.txt", "matrix_file10.txt", "matrix_file15.txt",
                         "matrix_file20.txt", "matrix_file25.txt", "matrix_file30.txt"],

    "task_num": [5, 10, 15, 20, 25, 30],
    "T_max": 5,

    # ==================== 1. 边缘节点配置（稍微增强，避免长期崩溃）====================
    "edge_num": 8,  # [关键修改] 从4增加到8
    # "edge_power": [4.2e+09, 4.2e+09, 4.2e+09, 4.2e+09, 4.2e+09, 4.2e+09, 4.2e+09, 4.2e+09] * 8,  # [压力微调] 从 2.0GHz 提到 2.5GHz，加速队列清空
    "edge_power": [4.5 * (10 ** 9)] * 8,
    "edgecore_limit": 4,  # [压力调整] 从3降到2，甚至1。加剧排队！
    "edge_radius": 240,  # [修改] 覆盖半径增加

    # 中心聚集布局（500×500 区域）
    # [关键修改] 所有8个节点紧密聚集在中心区域（200-300范围）
    # 通过位置重叠制造排队压力，而不改变核心数（保持同构）
    # 大部分用户都会选择最近的节点（中心），导致严重拥堵
    # "edge_positions": [
    #     # 核心密集区（8个节点全部聚集，制造排队瓶颈）
    #     (200, 200), (200, 220), (200, 240), (200, 260),
    #     (220, 200), (240, 200), (260, 200), (220, 220)
    # ],

    # # 边缘节点分散分布
    # "edge_positions": [
    #     (250, 250), (270, 230), (230, 270),  # 核心热点
    #     (400, 200), (420, 280),               # 东部
    #     (100, 250), (80, 180),                # 西部
    #     (250, 400)                            # 南部
    # ],

    "edge_positions": [
        # 均匀分布在地图的 9 个宫格关键点（去掉中心一个或微调）
        (100, 100), (250, 100), (400, 100),
        (100, 250),             (400, 250), # 中间留空给用户
        (100, 400), (250, 400), (400, 400)
    ],

    # "edge_positions": [#表现好
    #     # 聚落 A (左上角，服务该区域用户)
    #     (100, 100), (100, 120), (120, 100), (120, 120),
    #     # 聚落 B (右下角，服务该区域用户)
    #     (380, 380), (380, 400), (400, 380), (400, 400)
    # ],

    # "edge_positions": [
    #     # 像围墙一样包围用户区
    #     (50, 250),  (450, 250), # 极左和极右
    #     (250, 50),  (250, 450), # 极上和极下
    #     (100, 100), (400, 400), # 对角线
    #     (100, 400), (400, 100)  # 对角线
    # ],


    "edge_load_weights": [0.5] * 8,  # 负载权重（8个节点）
    "max_storage": 1000000,

    # ==================== 用户配置 ====================
    "user_num": 150,  # [修改] 从100增加到150，保持人均资源紧张度
    "device_task": 1,

    # ==================== 3. 网络配置（适度提高带宽，让卸载更有竞争力）====================
    # [压力微调] 上行带宽从 4-10 Mbps 提高到 6-12 Mbps。
    # 对于 1MB 的任务，6Mbps 需要传输 1.33秒，12Mbps 需要 0.67秒。
    # 让卸载更快，给RL算法更多优化空间。
    "uplink_range": [20, 25, 30],
    # 原因：4-10 Mbps 传输太慢，导致卸载任务也超时
    # 提到 6-12 Mbps 后，传输时间 0.67-1.33秒，与计算时间更平衡
    "downlink_range": [6 * i for i in range(5, 6)],
    "upload_alpha": 1,
    "upload_beta": 1.0,
    "corespeed_alpha": 1,

    # ==================== 4. 任务配置（增大任务 & 提高复杂度）====================
    # 【目标调整】任务大小范围：200KB - 900KB (减小最大值)
    # 原因：
    # 1. 旧范围 200KB-1200KB，大部分任务（800KB+）上传时间 > 2s
    # 2. 即使 deadline=2.2s，也无法在剩余时间内完成排队+计算
    # 3. 减小到 900KB，让更多任务可以在 deadline 内完成
    # 计算：900KB / 1024 = 0.88 MB。在 4Mbps 带宽下传输耗时约 1.8秒
    "task_size_range": [i * 1024 for i in range(100, 401, 50)],

    "downlink_size": [i * (2 ** 10) * 8 for i in range(20, 31, 2)],
    # [压力调整] 复杂度适度增加：从 30-60 提升到 45-80。
    # 适度增加压力，避免过度苛刻
    "task_complex": [i for i in range(320, 621, 50)],
    # 本地算力范围：2.5-7.0 GHz
    # Local 策略会保留一定超时（50-70%），但不会崩溃，给 RL 算法留出可优化空间
    # 从随机列表改为范围，避免导入时随机
    "local_power_range": (2500000000.0, 7000000000.0),  # 2.5-7.0 GHz

    # 【新增】local_wait 和 local_trans 范围，用于相关采样
    # "local_wait_range": (0.06, 0.18),  # W，基础平台能耗（待机）#暂时没使用
    # "local_trans_range": (1.3, 2.3),  # W，射频模块能耗（上传）#暂时没使用
    # 保留固定值作为默认值，但会被 generate_components 的相关采样覆盖
    "local_trans": 1.5,
    "local_wait": 0.05,  # [调整] 从 0.05 提高到 0.1，增加 Local 等待能耗
    "task_storage_index": 1,

    # ==================== 云配置 ====================
    "R_ec": 2 * (10 ** 7),
    "cloud_power": 7 * (10 ** 9),  # 7 GHz，使云端任务保留约 10-15% 的超时率
    "cloud_wan_rtt": 0.0,  # [新增] 云端 WAN RTT 0.2秒，增加传输开销

    # ==================== 5. 时间配置（适度收紧，保持挑战性）====================
    "kappa": 2 * 10 ** -27,
    "slot_interval": 0.01,  # 1 slot = 10ms
    # 【目标调整】Deadline 从 180 (1.8s) 收紧到 130 (1.3s)
    # 目标：AppTO = 10-30%（让Local有一定超时，但不会崩溃）
    # 原因：
    # 1. 之前超时率偏低（0-20%），需要增加压力
    # 2. 130s (1.3s) 适度收紧，保持合理挑战性
    # 3. 配合边缘节点聚集的排队压力，制造调度价值
    "deadline_slot": 55,  # v2: stricter deadline
    "app_deadline_slack_factor": 0.05,  # v2: tighter slack
    "app_deadline_alpha": 0.3,  # v2: tighter alpha
    "tight_app_slack_factor": 0.0,  # 紧deadline用户额外余量因子
    "result_size": 1,

    # ==================== 信道配置 ====================
    "H_0": 10 ** -6,
    "yita2": 10 ** -9,
    "g0": -40,
    "theta": 2.5,
    "N0": -174,
    "e_ratio": 0.5
}

# 【诊断】打印 task_size_range 的实际值
task_size_list = list(para['task_size_range'])
print(f"[CONSTANT INIT] task_size_range: 前5个={task_size_list[:5]} (单位: Bytes), "
      f"范围: {task_size_list[0]}-{task_size_list[-1]} Bytes "
      f"({task_size_list[0] / 1024:.1f}-{task_size_list[-1] / 1024:.1f} KB), "
      f"数量: {len(task_size_list)} 个")

task_size = [
    [4096, 32768, 8192, 8192, 28672, 28672, 28672, 32768, 20480, 36864, 16384, 36864, 36864, 32768, 4096],
    [16384, 8192, 24576, 36864, 28672, 24576, 8192, 4096, 20480, 16384, 8192, 20480, 8192, 8192, 12288],
    [36864, 32768, 20480, 12288, 32768, 28672, 20480, 28672, 8192, 24576, 36864, 36864, 32768, 4096, 8192]
]

comm = {
    "bandwidth": 10,
    "noise": -43,
}

trans_power = {
    "AP": 1.0,
    "MD": 0.1  # [关键修复] 从0.3降低到0.1，鼓励卸载（降低传输功耗）
}

channel_gain = {
    "AP": 2.5 * 10 ** -7,
    "mm": 2.0 * 10 ** -7
}