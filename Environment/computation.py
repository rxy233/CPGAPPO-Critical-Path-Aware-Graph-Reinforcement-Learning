# -*- coding: utf-8 -*-
"""
Physical computation models: execution time/energy and upload time/energy.

English
-------
computation.py provides the closed-form models used everywhere in the repo:
  - execute_consumption(task_size, freq, complexity, node_type): returns
    (energy, delay) for running a subtask of given byte size and complexity
    on a device of frequency `freq` (local 'l', edge 'e', cloud 'c').
  - upload_consumption(...): returns (energy, delay) for transferring a
    subtask over a link of given distance/bandwidth, using the Shannon-based
    rate formula in para (local_trans, theta, yita2). Used by both the
    guide-score functions and the env step.

All constants are read from `para` (utils/constant.py) so the physics are
identical across algorithms (R4-2).

中文
----
执行/上传的物理模型: execute_consumption 与 upload_consumption, 基于香农速率公式,
常数来自 para, 所有算法共用同一份物理参数 (R4-2)。
"""
import math
import numpy as np
from utils.constant import para


# def upload_consumption(data, type="e"):
#     energy, delay = 0, 0
#     if type == "e":
#         data_size, distance, bw = data
#         new_bw = bw * (10 ** 6)
#
#         upload_speed = new_bw * math.log2(1 + (para["local_trans"] * (distance[0] ** -para["theta"]) * 1 / para["yita2"]))#目前distance固定为最近的
#         delay = (data_size * 8) / upload_speed  # 修复Bytes -> bits
#         energy = para["local_trans"] * delay
#     if type == "c":
#         data_size = data
#         delay = data_size / para["R_ec"]
#     # energy = para["local_trans"] * delay
#     return energy, delay

def upload_consumption(data, target_load_num, type="e", local_trans=None, local_wait=None): #上传至云端时先要经过边缘端，而边缘端上传至云端时经有线传输
    """
    【新模型】三独立耗能单元模型：
    - 基础平台能耗（维持待机）：始终存在
    - 射频模块能耗（上传）：只在传输时产生
    - CPU模块能耗：计算时单独计算

    传输能耗 = 基础平台能耗 + 射频模块能耗

    local_trans 和 local_wait 参数：支持设备特定的能耗参数
    """
    energy, delay = 0, 0
    threshold_e = 0.5
    threshold_c = 0.8

    # 使用设备特定的参数，否则使用全局默认值
    local_trans_val = local_trans if local_trans is not None else para["local_trans"]
    local_wait_val = local_wait if local_wait is not None else para["local_wait"]

    if type == "e":#本地→边缘
        data_size, distance, bw = data
        new_bw = bw * (10 ** 6) #从mbps转换bps

        upload_speed = new_bw * math.log2(
            1 + (local_trans_val * (distance ** -para["theta"]) * 1 / para["yita2"]))

        # # 考虑目标边缘节点负载（同时运行核心数）对上传速度的影响（根据场景自定义），#由于目前未实现边缘按量计算，暂不考虑
        # current_load_rate = target_load_num / para["edgecore_limit"] #预计改为根据边缘其他负载计算
        # if current_load_rate > threshold_e: #边缘负载达到阈值，终端上传至边缘端速度降低
        #     speed_ratio = 1 / (1 + (para["upload_alpha"] * (current_load_rate - threshold_e)) ** 2)
        #     upload_speed *= speed_ratio



        delay = (data_size * 8) / upload_speed  # 修复Bytes -> bits

        # 【新模型】传输能耗 = 基础平台能耗 + 射频模块能耗
        # 基础平台始终运行，射频模块仅在传输时工作，两者线性叠加
        base_platform_energy = local_wait_val * delay  # 基础平台能耗（待机）
        rf_module_energy = local_trans_val * delay      # 射频模块能耗（上传）
        energy = base_platform_energy + rf_module_energy

    elif type == "c":#边缘→云
        data_size = data
        upload_speed = para["R_ec"]
        current_load_rate = target_load_num / para["edgecore_limit"] #预计改为根据边缘节点内存负载计算

        # if current_load_rate > threshold_c:  # 边缘负载达到阈值，上传至云的速度降低#由于目前未实现边缘按量计算，暂不考虑
        #     speed_ratio = 1 / (1 + (para["upload_beta"] * (current_load_rate - threshold_c)) ** 2)
        #     upload_speed *= speed_ratio

        delay = (data_size * 8) / upload_speed  # 修复Bytes -> bits
        energy = 0  #边缘上传至云的传输不耗用户终端能量

    return energy, delay


# def execute_consumption(data_size, f, task_complex_index, type):
#     task_complex = para["task_complex"][task_complex_index]
#     delay = data_size * task_complex / f
#     # if type == "l":
#     #     energy = para["kappa"] * (f ** 2) * task_complex * data_size
#
#     energy = para["kappa"] * (f ** 2) * task_complex * data_size
#
#     # else:
#     #     energy = 0
#     #     # energy = para["local_wait"] * delay
#     return energy, delay

def execute_consumption(data_size, f, task_complex_index, type="l", local_wait=None): #暂只考虑本地设备能耗，云与边运行能耗暂不考虑
    """
    【新模型】三独立耗能单元模型：
    - 基础平台能耗（维持待机）：始终存在
    - CPU模块能耗（计算）：只在计算时产生
    - 射频模块能耗：传输时单独计算

    计算能耗 = 基础平台能耗 + CPU模块能耗
    """
    task_complex = para["task_complex"][task_complex_index]
    delay = data_size * task_complex / f
    # 使用设备特定的参数，否则使用全局默认值
    local_wait_val = local_wait if local_wait is not None else para["local_wait"]

    if type == "l":
        # 【DVFS 立方模型】P_cpu = κ·f³, E_local = (P_idle + P_cpu)·T_local
        # 与论文一致，替代旧的 18·(f/1e9)^1.5 经验公式
        kappa = para.get("kappa", 2.2e-27)
        local_cpu_power = kappa * (f ** 3)  # CPU动态功率 (W)
        energy = (local_wait_val + local_cpu_power) * delay

    else:
        energy = 0
        # energy = local_wait_val * delay
    # 防止返回 nan/inf
    if np.isnan(energy) or np.isinf(energy):
        energy = 0.0
    if np.isnan(delay) or np.isinf(delay):
        delay = 0.0
    return energy, delay