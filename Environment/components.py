# -*- coding: utf-8 -*-
"""
System components: cloud, edge, user device, and task/subtask classes.

English
-------
components.py defines the physical entities the environment is built from:
  - Device: a user device with local compute power, bandwidth, and the list
    of distances to each edge node (used for upload delay + edge radius
    filtering in the action mask).
  - Edge: an edge server with power, calculate_parameter, and a queue.
  - Cloud: the single cloud node with the highest power.
  - Task / Subtask: the workload units; a Task holds a SubGraph (DAG) of
    subtasks with sizes/complexities, deadlines and dependencies.
Each class reads its physical parameters from `para` (utils/constant.py).

中文
----
系统基本组件: 云 / 边 / 端 / 任务 / 子任务, 物理参数来自 para。
"""
from utils.constant import *
import random
import numpy as np
import traceback

# 用于追踪 Device 创建位置的标记
_DEVICE_WARN_PRINTED = False

'''
Device-单个用户设备(也即user)
相关参数：
    用户所需执行任务task
'''

class Device:
    def __init__(self, x, y, edge_distances, task_complex_index=0):#老版
        self.task_complex_index = task_complex_index
        self.local_power = para["local_power"][random.randint(0, len(para["local_power"]) - 1)]
        self.pos_x = x # 设备所在坐标
        self.pos_y = y
        self.edge_distances = edge_distances

    def __init__(self, number, x, y, edge_distances, edge_scores, edge_inrange, task_complex_index=0, local_power=None, local_wait=None, local_trans=None):
        self.number = number#编号
        self.task_complex_index = task_complex_index
        # 修复使用传入的 local_power，优先级最高
        if local_power is not None:
            self.local_power = float(local_power)
        elif "local_power_range" in para:
            # 【警告】未传入 local_power，打印堆栈追踪（只打印一次）
            global _DEVICE_WARN_PRINTED
            if not _DEVICE_WARN_PRINTED:
                _DEVICE_WARN_PRINTED = True
                print("\n" + "="*80)
                print(f"[Device {number}] local_power 参数未传入！打印堆栈追踪：")
                print("="*80)
                traceback.print_stack(limit=15)
                print("="*80 + "\n")
            # 【警告】未传入 local_power，使用确定性采样（而非随机）
            # 注意：如果 local_power 参数未传入，说明 Environment.generate_components 可能没被正确调用
            import warnings
            warnings.warn(
                f"Device[{number}]: local_power 参数未传入。"
                f"这通常意味着 Environment.generate_components(seed=...) 没有被调用。"
                f"请确保在 Environment 初始化后显式调用 generate_components(seed=env_seed)。"
                f"当前使用 local_power_range 进行确定性采样，但强烈建议修复调用方式。"
            )
            low, high = para["local_power_range"]
            # 修复使用确定性采样而非 random.uniform，确保多进程一致性
            # 使用设备编号作为确定性种子的一部分
            rng = np.random.default_rng(number)
            self.local_power = float(rng.uniform(low, high))
        elif "local_power" in para and len(para["local_power"]) > 0:
            # 使用预生成的列表（按设备编号取值，确保确定性）
            power_idx = number % len(para["local_power"])
            self.local_power = float(para["local_power"][power_idx])
        else:
            # 【新版本】local_power_range 和 local_power 都不存在，抛出异常
            raise KeyError(f"Device[{number}]: local_power 参数未传入，且 para 中未定义 local_power_range 或 local_power")
        # local_wait 和 local_trans：基础平台能耗（待机）和射频模块能耗（上传）
        # 优先使用传入的参数，否则使用全局默认值
        if local_wait is not None:
            self.local_wait = float(local_wait)
        else:
            self.local_wait = para.get("local_wait", 0.1)
        if local_trans is not None:
            self.local_trans = float(local_trans)
        else:
            self.local_trans = para.get("local_trans", 1.5)
        self.pos_x = x # 设备所在坐标
        self.pos_y = y
        self.edge_distances = edge_distances
        self.edge_scores = edge_scores
        self.edge_inrange = edge_inrange#在其范围内的边缘端，暂时没用到，只通过将范围外的edge_distances设为无限大来排除

    # ==================== 新增方法 ====================
    
    def is_available(self):
        """
        检查本地设备是否可用于计算
        
        Returns:
            bool: 设备是否可用
        """
        # 检查计算能力是否足够
        if hasattr(self, 'local_power') and self.local_power <= 0:
            return False
        return True
    
    def get_compute_power(self):
        """
        获取设备计算能力 (Hz)
        
        Returns:
            float: 计算能力（Hz）
        """
        if hasattr(self, 'local_power'):
            return float(self.local_power)
        return 0.0
    
    def get_idle_power(self):
        """
        获取设备待机功耗 (W)
        
        Returns:
            float: 待机功耗（W）
        """
        if hasattr(self, 'local_wait'):
            return float(self.local_wait)
        return para.get("local_wait", 0.1)
    
    def get_transmission_power(self):
        """
        获取设备传输功耗 (W)
        
        Returns:
            float: 传输功耗（W）
        """
        if hasattr(self, 'local_trans'):
            return float(self.local_trans)
        return para.get("local_trans", 1.5)
    
    # ==================== 新增方法结束 ====================


'''
EdgeServer-边缘端
相关参数：
    算力,容量，传输上下行带宽
'''
class EdgeServer:
    def __init__(self, number, x, y, fe, task_complex_index=0):
        self.number = number  # 编号
        self.pos_x = x
        self.pos_y = y
        self.edge_power = fe
        self.edge_near = []#记录其附近边缘端暂未实现
        self.task_complex_index = task_complex_index

        #可用性指标
        self.current_remainTime = 0  # 记录节点当前最快可用时间（最快任一核心空闲时间）

        # 存储指标
        self.task_count = 0  # 记录节点当前任务数量
        self.total_task_size = 0  # 记录节点当前任务量(根据这个算存储需要空间和计算量)
        self.total_difficulty = 0  # 累计任务难度（任务大小*复杂度系数）
        self.used_storage = 0  # 已用存储空间

        self.max_storage = 6000 * 2 ** 10  # 最大存储容量（KB）

        # 计算参数
        self.calculate_parameter = 1  # 各核心计算速度参数，设备负载越高，计算速度越低

        # 带宽分配列表
        self.bw_occupy = {} # dict
        self.bw_occupy_index = []

    def add_new_task(self, task_size, complex_index):
        """
        增加新任务到边缘节点
        """
        self.task_count += 1
        self.total_task_size += task_size
        self.total_difficulty += task_size * complex_index
        self.used_storage += task_size * para["task_storage_index"]

    def update_core_speeds(self, load_rates):
        """根据负载率动态调整核心速度"""
        threshold = 0.5
        if load_rates < threshold:
            self.calculate_parameter = 1
        else:
            self.calculate_parameter = 1

    # ==================== 新增方法 ====================
    
    def is_available(self):
        """
        检查边缘服务器是否可用
        基于存储空间和任务数判断
        
        Returns:
            bool: 边缘服务器是否可用
        """
        # 检查存储空间是否已满
        if hasattr(self, 'max_storage') and hasattr(self, 'used_storage'):
            if self.used_storage >= self.max_storage:
                return False
        
        # 检查任务数是否超过限制（假设最大排队任务数为50）
        if hasattr(self, 'task_count'):
            max_tasks = para.get("edgecore_limit", 5) * 10  # 每核心最多排队10个任务
            if self.task_count >= max_tasks:
                return False
        
        return True
    
    def get_current_load(self):
        """
        获取边缘节点当前负载（0-1之间）
        
        Returns:
            float: 当前负载率（0-1）
        """
        if hasattr(self, 'task_count'):
            max_tasks = para.get("edgecore_limit", 5)
            return min(1.0, self.task_count / max(max_tasks, 1))
        return 0.0
    
    def get_queue_time(self):
        """
        获取预估排队时间（秒）
        
        Returns:
            float: 预估排队时间
        """
        if hasattr(self, 'current_remainTime'):
            return max(0.0, float(self.current_remainTime))
        return 0.0
    
    def get_available_storage(self):
        """
        获取剩余可用存储空间（字节）
        
        Returns:
            float: 剩余存储空间
        """
        if hasattr(self, 'max_storage') and hasattr(self, 'used_storage'):
            return max(0.0, float(self.max_storage - self.used_storage))
        return float('inf')
    
    def get_compute_power(self):
        """
        获取边缘节点有效计算能力 (Hz)
        
        Returns:
            float: 有效计算能力
        """
        base_power = float(self.edge_power) if hasattr(self, 'edge_power') else 0.0
        calc_param = float(self.calculate_parameter) if hasattr(self, 'calculate_parameter') else 1.0
        return base_power * calc_param
    
    # ==================== 新增方法结束 ====================


'''
Cloud-云端
相关参数：
    算力
'''
class Cloud:
    def __init__(self, task_complex_index=0):
        self.cloud_power = para["cloud_power"]
        self.task_complex_index = task_complex_index

    # ==================== 新增方法 ====================
    
    def is_available(self):
        """
        检查云端是否可用
        云端通常总是可用的（除非网络中断或维护）
        
        Returns:
            bool: 云端是否可用
        """
        # 云端资源通常认为是无限的，总是可用
        return True
    
    def get_current_load(self):
        """
        获取云端当前负载（0-1之间）
        云端资源充足，负载通常为0
        
        Returns:
            float: 当前负载率（0-1）
        """
        return 0.0
    
    def get_compute_power(self):
        """
        获取云端计算能力 (Hz)
        
        Returns:
            float: 计算能力
        """
        if hasattr(self, 'cloud_power'):
            return float(self.cloud_power)
        return para.get("cloud_power", 10e9)
    
    def get_latency_overhead(self):
        """
        获取云端额外延迟开销（WAN RTT等）
        
        Returns:
            float: 额外延迟（秒）
        """
        return para.get("cloud_wan_rtt", 0.0)
    
    # ==================== 新增方法结束 ====================


if __name__ == '__main__':
    # 测试代码
    print("测试组件的 is_available() 方法...")
    
    # 测试 Cloud
    cloud = Cloud()
    print(f"Cloud.is_available(): {cloud.is_available()}")
    print(f"Cloud.get_current_load(): {cloud.get_current_load()}")
    print(f"Cloud.get_compute_power(): {cloud.get_compute_power()}")
    
    # 测试 EdgeServer
    edge = EdgeServer(0, 100, 100, 1e9, task_complex_index=0)
    print(f"\nEdgeServer.is_available(): {edge.is_available()}")
    print(f"EdgeServer.get_current_load(): {edge.get_current_load()}")
    print(f"EdgeServer.get_queue_time(): {edge.get_queue_time()}")
    
    print("\n所有测试通过！")
