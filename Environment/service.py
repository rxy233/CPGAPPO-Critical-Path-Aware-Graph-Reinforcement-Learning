"""
    服务类
        用以计算整个过程中的时间以及消耗
"""
import numpy as np
import math
from utils.constant import *
import random

class Service:
    def __init__(self, env):
        self.env = env
        self.task_complex_index = env.task_complex_index
        self.ready_time = []
        self.wait_time = []
        self.finish_time = []

        self.energy_list = []

        self.energy_counter = []

        self.time_reset()

    def compute_communication_time(self, task_info: list, tp: str = "end-end", is_download=False) -> float:
        """
        计算传输时间统一接口
        :param task_info: 任务信息 [user_index, task_index]
        :param tp: end-end设备之间，end-edge设备边，end-cloud设备云
        :param is_download: 是否为下行，下行直接拿到下行大小进行计算
        :return:
        """
        trans_rate = None
        task_size = para["result_size"] if is_download else self.env.get_tasksize(task_info)
        if tp == "end-end":
            inner = (trans_power["mm"] * channel_gain["mm"])/comm["noise"]
            trans_rate = comm["bandwidth"] * math.log2(1 + inner)
        elif tp == "end-edge":
            inner = (trans_power["AP"] * channel_gain["AP"])/comm["noise"]
            trans_rate = comm["bandwidth"] * math.log2(1 + inner)
        elif tp == "end-cloud":  # 写死的传输速度
            trans_rate = para["R_ec"]
        return task_size/trans_rate


    # 初始化所有列表
    def time_reset(self):
        # ready_list
        device_num = len(self.env.device_list)
        task_num = self.env.subtask_num
        self.ready_time = np.full((device_num, task_num), np.Inf)
        self.finish_time = np.full((device_num, task_num), np.Inf)
        self.energy_counter = []
        for i in range(device_num):
            self.energy_counter.append(EnergyCounter()) # 能耗计算器
            start_index = self.env.device_list[i].task.get_start_index()
            for j in start_index:
                self.ready_time[i][j] = self.env.device_list[i].task.arrival
                self.earlist_send[i][j] = self.env.device_list[i].task.arrival

        self.energy_list = np.full(device_num, np.Inf)
        self.env.reset_env()


# 能耗计算
    def compute_energy(self, store, trans_time=0., other_time=0., type="local", user_power=0, user=None):
        cycle = store * para["task_complex"][self.task_complex_index]
        if type == "local":
            return para["kappa"] * (user_power ** 2) * cycle

        if type == "edge" or type == "cloud":
            # 使用设备特定的 local_wait 和 local_trans
            local_trans = user.local_trans if user and hasattr(user, 'local_trans') else para["local_trans"]
            local_wait = user.local_wait if user and hasattr(user, 'local_wait') else para["local_wait"]
            return local_trans * trans_time + local_wait * other_time



    def update_energy(self, energy, item):
        task_index, sub_index = item
        if self.energy_list[task_index] == np.Inf:
            self.energy_list[task_index] = energy
        else:
            self.energy_list[task_index] += energy

    def compute_local(self, index):
        task_index, sub_index = index
        user = self.env.device_list[task_index]
        store = user.task.get_storage(sub_index)
        compute_t = store * para["task_complex"][self.task_complex_index] / user.local_power
        return round(compute_t, 6)

    def compute_trans(self, bw, item):
        task_index, sub_index = item
        user = self.env.device_list[task_index]
        # 使用设备特定的 local_trans
        local_trans = user.local_trans if hasattr(user, 'local_trans') else para["local_trans"]
        upload_speed = bw * math.log2(1 + (local_trans * para["H_0"]) / para["yita2"])
        store = user.task.get_storage(sub_index)
        return round(store / upload_speed, 6)

    def compute_trans_new(self, bw, edge_index, item, type="edge"):
        task_index, sub_index = item
        user = self.env.device_list[task_index]
        edge = self.env.edge_list[edge_index]
        store = user.task.get_storage(sub_index)
        new_bw = bw * (10 ** 6)
        # 使用设备特定的 local_trans
        local_trans = user.local_trans if hasattr(user, 'local_trans') else para["local_trans"]
        if type == "edge":
        # 计算距离
            L = math.sqrt((user.pos_x - edge.pos_x) ** 2 + (user.pos_y - edge.pos_y) ** 2)
            # 基于自由空间损耗得到的速率
            upload_speed = new_bw * math.log2(1 + (local_trans * (L ** -para["theta"]) * 1 / para["yita2"]))
            return round(store / upload_speed, 6)
        elif type == "ed-ed":
            L = 140
            path_loss = 128.1 + 37.6 * math.log10(L / 1000)
            channel_gain = 10 ** (7. - path_loss / 10)
            upload_speed = new_bw * math.log2(1 + 100 * channel_gain / para["yita2"])
            return round(store / upload_speed, 6)

    def compute_trans_cloud(self, item):
        task_index, sub_index = item
        user = self.env.device_list[task_index]
        store = user.task.get_storage(sub_index)
        return round(store / para["R_ec"] , 6)

    # 前提是还能有带宽分配
    def compute_edge(self, index):
        task_index, sub_index = index
        user = self.env.device_list[task_index]
        store = user.task.get_storage(sub_index) * para["task_complex"][self.task_complex_index]

        # execute
        execute_list = []
        edge_set = user.edge_set
        for i in edge_set:
            execute_list.append(store / (self.env.edge_list[i].edge_power * self.env.edge_list[i].calculate_parameter))
        execute_t = min(execute_list)

        return round(execute_t, 6)

    def compute_cloud(self, index):
        task_index, sub_index = index
        user = self.env.device_list[task_index]
        store = user.task.get_storage(sub_index) * para["task_complex"][self.task_complex_index]
        return round(store / para["cloud_power"], 6)

    # ft = rt + exe_t, 更新device中ft
    def update_local(self, exe_t, item, queue_t=0.):
        self.finish_time[item[0]][item[1]] = self.ready_time[item[0]][item[1]] + exe_t + queue_t
        self.env.device_list[item[0]].task.subtask_completed[item[1]] = self.finish_time[item[0]][item[1]]

    # e_arr = e_send + upload_time
    def update_upload(self, upload_time, item, type="edge"):
        task_index, sub_index = item
        if type == "edge":
            self.earlist_arr[task_index][sub_index] = self.earlist_send[task_index][sub_index] + upload_time

        if type == "ed-ed":
            self.earlist_arr[task_index][sub_index] += upload_time


    def update_edge(self, edge_exe, item):
        task_index, sub_index = item
        self.finish_time[task_index][sub_index] = self.earlist_arr[task_index][sub_index] + edge_exe
        self.env.device_list[item[0]].task.subtask_completed[item[1]] = self.finish_time[item[0]][item[1]]

    def update_cloud(self, upload_cloud, exe_cloud, item):
        task_index, sub_index = item
        self.finish_time[task_index][sub_index] = self.earlist_arr[task_index][sub_index] + upload_cloud + exe_cloud
        self.env.device_list[item[0]].task.subtask_completed[item[1]] = self.finish_time[item[0]][item[1]]


    # 更新ready time
    def update_ready(self, item):
        # 获取前驱
        task_index, sub_index = item
        user = self.env.device_list[task_index]
        pre = user.task.pre_traverse([sub_index])
        temp_ft = []
        for p in pre:
            temp_ft.append(self.finish_time[task_index][p])
        if len(temp_ft):
            self.ready_time[task_index][sub_index] = max(temp_ft)
            if self.earlist_send[task_index][sub_index] not in [-1., -2.]:
                self.earlist_send[task_index][sub_index] = self.ready_time[task_index][sub_index]

    def avg_ft(self):
        rt_min = []
        for row in self.ready_time:
            rt_min.append(min(row))
        ft_max = []
        for row in self.finish_time:
            ft_max.append(max(row))
        res = [ft - rt for rt, ft in zip(rt_min, ft_max)]
        return sum(res)/len(res)


    def avg_energy(self):
        return sum(self.energy_list)/len(self.energy_list)

    def user_delay(self):
        u_delay = []
        for user in self.finish_time:
            u_delay.append(max(user))
        return u_delay

    # 完成后调用以计算所有能耗
    def e_counter_calc(self):
        for device_index in range(len(self.energy_counter)):
            energy = self.energy_counter[device_index].get_total_energy\
                (self.ready_time[device_index], self.finish_time[device_index])
            self.energy_list[device_index] = energy


# 每个device给一个
class EnergyCounter:
    """
    【新模型】三独立耗能单元统计
    - base_platform_energy: 基础平台能耗（待机）
    - cpu_module_energy: CPU模块能耗（计算）
    - rf_module_energy: 射频模块能耗（上传）
    """
    def __init__(self):
        # 时间统计
        self.local_time = np.inf  # 计算时间
        self.upload_time = np.inf  # 上传时间

        # 能耗统计（三独立单元）
        self.base_platform_energy = np.inf  # 基础平台能耗
        self.cpu_module_energy = np.inf      # CPU模块能耗
        self.rf_module_energy = np.inf        # 射频模块能耗

        # 向后兼容
        self.local_energy = np.inf

    def reset(self):
        self.local_time = np.inf
        self.upload_time = np.inf
        self.base_platform_energy = np.inf
        self.cpu_module_energy = np.inf
        self.rf_module_energy = np.inf
        self.local_energy = np.inf

    def update_local_para(self, time, energy):
        """
        更新计算时间和能耗
        energy 已经包含：基础平台能耗 + CPU模块能耗
        """
        if self.local_time == np.inf:
            self.local_time = time
        else:
            self.local_time += time

        if self.local_energy == np.inf:
            self.local_energy = energy
        else:
            self.local_energy += energy

    def update_upload_para(self, time, energy=None):
        """
        更新上传时间和能耗
        energy 已经包含：基础平台能耗 + 射频模块能耗
        """
        if self.upload_time == np.inf:
            self.upload_time = time
        else:
            self.upload_time += time

        # 如果传入了能耗，记录射频能耗（不包含基础平台，避免重复计算）
        if energy is not None:
            # 上传总能耗 = 基础平台 + 射频
            # 射频能耗 = 上传总能耗 - 基础平台能耗
            if self.rf_module_energy == np.inf:
                self.rf_module_energy = energy - para["local_wait"] * time
            else:
                self.rf_module_energy += energy - para["local_wait"] * time

    def get_total_energy(self, RT, FT):
        ready_time = min(RT)
        finish_time = max(FT)
        middle = round(finish_time - ready_time, 6)
        energy = None
        # 处理inf
        if self.local_time is np.inf:
            self.local_time = 0.
        if self.upload_time is np.inf:
            self.upload_time = 0.
        if self.local_energy is np.inf:
            self.local_energy = 0.
        if self.base_platform_energy == np.inf:
            self.base_platform_energy = 0.
        if self.cpu_module_energy == np.inf:
            self.cpu_module_energy = 0.
        if self.rf_module_energy == np.inf:
            self.rf_module_energy = 0.

        # 【新模型】总能耗 = 基础平台能耗 + CPU模块能耗 + 射频模块能耗
        # 其中基础平台能耗在整个应用周期内持续存在
        wait_time = middle - self.local_time - self.upload_time

        # 计算各模块能耗
        # 1. 基础平台能耗：整个应用周期（等待 + 计算 + 上传）
        base_energy = middle * para["local_wait"]

        # 2. CPU模块能耗：仅在计算时产生（已包含在 local_energy 中）
        # 从 local_energy 中提取 CPU 模块能耗：local_energy = base + cpu
        cpu_energy = self.local_energy - self.local_time * para["local_wait"]

        # 3. 射频模块能耗：仅在传输时产生
        # upload_energy 已包含在 wait_time * para["local_trans"] 中（旧模型）
        # 新模型：rf_energy = upload_time * para["local_trans"]
        rf_energy = self.upload_time * para["local_trans"]

        # 保存详细能耗统计
        self.base_platform_energy = base_energy
        self.cpu_module_energy = cpu_energy
        self.rf_module_energy = rf_energy

        # 总能耗（线性叠加）
        energy = base_energy + cpu_energy + rf_energy

        # 兼容旧逻辑（如果使用旧调用方式）
        # energy = self.local_energy + wait_time * para["local_wait"] + self.upload_time * para["local_trans"]

        return energy