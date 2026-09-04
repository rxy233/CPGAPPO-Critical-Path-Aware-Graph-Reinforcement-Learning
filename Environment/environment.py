# -*- coding: utf-8 -*-
"""
Physical edge-cloud environment.

English
-------
Environment assembles the physical topology from components.py: 150 user
devices, 8 edge nodes, 1 cloud, and the network links between them (with
per-pair distances, uplink/downlink bandwidth and transmission delays). It
owns the per-device / per-edge execution and upload queues and exposes the
step interface that TaskScheduler drives. All physical parameters (power,
bandwidth, edge radius, etc.) come from `para` in utils/constant.py, so
every algorithm sees the identical environment (R4-2 fair-comparison
protocol). The matrix/*.txt files override the DAG topology (chain /
default / wide) via MATRIX_OVERRIDE_PATH.

中文
----
物理环境: 由 components 组装 150 用户 + 8 边缘 + 1 云 + 网络链路, 拥有设备/边缘
执行与上传队列, 物理参数来自 para (R4-2 公平对比), DAG 拓扑由 matrix/*.txt 覆盖。
"""
import copy
import math
import random

from Environment.components import *
from utils.constant import *
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from queue import Queue
import bisect
from utils.tools import *
from Environment.Graph import BaseGraph, SubGraph
# import utils.timeline
# random.seed(1)
# 创建环境
class Environment:
    # def __init__(self, user_num=20, basegraph_num=30, subgraph_num=10, task_complex_index=3, fe=para["edge_power"]):
    def __init__(self, user_num=20, subgraph_num=10, basegraph_num=30, task_complex_index=3, fe=para["edge_power"]):
        self.task_complex_index = task_complex_index
        self.cloud = Cloud(task_complex_index=task_complex_index)
        self.basegraph = BaseGraph(basegraph_num)
        edges_num = 4

        #self.edge = EdgeServer(250, 250, fe, task_complex_index=task_complex_index)#旧设计仅1个边缘端

        # 初始化多个边缘端
        self.edges = []

        #几个边缘节点的设置
        edge_order = 0
        for i, position in enumerate(para["edge_positions"]):
            x, y = position
            # 修复处理 edge_power 是列表的情况，循环取单个值
            if isinstance(para["edge_power"], list):
                # 循环取值，防止索引越界
                power = para["edge_power"][i % len(para["edge_power"])]
            else:
                power = para["edge_power"]
            
            # 使用提取出的单个数值 power 初始化 Edge
            self.edges.append(EdgeServer(edge_order, x, y, power, task_complex_index=task_complex_index))
            edge_order += 1
        # self.edges.append(EdgeServer(150, 150, fe, task_complex_index=task_complex_index))
        # self.edges.append(EdgeServer(150, 350, fe, task_complex_index=task_complex_index))
        # self.edges.append(EdgeServer(350, 150, fe, task_complex_index=task_complex_index))
        # self.edges.append(EdgeServer(350, 350, fe, task_complex_index=task_complex_index))

        self.device_list = []
        self.subgraph_list = []
        self.task_list = random.choices(para["task_size_range"], k=user_num) # 全局任务的大小#没用到

        self.user_num = user_num
        self.subtask_num = subgraph_num

        # self.radius = 250
        # self.radius = 150 #边缘端通信范围
        self.radius = para["edge_radius"] # 边缘端通信范围 暂为200

        # 修复移除自动调用 generate_components，改为由调用方显式传入 seed
        # 这样可以确保所有进程使用确定性的环境参数
        # self.generate_components()

    # def update_edge_metrics(self):
    #     # 定期更新边缘节点状态指标
    #     for edge_idx, edge in enumerate(self.edges):
    #         load = edge.current_load / para["edgecore_limit"]
    #         for device in self.device_list:
    #             if edge_idx in device.connected_edges:
    #                 # 综合距离和负载的评分
    #                 score = (
    #                         para["edge_load_weights"][0] * device.edge_metrics[edge_idx]['distance'] +
    #                         para["edge_load_weights"][1] * load
    #                 )
    #                 device.edge_metrics[edge_idx]['score'] = score



    def generate_components(self, seed=None):
        self.device_list = []
        self.subgraph_list = []

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            rng = np.random.default_rng(seed)  # 使用带 seed 的 rng

            # 使用"设备能力因子 z"进行相关采样
            # 给每个用户采一个 z∈[0,1]，代表设备档次
            # z 越大，local_power 越强，local_wait 和 local_trans 也相应增大
            if "local_power_range" in para:
                # 设备能力因子 z
                z = rng.uniform(0.0, 1.0, size=self.user_num)

                # 线性插值函数
                def lerp(a, b, z):
                    return a + (b - a) * z

                # 主趋势（单调）
                local_powers = lerp(0.2e9, 0.6e9, z)  # 0.2-0.6 GHz
                local_waits = lerp(0.06, 0.18, z)  # 0.06-0.18 W
                local_trans_powers = lerp(1.3, 2.3, z)  # 1.3-2.3 W

                # 小噪声（5% 左右），避免出现不合理的组合
                local_powers *= rng.lognormal(mean=0.0, sigma=0.05, size=self.user_num)
                local_waits *= rng.lognormal(mean=0.0, sigma=0.05, size=self.user_num)
                local_trans_powers *= rng.lognormal(mean=0.0, sigma=0.05, size=self.user_num)

                # clip 保底
                local_powers = np.clip(local_powers, 0.2e9, 0.6e9)
                local_waits = np.clip(local_waits, 0.05, 0.20)
                local_trans_powers = np.clip(local_trans_powers, 1.0, 2.8)
            else:
                # 兼容旧版本
                local_powers = None
                local_waits = None
                local_trans_powers = None
        else:
            rng = None
            local_powers = None
            local_waits = None
            local_trans_powers = None
        cnt = 0
        #生成用户节点#控制生成的终端设备位置1
        while cnt < self.user_num:
            temp_x = round(random.normalvariate(250, 200))
            temp_y = round(random.normalvariate(250, 200))
            # temp_x = round(random.normalvariate(100, 40))
            # temp_y = round(random.normalvariate(100, 40))
            # temp_x = random.uniform(0, 500)
            # temp_y = random.uniform(0, 500)

            if temp_x < 0 or temp_x > 500 or temp_y < 0 or temp_y > 500:#保证在500x500的范围内
                continue
            # if temp_x < 0 or temp_x > 200 or temp_y < 0 or temp_y > 200:#保证在500x500的范围内
            #     continue



            in_range = False
            too_close = False

            for edge in self.edges:
                if temp_x == edge.pos_x and temp_y == edge.pos_y:
                    too_close = True
                    break

            if too_close:
                continue

            # temp_dist = [150,150,150,150]
            # temp_dist = [150] * len(self.edges)
            # for index,edge in enumerate(self.edges):
            #     # first_dist = math.sqrt((temp_x - edge.pos_x) ** 2 + (temp_y - edge.pos_y) ** 2)
            #     temp_dist0 = math.sqrt((temp_x - edge.pos_x) ** 2 + (temp_y - edge.pos_y) ** 2)
            #     temp_dist[index] = temp_dist0#设备和目标边缘的距离
            #
            #     if temp_dist0 <= self.radius:
            #         in_range = True
            #     # if temp_dist0 < temp_dist:
            #     #     first_dist = temp_dist0#设备和目标边缘的距离

            # if in_range:
            #     # 生成用户
            #     #给temp_dist排序
            #     temp_dist.sort()
            #     self.device_list.append(Device(temp_x, temp_y, temp_dist, self.task_complex_index))
            #     self.subgraph_list.append(SubGraph(self.subtask_num, self.basegraph))
            #     cnt += 1

            distances = [float("inf")] * len(self.edges)
            edge_scores = [float("inf")] * len(self.edges)

            # best_edge_index = -1
            # best_score = float('inf')  # 记录当前最优分数

            for index, edge in enumerate(self.edges):
                # 计算与边缘节点的距离
                # first_dist = math.sqrt((temp_x - edge.pos_x) ** 2 + (temp_y - edge.pos_y) ** 2)
                temp_dist0 = math.sqrt((temp_x - edge.pos_x) ** 2 + (temp_y - edge.pos_y) ** 2)
                if temp_dist0 < distances[index]:
                    distances[index] = temp_dist0  # 设备和目标边缘的距离,不在范围内就为inf

                #判断生成的终端设备是否在范围内
                if temp_dist0 <= self.radius:
                    in_range = True
                # if temp_dist0 < temp_dist:
                #     first_dist = temp_dist0#设备和目标边缘的距离


            #生成的终端设备在范围内时进行添加
            if in_range:
                # 生成用户

                # distances.sort()#不能给distances排序，应使得distances与edges一一对应
                edge_inrange = []

                for edge_idx in range(len(self.edges)): #用各边缘端的距离和其负载（初始无负载）计算其评分
                    if distances[edge_idx] <= self.radius:
                        edge_inrange.append(self.edges[edge_idx])
                        edge_scores[edge_idx] = para["edge_load_weights"][0] * distances[edge_idx]


                # 修复传入预先生成的设备参数，使用 seed 控制
                device_power = local_powers[cnt] if local_powers is not None else None
                device_wait = local_waits[cnt] if local_waits is not None else None
                device_trans = local_trans_powers[cnt] if local_trans_powers is not None else None
                self.device_list.append(Device(cnt, temp_x, temp_y, distances, edge_scores, edge_inrange,
                                         self.task_complex_index,
                                         local_power=device_power,
                                         local_wait=device_wait,
                                         local_trans=device_trans))
                # 每个终端设备的应用对应一个子图（任务图），在此处依照BaseGraph生成有subtask_num个节点的子图，故subtask_num为单个终端上的任务数
                self.subgraph_list.append(SubGraph(self.subtask_num, self.basegraph))
                cnt += 1


    def plot_env(self):
        fig, ax = plt.subplots(1)
        ax.set_aspect('equal')
        for edge in self.edges:
            # circle = Circle((self.edge.pos_x, self.edge.pos_y), radius=self.radius, facecolor='white', edgecolor='cornflowerblue', alpha=0.8, fill=False)
            circle = Circle((edge.pos_x, edge.pos_y), radius=self.radius, facecolor='white',
                            edgecolor='cornflowerblue', alpha=0.8, fill=False)
            ax.add_patch(circle)
        # ax.set_xlim(0, 500)
        # ax.set_ylim(0, 500)
        ax.set_xlim(-50, 550)
        ax.set_ylim(-50, 550)
        # ax.set_xlim(-50, 250)
        # ax.set_ylim(-50, 250)
        square = Rectangle(
            xy=(0, 0),  # 左下角坐标
            # width=200,  # 宽度
            # height=200,  # 高度
            width=500,  # 宽度
            height=500,  # 高度
            linewidth=2,  # 边框粗细
            edgecolor='r',  # 边框颜色（红色）
            facecolor='none'  # 无填充
        )

        # 将正方形添加到当前坐标轴
        ax = plt.gca()
        ax.add_patch(square)

        user_position_x = []
        user_position_y = []
        for item in self.device_list:
            user_position_x.append(item.pos_x)
            user_position_y.append(item.pos_y)
        plt.scatter(user_position_x, user_position_y, color='orange')
        # plt.scatter(150, 150, color='green', marker='^', s=64)
        # plt.scatter(150, 350, color='green', marker='^', s=64)
        # plt.scatter(350, 150, color='green', marker='^', s=64)
        # plt.scatter(350, 350, color='green', marker='^', s=64)
        for position in para["edge_positions"]:
            x, y = position
            plt.scatter(x, y, color='green', marker='^', s=64)

        # plt.scatter(250, 250, color='blue')
        # plt.plot()
        plt.xlabel("Meters", fontsize=13)
        plt.ylabel("Meters", fontsize=13)
        # plt.savefig(r"pic/environment.png", dpi=600)
        plt.show()

    def set_fe(self, fe):
        self.edge.edge_power = fe

    def reset_env(self):
        pass

    # def step(self, actions):
    #     pass

    def calculate_average_distance(self):
        total_distance = 0
        for device in self.device_list:
            min_distance = float('inf')
            for edge in self.edges:
                distance = math.sqrt((device.pos_x - edge.pos_x) ** 2 + (device.pos_y - edge.pos_y) ** 2)
                if distance < min_distance:
                    min_distance = distance
            total_distance += min_distance
        average_distance = total_distance / len(self.device_list)
        return average_distance

if __name__ == '__main__':
    # env = Environment(user_num=60)
    env = Environment(user_num = 60, basegraph_num = 60, subgraph_num = 20, task_complex_index = 3)
    avg_distance = env.calculate_average_distance()
    print(f"平均距离: {avg_distance}")
    env.plot_env()