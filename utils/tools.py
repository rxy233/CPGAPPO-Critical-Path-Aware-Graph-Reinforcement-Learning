"""
    相关工具函数
"""
import random
from collections import deque
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def plot_graph(g):
    subax1 = plt.subplot(111)
    nx.draw(g, with_labels=True, font_weight='bold')
    plt.show()

def averange(l):
    return sum(l) / len(l)

def normalize_list(input_list):
    # 寻找列表中的最小和最大值
    return mean_normalize(input_list)

def normalize_dict_values(input_dict):
    # 找到字典中的最小值和最大值
    min_val = min(input_dict.values())
    max_val = max(input_dict.values())

    # 处理最大值和最小值相等的情况
    if min_val == max_val:
        return {key: 0.5 for key in input_dict}  # 或者返回其他合适的值，取决于你的需求

    # 归一化字典的值
    normalized_dict = {key: (value - min_val) / (max_val - min_val) for key, value in input_dict.items()}

    return normalized_dict

def minmax_normalize(input_list):
    min_val = min(input_list)
    max_val = max(input_list)

    # 如果最大值和最小值相等，则避免除以零错误
    if min_val == max_val:
        return input_list

    # 归一化列表
    normalized_list = [(x - min_val) / (max_val - min_val) for x in input_list]

    return normalized_list

def mean_normalize(input_list):
    # 计算列表的均值
    mean_value = sum(input_list) / len(input_list)

    # 对列表中的每个元素进行均值归一化
    normalized_list = [x - mean_value for x in input_list]

    return normalized_list

def m_normalize(data):

    data = [data]
    scaler = MinMaxScaler()
    # 用fit方法估计缩放所需的参数
    scaler.fit(data)

    # 使用transform方法来进行缩放
    normalized_data = scaler.transform(data)
    return normalized_data[0]
