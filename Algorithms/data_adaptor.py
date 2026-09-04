# 图数据转换为tensor
# 创建图数据
import torch
import torch_geometric.data as data
from torch_geometric.data import Data

def graph_data_generator(graph, nodes_states):
    adjacency_matrix = [[0, 1, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0]]  # 创建邻接矩阵，可以使用稀疏矩阵表示
    adjacency_matrix = torch.tensor(adjacency_matrix)

    # 创建图数据对象
    x = torch.randn(4, 3)  # 100个节点，每个节点有3个属性
    edge_index = adjacency_matrix.nonzero().t()  # 从邻接矩阵中生成边索引

    graph_data = Data(x=x, edge_index=edge_index)

    return graph_data