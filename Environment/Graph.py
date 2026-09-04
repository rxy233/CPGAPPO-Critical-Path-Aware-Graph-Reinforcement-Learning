# -*- coding: utf-8 -*-
"""
DAG graph classes: BaseGraph (shared 60-node base DAG) and SubGraph.

English
-------
  - BaseGraph: the fixed 60-node base directed acyclic graph (loaded from
    matrix/*.txt). The three topology files matrix_60.txt (default),
    matrix_60_chain.txt and matrix_60_wide.txt give the chain / default /
    wide DAG shapes used in all experiments.
  - SubGraph: a per-user sub-DAG carved out of BaseGraph via BFS/DFS; each
    node carries the subtask size (bytes) and complexity index used by
    computation.execute_consumption. SubGraph exposes the networkx graph
    that GraphScheduler traverses to find ready subtasks and that the
    guide-score functions traverse to compute CP depth.

中文
----
BaseGraph: 60 节点的基础 DAG (由 matrix/*.txt 加载, 三种拓扑 chain/default/wide)。
SubGraph: 从 BaseGraph 拆出的 per-user 子图, 节点带子任务大小与复杂度, 供
GraphScheduler 找就绪子任务、guide 分数算 CP 深度。
"""
import random
from collections import deque
import networkx as nx
import matplotlib.pyplot as plt
import time
from networkx.drawing.nx_agraph import graphviz_layout
from adjustText import adjust_text

class BaseGraph:
    def __init__(self, num):
        self.num = num  # 图节点个数
        self.nx_graph = self.parse_dag() # 邻接矩阵形式
        self.enter_nodes = [n for n, d in self.nx_graph.in_degree() if d == 0]
        self.exit_nodes = [n for n, d in self.nx_graph.out_degree() if d == 0]
        self.reachable_nodes = self.get_reachable_nodes()

    # def get_reachable_nodes(self):
    #     shortest_paths = dict(nx.all_pairs_shortest_path_length(self.nx_graph))
    #     target_node = self.exit_nodes[0]
    #     reachable_nodes = {}  # 用于存储能够到达指定节点的其他节点和它们的路径长度
    #     for source_node, shortest_path_lengths in shortest_paths.items():
    #         if target_node in shortest_path_lengths:
    #             path_length = shortest_path_lengths[target_node]
    #             reachable_nodes[source_node] = path_length
    #     del reachable_nodes[target_node]
    #     # print(reachable_nodes)
    #     return reachable_nodes

    def get_reachable_nodes(self):
        """
        【关键路径改进】计算到出口的最长路层数（不是最短路径）
        这样可以识别出更靠近关键路径的节点

        使用逆拓扑序 DP 计算 dist_to_exit[v]:
        - dist_to_exit[exit] = 0
        - dist_to_exit[v] = 1 + max(dist_to_exit[succ]) for succ in successors
        """
        g = self.nx_graph
        dist_to_exit = {}

        # 拓扑排序（用于逆序处理）
        topo = list(nx.topological_sort(g))

        # 逆拓扑序计算最长路
        for v in reversed(topo):
            succs = list(g.successors(v))
            if not succs:
                # 出口节点，距离为 0
                dist_to_exit[v] = 0
            else:
                # 取所有后继的最大距离 + 1
                dist_to_exit[v] = 1 + max(dist_to_exit.get(succ, 0) for succ in succs)

        # 删除出口节点本身（与原逻辑一致）
        for exit_node in self.exit_nodes:
            if exit_node in dist_to_exit:
                del dist_to_exit[exit_node]

        return dist_to_exit

    def _generate_random_dag(self, num_nodes):
        """
        生成一个随机的有向无环图 (DAG)
        num_nodes: 节点数量
        返回: nx.DiGraph 对象
        """
        # 使用 networkx 的随机图生成器
        # 指定有向图，并确保没有环
        # 简单策略：按拓扑顺序添加边
        G = nx.DiGraph()
        G.add_nodes_from(range(num_nodes))
        
        # 添加边：只从小编号节点连向大编号节点，确保无环
        edge_count = int(num_nodes * 1.5)  # 添加约1.5倍的边
        for _ in range(edge_count):
            # 随机选择两个不同的节点，且确保 source < dest
            source = random.randint(0, num_nodes - 2)
            dest = random.randint(source + 1, num_nodes - 1)
            # 只有当边不存在时才添加（避免重复边）
            if not G.has_edge(source, dest):
                G.add_edge(source, dest)
        
        # 确保图是连通的
        if not nx.is_weakly_connected(G):
            # 添加额外的边来确保连通性
            components = list(nx.weakly_connected_components(G))
            for i in range(len(components) - 1):
                nodes1 = sorted(list(components[i]))
                nodes2 = sorted(list(components[i + 1]))
                if nodes1 and nodes2:
                    # 从第一个组件的最小节点连接到第二个组件的最小节点
                    G.add_edge(min(nodes1), min(nodes2))
        
        return G

    def parse_dag(self):
        import os  # 确保在函数开头导入，避免局部变量冲突
        # path = r"E:\experiments\DVR\dvr\matrix/matrix_{}.txt".format(self.num)
        # 默认 matrix 路径 (仅占位; 实际由 MATRIX_OVERRIDE_PATH 环境变量指定)
        path = os.path.join(os.getcwd(), "matrix", "matrix_{}.txt").format(self.num) if False else None
        
        # 【扩展】支持环境变量覆盖（用于不同DAG结构对比实验，不影响原流程）
        _override = os.environ.get("MATRIX_OVERRIDE_PATH")
        if _override and os.path.exists(_override):
            path = _override
            print(f"[Graph] 使用覆盖路径: {path}")
        
        # 修复检查文件是否存在，如果不存在则动态生成 DAG
        if not os.path.exists(path):
            # 文件不存在，动态生成一个随机 DAG
            print(f"[Graph] 矩阵文件不存在: {path}")
            print(f"[Graph] 自动生成随机 DAG (节点数: {self.num})")
            return self._generate_random_dag(self.num)
        
        file = open(path, "r")
        lines = file.readlines()
        edges = []
        for line in lines:
            if '->' in line:
                source, dest = line.split('[size =')[0].split(' -> ')
                edges.append((int(source)-1, int(dest)-1))
        G = nx.DiGraph(edges)
        return G



class SubGraph:
    """
    子图，根据传入的base图生成对应的子图。子节点
    """
    def __init__(self, num: int, basegraph: BaseGraph):
        self.num = num# 生成的子图节点个数
        self.nx_graph = self.generate_subgraph(basegraph)#通过排序和深度优先搜索（DFS）生成子图
        # self.nx_graph = self.generate_subgraph_old(basegraph)#尝试使用旧版随机生成方法
        self.enter_nodes = [n for n, d in self.nx_graph.in_degree() if d == 0]
        self.exit_nodes = [n for n, d in self.nx_graph.out_degree() if d == 0]

    def generate_subgraph_old(self, basegraph: BaseGraph):#纯随机取点
        G = basegraph.nx_graph
        subgraph_nodes = set()
        while len(subgraph_nodes) < self.num:
            node = random.choice(list(G.nodes))
            subgraph_nodes.add(node)
            reachable_nodes = set(nx.descendants(G, node))
            # subgraph_nodes.update(reachable_nodes)#每次把可达节点一起加进去，导致子图节点数过多
            for reachable_node in reachable_nodes:#修改，一个个添加，直到达到要求的节点数，防止子图过大
                if len(subgraph_nodes) < self.num:
                    subgraph_nodes.add(reachable_node)
                else:
                    break
                # print("len", len(subgraph_nodes))
        S = basegraph.nx_graph.subgraph(subgraph_nodes)
        return S

    def generate_subgraph(self, basegraph: BaseGraph):#从最长路径起点之一随机取初始点，然后用DFS遍历，直到达到要求的节点数
        G = basegraph.nx_graph
        subgraph_nodes = set()
        distances = basegraph.reachable_nodes

        sorted_nodes = sorted(distances.items(), key=lambda x: x[1], reverse=True)
        # 选择距离最远的一些节点
        top_n = len(G.nodes) // 6 #选择总节点数的1/6作为初始节点备选
        farthest_nodes = [node for node, distance in sorted_nodes[:top_n]]

        # 根据权重随机选择初始节点（权重是？）
        initial_node = random.choice(farthest_nodes)
        # subgraph_nodes.add(initial_node)
        new_nodes = {initial_node}
        new_neighbors = set()#看看是否深度优先搜索
        while len(subgraph_nodes) < self.num and new_nodes:
            # node = new_nodes.pop()#每次都选最小的那个值，一点都不随机
            # node = random.choice(list(new_nodes))
            if new_neighbors:
                node = random.choice(list(new_neighbors))#保证每次都能选到新增节点的邻居节点
            else:
                node = random.choice(list(new_nodes))#到头了就随机选一个老邻居节点
            new_nodes.remove(node)

            subgraph_nodes.add(node)#每次添加一个节点
            # print("新增节点：", node)
            neighbors = list(G.neighbors(node))#一直获取最新添加节点的邻居节点

            new_neighbors = set()


            # random.shuffle(neighbors)  # 随机打乱邻居节点的顺序，以模拟DFS的同时保证随机性 #这不像DFS，像是BFS #随机性由random.choice提供

            for neighbor in neighbors:
                if neighbor not in subgraph_nodes:
                    new_nodes.add(neighbor)
                    new_neighbors.add(neighbor)
            # print("新邻居：", new_neighbors)
            # print("备选添加节点：", new_nodes)
        #
        # print("subgraph_nodes length: ",len(subgraph_nodes))

        S = G.subgraph(subgraph_nodes)

        # visualize_graph2(S)#看一眼子图什么样

        return S


def visualize_graph2(G):
    # 使用 spring_layout 布局
    # pos = nx.spring_layout(G) #每次都不一样
    pos = nx.kamada_kawai_layout(G, scale=2) #美观，显示头尾

    # pos = nx.spring_layout(G, k=0.5, iterations=50)  # k 控制节点间距，iterations 控制迭代次数

    # 绘制图
    plt.figure(figsize=(16, 16))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=200,
        node_color='lightblue',
        font_size=10,
        font_weight='bold',
        arrows=True,
        arrowstyle='->',
        arrowsize=10,
        edge_color='gray'
    )

    # texts = [plt.text(x, y, s) for (x, y), s in zip(pos.values(), G.nodes)]
    # texts = [plt.text(x + 0.005, y, s) for (x, y), s in zip(pos.values(), G.nodes)]
    # adjust_text(texts)

    # 调整图的显示
    plt.title("Directed Graph Visualization")
    plt.axis("off")
    # plt.tight_layout(pad=2.0, w_pad=2.0, h_pad=2.0)

    # 保存图
    plt.savefig("graph.png", bbox_inches="tight")
    # plt.savefig('graph{}.png'.format(subgraph_num), bbox_inches="tight")
    plt.show()
    plt.close()

def visualize_graph(G):
    # 绘制图
    pos = nx.spring_layout(G)  # 使用Fruchterman-Reingold布局算法
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=10, font_weight='bold',
            arrows=True)
    plt.title("Directed Graph Visualization")

    # plt.savefig('graph{}.png'.format(subgraph_num), bbox_inches="tight")
    plt.show()
    plt.close()


def test_generate_subgraph_old_show(G, num_nodes=15):
    # G = basegraph.nx_graph
    # subgraph_nodes = set()
    # while len(subgraph_nodes) < num_nodes:
    #     node = random.choice(list(G.nodes))
    #     subgraph_nodes.add(node)
    #     reachable_nodes = set(nx.descendants(G, node))
    #     subgraph_nodes.update(reachable_nodes)
    # S = basegraph.nx_graph.subgraph(subgraph_nodes)
    subgraph_nodes = set()
    while len(subgraph_nodes) < num_nodes:
        node = random.choice(list(G.nodes))
        print("node",node)
        subgraph_nodes.add(node)
        reachable_nodes = set(nx.descendants(G, node))
        for reachable_node in reachable_nodes:
            if len(subgraph_nodes) < num_nodes:
                subgraph_nodes.add(reachable_node)
            else:
                break
            # print("len", len(subgraph_nodes))
        # subgraph_nodes.update(reachable_nodes)
        # print("len",len(subgraph_nodes))
    subgraph = G.subgraph(subgraph_nodes)
    # print(len(subgraph_nodes))
    return subgraph

def test_generate_subgraph_show(G, num_nodes=15):

    subgraph_nodes = set()
    distances = G.reachable_nodes

    sorted_nodes = sorted(distances.items(), key=lambda x: x[1], reverse=True)
    # 选择距离最远的一些节点
    top_n = len(G.nodes) // 6
    farthest_nodes = [node for node, distance in sorted_nodes[:top_n]]

    # 根据权重随机选择初始节点
    initial_node = random.choice(farthest_nodes)
    # subgraph_nodes.add(initial_node)
    new_nodes = {initial_node}

    while len(subgraph_nodes) < num_nodes and new_nodes:
        node = new_nodes.pop()
        subgraph_nodes.add(node)
        neighbors = list(G.neighbors(node))
        random.shuffle(neighbors)  # 随机打乱邻居节点的顺序，以模拟DFS

        for neighbor in neighbors:
            if neighbor not in subgraph_nodes:
                new_nodes.add(neighbor)

    print("subgraph_nodes:  ",len(subgraph_nodes))

    S = G.subgraph(subgraph_nodes)

    # visualize_graph2(S)#看一眼子图什么样

    return S

if __name__ == "__main__":
    basegraph_num = 45
    subtask_num = basegraph_num / 3
    basegraph = BaseGraph(basegraph_num)
    visualize_graph2(basegraph.nx_graph)

    # 可视化原始图
    for _ in range(5):
        subgraph = SubGraph(subtask_num, basegraph)
        # visualize_graph2(subgraph.nx_graph)



    # path = r"matrix/matrix_{}.txt".format(subgraph_num)  # 这里改文件名
    # file = open(path, "r")
    # lines = file.readlines()
    # edges = []
    # for line in lines:
    #     if '->' in line:
    #         source, dest = line.split('[size =')[0].split(' -> ')
    #         edges.append((int(source) - 1, int(dest) - 1))
    # G = nx.DiGraph(edges)

    # for subgraph_num in range(30, 300, 15):
    #     path = r"matrix/matrix_{}.txt".format(subgraph_num)#这里改文件名
    #     file = open(path, "r")
    #     lines = file.readlines()
    #     edges = []
    #     for line in lines:
    #         if '->' in line:
    #             source, dest = line.split('[size =')[0].split(' -> ')
    #             edges.append((int(source) - 1, int(dest) - 1))
    #     G = nx.DiGraph(edges)
        # 可视化原始图

        # visualize_graph2(G)

        # exit_nodes = [n for n, d in G.out_degree() if d == 0]
        # print("subgraph_num ", subgraph_num, ": ", len(exit_nodes))
        # print(len(exit_nodes))



    # G2 = test_generate_subgraph_show(G, 15)
    # G2 = test_generate_subgraph_old_show(G, 15)

    # G2 = test_generate_subgraph_show(G, 15)
    # visualize_graph2(G2)



    # bg = BaseGraph(15)
    # sg = SubGraph(5, bg)
    # subax1 = plt.subplot(121)
    # nx.draw(bg.nx_graph, with_labels=True, font_weight='bold')
    # subax1 = plt.subplot(122)
    # nx.draw(sg.nx_graph, with_labels=True, font_weight='bold')
    # plt.show()
    # print(bg.matrix)
    # print(sg.matrix)
