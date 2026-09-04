def merge_graphs(graphs):
    merged_graph = {}

    # 遍历图的列表
    for graph in graphs:
        # 遍历当前图的邻接表
        for node, edges in graph.items():
            if node in merged_graph:
                # 合并边列表（去除重复边）
                merged_edges = list(set(merged_graph[node] + edges))#使用了`list(set(...))`来去重，但这样的方式可能会打乱顺序，如果顺序对业务逻辑重要的话，可能会有隐患
                merged_graph[node] = merged_edges
            else:
                # 添加节点和边
                merged_graph[node] = edges

    return merged_graph


# 图的列表
graphs = [
    {1: [2, 3], 2: [5], 3: [5, 6], 5: [8], 6: [8]},
    {1: [2, 3], 2: [4, 5], 3: [5], 4: [7], 5: [7]},
    {3: [9], 9: [10]}
]

# 合并图的邻接表
merged_graph = merge_graphs(graphs)

# 打印合并后的邻接表
print(merged_graph)

def generate_graph(basegraph_num, user_num, subgraph_num):
    """
    生成图结构的边列表和顶点列表
    basegraph_num: 基础图数量
    user_num: 用户数量  
    subgraph_num: 子图数量
    返回: edge_list, vertex_list
    """
    # 简单实现：顶点为用户和子图的编号
    vertex_list = list(range(user_num + subgraph_num))
    
    # 生成边：每个用户连接到一个子图，子图之间也建立连接
    edge_list = []
    for u in range(user_num):
        # 用户连接到对应的子图
        subgraph_id = user_num + (u % subgraph_num)
        edge_list.append((u, subgraph_id))
    
    # 子图之间建立连接形成网络
    for s in range(subgraph_num):
        next_s = (s + 1) % subgraph_num
        edge_list.append((user_num + s, user_num + next_s))
    
    return edge_list, vertex_list