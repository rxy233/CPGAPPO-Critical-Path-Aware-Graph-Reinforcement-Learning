import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
def generate_uniform_endpoints(num_endpoints, area_size=500):#随机
    # 计算网格大小
    grid_size = int(np.ceil(np.sqrt(num_endpoints)))

    # 计算每个网格单元的边长
    step = area_size / grid_size

    endpoints = []
    for i in range(grid_size):
        for j in range(grid_size):
            if len(endpoints) >= num_endpoints:
                break
            # 在每个网格单元内生成一个随机点
            x = np.random.uniform(i * step, (i + 1) * step)
            y = np.random.uniform(j * step, (j + 1) * step)
            endpoints.append((x, y))

    return endpoints

def generate_grid_endpoints(num_endpoints, area_size=500):#固定
    grid_size = int(np.ceil(np.sqrt(num_endpoints)))
    step = area_size / grid_size
    endpoints = []
    print("step：", step)
    for i in range(grid_size):
        for j in range(grid_size):
            if len(endpoints) >= num_endpoints:
                break
            # 在每个网格单元的中心生成一个点
            x = (i + 0.5) * step
            y = (j + 0.5) * step
            endpoints.append((x, y))
    return endpoints


def greedy_placement(x):
    r = 150
    locations = [(250, 250)] # 初始点为中心
    for _ in range(x - 1):
        # 生成候选点（简化为网格点）
        grid = np.meshgrid(np.arange(0, 500, 50), np.arange(0, 500, 50))
        points = np.column_stack([grid[0].flatten(), grid[1].flatten()])

        # 筛选未被覆盖的点
        covered = np.array([np.min([np.linalg.norm(p - np.array(loc)) for loc in locations]) < r for p in points])
        uncovered = points[~covered]

        # 找到未覆盖区域的最远点
        if len(uncovered) == 0:
            break
        max_dist = np.argmax(np.linalg.norm(uncovered - np.array([250, 250]), axis=1))
        new_point = uncovered[max_dist]
        # min_dist = np.argmin([np.sum([np.linalg.norm(p - np.array(loc)) for loc in locations]) for p in uncovered])
        # new_point = uncovered[min_dist]

        # 确保新点离现有边缘端足够远
        distances = [np.linalg.norm(new_point - np.array(loc)) for loc in locations]
        if np.min(distances) > 2 * r:
            locations.append(tuple(new_point))
    return locations

if __name__ == '__main__':
    # 设置边缘端的数量
    num_endpoints = 4
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体




    # 生成网格分布的边缘端坐标
    endpoints = generate_grid_endpoints(num_endpoints)

    # # 生成随机生成的边缘端坐标
    # endpoints = generate_uniform_endpoints(num_endpoints)

    # # 生成贪心算法生成的边缘端坐标
    # endpoints = greedy_placement(num_endpoints)

    # 打印生成的坐标
    for i, (x, y) in enumerate(endpoints):
        print(f"边缘端 {i + 1}: ({x:.2f}, {y:.2f})")



    #画图
    # 生成边缘端坐标
    # endpoints = generate_uniform_endpoints(num_endpoints)

    # 提取 x 和 y 坐标
    x_coords = [x for x, y in endpoints]
    y_coords = [y for x, y in endpoints]

    # 绘制坐标点
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    for x, y in endpoints:
        circle = Circle((x, y), 150, color='blue', alpha=0.2)  # 添加圆圈
        ax.add_patch(circle)
    plt.scatter(x_coords, y_coords, color='blue', label="边缘端")
    plt.xlabel('X 坐标 (米)')
    plt.ylabel('Y 坐标 (米)')
    plt.title("边缘端均匀分布")
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 500)
    plt.ylim(0, 500)
    plt.show()


