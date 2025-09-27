import heapq
# import numpy as np


def dijkstra(adj, start, end):
    # 初始化距离表，所有节点距离为无穷大
    distances = {i: float('inf') for i in range(len(adj))}
    distances[start] = 0  # 起点到自身的距离为 0

    # 优先队列，存储 (距离, 节点)
    priority_queue = [(0, start)]

    while priority_queue:
        # 取出当前距离最小的节点
        current_distance, current_node = heapq.heappop(priority_queue)

        # 如果当前节点是目标节点，提前返回结果
        if current_node == end:
            return current_distance

        # 如果当前距离大于已知距离，跳过
        if current_distance > distances[current_node]:
            continue

        # 遍历当前节点的邻居
        for neighbor in range(len(adj[current_node])):
            weight = adj[current_node][neighbor]
            if weight == 1e8:
                continue
            distance = current_distance + weight

            # 如果找到更短的路径，更新距离表
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))

    # 如果未找到目标节点，返回无穷大
    return float('inf')


# if __name__ == "__main__":
#     # 示例图的邻接表表示
#     edge_features_matrix = np.array([[[0], [3], [-1], [2], [-1]], [[3], [0], [2], [2], [3]],
#     [[-1], [2], [0], [-1], [3]], [[2], [2], [-1], [0], [4]], [[-1], [3], [3], [4], [0]]])
#     print(edge_features_matrix)
#     edge_features_matrix = np.reshape(edge_features_matrix, (5, 5))
#     print(edge_features_matrix)
#
#     # 计算从节点 A 到节点 C 的最短距离
#     start_node = 1
#     end_node = 1
#     shortest_distance = dijkstra(edge_features_matrix, start_node, end_node)
#
#     # 输出结果
#     if shortest_distance != float('inf'):
#         print(f"从节点 {start_node} 到节点 {end_node} 的最短距离是: {shortest_distance}")
#     else:
#         print(f"节点 {start_node} 和节点 {end_node} 之间没有路径。")
