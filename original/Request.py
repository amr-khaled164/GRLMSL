import numpy as np
# import random
# from Microservice import *


class Request:

    def __init__(self, req_id, req_ms, req_task, adj, begin_ms, deadline, mask):

        self.req_id = req_id  # 注：请求的唯一id（对内）
        self.req_name = 'Req{}'.format(req_id)  # 注：请求的唯一名称（对外）
        self.req_ms = req_ms  # 注：请求所涉及的微服务,list[]
        self.req_task = req_task  # 注：以及该微服务（下标一致）执行的任务，list[]
        self.req_ms_number = len(req_ms)  # 注：请求所涉及微服务的数量
        self.adj = adj  # 注：邻接矩阵，用于描述请求调用图（微服务间的调用关系)，行列下标与微服务下标一致
        self.begin_ms = begin_ms  # 注：入口微服务
        self.deadline = deadline  # 注：请求的延迟约束
        self.mask = mask  # 注：请求可用动作下标集合，用于DQN输出动作的筛选
        self.req_features = 3  # 注：请求调用图节点（微服务）的特征数
        self.req_feature_matrix = np.zeros((self.req_ms_number, self.req_features))  # 注：节点特征矩阵
        self.route_road = None  # 注：执行路径

    def produce_req_feature_matrix(self):

        for i in range(self.req_ms_number):
            self.req_feature_matrix[i][0] = self.req_ms[i].ms_id
            self.req_feature_matrix[i][1] = self.req_task[i].t_id
            self.req_feature_matrix[i][2] = self.req_ms[i].instance_number

        return self.req_feature_matrix

    def produce_route_road(self, start, visited=None, path=None, paths=None):

        """
            深度优先为准，从入口走到出口，可以建模选择关系（if，else带来的不同执行路径）
            self.req_ms=[Ms3,Ms4,Ms6]
            self.req_task=[Ms3_T1,Ms4_T0,Ms6_T1]
            self.adj=[[0, 0, 1], [1, 0, 1], [0, 0, 0]],按照3，4，6的顺序
            self.begin_ms=Ms4
        """

        if visited is None:
            visited = set()
        if path is None:
            path = []
        if paths is None:
            paths = []

        # 注：标记当前节点为已访问
        visited.add(start)
        path.append(start)

        # 注：如果当前节点没有邻居，说明到达路径终点
        adj_ms = self.adj[self.req_ms.index(start)]
        if sum(adj_ms) == 0:
            paths.append(path.copy())
        else:
            # 递归访问所有未访问的邻居
            for i in range(self.req_ms_number):
                if adj_ms[i] != 0 and self.req_ms[i] not in visited:
                    self.produce_route_road(self.req_ms[i], visited, path, paths)

        # 注：回溯：移除当前节点，标记为未访问
        path.pop()
        visited.remove(start)

        return paths

    def get_begin_task(self):

        return  self.req_task[self.req_ms.index(self.begin_ms)]

    def print_information(self):

        print('request', self.req_name, 'call graph')
        print('Its include microservice')
        for i in range(self.req_ms_number):
            print(self.req_ms[i].ms_name, ', the task it deal with: ', self.req_task[i].t_name, sep='')
        print('the edge between ms')
        for i in range(self.req_ms_number):
            for j in range(self.req_ms_number):
                if self.adj[i][j] == 1:
                    print('edge exist is:', self.req_ms[i].ms_name, '->', self.req_ms[j].ms_name)
        print('the begin microservice: {}'.format(self.begin_ms.ms_name))
        print('the deadline: {}ms'.format(self.deadline))
        print('the valid action index:', self.mask)
        paths = self.produce_route_road(self.begin_ms)
        print('the route roads include:')
        for path in paths:
            print([ms.ms_name for ms in path])
        print('Its feature matrix:')
        print(self.produce_req_feature_matrix())

