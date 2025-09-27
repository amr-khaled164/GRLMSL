import collections
import csv
import math
import random
import time

from numpy import dtype

from original.Node import Node
from original.Microservice import *
import numpy as np
from original.Request import Request
import torch
from original.Route import dijkstra
from original.User import User
from original.AdvancedAgent import AdvancedAgent
from original.Argument import get_options
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

seed_value = 1   # 设定随机数种子

np.random.seed(seed_value)
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)  # 为了禁止hash随机化，使得实验可复现。

torch.manual_seed(seed_value)     # 为CPU设置随机种子
torch.cuda.manual_seed(seed_value)      # 为当前GPU设置随机种子（只用一块GPU）
torch.cuda.manual_seed_all(seed_value)   # 为所有GPU设置随机种子（多块GPU）

torch.backends.cudnn.deterministic = True


class SP:

    def __init__(self, args):

        # self.args = args  # 注：参数集合

        self.node_number = 16  # 注：部署节点数
        self.nodes = []  # 注：部署节点集合
        self.create_node()  # 注：初始化部署节点
        self.node_features = 96  # 部署节点特征数
        # 节点特征矩阵，表示实例部署情况和实例负载。实例部署，值为实例的1+load；否则为0。初始化为全0，表示未进行任意部署
        self.node_feature_matrix = np.zeros((self.node_number, self.node_features), dtype=np.float32)
        # 注：部署节点的邻接矩阵
        self.adj = np.array([[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                             [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                             [1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                             [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1],
                             [0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1],
                             [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
                             [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0]])
        # 注：带权重信息（部署节点间的网络延迟，毫秒ms）邻接矩阵
        self.edge_feature_matrix = np.array([[0, 4, 4, 6, 1e8, 1e8, 1e8, 1e8, 1e8, 1e8, 1e8, 1e8, 1e8, 1e8, 1e8, 1e8],
                                             [4, 0, 1e8, 3, 1e8, 10, 1e8, 12, 1e8, 1e8, 1e8, 1e8, 1e8, 1e8, 1e8, 1e8],
                                             [4, 1e8, 0, 3, 1e8, 1e8, 1e8, 1e8, 1e8, 1e8, 1e8, 1e8, 1e8, 1e8, 10, 12],
                                             [6, 3, 3, 0, 1e8, 1e8, 1e8, 20, 1e8, 1e8, 1e8, 22, 1e8, 1e8, 1e8, 20],
                                             [1e8, 1e8, 1e8, 1e8, 0, 4, 4, 1e8, 1e8, 1e8, 1e8, 1e8, 1e8, 1e8, 1e8, 1e8],
                                             [1e8, 10, 1e8, 1e8, 4, 0, 6, 3, 1e8, 1e8, 1e8, 1e8, 1e8, 1e8, 1e8, 1e8],
                                             [1e8, 1e8, 1e8, 1e8, 4, 6, 0, 3, 1e8, 1e8, 1e8, 1e8, 1e8, 1e8, 1e8, 1e8],
                                             [1e8, 12, 1e8, 20, 1e8, 3, 3, 0, 1e8, 12, 1e8, 20, 1e8, 1e8, 1e8, 1e8],
                                             [1e8, 1e8, 1e8, 1e8, 1e8, 1e8, 1e8, 1e8, 0, 4, 4, 6, 1e8, 1e8, 1e8, 1e8],
                                             [1e8, 1e8, 1e8, 1e8, 1e8, 1e8, 10, 12, 4, 0, 1e8, 3, 1e8, 1e8, 1e8, 1e8],
                                             [1e8, 1e8, 1e8, 1e8, 1e8, 1e8, 1e8, 1e8, 4, 1e8, 0, 3, 1e8, 10, 1e8, 12],
                                             [1e8, 1e8, 1e8, 22, 1e8, 1e8, 1e8, 20, 6, 3, 3, 0, 1e8, 1e8, 1e8, 20],
                                             [1e8, 1e8, 1e8, 1e8, 1e8, 1e8, 1e8, 1e8, 1e8, 1e8, 1e8, 1e8, 0, 4, 4, 1e8],
                                             [1e8, 1e8, 1e8, 1e8, 1e8, 1e8, 1e8, 1e8, 1e8, 1e8, 1e8, 1e8, 4, 0, 6, 3],
                                             [1e8, 1e8, 10, 1e8, 1e8, 1e8, 1e8, 1e8, 1e8, 1e8, 1e8, 1e8, 4, 6, 0, 3],
                                             [1e8, 1e8, 12, 20, 1e8, 1e8, 1e8, 1e8, 1e8, 1e8, 12, 20, 1e8, 3, 3, 0]])

        self.Microservice_number = 8  # 注：系统微服务数
        self.Microservice_Task_number = [3, 5, 3, 3, 3, 2, 2, 3]  # 注：微服务暴露的任务数集合
        self.Microservice_Instance_number = [12, 12, 12, 12, 12, 12, 12, 12]  # 注：系统微服务实例数集合
        self.Microservices = []  # 注：系统微服务集合
        self.create_micro()  # 注：初始化系统微服务

        self.req_number = 7  # 注：系统请求数量
        self.reqs = []  # 注：系统请求集合
        self.create_req()  # 注：初始化系统请求

        # self.request_rate = [1, 3]  # 注：每个时间步的请求率范围

        self.users_per_node = 20  # 注：每个部署节点附近的用户数
        self.Users = []  # 注：系统用户集合
        self.create_user()  # 注：初始化系统用户

        # 注：决策智能体（微服务实例选择）
        self.agent = AdvancedAgent(args)

        self.my_deploy()  # 注：初始化节点部署情况

    def create_node(self):

        for i in range(self.node_number):
            self.nodes.append(Node(i))

    def create_micro(self):

        # 微服务任务处理时间矩阵
        task_delay = [[468, 389, 337],
                      [324, 345, 445, 68, 455],
                      [253, 149, 325],
                      [385, 456, 460],
                      [389, 353, 43],
                      [437, 49],
                      [136, 171],
                      [430, 280, 34]]

        for i in range(self.Microservice_number):
            ms = Microservice(i)
            tasks = [Task(ms, j, task_delay[i][j]) for j in range(self.Microservice_Task_number[i])]
            ms.add_task(tasks)
            instances = [Instance(ms, j) for j in range(self.Microservice_Instance_number[i])]
            ms.add_instance(instances)
            self.Microservices.append(ms)

    def mask_update(self, mask):

        # [0, 0, 0, 0, 1, 1, 0, 1]
        mask_update = []

        for m in mask:
            if m == 0:
                mask_update.append([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            else:
                mask_update.append([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        return mask_update

    def create_req(self):

        reqs_delay = [966, 1296, 1277, 1420, 372, 1224, 626]
        deadline_factor = 1.4

        # req0
        req0_ms = [self.Microservices[ms] for ms in [1, 2, 4]]
        req0_task = [self.Microservices[ms].tasks[t] for ms, t in [(1, 0), (2, 0), (4, 0)]]
        req0_adj = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        req0_mask = self.mask_update([0, 1, 1, 0, 1, 0, 0, 0])
        # req0_mask = np.array([1 if i in range(0, 81) else 0 for i in range(198)])
        req0 = Request(0, req0_ms, req0_task, req0_adj, req0_ms[0], reqs_delay[0]*deadline_factor, req0_mask)
        self.reqs.append(req0)

        # req1
        req1_ms = [self.Microservices[ms] for ms in [3, 6, 7, 1]]
        req1_task = [self.Microservices[ms].tasks[t] for ms, t in [(3, 0), (6, 0), (7, 0), (1, 1)]]
        req1_adj = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
        req1_mask = self.mask_update([0, 1, 0, 1, 0, 0, 1, 1])
        req1 = Request(1, req1_ms, req1_task, req1_adj, req1_ms[0], reqs_delay[1]*deadline_factor, req1_mask)
        self.reqs.append(req1)

        # req2
        req2_ms = [self.Microservices[ms] for ms in [0, 3, 4]]
        req2_task = [self.Microservices[ms].tasks[t] for ms, t in [(0, 0), (3, 1), (4, 1)]]
        req2_adj = req0_adj  # np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        req2_mask = self.mask_update([1, 0, 0, 1, 1, 0, 0, 0])
        req2 = Request(2, req2_ms, req2_task, req2_adj, req2_ms[0], reqs_delay[2]*deadline_factor, req2_mask)
        self.reqs.append(req2)

        # req3
        req3_ms = [self.Microservices[ms] for ms in [0, 1, 2, 5]]
        req3_task = [self.Microservices[ms].tasks[t] for ms, t in [(0, 1), (1, 2), (2, 1), (5, 0)]]
        req3_adj = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
        req3_mask = self.mask_update([1, 1, 1, 0, 0, 1, 0, 0])
        req3 = Request(3, req3_ms, req3_task, req3_adj, req3_ms[0], reqs_delay[3]*deadline_factor, req3_mask)
        self.reqs.append(req3)

        # req4
        req4_ms = [self.Microservices[ms] for ms in [5, 4, 7]]
        req4_task = [self.Microservices[ms].tasks[t] for ms, t in [(5, 1), (4, 2), (7, 1)]]
        req4_adj = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        req4_mask = self.mask_update([0, 0, 0, 0, 1, 1, 0, 1])
        req4 = Request(4, req4_ms, req4_task, req4_adj, req4_ms[0], reqs_delay[4]*deadline_factor, req4_mask)
        self.reqs.append(req4)

        # req5
        req5_ms = [self.Microservices[ms] for ms in [0, 1, 2, 3, 7]]
        req5_task = [self.Microservices[ms].tasks[t] for ms, t in [(0, 2), (1, 3), (2, 2), (3, 2), (7, 2)]]
        req5_adj = np.array([[0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0]])
        req5_mask = self.mask_update([1, 1, 1, 1, 0, 0, 0, 1])
        req5 = Request(5, req5_ms, req5_task, req5_adj, req5_ms[0], reqs_delay[5]*deadline_factor, req5_mask)
        self.reqs.append(req5)

        # req6
        req6_ms = [self.Microservices[ms] for ms in [1, 6]]
        req6_task = [self.Microservices[ms].tasks[t] for ms, t in [(1, 4), (6, 1)]]
        req6_adj = np.array([[0, 1], [0, 0]])
        req6_mask = self.mask_update([0, 1, 0, 0, 0, 0, 1, 0])
        req6 = Request(6, req6_ms, req6_task, req6_adj, req6_ms[0], reqs_delay[6]*deadline_factor, req6_mask)
        self.reqs.append(req6)

    def create_user(self):

        # 用户access时间
        access_delay = [[1, 6, 3, 3, 6, 6, 3, 5, 2, 4, 3, 5, 1, 1, 3, 3, 6, 4, 1, 4],
                        [1, 1, 2, 5, 3, 3, 6, 1, 3, 6, 2, 1, 3, 1, 3, 4, 3, 4, 4, 2],
                        [2, 1, 6, 2, 4, 2, 5, 2, 2, 3, 6, 1, 5, 1, 2, 4, 5, 3, 4, 3],
                        [1, 6, 1, 6, 2, 2, 4, 1, 5, 4, 3, 6, 2, 2, 3, 2, 2, 5, 1, 4],
                        [1, 1, 3, 6, 5, 6, 1, 6, 5, 3, 2, 2, 4, 2, 2, 3, 4, 2, 2, 6],
                        [5, 6, 5, 1, 3, 1, 1, 6, 4, 2, 3, 5, 2, 2, 6, 6, 1, 6, 1, 6],
                        [5, 6, 6, 6, 2, 2, 4, 3, 5, 5, 2, 6, 4, 4, 5, 1, 3, 2, 3, 6],
                        [2, 2, 5, 2, 2, 1, 6, 5, 5, 5, 5, 6, 3, 1, 6, 2, 3, 3, 1, 3],
                        [2, 1, 2, 2, 2, 1, 4, 5, 3, 1, 5, 3, 6, 4, 1, 5, 1, 2, 4, 2],
                        [3, 6, 1, 1, 6, 1, 3, 2, 5, 2, 1, 6, 4, 1, 1, 5, 1, 2, 2, 6],
                        [2, 2, 5, 5, 3, 3, 3, 5, 1, 4, 6, 3, 5, 4, 1, 1, 2, 3, 4, 4],
                        [6, 4, 1, 4, 4, 3, 1, 3, 2, 3, 2, 1, 2, 6, 4, 5, 3, 5, 3, 5],
                        [4, 5, 3, 6, 2, 5, 6, 6, 2, 6, 5, 3, 1, 4, 4, 5, 6, 3, 6, 4],
                        [5, 4, 5, 1, 1, 2, 4, 6, 6, 3, 4, 2, 6, 2, 1, 5, 3, 5, 6, 2],
                        [4, 3, 5, 5, 2, 2, 3, 1, 5, 4, 6, 6, 4, 1, 4, 4, 2, 3, 4, 6],
                        [4, 1, 4, 4, 3, 5, 4, 5, 4, 1, 1, 4, 3, 4, 2, 6, 2, 2, 4, 1]]

        for i in range(self.node_number):
            for j in range(self.users_per_node):
                u_id = i*self.users_per_node+j
                self.Users.append(User(u_id, self.nodes[i], access_delay[i][j]))

    def deploy_mirco_instance(self, node_id, micro_id, instance_id):

        # 部署实例
        node = self.nodes[node_id]
        micro = self.Microservices[micro_id]
        instance = micro.instances[instance_id]
        node.deploy_instance([instance])

    def my_deploy(self):

        # Ms0
        self.deploy_mirco_instance(0, 0, 0)
        self.deploy_mirco_instance(1, 0, 1)
        self.deploy_mirco_instance(4, 0, 2)
        self.deploy_mirco_instance(6, 0, 3)
        self.deploy_mirco_instance(8, 0, 4)
        self.deploy_mirco_instance(9, 0, 5)
        self.deploy_mirco_instance(10, 0, 6)
        self.deploy_mirco_instance(11, 0, 7)
        self.deploy_mirco_instance(12, 0, 8)
        self.deploy_mirco_instance(13, 0, 9)
        self.deploy_mirco_instance(14, 0, 10)
        self.deploy_mirco_instance(15, 0, 11)
        # Ms1
        self.deploy_mirco_instance(0, 1, 0)
        self.deploy_mirco_instance(1, 1, 1)
        self.deploy_mirco_instance(2, 1, 2)
        self.deploy_mirco_instance(3, 1, 3)
        self.deploy_mirco_instance(4, 1, 4)
        self.deploy_mirco_instance(7, 1, 5)
        self.deploy_mirco_instance(8, 1, 6)
        self.deploy_mirco_instance(9, 1, 7)
        self.deploy_mirco_instance(11, 1, 8)
        self.deploy_mirco_instance(12, 1, 9)
        self.deploy_mirco_instance(13, 1, 10)
        self.deploy_mirco_instance(15, 1, 11)
        # Ms2
        self.deploy_mirco_instance(3, 2, 0)
        self.deploy_mirco_instance(4, 2, 1)
        self.deploy_mirco_instance(5, 2, 2)
        self.deploy_mirco_instance(7, 2, 3)
        self.deploy_mirco_instance(8, 2, 4)
        self.deploy_mirco_instance(9, 2, 5)
        self.deploy_mirco_instance(10, 2, 6)
        self.deploy_mirco_instance(11, 2, 7)
        self.deploy_mirco_instance(12, 2, 8)
        self.deploy_mirco_instance(13, 2, 9)
        self.deploy_mirco_instance(14, 2, 10)
        self.deploy_mirco_instance(15, 2, 11)
        # Ms3
        self.deploy_mirco_instance(0, 3, 0)
        self.deploy_mirco_instance(1, 3, 1)
        self.deploy_mirco_instance(2, 3, 2)
        self.deploy_mirco_instance(3, 3, 3)
        self.deploy_mirco_instance(4, 3, 4)
        self.deploy_mirco_instance(5, 3, 5)
        self.deploy_mirco_instance(7, 3, 6)
        self.deploy_mirco_instance(8, 3, 7)
        self.deploy_mirco_instance(9, 3, 8)
        self.deploy_mirco_instance(10, 3, 9)
        self.deploy_mirco_instance(14, 3, 10)
        self.deploy_mirco_instance(15, 3, 11)
        # Ms4
        self.deploy_mirco_instance(0, 4, 0)
        self.deploy_mirco_instance(1, 4, 1)
        self.deploy_mirco_instance(2, 4, 2)
        self.deploy_mirco_instance(5, 4, 3)
        self.deploy_mirco_instance(6, 4, 4)
        self.deploy_mirco_instance(7, 4, 5)
        self.deploy_mirco_instance(8, 4, 6)
        self.deploy_mirco_instance(10, 4, 7)
        self.deploy_mirco_instance(11, 4, 8)
        self.deploy_mirco_instance(12, 4, 9)
        self.deploy_mirco_instance(14, 4, 10)
        self.deploy_mirco_instance(15, 4, 11)
        # Ms5
        self.deploy_mirco_instance(0, 5, 0)
        self.deploy_mirco_instance(1, 5, 1)
        self.deploy_mirco_instance(2, 5, 2)
        self.deploy_mirco_instance(3, 5, 3)
        self.deploy_mirco_instance(5, 5, 4)
        self.deploy_mirco_instance(6, 5, 5)
        self.deploy_mirco_instance(7, 5, 6)
        self.deploy_mirco_instance(8, 5, 7)
        self.deploy_mirco_instance(10, 5, 8)
        self.deploy_mirco_instance(11, 5, 9)
        self.deploy_mirco_instance(12, 5, 10)
        self.deploy_mirco_instance(14, 5, 11)
        # Ms6
        self.deploy_mirco_instance(0, 6, 0)
        self.deploy_mirco_instance(3, 6, 1)
        self.deploy_mirco_instance(4, 6, 2)
        self.deploy_mirco_instance(6, 6, 3)
        self.deploy_mirco_instance(7, 6, 4)
        self.deploy_mirco_instance(8, 6, 5)
        self.deploy_mirco_instance(9, 6, 6)
        self.deploy_mirco_instance(11, 6, 7)
        self.deploy_mirco_instance(12, 6, 8)
        self.deploy_mirco_instance(13, 6, 9)
        self.deploy_mirco_instance(14, 6, 10)
        self.deploy_mirco_instance(15, 6, 11)
        # Ms7
        self.deploy_mirco_instance(1, 7, 0)
        self.deploy_mirco_instance(2, 7, 1)
        self.deploy_mirco_instance(3, 7, 2)
        self.deploy_mirco_instance(6, 7, 3)
        self.deploy_mirco_instance(7, 7, 4)
        self.deploy_mirco_instance(8, 7, 5)
        self.deploy_mirco_instance(10, 7, 6)
        self.deploy_mirco_instance(11, 7, 7)
        self.deploy_mirco_instance(12, 7, 8)
        self.deploy_mirco_instance(13, 7, 9)
        self.deploy_mirco_instance(14, 7, 10)
        self.deploy_mirco_instance(15, 7, 11)

        self.produce_node_feature_matrix()

    def produce_node_feature_matrix(self):

        # 生成节点特征矩阵
        i = 0
        for ms in self.Microservices:
            for instance in ms.instances:
                self.node_feature_matrix[instance.node.n_id][i] = instance.update_weight_time() + 1
                i += 1

        return self.node_feature_matrix

    def noisy_infer(self,data):

        mu=0
        sigma=25
        noisy_data=data+np.random.normal(mu,sigma,data.shape)
        return noisy_data

    def get_state(self):

        # 生成一个随机的用户请求
        req = random.sample(self.reqs, 1)[0]
        user = random.sample(self.Users, 1)[0]
        # print('req{}, user{}'.format(req.req_id,user.u_id))

        sp_node_feature_matrix = torch.tensor(self.produce_node_feature_matrix())
        # sp_node_feature_matrix = torch.tensor(self.noisy_infer(self.produce_node_feature_matrix()),dtype=torch.float32) #noisy infer
        sp_edge_feature_matrix = torch.tensor(self.edge_feature_matrix, dtype=torch.float32)
        # sp_edge_feature_matrix = torch.tensor(self.noisy_infer(self.edge_feature_matrix), dtype=torch.float32) #noisy infer
        sp_adj = torch.tensor(self.adj+np.identity(self.node_number))  # 加自环

        req_node_feature_matrix = torch.tensor(req.produce_req_feature_matrix(), dtype=torch.float32)
        req_adj = torch.tensor(req.adj+np.identity(req.req_ms_number))
        req_mask = torch.tensor(req.mask)

        state = torch.tensor([req.deadline, user.source.n_id])

        return state, (sp_node_feature_matrix, sp_edge_feature_matrix, sp_adj), (req_node_feature_matrix, req_adj, req_mask), req, user

    def compute_balance(self):

        balances = []
        for ms in self.Microservices:
            balance = 0
            mean = sum([instance.update_weight_time() for instance in ms.instances]) / ms.instance_number
            for instance in ms.instances:
                balance += (instance.update_weight_time() - mean) ** 2
            balance = math.sqrt(balance / ms.instance_number)
            balances.append(balance)

        return balances

        # balance = 0
        # for ms in self.Microservices:
        #     mean = sum([instance.load for instance in ms.instances]) / ms.instance_number
        #     s1 = 0
        #     for instance in ms.instances:
        #         s1 += (instance.load - mean) ** 2
        #     balance += math.sqrt(s1/ms.instance_number)
        # balance /= self.Microservice_number
        #
        # return balance

    def execute_instance_plan(self, action, req, user):
        # print('req',req.produce_req_feature_matrix())
        # print('action',action)
        valid = []
        for i in range(len(action)):
            if action[i] > 0:
                valid.append(i)
        # print('valid',valid)
        origin = [ms.ms_id for ms in req.req_ms]
        origin.sort()
        # print('orign',origin)

        assert valid.__eq__(origin)

        # 执行实例选择计划 action = [0,1,2,0,4,0,0,0]
        access_time = user.access_time
        route_time = 0
        process_time = 0
        path = req.produce_route_road(req.begin_ms)[0]
        # path = random.sample(paths, 1)[0]
        # print('req execute path', [ms.ms_name for ms in path])
        user_source = user.source  # 注：用户就近接入的部署服务节点
        # print('user source node', user_source.n_name)
        pre_ms = path[0]
        pre_task = req.req_task[req.req_ms.index(pre_ms)]
        pre_instance = pre_ms.instances[action[pre_ms.ms_id]-1]
        pre_node = pre_instance.node
        # print('begin ms instance node', pre_node.n_name)
        adj_weight = self.edge_feature_matrix
        # print('adj with network delay(weight):',adj_weight)
        part_route_time = dijkstra(adj_weight, user_source.n_id, pre_node.n_id)
        # print('part route time between {} and {}: {}ms'.format(user_source.n_name, pre_node.n_name, part_route_time))
        route_time += part_route_time

        for i in range(1, len(path)):
            next_ms = path[i]
            next_task = req.req_task[req.req_ms.index(next_ms)]
            next_instance = next_ms.instances[action[next_ms.ms_id]-1]
            next_node = next_instance.node
            # print('pre ms instance ', pre_instance.ins_name)
            # print('pre ms instance load', pre_instance.load)
            # print('pre ms instance node', pre_node.n_name)
            # print('pre ms task', pre_task.t_name)
            weight_time = pre_instance.update_weight_time()
            part_process_time = pre_task.execute_time+weight_time
            # print('pre ms task time(part process time): {}ms'.format(part_process_time))
            # print('next ms instance node:', next_node.n_name)
            part_route_time = dijkstra(adj_weight, pre_node.n_id, next_node.n_id)
            # print('part route time between {} and {}: {}ms'.format
            # (pre_node.n_name, next_node.n_name, part_route_time))

            process_time += part_process_time
            pre_instance.load_num += 1
            pre_instance.tasks_process_queue.append(pre_task)
            route_time += part_route_time

            if i == len(path)-1:
                # print('last ms instance ', next_instance.ins_name)
                # print('last ms instance load', next_instance.load)
                # print('last ms task', next_task.t_name)
                weight_time = next_instance.update_weight_time()
                part_process_time = next_task.execute_time + weight_time
                # print('last ms task time(part process time): {}ms'.format(part_process_time))
                process_time += part_process_time
                next_instance.load_num += 1
                next_instance.tasks_process_queue.append(next_task)

            pre_ms = next_ms
            pre_task = next_task
            pre_instance = next_instance
            pre_node = next_node

        # print('access time: {}ms; route time: {}ms; process time: {}ms'.format
        # (access_time,route_time,process_time))
        delay = access_time+process_time+route_time
        # print('delay: {}ms'.format(delay))

        return delay

    def environment_reset(self):

        # 清空所有的负载
        for ms in self.Microservices:
            for instance in ms.instances:
                instance.load_num = 0
                instance.tasks_process_queue.clear()
                instance.weight_time = 0

        return self.get_state()

    def environment_update(self, action, req, user, t):

        # 执行请求的实例选择计划前的系统的负载均衡程度balance
        # pre_balance = self.compute_balance()
        pre_balances = self.compute_balance()

        # 执行实例选择计划 action = [0,1,2,0,4,0,0,0]
        delay = self.execute_instance_plan(action, req, user)
        # print('delay: {}ms'.format(delay))

        # 延迟子奖励和是否成功，默认1，即成功并给予奖励
        deadline_violation_reward = 1
        success = 1
        if delay > req.deadline:
            # deadline_violation = delay-req.deadline
            deadline_violation_reward = -1
            success = 0
        # print('deadline violation: {}ms'.format(deadline_violation))

        # 负载子奖励，默认为0，相当于无奖励也无惩罚
        balance_reward = 0
        # 执行请求的实例选择计划后的系统的负载均衡程度balance
        balances = self.compute_balance()
        for i in range(self.Microservice_number):
            # action=[0,1,2,0,4,0,0,0]
            if action[i] != 0:
                if pre_balances[i] != 0:
                    if balances[i] > pre_balances[i]:
                        balance_reward -= 1
                    else:
                        balance_reward += 1
        balance_reward /= req.req_ms_number

        # balance_reward = 1
        # balance = self.compute_balance()
        # if pre_balance == 0:
        #     balance_reward =0
        # elif balance > pre_balance:
        #     balance_reward = -1

        # 注：复合奖励函数的权重因子
        reward_alpha = 0.5
        reward = reward_alpha * deadline_violation_reward + (1 - reward_alpha) * balance_reward
        # print("reward:", reward)

        # 模拟微服务实例完成任务带来的负载下降,t也是一个超参数
        if t > 50:
            # print(t)
            for ms in self.Microservices:
                for ins in ms.instances:
                    load_num = ins.load_num
                    # random.seed(0)
                    r = random.randint(0, load_num)
                    # print('ms{}->ins{}, load_num:{}, r：{}'.format(ms.ms_id,ins.ins_id,load_num,r))
                    ins.tasks_process_queue = ins.tasks_process_queue[r:]
                    ins.load_num = load_num-r
                    ins.update_weight_time()

        return reward, self.get_state(), delay, success, sum(balances)/len(balances)

    def train_agent_more(self, episodes, break_point):

        self.agent.load(break_point-1)
        self.train_agent(episodes, break_point)

    def train_agent(self, episodes=20, break_point=0, break_episode=5):

        time_steps = 1000

        # 每个episode的time_steps个时间步，智能体获得的累计奖励，共episodes个
        sum_reward_list = []
        # 智能体每次交互产生的奖励，共episodes*time_steps个
        # reward_list = []

        # 每个episode的time_steps个时间步，智能体训练产生loss的均值，共episodes个
        train_average_loss_list = []
        # 智能体训练产生的loss
        # 第一个episode训练次数<1000，需要收集transition，后面的episode训练次数=1000，因为已经收集了够多的启动数据
        # train_loss_list = []

        # 每个episode的平均请求延迟
        average_delay_list = []
        # 智能体每次交互产生的请求延迟，共episodes*time_steps个
        # delay_list = []

        # 每个episode的平均系统负载均衡程度
        average_balance_list = []
        # 智能体每次交互产生的负载均衡，共episodes * (time_steps+1)个
        # balance_list = []

        # 每个episode的time_steps个时间步, 智能体调度请求的成功率，共episodes个
        success_rate_list = []
        # 每个episode、每个个时间步, 智能体调度请求的成功与否
        # success_list = []

        # break_point = 0
        # break_episode = 5

        for episode in range(episodes):

            sum_reward_per_episode = 0
            average_loss_per_episode = 0
            train_count_per_episode = 0
            average_delay_per_episode = 0
            average_balance_per_episode = 0
            success_rate_per_episode = 0
            # delays = ['episode_{}'.format(episode)]
            # balances = ['episode_{}'.format(episode), 0]
            # rewards = ['episode_{}'.format(episode)]
            # losses = ['episode_{}'.format(episode)]
            # successes = ['episode_{}'.format(episode)]

            state, (sp_node, sp_edge, sp_adj), (req_node, req_adj, req_mask), req, user = self.environment_reset()

            for t in range(time_steps):
                print('episode {}, interaction with environment in time step {}'.format(episode, t))
                action = self.agent.select_action(state, sp_node, sp_adj, sp_edge, req_node, req_adj, req_mask)
                (reward, (next_state, (next_sp_node, _, _), (next_req_node, next_req_adj, next_req_mask),
                          next_req, next_user), delay, success, balance) = self.environment_update(action, req, user, t)
                transition = (sp_node, sp_edge, sp_adj, req_node, req_adj, state, action, torch.FloatTensor([reward]),
                              next_sp_node, next_req_node, next_req_adj, next_state, next_req_mask, req.req_id)
                self.agent.buffer.push(transition)
                state, sp_node, req_node, req_adj, req_mask, req, user = (
                    next_state, next_sp_node, next_req_node, next_req_adj, next_req_mask, next_req, next_user)

                # rewards.append(reward)
                sum_reward_per_episode += reward
                # delays.append(delay)
                average_delay_per_episode += delay
                # balances.append(balance)
                average_balance_per_episode += balance
                # successes.append(success)
                success_rate_per_episode += success

                if self.agent.buffer.train_start():
                    # return
                    train_count_per_episode += 1
                    print('start training, episode {}, train {}'.format(episode, train_count_per_episode))
                    begin_time = time.time()
                    loss = self.agent.train()
                    end_time = time.time()
                    print('one train, use time:', end_time-begin_time)
                    # losses.append(loss)
                    average_loss_per_episode += loss

            sum_reward_list.append(sum_reward_per_episode)
            # reward_list.append(rewards)
            if train_count_per_episode != 0:
                average_loss_per_episode /= train_count_per_episode
            train_average_loss_list.append(average_loss_per_episode)
            # train_loss_list.append(losses)
            average_delay_per_episode /= time_steps
            average_delay_list.append(average_delay_per_episode)
            # delay_list.append(delays)
            average_balance_per_episode /= time_steps
            average_balance_list.append(average_balance_per_episode)
            # balance_list.append(balances)
            success_rate_per_episode /= time_steps
            success_rate_list.append(success_rate_per_episode)
            # success_list.append(successes)

            print('episode {}, reward：{}'.format(episode, sum_reward_per_episode))

            if (episode+1) % break_episode == 0:
                print('save agent for part training')

                self.agent.save(break_point)

                plt.title('train reward changing per episode {}'.format(break_point))
                plt.xlabel('episodes {}'.format(break_point))
                plt.ylabel('reward {}'.format(break_point))
                plt.plot(range(episode+1), sum_reward_list)
                plt.savefig('./result/train/reward_changing_train_{}.jpg'.format(break_point))
                plt.clf()

                with open('./result/train/train_data_{}.csv'.format(break_point), 'a+', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['train episodes: {}'.format(episode+1)])
                    writer.writerow(['sum reward per episode:'])
                    writer.writerow(sum_reward_list)
                    # writer.writerow(['reward list:'])
                    # writer.writerows(reward_list)
                    writer.writerow(['success rate per episode:'])
                    writer.writerow(success_rate_list)
                    # writer.writerow(['success list:'])
                    # writer.writerows(success_list)
                    writer.writerow(['average delay per episode'])
                    writer.writerow(average_delay_list)
                    # writer.writerow(['delay list:'])
                    # writer.writerows(delay_list)
                    writer.writerow(['average balance per episode'])
                    writer.writerow(average_balance_list)
                    # writer.writerow(['balance list:'])
                    # writer.writerows(balance_list)
                    writer.writerow(['average loss per episode:'])
                    writer.writerow(train_average_loss_list)
                    # writer.writerow(['train loss list:'])
                    # writer.writerows(train_loss_list)
                    writer.writerow(
                        ['================================================================================='])
                break_point += 1

    def test_agent(self, break_point=None):

        if break_point is not None:
            self.agent.load(break_point)
            print('load trained model')

        episodes = 10
        time_steps = 1000

        # 每个epoch的1000个时间步，智能体获得的累计奖励，共10个
        sum_reward_list = []
        average_delay_list = []
        average_balance_list = []
        success_rate_list = []

        # 设为评估模式（因为现在是测试），同时噪声网络在测试时不添加噪声
        self.agent.test_setting()

        print('begin test')
        begin_time = time.time()
        for episode in range(episodes):

            sum_reward_per_episode = 0
            average_delay_per_episode = 0
            average_balance_per_episode = 0
            success_rate_per_episode = 0

            state, (sp_node, sp_edge, sp_adj), (req_node, req_adj, req_mask), req, user = self.environment_reset()

            for t in range(time_steps):
                action = self.agent.select_action(state, sp_node, sp_adj, sp_edge, req_node, req_adj, req_mask)
                (reward, (next_state, (next_sp_node, _, _), (next_req_node, next_req_adj, next_req_mask),
                          next_req, next_user), delay, success, balance) = self.environment_update(action, req, user, t)
                state, sp_node, req_node, req_adj, req_mask, req, user = (
                    next_state, next_sp_node, next_req_node, next_req_adj, next_req_mask, next_req, next_user)
                sum_reward_per_episode += reward
                average_delay_per_episode += delay
                average_balance_per_episode += balance
                success_rate_per_episode += success

            sum_reward_list.append(sum_reward_per_episode)
            average_delay_list.append(average_delay_per_episode/time_steps)
            average_balance_list.append(average_balance_per_episode/time_steps)
            success_rate_list.append(success_rate_per_episode/time_steps)

        test_time = time.time()-begin_time
        print('complete test, test time:',test_time)

        # 取episodes轮测试的均值和最值
        mean_reward = sum(sum_reward_list)/episodes
        max_reward = max(sum_reward_list)
        min_reward = min(sum_reward_list)
        mean_success = sum(success_rate_list) / episodes
        max_success = max(success_rate_list)
        min_success = min(success_rate_list)
        mean_delay = sum(average_delay_list) / episodes
        max_delay = max(average_delay_list)
        min_delay = min(average_delay_list)
        mean_balance = sum(average_balance_list) / episodes
        max_balance = max(average_balance_list)
        min_balance = min(average_balance_list)

        plt.title('reward changing per episode (test)')
        plt.xlabel('episodes')
        plt.ylabel('reward')
        plt.plot(range(episodes), sum_reward_list)
        plt.savefig('./result/test/reward_changing_test.jpg')
        plt.clf()

        with open('result/test/test_data.csv', 'a+', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['test episodes: {}'.format(episodes)])
            writer.writerow(['sum reward per episode:'])
            writer.writerow(sum_reward_list)
            writer.writerow(['mean: {}'.format(mean_reward), 'max: {}'.format(max_reward),
                             'min: {}'.format(min_reward)])
            writer.writerow(['success rate per episode:'])
            writer.writerow(success_rate_list)
            writer.writerow(['mean: {}'.format(mean_success), 'max: {}'.format(max_success),
                             'min: {}'.format(min_success)])
            writer.writerow(['average delay per episode:'])
            writer.writerow(average_delay_list)
            writer.writerow(['mean: {}'.format(mean_delay), 'max: {}'.format(max_delay),
                             'min: {}'.format(min_delay)])
            writer.writerow(['average balance per episode:'])
            writer.writerow(average_balance_list)
            writer.writerow(['mean: {}'.format(mean_balance), 'max: {}'.format(max_balance),
                             'min: {}'.format(min_balance)])
            writer.writerow(['================================================================================='])

    def print_information(self):

        # 注：输出平台详细信息
        print('System Platform Information')
        print('Node number:', self.node_number)
        for node in self.nodes:
            node.print_information()
        print('Microservice number:', self.Microservice_number)
        for microservice in self.Microservices:
            microservice.print_information()
        print('Request number:', self.req_number)
        for req in self.reqs:
            req.print_information()
        print('User number', len(self.Users))
        for user in self.Users:
            user.print_information()


if __name__ == "__main__":

    print("SP")
    args = get_options()
    sp = SP(args)
    # sp.agent.q_num()
    # sp.train_agent()
    # sp.train_agent_more(20, 4)
    sp.test_agent(23)



