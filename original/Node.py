# from Microservice import *


class Node:

    def __init__(self, n_id):

        self.n_id = n_id  # 注：部署节点唯一id（对内）
        self.n_name = 'Node{}'.format(n_id)  # 注：部署节点唯一名称（对外）
        self.instance_deployed = []  # 注：部署在该节点的实例集合
        self.instance_deployed_number = 0  # 注：部署实例数

    def deploy_instance(self, instances):

        # 注：部署实例
        for instance in instances:
            self.instance_deployed.append(instance)
            self.instance_deployed_number += 1
            instance.node = self

    def undeploy_instance(self, instances):

        # 注：卸载实例
        for instance in instances:
            self.instance_deployed.remove(instance)
            self.instance_deployed_number -= 1
            instance.node = None

    def print_information(self):

        # 打印部署节点详细信息
        print('Node', self.n_name)
        print('Its deployed Instance numbers:', self.instance_deployed_number)
        for instance in self.instance_deployed:
            instance.print_information()


# if __name__ == "__main__":
#
#     Node0 = Node(0)
#     Node0.print_information()
#
#     Ms0 = Microservice(0)
#     Ms0_T0 = Task(Ms0, 0, 12)
#     Ms0_T1 = Task(Ms0, 1, 10)
#     Ms0.add_task([Ms0_T0, Ms0_T1])
#
#     Ms0_Ins0 = Instance(Ms0, 0)
#     Ms0_Ins1 = Instance(Ms0, 1)
#     Ms0_Ins2 = Instance(Ms0, 2)
#     Ms0.add_instance([Ms0_Ins0, Ms0_Ins1, Ms0_Ins2])
#
#     Ms1 = Microservice(1)
#     Ms1_T0 = Task(Ms1, 0, 17)
#     Ms1_T1 = Task(Ms1, 1, 20)
#     Ms1.add_task([Ms1_T0, Ms1_T1])
#
#     Ms1_Ins0 = Instance(Ms1, 0)
#     Ms1_Ins1 = Instance(Ms1, 1)
#     Ms1_Ins2 = Instance(Ms1, 2)
#     Ms1.add_instance([Ms1_Ins0, Ms1_Ins1, Ms1_Ins2])
#
#     Node0.deploy_instance([Ms0_Ins1, Ms1_Ins1])
#
#     Node0.print_information()
#
#     Node0.undeploy_instance([Ms0_Ins1])
#     Node0.print_information()
#     Ms0_Ins1.print_information()


