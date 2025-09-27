
class Microservice:

    def __init__(self, ms_id):

        self.ms_id = ms_id  # 注：微服务的唯一id（对内）
        self.ms_name = 'Ms{}'.format(ms_id)  # 注：微服务的唯一名称（对外）
        self.tasks = []  # 注：微服务暴露的任务（接口）集合
        self.instances = []  # 注：微服务实例集合
        self.task_number = 0  # 注：微服务暴露的任务的数量
        self.instance_number = 0  # 注：微服务实例数

    def add_task(self, tasks):

        # 注：添加任务
        self.tasks.extend(tasks)
        self.task_number += len(tasks)

    def add_instance(self, instances):

        # 注：添加实例
        self.instances.extend(instances)
        self.instance_number += len(instances)

    def print_information(self):

        # 注：打印微服务的详细信息
        print('Microservice', self.ms_name)
        print('Its task number:', self.task_number)
        for task in self.tasks:
            task.print_information()
        print('Its instance number:', self.instance_number)
        for instance in self.instances:
            instance.print_information()


class Task:

    def __init__(self, ms, t_id, execute_time):

        self.ms = ms  # 注：任务所属微服务
        self.t_id = t_id  # 注：任务唯一id（对内）
        self.t_name = '{}_T{}'.format(ms.ms_name, t_id)  # 注：任务唯一名称（对外）
        self.execute_time = execute_time  # 注：执行任务所需时间

    def print_information(self):

        # 注：打印任务详细信息
        print('Task {}(execute time: {}ms)'.format(self.t_name, self.execute_time))


class Instance:

    def __init__(self, ms, ins_id):

        self.ms = ms  # 注：实例所属微服务
        self.ins_id = ins_id  # 注：实例唯一id（对内）
        self.ins_name = '{}_Ins{}'.format(ms.ms_name, ins_id)  # 注：实例唯一名称（对外）
        self.node = None  # 注：实例部署节点
        self.load_num = 0  # 注：实例运行负载（任务数）
        self.tasks_process_queue = []  # 注：实例的任务处理队列
        self.weight_time = 0

    def update_weight_time(self):

        self.weight_time = sum([task.execute_time for task in self.tasks_process_queue])
        return self.weight_time

    # def workload(self):
    #
    #     return  self.update_weight_time() #还是用原来的
    #     # self.update_weight_time()
    #     # # workloads = self.weight_time / 1000 if self.weight_time / 1000 <= 1 else 1
    #     # workloads = self.weight_time / 1000
    #     # return workloads

    def print_information(self):

        # 注：打印实例详细信息
        if self.node is None:
            print('Instance {}(Node deployed: None; Load: {})'.format(self.ins_name, self.load_num))
        else:
            print('Instance {}(Node deployed: {}; Load: {})'.format(self.ins_name, self.node.n_name, self.load_num))


# if __name__ == "__main__":
#
#     # Ms0
#     Ms0 = Microservice(0)
#     print('original:')
#     Ms0.print_information()
#
#     Ms0_T0 = Task(Ms0, 0, 12)
#     Ms0_T1 = Task(Ms0, 1, 10)
#     Ms0.add_task([Ms0_T0, Ms0_T1])
#
#     Ms0_Ins0 = Instance(Ms0, 0)
#     Ms0_Ins1 = Instance(Ms0, 1)
#     Ms0_Ins2 = Instance(Ms0, 2)
#     Ms0.add_instance([Ms0_Ins0, Ms0_Ins1, Ms0_Ins2])
#
#     print('last:')
#     Ms0.print_information()
