import pickle
import random
import numpy as np
from original.SumTree import SumTree
from collections import deque
# import argparse


class PatternMemory:

    def __init__(self, args, capacity, batch_size, pattern_type):
        # 注：pattern_type对应request分组
        self.pattern_type = pattern_type
        # self.args = args
        self.capacity = capacity
        self.batch_size = batch_size
        self.prioritized_er = args.prioritized_er
        if self.prioritized_er:  # 注：优先经验回放（Prioritized Experience Replay）
            self.tree = SumTree(self.capacity)  # 注：创建一个SumTree实例
            self.e = args.e  # 注：epsilon-ε，超参数，用于计算优先级，保证优先级不为0，即采样概率不为0
            self.alpha = args.per_alpha
            self.beta = args.per_beta
            self.beta_increment = args.beta_increment  # 注：alpha-α；beta-β；beta_increment：用于计算重要性采样权重的超参数
        else:
            self.buffer = deque(maxlen=self.capacity)  # 注：常规的经验回放池，使用队列实现，deque（double-end queue，双向队列）

    def sample_experience(self):
        if self.prioritized_er:
            result = self.sample()  # 注：如果是优先级经验回放池，调用sample方法
        else:
            result = zip(*random.sample(self.buffer, self.batch_size))  # 注：否则，随机采样
            '''
                *是Python中的一个解包操作符，用于将序列解包为单独的参数，如print(*[1,2,3]),print(1,2,3)
                *random.sample(self.buffer, batch_size)将采样得到的包含batch_size个元组的list解包[(s,a,r,s'),(),...]为
                batch_size个单独的元组(s,a,r,s'),(),...,每个元组是存储的经验
                zip用于将多个序列中的元素逐个匹配，返回一个元组构成的列表，例如zip([1,2,3],['a','b','c'])=>zip[(1,'a'),(2,'b'),(3,'c')]
                zip(*random.sample(self.buffer, batch_size))将采样得到的batch_size个元组(s,a,r,s'),(),...中的每个元素逐个配对，
                返回一个包含多个元组的zip对象，每个元组包含了batch_size个 s，a，r，s' 等信息->(s,s'...),(a,a'...)
            '''
        return result

    def sample(self):
        batch = []  # 注：to store transitions 存储采样的batch_size个样本（经验元组）
        idxs = []  # 注：to store index of transitions in tree 存储样本在SumTree中的下标
        priorities = []  # 注：to store priorities of transitions 存储样本的优先级

        segment = self.tree.total() / self.batch_size
        # 注：抽样时, 我们会将优先级总和（tree.total()）除以batch_size, 分成 batch_size 多个区间。每个区间随机生成一个数进行采样操作，共采样batch_size个样本

        self.beta = np.min([1., self.beta + self.beta_increment])  # 注：重要性采样权重相关超参数，两者选最小，beta随着采样会变大
        i = 0
        while True:
            # 注：循环，对划分的各区间进行采样，a是区间下限，b是区间上限
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)  # 注：返回a,b之间的随机浮点数，若a<=b则范围[a,b]，若a>=b则范围[b,a] ，a和b可以是实数。
            idx, p, data = self.tree.get(s)  # 注：返回采样样本的下标（树中的下标），优先级，数据（经验元组）
            assert data != 0
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)
            i += 1
            if i == self.batch_size:
                break
        assert len(batch) == self.batch_size
        sampling_probabilities = priorities / self.tree.total()  # 注：计算每个样本的采样概率，基于优先级
        is_weight = (self.tree.n_entries * sampling_probabilities) ** -self.beta  # 注：计算重要性采样权重，tree.n_entries为存储的经验元组个数
        is_weight /= is_weight.max()  # 注：权重归一化至0-1

        return batch, idxs, is_weight

    def update(self, idx, error):
        # 注：基于TD-error更新优先级
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def _get_priority(self, error):
        # 注：Proportional Prioritization，基于比例的优先级计算公式，基于TD-error
        # e->epsilon-ε，超参数，用于计算优先级，保证优先级不为0，即采样概率不为0
        # alpha，超参数，表示使用多少优先级，alpha=0表示不使用优先级，因为优先级均为1，是等概率的
        return (np.abs(error) + self.e) ** self.alpha

    def train_start(self):
        # 注：判断是否达到采样个数
        if self.prioritized_er:
            return self.tree.n_entries > self.batch_size  # batch_size
        else:
            return len(self.buffer) > self.batch_size

    def push(self, transition):
        # 注：添加
        if self.prioritized_er:
            self.tree.add(transition)
        else:
            self.buffer.append(transition)

    def save(self,  break_point):
        save_dict={
            'beta': self.beta
        }
        with open('./save/save_Memory_{}_b{}.plk'.format(self.pattern_type,break_point), 'wb') as file:
            pickle.dump(save_dict, file)
        self.tree.save(self.pattern_type,break_point)

    def load(self, break_point):
        with open('./save/save_Memory_{}_b{}.plk'.format(self.pattern_type,break_point), 'rb') as file:
            save_data=pickle.load(file)
        self.beta = save_data['beta']
        self.tree.load(self.pattern_type,break_point)



class Memory:
    # capacity, num_request, batch_size
    def __init__(self, args):

        self.args = args  # 注：参数集合
        self.capacity = args.buffer_capacity  # 注：存储总量
        self.num_request = args.num_request  # 注：请求数量，即子经验池数量
        self.batch_size = args.batch_size  # 注：每个子经验池的采样量，总采样量=batch_size * num_request
        self.transition_count = 0 # 注：存储的数量
        self.pattern_memories = self.initialize_buffers()   # 注：初始化子经验池 -> dict

    def get_beta(self):
        return self.pattern_memories[1].beta  # 超参数，用于计算重要性采样权重

    def initialize_buffers(self):
        # 注：初始化分组的经验回放池
        sub_capacity = int(self.capacity / self.num_request)
        sub_batch_size = int(self.batch_size/self.num_request)
        pattern_memories = {}
        for i in range(self.num_request):  # 注：num_request是对请求的分组（与梯度手术相关）
            pattern_memories[i] = PatternMemory(self.args, sub_capacity, sub_batch_size, i)  # 注：根据划分的不同请求创建对应的patternMemory
        return pattern_memories

    def push(self, transition):
        # 注：根据request，将transition放入对应经验回放池
        request_type = transition[-1]
        self.pattern_memories[request_type].push(transition)
        self.transition_count += 1

    def train_start(self):
        # 注：当所有pattern_memories都达到采样数量，可以开始训练时，则memory可以开始训练，即模型开始训练
        if_train_start = True
        for i in range(self.num_request):
            if not self.pattern_memories[i].train_start():  # 注：判断每个pattern_memories是否到达采样数量
                if_train_start = False

        return if_train_start

    def sample_experience(self):
        mini_batch_result = []
        idxs_result = []
        is_weights_result = []
        # 注：对每种pattern_memories进行采样，并合并采样结果
        for i in range(self.num_request):
            mini_batch, idxs, is_weights = self.pattern_memories[i].sample()
            # 注：extend(__iterable)，[0,4].extend([1,2])=>[0,4,1,2]
            mini_batch_result.extend(mini_batch)
            idxs_result.extend(idxs)
            is_weights_result.extend(is_weights)
        return mini_batch_result, idxs_result, is_weights_result

    def update(self, idxs, errors, pattern):
        # 注：需要更新的transitions在SumTree中的索引idxs以及它们的errors；pattern指定子经验池
        for i, idx in enumerate(idxs):
            self.pattern_memories[pattern].update(idx, errors[i])

    def save(self, break_point):
        save_dict = {
            "transition_count": self.transition_count
        }
        with open('./save/save_Memories_b{}.plk'.format(break_point), 'wb') as file:
            pickle.dump(save_dict, file)
        for pattern_memory in self.pattern_memories.values():
            pattern_memory.save(break_point)

    def load(self, break_point):
        with open('./save/save_Memories_b{}.plk'.format(break_point), 'rb') as file:
            save_date=pickle.load(file)
        self.transition_count=save_date['transition_count']
        for pattern_memory in self.pattern_memories.values():
            pattern_memory.load(break_point)
