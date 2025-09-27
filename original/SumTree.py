import pickle

import numpy as np


# 注：SumTree
# 注：a binary tree data structure where the parent’s value is the sum of its children
class SumTree:
    def __init__(self, capacity):
        # 注：指向data数组的下标，可循环
        self.write = 0
        # 注：设定容量
        self.capacity = capacity
        # 注：from which index to store transitions（指向存储第一个transition，即经验元组的位置）
        self.data_start = 2**((self.capacity - 1).bit_length()) - 1
        # 注：**代表指数运算，如2**10=》2的10次=1024；bit_length()查询二进制的位数或长度
        # 注：返回全0的数组[0,0,...]；初始化sumTree，并指定大小；树从根节点开始依次存储。
        self.tree = np.zeros(self.data_start + self.capacity, dtype=np.float64)
        # store experiences，用于存储经验元组的数组
        self.data = np.zeros(capacity, dtype=object)
        # 注：数据条目数<=capacity，当前的存储数量
        self.n_entries = 0
        self.priority_max = 1.  # 注：记录最大优先级的值

    # 注：依次往上更新至根节点，update to the root node
    def _propagate(self, idx):
        parent = (idx - 1) // 2
        left, right = 2 * parent + 1, 2 * parent + 2
        self.tree[parent] = self.tree[left] + self.tree[right]
        assert self.tree[parent] == self.tree[left] + self.tree[right]
        if parent != 0:  # 注：是否为根节点
            self._propagate(parent)

    # 注：find transitions on leaf node，采样的寻找过程
    def _retrieve(self, idx, s):  # 注：idx初始值为0，从根节点开始，s为寻找的对象（各区间产生的随机值）
        left = 2 * idx + 1
        right = left + 1

        # 注：判断是否到达叶子节点（叶子节点存储经验元组自身）
        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]: # 注：往左找
            return self._retrieve(left, s)
        else: # 注：往右找
            return self._retrieve(right, s - self.tree[left])

    # 注：返回优先级总和，即根节点数据
    def total(self):
        return self.tree[0]

    # 注：存储，store priority and transitions
    def add(self, data):
        self.data[self.write] = data

        idx = self.write + self.data_start  # 注：sumTree中的存储下标
        self.update(idx, self.priority_max)

        self.write += 1
        self.write %= self.capacity  # 注：循环指针，当存满时，返回开头，进行覆盖存储

        if self.n_entries < self.capacity:
            self.n_entries += 1
        assert self.total() == self.tree[self.data_start:].sum()

    # 注；更新优先级，update priority
    def update(self, idx, p):
        self.tree[idx] = p  # 注：更新idx指向的经验元组的优先级
        if p > self.priority_max:
            self.priority_max = p
        self._propagate(idx)  # 注：向上更新整个sumTree
        assert abs(self.total() - self.tree[self.data_start:].sum()) < 1e-5
        # assert self.total() == self.tree[self.data_start:].sum()

    # 注：取经验元组及其优先级，get priority and transition
    def get(self, s):
        idx = self._retrieve(0, s)  # 注：idx是sumTree中的下标
        data_idx = idx - self.data_start

        return idx, self.tree[idx], self.data[data_idx]

    def save(self, pattern_type, break_point):
        save_dict = {
            'write': self.write,
            'tree': self.tree,
            'data': self.data,
            'n': self.n_entries,
            'p': self.priority_max
        }
        with open('./save/save_tree_m{}_b{}.plk'.format(pattern_type,break_point),'wb') as file:
            pickle.dump(save_dict,file)

    def load(self,pattern_type,break_point):
        with open('./save/save_tree_m{}_b{}.plk'.format(pattern_type,break_point),'rb') as file:
            save_data=pickle.load(file)
        self.write=save_data['write']
        self.tree=save_data['tree']
        self.data=save_data['data']
        self.n_entries=save_data['n']
        self.priority_max=save_data['p']

# if __name__ == "__main__":
#
#     sumTree = SumTree(8)
#     sumTree.add(('s0','a0','r0','s1'))
#     sumTree.add(('s1', 'a1', 'r1', 's2'))
#     sumTree.add(('s2', 'a2', 'r2', 's3'))
#     sumTree.add(('s3', 'a3', 'r3', 's4'))
#     sumTree.add(('s4', 'a4', 'r4', 's5'))
#     idx,p,data = sumTree.get(1)
#     sumTree.update(idx,2.6)
#     sumTree.save()
#
#     sumTree2=SumTree(8)
#     sumTree2.load()
#     print('test')

