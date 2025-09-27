import torch
import numpy as np
import copy
import random
import time
'''
    论文 “Gradient Surgery for Multi-task Learning” 中提出的 PCGrad 方法应对了这个问题。
    当多个任务的梯度在更新时相互冲突时，PCGrad 主要考虑如何修整这些梯度，使它们不再互相矛盾。
    具体地，它通过将每个任务的梯度投影到与其他任务梯度冲突方向的正交平面上，来消除梯度之间的冲突。
    -》
    Temporal grouping with gradient surgery，梯度手术
    按照经验发生的时间间隔对经验进行存储和采样，分为不同时间组
    当从不同时间组的训练样本中提取梯度时，这些梯度可能会相互冲突，从而干扰彼此的性能。
    当梯度尺度存在显著的不一致性时，这种情况会加剧。与较小梯度相关的性能往往被具有较大梯度的性能所掩盖。
'''
class PCGrad():
    def __init__(self, optimizer, reduction='mean'):
        self._optim, self._reduction = optimizer, reduction
        return

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        '''
        clear the gradient of the parameters
        '''

        return self._optim.zero_grad(set_to_none=True)

    def step(self):
        '''
        update the parameters with the gradient
        '''

        return self._optim.step()

    def pc_backward(self, objectives):
        '''
        calculate the gradient of the parameters

        input:
        - objectives: a list of objectives
        '''

        before_computation = time.time()
        grads, shapes, has_grads = self._pack_grad(objectives)
        gradient_ct = time.time() - before_computation
        # print('gradient computation time: ', gradient_ct)

        before_gs = time.time()
        pc_grad = self._project_conflicting(grads, has_grads)
        gs_t = time.time() - before_gs
        # print('gradient surgery time: ', gs_t)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)
        return gradient_ct, gs_t

    # def _project_conflicting(self, grads, has_grads, shapes=None):
    #     shared = torch.stack(has_grads).prod(0).bool()
    #     pc_grad = copy.deepcopy(grads)
    #     for g_i in pc_grad:
    #         random.shuffle(grads)
    #         for g_j in grads:
    #             g_i_g_j = torch.dot(g_i, g_j)
    #             if g_i_g_j < 0:
    #                 g_i -= (g_i_g_j) * g_j / (g_j.norm() ** 2)
    #     merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
    #     if self._reduction:
    #         merged_grad[shared] = torch.stack([g[shared]
    #                                            for g in pc_grad]).mean(dim=0)
    #     elif self._reduction == 'sum':
    #         merged_grad[shared] = torch.stack([g[shared]
    #                                            for g in pc_grad]).sum(dim=0)
    #     else:
    #         exit('invalid reduction method')
    #
    #     merged_grad[~shared] = torch.stack([g[~shared]
    #                                         for g in pc_grad]).sum(dim=0)
    #     return merged_grad

    def _project_conflicting(self, grads, has_grads, shapes=None):
        shared = torch.stack(has_grads).prod(0).bool()
        grads = torch.stack(grads, dim=0)
        # pc_grad, num_task = copy.deepcopy(grads), len(grads)

        # torch.norm(input, p="fro", dim=None, keepdim=False, out=None, dtype=None),计算范数；
        # 在维度1对grads计算矩阵的Frobenius norm (Frobenius 范数)，就是矩阵A各项元素的绝对值平方的总和，并保留dim指定的维度
        grads_unit = grads / torch.norm(grads, dim=1, keepdim=True)  # normalize, shape: 64 * 228149

        # torch.mm(a, b) 是矩阵a和b矩阵相乘；grads_unit.T转置
        grads_mm = torch.mm(grads, grads_unit.T)  # g * f / |f|, shape: 64 * 64

        grads_mm[grads_mm >= 0] = 0  # fill elements larger than 0 as 0, no need to do GS

        proj = torch.mm(grads_mm, grads_unit)  # calculate projection, g * (f / |f|) * (f / |f|), shape: 64 * 228149

        pc_grad = grads - proj  # minus projection to get projected grads, shape: 64 * 228149

        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        if self._reduction:
            merged_grad[shared] = torch.stack([g[shared]
                                               for g in pc_grad]).mean(dim=0)
        elif self._reduction == 'sum':
            merged_grad[shared] = torch.stack([g[shared]
                                               for g in pc_grad]).sum(dim=0)
        else:
            exit('invalid reduction method')

        # ~运算符，按位“取反”运算符：对数据的每个二进制位取反，即把 1 变为 0，把 0 变为 1
        merged_grad[~shared] = torch.stack([g[~shared]
                                            for g in pc_grad]).sum(dim=0)
        return merged_grad

    def _set_grad(self, grads):
        '''
        set the modified gradients to the network
        '''

        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                p.grad = grads[idx]
                idx += 1
        return

    def _pack_grad(self, objectives):
        '''
        pack the gradient of the parameters of the network for each objective
        打包每个目标的网络参数梯度

        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grads, shapes, has_grads = [], [], []
        for obj in objectives:
            self._optim.zero_grad(set_to_none=True)
            obj.backward(retain_graph=True)
            grad, shape, has_grad = self._retrieve_grad()
            grads.append(self._flatten_grad(grad, shape))
            has_grads.append(self._flatten_grad(has_grad, shape))
            shapes.append(shape)
        return grads, shapes, has_grads

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            # numpy.prod(a, axis=None, dtype=None, out=None, keepdims=<class 'numpy._globals._NoValue'>) 返回给定轴上的数组元素的乘积
            # 默认计算所有元素的乘积
            length = np.prod(shape)
            # .view()返回对象的一个视图，视图的数据来自对象，视图自身并不用于该数据；传入shape可以改变视图的形状
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        # torch.flatten(input,start_dim=0,end_dim=-1) => input.flatten(start_dim=0,end_dim=-1)
        # 扁平化，假设你的数据为1维数据，那么这个数据天然就已经扁平化了，如果是2维数据，那么扁平化就是将2维数据变为1维数据，
        # 如果是3维数据，那么就要根据你自己所选择的“扁平化程度”来进行操作，
        # 假设需要全部扁平化，那么就直接将3维数据变为1维数据，如果只需要部分扁平化，那么有一维的数据不会进行扁平操作
        # 这里将grads里的每个g扁平化为一维向量，并拼接向量
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self):
        '''
        get the gradient of the parameters of the network with specific
        objective

        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grad, shape, has_grad = [], [], []
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                # tackle the multi-head scenario
                if p.grad is None:
                    shape.append(p.shape)
                    # torch.zeros_like:生成和括号内变量维度一致的全是零的内容。
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad