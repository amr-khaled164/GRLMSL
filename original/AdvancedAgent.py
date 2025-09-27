import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from original.EGAT import EGAT
from original.GAT import GAT
from original.NoisyNet import NoisyLinear
from original.Memory import Memory
from original.PCGrad import PCGrad

# Dueling Noisy Network Architecture with Multi-advantage-streams and GAT
class AdvancedDQN(nn.Module):

    def __init__(self, args):  # state_dim, action_dim_list, dropout=0.2):
        super().__init__()

        self.state_dim = args.state_dim  # 202(GAT)
        self.action_dim_list = args.action_dim_list  # [13,13,13,13,13,13,13,13],值为各微服务的实例数+1，也可能不同
        self.action_group = len(self.action_dim_list)  # 8，值为微服务数，也是输出层的数
        self.action_dim_max = max(self.action_dim_list)  # 13,微服务实例数的最大值+1
        self.dropout = args.dropout

        # GAT
        self.sp_encoder = EGAT(args)
        self.req_encoder = GAT(args)

        # Dueling Value Stream with Noisy Linear
        self.value_stream = nn.Sequential(
            NoisyLinear(self.state_dim, 100),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            NoisyLinear(100, 50),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            NoisyLinear(50, 1)
        )

        # Dueling Multi Advantage stream
        self.advantage_stream = nn.ModuleList()
        for dim in self.action_dim_list:
            self.advantage_stream.append(
                nn.Sequential(
                    NoisyLinear(self.state_dim, 100),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    NoisyLinear(100, 50),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    NoisyLinear(50, dim),
                )
            )

    def forward(self, state, sp_node, sp_adj, sp_edge, req_node, req_adj, mask=None):

        """
            state 包含 req_deadline, user.node.n_id的张量
            mask [[1,0,0,...,0],[0,1,1,...,1],[],[],[],[],[],[]]
        """

        # 处理图数据，转化为特征向量
        sp_feature_vector = self.sp_encoder(sp_node, sp_edge, sp_adj)
        # 处理next_req不相同的问题
        if isinstance(req_node, tuple):
            req_feature_vector = torch.stack([self.req_encoder(node, adj)
                                              for node, adj in zip(req_node, req_adj)], dim=0)
        else:
            req_feature_vector = self.req_encoder(req_node, req_adj)

        # state tensor.Size(【batch_size,】 state_dim)
        state = torch.cat([sp_feature_vector, req_feature_vector, state], dim=-1)

        # 注入噪声，形成环境信息的干扰,已训练好的模型进行鲁棒性实验时使用
        # print('state\n', state)
        # state = self.noisy_interfere(state,mode=0,noisy_degree=0.05) # uniform or gauss noisy mode0
        # state = self.noisy_interfere(state,mode=1,noisy_degree=5,noisy_percentage=0.1)
        # print(' noisy state\n', state)

        # value tensor.Size(【batch_size,】, action_group, 1)

        value = torch.einsum('...jk->...kj', self.value_stream(state).
                             repeat_interleave(self.action_group, dim=-1).unsqueeze(-2))

        # out_layer(advantage_hidden) tensor.Size(【batch_size,】 action_dim)
        # multi_advantage tensor.Size(【batch_size,】,action_group, action_dim)
        multi_advantage = torch.stack([out_layer(state) for out_layer in self.advantage_stream],
                                      dim=-2)
        # multi_q_values tensor.Size(【batch_size,】,action_group, action_dim)
        multi_q_values = value + (multi_advantage - multi_advantage.mean(-1, keepdim=True))

        # mask invalid action
        if mask is not None:
            # print('mask', mask)
            mask[mask==1.0]=float('-inf')
            # print('mask(transfer)',mask)
            # print('multi_q_values', multi_q_values)
            multi_q_values +=mask
            # print('multi_q_values(mask)', multi_q_values)
        return multi_q_values

    # 重置噪声
    def reset_noise(self):

        for layer in self.value_stream:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()
        for group in self.advantage_stream:
            for layer in group:
                if isinstance(layer, NoisyLinear):
                    layer.reset_noise()

    # def noisy_interfere(self,state,noisy_degree=0.05,noisy_percentage=1.0,mode=2):
    #
    #     # print('noisy percentage：', noisy_percentage)
    #     # print('noisy degree:', noisy_degree)
    #
    #     if mode == 0:
    #         # 固定噪声占比（100%），改变噪声程度
    #         for i in range(len(state)):
    #             # noisy = np.random.uniform(low=-noisy_degree, high=noisy_degree) 均匀噪声
    #             noisy = random.gauss(0,noisy_degree) # 高斯噪声
    #             state[i] += noisy
    #
    #     elif mode == 1:
    #         # 固定噪声程度，改变噪声占比
    #         num = int(len(state) * noisy_percentage)
    #         # print('noisy num：', num)
    #         noisy_indexes = random.sample([i for i in range(len(state))], num)
    #         # print('noisy indexes', noisy_indexes)
    #         for i in noisy_indexes:
    #             # noisy = np.random.uniform(low=-noisy_degree, high=noisy_degree) 均匀噪声
    #             noisy = random.gauss(0, noisy_degree)  # 高斯噪声
    #             state[i] += noisy
    #
    #     return state


class AdvancedAgent:

    # state_dim, action_dim_list, lr=1e-3, gamma=0.9, tau=1e-3, buffer_capacity=14336, batch_size=64, num_request=7):
    def __init__(self, args):

        self.state_dim = args.state_dim  # 202
        self.action_dim_list = args.action_dim_list  # [13,13,13,13,13,13,13,13],值为各微服务的实例数，也可能不同
        self.action_group = len(self.action_dim_list)  # 8，值为微服务数，也是输出层的数
        self.action_dim_max = max(self.action_dim_list)  # 13,微服务实例数的最大值
        self.lr = args.lr
        self.gamma = args.gamma
        self.tau = args.tau
        self.buffer_capacity = args.buffer_capacity
        self.batch_size = args.batch_size  # 每次采样的子经验池的采样量，总采样量为batch_size*num_request
        self.num_request = args.num_request  # 请求种类，即子经验池数

        # DQN
        self.q_net = AdvancedDQN(args)
        self.target_net = AdvancedDQN(args)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        # Optimizer with 梯度手术
        self.optimizer = PCGrad(optim.Adam(self.q_net.parameters(), lr=self.lr))

        # Replay buffer
        self.buffer = Memory(args)

    def select_action(self, state, sp_node, sp_adj, sp_edge, req_node, req_adj, mask):

        with torch.no_grad():
            # q_values torch.Size(action_group, action_dim_max)
            q_values = self.q_net(state, sp_node, sp_adj, sp_edge, req_node, req_adj, mask)
            # actions torch.Size(action_group,)
            actions = q_values.argmax(dim=-1, keepdim=False)

        return actions

    def train(self):

        """
            state & next_state [req_deadline, user]
            [sp_node_feature_matrix, sp_edge_feature_matrix, sp_adj, req_node_feature_matrix, req_adj, remain]
            action = [2,4,0,0,1,0,0,0] 0表示不涉及该微服务，即不进行选择，大于0表示选择实例，action[i]-1 = ins.id(实例id）
            reward = tenser([2.0])
            next_req_mask = [[1,0,0,...,0],[0,1,1,...,1],[],[],[],[],[],[]]
            req_id = 1
        """
        loss_list = []
        for i in range(self.num_request):

            transitions, indexes, weights = self.buffer.pattern_memories[i].sample_experience()
            (sp_nodes, sp_edges, sp_adjs, req_nodes, req_adjs, states, actions, rewards, next_sp_nodes,
             next_req_nodes, next_req_adjs, next_states, next_req_masks, req_ids) = zip(*transitions)

            # sp_nodes torch.Size(batch_size,N=16,F=96)
            sp_nodes = torch.stack(sp_nodes, dim=0)
            # sp_edges torch.Size(batch_size,N=16,N=16)
            sp_edges = torch.stack(sp_edges, dim=0)
            # sp_adjs torch.Size(batch_size,N=16,N=16)
            sp_adjs = torch.stack(sp_adjs, dim=0)
            # req_nodes torch.Size(batch_size,N=1-8,F=3)
            req_nodes = torch.stack(req_nodes, dim=0)
            # req_adjs torch.Size(batch_size,N=1-8,N)
            req_adjs = torch.stack(req_adjs, dim=0)
            # states torch.Size(batch_size,2)
            states = torch.stack(states, dim=0)
            # actions tensor.Size(batch_size,action_group,1)
            actions = torch.stack(actions, dim=0).unsqueeze(-1)
            # rewards tensor.Size(batch_size,1)
            rewards = torch.stack(rewards, dim=0)
            next_sp_nodes = torch.stack(next_sp_nodes, dim=0)
            next_states = torch.stack(next_states, dim=0)
            # next_req_masks tensor.Size(batch_size,action_group,action_dim_max)
            next_req_masks = torch.stack(next_req_masks, dim=0)

            # self.q_net() -> torch.Size(batch_size, action_group, action_dim_max)
            # current_q_values torch.Size(batch_size, action_group,1)
            current_q_values = (self.q_net(states, sp_nodes, sp_adjs, sp_edges, req_nodes, req_adjs)
                                .gather(-1, actions))
            # next_actions torch.Size(batch_size,action_group,1)
            next_actions = self.q_net(next_states, next_sp_nodes, sp_adjs, sp_edges, next_req_nodes,
                                      next_req_adjs, next_req_masks).argmax(-1, keepdim=True)
            # next_q_values torch.Size(batch_size, action_group, 1)
            next_q_values = self.target_net(next_states, next_sp_nodes, sp_adjs, sp_edges, next_req_nodes
                                            , next_req_adjs).gather(-1, next_actions)

            # torch.Size(batch_size, action_group, 1),两个q值的差
            t1 = next_q_values-current_q_values
            # torch.Size(batch_size, 1) ，按action_group取两个q值的差的均值
            t2 = t1.mean(dim=-2)
            # td_errors torch.Size(batch_size,)
            td_errors = (rewards + t2).squeeze(-1)
            self.buffer.update(indexes, td_errors.detach().numpy(), req_ids[0])
            # torch.tensor(weights) tensor.Size(batch_size,)
            losses = torch.tensor(weights)*(td_errors**2)
            average_loss = losses.mean()
            loss_list.append(average_loss)

        # 结合梯度手术，进行梯度更新操作
        self.optimizer.zero_grad()
        self.optimizer.pc_backward(loss_list)
        self.optimizer.step()

        # Soft update target network
        for target_param, policy_param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1 - self.tau) * target_param.data)

        self.q_net.reset_noise()

        with torch.no_grad():
            train_average_loss = np.mean(np.array(loss_list))

        print('train_average_loss', train_average_loss)

        return train_average_loss

    def test_setting(self):

        self.q_net.eval()

    def save(self, break_point):

        # 中断后继续训练需要保存，两个Q网络的参数，优化器的参数
        torch.save({
            'q_net': self.q_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.optimizer.state_dict()
        }, './save/DQN_trained_model_{}.pt'.format(break_point))
        self.buffer.save(break_point)

    def load(self, break_point):

        checkpoint = torch.load('./save/DQN_trained_model_{}.pt'.format(break_point))
        self.q_net.load_state_dict(checkpoint['q_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.optimizer.load_state_dict(checkpoint['optimizer'])
        self.buffer.load(break_point)

    def q_num(self):
        print('total parameters of q_net:{}'.format(sum(p.numel() for p in self.q_net.parameters())))
