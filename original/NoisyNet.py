import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


# 定义带有噪声的线性层
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features  # 输入特征维度
        self.out_features = out_features  # 输出特征维度
        self.std_init = std_init  # 噪声的标准差初始化值

        # 可训练参数：权重的均值（mu）和标准差（sigma）
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))

        # 可训练参数：偏置的均值（mu）和标准差（sigma）
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))

        # 注册缓冲区：用于存储噪声（epsilon），这些值不会被优化器更新
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))

        # 初始化参数和噪声
        self.reset_parameters()
        self.reset_noise()

    # 初始化权重和偏置的均值和标准差
    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)  # 均值初始化范围
        self.weight_mu.data.uniform_(-mu_range, mu_range)  # 权重均值初始化
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))  # 权重标准差初始化
        self.bias_mu.data.uniform_(-mu_range, mu_range)  # 偏置均值初始化
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))  # 偏置标准差初始化

    # 重置噪声 reset_noise方法用于生成新的噪声，确保每次训练时噪声是独立的。
    def reset_noise(self):
        epsilon_in = self.scale_noise(self.in_features)  # 输入噪声
        epsilon_out = self.scale_noise(self.out_features)  # 输出噪声

        # 生成权重和偏置的噪声
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))  # 外积生成权重噪声
        self.bias_epsilon.copy_(epsilon_out)  # 直接使用输出噪声作为偏置噪声

    # 生成缩放后的噪声
    def scale_noise(self, size):
        x = torch.randn(size)  # 生成标准正态分布噪声
        return x.sign().mul_(x.abs().sqrt_())  # 对噪声进行缩放（符号乘以绝对值的平方根）

    def noisy_infer_when_test(self):
        mu=0
        sigma = 10000
        noisy_in = torch.FloatTensor(np.random.normal(mu, sigma, size=self.in_features))
        noisy_out = torch.FloatTensor(np.random.normal(mu, sigma, size=self.out_features))
        self.weight_epsilon.copy_(noisy_out.ger(noisy_in))
        self.bias_epsilon.copy_(noisy_out)

    # 前向传播
    def forward(self, x):
        # model.eval(),评估模式，此时self(module).training=False，默认为True
        if self.training:  # 训练时添加噪声
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:  # 测试时不添加噪声
            # self.noisy_infer_when_test()
            # weight = self.weight_mu
            # print('weight',weight)
            weight = self.weight_mu
            # weight = self.weight_mu+self.weight_epsilon
            # print('weight(noisy)',weight)
            # bias = self.bias_mu
            # print('bias',bias)
            # bias = self.bias_mu+self.bias_epsilon
            bias = self.bias_mu
            # print('bias(noisy)',bias)

        return F.linear(x, weight, bias)  # 返回线性变换结果


# # # 定义Noisy DQN网络
# class NoisyDQN(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(NoisyDQN, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 128)  # 普通全连接层
#         self.noisy_fc2 = NoisyLinear(128, 128)  # 带噪声的全连接层
#         self.noisy_fc3 = NoisyLinear(128, output_dim)  # 带噪声的输出层
#
#     # 前向传播
#     def forward(self, x):
#         x = F.relu(self.fc1(x))  # 第一层全连接 + ReLU激活
#         x = F.relu(self.noisy_fc2(x))  # 第二层带噪声的全连接 + ReLU激活
#         return self.noisy_fc3(x)  # 输出层带噪声的全连接
#
#     # 重置噪声
#     def reset_noise(self):
#         self.noisy_fc2.reset_noise()  # 重置第二层的噪声
#         self.noisy_fc3.reset_noise()  # 重置输出层的噪声
#

# # 示例用法
# input_dim = 4  # 输入维度（例如状态空间维度）
# output_dim = 2  # 输出维度（例如动作空间维度）
# model = NoisyDQN(input_dim, output_dim)  # 初始化Noisy DQN模型
# model.eval()
# optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器
#
# # 生成虚拟数据
# state = torch.FloatTensor(np.random.rand(1, input_dim))  # 当前状态
# action = torch.LongTensor([0])  # 选择的动作
# reward = torch.FloatTensor([1.0])  # 奖励
# next_state = torch.FloatTensor(np.random.rand(1, input_dim))  # 下一个状态
# done = torch.FloatTensor([0])  # 是否结束（0表示未结束）
#
# # 前向传播：计算当前状态的Q值
# q_values = model(state)
# # 前向传播：计算下一个状态的Q值
# next_q_values = model(next_state)
#
# # 计算目标Q值（Bellman方程）
# target = reward + (1 - done) * 0.99 * next_q_values.max(1)[0]
# # 计算损失（均方误差损失）
# loss = F.mse_loss(q_values.gather(1, action.unsqueeze(1)), target.unsqueeze(1))
#
# # 反向传播和优化
# optimizer.zero_grad()  # 清空梯度
# loss.backward()  # 计算梯度
# optimizer.step()  # 更新参数
#
# # 重置噪声
# model.reset_noise()