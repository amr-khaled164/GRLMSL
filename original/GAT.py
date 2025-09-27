import torch
import torch.nn as nn
import torch.nn.functional as f
# from SystemPlatform import SP


class GATLayer(nn.Module):
    # in_features, out_features, dropout, alpha,
    def __init__(self, args, concat=True):
        super(GATLayer, self).__init__()

        self.dropout = args.dropout
        self.in_features = args.req_node_features
        self.out_features = args.req_node_hid
        self.alpha = args.alpha
        self.concat = concat

        '''
            torch.empty() 创建一个指定大小但未初始化数据的张量
            nn.init.xavier_uniform 初始化数据
            nn.Linear自动处理了权重和偏置项的初始化和更新，其本身就是使用的nn.Parameter()
            nn.Parameter()用来自定义权重和偏置
        '''

        self.W = nn.Parameter(torch.empty(size=(self.in_features, self.out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*self.out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leaky_relu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):

        """
            h：节点特征矩阵
            adj：节点邻接矩阵
            torch.mm 矩阵乘法
        """
        # h.shape: (【batch_size,】 N, in_features), Wh.shape: (【batch_size, 】 N, out_features)
        wh = torch.matmul(h, self.W)
        # e.shape (【batch_size, 】 N, N) , 是一个 NxN 的未归一化的节点间注意力分数表
        e = self._prepare_attentional_mechanism_input(wh)

        # 与e同shape，即(【batch_size, 】 N, N)的负无穷矩阵，元素均为-9e15
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)  # 将没有连接的边设为负无穷，mask attention
        # adj,e,zer0_vec均是N*N，if adj的元素值>0 ,则e中对应位置保留未归一化的节点间注意力分数；否则替换为负无穷（这样softmax时这个值约等于0，会不考虑）
        # attention (【batch_size, 】 N, N), 归一化的权重
        attention = f.softmax(attention, dim=-1)
        # attention = f.dropout(attention, self.dropout, training=self.training)
        # h_prime: (【batch_size, 】 N, out_features)
        h_prime = torch.matmul(attention, wh)

        # 多头注意力机制的输出是否concat，决定非线性激活函数的使用位置
        if self.concat:
            return f.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, wh):
        # Wh.shape (【batch_size, 】 N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (【batch_size, 】 N, 1)
        # e.shape (【batch_size, 】 N, N)
        wh1 = torch.matmul(wh, self.a[:self.out_features, :])
        wh2 = torch.matmul(wh, self.a[self.out_features:, :])
        # broadcast add
        e = wh1 + torch.einsum('...jk->...kj', wh2)
        return self.leaky_relu(e)


class GAT(nn.Module):
    # node_features, node_hid, dropout, alpha, attention_heads
    def __init__(self, args):
        super(GAT, self).__init__()

        self.dropout = args.dropout
        self.attention_heads = args.req_attention_heads

        self.attentions = [GATLayer(args, concat=True) for _ in range(self.attention_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, x, adj):

        # att(x, adj)->(【batch_size, 】 N, out_features)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        # (【batch_size, 】 out_features)
        x = x.mean(dim=-2)
        return x

