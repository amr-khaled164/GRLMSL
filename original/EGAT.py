import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from SystemPlatform import SP


class EGATLayer(nn.Module):
    # node_in_dim, node_hid_dim, edge_in_dim, edge_hid_dim, dropout, alpha,
    def __init__(self, args, concat=True):
        super(EGATLayer, self).__init__()
        # self.args = args  # 注：参数集合
        self.dropout = args.dropout  # 注：dropout（丢弃），防止模型过拟合
        self.node_in_dim = args.sp_node_features  # 注：节点输入特征数
        self.node_hid_dim = args.sp_node_hid  # 注：节点变化后的特征数
        self.edge_in_dim = args.sp_edge_features  # 注：边输入特征数
        self.edge_hid_dim = args.sp_edge_hid  # 注：边变化后的特征数
        self.alpha = args.alpha  # 注：LeakyReLU超参数
        self.concat = concat  # 注：多头注意力的输出是否执行拼接操作

        # Linear transformations for node features
        self.W = nn.Parameter(torch.empty(size=(self.node_in_dim, self.node_hid_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # Linear transformations for edge features
        self.U = nn.Parameter(torch.empty(size=(self.edge_in_dim, self.edge_hid_dim)))
        nn.init.xavier_uniform_(self.U.data, gain=1.414)

        # Attention mechanism parameters
        self.a = nn.Parameter(torch.empty(size=(2 * self.node_hid_dim + self.edge_hid_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, n, e, adj):
        # n: Node features (【batch_size,】 N, node_in_features)
        # e: Edge features (【batch_size,】 N, N)
        # adj: Adjacency matrix (【batch_size,】 N, N)

        # e: Edge features (【batch_size,】 N, N, edge_in_features=1)
        e = e.unsqueeze(-1)

        Wn = torch.matmul(n, self.W)  # Wn: (【batch_size,】 N, node_hid_features)

        Ue = torch.einsum('...jk,...kl', e, self.U)  # (【batch_size,】,N, N, edge_hid_features)

        # a1 (【batch_size,】,N,1)
        a1 = torch.matmul(Wn, self.a[:self.node_hid_dim, :])
        # a1 (【batch_size,】,N,1)
        a2 = torch.matmul(Wn, self.a[self.node_hid_dim:self.node_hid_dim*2, :])
        # a2_T (【batch_size,】,1,N)
        a2_T = torch.einsum('...jk->...kj', a2)
        # a3 (【batch_size,】,N, N)
        a3 = torch.einsum('...jk,...kl', Ue.unsqueeze(-2), self.a[self.node_hid_dim*2:, :]).squeeze(-1).squeeze(-1)
        # broadcast add -> attention (【batch_size,】,N, N)
        # t1 = a1+a2_T
        # t2 = t1+a3
        attention = self.leakyrelu(a1+a2_T+a3)

        # Masked attention
        zero_vec = -9e15 * torch.ones_like(attention)
        # attention (【batch_size,】,N, N) masked
        attention_masked = torch.where(adj > 0, attention, zero_vec)
        attention_normalized = F.softmax(attention_masked, dim=-1)
        # attention = F.dropout(attention, self.dropout, training=self.training)

        n_prime = []
        # Aggregate node features with edge_features
        for i in range(n.size(-2)):
            if len(Ue.shape) == 4:
                # Wn: (batch_size, N, node_hid_features), Ue[:,i]: (batch_size, N, edge_hid_features)
                # WnUe (batch_size, N, node_hid_features+edge_hid_features)
                WnUe = torch.concat([Wn, Ue[:,i]], dim=-1)
                # attention[:, i].unsqueeze(1) (batch_size, 1, N)
                # torch.matmul() -> (batch_size , 1 ,node_hid_features+edge_hid_features)
                n_prime.append(torch.matmul(attention_normalized[:, i].unsqueeze(1), WnUe).squeeze(1))
            else:
                # Wn: (N, node_hid_features), Ue[i]: (N, edge_hid_features)
                # WnUe: (N,node_hid_features+edge_hid_features)
                WnUe = torch.concat([Wn, Ue[i]], dim=-1)
                # attention[i]: (N), torch.matmul()->(node_hid_features+edge_hid_features)
                n_prime.append(torch.matmul(attention_normalized[i], WnUe))

        # n_prime N[node_hid_features+edge_hid_features] or N[batch_size ,node_hid_features+edge_hid_features]
        # -> (【batch_size,】 N, node_out_features+edge_out_features)
        n_prime = torch.stack(n_prime, dim=-2)

        if self.concat:
            return F.elu(n_prime)
        else:
            return n_prime


class EGAT(nn.Module):

    # node_features, node_hid, edge_features, edge_hid, dropout, alpha, attention_heads):
    def __init__(self, args):
        super(EGAT, self).__init__()
        # self.args = args
        self.dropout = args.dropout
        self.attention_heads = args.sp_attention_heads  # 注：注意力个数

        # Multiple attention heads
        self.attentions = [EGATLayer(args, concat=True) for _ in range(self.attention_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, n, e, adj):
        # att()->(【batch_size, 】 N, node_out_features+edge_out_features)
        n = torch.cat([att(n, e, adj) for att in self.attentions], dim=-1)  # Concatenate heads
        # (【batch_size, 】 node_out_features+edge_out_features)
        n = n.mean(dim=-2)  # 注：对节点特征取均值，生成图级表示
        return n
