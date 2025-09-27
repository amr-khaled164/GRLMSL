import argparse

'''
    argparse模块是命令行选项、参数和子命令解析器。可以让人轻松编写用户友好的命令行接口。适用于代码需要频繁地修改参数的情况。
    如果代码的规模很大，修改参数的过程会困难，不妨将这些需要频繁修改的参数放到代码外部，在命令行运行程序的时候一起输入，就用到了argparse模块。
    parser = argparse.ArgumentParser(description='...') -> 用来装载参数的容器
    -创建一个 ArgumentParser 对象，该对象包含将命令行输入内容解析成 Python 数据的过程所需的全部功能。description是该对象的描述信息。
    parser.add_argument('radius', type=int, help='Radius of cylinder') -> 给这个解析对象添加命令行参数
    -参数名；参数类型，声明这个参数的数据类型为int算默认数据类型为str；描述信息
    parser.add_argument('radius', type=int, help='Radius of cylinder')
    parser.add_argument('height', type=int, help='Height of cylinder')
    -以上输入必须半径在前，高度在后，如果想改变输入的顺序或在输入参数同时携带参数名，可以使用选择型参数，在添加参数时参数名前加两个"-"
    -还有一种方法,通过“-”加上参数别名的形式，注意被"–"修饰的参数名必须存在：
    =》parser.add_argument('-r', '--radius', type=int, help='Radius of cylinder')
    =》parser.add_argument('-H', '--height', type=int, help='Height of cylinder')
    parser.parse_args() -> 获取所有参数
'''


def get_options():

    parser = argparse.ArgumentParser(description='Microservice Instance Selection')

    parser.add_argument('--state_dim', type=int, default=202, help='经GAT编码后的state的特征维度')
    parser.add_argument('--state_dim_1', type=int, default=200, help='经GCN编码后的state的特征维度')
    parser.add_argument('--action_dim_list', type=list, default=[13, 13, 13, 13, 13, 13, 13, 13],
                        help='多输出的动作维度列表，动作维度为微服务实例数+1，列表长度为微服务数')
    parser.add_argument('--action_dim_2', type=int, default=295632, help='变体2（单输出）的动作维度')
    parser.add_argument('--action_dim_2_simply', type=int, default=5328, help='变体2（单输出）的动作维度')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout，丢弃比例，训练时防止过拟合')
    parser.add_argument('--lr', type=float, default=5e-4, help='学习率')  # ！！！！！！！！！！！！！！ 1e-1,1e-3,(5e-4),1e-4||
    parser.add_argument('--gamma', type=float, default=0.9, help='折扣因子')
    parser.add_argument('--tau', type=float, default=1e-3, help='目标网络软更新的学习率')
    parser.add_argument('--buffer-capacity', type=int, default=14336, help='经验回放池的容量')
    parser.add_argument('--batch-size', type=int, default=448, help='每次回放时子经验池的采样大小') # 448!!!!!!!
    parser.add_argument('--batch-size_simply', type=int, default=256, help='每次回放时子经验池的采样大小') # 448!!!!!!!
    parser.add_argument('--num-request', type=int, default=7, help='请求种类,即子经验池数')
    parser.add_argument('--num-request-simply', type=int, default=4, help='请求种类,即子经验池数')

    parser.add_argument('--sp-node-features', type=int, default=96, help='部署节点的特征维度')
    parser.add_argument('--sp-node-hid', type=int, default=192, help='部署节点特征变换后的维度，即EGAT输出的节点特征维度')
    parser.add_argument('--sp-edge-features', type=int, default=1, help='部署节点间边的特征维度')
    parser.add_argument('--sp-edge-hid', type=int, default=2, help='部署节点间边特征变换后的维度')
    parser.add_argument('--alpha', type=float, default=0.2, help='EGAT，GAT的leakyrelu的超参数')
    parser.add_argument('--sp-attention-heads', type=int, default=1, help='EGAT的注意力个数')

    parser.add_argument('--req_node_features', type=int, default=3, help='req中微服务节点的特征维度')
    parser.add_argument('--req_node_hid', type=int, default=6, help='req中微服务节点的特征变换后的维度')
    parser.add_argument('--req_attention_heads', type=int, default=1, help='GAT的注意力个数')

    parser.add_argument('--prioritized-er', type=bool, default=True, help='是否使用优先经验回放')
    parser.add_argument('--e', type=float, default=0.01, help='防止优先级为0的常量')
    parser.add_argument('--per-alpha', type=float, default=0.6, help='PER（Prioritized Experience Replay）超参数alpha，用于决定使用多少优先级，[0,1]')
    parser.add_argument('--per-beta', type=float, default=0.4, help='PER超参数beta，用于计算重要性采样权重')
    parser.add_argument('--beta-increment', type=float, default=1e-5, help='每次采样beta的增量，用于计算重要性采样权重')

    parser.add_argument('--epsilon', type=float, default=1, help='initial epsilon')
    parser.add_argument('--decay-rate', type=float, default=1 - 1e-4, help='Decay rate for epsilon')
    parser.add_argument('--epsilon-min', type=float, default=0.005, help='Minimum value for epsilon')


    return parser.parse_args()
