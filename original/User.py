from original.Node import *


class User:

    def __init__(self, u_id, source, access):

        self.u_id = u_id  # 注：用户唯一id（对内）
        self.username = 'User{}'.format(u_id)  # 注：用户唯一名称（对外）
        self.source = source  # 注：用户所在的部署节点
        self.access_time = access  # 注：用户请求接入部署节点的时间

    def print_information(self):

        print('User', self.username)
        print('It is around the node {}, the access time is {}ms'.format(self.source.n_name, self.access_time))


if __name__ == "__main__":

    Node0 = Node(0)
    Node0.print_information()
    user0 = User(0, Node0, 9)
    user0.print_information()
