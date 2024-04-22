# from math import dist
import numpy as np
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class VFedPCA:
    def __init__(self, **kwargs) -> None:
        self.fed_dis = []   # [[dis_1, dis_2, dis_3, ..., dis_c], ...], c=通信轮数

    # 联邦表示学习 不断修正表示矩阵，最后的矩阵作为表示
    def fed_representation_learning(self, params, X_fed, X_clients):  # params, X_fed, X_clients
        X_fed = X_fed[:, :int(X_fed.shape[-1] / params['party_num']) * params['party_num']]  # 将data_full的列数缩减为params[‘party_num’]的倍数，防止不匹配
        global_eigs, global_eigv = self.local_power_iteration(params, X_fed, iter_num=params['iter_num'], com_time=0, warm_start=None)  # 幂迭代法求data_full的特征值和特征向量

        torch.cuda.empty_cache

        # for p_idx in range(len(args.p_list)):
        d_list = X_clients  # 纵联各客户端数据集的列表组合，[d1, d2, ...d_p], d_p=[n, fea_num]
        p_num = params['party_num']  # 纵联参与方个数
        ep_list, vp_list = [], []  # 列表分别存放每个参与方的特征值和特征向量
        print('Before:')
        for x in X_clients:
            print(x.shape)
        # 纵联开始
        if params['warm_start']:  # 是否选择带有热启动的幂迭代算法 tips:热启动一般是采用“相关或简化问题的最优解”来作为原问题的初始值
            print("Warning: you are using Local Power Iteration with Warm Start!")
        fed_u = None  # 定义初始全局特征向量
        for cp in range(params['period_num'] + 1):  # 通信次数
            # 获取每个客户端的特征值和特征向量
            for i in range(p_num):
                ep, vp = self.local_power_iteration(params, d_list[i], iter_num=params['iter_num'], com_time=cp, warm_start=fed_u)
                ep_list.append(ep)
                vp_list.append(vp)
            if cp == 0:  # 仅第一轮通信执行，初始权重设置相同
                # print("Warning: isolate period!")
                isolate_u = self.isolate(ep_list, vp_list)  # 第一轮的全局特征向量
                dis_p = self.squared_dis(global_eigv, isolate_u)  # 算表示矩阵的特征向量和联邦特征特征向量的欧氏距离
                self.fed_dis.append(dis_p)  # 记录每次通信的欧氏距离，装进列表
                continue

            # 联邦向量
            fed_u = self.federated(ep_list, vp_list, params['weight_scale'])  # weight scale method

            # 算表示矩阵的特征向量和联邦特征特征向量的欧氏距离
            dis_p = self.squared_dis(global_eigv, fed_u)
            self.fed_dis.append(dis_p)
            
            # 重建表示矩阵 伪代码无，为了满足第二目标函数dist
            rs_fed_u = np.expand_dims(fed_u, axis=-1)   # 4000 x 1 最后一个维度上插入一个新的轴，使得数组的形状发生变化，最里面的元素单独成一个轴
            # print('rs_fed_u: ', rs_fed_u.shape)
            # print('X_fed: ', X_fed.shape)
            mid_up = X_fed.T.dot(rs_fed_u)          # X_s x 1 表示矩阵向联邦向量投影 伪代码13
            up_item = mid_up.dot(mid_up.T)              # X_s x X_s
            up_item_norm = up_item / (np.linalg.norm(up_item) + 1e-9)
            X_fed = X_fed.dot(up_item_norm)     # 4000 x X_s 伪代码14
            # print('X_fed: ', X_fed.shape)

            # 重建本地数据矩阵 对应伪代码13-14
            for i in range(p_num):
                rs_fed_u = np.expand_dims(fed_u, axis=-1)   # 4000 x 1
                mid_up = d_list[i].T.dot(rs_fed_u)          # X_i x 1 表示矩阵向联邦向量投影 伪代码13
                up_item = mid_up.dot(mid_up.T)              # X_i x X_i
                up_item_norm = up_item / (np.linalg.norm(up_item) + 1e-9)
                d_list[i] = d_list[i].dot(up_item_norm)     # 4000 x X_i 伪代码14
        print('After:')
        for x in X_clients:
            print(x.shape)
        print('Global representation: ', X_fed.shape)
        return X_fed

    # 计算表示矩阵的特征向量和联邦特征向量距离（欧氏距离）
    def squared_dis(self, a, b, r=2.0):  # r=2是为了开平方
        d = sum(((a[i] - b[i]) ** r) for i in range(len(a))) ** (1.0 / r)  # 特征向量每值的差的平方和再开方
        return d
    
    # 联邦合成联邦特征向量（第二轮+），可选择是否用权重缩放方法
    def federated(self, ep_list, vp_list, weight_scale):
        v_w = ep_list / np.sum(ep_list)  # 基于特征值的每个客户端的权重
        if weight_scale:  # 权重缩放方法，伪代码Weight Scaling Method
            print("Warning: you are using weight scaling method!")
            eta = np.mean(v_w)  # 特征值先默认ω，所有ω求平均值得加速度η
            en_num = len(ep_list) // 2  # 增强客户端数量
            idx = np.argsort(-v_w)  # 降序排序的索引列表
            print("Before: ", v_w) 
            for i in idx[:en_num]:
                v_w[i] = (1 + eta) * v_w[i]  # 前半特征值大的客户端增强权重
            for j in idx[en_num:]:
                v_w[j] = (1 - eta) * v_w[j]  # 后半特征值小的客户端虚弱权重
            print("After: ", v_w)
        B = [np.dot(k, v) for k, v in zip(v_w, vp_list)]  # 每一方特征向量✖自身权重的列表
        u = np.sum(B, axis=0)  # 联邦向量u，各方加权求和，用于表示矩阵投影
        return u

    # 联邦合成联邦特征向量（第一轮），第一轮权重都相同
    def isolate(self, ep_list, vp_list):
        # 基于特征值的每个客户端的权重
        ep_avg = [1.0 for i in range(len(ep_list))]  # 第一轮的特征值不能说明什么，其实各方权重一致，假设都为1.0
        v_w = ep_avg / np.sum(ep_avg)  # 原列表的每一个值除以列表值总和，用于算加权权值
        B = [np.dot(k, v) for k, v in zip(v_w, vp_list)]  # 每一方特征向量✖自身权重的列表
        u = np.sum(B, axis=0)  # 联邦向量u，各方加权求和，用于表示矩阵投影
        return u
    
    # 幂迭代算法 返回特征值和特征向量
    def local_power_iteration(self, params, X, iter_num, com_time, warm_start):  # params["iter_num","party_num","warm_start","period_num","weight_scale"], X_fed, iter_num=params['iter_num'], com_time=0, warm_start=None
        """
            幂法基本思想是: 求一个n阶方阵A的特征值和特征向量, 先任取一个非零初始向量v(0), 进行迭代计算v(k+1)=Av(k), 直到收敛得最大特征值
            https://www.jianshu.com/p/e4585be7850d

            热启动v是指在迭代过程中使用上一次迭代的结果作为起点，而不是使用随机向量作为起点。这样可以加快迭代速度，提高算法的收敛性能。
            在幂迭代算法中，b_k是一个向量，用于计算矩阵的特征值和特征向量。
            在每次迭代中，算法计算出特征向量a_bk和特征值e_k，并将特征向量归一化后更新b_k。最后，该算法返回特征值e_k和特征向量b_k。

            矩阵阶数由维数最小次决定，比如(500,1000)是500，(500,400)是400
            特征值的个数由阶数决定
            特征向量的形状在幂迭代算法中，由原始矩阵的第一维决定
                例子中因为输入的参与方矩阵都是对齐的，也就是样本量一致，特征向量也就都是一致了
        """
        A = np.cov(X)  # A是X的协方差矩阵
        # 从随机向量或热启动 v 开始 tips:热启动一般是采用“相关或简化问题的最优解”来作为原问题的初始值
        judge_use = com_time not in [0, 1] and params['warm_start']  # 如果com_time不等于0或1并且params[‘warm_start’]为真，com_time表示第几轮迭代，一般2+次才热启动
        b_k = warm_start if judge_use else np.random.rand(A.shape[0])  # 若judge_use为真，使用热启动v；否则，judge_use为假，使用随机向量

        for _ in range(iter_num):
            # 特征向量
            a_bk = np.dot(A, b_k)  # v(k+1)=Av(k)计算特征向量
            b_k = a_bk / (np.linalg.norm(a_bk) + 1e-9)  # 归一化特征向量，见VFedPCA公式5，伪代码7前
            
            # 特征值
            e_k = np.dot(A, b_k.T).dot(b_k) / np.dot(b_k.T, b_k)  # 算特征向量对应的特征值，见VFedPCA公式6，伪代码7后
        return e_k, b_k  # 返回特征值和特征向量


if __name__ == '__main__':
    params = {
        'iter_num': 100,  # 本地幂迭代算法计算特征值和特征向量的迭代次数
        'party_num': 4,  # 除第三方的参与方个数
        'warm_start': False,  # 热启动一般是采用“相关或简化问题的最优解”来作为原问题的初始值
        'period_num': 10,  # 通信次数
        'weight_scale': False  # 模型压缩的技术，用于将模型的权重（weight）从浮点数（float）转换为定点数（int），从而减少模型的大小和内存占用
    }
    # 仿真party_num个客户端的数据集

    X_clients = [torch.normal(0, 1, size=(2000, 1000)).numpy() for _ in range(params['party_num'])]  # 一个大小为(500, 1000)的张量，其中每个元素都是从均值为0，标准差为1的正态分布中随机采样得到的。然后，它将这个张量转换为NumPy数组。
    X_fed = torch.normal(0, 1, size=(2000, 1900)).numpy()  # 表示矩阵的第二维可以自定义
    model = VFedPCA()
    print(model.fed_representation_learning(params, X_fed, X_clients))
    print(model.fed_dis)