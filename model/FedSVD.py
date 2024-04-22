# -*- coding: utf-8 -*-
import numpy as np


# 奇异值分解 返回v s u
def svd(X: np.ndarray):
    m, n = X.shape
    if m >= n:  # 如果矩阵X的行数大于或等于列数，则直接使用np.linalg.svd()函数进行SVD分解。
        return np.linalg.svd(X)
    else:  # 否则，它将先对X的转置进行SVD分解，然后返回V的转置、S和U的转置
        u, s, v = np.linalg.svd(X.T)
        return v.T, s, u.T


class FedSVD:
    def __init__(self, num_features, num_participants=2, random_seed=100):
        self.num_participants = num_participants  # 参与方数量
        self.num_features = num_features  # 特征数量
        self.seed = np.random.RandomState(random_seed)  # 随机种子

    def load_data(self, X: np.ndarray):
        self.X = X
        # X分成num_participants份，每一份的特征没有交叉
        self.Xs = [self.X[:, e * self.num_features: e * self.num_features + self.num_features] for e in range(self.num_participants)]

    def learning(self):
        ground_truth = np.concatenate(self.Xs, axis=1)  # 一维展开
        m, n = ground_truth.shape  # ground_truth的行，列

        P = self.efficient_orthogonal(n=self.X.shape[0])
        Q = self.efficient_orthogonal(n=np.sum([e.shape[1] for e in self.Xs]))
        Qs = [Q[e * self.num_features: e * self.num_features + self.num_features] for e in range(self.num_participants)]

        X_mask_partitions = []
        for i in range(self.num_participants):
            X_mask_partitions.append(P @ self.Xs[i] @ Qs[i])
        X_mask = self.secure_aggregation(X_mask_partitions)

        U_mask, sigma, VT_mask = svd(X_mask)

        U_mask = U_mask[:, :min(m, n)]
        VT_mask = VT_mask[:min(m, n), :]

        U = P.T @ U_mask

        VTs = []
        k = 1
        transferred_variables = []
        for i in range(self.num_participants):
            Q_i = Qs[i].T
            R1_i = self.seed.random([n, k])
            R2_i = self.seed.random([Q_i.shape[1] + k, Q_i.shape[1] + k])
            Q_i_mask = np.concatenate([Q_i, R1_i], axis=-1) @ R2_i
            VT_i_mask = VT_mask @ Q_i_mask
            VTs.append((VT_i_mask @ np.linalg.inv(R2_i))[:, :Q_i.shape[1]])
            transferred_variables.append([Q_i_mask, VT_i_mask])

        U = np.array(U)
        VTs = np.concatenate(VTs, axis=1)
        # self.Xs_fed = U[:, :min(m, n)] @ np.diag(sigma) @ VTs[:min(m, n), :]
        self.Xs_fed = np.matmul(P.transpose(), U)

    def secure_aggregation(self, Xs):
        n = len(Xs)
        size = Xs[0].shape
        perturbations = []
        for i in range(n):
            row = []
            for j in range(n):
                row.append(self.random(size))
            perturbations.append(row)
        perturbations = np.array(perturbations)
        perturbations -= np.transpose(perturbations, [1, 0, 2, 3])
        ys = [Xs[i] + np.sum(perturbations[i], axis=0) for i in range(n)]
        return np.sum(ys, axis=0)

    def random(self, size):
        return np.random.randint(low=-10 ** 5, high=10 ** 5, size=size) + np.random.random(size)

    """
        基于块构建的正交矩阵和基于随机矩阵的QR分解得到的正交矩阵有以下区别：
        构建方法不同：基于块的正交矩阵是通过将整个矩阵分割为块，并在每个块上生成独立的正交矩阵，最终拼接而成。而基于随机矩阵的QR分解是通过将随机生成的矩阵进行QR分解得到正交矩阵。
        生成过程不同：基于块构建的正交矩阵使用了递归的方式，从整体到局部逐步构建每个块的正交矩阵。而基于随机矩阵的QR分解是通过对随机矩阵进行正交变换得到正交矩阵。
        独立性不同：基于块构建的正交矩阵中，每个子矩阵是独立生成的正交矩阵，它们之间没有直接的关联。而基于随机矩阵的QR分解得到的正交矩阵是通过对整个随机矩阵进行变换得到的，各个列之间存在一定的关联性。
        计算复杂度不同：基于块的正交矩阵构建方法通常比基于随机矩阵的QR分解更高效，特别是对于大型矩阵而言。基于块的方法可以减少计算量，因为每个子矩阵的大小较小。而QR分解需要对整个矩阵进行正交变换，计算复杂度较高。
        总的来说，基于块构建的正交矩阵在生成过程中更加灵活和可控，可以生成具有特定结构的正交矩阵。而基于随机矩阵的QR分解得到的正交矩阵具有较高的随机性，适用于一些随机化算法和数值计算中的正交性要求。选择哪种方法取决于具体的应用场景和需求。
    """
    # 实现基于块的大小为nxn的正交矩阵。
    def efficient_orthogonal(self, n, block_size=None):
        if block_size != None:  # 如果指定了block_size，则使用基于块的方法
            qs = [block_size] * int(n / block_size)  # n/block_size是块的个数，每个块的大小保存在列表qs中
            if n % block_size != 0:  # 如果n不能被block_size整除
                qs[-1] += (n - np.sum(qs))  # 将剩余的部分添加到最后一个块的大小，以确保块的大小总和等于n
            q = np.zeros([n, n])  # nxn的全0矩阵q
            for i in range(len(qs)):  # 遍历每个块
                sub_n = qs[i]  # 得到每个块的块长度
                sub_matrix = self.efficient_orthogonal(sub_n, block_size=sub_n)  # 得到正交矩阵Q
                idx = int(np.sum(qs[:i]))  # 截至目前块的所有快的总和
                q[idx:idx + sub_n, idx:idx + sub_n] += sub_matrix
        else:  # 否则，使用带有随机矩阵的QR分解
            # 使用随机生成的nxn矩阵调用NumPy的QR分解函数np.linalg.qr，并选择full模式，得到一个正交矩阵Q和上三角矩阵R。
            q, _ = np.linalg.qr(np.random.randn(n, n), mode='full')
        return q

    def get_fed_representation(self):
        return np.array(self.Xs_fed)