import os
from random import sample
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils
from torch.autograd import Variable
from CrossKTFRA.utils import *  # 引用IndexedDataset


class GAN(object):
    def __init__(self,
                 X_task,
                 X_fed,
                 y_task,
                 task_extractor,
                 task_classifier,
                 dir,
                 device,
                 **kwargs) -> None:
        self.dir = dir  # 保存路径
        self.device = device

        input_dim = X_task.shape[1]
        output_dim = X_fed.shape[1]
        self.X_task = X_task
        self.X_fed = X_fed
        self.y_task = y_task

        d_depth = kwargs["d_depth"]
        g_depth = kwargs["g_depth"]
        negative_slope = kwargs["negative_slope"]

        self.E = task_extractor  # task方要训练的特征提取器
        self.C = task_classifier  # task方要训练的分类器

        self.gamma_adversarial_shared = torch.nn.Parameter(torch.Tensor([kwargs["gamma_1"]]).to(self.device), requires_grad=True)  # 对抗损失
        self.gamma_recons_shared = torch.nn.Parameter(torch.Tensor([kwargs["gamma_2"]]).to(self.device), requires_grad=True)  # 重构损失
        self.gamma_distill_shared = torch.nn.Parameter(torch.Tensor([kwargs["gamma_3"]]).to(self.device), requires_grad=True)  # 蒸馏损失
        self.gamma_classify_shared = torch.nn.Parameter(torch.Tensor([kwargs["gamma_4"]]).to(self.device), requires_grad=True)  # 分类损失
        self.weight_optimizer_shared = optim.Adam([self.gamma_adversarial_shared, self.gamma_recons_shared, self.gamma_distill_shared, self.gamma_classify_shared], lr=0.001)
        self.normalize_weights_shared()

        self.gamma_adversarial_private = torch.nn.Parameter(torch.Tensor([kwargs["gamma_1"]]).to(self.device), requires_grad=True)  # 对抗损失
        self.gamma_recons_private = torch.nn.Parameter(torch.Tensor([kwargs["gamma_2"]]).to(self.device), requires_grad=True)  # 重构损失
        self.weight_optimizer_private = optim.Adam([self.gamma_adversarial_private, self.gamma_recons_private], lr=0.001)
        self.normalize_weights_private()

        self.z_dim = self.X_task.shape[1]  # 噪声维度

        self.g_shared_layer_indices = [i for i in range(0, (g_depth-1) * 2, 2)][1:]
        self.d_shared_layer_indices = [i for i in range(0, (d_depth-1) * 2, 2)][1:]

        # 生成器架构 生成的x中对应X_task的部分和X_task对抗 GAN 因为标签没有限制
        self.g_local_input = input_dim  # 生成器的输入维度为task方数据特征数
        g_output = output_dim  # 生成器的输出维度为总特征维度，即X_fed的特征数
        increment = (g_output - self.g_local_input) // (g_depth - 1)  # 计算每一层神经元的增量，以确保在深度方向上平均分布特征维度
        g_layers = [self.g_local_input + increment * i for i in range(g_depth - 1)] + [g_output]  # 定义生成器每一层的维度，从输入层到输出层逐渐增加
        self.G_local = nn.Sequential()  # 创建一个空的神经网络序列，用于存储生成器的层
        for i in range(g_depth - 1):  # 对生成器的每一层进行循环操作
            self.G_local.add_module('G-linear-' + str(i),
                                    nn.Linear(g_layers[i], g_layers[i + 1]))  # 添加线性层，将上一层的维度映射到当前层的维度
            self.G_local.add_module('G-activation-' + str(i), nn.LeakyReLU(negative_slope))  # 添加LeakyReLU激活函数，引入非线性
            if i == g_depth - 2:  # 如果是最后一层
                self.G_local.add_module('G-last', nn.Tanh())  # 添加Tanh激活函数，将生成的输出限制在[-1, 1]范围内

        # 生成器架构 生成的x和x_fed对抗 CGAN
        self.g_fed_input = input_dim + self.y_task.shape[1]  # 生成器的输入维度为task方数据特征数和标签维数之和
        # g_output = output_dim  # 生成器的输出维度为总特征维度，即X_fed的特征数
        # increment = (g_output - self.g_fed_input) // (g_depth - 1)  # 计算每一层神经元的增量，以确保在深度方向上平均分布特征维度
        # g_layers = [self.g_fed_input + increment * i for i in range(g_depth - 1)] + [g_output]  # 定义生成器每一层的维度，从输入层到输出层逐渐增加
        g_layers[0] = self.g_fed_input
        self.G_fed = nn.Sequential()  # 创建一个空的神经网络序列，用于存储生成器的层
        for i in range(g_depth - 1):  # 对生成器的每一层进行循环操作
            self.G_fed.add_module('G-linear-' + str(i), nn.Linear(g_layers[i], g_layers[i + 1]))  # 添加线性层，将上一层的维度映射到当前层的维度
            self.G_fed.add_module('G-activation-' + str(i), nn.LeakyReLU(negative_slope))  # 添加LeakyReLU激活函数，引入非线性
            if i == g_depth - 2:  # 如果是最后一层
                self.G_fed.add_module('G-last', nn.Tanh())  # 添加Tanh激活函数，将生成的输出限制在[-1, 1]范围内

        # 判别器架构 生成的x取对应本体部分和x对抗 GAN
        self.d_local_input = output_dim  # 判别器的输入维度为总特征维度和标签维数之和
        d_output = 1  # 判别器的输出维度为1，因为判别器的任务是输出一个二进制值，表示输入是否为真实数据
        decrement = (self.d_local_input - d_output) // (d_depth - 1)  # 计算每一层神经元的减量，以确保在深度方向上逐渐减小特征维度
        d_layers = [self.d_local_input - decrement * i for i in range(d_depth - 1)] + [d_output]  # 定义判别器每一层的维度，从输入层到输出层逐渐减小
        self.D_local = nn.Sequential()  # 创建一个空的神经网络序列，用于存储判别器的层
        for i in range(d_depth - 1):  # 对判别器的每一层进行循环操作
            self.D_local.add_module('D-linear-' + str(i), nn.Linear(d_layers[i], d_layers[i + 1]))
            self.D_local.add_module('D-activation-' + str(i), nn.LeakyReLU(negative_slope))
            if i == d_depth - 2:
                self.D_local.add_module('D-last', nn.Sigmoid())

        # 判别器架构 生成的x和x_fed对抗 CGAN
        d_input = output_dim + self.y_task.shape[1]  # 判别器的输入维度为总特征维度和标签维数之和
        # d_output = 1  # 判别器的输出维度为1，因为判别器的任务是输出一个二进制值，表示输入是否为真实数据
        # decrement = (d_input - d_output) // (d_depth - 1)  # 计算每一层神经元的减量，以确保在深度方向上逐渐减小特征维度
        # d_layers = [d_input - decrement * i for i in range(d_depth - 1)] + [d_output]  # 定义判别器每一层的维度，从输入层到输出层逐渐减小
        d_layers[0] = d_input
        self.D_fed = nn.Sequential()  # 创建一个空的神经网络序列，用于存储判别器的层
        for i in range(d_depth - 1):  # 对判别器的每一层进行循环操作
            self.D_fed.add_module('D-linear-' + str(i), nn.Linear(d_layers[i], d_layers[i + 1]))
            self.D_fed.add_module('D-activation-' + str(i), nn.LeakyReLU(negative_slope))
            if i == d_depth - 2:
                self.D_fed.add_module('D-last', nn.Sigmoid())

        self.ablation_part = kwargs["ablation_part"]  # 消融的是共享数据还是私有数据
        self.ablation_loss = kwargs["ablation_loss"]  # 消融损失组件

    def normalize_weights_shared(self):
        total_weights = self.gamma_adversarial_shared + self.gamma_recons_shared + self.gamma_distill_shared + self.gamma_classify_shared
        self.gamma_adversarial_shared.data /= total_weights
        self.gamma_recons_shared.data /= total_weights
        self.gamma_distill_shared.data /= total_weights
        self.gamma_classify_shared.data /= total_weights

    def normalize_weights_private(self):
        total_weights = self.gamma_adversarial_private + self.gamma_recons_private
        self.gamma_adversarial_private.data /= total_weights
        self.gamma_recons_private.data /= total_weights

    def train(self, training_params, device):
        X_task = torch.from_numpy(self.X_task).to(device).float()
        # X_task = torch.from_numpy(self.X_task).to(device).to(torch.float32)
        X_fed = torch.from_numpy(self.X_fed).to(device).float()
        y_task = torch.from_numpy(self.y_task).to(device).float()

        self.device = device  # 用GPU显卡
        self.D_fed = self.D_fed.to(self.device)  # 判别器fed用GPU
        self.D_local = self.D_local.to(self.device)  # 判别器local用GPU
        self.G_fed = self.G_fed.to(self.device)  # 生成器fed用GPU
        self.G_local = self.G_local.to(self.device)  # 生成器local用GPU
        self.E = self.E.to(self.device)  # 特征提取器用GPU
        self.C = self.C.to(self.device)  # 分类器用GPU
        self.num_samples = X_task.shape[0]  # 总样本数
        self.X_fed = torch.from_numpy(self.X_fed).to(self.device).float()
        self.local_acc = []  # 准确率记录

        self.training_params = training_params  # 训练参数集合，包括d_LR,d_WD,g_LR,g_WD,enable_distill_penalty,batch_size,num_epochs,lmd
        # divide_line = X_task.shape[0] - X_fed.shape[0]  # 计算一个分割线，用于区分任务数据X_task和联邦数据X_fed

        indexed_dataset = IndexedDataset(X_task, y_task)

        data_holder = torch.utils.data.DataLoader(  # 创建一个数据加载器，用于按批次加载任务数据 X_task。其中的参数包括：
            dataset=indexed_dataset,  # 指定要加载的数据集
            batch_size=self.training_params['batch_size'],  # 指定每个批次的样本数量
            shuffle=False,  # 不打乱，因为在生成对抗网络中，训练样本的顺序通常是重要的
        )
        
        # 二元交叉熵损失和优化器
        self.loss = nn.BCELoss()  # 创建一个二分类交叉熵损失函数，衡量判别器输出和真实标签之间的差异
        self.d_fed_optimizer = torch.optim.Adam(  # 创建判别器优化器，采用Adam优化算法
            self.D_fed.parameters(),  # 指定要优化的判别器模型的参数
            lr=self.training_params['d_LR'],  # 判别器学习率
            weight_decay=self.training_params['d_WD'])  # 指定判别器权重衰减（weight decay）参数，用于控制正则化项的强度。
        self.d_local_optimizer = torch.optim.Adam(  # 创建判别器优化器，采用Adam优化算法
            self.D_local.parameters(),  # 指定要优化的判别器模型的参数
            lr=self.training_params['d_LR'],  # 判别器学习率
            weight_decay=self.training_params['d_WD'])  # 指定判别器权重衰减（weight decay）参数，用于控制正则化项的强度。
        self.g_fed_optimizer = torch.optim.Adam(
            self.G_fed.parameters(),  # 指定要优化的生成器模型的参数
            lr=self.training_params['g_LR'],  # 生成器学习率
            weight_decay=self.training_params['g_WD'])  # 指定生成器优化器的权重衰减参数。
        self.g_local_optimizer = torch.optim.Adam(
            self.G_local.parameters(),  # 指定要优化的生成器模型的参数
            lr=self.training_params['g_LR'],  # 生成器学习率
            weight_decay=self.training_params['g_WD'])  # 指定生成器优化器的权重衰减参数。
        self.g_e_c_optimizer = torch.optim.Adam(  # 创建生成器、特征提取器和分类的三者同时的优化器
            list(self.G_fed.parameters()) + list(self.E.parameters()) + list(self.C.parameters()),  # 指定要优化的生成器模型的参数
            lr=self.training_params['g_LR'],  # 生成器学习率
            weight_decay=self.training_params['g_WD'])  # 指定生成器优化器的权重衰减参数。
        for _ in range(self.training_params['num_epochs']):
            total_idx = 0  # 初始化一个变量，用于记录训练数据的总索引
            D_fed_index = []  # 遍历共享数据时，遍历X_fed的索引列表
            end_number = -1  # 初始时设置为-1，因为起点是0
            for _, (indices, data, labels) in enumerate(data_holder):  # 对数据加载器中的每个批次进行迭代
                total_idx += data.shape[0]  # 更新总索引，记录当前迭代处理的数据量
                z = torch.rand((data.shape[0], self.z_dim))  # 生成随机噪声z

                real_labels = Variable(torch.ones(self.training_params['batch_size'])).to(self.device)  # 创建包含全1的张量，表示真实标签
                fake_labels = Variable(torch.zeros(self.training_params['batch_size'])).to(self.device)  # 创建包含全0的张量，表示假标签
                z, data, labels = Variable(z.to(self.device)), Variable(data.to(self.device)), Variable(labels.to(self.device))  # 将输入数据和随机噪声转换为 PyTorch 变量

                z = z + data  # z'=z+X_task，用于生成器的输入
                # 当遍历共享数据时
                if indices[-1].item() < X_fed.shape[0]:  # 当遍历私有数据时，只需要重构损失和对抗损失(生成器)
                    z_former = z[indices < X_fed.shape[0]]
                    data_former = data[indices < X_fed.shape[0]]
                    labels_former = labels[indices < X_fed.shape[0]]
                    real_labels_former = real_labels[indices < X_fed.shape[0]]
                    fake_labels_former = fake_labels[indices < X_fed.shape[0]]

                    D_fed_index += list(range(end_number + 1, end_number + 1 + data_former.shape[0]))  # 添加 n 个连续数字，起点为上一次添加的末尾数字的下一个数字
                    end_number = D_fed_index[-1]  # 更新 end_number
                    x_fed = self.X_fed[D_fed_index[-data.shape[0]:]]
                    self.shared_train(z_former, data_former, x_fed, labels_former, real_labels_former, fake_labels_former)

                    # 共有层由G_fed赋值给G_local
                    for idx in self.g_shared_layer_indices:
                        self.G_local[idx].load_state_dict(self.G_fed[idx].state_dict())
                    # 共有层由D_fed赋值给D_local
                    for idx in self.d_shared_layer_indices:
                        self.D_local[idx].load_state_dict(self.D_fed[idx].state_dict())

                elif indices[0].item() < X_fed.shape[0] and indices[-1].item() >= X_fed.shape[0]:  # 当同时遍历私有数据和共享数据时，分开训练，(往往只会出现一次batch)
                    # 当遍历共享数据时
                    z_former = z[indices < X_fed.shape[0]]
                    data_former = data[indices < X_fed.shape[0]]
                    labels_former = labels[indices < X_fed.shape[0]]
                    real_labels_former = real_labels[indices < X_fed.shape[0]]
                    fake_labels_former = fake_labels[indices < X_fed.shape[0]]
                    D_fed_index += list(range(end_number + 1, end_number + 1 + data_former.shape[0]))  # 添加 n 个连续数字，起点为上一次添加的末尾数字的下一个数字
                    end_number = D_fed_index[-1]  # 更新 end_number
                    x_fed = self.X_fed[D_fed_index[-data_former.shape[0]:]]
                    self.shared_train(z_former, data_former, x_fed, labels_former, real_labels_former, fake_labels_former)

                    # 共有层由G_fed赋值给G_local
                    for idx in self.g_shared_layer_indices:
                        self.G_local[idx].load_state_dict(self.G_fed[idx].state_dict())
                    # 共有层由D_fed赋值给D_local
                    for idx in self.d_shared_layer_indices:
                        self.D_local[idx].load_state_dict(self.D_fed[idx].state_dict())

                    # 当遍历私有数据时
                    z_latter = z[indices >= X_fed.shape[0]]
                    data_latter = data[indices >= X_fed.shape[0]]
                    labels_latter = labels[indices >= X_fed.shape[0]]
                    real_labels_latter = real_labels[indices >= X_fed.shape[0]]
                    fake_labels_latter = fake_labels[indices >= X_fed.shape[0]]
                    self.private_train(z_latter, data_latter, labels_latter, real_labels_latter)

                    # 共有层由G_local赋值给G_fed
                    for idx in self.g_shared_layer_indices:
                        self.G_fed[idx].load_state_dict(self.G_local[idx].state_dict())

                # 当遍历私有数据时
                elif indices[0].item() > X_fed.shape[0]:  # 当遍历共享数据时，需要重构损失、对抗损失、蒸馏损失和分类损失(生成器)，对抗损失(判别器)
                    self.private_train(z, data, labels, real_labels)

                    # 共有层由G_local赋值给G_fed
                    for idx in self.g_shared_layer_indices:
                        self.G_fed[idx].load_state_dict(self.G_local[idx].state_dict())

        # Save the trained parameters
        self.save_model()  # 保存训练得到的生成器和判别器的参数

    def train_noPrivate(self, training_params, device):
        X_task = torch.from_numpy(self.X_task).to(device).float()
        # X_task = torch.from_numpy(self.X_task).to(device).to(torch.float32)
        X_fed = torch.from_numpy(self.X_fed).to(device).float()
        y_task = torch.from_numpy(self.y_task).to(device).float()

        self.device = device  # 用GPU显卡
        self.D_fed = self.D_fed.to(self.device)  # 判别器fed用GPU
        self.D_local = self.D_local.to(self.device)  # 判别器local用GPU
        self.G_fed = self.G_fed.to(self.device)  # 生成器fed用GPU
        self.G_local = self.G_local.to(self.device)  # 生成器local用GPU
        self.E = self.E.to(self.device)  # 特征提取器用GPU
        self.C = self.C.to(self.device)  # 分类器用GPU
        self.num_samples = X_task.shape[0]  # 总样本数
        self.X_fed = torch.from_numpy(self.X_fed).to(self.device).float()
        self.local_acc = []  # 准确率记录

        self.training_params = training_params  # 训练参数集合，包括d_LR,d_WD,g_LR,g_WD,enable_distill_penalty,batch_size,num_epochs,lmd
        # divide_line = X_task.shape[0] - X_fed.shape[0]  # 计算一个分割线，用于区分任务数据X_task和联邦数据X_fed

        indexed_dataset = IndexedDataset(X_task, y_task)

        data_holder = torch.utils.data.DataLoader(  # 创建一个数据加载器，用于按批次加载任务数据 X_task。其中的参数包括：
            dataset=indexed_dataset,  # 指定要加载的数据集
            batch_size=self.training_params['batch_size'],  # 指定每个批次的样本数量
            shuffle=False,  # 不打乱，因为在生成对抗网络中，训练样本的顺序通常是重要的
        )

        # 二元交叉熵损失和优化器
        self.loss = nn.BCELoss()  # 创建一个二分类交叉熵损失函数，衡量判别器输出和真实标签之间的差异
        self.d_fed_optimizer = torch.optim.Adam(  # 创建判别器优化器，采用Adam优化算法
            self.D_fed.parameters(),  # 指定要优化的判别器模型的参数
            lr=self.training_params['d_LR'],  # 判别器学习率
            weight_decay=self.training_params['d_WD'])  # 指定判别器权重衰减（weight decay）参数，用于控制正则化项的强度。
        self.d_local_optimizer = torch.optim.Adam(  # 创建判别器优化器，采用Adam优化算法
            self.D_local.parameters(),  # 指定要优化的判别器模型的参数
            lr=self.training_params['d_LR'],  # 判别器学习率
            weight_decay=self.training_params['d_WD'])  # 指定判别器权重衰减（weight decay）参数，用于控制正则化项的强度。
        self.g_fed_optimizer = torch.optim.Adam(
            self.G_fed.parameters(),  # 指定要优化的生成器模型的参数
            lr=self.training_params['g_LR'],  # 生成器学习率
            weight_decay=self.training_params['g_WD'])  # 指定生成器优化器的权重衰减参数。
        self.g_local_optimizer = torch.optim.Adam(
            self.G_local.parameters(),  # 指定要优化的生成器模型的参数
            lr=self.training_params['g_LR'],  # 生成器学习率
            weight_decay=self.training_params['g_WD'])  # 指定生成器优化器的权重衰减参数。
        self.g_e_c_optimizer = torch.optim.Adam(  # 创建生成器、特征提取器和分类的三者同时的优化器
            list(self.G_fed.parameters()) + list(self.E.parameters()) + list(self.C.parameters()),  # 指定要优化的生成器模型的参数
            lr=self.training_params['g_LR'],  # 生成器学习率
            weight_decay=self.training_params['g_WD'])  # 指定生成器优化器的权重衰减参数。
        for _ in range(self.training_params['num_epochs']):
            total_idx = 0  # 初始化一个变量，用于记录训练数据的总索引
            D_fed_index = []  # 遍历共享数据时，遍历X_fed的索引列表
            end_number = -1  # 初始时设置为-1，因为起点是0
            for _, (indices, data, labels) in enumerate(data_holder):  # 对数据加载器中的每个批次进行迭代
                total_idx += data.shape[0]  # 更新总索引，记录当前迭代处理的数据量
                z = torch.rand((data.shape[0], self.z_dim))  # 生成随机噪声z

                real_labels = Variable(torch.ones(self.training_params['batch_size'])).to(
                    self.device)  # 创建包含全1的张量，表示真实标签
                fake_labels = Variable(torch.zeros(self.training_params['batch_size'])).to(
                    self.device)  # 创建包含全0的张量，表示假标签
                z, data, labels = Variable(z.to(self.device)), Variable(data.to(self.device)), Variable(
                    labels.to(self.device))  # 将输入数据和随机噪声转换为 PyTorch 变量

                z = z + data  # z'=z+X_task，用于生成器的输入
                # 当遍历共享数据时
                if indices[-1].item() < X_fed.shape[0]:  # 当遍历私有数据时，只需要重构损失和对抗损失(生成器)
                    z_former = z[indices < X_fed.shape[0]]
                    data_former = data[indices < X_fed.shape[0]]
                    labels_former = labels[indices < X_fed.shape[0]]
                    real_labels_former = real_labels[indices < X_fed.shape[0]]
                    fake_labels_former = fake_labels[indices < X_fed.shape[0]]

                    D_fed_index += list(
                        range(end_number + 1, end_number + 1 + data_former.shape[0]))  # 添加 n 个连续数字，起点为上一次添加的末尾数字的下一个数字
                    end_number = D_fed_index[-1]  # 更新 end_number
                    x_fed = self.X_fed[D_fed_index[-data.shape[0]:]]
                    self.shared_train(z_former, data_former, x_fed, labels_former, real_labels_former,
                                      fake_labels_former)

                    # 共有层由G_fed赋值给G_local
                    for idx in self.g_shared_layer_indices:
                        self.G_local[idx].load_state_dict(self.G_fed[idx].state_dict())
                    # 共有层由D_fed赋值给D_local
                    for idx in self.d_shared_layer_indices:
                        self.D_local[idx].load_state_dict(self.D_fed[idx].state_dict())

                elif indices[0].item() < X_fed.shape[0] and indices[-1].item() >= X_fed.shape[
                    0]:  # 当同时遍历私有数据和共享数据时，分开训练，(往往只会出现一次batch)
                    # 当遍历共享数据时
                    z_former = z[indices < X_fed.shape[0]]
                    data_former = data[indices < X_fed.shape[0]]
                    labels_former = labels[indices < X_fed.shape[0]]
                    real_labels_former = real_labels[indices < X_fed.shape[0]]
                    fake_labels_former = fake_labels[indices < X_fed.shape[0]]
                    D_fed_index += list(
                        range(end_number + 1, end_number + 1 + data_former.shape[0]))  # 添加 n 个连续数字，起点为上一次添加的末尾数字的下一个数字
                    end_number = D_fed_index[-1]  # 更新 end_number
                    x_fed = self.X_fed[D_fed_index[-data_former.shape[0]:]]
                    self.shared_train(z_former, data_former, x_fed, labels_former, real_labels_former,
                                      fake_labels_former)

                    # 共有层由G_fed赋值给G_local
                    for idx in self.g_shared_layer_indices:
                        self.G_local[idx].load_state_dict(self.G_fed[idx].state_dict())
                    # 共有层由D_fed赋值给D_local
                    for idx in self.d_shared_layer_indices:
                        self.D_local[idx].load_state_dict(self.D_fed[idx].state_dict())

        # Save the trained parameters
        self.save_model()  # 保存训练得到的生成器和判别器的参数

    # 私有数据训练
    def private_train(self, z, data, labels, real_labels):
        # ==================================训练生成器==================================
        fake_data = self.G_local(z)
        # outputs = self.D_local(fake_data[:, :data.shape[1]])
        outputs = self.D_local(fake_data)

        recons_loss = torch.mean(torch.abs(data - fake_data[:, :data.shape[1]]))  # 重构损失
        adversarial_loss = self.loss(outputs.flatten(), real_labels[:outputs.flatten().shape[0]])  # 对抗损失

        # 消融
        if self.ablation_part == "Private":
            if self.ablation_loss == "adversarial":
                self.gamma_adversarial_private = torch.nn.Parameter(torch.Tensor([0]).to(self.device), requires_grad=True)
            elif self.ablation_loss == "recons":
                self.gamma_recons_private = torch.nn.Parameter(torch.Tensor([0]).to(self.device), requires_grad=True)
            else:
                pass

        g_loss = self.gamma_adversarial_private * adversarial_loss + self.gamma_recons_private * recons_loss

        self.weight_optimizer_private.zero_grad()
        self.D_local.zero_grad()  # 清零判别器的梯度
        self.G_local.zero_grad()  # 清零生成器的梯度
        g_loss.backward()  # 反向传播计算生成器的梯度
        self.weight_optimizer_private.step()
        self.normalize_weights_private()
        self.g_local_optimizer.step()  # 根据梯度，通过优化器来更新生成器的参数

    # 共享数据训练
    def shared_train(self, z, data, x_fed, labels, real_labels, fake_labels):
        # ==================================训练判别器==================================
        # 用真样本计算二分类交叉熵d_loss_real BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
        outputs = self.D_fed(torch.cat([x_fed, labels], dim=1))
        d_loss_real = self.loss(outputs.flatten(), real_labels)
        real_score = outputs

        # 用假样本计算二分类交叉熵d_loss_fake
        fake_data = self.G_fed(torch.cat([z, labels], dim=1))
        outputs = self.D_fed(torch.cat([fake_data, labels], dim=1))
        d_loss_fake = self.loss(outputs.flatten(), fake_labels)
        fake_score = outputs

        # 更新判别器
        d_loss = d_loss_real + d_loss_fake
        self.D_fed.zero_grad()  # 清零判别器的梯度
        d_loss.backward()  # 反向传播计算判别器的梯度
        self.d_fed_optimizer.step()  # 根据梯度，通过优化器来更新判别器的参数

        # ==================================训练生成器==================================
        z = Variable(torch.randn(data.shape[0], self.z_dim).to(self.device))  # 生成新的随机噪声z
        z = z + data  # z'=z+X_task，用于生成器的输入
        fake_data = self.G_fed(torch.cat([z, labels], dim=1))
        outputs = self.D_fed(torch.cat([fake_data, labels], dim=1))
        recons_loss = torch.mean(torch.abs(data - fake_data[:, :data.shape[1]]))  # 重构损失
        adversarial_loss = self.loss(outputs.flatten(), real_labels)  # 对抗损失

        h = self.E(fake_data.unsqueeze(1).permute(0, 2, 1))  # 特征提取
        h_fed = self.E(x_fed.unsqueeze(1).permute(0, 2, 1))  # 特征提取
        classifier_outputs = self.C(h)  # 分类

        cosine_similarity = F.cosine_similarity(h, h_fed, dim=1)  # task表示和data表示的余弦值
        distill_loss = 1 - cosine_similarity.mean()  # task表示和data表示的余弦相似度

        ce_loss = F.cross_entropy(torch.argmax(classifier_outputs, dim=1).unsqueeze(1).float(), labels)  # 分类损失

        # predicted_labels = torch.argmax(classifier_outputs, dim=1)  # 1. 获取预测标签
        # correct_predictions = torch.sum(predicted_labels == labels)  # 2. 计算准确预测的数量
        # accuracy = correct_predictions.item() / len(labels)  # 3. 计算准确率
        # print("准确率：", accuracy)

        # 消融
        if self.ablation_part == "Shared":
            if self.ablation_loss == "adversarial":
                self.gamma_adversarial_shared = torch.nn.Parameter(torch.Tensor([0]).to(self.device), requires_grad=True)
            elif self.ablation_loss == "recons":
                self.gamma_recons_shared = torch.nn.Parameter(torch.Tensor([0]).to(self.device), requires_grad=True)
            elif self.ablation_loss == "distill":
                self.gamma_distill_shared = torch.nn.Parameter(torch.Tensor([0]).to(self.device), requires_grad=True)
            elif self.ablation_loss == "ce":
                self.gamma_classify_shared = torch.nn.Parameter(torch.Tensor([0]).to(self.device), requires_grad=True)
            else:
                pass
        g_loss = self.gamma_adversarial_shared * adversarial_loss + self.gamma_recons_shared * recons_loss + self.gamma_distill_shared * distill_loss + self.gamma_classify_shared * ce_loss

        # # "蒸馏惩罚"
        # if self.training_params[
        #     'enable_distill_penalty'] and total_idx > divide_line:  # 检查是否启用了“distill penalty”并且当前数据处于外部数据范围内
        #     soft_label = torch.abs(fake_data - x_fed[total_idx - data.shape[0] - divide_line:total_idx - divide_line, :]).sum()  # 计算额外的损失，这里使用了平滑标签损失。将生成的假样本与外部数据进行比较，并计算其绝对差的和
        #     g_loss += soft_label * self.training_params['lmd']  # 将平滑标签损失加到生成器的总损失上，并乘以超参数 lmd 控制其影响程度

        # Optimize generator
        self.weight_optimizer_shared.zero_grad()
        self.D_fed.zero_grad()  # 清零判别器的梯度
        self.G_fed.zero_grad()  # 清零生成器的梯度
        self.E.zero_grad()  # 清零特征提取器的梯度
        self.C.zero_grad()  # 清零分类器的梯度
        g_loss.backward()  # 反向传播计算生成器的梯度
        self.weight_optimizer_shared.step()
        self.g_e_c_optimizer.step()  # 根据梯度，通过优化器来更新生成器的参数 优化对抗损失和重构损失

    # 生成表示
    def generate_representation(self, G_fed_model_path, y_task):
        self.load_model(G_fed_model_path)  # 判别器和生成器通过load_model方法加载预训练的判别器和生成器模型的参数
        z = Variable(torch.randn(self.num_samples, self.z_dim).to(self.device))  # 生成一个随机噪声z，用于生成器的输入
        x = Variable(torch.from_numpy(self.X_task).to(self.device)).float()
        z = z + x  # z'=z+X_task，用于生成器的输入
        y_task = torch.from_numpy(y_task).to(self.device).float()
        samples = self.G_fed(torch.cat([z, y_task], dim=1))  # 生成假数据
        # 归一化谨慎用
        samples = samples.mul(0.5).add(0.5)  # 对生成的样本进行归一化，将其值范围从[-1, 1]映射到[0, 1]。这是因为生成器的输出通常在[-1, 1]范围内，而归一化后希望将其映射到[0, 1]范围，以得到更符合常规图像表示的结果。
        print("生成的最终补全task数据形状：", self.to_np(samples).shape)
        return self.to_np(samples)  # 生成数据转类型为numpy

    def to_np(self, X):
        return X.data.cpu().numpy()

    # 保存训练好的模型
    def save_model(self):
        torch.save(self.G_fed.state_dict(), os.path.join(self.dir, 'task_fed_generator.pkl'))
        torch.save(self.G_local.state_dict(), os.path.join(self.dir, 'task_local_generator.pkl'))
        torch.save(self.D_fed.state_dict(), os.path.join(self.dir, 'task_fed_discriminator.pkl'))
        torch.save(self.D_local.state_dict(), os.path.join(self.dir, 'task_local_discriminator.pkl'))
        torch.save(self.E.state_dict(), os.path.join(self.dir, 'task_extractor.pkl'))
        torch.save(self.C.state_dict(), os.path.join(self.dir, 'task_classifier.pkl'))
        print(f'CrossKT-FDA Models save to {self.dir}task_fed_generator.pkl & {self.dir}task_local_generator.pkl & {self.dir}task_fed_discriminator.pkl & {self.dir}task_local_discriminator.pkl & {self.dir}task_extractor.pkl & {self.dir}task_classifier.pkl')

    # 加载预训练的生成器和判别器的模型参数
    def load_model(self, G_fed_model_filename):
    # def load_model(self, D_fed_model_filename, D_local_model_filename, G_fed_model_filename, G_local_model_filename, E_model_filename, C_model_filename):
        # D_fed_model_path = os.path.join(os.getcwd(), D_fed_model_filename)
        # D_local_model_path = os.path.join(os.getcwd(), D_local_model_filename)
        G_fed_model_path = os.path.join(os.getcwd(), G_fed_model_filename)
        # G_local_model_path = os.path.join(os.getcwd(), G_local_model_filename)
        # E_model_path = os.path.join(os.getcwd(), E_model_filename)
        # C_model_path = os.path.join(os.getcwd(), C_model_filename)
        # self.D_fed.load_state_dict(torch.load(D_fed_model_path))
        # self.D_local.load_state_dict(torch.load(D_local_model_path))
        self.G_fed.load_state_dict(torch.load(G_fed_model_path))
        # self.G_local.load_state_dict(torch.load(G_local_model_path))
        # self.E.load_state_dict(torch.load(E_model_path))
        # self.C.load_state_dict(torch.load(C_model_path))
        # print('Discriminator fed model loaded from {}.'.format(D_fed_model_path))
        # print('Discriminator local model loaded from {}.'.format(D_local_model_path))
        print('Generator fed model loaded from {}.'.format(G_fed_model_path))
        # print('Generator local model loaded from {}.'.format(G_local_model_path))
        # print('Extractor model loaded from {}.'.format(E_model_path))
        # print('Classifier model loaded from {}.'.format(C_model_path))

    @property
    def name(self):
        return 'GAN'

# if __name__ == '__main__':
#     gan = GAN(1024, 2048)
#     X_task = torch.normal(0, 1, size=(1000, 1024))
#     X_fed = torch.normal(0, 1, size=(1000, 1024))
#     params = {
#         'batch_size': 100,
#         'num_epochs': 5,
#         'd_LR': 0.0002,
#         'd_WD': 0.00001,
#         'g_LR': 0.0002,
#         'g_WD': 0.00001,
#         'enable_distill_penalty': True,
#         'lmd': 0.0001
#     }
#     gan.train(X_task, X_fed, params, 0)
#     gan.generate_representation('./discriminator.pkl', './generator.pkl')