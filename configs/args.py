import argparse
from utils import *

parser = argparse.ArgumentParser()
# MIMIC 58976×26 4标签；
parser.add_argument('--dataset', type=str, default="MIMIC", choices=["MIMIC", "HAPT", "RNASeq", "Breast", "MNIST", "CIFAR10", "ALLAML", "Diagnosis"])
parser.add_argument('--data_type', type=str, default="CNN", choices=["CNN", "DNN"])  # data方特征提取器和分类器用模型
parser.add_argument('--n_clients', type=int, default=1, help="the number of data sides")
parser.add_argument('--split', type=str, default="unequal", choices=["equal", "unequal"])  # 仿真切分规则
parser.add_argument('--frl', type=str, default="VFedPCA", choices=["VFedPCA", "FedSVD"])  # 表示学习的方法
parser.add_argument('--gpu', type=str, default='0', help='gpu device id')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
# parser.add_argument('--experiment1', type=str, default="Base", choices=["Base", "Ablation", "Downstream"], help='实验大类')
parser.add_argument('--experiment2', type=str, default="noPrivate", choices=["", "Shared", "Private", "noPrivate", "VFL", "Local"], help='实验小类（1是base对比，234是消融，56是下游任务）')
parser.add_argument('--experiment3', type=str, default="task_sample", choices=["task_feature", "task_sample", "data_feature", "data_sample", "shared_feature", "shared_sample", "IID_nonIID", "compute_time"], help='控制变量')
parser.add_argument('--ablation', type=str, default="", choices=["", "adversarial", "recons", "distill", "ce"])
parser.add_argument('--downstream', type=str, default="NN", help="RF|Xgb|NN|Ada|KNN")
# parser.add_argument('--sample_interval', type=int, default=200, help='Save the epoch interval for generating samples')
# parser.add_argument('--optimizer', type=str, default="Adam", choices=["Adam", "SGD"])
# parser.add_argument('--gan_epochs', type=int, default=1500, help="the epochs for gan training")
# parser.add_argument('--fed_communications', type=int, default=1, help="the communications for fed training")
# parser.add_argument('--fed_epochs', type=int, default=1, help="the epochs for fed training")
# parser.add_argument('--use_trained_GD', type=int, default=1, help="use trained model or not")
# parser.add_argument('--use_trained_comparison', type=int, default=1, help="use trained downstream or not")
# parser.add_argument('-k', action="store", dest="neighbour_size", type=int, default=16, help="text data LaplacianScore neighbour_size")
# parser.add_argument('-t', action="store", dest="t_param", type=int, default=2, help="text data LaplacianScore t_param")

args = parser.parse_args()

# if args.experiment1 == "Base" or args.experiment1 == "Downstream":
#     args.method = "CrossKTFRA"
# else:
#     args.method = ""

# 保存路径控制变量（e.g任务方特征数）
setting = load_dataset_config(args.dataset, f'{args.split}_split')
if args.experiment3 == "task_feature":
    args.parameter = setting['task_num_feature']
elif args.experiment3 == "task_sample":
    args.parameter = setting['task_num_sample']
elif args.experiment3 == "data_feature":
    args.parameter = setting['data_num_feature']
elif args.experiment3 == "data_sample":
    args.parameter = setting['data_num_sample']
elif args.experiment3 == "shared_feature":
    args.parameter = setting['shared_feature']
elif args.experiment3 == "shared_sample":
    args.parameter = setting['shared_sample']
else:
    args.parameter = ""

if args.dataset == "MIMIC" or "HAPT" or "TCGA" or "Breast":
    args.data_type = "csv"
elif args.dataset == "MNIST" or "CIFAR10":
    args.data_type = "image"

if args.frl == "VFedPCA":
    args.frl_params = {
        "name": 'VFedPCA',
        "iter_num": 10,  # VFedPCA中每个客户端幂迭代算法训练次数
        "party_num": args.n_clients + 1,  # 参与方是task方和data方一起
        "warm_start": False,  # 热启动一般是采用“相关或简化问题的最优解”来作为原问题的初始值
        "period_num": 5,  # 通信次数
        "weight_scale": False,  # 模型压缩的技术，用于将模型的权重（weight）从浮点数（float）转换为定点数（int），从而减少模型的大小和内存占用
    }
elif args.frl == "FedSVD":
    args.frl_params = {
        "name": 'FedSVD',
        "random_seed": 800,
        "block_size": 10,
    }

args.lrd_params = {
    "name": 'GAN',
    "d_depth": 4,  # 判别器层数
    "g_depth": 4,  # 生成器层数
    "negative_slope": 0.2,  # 添加LeakyReLU激活函数的参数
    "d_LR": 0.0002,  # 判别器学习率
    "d_WD": 0.00001,  # 判别器权重衰减
    "g_LR": 0.0002,  # 生成器学习率
    "g_WD": 0.00001,  # 生成器权重衰减
    "enable_distill_penalty": True,  # "蒸馏惩罚"
    "batch_size": 100,
    "num_epochs": 20,
    "lmd": 0.0000001,  # "蒸馏惩罚"

    "gamma_1": 0.7,  # 对抗损失
    "gamma_2": 0.1,  # 重构损失
    "gamma_3": 0.1,  # 蒸馏损失
    "gamma_4": 0.1,  # 分类损失

    "ablation_part": args.experiment2,  # 消融的是共享数据还是私有数据
    "ablation_loss": args.ablation,  # 消融损失组件
}

# if args.dataset == "MNIST":
#     args.img_rows = 28  # 图片的第1维
#     args.img_cols = 28  # 图片的第2维
#     args.img_channels = 1  # 图片的通道数
#     args.img_shape = (args.img_rows, args.img_cols, args.img_channels)  # MNIST图片的形状
#     # args.latent_dim = args.img_rows * args.img_cols * args.img_channels  # 噪声潜在空间的维数
#     args.num_classes = 10  # MNIST标签类别数
#     args.txt_img = 1  # 为图片数据集
#
# elif args.dataset == "CIFAR10":
#     args.img_rows = 32  # 图片的第1维
#     args.img_cols = 32  # 图片的第2维
#     args.img_channels = 3  # 图片的通道数
#     args.img_shape = (args.img_rows, args.img_cols, args.img_channels)  # MNIST图片的形状
#     # args.latent_dim = args.img_rows * args.img_cols * args.img_channels  # 噪声潜在空间的维数
#     args.num_classes = 10  # MNIST标签类别数
#     args.txt_img = 1  # 为图片数据集
#
# elif args.dataset == "ALLAML":
#     args.num_classes = 2  # MNIST标签类别数
#     args.txt_img = 0  # 为图片数据集
#
# elif args.dataset == "Diagnosis":
#     args.num_classes = 11  # MNIST标签类别数
#     args.txt_img = 0  # 为图片数据集
#
# if args.optimizer == "Adam":
#     args.lr = 2e-4  # 学习率
#     args.beta1 = 1e-4  # beta1
