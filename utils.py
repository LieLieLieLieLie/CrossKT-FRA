import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import json
import os

from model.CNN import *
# from model.DNN import *

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# 指定数据集获取X和y
def load_data(dataset: str):
    if dataset == 'HAPT':
        X_train = np.loadtxt('dataset/HAPT/X_train.txt', usecols=range(208))
        y_train = np.loadtxt('dataset/HAPT/y_train.txt')[:4333]
        X_test = np.loadtxt('dataset/HAPT/X_test.txt', usecols=range(208))
        y_test = np.loadtxt('dataset/HAPT/y_test.txt')[:3162]
        X = np.concatenate([X_train, X_test], axis=0).astype(float)
        y = np.concatenate([y_train, y_test], axis=0).astype(float).reshape([X.shape[0], 1]) - 1
        num_classes = len(np.unique(y))
    elif dataset == 'RNASeq':
        # X = np.loadtxt('dataset/RNA-Seq/data.csv', delimiter=',')
        X_original = np.genfromtxt('dataset/RNA-Seq/data.csv', delimiter=',', skip_header=1, usecols=range(1, 16382),
                                   dtype=float, filling_values=np.nan)
        min_val = np.min(X_original)
        max_val = np.max(X_original)
        X = (X_original - min_val) / (max_val - min_val)
        # y = np.loadtxt('dataset/RNA-Seq/labels.csv')
        y_original = np.genfromtxt('dataset/RNA-Seq/labels.csv', delimiter=',', skip_header=1, usecols=range(1, 2),
                                   dtype=str, filling_values=np.nan)
        y_original = y_original.reshape((y_original.shape[0], 1))
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_original)
        y = y.reshape((y.shape[0], 1))
        num_classes = len(np.unique(y))
    elif dataset == 'MIMIC':
        data = pd.read_csv('dataset/MIMIC/mimic3d.csv')[20000:]
        drop_cols = [
            'LOSgroupNum', 'hadm_id', 'AdmitDiagnosis',
            'AdmitProcedure', 'religion', 'insurance',
            'ethnicity', 'marital_status', 'ExpiredHospital',
            'LOSdays', 'gender', 'admit_type', 'admit_location']
        X = data.drop(drop_cols, axis=1)
        y = data['LOSgroupNum'].to_numpy()
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        y = y.reshape((len(y), 1))
        num_classes = len(np.unique(y))
        print(X.shape)
    elif dataset == 'Breast':
        data = pd.read_csv('dataset/breast/data.csv', names=['id', 'diagnosis', 'radius1', 'texture1', 'perimeter1',
                                                             'area1', 'smoothness1', 'compactness1', 'concavity1',
                                                             'concave_points1', 'symmetry1',
                                                             'fractal_dimension1', 'radius2', 'texture2', 'perimeter2',
                                                             'area2', 'smoothness2', 'compactness2',
                                                             'concavity2', 'concave_points2', 'symmetry2',
                                                             'fractal_dimension2', 'radius3', 'texture3', 'perimeter3',
                                                             'area3', 'smoothness3', 'compactness3', 'concavity3',
                                                             'concave_points3', 'symmetry3', 'fractal_dimension3'])
        data.drop(["id"], inplace=True, axis=1)
        data = data.rename(columns={"diagnosis": "target"})
        data["target"] = [1 if i.strip() == "M" else 0 for i in data.target]
        X = data.drop(["target"], axis=1).to_numpy()
        y = data.target.to_numpy().reshape((len(X), 1))
        num_classes = len(np.unique(y))
    elif dataset == 'Diagnosis':
        data = pd.read_csv('dataset/Diagnosis/Sensorless_drive_diagnosis.csv', header=None)[:-1]
        data = data.sample(frac=1).reset_index(drop=True)  # 原顺序是按类别排的，现在要均匀打乱
        X = data.iloc[:, :-1].values  # 所有列除了最后一列是特征，最小值是-15.，为了求KL散度所有值得是正值
        y = data.iloc[:, -1].values - 1  # 最后一列是标签
        # y = pd.factorize(y)[0]
        min_max_scaler = MinMaxScaler()  # Min-Max规范化
        X = min_max_scaler.fit_transform(X)
        # scaler = StandardScaler()
        # X = scaler.fit_transform(X)
        y = y.reshape((len(y), 1))
        num_classes = len(np.unique(y))
    else:
        print('Wrong data name!')
        return

    return X, y, num_classes


# torch调用gpu
def get_gpu(gpu_idx='0'):
    return torch.device('cuda:' + gpu_idx if torch.cuda.is_available() else "cpu")
    # return torch.device("cpu")


# 获取数据集切割方式的对应参数
def load_dataset_config(dataset: str, type: str):
    with open('configs/split.json') as load_config:
        config = json.load(load_config)
    return config[dataset][type]


# 平等切割 每个客户端切割得到的样本数和特征数都是相同的
def equal_split(dataset='MIMIC', start_sample=0, start_feature=0, shared_sample=None, shared_feature=None,
                num_sample=None, num_feature=None):
    X, y, num_classes = load_data(dataset)  # 指定数据集获取X和y
    setting = load_dataset_config(dataset, 'equal_split')  # 获取数据集的该种切割方式对应参数

    num_sample = setting['num_sample'] if num_sample == None else num_sample
    num_feature = setting['num_feature'] if num_feature == None else num_feature
    shared_sample = setting['shared_sample'] if shared_sample == None else shared_sample
    shared_feature = setting['shared_feature'] if shared_feature == None else shared_feature

    # Task party 样本0-5000 特征0-4 切割取1000之后的num_sample个样本(可能前1000噪声) 取前num_feature个特征
    X_task = X[start_sample:start_sample + num_sample, start_feature:start_feature + num_feature]
    y_task = y[start_sample:start_sample + num_sample, :]

    # Data party 样本1000-6000 特征4-8 切割取(num_sample - shared_sample)后的num_sample个样本 取(num_feature - shared_feature)后的num_feature个特征
    X_data = X[start_sample + num_sample - shared_sample:start_sample + 2 * num_sample - shared_sample,
             start_feature + num_feature - shared_feature:start_feature + 2 * num_feature - shared_feature]  # 与task方错开shared_sample和shared_feature切割数据集
    y_data = y[start_sample + num_sample - shared_sample:start_sample + 2 * num_sample - shared_sample, :]

    # Shared samples task_shared样本1000-5000特征0-4 data_shared样本1000-5000特征5-8
    task_shared = X_task[num_sample - shared_sample:num_sample, :]  # task中data不拥有的那部分用于共享??多传了个4
    data_shared = X_data[:shared_sample, shared_feature:num_feature]  # data中task不拥有的那部分用于共享
    X_shared = np.concatenate([task_shared, data_shared], axis=1)  # 把两方的共享数据纵向合并为共享数据
    y_shared = y_data[:shared_sample]

    X = X[start_sample:start_sample + num_sample, :]
    y = y[start_sample:start_sample + num_sample, :]

    X_task = np.concatenate((X_task[num_sample - shared_sample:, :], X_task[:num_sample - shared_sample, :]))  # 重要，共享部分优先学习

    return X_task, y_task, X_shared, y_shared, X_data, y_data, task_shared, data_shared, num_classes  # 返回task方的X和y，共享的X_shared和data方的X和y


# 平等切割 每个客户端切割得到的样本数和特征数是不相同的，且差异较大
def unequal_split(dataset='MIMIC', start=0, task_feature=None, data_feature=None, shared_sample=None, shared_feature=None):
    X, y, num_classes = load_data(dataset)  # 指定数据集获取X和y
    setting = load_dataset_config(dataset, 'unequal_split')  # 获取数据集的该种切割方式对应参数

    task_num_sample = setting['task_num_sample']  # 切割的task方样本个数
    data_num_sample = setting['data_num_sample']  # 切割的data方样本个数
    task_num_feature = setting['task_num_feature'] if task_feature == None else task_feature  # 切割的task方特征个数
    data_num_feature = setting['data_num_feature'] if data_feature == None else data_feature  # 切割的data方特征个数
    shared_sample = setting['shared_sample'] if shared_sample == None else shared_sample  # 切割的task方与data方重叠样本个数
    shared_feature = setting['shared_feature'] if shared_feature == None else shared_feature  # 切割的task方与data方重叠特征个数

    # Task party
    X_task = X[start:start + task_num_sample, :task_num_feature]  # 切割取前task_num_sample个样本，前task_num_feature个特征
    y_task = y[start:start + task_num_sample, :]

    # Data party
    X_data = X[start + task_num_sample - shared_sample:start + task_num_sample + data_num_sample - shared_sample,
         task_num_feature - shared_feature:task_num_feature + data_num_feature - shared_feature]
        # 切割取从(task_num_sample - shared_sample)开始的data_num_sample个样本(保证有shared_sample个样本能重叠)
        # 切割取从(task_num_feature - shared_feature)开始的data_num_feature个样本(保证有shared_feature个样本能重叠)
    y_data = y[start + task_num_sample - shared_sample:start + task_num_sample + data_num_sample - shared_sample, :]

    # Shared samples
    task_shared = X_task[task_num_sample - shared_sample:task_num_sample, :]  # 把重叠部分作为共享数据
    data_shared = X_data[:shared_sample, shared_feature:data_num_feature]  # 把重叠部分作为共享数据
    X_shared = np.concatenate([task_shared, data_shared], axis=1)  # 把两方的共享数据纵向合并为共享数据
    y_shared = y_data[:shared_sample]

    # X = X[start:start + task_num_sample, :]
    # y = y[start:start + task_num_sample, :]

    X_task = np.concatenate((X_task[task_num_sample - shared_sample:, :], X_task[:task_num_sample - shared_sample, :]))  # 重要，共享部分优先学习

    return X_task, y_task, X_shared, y_shared, X_data, y_data, task_shared, data_shared, num_classes


# 非独立同分布的切割 原始数据是均匀分布的(每个标签对应的样本数是均匀的)，现取select_num个标签数，每个标签只取noniid_size个样本，这样在切割前数据是标签对应样本数有不均衡性 **不过原作者没用到这个
def imbalanced_split(dataset='MIMIC', type='iid', select_num=1, noniid_size=100):
    X, y, num_classes = load_data(dataset)
    data = pd.DataFrame(np.concatenate([X, y], axis=1))
    setting = load_dataset_config(dataset, 'imbalanced_split')

    task_num_sample = setting['task_num_sample']  # 切割的task方样本个数
    data_num_sample = setting['data_num_sample']  # 切割的data方样本个数
    task_num_feature = setting['task_num_feature']  # 切割的task方特征个数
    data_num_feature = setting['data_num_feature']  # 切割的data方特征个数
    shared_sample = setting['shared_sample']  # 切割的task方与data方重叠样本个数
    shared_feature = setting['shared_feature']  # 切割的task方与data方重叠特征个数
    if type == 'iid':  # 独立同分布
        pass
    else:  # 非独立同分布
        for i in range(select_num):
            select_label = data[data.shape[1] - 1].max()  # 选择标签的最大值
            if i == 0:
                data_new = data[data[data.shape[1] - 1] == select_label].to_numpy()  # data_new是取标签为最大值的所有行
            else:
                data_new = np.concatenate([data_new, data[data[data.shape[1] - 1] == select_label].to_numpy()], axis=0)  # data_new是取标签为最大值的所有行
            data = data[data[data.shape[1] - 1] != select_label]  # data是取标签不为最大值的所有行
        data = data.to_numpy()
        np.random.shuffle(data_new)  # 打乱data_new顺序
        data = np.concatenate([data, data_new[noniid_size:, :]], axis=0)  # 把data_new取noniid_size个放原数据后，方便后续切割时，data方有一部分标签的样本是task方没有的，因为data方拿到的是原数据的后部分
        np.random.shuffle(data)  # 重新打乱
        X, y = data[:, :-1], data[:, -1:].astype(int)  # 重新分开为特征和标签

        # Task party
        X_task = X[:task_num_sample, :task_num_feature]  # 切割取前task_num_sample个样本，前task_num_feature个特征
        y_task = y[:task_num_sample, :]

        # Data party
        X_data = X[task_num_sample - shared_sample:task_num_sample + data_num_sample - shared_sample,
            task_num_feature - shared_feature:task_num_feature + data_num_feature - shared_feature]
        y_data = y[task_num_sample - shared_sample:task_num_sample + data_num_sample - shared_sample, :]
        # 切割取从(task_num_sample - shared_sample)开始的data_num_sample个样本(保证有shared_sample个样本能重叠)
        # 切割取从(task_num_feature - shared_feature)开始的data_num_feature个样本(保证有shared_feature个样本能重叠)

        # Shared samples
        task_shared = X_task[task_num_sample - shared_sample:task_num_sample, :]  # 把重叠部分作为共享数据
        data_shared = X_data[:shared_sample, shared_feature:data_num_feature]  # 把重叠部分作为共享数据
        X_shared = np.concatenate([task_shared, data_shared], axis=1)  # 把两方的共享数据纵向合并为共享数据
        y_shared = y_data[:shared_sample]

        # New samples
        data_new = data_new[:noniid_size, :]
        X_new, y_new = data_new[:, :task_num_feature], data_new[:, -1:].astype(int)

        X_task = np.concatenate((X_task[task_num_sample - shared_sample:, :], X_task[:task_num_sample - shared_sample, :]))  # 重要，共享部分优先学习

    return X_task, y_task, X_shared, y_shared, X_data, X_new, y_new, num_classes


# 学习综合数据
class FedRepresentationLearning():
    def __init__(self, frl_model, params, dir) -> None:
        self.frl_model = frl_model  # frl方法类
        self.params = params  # frl方法配置参数
        self.dir = dir  # X_fed保存路径

    def training(self, **kwargs):
        if self.params['name'] == 'FedSVD':
            self.frl_model.load_data(kwargs['X_shared'])
            self.frl_model.learning()
            X_fed = self.frl_model.get_fed_representation()
        elif self.params['name'] == 'VFedPCA':
            X_task_shared, X_data_shared = kwargs['X_task_shared'], kwargs['X_data_shared']
            X_fed = self.frl_model.fed_representation_learning(
                self.params,
                kwargs['X_shared'],
                [X_task_shared, X_data_shared])
        np.savez(f"{self.dir}/X_fed.npz",
                 X_fed=X_fed, )
        return X_fed


def data_method_select(data_type, num_classes, input_features, representation_features):
    extractor = CNNFeatureExtractor(data_type, input_features, representation_features)  # 创建卷积特征提取器
    classifier = CNNClassifier(representation_features, num_classes)  # 创建卷积分类器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(extractor.parameters()) + list(classifier.parameters()), lr=0.001)
    return extractor, classifier, criterion, optimizer


class LocalRepresentationDistillation():
    def __init__(self, lrd_model, params, device, dir) -> None:
        self.model = lrd_model  # 生成式模型
        self.params = params  # 生成式模型参数
        self.device = device  # gpu
        self.total_representation = None
        self.dir = dir

    def train(self):
        self.model.train(self.params, self.device)

    def representation_distillation_step(self, y_task):
        representation = self.model.generate_representation(os.path.join(self.dir, 'task_fed_generator.pkl'), y_task)  # 调跑完的GAN
        return representation

# 给每次进GAN的X_task排序，当对应X_Fed时才计算损失
class IndexedDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.length = len(data)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Return both the data and the index
        return index, self.data[index], self.label[index]


def downstream_method_select(method_name):
    if method_name == 'Log':
        model = LogisticRegression()
    elif method_name == 'KNN':
        model = KNeighborsClassifier(n_neighbors=8)
    elif method_name == 'Xgb':
        model = XGBClassifier()
    elif method_name == 'SVM':
        model = SVC(gamma='scale')
    elif method_name == 'NN':
        model = MLPClassifier(hidden_layer_sizes=(100, 50, 20), alpha=0.01, max_iter=400)
    elif method_name == 'Ada':
        model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=100, algorithm="SAMME.R", learning_rate=0.5)
    elif method_name == 'RF':
        model = RandomForestClassifier(n_estimators=200, max_depth=20)
    else:
        print('Wrong method name!')
    return model

