import numpy as np
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from model.Downstrams import *
import os
from configs.args import *  # 选择配置（数据集，参数，gpu/cpu等等）
from model.AggVFL import *
from model.SplitVFL import *
from model.Local import *


class Performance:
    def __init__(self, device, dir) -> None:
        self.train_size = 0.8  # 训练集比例
        self.batch_size = 32  # batch size
        self.random_state = 101
        self.N = 30  # 迭代次数
        # self.gamma_1 = 0.25  # 对抗损失系数
        # self.gamma_2 = 0.25  # 重构损失系数
        # self.gamma_3 = 0.25  # 蒸馏损失系数
        # self.gamma_4 = 0.25  # 分类损失系数
        self.device = device  # GPU
        self.dir = dir  # 保存路径

    # data方本地先训练联邦表示得到特征提取器给task方
    def data_run(self, X, y, num_classes, data_type='csv'):
        self.data_extractor, self.data_classifier, self.criterion, self.optimizer = data_method_select(data_type, num_classes, X.shape[1], X.shape[1])  # 选择模型
        self.data_extractor, self.data_classifier, self.criterion = self.data_extractor.to(self.device), self.data_classifier.to(self.device), self.criterion.to(self.device)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
        self.data_acc = []  # 准确率记录
        # 训练过程
        # print("data方正在本地对X_Fed训练监督模型")
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=self.train_size)
        train_dataset = TensorDataset(torch.from_numpy(X_train).to(self.device), torch.from_numpy(y_train).to(self.device))

        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        for epoch in range(self.N):
            for batch_X, batch_y in train_dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                features = self.data_extractor(batch_X.unsqueeze(1).permute(0, 2, 1).float())
                outputs = self.data_classifier(features)  # 分类
                _, target = torch.max(batch_y, 1)
                loss = self.criterion(outputs, torch.squeeze(batch_y.long(), dim=1))  # 计算损失
                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # 在每个epoch结束后进行测试
            with torch.no_grad():
                all_predictions = []
                all_targets = []
                total_loss = 0

                for batch_X_test, batch_y_test in test_dataloader:
                    batch_X_test, batch_y_test = batch_X_test.to(self.device), batch_y_test.to(self.device)
                    features_test = self.data_extractor(batch_X_test.unsqueeze(1).permute(0, 2, 1).float())
                    outputs_test = self.data_classifier(features_test)
                    # _, target_test = torch.max(batch_y_test, 1)
                    loss_test = self.criterion(outputs_test, torch.squeeze(batch_y_test.long(), dim=1))
                    total_loss += loss_test.item()

                    all_predictions.extend(torch.argmax(outputs_test, dim=1).cpu().numpy())
                    all_targets.extend(torch.squeeze(batch_y_test.long(), dim=1).cpu().numpy())

                average_loss = total_loss / len(test_dataloader)
                test_accuracy = accuracy_score(all_targets, all_predictions)

                # print(f'Epoch {epoch + 1}/{self.N}, Test Loss: {average_loss}, Test Accuracy: {test_accuracy * 100:.2f}%')
                self.data_acc.append(test_accuracy)  # 准确率记录
                # print("data方对X_Fed训练监督模型的准确率：", self.data_acc)

        # 保存模型参数
        torch.save(self.data_extractor.state_dict(), os.path.join(self.dir, 'data_extractor.pkl'))
        torch.save(self.data_classifier.state_dict(), os.path.join(self.dir, 'data_classifier.pkl'))

        # # 提取data表示
        # X_tensor = torch.from_numpy(X)  # 将X转换为 PyTorch Tensor
        # X_tensor = X_tensor.unsqueeze(1).permute(0, 2, 1).float()  # 调整维度，以符合模型的输入要求（示例中假设模型期望的输入是 (batch_size, channels, sequence_length)）
        # with torch.no_grad():
        #     h_fed = self.data_extractor(X_tensor)
        # return h_fed

    # 下游任务
    def Downstream_local_run(self, X, y, status, method_name='NN'):  # status：1.Local：原始task数据本地训练；2.CrossKTFRA：task增强的表示本地的训练
        # 下游任务实验保存路径
        setting = load_dataset_config(args.dataset, f'{args.split}_split')

        dir = r'model/save/Downstream/Local/' + args.dataset + '/' + args.downstream + '/' + \
              str(setting['task_num_feature']) + '_' + str(setting['shared_feature']) + '_' + \
              str(setting['task_num_sample']) + '_' + str(setting['shared_sample']) +  '/'  # 保存路径
        os.makedirs(dir) if not os.path.exists(dir) else None
        print("下游任务—" + status + "-" + method_name + "-实验保存路径：" + dir)

        self.method = downstream_method_select(method_name)  # 选择模型
        res = []  # 准确率记录
        for _ in range(self.N):
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=self.train_size)
            self.method.fit(X_train, y_train)
            res.append(self.method.score(X_test, y_test))  # 准确率记录
        print("下游任务—" + status + "-" + method_name + "-Acc:", res)  # 准确率记录
        print("下游任务—" + status + "-" + method_name + "-最大Acc:", max(res))  # 准确率最大值
        np.savez(f"{dir}/{status}.npz",
                 acc_list=res,
                 acc=max(res), )

    def Downstream_VFL_run(self, X_task_shared, X_data_shared, y_shared, num_classes, X_fed, X_task_new):
        device = get_gpu(args.gpu)  # 获取命令行的GPU设备
        setting = load_dataset_config(args.dataset, f'{args.split}_split')

        # 1.AggVFL
        dir_AggVFL = r'model/save/Downstream/VFL/' + args.dataset + '/' + \
                     str(setting['shared_feature']) + '_' + str(setting['shared_sample']) + '/AggVFL/'  # 保存路径
        os.makedirs(dir_AggVFL) if not os.path.exists(dir_AggVFL) else None
        print("下游任务的VFL的AggVFL实验保存路径：" + dir_AggVFL)
        aggVFL = AggVFL(device, dir_AggVFL, args.data_type)
        aggVFL.Base_train([X_task_shared, X_data_shared], [y_shared, y_shared], num_classes, X_fed.shape[1], "VFL_AggVFL")
        aggVFL.Base_train([X_task_new[setting['task_num_sample']-setting['shared_sample']:setting['task_num_sample'],:], X_data_shared], [y_shared, y_shared], num_classes, X_fed.shape[1], "VFL_AggVFL_ours")
        # 2.SplitVFL
        dir_SplitVFL = r'model/save/Downstream/VFL/' + args.dataset + '/' + \
                     str(setting['shared_feature']) + '_' + str(setting['shared_sample']) + '/SplitVFL/'  # 保存路径
        os.makedirs(dir_SplitVFL) if not os.path.exists(dir_SplitVFL) else None
        print("下游任务的VFL的SplitVFL实验保存路径：" + dir_SplitVFL)
        splitVFL = SplitVFL(device, dir_SplitVFL, args.data_type)
        splitVFL.Base_train([X_task_shared, X_data_shared], [y_shared, y_shared], num_classes, X_fed.shape[1], "VFL_SplitVFL")
        splitVFL.Base_train([X_task_new[setting['task_num_sample']-setting['shared_sample']:setting['task_num_sample'],:], X_data_shared], [y_shared, y_shared], num_classes, X_fed.shape[1], "VFL_SplitVFL_ours")

    def Base_run(self, X_task_shared, X_data_shared, y_shared, num_classes, X_fed, X_task_new, y_task):
        device = get_gpu(args.gpu)  # 获取命令行的GPU设备
        # 1.AggVFL
        dir_AggVFL = r'model/save/Base/' + args.experiment2 + '/' + args.experiment3 + '/' + args.dataset + '/' + str(
            args.parameter) + '/AggVFL/'  # 保存路径
        os.makedirs(dir_AggVFL) if not os.path.exists(dir_AggVFL) else None
        print("AggVFL实验保存路径：" + dir_AggVFL)
        aggVFL = AggVFL(device, dir_AggVFL, args.data_type)
        aggVFL.Base_train([X_task_shared, X_data_shared], [y_shared, y_shared], num_classes, X_fed.shape[1], "Base_AggVFL")
        # 2.SplitVFL
        dir_SplitVFL = r'model/save/Base/' + args.experiment2 + '/' + args.experiment3 + '/' + args.dataset + '/' + str(
            args.parameter) + '/SplitVFL/'  # 保存路径
        os.makedirs(dir_SplitVFL) if not os.path.exists(dir_SplitVFL) else None
        print("SplitVFL实验保存路径：" + dir_SplitVFL)
        splitVFL = SplitVFL(device, dir_SplitVFL, args.data_type)
        splitVFL.Base_train([X_task_shared, X_data_shared], [y_shared, y_shared], num_classes, X_fed.shape[1], "Base_SplitVFL")
        # 3.Local
        dir_Local = r'model/save/Base/' + args.experiment2 + '/' + args.experiment3 + '/' + args.dataset + '/' + str(
            args.parameter) + '/Local/'  # 保存路径
        os.makedirs(dir_Local) if not os.path.exists(dir_Local) else None
        print("Local实验保存路径：" + dir_Local)
        local = Local(device, dir_Local, args.data_type, 50)
        local.Base_run(X_task_shared, y_shared, num_classes, X_task_shared.shape[1], "Local")
        # 4.CrossKT-FRA
        dir_CrossKTFRA = r'model/save/Base/' + args.experiment2 + '/' + args.experiment3 + '/' + args.dataset + '/' + str(
            args.parameter) + '/CrossKTFRA/'  # 保存路径
        os.makedirs(dir_CrossKTFRA) if not os.path.exists(dir_CrossKTFRA) else None
        print("CrossKTFRA实验保存路径：" + dir_CrossKTFRA)
        CrossKTFRA = Local(device, dir_CrossKTFRA, args.data_type, 50)
        # CrossKTFRA.Base_run(X_task_new[:X_task_shared.shape[0]], y_shared, num_classes, X_fed.shape[1], "CrossKTFRA")
        CrossKTFRA.Base_run(X_task_new, y_task, num_classes, X_fed.shape[1], "CrossKTFRA")

    def Ablation_run(self, X_task_shared, X_data_shared, y_shared, num_classes, X_fed, X_task_new, y_task):
        device = get_gpu(args.gpu)  # 获取命令行的GPU设备
        dir_CrossKTFRA = r'model/save/Ablation/' + args.experiment2 + '/' + args.experiment3 + '/' + args.dataset + '/' + str(
            args.parameter) + '/CrossKTFRA_' + args.ablation + '/'  # 保存路径
        os.makedirs(dir_CrossKTFRA) if not os.path.exists(dir_CrossKTFRA) else None
        CrossKTFRA = Local(device, dir_CrossKTFRA, args.data_type, 50)
        CrossKTFRA.Ablation_run(X_task_new, y_task, num_classes, X_fed.shape[1], args.ablation)