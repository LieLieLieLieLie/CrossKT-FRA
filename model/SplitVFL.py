from CrossKTFRA.utils import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
import time


class SplitVFL:
    def __init__(self, device, dir, data_type) -> None:
        self.train_size = 0.8  # 训练集比例
        self.batch_size = 32  # batch size
        self.random_state = 0
        self.N = 50  # 迭代次数
        self.gamma_1 = 0.25  # 蒸馏损失系数
        self.gamma_2 = 0.25  # 分类损失系数
        self.device = device  # GPU
        self.dir = dir  # 实验保存路径
        self.data_type = data_type

    # 保证加权权重之和为1
    def normalize_weights(self):
        total_weights = self.weight_task + self.weight_data
        self.weight_task.data /= total_weights
        self.weight_data.data /= total_weights

    def Base_train(self, X, y, num_classes, representation_features, name):
        self.weight_task = torch.nn.Parameter(torch.Tensor([0.5]).to(self.device), requires_grad=True)
        self.weight_data = torch.nn.Parameter(torch.Tensor([0.5]).to(self.device), requires_grad=True)
        self.normalize_weights()
        self.weight_optimizer = optim.Adam([self.weight_task, self.weight_data], lr=0.001)
        self.weight_task, self.weight_data = self.weight_task.to(self.device), self.weight_data.to(self.device)

        self.task_extractor, _, _, _ = data_method_select(self.data_type, num_classes, X[0].shape[1], representation_features)  # 选择模型
        self.task_optimizer = optim.Adam(list(self.task_extractor.parameters()), lr=0.001)
        self.task_extractor = self.task_extractor.to(self.device)

        self.data_extractor, self.data_classifier, self.criterion, self.data_optimizer = data_method_select(self.data_type, num_classes, X[1].shape[1], representation_features)  # 选择模型
        self.data_extractor, self.data_classifier, self.criterion = self.data_extractor.to(self.device), self.data_classifier.to(self.device), self.criterion.to(self.device)

        for state in self.task_optimizer.state.values():  # 任务方参数优化器
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        for state in self.data_optimizer.state.values():  # 数据方参数优化器
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
        self.normal_data_acc = []  # 准确率记录
        # 训练过程
        # 任务方的无标签数据
        X_train_task, X_test_task, _, _ = train_test_split(X[0], y[0], train_size=self.train_size, random_state=self.random_state)
        # 数据方的有标签数据
        X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(X[1], y[1], train_size=self.train_size, random_state=self.random_state)
        # 所有训练集进dataloader
        train_dataset = TensorDataset(torch.from_numpy(X_train_task).to(self.device),
                                      torch.from_numpy(X_train_data).to(self.device),
                                      torch.from_numpy(y_train_data).to(self.device),
                                      )
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        # 所有测试集进dataloader
        test_dataset = TensorDataset(torch.from_numpy(X_test_task),
                                     torch.from_numpy(X_test_data),
                                     torch.from_numpy(y_test_data),
                                     )
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        start_time = time.time()  # 记录开始时间
        for epoch in range(self.N):
            for batch_X_task, batch_X_data, batch_y_data in train_dataloader:  # 遍历任务方X，数据方X，y
                batch_X_task, batch_X_data, batch_y_data = batch_X_task.to(self.device), batch_X_data.to(self.device), batch_y_data.to(self.device)
                # representation_task = self.task_extractor(batch_X_task.unsqueeze(1).permute(0, 2, 1).float())  # 任务方表示
                # representation_data = self.data_extractor(batch_X_data.unsqueeze(1).permute(0, 2, 1).float())  # 数据方表示
                representation_task = self.task_extractor(batch_X_task.unsqueeze(1).permute(0, 2, 1).float())
                representation_data = self.data_extractor(batch_X_data.unsqueeze(1).permute(0, 2, 1).float())

                representation = torch.add(self.weight_task * representation_task, self.weight_data * representation_data)  # 合成分类
                outputs_data = self.data_classifier(representation)  # 数据方分类
                outputs = outputs_data
                # _, target = torch.max(batch_y_data, 1)
                loss = self.criterion(outputs, torch.squeeze(batch_y_data.long(), dim=1))  # 计算损失
                # 反向传播和优化
                self.weight_optimizer.zero_grad()
                self.task_optimizer.zero_grad()
                self.data_optimizer.zero_grad()
                loss.backward()
                # # 打印梯度
                # print("Gradient of weight_task:", self.weight_task.grad)
                # print("Gradient of weight_data:", self.weight_data.grad)
                self.weight_optimizer.step()
                self.normalize_weights()
                self.task_optimizer.step()
                self.data_optimizer.step()

            # 在每个epoch结束后进行测试
            with torch.no_grad():
                all_predictions = []
                all_targets = []
                total_loss = 0

                for batch_X_task_test, batch_X_data_test, batch_y_data_test in test_dataloader:
                    batch_X_task_test, batch_X_data_test, batch_y_data_test = \
                        batch_X_task_test.to(self.device), batch_X_data_test.to(self.device), batch_y_data_test.to(self.device)
                    # representation_task_test = self.data_extractor(batch_X_task_test.unsqueeze(1).permute(0, 2, 1).float())
                    # representation_data_test = self.data_extractor(batch_X_data_test.unsqueeze(1).permute(0, 2, 1).float())
                    representation_task_test = self.task_extractor(batch_X_task_test.unsqueeze(1).permute(0, 2, 1).float())
                    representation_data_test = self.data_extractor(batch_X_data_test.unsqueeze(1).permute(0, 2, 1).float())
                    representation_test = torch.add(self.weight_task * representation_task_test, self.weight_data * representation_data_test)  # 合成分类
                    outputs_data_test = self.data_classifier(representation_test)
                    outputs_test = outputs_data_test
                    # _, target_test = torch.max(batch_y_data_test, 1)
                    loss_test = self.criterion(outputs_test, torch.squeeze(batch_y_data_test.long(), dim=1))
                    total_loss += loss_test.item()

                    all_predictions.extend(torch.argmax(outputs_test, dim=1).cpu().numpy())
                    all_targets.extend(torch.squeeze(batch_y_data_test.long(), dim=1).cpu().numpy())

                average_loss = total_loss / len(test_dataloader)
                test_accuracy = accuracy_score(all_targets, all_predictions)

                # print(f'Epoch {epoch + 1}/{self.N}, Test Loss: {average_loss}, Test Accuracy: {test_accuracy * 100:.2f}%')
                self.normal_data_acc.append(test_accuracy)  # 准确率记录
                # print("data方对X_Fed训练监督模型的准确率：", self.normal_data_acc)

        end_time = time.time()  # 记录结束时间
        execution_time = end_time - start_time  # 计算程序执行时间
        print(f"{name}: {execution_time}s, {max(self.normal_data_acc) * 100}%")

        # Base_SplitVFL
        np.savez(f"{self.dir}/{name}.npz",
                 acc_list=self.normal_data_acc,
                 acc=max(self.normal_data_acc),
                 time=execution_time, )

        # # 保存模型参数
        # dir = 'model/save/SplitVFL/normal/' + self.dataset + '/' + self.split + '/'
        # dir = 'model/save/SplitVFL/normal/' + self.dataset + '/' + self.split + '/'
        # dir = 'model/save/SplitVFL/normal/' + self.dataset + '/' + self.split + '/'
