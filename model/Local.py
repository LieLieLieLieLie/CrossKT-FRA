from CrossKTFRA.utils import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
import time


class Local:
    def __init__(self, device, dir, data_type, epoch) -> None:
        self.train_size = 0.8  # 训练集比例
        self.batch_size = 32  # batch size
        self.random_state = 0
        self.N = epoch  # 迭代次数
        self.device = device  # GPU
        self.dir = dir  # 实验保存路径
        self.data_type = data_type

    def Base_run(self, X, y, num_classes, representation_features, name):
        _, self.classifier, self.criterion, _ = data_method_select(self.data_type, num_classes, X.shape[1], representation_features)  # 选择模型
        self.optimizer = optim.Adam(list(self.classifier.parameters()), lr=0.001)
        self.classifier, self.criterion = self.classifier.to(self.device), self.criterion.to(self.device)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
        self.local_acc = []  # 准确率记录
        # 训练过程
        # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=self.train_size, random_state=self.random_state)
        X_train, X_test, y_train, y_test = X, X, y, y
        train_dataset = TensorDataset(torch.from_numpy(X_train).to(self.device), torch.from_numpy(y_train).to(self.device))

        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        test_dataset = TensorDataset(torch.from_numpy(X_test).to(self.device), torch.from_numpy(y_test).to(self.device))
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        start_time = time.time()  # 记录开始时间
        for epoch in range(self.N):
            for batch_X, batch_y in train_dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                # features = self.extractor(batch_X.unsqueeze(1).permute(0, 2, 1).float())
                outputs = self.classifier(batch_X.float())  # 分类
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
                    # features_test = self.extractor(batch_X_test.unsqueeze(1).permute(0, 2, 1).float())
                    outputs_test = self.classifier(batch_X_test.float())
                    # _, target_test = torch.max(batch_y_test, 1)
                    loss_test = self.criterion(outputs_test, torch.squeeze(batch_y_test.long(), dim=1))
                    total_loss += loss_test.item()

                    all_predictions.extend(torch.argmax(outputs_test, dim=1).cpu().numpy())
                    all_targets.extend(torch.squeeze(batch_y_test.long(), dim=1).cpu().numpy())

                average_loss = total_loss / len(test_dataloader)
                test_accuracy = accuracy_score(all_targets, all_predictions)

                # print(f'Epoch {epoch + 1}/{self.N}, Test Loss: {average_loss}, Test Accuracy: {test_accuracy * 100:.2f}%')
                self.local_acc.append(test_accuracy)  # 准确率记录

        end_time = time.time()  # 记录结束时间
        execution_time = end_time - start_time  # 计算程序执行时间
        if name == "Local":
            print(f"Base Local: {execution_time}s, {max(self.local_acc) * 100}%")
        elif name == "CrossKTFRA":
            print(f"Base Local(CrossKT-FRA): {execution_time}s, {max(self.local_acc) * 100}%")

        np.savez(f"{self.dir}/Base_{name}.npz",
                 acc_list=self.local_acc,
                 acc=max(self.local_acc),
                 time=execution_time, )
        # 保存模型参数
        # torch.save(self.extractor.state_dict(), os.path.join(dir, 'extractor.pkl'))
        # torch.save(self.classifier.state_dict(), os.path.join(dir, 'classifier.pkl'))

    def Ablation_run(self, X, y, num_classes, representation_features, name):
        _, self.classifier, self.criterion, _ = data_method_select(self.data_type, num_classes, X.shape[1], representation_features)  # 选择模型
        self.optimizer = optim.Adam(list(self.classifier.parameters()), lr=0.001)
        self.classifier, self.criterion = self.classifier.to(self.device), self.criterion.to(self.device)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
        self.local_train_acc = []  # 训练集准确率记录
        self.local_test_acc = []  # 测试集准确率记录
        # 训练过程
        # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=self.train_size, random_state=self.random_state)
        X_train, X_test, y_train, y_test = X, X, y, y
        train_dataset = TensorDataset(torch.from_numpy(X_train).to(self.device), torch.from_numpy(y_train).to(self.device))

        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        test_dataset = TensorDataset(torch.from_numpy(X_test).to(self.device), torch.from_numpy(y_test).to(self.device))
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        start_time = time.time()  # 记录开始时间
        for epoch in range(self.N):
            train_all_predictions = []
            train_all_targets = []
            train_total_loss = 0
            for batch_X, batch_y in train_dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                # features = self.extractor(batch_X.unsqueeze(1).permute(0, 2, 1).float())
                outputs = self.classifier(batch_X.float())  # 分类
                # _, target = torch.max(batch_y, 1)
                loss_train = self.criterion(outputs, torch.squeeze(batch_y.long(), dim=1))  # 计算损失
                train_total_loss += loss_train.item()

                train_all_predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                train_all_targets.extend(torch.squeeze(batch_y.long(), dim=1).cpu().numpy())

                # 反向传播和优化
                self.optimizer.zero_grad()
                loss_train.backward()
                self.optimizer.step()

            train_accuracy = accuracy_score(train_all_targets, train_all_predictions)
            self.local_train_acc.append(train_accuracy)  # 准确率记录

            # 在每个epoch结束后进行测试
            with torch.no_grad():
                test_all_predictions = []
                test_all_targets = []
                test_total_loss = 0

                for batch_X_test, batch_y_test in test_dataloader:
                    batch_X_test, batch_y_test = batch_X_test.to(self.device), batch_y_test.to(self.device)
                    # features_test = self.extractor(batch_X_test.unsqueeze(1).permute(0, 2, 1).float())
                    outputs_test = self.classifier(batch_X_test.float())
                    # _, target_test = torch.max(batch_y_test, 1)
                    loss_test = self.criterion(outputs_test, torch.squeeze(batch_y_test.long(), dim=1))
                    test_total_loss += loss_test.item()

                    test_all_predictions.extend(torch.argmax(outputs_test, dim=1).cpu().numpy())
                    test_all_targets.extend(torch.squeeze(batch_y_test.long(), dim=1).cpu().numpy())

                average_loss = test_total_loss / len(test_dataloader)
                test_accuracy = accuracy_score(test_all_targets, test_all_predictions)

                # print(f'Epoch {epoch + 1}/{self.N}, Test Loss: {average_loss}, Test Accuracy: {test_accuracy * 100:.2f}%')
                self.local_test_acc.append(test_accuracy)  # 准确率记录

        end_time = time.time()  # 记录结束时间
        execution_time = end_time - start_time  # 计算程序执行时间
        print(f"CrossKT-FRA - {name}: {execution_time}s, 训练: {max(self.local_train_acc) * 100}%, 测试: {max(self.local_test_acc) * 100}%")

        np.savez(f"{self.dir}/CrossKTFRA.npz",
                 acc_train_list=self.local_train_acc,
                 acc_train=max(self.local_train_acc),
                 acc_list=self.local_test_acc,
                 acc=max(self.local_test_acc),
                 time=execution_time, )
        # 保存模型参数
        # torch.save(self.extractor.state_dict(), os.path.join(dir, 'extractor.pkl'))
        # torch.save(self.classifier.state_dict(), os.path.join(dir, 'classifier.pkl'))
