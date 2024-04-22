import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)  # 显示完整的列
pd.set_option('display.max_rows', None)  # 显示完整的行

# 下游任务
class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, device):
        super(MLPModel, self).__init__()
        layers = []
        sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            layers.append(nn.ReLU())
        layers.pop()  # 移除最后一个ReLU层
        self.model = nn.Sequential(*layers)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        self.train_accuracies = []  # 用于记录每轮训练后的准确率
        self.device = device

    def forward(self, x, labels=None):
        outputs = self.model(x)
        if labels is not None:
            print(outputs.shape)
            print(labels.shape)
            print(type(outputs))
            print(type(labels))
            loss = self.criterion(outputs, labels)
            return outputs, loss
        return outputs

    def train_model(self, train_loader, num_epochs=400):
        for epoch in range(num_epochs):
            total_correct = 0
            total_samples = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)  # 移动数据到设备
                self.optimizer.zero_grad()
                outputs, loss = self.forward(inputs, labels)
                loss.backward()
                self.optimizer.step()

                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

            accuracy = total_correct / total_samples
            self.train_accuracies.append(accuracy)
            print(f'Epoch [{epoch+1}/{num_epochs}], Accuracy: {accuracy:.4f}')

    def evaluate_model(self, test_data, test_labels):
        with torch.no_grad():
            self.eval()
            test_data, test_labels = test_data.to(self.device), test_labels.to(self.device)  # 移动数据到设备
            outputs = self.forward(test_data)
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == test_labels).sum().item() / test_labels.size(0)

        return accuracy
