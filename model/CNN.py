import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class DistillLoss(nn.Module):
    def __init__(self, alpha):
        super(DistillLoss, self).__init__()
        self.alpha = alpha

    def forward(self, outputs, labels, features, h_z):
        # 交叉熵损失
        ce_loss = F.cross_entropy(outputs, labels)

        # 重构损失
        # recons_loss = nn.KLDivLoss(F.log_softmax(generated_data, dim=1), F.softmax(X, dim=1))

        # 余弦相似度损失
        cosine_similarity = F.cosine_similarity(features, h_z, dim=1)
        cosine_loss = 1 - cosine_similarity.mean()

        # 总损失
        total_loss = ce_loss + self.alpha * cosine_loss
        return total_loss


class CNNFeatureExtractor(nn.Module):
    def __init__(self, data_type, input_features, representation_features):
        super(CNNFeatureExtractor, self).__init__()
        if data_type == "csv":
            self.conv1 = nn.Conv1d(input_features, 32, kernel_size=3, stride=1, padding=1)
        elif data_type == "image":
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        if data_type == "csv":
            self.conv2 = nn.Conv1d(32, representation_features, kernel_size=3, stride=1, padding=1)
        elif data_type == "image":
            self.conv2 = nn.Conv2d(32, representation_features, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        if data_type == "csv":
            self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        elif data_type == "image":
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool(x)
        return x


class CNNClassifier(nn.Module):
    def __init__(self, representation_features, num_classes):
        super(CNNClassifier, self).__init__()
        self.fc1 = nn.Linear(representation_features, 512)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.representation_features = representation_features

    def forward(self, x):
        x = x.view(-1, self.representation_features)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


# # Hyperparameters
# num_classes = 10  # 分类的类别数
#
# # 创建卷积特征提取器和分类器
# cnn_feature_extractor = CNNFeatureExtractor()
# cnn_classifier = CNNClassifier(num_classes)
#
# # 定义损失函数和优化器
# criterion = CustomLoss(alpha)
# optimizer = optim.Adam(list(cnn_feature_extractor.parameters()) + list(cnn_classifier.parameters()), lr=0.001)
#
# # 训练过程
# for epoch in range(num_epochs):
#     # 输入数据 X，假设 X 的形状是 (batch_size, channels, height, width)
#
#     # 特征提取
#     features = cnn_feature_extractor(X)
#
#     # 分类
#     outputs = cnn_classifier(features)
#
#     # 计算损失
#     loss = criterion(outputs, labels, features, h_z)
#
#     # 反向传播和优化
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
# # 保存模型参数
# torch.save(cnn_feature_extractor.state_dict(), 'models/save/data_extractor.pkl')
# torch.save(cnn_classifier.state_dict(), 'models/save/data_classifier.pkl')

# loaded_feature_extractor.load_state_dict(torch.load('feature_extractor.pkl'))
# loaded_classifier.load_state_dict(torch.load('classifier.pkl'))
# 在测试集上进行测试
# 输入测试数据 test_X，获取特征 features
# 使用分类器进行预测
# 得到预测结果 predictions
