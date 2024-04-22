import torch
import torch.nn as nn


class DNNFeatureExtractor(nn.Module):
    def __init__(self, num_features):
        super(DNNFeatureExtractor, self).__init__()
        self.fc1 = nn.Linear(num_features, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 8)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)

        return x


class DNNClassifier(nn.Module):
    def __init__(self, num_features, num_classes):
        super(DNNClassifier, self).__init__()
        self.fc1 = nn.Linear(num_features, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
