#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS231N 2023-2024: Final Project
late_fusion_CNN.py: Run the naive late fusion CNN model
Yichen Jiang <ycjiang@stanford.edu>
Senyang Jiang <senyangj@stanford.edu>
Suxi Li <suxi2024@stanford.edu>
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class NaiveCNN(nn.Module):
    """
    Simple Naive CNN model to extract features from given frame
    """
    def __init__(self, height=224, width=224):
        super(NaiveCNN, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.conv1_bn = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4_bn = nn.BatchNorm2d(128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(128 * (height//64) * (width//64), 512)
        self.fc1_bn = nn.BatchNorm1d(512)

    def forward(self, x):
        B = x.size(0)
        x = self.pool(F.relu(self.conv1_bn(self.conv1(x))))
        x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
        x = self.pool(F.relu(self.conv3_bn(self.conv3(x))))
        x = self.pool(F.relu(self.conv4_bn(self.conv4(x))))
        x = x.reshape(B, -1)
        x = self.dropout(F.relu(self.fc1_bn(self.fc1(x))))

        return x


class LateFusionModel(nn.Module):
    """
    Late Fusion Model to combine the cnn features of each frame together
    to predict a single emotion class label
    """
    def __init__(self, n_frames, cnn_features=512, num_classes=8):
        super(LateFusionModel, self).__init__()
        self.cnn = NaiveCNN(height=224, width=224)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(n_frames * cnn_features, 1024)
        self.fc1_bn = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.size()   # batch_size, num_frames, num_channels, H, W

        frame_features = []
        for i in range(T):
            frame = x[:, i, ...]
            cnn_feature = self.cnn(frame)
            frame_features.append(cnn_feature)

        cnn_features = torch.cat(frame_features, dim=1)
        cnn_features = cnn_features.view(B, -1)

        y = self.dropout(F.relu(self.fc1_bn(self.fc1(cnn_features))))
        out = self.fc2(y)

        return out