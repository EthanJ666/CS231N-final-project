#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS231N 2023-2024: Final Project
3D_CNN.py: Run the 3D Convolutional Neural Network
Model structure adapted from our late_fusion_model
Yichen Jiang <ycjiang@stanford.edu>
Senyang Jiang <senyangj@stanford.edu>
Suxi Li <suxi2024@stanford.edu>
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Video3DCNN(nn.Module):
    def __init__(self, n_frames, height=224, width=224, num_classes=8):
        super(Video3DCNN, self).__init__()
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv1 = nn.Conv3d(3, 16, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1)
        self.conv1_bn = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1)
        self.conv2_bn = nn.BatchNorm3d(32)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1)
        self.conv3_bn = nn.BatchNorm3d(64)
        self.conv4 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1)
        self.conv4_bn = nn.BatchNorm3d(128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(128 * (n_frames) * (height // 64) * (width // 64), 1024)
        self.fc1_bn = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc2_bn = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        B = x.size(0)  # (B, T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4)   # (B, C, T, H, W)
        x = self.pool(F.relu(self.conv1_bn(self.conv1(x)))) # torch.Size([10, 16, 6, 180, 320])
        x = self.pool(F.relu(self.conv2_bn(self.conv2(x)))) # torch.Size([10, 32, 6, 45, 80])
        x = self.pool(F.relu(self.conv3_bn(self.conv3(x)))) 
        x = self.pool(F.relu(self.conv4_bn(self.conv4(x))))
        x = x.reshape(B, -1)
        x = self.dropout(F.relu(self.fc1_bn(self.fc1(x))))
        x = self.dropout(F.relu(self.fc2_bn(self.fc2(x))))
        out = self.fc3(x)

        return out