#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS231N 2023-2024: Final Project
RNN_CNN.py: Run the Recurrent Convolutional Network (RCNN)
Yichen Jiang <ycjiang@stanford.edu>
Senyang Jiang <senyangj@stanford.edu>
Suxi Li <suxi2024@stanford.edu>
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvRNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvRNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)  # Add batch normalization
        self.rnn = nn.LSTMCell(input_size=out_channels, hidden_size=out_channels)

    def forward(self, x, hc_prev=None):
        # x shape: (N, C, H, W)
        N, C, H, W = x.size()
        x = self.conv(x)  # (N*T, out_channels, H/2, W/2)
        x = self.bn(x)  # Apply batch normalization
        x = F.relu(x)  # Add ReLU activation

        new_H, new_W = x.size(2), x.size(3)  # H/2, W/2
        x = x.view(N, -1, new_H, new_W)  # Shape: (N, out_channels, H/2, W/2)

        x = x.permute(0, 2, 3, 1)  # rearrange dimensions to prepare for RNN processing. (N, H/2, W/2, out_channels)
        # to structure the data such that each spatial location (pixel) across all frames is treated as a sequence

        x = x.contiguous().view(N * new_H * new_W, -1)  # (N*H/2*W/2, out_channels)
        # The resulting tensor has N * H/2 * W/2 sequences, with out_channels features per time step.

        if hc_prev is None:
            hc = self.rnn(x)
        else:
            hc = self.rnn(x, hc_prev)

        x = hc[0]
        x = x.reshape(N, new_H, new_W, -1).permute(0, 3, 1, 2)  # Shape: (N, out_channels, H/2, W/2)
        return x, hc


class RCNN(nn.Module):
    def __init__(self, height=224, width=224, num_classes=8):
        super(RCNN, self).__init__()
        self.layer1 = ConvRNNBlock(3, 16)
        self.layer2 = ConvRNNBlock(16, 32)
        self.layer3 = ConvRNNBlock(32, 64)
        self.layer4 = ConvRNNBlock(64, 128)

        flattened_size = 128 * (height // 16) * (width // 16)  # Calculate flattened size after convolutional layers
        self.fc1 = nn.Linear(flattened_size, 1024)  # Add an additional fully connected layer
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)


    def forward(self, x):
        hc1, hc2, hc3, hc4 = None, None, None, None

        outputs = []
        for t in range(x.size(1)):
            xt = x[:, t, :, :, :]  # Shape: (N, C, H, W)
            # Each ConvRNNBlock should handle its own hidden state separately,
            # capturing the temporal dynamics at its level of processing
            out1, hc1 = self.layer1(xt, hc1)  # Shape after layer1: (N, 16, H/2, W/2)
            out2, hc2 = self.layer2(out1, hc2)  # Shape after layer2: (N, 32, H/4, W/4)
            out3, hc3 = self.layer3(out2, hc3)  # Shape after layer3: (N, 64, H/8, W/8)
            out4, hc4 = self.layer4(out3, hc4)  # Shape after layer3: (N, 128, H/16, W/16)
            outputs.append(out4)  # Remove the temporal dimension for concatenation

        x = torch.stack(outputs, dim=1)  # (N, T, C, H, W)
        N, T, C, H2, W2 = x.size()
        x = x.reshape(N, T, -1)  # (N, T, 128*H/16*W/16)
        x = x.mean(dim=1)  # (N, T, 128*H/16*W/16)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))  # Add another fully connected layer with ReLU activation
        x = self.dropout(x)
        x = self.fc3(x)  # Final fully connected layer to produce class scores
        return x
