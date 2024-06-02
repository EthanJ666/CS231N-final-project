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
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.rnn = nn.LSTM(input_size=out_channels, hidden_size=out_channels, batch_first=True)

    def forward(self, x, h_prev=None):
        N, T, C, H, W = x.size()
        x = x.view(N * T, C, H, W)
        x = self.conv(x)  # (N*T, out_channels, H, W)
        x = x.view(N, T, -1, H, W)  # (N, T, out_channels, H, W)
        x = x.permute(0, 3, 4, 1, 2)  # (N, H, W, T, out_channels)
        x = x.reshape(-1, T, x.size(-1))  # (N*H*W, T, out_channels)

        if h_prev is None:
            x, h = self.rnn(x)
        else:
            x, h = self.rnn(x, h_prev)

        x = x.reshape(N, H, W, T, -1).permute(0, 3, 4, 1, 2)  # (N, T, out_channels, H, W)
        return x


class RCNN(nn.Module):
    def __init__(self, height=720, width=1280, num_classes=8, n_frames=6):
        super(RCNN, self).__init__()
        self.layer1 = RCNNBlock(3, 64)
        self.layer2 = RCNNBlock(64, 128)
        self.layer3 = RCNNBlock(128, 256)

        self.fc = nn.Linear(256 * height * width, num_classes)

    def forward(self, x):
        h1, h2, h3 = None, None, None

        outputs = []
        for t in range(x.size(1)):
            xt = x[:, t, :, :, :].unsqueeze(1)
            out1, h1 = self.layer1(xt, h1)
            out2, h2 = self.layer2(out1, h2)
            out3, h3 = self.layer3(out2, h3)
            outputs.append(out3)

        x = torch.stack(outputs, dim=1)  # (N, T, C, H, W)
        N, T, C, H, W = x.size()
        x = x.view(N, T, -1)  # (N, T, 256*H*W)
        x = x.mean(dim=1)  # (N, 256*H*W)

        x = self.fc(x)  # (N, num_classes)
        return x
