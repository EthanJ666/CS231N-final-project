#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS231N 2023-2024: Final Project
GRU_RCN.py: Run the Recurrent Convolutional Network (RCNN)
Yichen Jiang <ycjiang@stanford.edu>
Senyang Jiang <senyangj@stanford.edu>
Suxi Li <suxi2024@stanford.edu>
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size):
        super(ConvGRUCell, self).__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.ConvGates   = nn.Conv2d(self.input_size+self.hidden_size, 2*self.hidden_size, kernel_size, padding='same')
        self.Conv_ct     = nn.Conv2d(self.input_size+self.hidden_size, self.hidden_size, kernel_size, padding='same') 
    
    def forward(self, input, hidden):
        if hidden is None:
            size_h = [input.size(0), self.hidden_size] + list(input.size()[2:])
            hidden = torch.zeros(size_h).to(input.device)
        c1           = self.ConvGates(torch.cat((input, hidden), dim=1))
        (rt,ut)      = c1.chunk(chunks=2, dim=1)
        reset_gate   = F.sigmoid(rt)
        update_gate  = F.sigmoid(ut)
        gated_hidden = torch.mul(reset_gate, hidden)
        p1           = self.Conv_ct(torch.cat((input, gated_hidden), dim=1))
        ct           = F.tanh(p1)
        next_h       = torch.mul(update_gate,hidden) + (1-update_gate)*ct
        return next_h

class RCN(nn.Module):
    def __init__(self, height=224, width=224, num_classes=8, n_frames=6):
        super(RCN, self).__init__()
        # CNN
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.conv1_bn = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3_bn = nn.BatchNorm2d(64)

        # RNN model taking conv maps from L=3 layers
        self.rnn1 = ConvGRUCell(input_size=16, hidden_size=16, kernel_size=3)
        self.rnn2 = ConvGRUCell(input_size=32, hidden_size=32, kernel_size=3)
        self.rnn3 = ConvGRUCell(input_size=64, hidden_size=64, kernel_size=3)
        self.avgpool1 = nn.AvgPool2d(kernel_size=height//4)
        self.avgpool2 = nn.AvgPool2d(kernel_size=height//16)
        self.avgpool3 = nn.AvgPool2d(kernel_size=height//64)

        self.fc1 = nn.Linear(16+32+64, num_classes)

    def forward(self, x):
        N, T, C, H, W = x.size()
        x = x.view(N * T, C, H, W)
        x1 = self.maxpool(F.relu(self.conv1_bn(self.conv1(x))))
        x2 = self.maxpool(F.relu(self.conv2_bn(self.conv2(x1))))
        x3 = self.maxpool(F.relu(self.conv3_bn(self.conv3(x2))))

        x1 = x1.reshape(N, T, 16, H//4, W//4)
        x2 = x2.reshape(N, T, 32, H//16, W//16)
        x3 = x3.reshape(N, T, 64, H//64, W//64)

        h1, h2, h3 = None, None, None
        for t in range(T):
            x1t = x1[:, t, :, :, :]
            x2t = x2[:, t, :, :, :]
            x3t = x3[:, t, :, :, :]
            h1 = self.rnn1(x1t, h1)
            h2 = self.rnn2(x2t, h2)
            h3 = self.rnn3(x3t, h3)

        h1 = self.avgpool1(h1).squeeze((2,3))
        h2 = self.avgpool2(h2).squeeze((2,3))
        h3 = self.avgpool3(h3).squeeze((2,3))

        h = torch.cat((h1, h2, h3), dim=-1)
        out = self.fc1(h) # (N, num_classes)
        
        return out
