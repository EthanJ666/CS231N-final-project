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

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from EmotionDataset import EmotionDataset
from EmotionFrameDataset import EmotionFrameDataset
#from PIL import Image


class ConvRNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvRNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.rnn = nn.LSTM(input_size=out_channels, hidden_size=out_channels, batch_first=True)

    def forward(self, x, h_prev=None):
        # x shape: (N, T, C, H, W)
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


if __name__ == "__main__":
    #num_classes = 8
    num_frames = 6

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'using device {device}')

    writer = SummaryWriter()

    model = RCNN().to(device)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'total params: {pytorch_total_params}')
    
    #######################################################
    #################### Edit Dataset #####################
    #######################################################
    root_dir = './dataset'
    test_root_dir = './dataset_test'
    frames_root_dir = './dataset_images'

    emotion_labels = {
        '01-neutral': 0,
        '02-calm': 1,
        '03-happy': 2,
        '04-sad': 3,
        '05-angry': 4,
        '06-fearful': 5,
        '07-disgust': 6,
        '08-surprised': 7
    }

    dataset = EmotionFrameDataset(frames_root_dir, device)
    #######################################################

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 25
    print(f'Training for {num_epochs} epochs...')

    for epoch in range(num_epochs):
        with tqdm(train_dataloader, unit="batch") as tepoch:
            running_loss = 0.0
            for inputs, e_labels in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                inputs, e_labels = inputs.to(device), e_labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, e_labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                tepoch.set_postfix(loss=loss.item())
            avg_running_loss = running_loss/len(train_dataloader)
            writer.add_scalar("Loss/train", avg_running_loss, epoch+1)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_running_loss}')

    # Save trained weights of CNN and LF models
    #torch.save(cnn_model.state_dict(), 'video_cnn_weights.pth')
    torch.save(model.state_dict(), 'CNN_3D_weights.pth')

    print('Testing model...')
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        with tqdm(test_dataloader, unit="batch") as tepoch:
            for inputs, e_labels in tepoch:
                inputs, e_labels = inputs.to(device), e_labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, e_labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += e_labels.size(0)
                correct += (predicted == e_labels).sum().item()

    print(f'Test Loss: {test_loss/len(test_dataloader)}, Accuracy: {100 * correct / total}%')
    writer.add_scalar("Acc/test", 100 * correct / total, 0)
    writer.flush()
    writer.close()
