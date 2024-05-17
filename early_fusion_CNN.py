#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS231N 2023-2024: Final Project
early_fusion_CNN.py: Run the naive early fusion CNN model
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


class EarlyFusionCNN(nn.Module):
    def __init__(self, n_frames, height=1280, width=720, num_classes=8):
        super(EarlyFusionCNN, self).__init__()
        #self.T = n_frames
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(3 * n_frames, 64, kernel_size=3, stride=2, padding=1, padding_mode='zeros')
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, padding_mode='zeros')
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, padding_mode='zeros')
        self.conv3_bn = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1, padding_mode='zeros')
        self.conv4_bn = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(512 * (height//64) * (width//64), 512)
        self.fc1_bn = nn.BatchNorm1d(512)
        #self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 1024)
        self.fc2_bn = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.size()
        x = x.reshape(B, T*C, H, W)

        x = self.pool(F.relu(self.conv1_bn(self.conv1(x))))
        x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
        x = self.pool(F.relu(self.conv3_bn(self.conv3(x))))
        x = self.pool(F.relu(self.conv4_bn(self.conv4(x))))
        x = x.view(B, -1)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        #x = self.dropout(x)
        x = F.relu(self.fc2_bn(self.fc2(x)))
        # x = self.dropout(x)
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    #num_classes = 8
    num_frames = 6

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'using device {device}')

    writer = SummaryWriter()

    EF_model = EarlyFusionCNN(num_frames).to(device)
    
    pytorch_total_params = sum(p.numel() for p in EF_model.parameters() if p.requires_grad)
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
    optimizer = optim.Adam(EF_model.parameters(), lr=0.001)

    num_epochs = 25
    print(f'Training for {num_epochs} epochs...')

    for epoch in range(num_epochs):
        with tqdm(train_dataloader, unit="batch") as tepoch:
            running_loss = 0.0
            for inputs, e_labels in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                inputs, e_labels = inputs.to(device), e_labels.to(device)
                optimizer.zero_grad()
                outputs = EF_model(inputs)
                loss = criterion(outputs, e_labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                tepoch.set_postfix(loss=loss.item())
            avg_running_loss = running_loss/len(train_dataloader)
            writer.add_scalar("Loss/train", avg_running_loss, epoch+1)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_dataloader)}')

    # Save trained weights of EarlyFusion model
    torch.save(EF_model.state_dict(), 'early_fusion_weights.pth')

    print('Testing model...')
    EF_model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        with tqdm(test_dataloader, unit="batch") as tepoch:
            for inputs, e_labels in tepoch:
                inputs, e_labels = inputs.to(device), e_labels.to(device)
                outputs = EF_model(inputs)
                loss = criterion(outputs, e_labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += e_labels.size(0)
                correct += (predicted == e_labels).sum().item()

    print(f'Test Loss: {test_loss/len(test_dataloader)}, Accuracy: {100 * correct / total}%')
    writer.add_scalar("Acc/test", 100 * correct / total, 0)
    writer.flush()
    writer.close()
