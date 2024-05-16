#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS231N 2023-2024: Final Project
late_fusion_CNN.py: Run the naive late fusion CNN model
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
from EmotionDataset import EmotionDataset
from EmotionFrameDataset import EmotionFrameDataset
#from PIL import Image


class NaiveCNN(nn.Module):
    """
    Simple Naive CNN model to extract features from given frame
    """
    def __init__(self):
        super(NaiveCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, padding_mode='zeros')
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, padding_mode='zeros')
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, padding_mode='zeros')
        #self.fc1 = nn.Linear(64 * 160 * 90, 1024)
        self.fc1 = nn.Linear(64 * 160 * 90, 512)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.reshape(-1, 64 * 160 * 90)
        x = F.relu(self.fc1(x))
        #x = nn.ReLU(self.fc2(x))

        return x


class LateFusionModel(nn.Module):
    """
    Late Fusion Model to combine the cnn features of each frame together
    to predict a single emotion class label
    """
    def __init__(self, cnn, n_frames, cnn_features=512, num_classes=8):
        super(LateFusionModel, self).__init__()
        self.cnn = cnn
        self.fc1 = nn.Linear(n_frames * cnn_features, 1024)
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

        y = F.relu(self.fc1(cnn_features))
        out = self.fc2(y)

        return out

if __name__ == "__main__":
    #num_classes = 8
    num_frames = 6

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'using device {device}')

    cnn_model = NaiveCNN()
    LF_model = LateFusionModel(cnn_model, num_frames).to(device)

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

    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(LF_model.parameters(), lr=0.001)

    num_epochs = 10
    print(f'Training for {num_epochs} epochs...')

    for epoch in range(num_epochs):
        with tqdm(train_dataloader, unit="batch") as tepoch:
            running_loss = 0.0
            for inputs, e_labels in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                inputs, e_labels = inputs.to(device), e_labels.to(device)
                optimizer.zero_grad()
                outputs = LF_model(inputs)
                loss = criterion(outputs, e_labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                tepoch.set_postfix(loss=loss.item())
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_dataloader)}')

    # Save trained weights of CNN and LF models
    torch.save(cnn_model.state_dict(), 'naive_cnn_weights.pth')
    torch.save(LF_model.state_dict(), 'late_fusion_weights.pth')

    print('Testing model...')
    LF_model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        with tqdm(test_dataloader, unit="batch") as tepoch:
            for inputs, e_labels in tepoch:
                outputs = LF_model(inputs)
                loss = criterion(outputs, e_labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += e_labels.size(0)
                correct += (predicted == e_labels).sum().item()

    print(f'Test Loss: {test_loss/len(test_dataloader)}, Accuracy: {100 * correct / total}%')
