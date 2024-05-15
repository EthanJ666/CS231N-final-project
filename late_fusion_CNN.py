#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS231N 2023-2024: Final Project
late_fusion_CNN.py: Run the naive late fusion CNN model
Yichen Jiang <ycjiang@stanford.edu>
Senyang Jiang <senyangj@stanford.edu>
Suxi Li <suxi2024@stanford.edu>
"""

import cv2
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image


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
        x = x.view(-1, 64 * 160 * 90)
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


class EmotionDataset(Dataset):
    """
    Dataset containing all video clips and corresponding emotion labels
    """
    def __init__(self, root_dir, emotion_labels, n_frames, transform=None, ):
        self.root_dir = root_dir
        self.emotion_labels = emotion_labels
        #self.transform = transform
        self.n_frames = n_frames
        self.video_paths = []
        self.labels = []

        # Load all video paths and their corresponding labels
        for folder_name in os.listdir(root_dir):
            if folder_name in emotion_labels:
                label = emotion_labels[folder_name]
                folder_path = os.path.join(root_dir, folder_name)
                for video_name in os.listdir(folder_path):
                    if video_name.endswith('.mp4'):
                        self.video_paths.append(os.path.join(folder_path, video_name))
                        self.labels.append(label)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        frames = self.extract_frames(video_path, self.n_frames)
        frames = torch.stack(frames)  # shape(T, C, H, W)
        return frames, label

    def extract_frames(self, video_path, n_frames, tf=None):
        """
        Args:
            video_path: video path
            n_frames: how many frames we want to extract in total (T)
            tf: whether to use transformer to lower the size of the frames

        Returns: a list of frames, 'n' frames in total
        """
        frames = []
        vidcap = cv2.VideoCapture(video_path)
        total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_indices = np.linspace(0, total_frames - 2, n_frames, dtype=int)

        for i in frames_indices:
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = vidcap.read()
            frame = torch.tensor(frame, dtype=torch.float32)
            frame = frame.permute(2, 0, 1)

            if ret:
                frames.append(frame)

        vidcap.release()

        return frames


#num_classes = 8
num_frames = 5
cnn_model = NaiveCNN()
LF_model = LateFusionModel(cnn_model, num_frames)

root_dir = './dataset'

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

dataset = EmotionDataset(root_dir, emotion_labels, num_frames)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(LF_model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = LF_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}')