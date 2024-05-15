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
from torch.utils.data import DataLoader, Dataset, random_split
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
        self.video_paths, self.labels = self._load_dataset()

    def _load_dataset(self):
        video_paths = []
        labels = []
        for emotion, label in self.emotion_labels.items():
            emotion_folder = os.path.join(self.root_dir, emotion)
            for video_name in os.listdir(emotion_folder):
                video_path = os.path.join(emotion_folder, video_name)
                video_paths.append(video_path)
                labels.append(label)

        return video_paths, labels

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

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(LF_model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, e_labels in train_dataloader:
        optimizer.zero_grad()
        outputs = LF_model(inputs)
        loss = criterion(outputs, e_labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}')

# Save trained weights of CNN and LF models
torch.save(cnn_model.state_dict(), 'naive_cnn_weights.pth')
torch.save(LF_model.state_dict(), 'late_fusion_weights.pth')

LF_model.eval()
test_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for inputs, e_labels in test_dataloader:
        outputs = LF_model(inputs)
        loss = criterion(outputs, e_labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Loss: {test_loss/len(test_dataloader)}, Accuracy: {100 * correct / total}%')