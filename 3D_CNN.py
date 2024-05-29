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


class Video3DCNN(nn.Module):
    def __init__(self, height=1280, width=720, frames=6, num_classes=8):
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
        self.fc1 = nn.Linear(128 * (frames) * (height // 64) * (width // 64), 512)
        #self.fc1 = nn.Linear(128 * (frames) * 3 * 5, 512)
        self.fc1_bn = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc2_bn = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, num_classes)

    def forward(self, x):
        B = x.size(0)  # torch.Size([10, 6, 3, 720, 1280])
        x = x.permute(0, 2, 1, 3, 4)   # torch.Size([10, 3, 6, 720, 1280])
        x = self.pool(F.relu(self.conv1_bn(self.conv1(x)))) # torch.Size([10, 16, 6, 180, 320])
        x = self.pool(F.relu(self.conv2_bn(self.conv2(x)))) # torch.Size([10, 32, 6, 45, 80])
        x = self.pool(F.relu(self.conv3_bn(self.conv3(x)))) 
        x = self.pool(F.relu(self.conv4_bn(self.conv4(x)))) 
        x = x.reshape(B, -1)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.relu(self.fc2_bn(self.fc2(x)))
        out = self.fc3(x)

        return out

"""
class Emotion3DCNN(nn.Module):
    def __init__(self, cnn3d, cnn_features=512, num_classes=8):
        self.cnn = cnn3d
        self.fc1 = nn.Linear(cnn_features, 1024)
        self.fc1_bn = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        cnn_features = self.cnn(x)
        y = F.relu(self.fc1_bn(self.fc1(cnn_features)))
        out = self.fc2(y)

        return out
"""

if __name__ == "__main__":
    #num_classes = 8
    num_frames = 6

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'using device {device}')

    writer = SummaryWriter()

    cnn3d_model = Video3DCNN().to(device)
    #cnn3d_model = Emotion3DCNN(cnn_model).to(device)

    pytorch_total_params = sum(p.numel() for p in cnn3d_model.parameters() if p.requires_grad)
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
    optimizer = optim.Adam(cnn3d_model.parameters(), lr=0.001)

    num_epochs = 25
    print(f'Training for {num_epochs} epochs...')

    for epoch in range(num_epochs):
        with tqdm(train_dataloader, unit="batch") as tepoch:
            running_loss = 0.0
            for inputs, e_labels in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                inputs, e_labels = inputs.to(device), e_labels.to(device)
                optimizer.zero_grad()
                outputs = cnn3d_model(inputs)
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
    torch.save(cnn3d_model.state_dict(), 'CNN_3D_weights.pth')

    print('Testing model...')
    cnn3d_model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        with tqdm(test_dataloader, unit="batch") as tepoch:
            for inputs, e_labels in tepoch:
                inputs, e_labels = inputs.to(device), e_labels.to(device)
                outputs = cnn3d_model(inputs)
                loss = criterion(outputs, e_labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += e_labels.size(0)
                correct += (predicted == e_labels).sum().item()

    print(f'Test Loss: {test_loss/len(test_dataloader)}, Accuracy: {100 * correct / total}%')
    writer.add_scalar("Acc/test", 100 * correct / total, 0)
    writer.flush()
    writer.close()
