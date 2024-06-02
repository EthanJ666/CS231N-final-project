#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS231N 2023-2024: Final Project
train.py: Train all models
Yichen Jiang <ycjiang@stanford.edu>
Senyang Jiang <senyangj@stanford.edu>
Suxi Li <suxi2024@stanford.edu>
"""
from tqdm import tqdm
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from early_fusion_CNN import EarlyFusionCNN
from late_fusion_CNN import LateFusionModel
from video3DCNN import Video3DCNN
from GRU_RCN import RCN
from RNN_CNN import RCNN

from EmotionFrameDataset import EmotionFrameDataset
from EmotionVideoDataset import EmotionVideoDataset

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        choices=("early, late, 3d, gru_rcn, rnn_cnn"),
    )

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    num_frames = 6

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'using device {device}')

    writer = SummaryWriter()
    args = parse_args()

    if args.model == 'early':
        model = EarlyFusionCNN(num_frames).to(device)
    elif args.model == 'late':
        model = LateFusionModel(num_frames).to(device)
    elif args.model == '3d':
        model = Video3DCNN().to(device)
    elif args.model == 'gru_rcn':
        model = RCN().to(device)
    elif args.model == 'rnn_cnn':
        model = RCNN().to(device)

    
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'total params: {pytorch_total_params}')

    # dataset
    root_dir = './dataset_images_resize'
    # dataset = EmotionVideoDataset(root_dir, n_frames=6)
    dataset = EmotionFrameDataset(root_dir)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)

    train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 5
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
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_dataloader)}')

    # Save trained weights
    torch.save(model.state_dict(), f'{args.model}_weights.pth')

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
