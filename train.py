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
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import v2

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
        choices=("early", "late", "3d", "gru_rcn", "rnn_cnn"),
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
    transforms = v2.Compose([
        v2.RandomPhotometricDistort(p=0.5),
        v2.RandomHorizontalFlip(p=0.5),
    ])
    dataset = EmotionFrameDataset(root_dir, transforms)

    # Define the split sizes
    train_size = int(0.7 * len(dataset))  # 70% for training
    val_size = int(0.1 * len(dataset))    # 10% for validation
    test_size = len(dataset) - train_size - val_size  # Remaining 20% for testing

    # Split the dataset into training, validation, and testing
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(dataset, 
                                                            [train_size, val_size, test_size],
                                                            generator=generator)


    # Create DataLoader for each dataset
    train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=10, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    num_epochs = 40
    print(f'Training for {num_epochs} epochs...')
    best_val_accuracy = 0.0

    for epoch in range(num_epochs):
        with tqdm(train_dataloader, unit="batch") as tepoch:
            model.train()
            dataset.enable_transform = True
            running_loss = 0.0
            correct = 0
            total = 0
            for inputs, e_labels in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                inputs, e_labels = inputs.to(device), e_labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, e_labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += e_labels.size(0)
                correct += (predicted == e_labels).sum().item()

                tepoch.set_postfix(loss=loss.item())
                
        avg_running_loss = running_loss / len(train_dataloader)
        train_accuracy = 100 * correct / total
        train_losses.append(avg_running_loss)
        train_accuracies.append(train_accuracy)
        writer.add_scalar("Loss/train", avg_running_loss, epoch+1)
        writer.add_scalar("Accuracy/train", train_accuracy, epoch+1)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_dataloader)}, Accuracy: {train_accuracy}%')

        # Validation loop
        model.eval()
        dataset.enable_transform = False
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, e_labels in val_dataloader:
                inputs, e_labels = inputs.to(device), e_labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, e_labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += e_labels.size(0)
                correct += (predicted == e_labels).sum().item()

        avg_val_loss = val_loss / len(val_dataloader)
        val_accuracy = 100 * correct / total
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        writer.add_scalar("Loss/val", avg_val_loss, epoch + 1)
        writer.add_scalar("Accuracy/val", val_accuracy, epoch + 1)
        print(f'Validation Loss: {avg_val_loss}, Accuracy: {val_accuracy}%')

        # Check if this is the best model so far
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), f'{args.model}_weights.pth')
            print(f'Saved Best Model with Validation Accuracy: {best_val_accuracy}%')

        # Step the scheduler based on the validation loss
        scheduler.step(avg_val_loss)

    # Load the best model for testing
    model.load_state_dict(torch.load(f'{args.model}_weights.pth'))

    print('Testing model...')
    model.eval()
    dataset.enable_transform = False
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

    # Plotting the results
    plt.figure(figsize=(12, 5))

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{args.model}.png')