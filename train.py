#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS231N 2023-2024: Final Project
train.py: Train all models
Yichen Jiang <ycjiang@stanford.edu>
Senyang Jiang <senyangj@stanford.edu>
Suxi Li <suxi2024@stanford.edu>
"""
import os
from tqdm import tqdm
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import v2

# confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

from early_fusion_CNN import EarlyFusionCNN
from late_fusion_CNN import LateFusionModel
from video3DCNN import Video3DCNN
from GRU_RCN import RCN
from RNN_CNN import RCNN

from EmotionFrameDataset import EmotionFrameDataset

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        default='early',
        choices=("early", "late", "3d", "gru_rcn", "rnn_cnn"),
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=6,
        choices=(6, 10, 16),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40
    )
    parser.add_argument(
        "--augment",
        action='store_true',
    )


    args = parser.parse_args()

    return args

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'using device {device}')

    args = parse_args()

    run_name = f'{args.model}_{args.frames}_{args.epochs}_augment={args.augment}'

    if args.model == 'early':
        model = EarlyFusionCNN(args.frames).to(device)
    elif args.model == 'late':
        model = LateFusionModel(args.frames).to(device)
    elif args.model == '3d':
        model = Video3DCNN(args.frames).to(device)
    elif args.model == 'gru_rcn':
        model = RCN().to(device)
    elif args.model == 'rnn_cnn':
        model = RCNN().to(device)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'total params: {pytorch_total_params}')

    # dataset
    root_dir = f'./dataset_images_{args.frames}_resize'
    transforms = v2.Compose([
        v2.RandomPhotometricDistort(p=0.5),
        v2.RandomHorizontalFlip(p=0.5),
    ])
    dataset = EmotionFrameDataset(root_dir, transforms)

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

    # Create a list of class names in the correct order - confusion matrix
    class_names = [label for label, index in sorted(emotion_labels.items(), key=lambda item: item[1])]

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

    print(f'Training for {args.epochs} epochs...')
    best_val_accuracy = 0.0

    for epoch in range(args.epochs):
        with tqdm(train_dataloader, unit="batch") as tepoch:
            model.train()
            dataset.enable_transform = args.augment
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
        print(f'Epoch [{epoch+1}/{args.epochs}], Loss: {running_loss/len(train_dataloader)}, Accuracy: {train_accuracy:.1f}%')

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
        print(f'Validation Loss: {avg_val_loss}, Accuracy: {val_accuracy:.1f}%')

        # Check if this is the best model so far
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), f'{run_name}_weights.pth')
            print(f'Saved Best Model with Validation Accuracy: {best_val_accuracy:.1f}%')

        # Step the scheduler based on the validation loss
        scheduler.step(avg_val_loss)

    # Load the best model for testing
    model.load_state_dict(torch.load(f'{run_name}_weights.pth'))

    print('Testing model...')
    model.eval()
    dataset.enable_transform = False
    test_loss = 0.0
    correct = 0
    total = 0

    # Initialize lists to store true labels and predictions confusion matrix
    all_preds = []
    all_labels = []
    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []

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

                # confusion matrix
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(e_labels.cpu().numpy())

                # Identify misclassified samples
                misclassified_idx = (predicted != e_labels).nonzero(as_tuple=True)[0]
                for idx in misclassified_idx:
                    misclassified_images.append(inputs[idx].cpu())
                    misclassified_labels.append(e_labels[idx].cpu().item())
                    misclassified_preds.append(predicted[idx].cpu().item())

    # Collect Results
    os.makedirs(f'results/{run_name}', exist_ok=True)

    test_accuracy = 100 * correct / total
    print(f'Test Loss: {test_loss/len(test_dataloader)}, Accuracy: {test_accuracy}%')
    with open(f'results/{run_name}/test_acc', 'w') as f:
        f.write(f'{test_accuracy}')

    # Compute confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # Calculate per-category accuracy
    class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1) * 100
    # Save per-category accuracy
    df = pd.DataFrame([class_accuracy])
    df.to_csv(f'results/{run_name}/class_accuracy.csv', header=class_names, index=False)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(f'results/{run_name}/confusion_matrix.png')
    plt.close()

    # Visualize a few misclassified examples
    num_misclassified = min(5, len(misclassified_images))  # Adjust the number of examples to visualize
    plt.figure(figsize=(args.frames*3, 15))
    for i in range(num_misclassified):
        img = misclassified_images[i]
        T, C, H, W = img.size()
        img = img.permute(2, 0, 3, 1).reshape(H, T*W, C)  # (T, C, H, W) -> (H, T, W, C) -> (H, T*W, C)
        true_label = class_names[misclassified_labels[i]]
        predicted_label = class_names[misclassified_preds[i]]
        plt.subplot(5, 1, i + 1)
        plt.imshow(img)
        plt.title(f'True: {true_label}, Pred: {predicted_label}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'results/{run_name}/misclassified_imgs.png')
    plt.close()

    # Plotting the results
    plt.figure(figsize=(12, 5))

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, args.epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, args.epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, args.epochs + 1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, args.epochs + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'results/{run_name}/results.png')
    plt.close()