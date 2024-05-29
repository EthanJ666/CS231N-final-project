import os
import torch

from torchvision.datasets.folder import make_dataset, find_classes
from torch.utils.data import Dataset

class EmotionFrameDataset(Dataset):
    """
    Dataset containing all sampled frames and corresponding emotion labels
    """
    def __init__(self, root_dir, device):
        super(EmotionFrameDataset).__init__()

        _, class_to_idx = find_classes(root_dir)
        self.paths_and_labels = make_dataset(root_dir, class_to_idx, extensions=['.pt'])
        self.device = device

        # Ensure the number of samples is divisible by 10 - added by Suxi
        if len(self.paths_and_labels) % 10 != 0:
            self.paths_and_labels = self.paths_and_labels[:-(len(self.paths_and_labels) % 10)]

    def __len__(self):
        return len(self.paths_and_labels)
    
    def __getitem__(self, idx):
        path, label = self.paths_and_labels[idx]
        frames = torch.load(path).type(torch.FloatTensor) / 255.0
        return frames, label
