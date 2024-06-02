import random
import torch

import torchvision
from torchvision.datasets.folder import make_dataset, find_classes
from torch.utils.data import Dataset
import torch.nn.functional as F

class EmotionVideoDataset(Dataset):
    """
    Dataset containing all video clips and corresponding emotion labels
    """
    def __init__(self, root_dir, n_frames):
        super(EmotionVideoDataset).__init__()

        self.n_frames = n_frames
        _, class_to_idx = find_classes(root_dir)
        self.paths_and_labels = make_dataset(root_dir, class_to_idx, extensions=['.mp4'])

    def __len__(self):
        return len(self.paths_and_labels)
    
    def __getitem__(self, idx):
        path, label = self.paths_and_labels[idx]
        frames = self.extract_frames(path, self.n_frames)
        return frames, label
    
    def extract_frames(self, video_path, n_frames, tf=None):
        """
        Args:
            video_path: video path
            n_frames: how many frames we want to extract in total (T)
            tf: whether to use transformer to lower the size of the frames

        Returns: frames: Tensor of size (n_frames, C, H, W)
        """
        # Get video object
        video, _, metadata = torchvision.io.read_video(video_path, pts_unit='sec', output_format='TCHW')
        fps = int(metadata['video_fps'])
        total_frames = video.size(0)
        sample_rate = fps // 2
        assert total_frames - sample_rate * (self.n_frames-1) > 0
        start_frame = random.randrange(0, total_frames - sample_rate * (self.n_frames-1))
        idx = torch.arange(0, n_frames) * sample_rate + start_frame
        frames = video[idx]
        resized_frames = F.interpolate(frames, size=(224,224), mode='bilinear', align_corners=False)
        resized_frames = resized_frames.type(torch.FloatTensor) / 255.0
        return resized_frames