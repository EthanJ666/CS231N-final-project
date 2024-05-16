import cv2
import os
import numpy as np
import torch
from torch.utils.data import Dataset

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