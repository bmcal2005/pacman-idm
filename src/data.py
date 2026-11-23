"""
PyTorch Dataset over the clips built in notebooks/02_build_clips.ipynb.

frames.npy stores every frame once. A clip is 16 frames, every 4th frame,
starting at a saved start position: frames[start + 0, 4, 8, ..., 60].
"""
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class ClipDataset(Dataset):
    def __init__(self, processed_dir, split):
        meta = json.load(open(os.path.join(processed_dir, "meta.json")))
        self.class_names = meta["class_names"]
        self.offsets = np.arange(meta["seq"]) * meta["skip"]          # [0, 4, ..., 60]
        sp = np.load(os.path.join(processed_dir, "splits.npz"))
        self.starts = sp[f"{split}_starts"]
        self.labels = sp[f"{split}_labels"]
        self.path = os.path.join(processed_dir, "frames.npy")
        self.frames = None                                            # opened lazily (see _frames)

    def _frames(self):
        # Memory-mapped: the 2.7 GB file stays on disk and only the frames we touch are read.
        # Opened on first use so each DataLoader worker gets its own handle.
        if self.frames is None:
            self.frames = np.load(self.path, mmap_mode="r")
        return self.frames

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, k):
        clip = self._frames()[self.starts[k] + self.offsets]           # (16, 84, 84) uint8
        y = int(self.labels[k])
        # Stay uint8 here (4x smaller to move around); the model divides by 255.
        return torch.from_numpy(np.ascontiguousarray(clip)), torch.tensor(y)

    def class_counts(self):
        return np.bincount(self.labels, minlength=len(self.class_names))


def to_hv(y):
    """9-way class index -> (horizontal, vertical) indices. Works on tensors or ints."""
    return y % 3, y // 3
