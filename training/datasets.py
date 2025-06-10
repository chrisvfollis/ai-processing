# standard dependencies
import random
from collections import defaultdict
from typing import Optional
import math

# 3rd-party dependencies
from torch.utils.data import Dataset, Sampler
from PIL import Image
import pandas as pd
import numpy as np

# internal dependencies
pass


class EventImgs(Dataset):
    def __init__(self, img_data_df, transform=None):
        self.image_paths = img_data_df['path'].tolist()
        self.labels = img_data_df['person_id'].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


class PKSampler(Sampler):
    def __init__(
        self,
        labels: list | np.ndarray | pd.Series,
        P: int,
        K: int,
        num_batches: Optional[int] = None
    ):
        '''
        Args:
            labels (list, np.ndarray or pd.Series): A 1D list or array of class
                labels (identities).
            P (int): Number of distinct identities (classes) per batch.
            K (int): The minimum number of samples per identity in each batch.
            num_batches (int, optional): Total number of batches per epoch. If
                None, will try to match full dataset coverage.
        '''
        self.labels = labels
        self.P = P
        self.K = K

        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.label_to_indices[label].append(idx)

        self.valid_labels = [
            label for label, idxs in self.label_to_indices.items()
            if len(idxs) >= K
        ]

        if not self.valid_labels:
            raise ValueError(f'No identities with at least K={K} examples')
        
        self.num_batches = num_batches or math.ceil(len(labels) / (P * K))

    def __iter__(self):
        indices = []
        for _ in range(self.num_batches):
            selected_labels = random.choices(self.valid_labels, k=self.P)
            for label in selected_labels:
                candidates = self.label_to_indices[label]
                if len(candidates) >= self.K:
                    sampled = random.sample(candidates, self.K)
                else:
                    sampled = random.choices(candidates, k=self.K)
                indices.extend(sampled)
        return iter(indices)

    def __len__(self):
        return self.num_batches * self.P * self.K
