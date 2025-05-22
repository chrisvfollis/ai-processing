# standard dependencies
import random
from collections import defaultdict

# 3rd-party dependencies
from torch.utils.data import Dataset, Sampler
from PIL import Image

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
    def __init__(self, labels, P, K):
        '''
        Args:
            labels (list, np.ndarray or pd.Series): A 1D list or array of class
                labels (identities).
            P (int): Number of distinct identities (classes) per batch.
            K (int): The minimum number of samples per identity in each batch.
        '''
        self.labels = labels
        self.P = P
        self.K = K

        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.label_to_indices[label].append(idx)

        self.valid_labels = [label for label, idxs in self.label_to_indices.items() if len(idxs) >= K]

    def __iter__(self):
        indices = []
        random.shuffle(self.valid_labels)

        for i in range(0, len(self.valid_labels), self.P):
            current_labels = self.valid_labels[i:i + self.P]
            if len(current_labels) < self.P:
                continue

            for label in current_labels:
                selected = random.sample(self.label_to_indices[label], self.K)
                indices.extend(selected)

        return iter(indices)

    def __len__(self):
        return len(self.valid_labels) // self.P * self.P * self.K
