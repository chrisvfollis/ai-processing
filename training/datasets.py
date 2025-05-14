# standard dependencies
pass

# 3rd-party dependencies
from torch.utils.data import Dataset
from PIL import Image

# internal dependencies
pass


class EventImgs(Dataset):
    def __init__(self, img_data_df, transform=None):
        self.image_paths = img_data_df['paths'].tolist()
        self.labels = img_data_df['employee_id'].tolist()
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
