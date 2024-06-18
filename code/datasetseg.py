import os
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# Define the dataset class
from PIL import Image

class SegmentationDataset(Dataset):
    def __init__(self, data_dir, image_dir, mask_dir):
        self.image_dir = os.path.join(data_dir, image_dir)
        self.mask_dir = os.path.join(data_dir, mask_dir)
        self.image_paths = sorted(os.listdir(self.image_dir))
        self.mask_paths = sorted(os.listdir(self.mask_dir))
        self.transform = transform_compose

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_paths[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_paths[idx])
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image = self.transform(image)
            mask = T.Resize((240, 240))(mask)  # Resize the mask to match the image size
            mask = torch.from_numpy(np.array(mask, dtype=np.int64))

        return image, mask


# Transformations for images and masks
transform_compose = T.Compose([
    T.Resize((240, 240)),  # Resize the image and mask to match the models's input size
    T.ToTensor(),  # Convert the image to a tensor
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])
