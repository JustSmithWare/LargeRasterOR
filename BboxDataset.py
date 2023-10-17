"""
Bounding Box Dataset
--------------------

This module provides the BboxDataset class for PyTorch's Dataset, along with a utility function to
create an instance of BboxDataset from a CSV file. 
"""

import pandas as pd
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import numpy as np

class BboxDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        self.annotations['label'] = self.annotations['label'].astype('category')
        self.annotations['label_code'] = self.annotations['label'].cat.codes
        
        self.unique_images = self.annotations.tile_path.unique()

        self.expected_shape = None

    def __len__(self):
        return len(self.unique_images)

    def __getitem__(self, index):
        annotations = self.annotations[self.annotations.tile_path == self.unique_images[index]]
        
        img_path = os.path.join(self.root_dir, self.unique_images[index])
        image = Image.open(img_path).convert("RGB")
        
        if self.expected_shape is None:
            self.expected_shape = image.size
        
        boxes = annotations[['xmin', 'ymin', 'xmax', 'ymax']].values.astype("float32")
        labels = annotations['label_code'].values.astype("int64")

        targets = {}
        targets["boxes"] = torch.tensor(boxes, dtype=torch.float32)
        targets["labels"] = torch.tensor(labels, dtype=torch.int64)

        if self.transform:
            image = ToTensor()(image)


        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

        return image, targets

def create_dataset_from_csv(csv_path, root_dir):
    return BboxDataset(csv_file=csv_path, root_dir=root_dir, transform=ToTensor())
