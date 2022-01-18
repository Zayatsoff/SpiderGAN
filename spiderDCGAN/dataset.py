import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io
from PIL import Image
import numpy as np


class SpiderDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 1])
        image = Image.open(img_path)
        # y_label = torch.tensor(self.annotations.iloc[index, 2])

        if self.transform:
            image = self.transform(image)

        return (image)
