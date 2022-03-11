from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import os
from skimage import io

class PlantVillageLoader(Dataset):
    """Plant Village dataset."""

    def __init__(self, csv_file, root_dir, image_path_col: str = "image_path", label_col: str = "label", transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.plant_village_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.image_path_col = image_path_col
        self.label_col = label_col

    def __len__(self):
        return len(self.plant_village_df)

    def build_path(self, relative_path):
        return os.path.join(self.root_dir, relative_path)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.build_path(self.plant_village_df.loc[self.plant_village_df.index[idx], self.image_path_col])
        image = io.imread(img_path)
        label = self.plant_village_df.loc[self.plant_village_df.index[idx], self.label_col]
        label_tensor = torch.tensor(label, dtype=torch.int64)
        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'label': label_tensor}

        return sample
