# %%

from multiprocessing.sharedctypes import Value
import os
from torch.utils.data import DataLoader
from dataloaders.csv_data_loader import CSVDataLoader
from dotenv import load_dotenv
import numpy as np
from torchvision import transforms

load_dotenv()

DATA_FOLDER_PATH = os.getenv("DATA_FOLDER_PATH")

def get_normalization_mean_std(dataset: str = None, datasheet : str = None):

    if datasheet:
        MASTER_PATH = datasheet

    if dataset:
        if dataset == 'leaf':
            DATA_PATH = "leaves_segmented_master.csv"
        elif dataset == 'plant':
            DATA_PATH = "plant_data_split_master.csv"
        elif dataset == 'plant_golden':
            DATA_PATH = "plant_data_split_golden.csv"
        else:
            raise ValueError(f"Dataset {dataset} not defined. Accepted values: plant, plant_golden, leaf")

        MASTER_PATH = os.path.join(DATA_FOLDER_PATH, DATA_PATH)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor()
    ])

    master_dataset = CSVDataLoader(
        csv_file=MASTER_PATH,
        root_dir=DATA_FOLDER_PATH,
        image_path_col="Split masked image path",
        label_col="Label",
        transform=transform
    )

    BATCH_SIZE = 1

    master_dataloader = DataLoader(master_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    image_mean = []
    image_std = []

    for i, data in enumerate(master_dataloader):

        # shape (batch_size, 3, height, width)
        numpy_image = data['image'].numpy()

        # shape (3,)
        batch_mean = np.mean(numpy_image, axis=(0, 2, 3))
        batch_std0 = np.std(numpy_image, axis=(0, 2, 3))

        image_mean.append(batch_mean)
        image_std.append(batch_std0)


    image_mean = np.array(image_mean).mean(axis=0)
    image_std = np.array(image_std).mean(axis=0)

    # print(f"Image mean: {image_mean}")
    # print(f"Image std: {image_std}")

    return image_mean, image_std
