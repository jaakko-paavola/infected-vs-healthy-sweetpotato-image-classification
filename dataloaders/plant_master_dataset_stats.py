# %%

import os
from torch.utils.data import DataLoader
from dataloaders.plant_village_loader import PlantVillageLoader
from dotenv import load_dotenv
import numpy as np
from torchvision import transforms

# %%

load_dotenv()

# %%

DATA_FOLDER_PATH = os.getenv("DATA_FOLDER_PATH")
PLANT_SPLIT_MASTER_PATH = os.path.join(DATA_FOLDER_PATH, "plant_data_split_master.csv")

# %%

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])

plant_master_dataset = PlantVillageLoader(
  csv_file=PLANT_SPLIT_MASTER_PATH, 
  root_dir=DATA_FOLDER_PATH,
  image_path_col="Split masked image path",
  label_col="Label",
  transform=transform
)

plant_master_dataloader = DataLoader(plant_master_dataset, batch_size=1, shuffle=False, num_workers=4)

# %%

image_mean = []
image_std = []

for i, data in enumerate(plant_master_dataloader):
    # shape (batch_size, 3, height, width)
    numpy_image = data['image'].numpy()

    
    print(numpy_image.shape)

    # shape (3,)
    batch_mean = np.mean(numpy_image, axis=(0, 2, 3))
    batch_std0 = np.std(numpy_image, axis=(0, 2, 3))

    image_mean.append(batch_mean)
    image_std.append(batch_std0)

image_mean = np.array(image_mean).mean(axis=0)
image_std = np.array(image_std).mean(axis=0)

# %%

print(f"Image mean: {image_mean}")

# Image mean: [0.2234376  0.27598768 0.16376022]

print(f"Image std: {image_std}")

# Image std: [0.23811504 0.28631625 0.18748806]

# %%

# %%
