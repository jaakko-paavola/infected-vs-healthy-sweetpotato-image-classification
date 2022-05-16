# %%

import pandas as pd
from dotenv import load_dotenv
import os

# %%

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 10)

# %%

load_dotenv()

# %%

DATA_FOLDER = os.getenv("DATA_FOLDER_PATH")

# %%

segmented_dummy_data_path = os.path.join(DATA_FOLDER, "Dummy_Data")

# %%

data = []

for root, dirs, files in sorted(os.walk(segmented_dummy_data_path)):

  # Iterate over subfolders until we are in folder that contains image files
  if len(files) == 0:
      continue

  # Parse label and image type from parent and current folder
  parent_folder = root.split("/")[-2]
  current_folder = root.split("/")[-1]

  if parent_folder == 'color':
    image_type = "original"
  elif parent_folder == 'segmented':
    image_type = "segmented"
  else:
    raise ValueError("Could not infer image type")

  if "early_blight" in current_folder.lower():
    label = "early_blight"
  elif "late_blight" in current_folder.lower():
    label = "late_blight"
  elif "healthy" in current_folder.lower():
    label = "healthy"
  else:
    raise ValueError("Could not infer image label")


  for index, file in enumerate(files):

    # Remove DATA_FOLDER path from the full path so path becomes relative to the data folder
    image_path = os.path.relpath(os.path.join(root, file), DATA_FOLDER)

    data.append({"image_type": image_type, "image_path": image_path, "label_name": label})

# %%

df = pd.DataFrame(columns = ['image_type', 'image_path', 'label_name'])

df = df.append(data)

# %%

df['label_categories'] = pd.Categorical(df.label_name)
df['label'] = df['label_categories'].cat.codes

# %%

original_plant_village_df = df.loc[df['image_type'] == 'original']
segmented_plant_village_df = df.loc[df['image_type'] == 'segmented']

# %%

original_file_name = "dummy_original_plant_village_data.csv"
original_file_path = os.path.join(DATA_FOLDER, original_file_name)
original_plant_village_df.to_csv(original_file_path)

segmented_file_name = "dummy_segmented_plant_village_data.csv"
segmented_file_path = os.path.join(DATA_FOLDER, segmented_file_name)
segmented_plant_village_df.to_csv(segmented_file_path, index=False)
# %%
