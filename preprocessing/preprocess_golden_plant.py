# %%

import pandas as pd
from dotenv import load_dotenv
import os
from utils.image_utils import find_minimum_bounding_box_from_masked_image
import cv2
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

# %%

load_dotenv()

DATA_FOLDER = os.getenv("DATA_FOLDER_PATH")

# %%

plant_split_df_name = "plant_data_split_master.csv"
leaf_split_df_name = "leaves_segmented_master.csv"

PLANT_SPLIT_DF = pd.read_csv(os.path.join(DATA_FOLDER, plant_split_df_name))
LEAF_SPLIT_DF = pd.read_csv(os.path.join(DATA_FOLDER, leaf_split_df_name))
# %%

# Remove index columns if they are present

PLANT_SPLIT_DF.drop(columns=["index", "level_0", "Unnamed: 0", "Unnamed: 0.1"], inplace=True, errors="ignore")
LEAF_SPLIT_DF.drop(columns=["index", "level_0", "Unnamed: 0", "Unnamed: 0.1"], inplace=True, errors="ignore")

# %% Filter out the smallest images


image_areas = []

for row in PLANT_SPLIT_DF.iterrows():
  image_path = row['Split masked image path']
  
  image = cv2.imread(image_path)
  
  x_min, y_min, x_max, y_max = find_minimum_bounding_box_from_masked_image(image)
  
  image_area = abs(x_max - x_min) * abs(y_max - y_min)
  
  image_areas.append(image_area)
  
  
  
sns.histplot(image_area, x="area")