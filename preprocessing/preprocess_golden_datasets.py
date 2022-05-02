# %%

import pandas as pd
from dotenv import load_dotenv
import os
from utils.image_utils import find_minimum_bounding_box_from_masked_image, crop_image_with_bounding_box
import cv2
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm

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

detector = cv2.SIFT_create()
n_sift_features = []
image_areas = []

for index, row in tqdm(PLANT_SPLIT_DF.iterrows()):
  image_path = os.path.join(DATA_FOLDER, row['Split masked image path'])
  
  image = cv2.imread(image_path)
  image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
  x_min, y_min, x_max, y_max = find_minimum_bounding_box_from_masked_image(image)
  # cropped_image = crop_image_with_bounding_box(image, [x_min, y_min, x_max, y_max])
  # plt.imshow(cropped_image)
  # plt.show()
  
  image_area = abs(x_max - x_min) * abs(y_max - y_min)
    
  image_areas.append(image_area)
  
  keypoints, descriptor = detector.detectAndCompute(image_gray, None)
  n_sift_features.append(len(keypoints))

# %%

plot = sns.scatterplot(x="Areas", y="Features", data={"Areas": image_areas, "Features": n_sift_features}, size=5, alpha=0.8)
plot.set(xscale='log')
plot.set(yscale='log')

# %%

PLANT_SPLIT_DF['Area'] = pd.Series(image_areas)
PLANT_SPLIT_DF['N of SIFT features'] = pd.Series(n_sift_features)

# %%

PLANT_SPLIT_DF = PLANT_SPLIT_DF.sort_values(by=['Area'])

for index, row in tqdm(PLANT_SPLIT_DF.iterrows()):
  # Plot only every 25th image
  if index % 25 != 0:
    continue
  area = row['Area']
  n_features = row['N of SIFT features']
  image_path = os.path.join(DATA_FOLDER, row['Split masked image path'])
  
  print(f"Image with {n_features} features and area {area}")
  
  image = cv2.imread(image_path)
  
  plt.imshow(image)
  plt.show()

# %% Filter images that have area < 50 000 pixels, meaning that they are smaller than 223 x 223 pixels if squared and write CSV

PLANT_SPLIT_DF_GOLDEN = PLANT_SPLIT_DF[PLANT_SPLIT_DF['Area'] > 50000]

plant_golden_path = os.path.join(DATA_FOLDER, "plant_data_split_golden.csv")

PLANT_SPLIT_DF_GOLDEN.to_csv(plant_golden_path, index=False)

# %% Do the same with leaf images

detector = cv2.SIFT_create()
n_sift_features = []
image_areas = []

for index, row in tqdm(LEAF_SPLIT_DF.iterrows()):
  image_path = os.path.join(DATA_FOLDER, row['Split masked image path'])
  
  image = cv2.imread(image_path)
  image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
  x_min, y_min, x_max, y_max = find_minimum_bounding_box_from_masked_image(image)
  
  image_area = abs(x_max - x_min) * abs(y_max - y_min)
    
  image_areas.append(image_area)
  
  keypoints, descriptor = detector.detectAndCompute(image_gray, None)
  n_sift_features.append(len(keypoints))

# %%

plot = sns.scatterplot(x="Areas", y="Features", data={"Areas": image_areas, "Features": n_sift_features}, size=5, alpha=0.8)
plot.set(xscale='log')
plot.set(yscale='log')

# %%

LEAF_SPLIT_DF['Area'] = pd.Series(image_areas)
LEAF_SPLIT_DF['N of SIFT features'] = pd.Series(n_sift_features)

# %%

LEAF_SPLIT_DF = LEAF_SPLIT_DF.sort_values(by=['Area'])

for index, row in tqdm(LEAF_SPLIT_DF.iterrows()):
  # Plot only every 25th image
  if index % 25 != 0:
    continue
  area = row['Area']
  n_features = row['N of SIFT features']
  image_path = os.path.join(DATA_FOLDER, row['Split masked image path'])
  
  print(f"Image with {n_features} features and area {area}")
  
  image = cv2.imread(image_path)
  
  plt.imshow(image)
  plt.show()

# %%

# All images are so similar that no need for golden dataset