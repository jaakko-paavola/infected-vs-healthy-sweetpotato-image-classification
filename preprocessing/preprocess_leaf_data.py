# %%
import pandas as pd
from dotenv import load_dotenv
import os
from typing import List
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from preprocessing_utils import fetch_image_data_from_trial_folder
# %%
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# %%
load_dotenv()

# %%

trial1_dataset2 = "Data/Trial_01/Dataset_02"
trial2_dataset2 = "Data/Trial_02/Dataset_02"

# %%
DATA_FOLDER = os.getenv("DATA_FOLDER_PATH")

# %%
df_trial1_dataset2 = pd.DataFrame(columns = ['Trial', 'Dataset', 'Genotype', 'Condition', 'Image Type', 'File_index', 'Original image path', 'Masked image path'])
trial1_dataset2_folder = os.path.join(DATA_FOLDER, trial1_dataset2)

# %%                    
        
df_trial1_dataset2 = df_trial1_dataset2.append(fetch_image_data_from_trial_folder(trial1_dataset2_folder, 1, 2), ignore_index = True)

# Image type and file index were useful only when collecting images, we can remove them now
df_trial1_dataset2 = df_trial1_dataset2.drop(columns=["Image Type", "File_index"])

# %%
df_trial2_dataset2 = pd.DataFrame(columns = ['Trial', 'Dataset', 'Genotype', 'Condition', 'Image Type', 'File_index', 'Original image path', 'Masked image path'])
trial2_dataset2_folder = os.path.join(DATA_FOLDER, trial2_dataset2)
df_trial2_dataset2 = df_trial2_dataset2.append(fetch_image_data_from_trial_folder(trial2_dataset2_folder, 2, 2), ignore_index = True)

# Image type and file index were useful only when collecting images, we can remove them now
df_trial2_dataset2 = df_trial2_dataset2.drop(columns=["Image Type", "File_index"])

# %%
leaf_master = pd.concat([df_trial1_dataset2, df_trial2_dataset2])

# %%
leaf_master = leaf_master.reset_index()

# %%

file_name = "leaf_data.csv"
file_path = os.path.join(DATA_FOLDER, file_name)
leaf_master.to_csv(file_path)