# %%
import pandas as pd
from dotenv import load_dotenv
import os
from typing import List
from preprocessing_utils import fetch_image_data_from_trial_folder

# %%
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# %%
load_dotenv()

# %%
trial2_dataset3 = "Data/Trial_02/Dataset_03"

# %%
DATA_FOLDER = os.getenv("DATA_FOLDER_PATH")

# %%
df_trial2_dataset3 = pd.DataFrame(columns = ['Trial', 'Dataset', 'Genotype', 'Condition', 'Image Type', 'File_index', 'Original image path', 'Masked image path'])
trial2_dataset3_folder = os.path.join(DATA_FOLDER, trial2_dataset3)

# %%
df_trial2_dataset3 = df_trial2_dataset3.append(fetch_image_data_from_trial_folder(trial2_dataset3_folder, 2, 3), ignore_index = True)

# %%
# Image type and file index were useful only when collecting images, we can remove them now
df_trial2_dataset3 = df_trial2_dataset3.drop(columns=["Image Type", "File_index"])

# %%
file_name = "growth_chamber_plant_data.csv"
file_path = os.path.join(DATA_FOLDER, file_name)
df_trial2_dataset3.to_csv(file_path)


