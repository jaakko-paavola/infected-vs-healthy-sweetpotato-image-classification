# %%

import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import re
import cv2


# %%

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 30)
pd.options.display.max_colwidth = 100
# %%

load_dotenv()

# %%

DATA_FOLDER_PATH = os.getenv("DATA_FOLDER_PATH")

# %%

plant_df_split = pd.read_csv(f'{DATA_FOLDER_PATH}/plant_data_split.csv')
# %%

growth_plant_df_split = pd.read_csv(f'{DATA_FOLDER_PATH}/growth_chamber_plant_data_split.csv')
# %%

plant_df_split = plant_df_split[["Trial", "Dataset", "Genotype", "Condition", "Original image path", "Masked image path", "Split masked image path"]]

# %%

growth_plant_df_split
growth_plant_df_split.drop(columns=["Unnamed: 0"], inplace=True)

# %%

growth_plant_df_split.dropna(inplace=True)

# %% 

plant_df_split.dropna(inplace=True)

# %%

plant_df_split['Split masked image path'] = plant_df_split['Split masked image path'].str.replace("/home/redande/University/ds_project/Infected-sweetpotato-classification/data/", "")

# %%

plant_df_split

# %%
growth_plant_df_split['Split masked image path'] = growth_plant_df_split['Split masked image path'].str[1:]

# %%

growth_plant_df_split
# %%

plant_df_split_master = pd.concat([plant_df_split, growth_plant_df_split])
# %%

plant_df_split_master
# %%

plant_df_split_master['Label Category'] = pd.Categorical(plant_df_split_master['Condition'])
plant_df_split_master['Label Category'] = pd.Categorical(plant_df_split_master['Condition'])
plant_df_split_master['Label'] = plant_df_split_master['Label Category'].cat.codes
plant_df_split_master

# %%

# Drop these rows because the splitted files don't exist
plant_df_split_master[plant_df_split_master['Masked image path'].str.contains('180724 - 05 - TV - R3-H - 14-15 - Mask')]['Split masked image path']
plant_df_split_master.drop(96, inplace=True)


# %%
plant_df_split_master[plant_df_split_master['Masked image path'].str.contains('180724 - 06 - TV - R3-FMV - 11-13 - Mask')]['Split masked image path']
plant_df_split_master.drop(110, inplace=True)

# %%

plant_split_master_file_name = "plant_data_split_master.csv"
plant_split_master_file_path = os.path.join(DATA_FOLDER_PATH, plant_split_master_file_name)
plant_df_split_master.to_csv(plant_split_master_file_path)
# %%
