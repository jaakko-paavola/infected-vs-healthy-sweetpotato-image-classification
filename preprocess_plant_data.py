# %%
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os
import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# %%
pd.set_option('display.max_rows', None)

# %%
load_dotenv()

# %%
trial1_dataset1 = "Data/Trial_01/Dataset_01"
trial1_dataset2 = "Data/Trial_01/Dataset_02"
trial2_dataset1 = "Data/Trial_02/Dataset_01"
trial2_dataset2 = "Data/Trial_02/Dataset_02"
trial2_dataset3 = "Data/Trial_02/Dataset_03"


# %%
DATA_FOLDER = os.getenv("DATA_FOLDER_PATH")

# %%
df_trial1_dataset1_data = pd.read_excel(os.path.join(DATA_FOLDER, trial1_dataset1, "01-Rgb_Morpho_Plant.xlsx"), sheet_name="Data")

# %%
# Remove extra columns for now
columns_to_keep = ["Genotype", "Condition", "Plant", "Tray", "Round", "Days"]
df_trial1_dataset1_data = df_trial1_dataset1_data[columns_to_keep]

# %%
# Get the first element (.iloc[0]) of the dataframe with same Tray and Round values (df[(row.Tray == df.Tray) & (row.Round == df.Round)])
# Calculate the difference between the first element of the same tray and same round index ("name") to rows index ("name")
def generate_tray_index(row, df):
    return 1 + row.name - df[(row.Tray == df.Tray) & (row.Round == df.Round)].iloc[0].name

# %%
df_trial1_dataset1_data['Tray Index'] = df_trial1_dataset1_data.apply(lambda row: generate_tray_index(row, df_trial1_dataset1_data), axis=1)

# %%
def map_row_to_image(row, path_prefix, name_regex_str):
    filename = ""
    
    row_tray = row.Tray
    row_round = row.Round
        
    name_path = name_regex_str % (row_round, row_tray)
    regex_pattern = re.compile(name_path)
    
    matches = []
    
    for root, dirs, files in os.walk(os.path.join(DATA_FOLDER, path_prefix)):
        for file in files:
            if regex_pattern.match(file):
                matches.append(file)
                
    if len(matches) != 1:
        print(f"{len(matches)} matches for file {name_path}")
        
        if len(matches) == 0:
            print("Warning: image possible missing")
        else:
            raise ValueError("Regex should find only one matching file")
    
    if len(matches) == 1:
        filename = matches[0]
        return os.path.join(path_prefix, filename)
    
    return np.NaN

# %%
trial1_dataset1_original_path_prefix = os.path.join(trial1_dataset1, "FishEyeCorrected")
# First string argument is round and second is tray
trial1_dataset1_original_file_regex = r"\d{2}-%s-PS_Tray_%s-RGB2-FishEyeCorrected.png"

trial1_dataset1_masked_path_prefix = os.path.join(trial1_dataset1, "FishEyeMasked")
# First string argument is round and second is tray
trial1_dataset1_masked_file_regex = r"\d{2}-%s-PS_Tray_%s-RGB2-FishEyeMasked.png"

df_trial1_dataset1_data['Original image path'] = df_trial1_dataset1_data.apply(lambda row: map_row_to_image(row, trial1_dataset1_original_path_prefix, trial1_dataset1_original_file_regex), axis=1)
df_trial1_dataset1_data['Masked image path'] = df_trial1_dataset1_data.apply(lambda row: map_row_to_image(row, trial1_dataset1_masked_path_prefix, trial1_dataset1_masked_file_regex), axis=1)

# %%
df_trial1_dataset1_data.insert(0, "Trial", 1)
df_trial1_dataset1_data.insert(1, "Dataset", 1)

# %%
df_trial2_dataset1_data = pd.read_excel(os.path.join(DATA_FOLDER, trial2_dataset1, "02-Rgb_Morpho_Plant.xlsx"), sheet_name="Data")

# %%
columns_to_keep = ["Genotype", "Condition", "Plant Name", "Tray ID", "Round Order", "Day", "Position"]
df_trial2_dataset1_data = df_trial2_dataset1_data[columns_to_keep]

# %%
# Extract only the numbers from the Position column and Tray ID column
df_trial2_dataset1_data['Tray Index'] = df_trial2_dataset1_data['Position'].str.extract('(\d+)')
df_trial2_dataset1_data['Tray'] = df_trial2_dataset1_data['Tray ID'].str.extract('(\d+)')

# %%
df_trial2_dataset1_data = df_trial2_dataset1_data.drop(columns=["Tray ID", "Position"])

# %%
df_trial2_dataset1_data = df_trial2_dataset1_data.rename(columns={"Plant Name": "Plant", "Round Order": "Round", "Day": "Days"})

# %%
trial2_dataset1_original_path_prefix = os.path.join(trial2_dataset1, "FishEyeCorrected")
# First string argument is round and second is tray
trial2_dataset1_original_file_regex = r"\d{2}-%s-PS_Tray_%s-RGB2-FishEyeCorrected.png"

trial2_dataset1_masked_path_prefix = os.path.join(trial2_dataset1, "FishEyeMasked")
# First string argument is round and second is tray
trial2_dataset1_masked_file_regex = r"\d{2}-%s-PS_Tray_%s-RGB2-FishEyeMasked.png"

df_trial2_dataset1_data['Original image path'] = df_trial2_dataset1_data.apply(lambda row: map_row_to_image(row, trial2_dataset1_original_path_prefix, trial2_dataset1_original_file_regex), axis=1)
df_trial2_dataset1_data['Masked image path'] = df_trial2_dataset1_data.apply(lambda row: map_row_to_image(row, trial2_dataset1_masked_path_prefix, trial2_dataset1_masked_file_regex), axis=1)

# %%
df_trial2_dataset1_data.insert(0, "Trial", 2)
df_trial2_dataset1_data.insert(1, "Dataset", 1)

# %%
df_master = pd.concat([df_trial1_dataset1_data, df_trial2_dataset1_data])

# %%
df_master = df_master.dropna()

# %%
df_master = df_master.reset_index()

# %%
file_name = "plant_data.csv"
file_path = os.path.join(DATA_FOLDER, file_name)
df_master.to_csv(file_path)
