# %%
import pandas as pd
from dotenv import load_dotenv
import os
from preprocessing_utils import find_matching_plant_images
# %%
pd.set_option('display.max_rows', None)

# %%
load_dotenv()

# %%
trial1_dataset1 = "Data/Trial_01/Dataset_01"
trial2_dataset1 = "Data/Trial_02/Dataset_01"

# %%
DATA_FOLDER = os.getenv("DATA_FOLDER_PATH")

# %%

# Extract data from Trial_01/Dataset_01
df_trial1_dataset1_data = pd.read_excel(os.path.join(DATA_FOLDER, trial1_dataset1, "01-Rgb_Morpho_Plant.xlsx"), sheet_name="Data")

# %%
# Remove extra columns for now
columns_to_keep = ["Genotype", "Condition", "Plant", "Tray", "Round", "Days", "Area (cm^2)"]
df_trial1_dataset1_data = df_trial1_dataset1_data[columns_to_keep]

# %%
# Get the first element (.iloc[0]) of the dataframe with same Tray and Round values (df[(row.Tray == df.Tray) & (row.Round == df.Round)])
# Calculate the difference between the first element of the same tray and same round index ("name") to rows index ("name")
def generate_tray_index(row, df):
    return 1 + row.name - df[(row.Tray == df.Tray) & (row.Round == df.Round)].iloc[0].name

# %%
df_trial1_dataset1_data['Tray Index'] = df_trial1_dataset1_data.apply(lambda row: generate_tray_index(row, df_trial1_dataset1_data), axis=1)

# %%
trial1_dataset1_original_path_prefix = os.path.join(trial1_dataset1, "FishEyeCorrected")
# First string argument is round and second is tray
trial1_dataset1_original_file_regex = r"\d{2}-%s-PS_Tray_%s-RGB2-FishEyeCorrected.png"

trial1_dataset1_masked_path_prefix = os.path.join(trial1_dataset1, "FishEyeMasked")
# First string argument is round and second is tray
trial1_dataset1_masked_file_regex = r"\d{2}-%s-PS_Tray_%s-RGB2-FishEyeMasked.png"

df_trial1_dataset1_data['Original image path'] = df_trial1_dataset1_data.apply(lambda row: find_matching_plant_images(row, trial1_dataset1_original_path_prefix, trial1_dataset1_original_file_regex), axis=1)
df_trial1_dataset1_data['Masked image path'] = df_trial1_dataset1_data.apply(lambda row: find_matching_plant_images(row, trial1_dataset1_masked_path_prefix, trial1_dataset1_masked_file_regex), axis=1)

# %%

# Add Trial and Dataset data to dataframe
df_trial1_dataset1_data.insert(0, "Trial", 1)
df_trial1_dataset1_data.insert(1, "Dataset", 1)

# %%

# Extract data from Trial_02/Dataset_01
df_trial2_dataset1_data = pd.read_excel(os.path.join(DATA_FOLDER, trial2_dataset1, "02-Rgb_Morpho_Plant.xlsx"), sheet_name="Data")

# %%
columns_to_keep = ["Genotype", "Condition", "Plant Name", "Tray ID", "Round Order", "Day", "Position", "AREA_MM"]
df_trial2_dataset1_data = df_trial2_dataset1_data[columns_to_keep]

# %%
# Convert cm^2-area from Trial1/Dataset1 to mm^2-area to match Trial2/Dataset1 -data

df_trial1_dataset1_data['AREA_MM'] = df_trial1_dataset1_data['Area (cm^2)']*100
df_trial1_dataset1_data.drop('Area (cm^2)', axis=1, inplace=True)


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

df_trial2_dataset1_data['Original image path'] = df_trial2_dataset1_data.apply(lambda row: find_matching_plant_images(row, trial2_dataset1_original_path_prefix, trial2_dataset1_original_file_regex), axis=1)
df_trial2_dataset1_data['Masked image path'] = df_trial2_dataset1_data.apply(lambda row: find_matching_plant_images(row, trial2_dataset1_masked_path_prefix, trial2_dataset1_masked_file_regex), axis=1)

# %%
df_trial2_dataset1_data.insert(0, "Trial", 2)
df_trial2_dataset1_data.insert(1, "Dataset", 1)

# %%

# Merge both plant dataframes together
plant_master = pd.concat([df_trial1_dataset1_data, df_trial2_dataset1_data])

# %%
plant_master = plant_master.dropna()

# %%
plant_master = plant_master.reset_index()

# %%
file_name = "plant_data.csv"
file_path = os.path.join(DATA_FOLDER, file_name)
plant_master.to_csv(file_path)
