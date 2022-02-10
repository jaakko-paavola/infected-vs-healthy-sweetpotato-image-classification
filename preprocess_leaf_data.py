# %%
import pandas as pd
from dotenv import load_dotenv
import os
from typing import List
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# %%
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# %%
load_dotenv()

# %%
trial1_dataset1 = "Data/Trial_01/Dataset_01"
trial1_dataset2 = "Data/Trial_01/Dataset_02"
trial2_dataset1 = "Data/Trial_02/Dataset_01"
trial2_dataset2 = "Data/Trial_02/Dataset_02"
trial2_dataset3 = "Data/Trial_02/Dataset_03"

masked_img_path = "Mask Img"
original_img_path = "Original Img"

# %%
DATA_FOLDER = os.getenv("DATA_FOLDER_PATH")

# %%
df_trial1_dataset2 = pd.DataFrame(columns = ['Trial', 'Dataset', 'Genotype', 'Condition', 'Image Type', 'File_index', 'Original image path', 'Masked image path'])
trial1_dataset2_folder = os.path.join(DATA_FOLDER, trial1_dataset2)

def leaf_array_filter(item, **kwargs):
    
    item_equals = True
    
    for key, value in kwargs.items():
        if item[key] != value:
            item_equals = False
    
    return item_equals


def fetch_images_from_leaf_folder(trial_folder, trial, dataset) -> List:
    
    leaf_data = []

    for root, dirs, files in sorted(os.walk(trial_folder)):

        # Iterate over subfolders until we are in folder that contains image files
        if len(files) == 0:
            continue

        # Parse genotype, image type (original or masked) and condition (healthy, VD, FMV, CSV) from folder name
        parent_folder = root.split("/")[-2]
        current_folder = root.split("/")[-1]

        genotype = None
        condition = None
        image_type = None

        if "hua" in parent_folder.lower():
            genotype = "Hua"
        if "r3" in parent_folder.lower():
            genotype = "R3"
        if "healthy" in parent_folder.lower():
            condition = "Healthy"
        if "vd" in parent_folder.lower():
            condition = "VD"
        if "csv" in parent_folder.lower():
            condition = "CSV"
        if "fmv" in parent_folder.lower():
            condition = "FMV"
        if "mask" in current_folder.lower():
            image_type = "Masked"
        if "original" in current_folder.lower():
            image_type = "Original"

        # Sort files by filename where digits are iterpreted as number 
        files = sorted(files, key=lambda file: int(''.join(filter(str.isdigit, file))))
        
        for index, file in enumerate(files):

            # Remove DATA_FOLDER path from the full path so path becomes relative to the data folder
            image_path = os.path.relpath(os.path.join(root, file), DATA_FOLDER)

            # Check if array contains already the same image in different format (masked/original)
            existing_row_filter = filter(lambda item: leaf_array_filter(item, Trial=trial, Dataset=dataset, Genotype=genotype, Condition=condition, File_index=index), leaf_data)

            existing_row = list(existing_row_filter)

            if len(existing_row) != 0:
                if image_type == 'Original':
                    existing_row[0]['Original image path'] = image_path
                if image_type == 'Masked':
                    existing_row[0]['Masked image path'] = image_path

            else:
                if image_type == "Original":
                    leaf_data.append({"Trial": trial, "Dataset": dataset, "Genotype": genotype, "Condition": condition, 'File_index': index, "Image Type": image_type, "Original image path": image_path})
                if image_type == "Masked":
                    leaf_data.append({"Trial": trial, "Dataset": dataset, "Genotype": genotype, "Condition": condition, 'File_index': index, "Image Type": image_type, "Masked image path": image_path})

    return leaf_data

# %%                    
        
df_trial1_dataset2 = df_trial1_dataset2.append(fetch_images_from_leaf_folder(trial1_dataset2_folder, 1, 2), ignore_index = True)

# Image type and file index were useful only when collecting images, we can remove them now
df_trial1_dataset2 = df_trial1_dataset2.drop(columns=["Image Type", "File_index"])

# %%
df_trial2_dataset2 = pd.DataFrame(columns = ['Trial', 'Dataset', 'Genotype', 'Condition', 'Image Type', 'File_index', 'Original image path', 'Masked image path'])
trial2_dataset2_folder = os.path.join(DATA_FOLDER, trial2_dataset2)
df_trial2_dataset2 = df_trial2_dataset2.append(fetch_images_from_leaf_folder(trial2_dataset2_folder, 2, 2), ignore_index = True)

# Image type and file index were useful only when collecting images, we can remove them now
df_trial2_dataset2 = df_trial2_dataset2.drop(columns=["Image Type", "File_index"])

# %%
df_master = pd.concat([df_trial1_dataset2, df_trial2_dataset2])

# %%
df_master = df_master.reset_index()

# %%

file_name = "leaf_data.csv"
file_path = os.path.join(DATA_FOLDER, file_name)
df_master.to_csv(file_path)