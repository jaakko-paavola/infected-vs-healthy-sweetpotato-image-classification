# %%
import pandas as pd
from dotenv import load_dotenv
import os
from typing import List
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from preprocessing.preprocessing_utils import fetch_image_data_from_trial_folder, fetch_image_data_from_folder
from segmentation.separate_leaves import segment

def process_and_segment_leaves():
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

    # %% separate leaves and save leaf paths to df with original paths
    leaves_segmented = leaf_master.copy()

    segmented_path_lists_masked = []
    segmented_path_lists_original = []

    for i, row in leaves_segmented.iterrows():
        original_path = row["Original image path"]
        original_masked_path = row["Masked image path"]

        # currently original (= not masked) images can be segmented only if there is a corresponding masked image
        segmented_paths_masked, segmented_paths_original = segment(os.path.join(DATA_FOLDER, original_masked_path), os.path.join(DATA_FOLDER, original_path))

        segmented_path_lists_masked.append(segmented_paths_masked)
        segmented_path_lists_original.append(segmented_paths_original)


    leaves_segmented = leaves_segmented.assign(
        segmented_masked_image_path = segmented_path_lists_masked,
        segmented_original_image_path = segmented_path_lists_original)

    leaves_segmented = leaves_segmented.apply(pd.Series.explode).reset_index()

    # %% Add categorical variables

    leaves_segmented['Label Category'] = pd.Categorical(leaves_segmented['Condition'])
    leaves_segmented['Label'] = leaves_segmented['Label Category'].cat.codes

    # %% Drop NaN column

    leaves_segmented.dropna(axis="index", inplace=True)

    # %% Rename columns

    leaves_segmented.rename(columns={
        "segmented_masked_image_path": "Split masked image path",
        "segmented_original_image_path": "Split original image path"
    }, inplace=True)

    # %%

    file_name = "leaves_segmented_master.csv"
    file_path = os.path.join(DATA_FOLDER, file_name)
    leaves_segmented.to_csv(file_path, index=False)


def preprocess_leaf_data(excel_path, output_path = None):
    DATA_FOLDER = os.getenv("DATA_FOLDER_PATH")

    leaf_master = pd.DataFrame()

    df = pd.DataFrame(columns = ['Genotype', 'Condition', 'Image Type', 'File_index', 'Original image path', 'Masked image path'])

    df = df.append(fetch_image_data_from_folder(excel_path), ignore_index = True)

    # Image type and file index were useful only when collecting images, we can remove them now
    df = df.drop(columns=["Image Type", "File_index"])

    leaf_master = pd.concat([leaf_master, df])

    leaf_master = leaf_master.reset_index()

    file_name = "leaf_data.csv"
    if output_path is not None:
        file_path = os.path.join(output_path, file_name)
    else:
        file_path = os.path.join(DATA_FOLDER, file_name)
    leaf_master.to_csv(file_path)

    return file_path
