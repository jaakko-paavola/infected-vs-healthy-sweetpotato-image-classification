import numpy as np
from dotenv import load_dotenv
import os
import re
from typing import List

load_dotenv()

DATA_FOLDER = os.getenv("DATA_FOLDER_PATH")

def find_matching_plant_images(row, path_prefix, name_regex_str):
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


def item_contains_keys(item, **kwargs):

    item_equals = True

    for key, value in kwargs.items():
        if item[key] != value:
            item_equals = False

    return item_equals

def extract_metadata_from_folder_names(parent_folder, current_folder):
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

    return genotype, condition, image_type

def fetch_image_data_from_trial_folder(trial_folder, trial, dataset) -> List:

    image_data = []

    for root, dirs, files in sorted(os.walk(trial_folder)):

        # Iterate over subfolders until we are in folder that contains image files
        if len(files) == 0:
            continue

        # Parse genotype, image type (original or masked) and condition (healthy, VD, FMV, CSV) from folder name
        parent_folder = root.split("/")[-2]
        current_folder = root.split("/")[-1]

        genotype, condition, image_type = extract_metadata_from_folder_names(parent_folder, current_folder)

        # Sort files by filename where digits are iterpreted as number
        files = sorted(files, key=lambda file: int(''.join(filter(str.isdigit, file))))

        for index, file in enumerate(files):

            # Remove DATA_FOLDER path from the full path so path becomes relative to the data folder
            image_path = os.path.relpath(os.path.join(root, file), DATA_FOLDER)

            # Check if array contains already the same image in different format (masked/original)
            existing_row_filter = filter(lambda item: item_contains_keys(item, Trial=trial, Dataset=dataset, Genotype=genotype, Condition=condition, File_index=index), image_data)

            existing_row = list(existing_row_filter)

            if len(existing_row) != 0:
                if image_type == 'Original':
                    existing_row[0]['Original image path'] = image_path
                if image_type == 'Masked':
                    existing_row[0]['Masked image path'] = image_path
            else:
                if image_type == "Original":
                    image_data.append({"Trial": trial, "Dataset": dataset, "Genotype": genotype, "Condition": condition, 'File_index': index, "Image Type": image_type, "Original image path": image_path})
                if image_type == "Masked":
                    image_data.append({"Trial": trial, "Dataset": dataset, "Genotype": genotype, "Condition": condition, 'File_index': index, "Image Type": image_type, "Masked image path": image_path})

    return image_data

def fetch_image_data_from_folder(data_path) -> List:

    image_data = []

    for root, dirs, files in sorted(os.walk(data_path)):

        # Iterate over subfolders until we are in folder that contains image files
        if len(files) == 0:
            continue

        # Parse genotype, image type (original or masked) and condition (healthy, VD, FMV, CSV) from folder name
        parent_folder = root.split("/")[-2]
        current_folder = root.split("/")[-1]

        genotype, condition, image_type = extract_metadata_from_folder_names(parent_folder, current_folder)

        # Sort files by filename where digits are iterpreted as number
        files = sorted(files, key=lambda file: int(''.join(filter(str.isdigit, file))))

        for index, file in enumerate(files):

            # Remove DATA_FOLDER path from the full path so path becomes relative to the data folder
            image_path = os.path.relpath(os.path.join(root, file), DATA_FOLDER)

            # Check if array contains already the same image in different format (masked/original)
            existing_row_filter = filter(lambda item: item_contains_keys(item, Genotype=genotype, Condition=condition, File_index=index), image_data)

            existing_row = list(existing_row_filter)

            if len(existing_row) != 0:
                if image_type == 'Original':
                    existing_row[0]['Original image path'] = image_path
                if image_type == 'Masked':
                    existing_row[0]['Masked image path'] = image_path
            else:
                if image_type == "Original":
                    image_data.append({"Genotype": genotype, "Condition": condition, 'File_index': index, "Image Type": image_type, "Original image path": image_path})
                if image_type == "Masked":
                    image_data.append({"Genotype": genotype, "Condition": condition, 'File_index': index, "Image Type": image_type, "Masked image path": image_path})

    return image_data
