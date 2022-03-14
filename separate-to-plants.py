# %%
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import re
from functools import cmp_to_key
import pathlib
from dotenv import load_dotenv
import pandas as pd

# %%

load_dotenv()

# %%

DATA_FOLDER_PATH = os.getenv("DATA_FOLDER_PATH")
# %%

## Set the paths here before running
input_path_of_tray_images = f'{DATA_FOLDER_PATH}/Data/Trial_01/Dataset_01/FishEyeMasked'
output_path_for_separated_plants = f'{DATA_FOLDER_PATH}/Separated_plants/Trial_01/Dataset_01/Background_included'

# %%
# Remove little bits and pieces of other plants from this contour's bounding rectangle's corners
def mask_plant_parts(img):
    contours = find_contours(img)
    # longest contour usually corresponds to the whole leaf (not necessarily always)
    i = np.argmax([len(c) for c in contours])
    plant_contour = contours[i]
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(mask, pts=[plant_contour], color=(255, 255, 255))
    masked = cv2.bitwise_and(img, img, mask=mask)
    return masked

# %%
def contour_sort(a, b):
    global img_counter
    br_a = cv2.boundingRect(a)
    br_b = cv2.boundingRect(b)
    a_x, a_y, a_w, a_h = br_a[0], br_a[1], br_a[2], br_a[3]
    b_x, b_y, b_w, b_h = br_b[0], br_b[1], br_b[2], br_b[3]

    # Check if a and be seem to be on the same "row" on the tray or not:
    if((a_y - 100 <= b_y and a_y + a_h - 100 <= b_y) or (b_y - 100 <= a_y and b_y + b_h - 100 <= a_y)):
        return a_y - b_y # Not on the same row
    else:
        return a_x - b_x # On the same row

#%%
def find_contours(img):
    # Threshold input image using otsu thresholding as mask and refine with morphology
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
    # Use "close" morphological operation to close the gaps between contours
    # Find contours in thresh_gray after closing the gaps
    closed_gaps_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50,50)))
    cnts = cv2.findContours(closed_gaps_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    result = tuple(c for c in cnts if cv2.contourArea(c) > 500) # Filter contours smaller than 500 pixels out
    return result

# %%
regex_pattern_for_plant_info = r".*([1-9]{2})-([1-9]).*(Tray_.*?)-"
regex_pattern_for_extracting_filename_from_path = '.*\/(.*)$'

for idx1, filename in enumerate(glob.glob(f'{input_path_of_tray_images}/*.png')):
    img = cv2.imread(filename)
    file = re.search(regex_pattern_for_extracting_filename_from_path, filename).group(1)
    prefix = re.match(regex_pattern_for_plant_info, filename).group(1)
    stage = re.match(regex_pattern_for_plant_info, filename).group(2)
    tray_id = re.match(regex_pattern_for_plant_info, filename).group(3)
    subfolder_name = f'{prefix}-{stage}-PS_{tray_id}'

    if(not os.path.exists(f'{output_path_for_separated_plants}/{subfolder_name}')):
        pathlib.Path(f'{output_path_for_separated_plants}/{subfolder_name}').mkdir(parents=True, exist_ok=True)
    cv2.imwrite(f'{output_path_for_separated_plants}/{subfolder_name}/' + file, img)

    cnts = find_contours(img)
    sorted_cnts = sorted(cnts, key=cmp_to_key(contour_sort))
    for idx2, c in enumerate(sorted_cnts):
        x, y, w, h = cv2.boundingRect(c)
        ROI = img[y:y+h, x:x+w]
        masked = mask_plant_parts(ROI)
        result = masked.copy()

        ## Make the background transparent (or comment out the following block to leave the (black) background)
        # gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        # ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # kernel = np.ones((9,9), np.uint8)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
        # result[:, :, 3] = mask

        # Write the split plant in a file
        cv2.imwrite(f'{output_path_for_separated_plants}/{subfolder_name}/plant_index_{idx2+1}.png', result)
    
# %%

gc_df = pd.read_csv(os.path.join(DATA_FOLDER_PATH, 'growth_chamber_plant_data.csv'))

output_path_for_separated_plants_gc = '/Separated_plants/Trial_02/Dataset_03/Background_included'

regex_pattern_for_plant_info_gc = r"^([0-9]{6}) - ([0-9]{2}) - TV - (Hua|R3)-(H|FMV|CSV|VD) - ((?:[0-9]{2}-|)[0-9]{2}) - Mask.png$"

split_gc_df = pd.DataFrame(columns=['Trial', 'Dataset', 'Genotype', 'Condition', 'Original image path', 'Masked image path', 'Split masked image path'])

for index, row in gc_df.iterrows():
    filename = row['Masked image path']
    img = cv2.imread(os.path.join(DATA_FOLDER_PATH, filename))
    file = re.search(regex_pattern_for_extracting_filename_from_path, filename).group(1)
    prefix = re.match(regex_pattern_for_plant_info_gc, file).group(1)
    tray = re.match(regex_pattern_for_plant_info_gc, file).group(2)
    genotype = re.match(regex_pattern_for_plant_info_gc, file).group(3)
    condition = re.match(regex_pattern_for_plant_info_gc, file).group(4)
    plants = re.match(regex_pattern_for_plant_info_gc, file).group(5)
    subfolder_name = f'{prefix} - {tray} - TV - {genotype}-{condition} - {plants}'
    if(not os.path.exists(f'{DATA_FOLDER_PATH}/{output_path_for_separated_plants_gc}/{subfolder_name}')):
        pathlib.Path(f'{DATA_FOLDER_PATH}/{output_path_for_separated_plants_gc}/{subfolder_name}').mkdir(parents=True, exist_ok=True)
    cv2.imwrite(f'{DATA_FOLDER_PATH}/{output_path_for_separated_plants_gc}/{subfolder_name}/' + file, img)

    cnts = find_contours(img)
    sorted_cnts = sorted(cnts, key=cmp_to_key(contour_sort))
    for idx2, c in enumerate(sorted_cnts):
        x, y, w, h = cv2.boundingRect(c)
        ROI = img[y:y+h, x:x+w]
        masked = mask_plant_parts(ROI)
        result = masked.copy()

        # Write the split plant in a file
        split_plant_img_path = f'{output_path_for_separated_plants_gc}/{subfolder_name}/plant_index_{idx2+1}.png'
        cv2.imwrite(f'{DATA_FOLDER_PATH}/{split_plant_img_path}', result)

        # Add an entry for the split plant to the new dataframe
        split_plant_data = {
            'Trial': row['Trial'],
            'Dataset': row['Dataset'],
            'Genotype': row['Genotype'],
            'Condition': row['Condition'],
            'Original image path': row['Original image path'],
            'Masked image path': row['Masked image path'],
            'Split masked image path': split_plant_img_path,
        }
        split_gc_df = split_gc_df.append(split_plant_data, ignore_index=True)

# %%

# There were two errors that needed manual work.
# In both cases one of the plants in the group image was split into two separate images
# for a single plant, as the leaves of the plant had too much gap between them.
# Here I drop the automatically created unnecessary rows after manually merging and deleting
# the separate images for these two plants.

split_gc_df = pd.read_csv(f'{DATA_FOLDER_PATH}/growth_chamber_plant_data_split.csv')

split_gc_df[split_gc_df['Masked image path'].str.contains('180724 - 05 - TV - R3-H - 14-15 - Mask')]['Split masked image path']
split_gc_df.iloc[96]['Split masked image path']
split_gc_df.drop(96, inplace=True)
split_gc_df[split_gc_df['Masked image path'].str.contains('180724 - 05 - TV - R3-H - 14-15 - Mask')]['Split masked image path']

split_gc_df[split_gc_df['Masked image path'].str.contains('180724 - 06 - TV - R3-FMV - 11-13 - Mask')]['Split masked image path']
split_gc_df.iloc[110]['Split masked image path']
split_gc_df.drop(110, inplace=True)
split_gc_df[split_gc_df['Masked image path'].str.contains('180724 - 06 - TV - R3-FMV - 11-13 - Mask')]['Split masked image path']

# %%

split_gc_df.to_csv(f'{DATA_FOLDER_PATH}/growth_chamber_plant_data_split.csv')

# %%
