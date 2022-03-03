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

# %%

load_dotenv()

# %% 

DATA_FOLDER_PATH = os.getenv("DATA_FOLDER_PATH")

# %%

## Set the paths here before running
input_path_of_tray_images = f'{DATA_FOLDER_PATH}/Data/Trial_02/Dataset_01/FishEyeMasked'
output_path_for_separated_plants = f'{DATA_FOLDER_PATH}/Separated_plants/Trial_02/Dataset_01/Background_included'

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
