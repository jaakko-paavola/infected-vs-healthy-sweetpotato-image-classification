# %%
from math import sqrt
import os
from unittest.loader import VALID_MODULE_NAME
from PIL import Image
import numpy as np
from skimage import data, io, filters, segmentation, measure
import cv2
import matplotlib.pyplot as plt
import matplotlib
import glob
import re
from functools import cmp_to_key

# %%
def mask_plant_parts(img):
    contours = find_contours2(img)
    # longest contour usually corresponds to the whole leaf (not necessarily always)
    i = np.argmax([len(c) for c in contours])
    plant_contour = contours[i]
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(mask, pts=[plant_contour], color=(255, 255, 255))
    masked = cv2.bitwise_and(img, img, mask=mask)
    return masked

#%%
def find_contours(img):
    # Threshold input image using otsu thresholding as mask and refine with morphology
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((9,9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    result = img.copy()
    result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    # Put mask into alpha channel of result
    result[:, :, 3] = mask
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    return cnts

# %%
img_counter = 0

def contour_sort(a, b):
    global img_counter
    br_a = cv2.boundingRect(a)
    br_b = cv2.boundingRect(b)
    a_x, a_y, a_w, a_h = br_a[0], br_a[1], br_a[2], br_a[3]
    b_x, b_y, b_w, b_h = br_b[0], br_b[1], br_b[2], br_b[3]

    # ROI = img[a_y:a_y+a_h, a_x:a_x+a_w]
    # img_counter += 1
    # cv2.imshow(f"a{img_counter}", ROI)

    # ROI = img[b_y:b_y+b_h, b_x:b_x+b_w]
    # cv2.imshow(f"b{img_counter}", ROI)
    # cv2.imshow(f"c{img_counter}", ROI)

    if((a_y - 100 <= b_y and a_y + a_h - 100 <= b_y) or (b_y - 100 <= a_y and b_y + b_h - 100 <= a_y)):
        return a_y - b_y # Not on the same row
    else:
        return a_x - b_x # On the same row

#%%
def find_contours2(img):
    # threshold input image using otsu thresholding as mask and refine with morphology
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
    # plt.imshow(thresh)
    # Use "close" morphological operation to close the gaps between contours
    # Find contours in thresh_gray after closing the gaps
    closed_gaps_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50,50)))
    # plt.imshow(closed_gaps_thresh)
    cnts = cv2.findContours(closed_gaps_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # Fill small contours with zero (erase small contours).
    mask = np.zeros(img.shape[:2], img.dtype)

    result = tuple(c for c in cnts if cv2.contourArea(c) > 500)
    return result

# %%

pattern_tray_2x3_and_3_2_3_2 = r".*78-(.).*(Tray_.*?)-"
pattern_tray_4x5 = r".*77-(.).*(Tray_.*?)-"

# tray_config, prefix, pattern = "2x3", "78", pattern_tray_2x3_and_3_2_3_2
# tray_config, prefix, pattern = "3+2+3+2", "78", pattern_tray_2x3_and_3_2_3_2
tray_config, prefix, pattern = "4x5", "77", pattern_tray_4x5

# Assignments for testing line-by-line:
# filename = glob.glob('/home/jaakkpaa/Dropbox/lx8-fuxi104/Documents/Opiskelu/Data science project I/2022_Data Science Project/Data/Trial_01_dataset_01_fisheyemasked_separated/4x5/*.png')[0]
# idx1 = 0
# filename = '/home/jaakkpaa/Documents/Source/Infected-sweetpotato-classification/Trial_1_in_order/78-4-PS_Tray_409/78-4-PS_Tray_409-RGB2-FishEyeMasked.png'
# for idx1, filename in enumerate(glob.glob(f'/home/jaakkpaa/Documents/Source/Infected-sweetpotato-classification/Trial_1_in_order/78-4-PS_Tray_409/78-4-PS_Tray_409-RGB2-FishEyeMasked.png')):
for idx1, filename in enumerate(glob.glob(f'/home/jaakkpaa/Dropbox/lx8-fuxi104/Documents/Opiskelu/Data science project I/2022_Data Science Project/Data/Trial_01_dataset_01_fisheyemasked_separated/{tray_config}/*.png')):
    img = cv2.imread(filename)
    file = re.search('.*\/(.*)$', filename).group(1)
    tray_id = re.match(pattern, filename).group(2)
    stage = re.match(pattern, filename).group(1)
    cnts = find_contours2(img)
    # cnts = (cnts[4], cnts[9])
    sorted_cnts = sorted(cnts, key=cmp_to_key(contour_sort))
    if(not os.path.exists(f'/home/jaakkpaa/Documents/Source/Infected-sweetpotato-classification/Trial_1_in_order/{prefix}-{stage}-PS_{tray_id}')):
        os.mkdir(f'/home/jaakkpaa/Documents/Source/Infected-sweetpotato-classification/Trial_1_in_order/{prefix}-{stage}-PS_{tray_id}')
    cv2.imwrite(f'/home/jaakkpaa/Documents/Source/Infected-sweetpotato-classification/Trial_1_in_order/{prefix}-{stage}-PS_{tray_id}/' + file, img)
    # Assignments for testing line-by-line:
    # c = cnts[4]
    # idx2 = 0
    for idx2, c in enumerate(sorted_cnts):
        x, y, w, h = cv2.boundingRect(c)
        ROI = img[y:y+h, x:x+w]
        masked = mask_plant_parts(ROI)
        # masked = ROI
        result = masked.copy()
        # plt.imshow(result)

        # Make the background transparent (or comment out the following block to leave it black)
        # gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        # ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # kernel = np.ones((9,9), np.uint8)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
        # result[:, :, 3] = mask
        cv2.imwrite(f'/home/jaakkpaa/Documents/Source/Infected-sweetpotato-classification/Trial_1_in_order/{prefix}-{stage}-PS_{tray_id}/plant_index_{idx2+1}.png', result)
    
# %%