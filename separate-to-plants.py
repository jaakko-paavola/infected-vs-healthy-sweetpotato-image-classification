# %%
from PIL import Image
import numpy as np
from skimage import data, io, filters, segmentation, measure
import cv2
import matplotlib.pyplot as plt
import matplotlib
import glob
import re

# %%
pattern_tray_2x3_and_3_2_3_2 = r".*78-(.).*(Tray_.*?)-"
pattern_tray_4x5 = r".*77-(.).*(Tray_.*?)-"
for idx1, filename in enumerate(glob.glob('/home/jaakkpaa/Dropbox/lx8-fuxi104/Documents/Opiskelu/Data science project I/2022_Data Science Project/Data/Trial_01_dataset_01_fisheyemasked_separated/4x5/*.png')):
    img = cv2.imread(filename)
    tray_id = re.match(pattern_tray_4x5, filename).group(2)
    stage = re.match(pattern_tray_4x5, filename).group(1)

    # cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = find_contours(img)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # Find bounding box and extract ROI
    # c = cnts[0]
    # idx = 0
    for idx2, c in enumerate(cnts):
        if(cv2.contourArea(c) < 10000):
            continue
        x,y,w,h = cv2.boundingRect(c)
        ROI = img[y:y+h, x:x+w]
        masked = mask_plant_parts(ROI)
        result = masked.copy()

        # Make the background transparent (or comment out the following block to leave it black)
        gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        kernel = np.ones((9,9), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
        result[:, :, 3] = mask

        cv2.imwrite(f'/home/jaakkpaa/Documents/Source/Infected-sweetpotato-classification/Trial_1/4x5/{tray_id}_stage_{stage}_the_{idx2+1}_biggest_object.png',result)

#%%
def find_contours(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # threshold input image using otsu thresholding as mask and refine with morphology
    ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((9,9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # put mask into alpha channel of result
    result = img.copy()
    result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = mask
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    return cnts

# %%
def mask_plant_parts(img):
    contours = find_contours(img)
    # longest contour usually corresponds to the whole leaf (not necessarily always)
    i = np.argmax([len(c) for c in contours])
    plant_contour = contours[i]
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(mask, pts=[plant_contour], color=(255, 255, 255))
    masked = cv2.bitwise_and(img, img, mask=mask)
    return masked