#%%
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

#%%

image_path = "../data/Data/Trial_02/Dataset_02/01-Hua-H/Mask Img/180813 - 01- Hua-H-10.1 - Mask.png"

img_orig = cv2.imread(image_path)
img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
plt.imshow(img)

#%%
def find_contours(img):
    # img channels assumed to be RGB
    img_bw = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(img_bw, 0, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours


def mask_leaf_parts(img):
    contours = find_contours(img)

    # longest contour usually corresponds to the whole leaf (not necessarily always)
    i = np.argmax([len(c) for c in contours])
    leaf_contour = contours[i]    

    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(mask, pts=[leaf_contour], color=(255, 255, 255))

    masked = cv2.bitwise_and(img, img, mask=mask)
    
    return masked

#%% find bounding rectangles per leaf, save them as individual images
contours = find_contours(img)

for i, c in enumerate(contours):
    x,y,w,h = cv2.boundingRect(c)

    ROI = img[y:y+h, x:x+w]
    masked = mask_leaf_parts(ROI)

    if not os.path.exists('separated'):
        os.makedirs('separated')

    cv2.imwrite('separated/ROI_{}.png'.format(i), ROI)
    cv2.imwrite('separated/ROI_{}_masked.png'.format(i), masked)