#%%
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import re

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

#%%

def get_bounding_boxes(path_masked):
    # assumes image is masked
    img_orig = cv2.imread(path_masked)
    # plt.imshow(img_orig)
    img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    # plt.imshow(img)

    contours = find_contours(img)
    # print(len(contours))
    contours = contours[1:]

    boxes = []

    for i, c in enumerate(contours[::-1]):
        if (cv2.contourArea(c) > 10000):
            boxes.append(cv2.boundingRect(c))
    
    return boxes


def cut(img, bounding_boxes, is_masked=True):
    segments = []

    for x,y,w,h in bounding_boxes:
        img_segmented = img[y:y+h, x:x+w]
        
        if is_masked:
            img_segmented = mask_leaf_parts(img_segmented)
    
        segments.append(img_segmented)
    
    return segments
    

def write(segments, path, img_original):
    filename = re.findall(r'[^\/]+(?=\.)', path)[0]

    pathname = os.path.join("segmented_leaves", filename)

    segmented_paths = []
    
    if not os.path.exists(pathname):
        os.makedirs(pathname)
    
    # for now, save the original image in the same location as the segments, just for easy checking that the segmentation has gone right
    cv2.imwrite(os.path.join(pathname, f"{filename}.png"), img_original)

    for i, segment in enumerate(segments):
        segmented_path = os.path.join(pathname, f"{filename}_{i}.png")
        segmented_paths.append(segmented_path)
        cv2.imwrite(segmented_path, segment)
    
    return segmented_paths



def segment(path_masked, path_original):
    img_orig_masked = cv2.imread(path_masked)
    img_masked = cv2.cvtColor(img_orig_masked, cv2.COLOR_BGR2RGB)

    img_orig = cv2.imread(path_original)
    img_original = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    
    bounding_boxes = get_bounding_boxes(path_masked)

    segments_masked = cut(img_masked, bounding_boxes, is_masked=True)
    segments_original = cut(img_original, bounding_boxes, is_masked=False)

    segmented_paths_masked = write(segments_masked, path_masked, img_masked)
    segmented_paths_original = write(segments_original, path_original, img_original)

    return segmented_paths_masked, segmented_paths_original
