# %%
from cv2 import COLOR_BGR2RGB
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import pandas as pd
from dotenv import load_dotenv
import logging
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import imutils
import cv2

logger = logging.getLogger(__name__)

# %%

load_dotenv()

DATA_FOLDER = os.getenv("DATA_FOLDER_PATH")

# %%

plant_df = pd.read_csv(os.path.join(DATA_FOLDER, "plant_data.csv"))
leaf_df = pd.read_csv(os.path.join(DATA_FOLDER, "leaf_data.csv"))
growth_chamber_plant_df = pd.read_csv(os.path.join(DATA_FOLDER, "growth_chamber_plant_data.csv"))
plant_master_df = pd.concat([plant_df, growth_chamber_plant_df])

# %%

def parse_img_path(path):
    return os.path.join(DATA_FOLDER, path)

# %%

# Image in OpenCV RGB-format
def kmeans_segmentation(image, k, attempts=10, plot=False):

  logger.info("Startig k-means segmentation")
  vectorized_image = image.reshape((-1,3))
  vectorized_image = np.float32(vectorized_image)
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.9)

  ret, labels, centers = cv2.kmeans(vectorized_image, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)

  centers_int = np.uint8(centers)
  segmented_data = centers_int[labels.flatten()]
  
  # Reshape data into the original image dimensions
  segmented_image = segmented_data.reshape((image.shape))

  if plot:
    plt.imshow(segmented_image)

  return segmented_image, ret, labels, centers

# %%

def watershed_segmentation(image, plot=False):
  shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)
  gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)

  thresh = cv2.threshold(gray, 0, 255,
    cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

  D = ndimage.distance_transform_edt(thresh)

  localMax = peak_local_max(D, indices=False, min_distance=20,
    labels=thresh)

  markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
  labels = watershed(-D, markers, mask=thresh)
  print(f"{len(np.unique(labels)) - 1} unique segments found")

  for label in np.unique(labels):
    # if the label is zero, we are examining the 'background'
    # so simply ignore it
    if label == 0:
      continue

    # otherwise, allocate memory for the label region and draw
    # it on the mask
    mask = np.zeros(gray.shape, dtype="uint8")
    mask[labels == label] = 255

    # detect contours in the mask and grab the largest one
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
      cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    image_copy = image.copy()

    if plot:
      cv2.drawContours(image_copy, c, -1, (0, 255, 0), 3)
      plt.imshow(image_copy)
      plt.show()

    # draw a circle enclosing the object
    ((x, y), r) = cv2.minEnclosingCircle(c)
    cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
    cv2.putText(image, "#{}".format(label), (int(x) - 10, int(y)),
      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

  return image

# %% 

for image_path in plant_master_df['Masked image path'].unique():
  
  relative_masked_image_path = image_path
  masked_image_path = parse_img_path(relative_masked_image_path)

  image = cv2.imread(masked_image_path)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  segmented_image = watershed_segmentation(image, plot=True)

  plt.imshow(segmented_image)
  plt.show()
# %%
