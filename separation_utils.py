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
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

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


# %%

def watershed_segmentation(image, plot=False):

  shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)
  gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)

  thresh = cv2.threshold(gray, 0, 255,
    cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

  kernel = np.ones((3),np.uint8)
  closing_img = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations = 4)

  dist_transform = cv2.distanceTransform(closing_img, cv2.DIST_L2, 3)

  local_max_peaks = peak_local_max(dist_transform, indices=False, min_distance=20,
    labels=thresh)

  markers = ndimage.label(local_max_peaks, structure=np.ones((3, 3)))[0]

  labels = watershed(255 - dist_transform, markers, mask=thresh)

  print(f"{len(np.unique(labels)) - 1} unique segments found")

  contours = []

  for label in np.unique(labels):
    # otherwise, allocate memory for the label region and draw
    # it on the mask
    mask = np.zeros(gray.shape, dtype="uint8")
    mask[labels == label] = 255

    # detect contours in the mask and grab the largest one
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
      cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    contours.append(c)
 
  mask = get_mask_from_contours(contours, gray)
  inv_mask = cv2.bitwise_not(mask)
  plt.imshow(inv_mask)
  plt.show()

  image_copy = image.copy()
  image_copy = cv2.bitwise_and(image_copy, image_copy, mask = inv_mask)

  return image_copy


# %%

def get_mask_from_contours(contours, image):
  mask = np.zeros(image.shape, np.uint8)
  cv2.fillPoly(mask, contours, color=(255,255,255))

  return mask

# %% 

def watershed_with_k_mean(image, n_clusters):
  shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)
  gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)

  thresh = cv2.threshold(gray, 0, 255,
    cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

  kernel = np.ones((3),np.uint8)
  closing_img = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations = 4)

  dist_transform = cv2.distanceTransform(closing_img, cv2.DIST_L2, 3)

  local_max_peaks = peak_local_max(dist_transform, indices=False, min_distance=20,
    labels=thresh)

  markers = ndimage.label(local_max_peaks, structure=np.ones((3, 3)))[0]

  watershed_labels = watershed(255 - dist_transform, markers, mask=thresh)

  contours = []

  for label in np.unique(watershed_labels):
    # otherwise, allocate memory for the label region and draw
    # it on the mask
    mask = np.zeros(gray.shape, dtype="uint8")
    mask[watershed_labels == label] = 255

    # detect contours in the mask and grab the largest one
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
      cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    contours.append(c)
 
  mask = get_mask_from_contours(contours, gray)
  inv_mask = cv2.bitwise_not(mask)

  point_map = transform_mask_to_point_map(inv_mask)

  dbscan = DBSCAN(eps=0.5, min_samples=10).fit(point_map)
  db_labels = dbscan.labels_
  print(db_labels)
  print(len(db_labels))
  
  # Number of clusters in labels, ignoring noise if present.
  n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
  n_noise = list(db_labels).count(-1)

  print(n_clusters)
  print(n_noise)

  plt.scatter(point_map[:, 0], point_map[:, 1], c=db_labels)
  plt.title(f"Estimated number of clusters: {n_clusters}")
  plt.show()

  #kmeans = KMeans(n_clusters=n_clusters, )
  #kmeans.fit(point_map)
  #labels = kmeans.predict(point_map)

  #plt.scatter(point_map[:, 0], point_map[:, 1], c=labels)
  #centers = kmeans.cluster_centers_
  #lt.scatter(centers[:, 0], centers[:, 1], c='black', alpha=0.5)



  #local_max_location = peak_local_max(dist_transform, min_distance=1, indices=True)

  #local_max_location = kmeans.cluster_centers_.copy()
  #local_max_location = local_max_location.astype(int)

  #dist_transform_copy = dist_transform.copy()
  #for i in range(local_max_location.shape[0]):
  #  cv2.circle( dist_transform_copy, (local_max_location[i][1],local_max_location[i][0]  ), 5, 255 )
  #  cv2.imshow(dist_transform_copy)

  #markers = np.zeros_like(dist_transform)
  #labels = np.arange(kmeans.n_clusters)


  #local_max_location = local_max_location.astype(int)

  #markers = np.zeros_like(dist_transform)
  #labels = np.arange(kmeans.n_clusters)
  #markers[local_max_location[:,0],local_max_location[:,1]] = labels
  #markers = markers.astype(np.int32)
  
  #markers, _ = ndimage.label(local_max_boolean)
  #segmented = watershed(255 - dist_transform, markers, mask=thresh)
  
  #print(type(segmented))
  #print(segmented)


  return inv_mask


# %%

def transform_mask_to_point_map(mask):
  assert len(np.unique(mask) == 2)

  point_map = np.flip(np.column_stack(np.where(mask > 0)), axis=1)

  return point_map

# %%

def watershed_2(image):  

  shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)

  plt.imshow(shifted)
  plt.title("Shifted")
  plt.show()


  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

  plt.imshow(thresh)
  plt.title("Threshold")
  plt.show()

  dist_transform = cv2.distanceTransform(gray, cv2.DIST_L2,3)

  plt.imshow(dist_transform)
  plt.title("distance")
  plt.show()

  local_max_location = peak_local_max(dist_transform, min_distance=1, indices=True)
  local_max_boolean = peak_local_max(dist_transform, min_distance=1, indices=False)
  markers, _ = ndimage.label(local_max_boolean)

  segmented = watershed(255 - dist_transform, markers, mask=thresh)

  return segmented


# %% 

for image_path in plant_master_df['Masked image path'].unique():
  
  n_clusters = len(plant_master_df[plant_master_df['Masked image path'] == image_path])
  print(n_clusters)

  relative_masked_image_path = image_path
  masked_image_path = parse_img_path(relative_masked_image_path)

  image = cv2.imread(masked_image_path)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  segmented_image = watershed_with_k_mean(image, n_clusters=n_clusters)

  plt.imshow(segmented_image)
  plt.title("Segmented")
  plt.show()
# %%

# %%

# %%

# %%
