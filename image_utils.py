# %%

import cv2
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
from matplotlib import image
import numpy as np
import imutils

# %%

load_dotenv()

# %%

image_path = os.path.join(os.getenv("DATA_FOLDER_PATH"), "Separated_plants/Trial_02/Dataset_01/Background_included/82-12-PS_Tray_427/plant_index_4.png")

# %%

def find_minimum_bounding_box_from_masked_image(masked_image):
  print(masked_image)
  mask = masked_image != 0
  print(mask)
  plt.imshow(mask)
  plt.show()

  print(mask)

  contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  contours = imutils.grab_contours(contours)
  max_contour = max(contours, key=cv2.contourArea)
  minimum_bounding_rect = cv2.boundingRect(max_contour)


  box = cv2.BoxPoints(minimum_bounding_rect)
  box = np.int0(box)
  cv2.drawContours(masked_image, [box], 0, (0,0,255), 2)

  plt.imshow(masked_image)
  plt.show()


# %% 

image = cv2.imread(image_path, 0)
find_minimum_bounding_box_from_masked_image(image)

# %%
