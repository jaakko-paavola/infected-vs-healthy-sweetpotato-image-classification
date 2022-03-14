# %%

import cv2
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
from matplotlib import image
import numpy as np
import imutils
from skimage.color import rgba2rgb
from skimage import data, io

# %%

load_dotenv()
DATA_FOLDER = os.getenv("DATA_FOLDER_PATH")

# %%

image_path = os.path.join(os.getenv("DATA_FOLDER_PATH"), "Separated_plants/Trial_02/Dataset_01/Background_included/82-12-PS_Tray_427/plant_index_4.png")

# %%

def find_minimum_bounding_box_from_masked_image(masked_image):
  print(masked_image)
  mask = masked_image != 0
  print(mask)
  plt.imshow(mask)
  plt.show()

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

def transform_alpha_to_bg(folder_path):

      rbga_image_paths = []

      for root, dirs, files in os.walk(folder_path):

        # Iterate over subfolders until we are in folder that contains image files
        if len(files) == 0:
            continue

        for file in files:
            if ".png" in file:
              file_path = os.path.join(root, file)
              img = io.imread(file_path)

              if img.shape[2] == 3:
                continue
              else:
                img_rgba = rgba2rgb(img, background=[0,0,0])
                rbga_image_paths.append(file_path)
                # io.imsave(file_path, img_rgba)
            else:
              print(f"Not image file: {file}")

# %%

def convert_color_to_another(image, color1=[255, 255, 255], color2=[0, 0, 0]):
  data = np.array(image)

  r1, g1, b1 = color1[0], color1[1], color1[2] # Original value
  r2, g2, b2 = color2[0], color2[1], color2[2] # Value that we want to replace it with

  red, green, blue = data[:,:,0], data[:,:,1], data[:,:,2]
  mask = (red == r1) & (green == g1) & (blue == b1)
  data[:,:,:3][mask] = [r2, g2, b2]

  return data
# %%
