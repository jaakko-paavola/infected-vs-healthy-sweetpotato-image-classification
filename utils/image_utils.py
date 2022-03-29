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
import math
from typing import Tuple, Type, Union
# %%

load_dotenv()
DATA_FOLDER = os.getenv("DATA_FOLDER_PATH")

# %%

image_path = os.path.join(os.getenv("DATA_FOLDER_PATH"), "Separated_plants/Trial_02/Dataset_01/Background_included/82-12-PS_Tray_427/plant_index_4.png")

# %%

def find_minimum_bounding_box_from_masked_image(masked_image: np.ndarray) -> Tuple[int, int, int, int]:
  bw_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
  mask = bw_image != 0

  rows, columns = np.nonzero(mask)
  y_min, y_max = rows.min(), rows.max()
  x_min, x_max = columns.min(), columns.max()
  height = abs(y_max - y_min)
  width = abs(x_max - x_min)

  # To draw the bounding box
  #cv2.rectangle(masked_image,(x_min,y_min),(x_max,y_max),(0,255,0),2)

  return x_min, y_min, x_max, y_max

# %%

# Bbox in format (x_min, y_min, x_max, y_max)
def crop_image_with_bounding_box(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
  cropped_image = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
  return cropped_image


# %%

def pad_image_to_minimum_size(image, size: Union[Tuple, int] = 224):
  """
    Size: Desired output size. If size is a tuple (h, w), pad image to match the tuple. If size is an integer,
    smaller edge will be matched to this.
  """

  height, width = image.shape[0], image.shape[1]

  if isinstance(size, tuple):
    target_height = size[0]
    target_width = size[1]

    # Prevent negative padding with max(x, 0)
    padding_vertical = max(target_height - height, 0)
    padding_horizontal = max(target_width - width, 0)

  elif isinstance(size, int):
    if height > width:
      height_width_ratio = height / width
      padding_horizontal = max(size - width, 0)
      padding_vertical = max(math.ceil(padding_horizontal * height_width_ratio), 0)
    else:
      width_height_ratio = width / height
      padding_vertical = max(size - height, 0)
      padding_horizontal = max(math.ceil(padding_vertical * width_height_ratio), 0)
  else:
    raise TypeError("Parameter size must be either tuple or integer")


  top = math.ceil(padding_vertical / 2)
  bottom = top
  left = math.ceil(padding_horizontal / 2)
  right = left

  BLACK = [0, 0, 0]
  padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)

  return padded_image

# %% 

""""
How to crop and pad an image:
  image = io.imread(image_path)
  bbox = find_minimum_bounding_box_from_masked_image(image)
  cropped_image = crop_image_with_bounding_box(image, bbox)
  padded_image = pad_image_to_minimum_size(cropped_image, (200, 500))
  plt.imshow(padded_image)
  plt.show()
"""

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

def img_to_patch(x, patch_size, flatten_channels=True):
    """from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial15/Vision_Transformer.html
    Inputs:
        x - torch.Tensor representing the image of shape [B, C, H, W]
        patch_size - Number of pixels per dimension of the patches (integer)
        flatten_channels - If True, the patches will be returned in a flattened format
                           as a feature vector instead of a image grid.
    """
    B, C, H, W = x.shape
    x = x.reshape(B, C, H//patch_size, patch_size, W//patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5) # [B, H', W', C, p_H, p_W]
    x = x.flatten(1,2)              # [B, H'*W', C, p_H, p_W]
    if flatten_channels:
        x = x.flatten(2,4)          # [B, H'*W', C*p_H*p_W]
    # print(x.shape)
    return x
