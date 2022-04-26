# %%

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from dotenv import load_dotenv
import os
import glob
from PIL import Image
import cv2
import re
load_dotenv()

CURRENT_FOLDER_PATH = os.getcwd()

RESULTS_FOLDER_PATH = os.path.join(os.path.dirname(os.path.abspath(os.path.join(CURRENT_FOLDER_PATH, os.pardir))), "results")

GRAY_FOLDER_NAME = "gray_2"
HEATMAP_FOLDER_NAME = "heatmap_2"
ON_IMG_FOLDER_NAME = "on_img_2"

# %%

gray_folder = os.path.join(RESULTS_FOLDER_PATH, GRAY_FOLDER_NAME)
heatmap_folder = os.path.join(RESULTS_FOLDER_PATH, HEATMAP_FOLDER_NAME)
on_img_folder = os.path.join(RESULTS_FOLDER_PATH, ON_IMG_FOLDER_NAME)


# %%

def make_gif(frame_folder, gif_name, duration=300):
    """
    Duration: duration of each frame in milliseconds
    """
    
    paths = [path for path in sorted(glob.glob(f"{frame_folder}/*.png"), key=lambda file: int(''.join(filter(str.isdigit, file))))]
    frames = [Image.open(path) for path in paths]
    frame_one = frames[0]
    frame_one.save(os.path.join(frame_folder, f"{gif_name}.gif"), format="GIF", append_images=frames,
               save_all=True, duration=duration, loop=0)

# %%

make_gif(gray_folder, "activations_gray")
make_gif(heatmap_folder, "activations_heatmap")
make_gif(on_img_folder, "activations_on_img")
# %%

gray_images = [path for path in sorted(glob.glob(f"{gray_folder}/*.png"), key=lambda file: int(''.join(filter(str.isdigit, file))))]
heatmap_images = [path for path in sorted(glob.glob(f"{heatmap_folder}/*.png"), key=lambda file: int(''.join(filter(str.isdigit, file))))]
on_img_images = [path for path in sorted(glob.glob(f"{on_img_folder}/*.png"), key=lambda file: int(''.join(filter(str.isdigit, file))))]

# %%

# Gray images

def plot_images(image_paths):

  n_row = 2
  n_col = 5

  _, axs = plt.subplots(n_row, n_col, figsize=(12, 5))
  axs = axs.flatten()

  images = []

  for path in image_paths:
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images.append(image)

  for index, (img, ax) in enumerate(zip(images, axs)):
      layer = int(re.search('[_A-z\/0-9]*layer(\d+)', image_paths[index]).group(1)) + 1
      ax.set_axis_off()
      ax.imshow(img, )
      ax.title.set_text(f"Layer {layer}")

  plt.show()

# %%

# Gray images

plot_images(gray_images)
plot_images(heatmap_images)
plot_images(on_img_images)
# %%
