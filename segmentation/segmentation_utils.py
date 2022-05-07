import os
import re

def get_masked_plant_filename(masked_image_path, output_path, plant_index: int = None):

  if not plant_index:
    plant_index = "?"
    
  masked_filename = re.findall(r'[^\/]+(?=\.)', masked_image_path)[0]
  masked_filetype = os.path.splitext(masked_image_path)[1]

  pathname = os.path.join(output_path, masked_filename)
  
  filepath = os.path.join(pathname, f"{masked_filename}_M{plant_index}{masked_filetype}")
  
  return filepath

def get_original_plant_filename(original_image_path, output_path, plant_index: int = None):
  
  if not plant_index:
    plant_index = "?"
  
  original_filename = re.findall(r'[^\/]+(?=\.)', original_image_path)[0]
  original_filetype = os.path.splitext(original_image_path)[1]

  pathname = os.path.join(output_path, original_filename)
  
  filepath = os.path.join(pathname, f"{original_filename}_O{plant_index}{original_filetype}")
  
  return filepath
