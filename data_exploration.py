# %%
import pandas as pd
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import cv2

# %%
load_dotenv()

# %%
DATA_FOLDER = os.getenv("DATA_FOLDER_PATH")

# %%
leaf_df_path = os.path.join(DATA_FOLDER, "leaf_data.csv")
plant_df_path = os.path.join(DATA_FOLDER, "plant_data.csv")
growth_chamber_plant_df_path = os.path.join(DATA_FOLDER, "growth_chamber_plant_data.csv")

# %%
leaf_df = pd.read_csv(leaf_df_path)
plant_df = pd.read_csv(plant_df_path)
growth_chamber_plant_df = pd.read_csv(growth_chamber_plant_df_path)

# %%

plant_df = plant_df.drop(columns=["Unnamed: 0", "index"])

# %%

plant_df
# %%
def parse_img_path(path):
    return os.path.join(DATA_FOLDER, path)

# %%
for index, row in leaf_df.iterrows():
        
    original_image_path = mpimg.imread(parse_img_path(row['Original image path']))
    masked_image_path = mpimg.imread(parse_img_path(row['Masked image path']))
    
    f = plt.figure()
    f.add_subplot(1,2, 1)
    plt.imshow(original_image_path)
    f.add_subplot(1,2, 2)
    plt.imshow(masked_image_path)
    plt.show(block=True)

# %%

# Show only one image for each unique image, many plants have the same image path 
for image_path in plant_df['Original image path'].unique():
    row = plant_df[plant_df['Original image path'] == image_path].head(1)
    
    original_image_path = row['Original image path'].item()
    masked_image_path = row['Masked image path'].item()       

    original_image_path = mpimg.imread(parse_img_path(original_image_path))
    masked_image_path = mpimg.imread(parse_img_path(masked_image_path))

    f = plt.figure()
    f.add_subplot(1,2, 1)
    plt.imshow(original_image_path)
    f.add_subplot(1,2, 2)
    plt.imshow(masked_image_path)
    plt.show(block=True)

# %%

# Create temporary column for sum counts of each label
plant_df['Label Frequency'] = 1 
label_frequency = plant_df.groupby(["Genotype", "Condition"]).sum()[["Label Frequency"]]

# %%
label_frequency = label_frequency.reset_index()
# %%
fg = sns.factorplot(x='Condition', y='Label Frequency', 
                        col='Genotype', data=label_frequency, kind='bar')
fg.set_xlabels('')

# %%

max_rows=4
max_cols=4

fig, axes = plt.subplots(nrows=max_rows, ncols=max_cols, figsize=(20,20))
plt.subplots_adjust(wspace=0, hspace=0)

fig2, axes2 = plt.subplots(nrows=max_rows, ncols=max_cols, figsize=(20,20))
plt.subplots_adjust(wspace=0, hspace=0)

idx = 0
for _, item in plant_df.sample(max_rows*max_cols).iterrows():
    condition = item['Condition']
    masked_image_path = item['Masked image path']    
    masked_image_path = mpimg.imread(parse_img_path(masked_image_path))

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 10.0
    color = (255, 255, 255)
    thickness = 10
    xy = (masked_image_path.shape[0]//2, masked_image_path.shape[1]//2)

    masked_image_copy = masked_image_path.copy()

    masked_image_with_label = cv2.putText(masked_image_copy, condition, xy,
                   font, fontScale, color, thickness, cv2.LINE_AA)

    row = idx // max_cols
    col = idx % max_cols
    axes[row, col].axis("off")
    axes[row, col].imshow(masked_image_path, aspect="auto")

    axes2[row, col].axis("off")
    axes2[row, col].imshow(masked_image_with_label, aspect="auto")
    idx = idx + 1

plt.subplots_adjust(wspace=0, hspace=0)
plt.show(fig)
plt.show(fig2)
# %%