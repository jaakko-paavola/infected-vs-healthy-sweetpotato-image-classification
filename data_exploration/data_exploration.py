# %%
import pandas as pd
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

# %%
load_dotenv()

# %%
DATA_FOLDER = os.getenv("DATA_FOLDER_PATH")

# %%
leaf_df_path = os.path.join(DATA_FOLDER, "leaf_data.csv")
plant_df_path = os.path.join(DATA_FOLDER, "plant_data.csv")
growth_chamber_plant_df_path = os.path.join(DATA_FOLDER, "growth_chamber_plant_data.csv")
plant_df_split_master_path = os.path.join(DATA_FOLDER, "plant_data_split_master.csv")

# %%
leaf_df = pd.read_csv(leaf_df_path)
plant_df = pd.read_csv(plant_df_path)
growth_chamber_plant_df = pd.read_csv(growth_chamber_plant_df_path)
plant_df_split_master = pd.read_csv(plant_df_split_master_path)

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

plant_df_split_master['Label'].value_counts()

# %% 