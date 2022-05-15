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
plant_df_split_master_path = os.path.join(DATA_FOLDER, "plant_data_split_master.csv")
leaf_df_split_master_path = os.path.join(DATA_FOLDER, "leaves_segmented_master.csv")
plant_golden_df_path = os.path.join(DATA_FOLDER, "plant_data_split_golden.csv")

# %%
leaf_df = pd.read_csv(leaf_df_path)
plant_df = pd.read_csv(plant_df_path)
growth_chamber_plant_df = pd.read_csv(growth_chamber_plant_df_path)
plant_df_split_master = pd.read_csv(plant_df_split_master_path)
leaf_df_split_master = pd.read_csv(leaf_df_split_master_path)
plant_df_split_golden = pd.read_csv(plant_golden_df_path)

# %%
def parse_img_path(path):
    return os.path.join(DATA_FOLDER, path)


# %%

image_path = os.path.join(DATA_FOLDER, plant_df.sample(1)['Masked image path'].item())

image = cv2.imread(image_path)

plt.imshow(image)
plt.show()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.imshow(gray)
plt.show()

thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]

plt.imshow(thresh)
plt.show()


closed_gaps_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50,50)))

plt.imshow(closed_gaps_thresh)
plt.show()

cnts, _ = cv2.findContours(closed_gaps_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(image, cnts, -1, (0, 255, 0), 3)

plt.imshow(image)
plt.show()

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
plant_df_split_master['Label Frequency'] = 1 
label_frequency = plant_df_split_master.groupby(["Condition"]).sum()[["Label Frequency"]]
label_frequency = label_frequency.reset_index()
fg = sns.factorplot(x='Condition', y='Label Frequency', data=label_frequency, kind='bar')
fg.set_xlabels('')
# %%

print(plant_df_split_master['Label'].value_counts())
print(plant_df_split_master['Label'].value_counts(normalize=True))

# %% 

print(len(plant_df_split_master))

# %% 

leaf_df_split_master['Label Frequency'] = 1 
label_frequency = leaf_df_split_master.groupby(["Condition"]).sum()[["Label Frequency"]]
label_frequency = label_frequency.reset_index()
fg = sns.factorplot(x='Condition', y='Label Frequency', data=label_frequency, kind='bar')
fg.set_xlabels('')
# %%

print(len(leaf_df_split_master))
print(leaf_df_split_master['Label'].value_counts())
print(leaf_df_split_master['Label'].value_counts(normalize=True))

# %%

plant_df_split_golden['Label Frequency'] = 1 
label_frequency = plant_df_split_golden.groupby(["Condition"]).sum()[["Label Frequency"]]
label_frequency = label_frequency.reset_index()
fg = sns.factorplot(x='Condition', y='Label Frequency', data=label_frequency, kind='bar')
fg.set_xlabels('')
# %%

print(len(plant_df_split_golden))
print(plant_df_split_golden['Label'].value_counts(normalize=True))
print(plant_df_split_golden['Label'].value_counts(normalize=True))

# %%
