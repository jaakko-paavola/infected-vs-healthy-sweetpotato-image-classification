# %%
from math import dist
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random

# %%

load_dotenv()

DATA_FOLDER_PATH = os.getenv("DATA_FOLDER_PATH")
PLANT_VILLAGE_DATA_PATH = os.path.join(DATA_FOLDER_PATH, "Dummy_Data")
PLANT_VILLAGE_DATA_PATH_DF = os.path.join(DATA_FOLDER_PATH, "dummy_segmented_plant_village_data.csv")

# %%

plant_village_df = pd.read_csv(PLANT_VILLAGE_DATA_PATH_DF)

train_dataset, test_dataset = train_test_split(plant_village_df, test_size=0.15)

# %%

# classify using train set's majority class

majority_class = train_dataset['label_name'].mode()[0]

test_dataset['prediction'] = majority_class

correct_count = len([row for index, row in test_dataset.iterrows() if row['label_name'] == row['prediction']])

print('accuracy with majority class: {0}'.format(correct_count/len(test_dataset)))

# %%

# classify using train set's label distribution

labels = train_dataset['label_name'].unique()
distribution = []
for label in labels:
    percentage = len(train_dataset[train_dataset['label_name'] == label])/len(train_dataset)
    distribution.append(percentage)

for index, row in test_dataset.iterrows():
    random_label = random.choices(labels, distribution)
    test_dataset.at[index, 'prediction'] = random_label[0]

correct_count = len([row for index, row in test_dataset.iterrows() if row['label_name'] == row['prediction']])

print('accuracy with majority class: {0}'.format(correct_count/len(test_dataset)))
