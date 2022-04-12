# %%
import json
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from datetime import datetime

# %%

load_dotenv()

# %%

DATA_FOLDER_PATH = os.getenv("DATA_FOLDER_PATH")

# %%

columns = {
  "id": pd.Series(dtype='str'), 
  "model_name": pd.Series(dtype='str'), 
  "timestamp": pd.Series(dtype='datetime64[ns]'), 
  "description": pd.Series(dtype='str'), 
  "dataset": pd.Series(dtype='str'),
  "num_classes": pd.Series(dtype='int'),
  "precision": pd.Series(dtype='float'),
  "recall": pd.Series(dtype='float'),
  "train_accuracy": pd.Series(dtype='float'),
  "train_loss": pd.Series(dtype='float'),
  "validation_accuracy": pd.Series(dtype='float'),
  "validation_loss": pd.Series(dtype='float'),
  "test_accuracy": pd.Series(dtype='float'), 
  "test_loss": pd.Series(dtype='float'),
  # F1 score should be macro weighted
  "f1_score": pd.Series(dtype='float'), 
  # Other should contain a json object dumped to a string
  "other_json": pd.Series(dtype=str)
}

model_df = pd.DataFrame(columns=columns)

# %%

model_df

# %%

model_df.to_csv(os.path.join(DATA_FOLDER_PATH, "models.csv"), index = False)

# %%
