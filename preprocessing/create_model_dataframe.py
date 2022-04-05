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
  "plant/leaf": pd.Series(dtype='str'),
  "num_classes": pd.Series(dtype='int'),
  "binary_test_accuracy": pd.Series(dtype='float'), 
  "binary_test_loss": pd.Series(dtype='float'), 
  "binary_F1": pd.Series(dtype='float'), 
  "multiclass_test_accuracy": pd.Series(dtype='float'), 
  "multiclass_test_loss": pd.Series(dtype='float'), 
  "multiclass_F1": pd.Series(dtype='float'),
  # Other should contain a json object dumped to a string
  "other_json": pd.Series(dtype=str)
}

model_df = pd.DataFrame(columns=columns)

# %%

model_df

# %%

model_df.to_csv(os.path.join(DATA_FOLDER_PATH, "models.csv"))

# %%
