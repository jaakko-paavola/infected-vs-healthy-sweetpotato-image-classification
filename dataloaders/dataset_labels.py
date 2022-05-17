import os
from dotenv import load_dotenv
import pandas as pd
from typing import List

load_dotenv()

DATA_FOLDER_PATH = os.getenv("DATA_FOLDER_PATH")

def get_dataset_labels(datasheet_path : str = None, label_col: str = "Label", condition_col: str = 'Condition') -> List[str]:
    
  df = pd.read_csv(datasheet_path)
  
  label_names = df.sort_values([label_col])[condition_col].unique().tolist()
  
  return label_names
  
  
  
