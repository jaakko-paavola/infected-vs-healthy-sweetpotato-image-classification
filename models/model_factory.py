from torch import nn
from resnet import resnet18
from inception import inception3
from dotenv import load_dotenv
from utils.model_utils import split_model_file_name
from utils.time_utils import str_to_datetime
import os
from datetime import datetime

load_dotenv()

DATA_FOLDER = os.getenv("DATA_FOLDER_PATH")
MODEL_FOLDER = os.path.join(DATA_FOLDER, "models")

AVAILABLE_MODELS = ['resnet18', 'inception_v3', 'vision_transformer', ""]

def get_model_class(name: str, num_of_classes: int) -> nn.Module:

  if name not in AVAILABLE_MODELS:
    raise ValueError(f"Model type not supported, available models: {AVAILABLE_MODELS}")

  # Names are defined in the class constructor function in the model desclarations
  if name == 'resnet18':
    return resnet18(num_classes=num_of_classes)
  elif name == 'inception_v3':
    return inception3(num_classes=num_of_classes)


def get_trained_model(name: str, latest: bool = True, timestamp: str = None) -> nn.Module:

  if name not in AVAILABLE_MODELS:
    raise ValueError(f"Model type not supported, available models: {AVAILABLE_MODELS}")

  if not latest and not timestamp:
    raise ValueError("Either latest flag or timestamp must be passed as an argument")

  models = os.listdir(MODEL_FOLDER)

  # Filter models that don't contain the name
  filtered_models = [model for model in models if name in models]

  if len(filtered_models) == 0:
    raise ValueError(f"Could not find a model with name {name}")

  if latest:
    # datetime.min is always smaller than any other datetime
    latest_model, latest_timestamp = None, datetime.min
    for model in filtered_models:
      model_name, timestamp_str = split_model_file_name(model)
      timestamp = str_to_datetime(timestamp_str)

      if timestamp > latest_timestamp:
        latest_model = model
        latest_timestamp = timestamp

    model_path = os.path.join(MODEL_FOLDER, latest_model)

  else:

    timestamp_model = None

    for model in filtered_models:
      model_name, timestamp_str = split_model_file_name(model)
      if timestamp == timestamp_str:
        timestamp_model = model

    if not model:
      raise ValueError(f"Could not find a model with name {name} and timestamp {timestamp}")

    model_class = get_model_class(name)
    model_path = os.path.join(MODEL_FOLDER, timestamp_model)



  model = TheModelClass)
  model.load_state_dict(torch.load(PATH))
  model.eval()



  

    



    

    

