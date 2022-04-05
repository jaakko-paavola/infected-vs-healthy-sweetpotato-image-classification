from operator import mod
from dotenv import load_dotenv
import os
import time
from time_utils import now_to_str, str_to_datetime
from torch import nn
from pathlib import Path
import pandas as pd
import random
import string
import json
from typing import Union
from datetime import datetime
from models.model_factory import AVAILABLE_MODELS

load_dotenv()

DATA_FOLDER = os.getenv("DATA_FOLDER_PATH")
MODEL_FOLDER = os.path.join(DATA_FOLDER, "models")

MODEL_DF = pd.read_csv(os.path.join(DATA_FOLDER, "models.csv"))

CLASSS_TO_MODEL_NAME_MAPPING = {
	"ResNet": "resnet18",
	"Inception3": "inception_v3",
	"VisionTransformer": "vision_transformer"
}

def get_model_file_name(id: str, model_name: str, timestamp: str):
		model_file_name = f"{id}-{model_name}-{timestamp}.pt"
		return model_file_name


def create_model_file_name(model_name: str):
		if type(model_name) != str:
				raise ValueError("Model name must be string")

		id = "".join(
				random.choice(string.ascii_lowercase + string.digits) for i in range(8)
		)
		timestamp = now_to_str()
		return get_model_file_name(id, model_name, timestamp)


def split_model_file_name(model_file_string: str) -> Union[str, str, datetime]:
		if model_file_string.count("-") != 2:
				raise ValueError(
						"model_file_string be in format <id>-<model_name>-<timestamp without hyphen>.<extension>"
				)

		path = Path(model_file_string)

		# Remove suffix
		path = path.with_suffix("")

		id, model_name, datetime_str = str(path).split("-")
		datetime = str_to_datetime(datetime_str)

		return id, model_name, datetime


def save_torch_model(model: nn.Module) -> Union[str, str, datetime]:
		# If custom name attribute has been given, use it
		if hasattr(model, "name"):
				model_name = model.name
		# Default to class name
		else:
				model_name = type(model).__name__
				

		model_file_name = create_model_file_name(model_name=model_name)


def add_model_info_to_df(
		id: str,
		model_name: str,
		# training timestamp
		timestamp=datetime,
		description: str = None,
		# "plant" or "leaf"
		plant_or_leaf: str = None,
		binary_test_accuracy: float = None,
		binary_test_loss: float = None,
		binary_F1: float = None,
		multiclass_test_accuracy: float = None,
		multiclass_test_loss: float = None,
		multiclass_F1: float = None,
		other_json: json = None,
):

		if model_name not in AVAILABLE_MODELS:
				raise ValueError(
						f"Model name not recognized, available models: {AVAILABLE_MODELS}"
				)

		if not plant_or_leaf:
				raise ValueError("Need to specify if model is trained on plants or leaf images")

		if (
				not binary_test_accuracy
				and not binary_test_loss
				and not binary_F1
				and not multiclass_test_accuracy
				and not multiclass_test_loss
				and not multiclass_F1
		):
				raise ValueError("Model should be stored with some results")

		other = json.dumps(other_json)

		pandas_row = [
				id,
				model_name,
				timestamp,
				description,
				plant_or_leaf,
				binary_test_accuracy,
				binary_test_loss,
				binary_F1,
				multiclass_test_accuracy,
				multiclass_test_loss,
				multiclass_F1,
				other,
		]

		MODEL_DF.loc[len(MODEL_DF.index)] = pandas_row
		MODEL_DF.to_csv(os.path.join(DATA_FOLDER, "models.csv"))


# Helper function to store the model by just passing the model to the function and add relevant results to df
def store_model_and_add_info_to_df(model, **kwargs):
		if type(model) == nn.Module:
			id, model_name, timestamp = save_torch_model(model)
		else:
			raise NotImplementedError("Support for sklearn models has not been implemented yet")

		add_model_info_to_df(id=id, model_name=model_name, timestamp=timestamp, **kwargs)


def get_model_info(id):
		row = MODEL_DF.loc[MODEL_DF["id"] == id]
