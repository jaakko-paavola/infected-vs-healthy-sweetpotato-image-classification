from dotenv import load_dotenv
import os

load_dotenv()

DATA_FOLDER = os.getenv("DATA_FOLDER_PATH")


def get_relative_path_to_data_folder(path: str) -> str:
    relative_path = os.path.relpath(path, DATA_FOLDER)
    return relative_path
