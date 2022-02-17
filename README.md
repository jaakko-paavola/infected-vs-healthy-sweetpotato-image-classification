# Infected-sweetpotato-classification
Image processing and computer vision approaches to split and classify images of infected sweetpotato plants

---

## Data Version Control (DVC) usage

### Initial setup

#### Install and configure DVC

1. Install DVC if you don't have it yet, e.g. with `pip install dvc` (see all installation methods [here](https://dvc.org/doc/install))
2. Install the Google Cloud Storage support to dvc with `pip install 'dvc[gs]'`

#### Install and configure Google Cloud CLI

* Google Cloud CLI is needed for authentication during `dvc pull` and `dvc push`

1. Install gcloud CLI if you don't have it yet, e.g. with `snap install google-cloud-cli --classic` (see all installation methods [here](https://cloud.google.com/sdk/docs/install))
2. Initialize gcloud CLI with `gcloud init`
    - use the Google account that was invited to the *Sweet Potato* Google Cloud project
    - select the project `apt-hold-340700`
3. Run `gcloud beta auth application-default login` to authorize Google Auth library (DVC uses this library for authentication during push and pull)

### Continuous usage

* `dvc pull` for getting latest changes
* `dvc add `*`local_data_directory`* to add **and** commit local changes to the data (*local_data_directory* is most likely named *data* if you pulled from Google Cloud)
* `dvc push` to push your changes to Google Cloud
* `git add data.dvc` to update the dvc file in GitHub

## Python environment

### Setup site-packages

1. Create a virtual environment named venv with `python3 -m venv venv`
2. Active virtual environment with `source venv/bin/activate`
3. Install requirements from requirements.txt with `pip3 install -r requirements.txt`
4. After installing a new site-package, update the `requirements.txt` by running `pip3 freeze > requirements.txt`

### Setup local environment variables

1. Create file `.env` on the folder root (`.env`-files are gitignored)
2. Add the following content to the file and fill out the variables:

```bash
DATA_FOLDER_PATH="/path/to/the/dvc/folder/root"
```

`DATA_FOLDER_PATH` is the path to the DVC `data` folder. Each image path is then relative to the `DATA_FOLDER`, something like `DATA_FOLDER_PATH + "Data/Trial_01/Dataset_01/...`

## Reading image metadata as DataFrame with Pandas

DVC root contains couple of CSV-files that contain image metadata. These files have been produced by `preprocess_plant_data.py`, `preprocess_leaf_data.py` and `preprocess_growth_chamber_plant_data.py` respectively.

`plant_data.csv` contains plant data from `Trial_01/Dataset_01` and `Trial_02/Dataset_01`

`leaf_data.csv` contains leaf data from `Trial_01/Dataset_02` and `Trial_02/Dataset_02`

`growth_chamber_plant_data.csv` contains plant data from `Trial_02/Dataset_03`

You can read CSV-file with Pandas with the following:

```python
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DATA_FOLDER = os.getenv("DATA_FOLDER_PATH")

plant_df_path = os.path.join(DATA_FOLDER, "plant_data.csv")
plant_df = pd.read_csv(plant_df_path)
```
