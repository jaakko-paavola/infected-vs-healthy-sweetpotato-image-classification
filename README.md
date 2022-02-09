# Infected-sweetpotato-classification
Image processing and computer vision approaches to split and classify images of infected sweetpotato plants

---

## Data Version Control (DVC) usage

### Initial setup

#### Install and configure DVC

1. Install DVC if you don't have it yet, e.g. with `pip install dvc` (see all installation methods [here](https://dvc.org/doc/install))
2. Install the Google Cloud Storage support to dvc with `pip install 'dvc[gs]'`
3. Add our Google Cloud Storage as a remote with `dvc remote add -d origin gs://sweetpotato_images/data` (note that `-d` flag makes the remote the default remote)

#### Install and configure Google Cloud CLI

* Google Cloud CLI is needed for authentication during `dvc pull` and `dvc push`

1. Install gcloud CLI if you don't have it yet, e.g. with `snap install google-cloud-cli --classic` (see all installation methods [here](https://cloud.google.com/sdk/docs/install))
2. Initialize gcloud CLI with `gcloud init`
    - use the Google account that was invited to the *Sweet Potato* Google Cloud project
    - select the project `apt-hold-340700`
3. Run `gcloud beta auth application-default login` to authorize Google Auth library (DVC uses this library for authentication during push and pull)

### Continuous usage

* `dvc pull` for getting latest changes
* `dvc add `*`local_data_directory`* to add **and** commit local changes to the data
* `dvc push` to push your changes to Google Cloud

**NB** if you didn't include the flag `-d` or `--default` when adding the remote during initial setup, you need to add `--remote `*`remote_name`* to `dvc pull` and `dvc push`.
