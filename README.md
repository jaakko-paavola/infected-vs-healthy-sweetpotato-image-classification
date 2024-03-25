# Classification of sweet potato plants into healthy and infected specimens
Image processing and computer vision approaches to split/segment images of sweetpotato plants and classify them into healthy ones and ones suffering from one or two specified plant infections.

The software in this repository was used for image segmentation and classification in the following study by a research team from the National Plant Phenotyping Infrastructure (NaPPI) in the Department of Agricultural Sciences of University of Helsinki:
![image](https://github.com/jaakko-paavola/infected-vs-healthy-sweetpotato-image-classification/assets/7263106/7ea7941c-fecb-482c-8550-9eea7bee635c)

# Introduction

The production and supply of our very fundamental need, food, can be disturbed in many
ways. On top of the effects of such sudden events as wars on the global supply chain of food, 
there is the slow but inevitable effect of climate change on the world’s food supply. 
Perhaps the next major epidemic is not going to be a human viral epidemic, but a plant disease one,
akin to the potato blight in the 1800s that caused the Great Irish Famine and killed 15% of the
Irish population. At normal times, the mean losses to major food crops due to plant diseases
and pests are around one fourth globally, but during a plant epidemic the figure can rise
regionally dramatically higher. Moreover, plant diseases significantly aggravate food shortages 
in times of poor crop years, political or societal turmoil, trade or military conflicts,
and natural disasters.

## Problem statement

Traditionally, combating plant diseases has involved a lot of tedious, manual work with visual
examination of and human judgment about the condition of plants. This slow and
labor-intensive work is not feasible on a large scale or when there is a rapidly spreading
plant infection. Something faster, yet untiring in its work and accurate in its judgment, is
needed to prepare us for the next serious plant disease outbreak.

We were commissioned to develop an image classifier to detect two common viral infections
in sweetpotato plants, one of the most important staple food crops in the world, and
consequently make spotting diseases faster and less labor-intensive, yet equally or even
more reliable than human judgment. With classification, we had two goals: a binary model
to detect coinfected plants (plants with both viruses at once) from plants with other
conditions; and a multi-class model to detect all four conditions (healthy, one of the two
viruses, or coinfection).

Our client is a research team from the National Plant Phenotyping Infrastructure (NaPPI) in
the Department of Agricultural Sciences of University of Helsinki – a high-end research
laboratory studying plant pathogens, among other things. The ultimate vision of our client is
to be able to run the model through a cell phone on any photo of a sweetpotato plant taken
with the phone’s camera and receive an instant and accurate assessment about the plant’s
health. But for now the target of this project is a command-line interface application as a
proof of concept, not yet the cell phone app.

Besides building a classifier, the client also wanted us to provide a tool that does image
segmentation, or breaks down a set of images with multiple sweetpotato plants and leaves
into a set of images with only a single plant or leaf in each image. The client expressly
requested such a tool in addition to the classifier, although segmentation would anyway be a
natural step to preprocess our data for classification, since one image could contain plants
with different health conditions.

The output of our project is a command-line tool with the functionality to segment images;
train the parameters and the hyperparameters of the four implemented classifiers on the
client’s data or on a new dataset; and predict a class for an image.

We will base our evaluation of the success of the project on: 1) the performance of our
model in terms of prediction accuracy and F1 score as compared to a dummy baseline
model and a benchmark study, and 2) the feedback we receive from our client regarding,
e.g., the usability, the documentation, and the expandability of the deliverables we have
handed them.

## Data

The client provided us with 238 images of collection trays with sweetpotato plants and 181
images of trays with individual sweetpotato leaves. Each tray contains 4-20 plants or leaves.
The images were pre-processed with a masking filter so that the collection tray and the
background, i.e, everything else in the image apart from the plants and the leaves was
blacked out.

The client also provided us a spreadsheet file with additional information regarding the
plants, such as the plants’ genotypes, sizes, and most importantly, health conditions. The
other useful information that we used was the mapping of each plant into the actual images,
so that we knew which plant had which condition while segmenting the images.

## Data organization

The client gave us access to an approximately 2GB dataset in cloud storage. They had
organized plant and leaf images by trial and dataset, plant data having three distinct datasets
and leaf data having two distinct datasets. Plant data included an Excel file for each dataset
with plant information like genotype, condition, age, area etc. Leaf data had the leaf
characteristics in the file name. We wanted to preprocess the data to a unified CSV file that
could be easily loaded as Pandas dataframe and be used in PyTorch’s DataLoader module.
We wrote scripts to traverse all dataset paths and gather the information from Excel files
and image folders and store all the information in a CSV file, one for plant and one for leaf
data, that contained all the information needed for data loading. Later, we also created a
curated plant dataset that contained only high quality images, which were larger than 50
000 pixels in total, meaning image squared is larger than 225x225 pixels.
We wanted to use version control with the data, so that we could all access easily each
other's changes to the data and wouldn’t need to worry about maximum dataset or file size.
We decided to use DVC, which is a tool similar to git: it is possible to commit, add, push and
pull datasets. We configured DVC to store all the data in Google Cloud storage.

# Implementation

## Image segmentation

The original images contain multiple plants or leaves, and our goal was to convert them to a
dataset with images containing a single plant or a leaf. In short, we used the OpenCV library
to detect contours for each plant and leaf, and then to find their bounding boxes according
to the contours. The illustrates parts of the segmentation process. In more detail, each masked 
image (top left) was first converted to grayscale (top right), and then thresholded so that 
plants and leaves are white and the background remains black (middle left). Some plants appeared 
as several parts, and would thus correspond to multiple contours. To prevent this, we used 
morphological closing to fill in the space between nearby leaves belonging to the same plant
(middle right). This ensured most plants would correspond to a single contour. For leaf images,
this was not necessary.

![image](https://github.com/jaakko-paavola/infected-vs-healthy-sweetpotato-image-classification/assets/7263106/8fd6c0b6-1230-48d0-9aa9-d672e9ca77d0)

After closing the gaps, we detected contours for each plant and leaf (bottom left). We only
kept contours with a large enough area, since short contours typically corresponded to
something else than plants or leaves: measuring sticks, holes in leaves, or text in the image.
This resulted in very small leaves being left out of the data set, but they were few. For the
rest of the contours, we found bounding boxes, and segmented the images accordingly
(bottom right). For each masked image with a single plant or leaf, we remasked the
background in order to remove parts of nearby leaves.

For images which were not originally masked, we used the contours and bounding boxes
found from the corresponding masked images. These images often contained parts of
nearby leaves.

The biggest problem we encountered in segmentation was with separating overlapping
plants and leaves. Some of this work needed to be done by hand, since the above method
would find a single contour for multiple overlapping plants.

## Data ingestion and transformation pipeline

For loading our dataset into tensor objects, we used PyTorch’s DataLoader module. We had
only a few thousand images of plants and leaves, so we decided to apply image
augmentation to increase our dataset size: we added random rotation and a small random
horizontal and vertical shift. We also normalized across each color channel to make the
distribution of pixel values follow the standard Gaussian distribution. In addition, for each
image, we also tried to add Gaussian noise to normalized images to improve the robustness
of the models [3, chapter 7].

## Selecting the approach: model architectures

We started the classification task by reading existing literature on plant disease classification.
We suspected that convolutional neural networks (CNNs) would be a common choice, but
we wanted the existing research to guide our initial efforts. We found two interesting
papers: from Mercurio and Hernandez about sweetpotato classification, and from Mohanty,
Hughes and Salathé about cassava plant classification [1, 2]. Both papers used CNNs
(Inception V3 and ResNet architectures) so we were quite confident that implementing those
models first would be a wise choice, and hoped to achieve similar results. We decided to
implement the Inception V3 and ResNet model architectures using PyTorch by ourselves (as
opposed to loading them off-the-shelf through a library), so that we could change the size of
the network, add regularization layers if needed, and because it was a useful learning
experience.

We also implemented a bag-of-visual-words (BOVW) model using more traditional learning
algorithms: kernel machines like the support vector machine (SVM), gradient boosting
techniques like XGBoost (XGB) and decision trees like random forest (RF). We also
implemented a Vision Transformer (ViT) model, which has reached many current record
results in image classification, although we acknowledged that the datasets used to train ViT
models were usually much larger than ours.

For BOVW, we selected two possible feature detection algorithms, SIFT and ORB, that
seemed to function significantly differently from each other, wanting to see which one
would fit our data better. These two seemed prominent according to a comparison [3] of
multiple different feature detection algorithms. They were both also available out-of-the-box
in OpenCV. The biggest differences between SIFT and ORB are that SIFT is scale-invariant,
meaning that the size of the object in the image does not matter, and SIFT detects more key
points on average per image. ORB on the other hand runs a lot faster, can handle colored
images (images needed to be converted to greyscale for SIFT), and ORB also tracks more
features per image.

## Hyperparameter search and model selection

We first knocked together a passably workable code to train the four model architectures,
and then focused on optimizing the models and the training process. We treated the
optimizer algorithm, the parameters of each optimizer, some model-specific parameters, and
some training process related parameters as the hyperparameters to optimize.

The optimizer algorithms we chose were the stochastic gradient descent (SGD), adaptive
moment estimation (Adam), Adam with improved weight decay (AdamW), adaptive gradient
algorithm (Adagrad), and root mean squared propagation (RMSProp). The mutual
hyperparameters between all of them are the learning rate and the weight decay. They each
have their distinctive set of other hyperparameters, some of them in common with another
optimizer, some unique to that particular optimizer. The only training process related
hyperparameter we had was the number of epochs. The BOVW classifiers had many
model-specific hyperparameters, while the NN’s had none, apart from the huge number of
network design choices that could have been treated as hyperparameters, but of these we
chose to hyperparameterize only two design aspects of the Vision Transformer architecture
– the number of heads and the dropout layer value.

The aim of the hyperparameter search or tuning process is to find the model
hyperparameter configuration (henceforth, just “model”) that yields the best performance.
In hyperparameter search, the specified hyperparameters are assigned different values
systematically and training performed with each ensuing model iteratively in trials. A trial
consists of a training and validation loop. It is noteworthy that when many hyperparameter
search trials are made, an occasional hyperparameter combination is bound to fit the
validation split exceptionally well just by sheer chance (i.e. it does not capture the real
dynamics of what we are modeling), and that model gets thus selected under false
pretenses. Instead of remediating this with cross-validation with its cost of added
complexity, we accept an occasional falsely selected model from the hyperparameter search,
since we can later catch and reject it in a final evaluation step, the test step.

Not only do we split our data into training and validation, but also into a hold-out test split.
The test step is not part of the hyperparameter search process, but we must do the split
already there, and then save the information about the test dataset for later use in the
following step, which trains the selected model and tests it. This way we can make sure that
the test split is data that the model has not been exposed to, and can thus give us an
unbiased assessment about the model’s generalizability.

The libraries we used for the NN hyperparameter search were the Optuna hyperparameter
search library together with an early stopping implementation [4] that skips the rest of a trial
after a specified number of epochs have consecutively failed to improve the validation loss.

The implementation of the hyperparameter search for the BOVW was somewhat different
from that of the NN’s, although the fundamental logic is naturally the same in all
hyperparameter tuning. With BOVW, we implemented a grid search on a parameter grid, a
grid of specified hyperparameters and their possible values. The hyperparameters in the grid
were: 1) the feature detection algorithm (SIFT or ORB), 2) the classifier (SVM, RF or XGB),
and 3) classifier-specific hyperparameters.

The options for the classifier – SVM, RF, and XGB – were the most prominent ones in
multiple online guides we followed on implementing a BOVW model. Additionally, we had
previous positive experiences with these models. After an exhaustive online research, we
selected a few classifier-specific hyperparameters that seemed to have the most impact on
these classifiers’ performance. We then formed a grid with an educated guess about
reasonable values for these hyperparameters, and trained the model using all combinations
of the hyperparameter values for each classifier in both the binary and the multiclass case.

## Training and evaluation of selected models

The output from the hyperparameter search is a ranking of the models in order of
performance by their validation loss. The hyperparameters of the top-ranked models are
saved in a file together with plots illustrating the development of the model’s validation loss,
accuracy, and F1 score epoch-by-epoch. The user can assess the performance of the
top-ranked models from these plots and choose one whose hyperparameters to copy-paste
to a hyperparameter configuration file in the working directory, under a section in the file
listing that particular model architecture’s hyperparameter values.

Next, the user executes another script for training the given architecture with the specific
hyperparameter values read from the configuration file. Now the script also reads from the
disk the information about which examples the test split created in the hyperparameter
search consisted of, reserves those examples for the test step, and uses the rest of the data
for training, as no data is needed for validation at this point anymore.

If the test step gives much poorer results than those achieved in the validation step of
hyperparameter search, it is either an indication that something in the training and
validation process or the data splits is not right, or that the model yielded good validation
results by sheer chance. One must then reject the model and test a different one with the
next best validation results. If none of the models give good results in testing, it would hint
that one has to make more fundamental changes in the model or even reject the whole
approach, like change to a different learning model or architecture altogether, and try again.

## Prediction with the most viable model

The final prediction of the health status from an image is obtained using a prediction script,
for which the user specifies which model architecture (ResNet, Inception V3, ViT or BOVW)
to use, whether they want a binary or multiclass classification, whether the image they want
to classify is an image of a plant or a leaf, and the path of that image. The prediction is done
using a pre-trained model, which can either be a model we have trained during the project,
or a new model that the client has trained themself using the training script.

The prediction script consists of two steps. First, the image is loaded and preprocessed. The
color channels of the image are reordered, as OpenCV assumes that the color channels are
in order of blue-green-red instead of the conventional red-green-blue, which is also used in
our client’s images. Then the image is resized to fit the expected image size of the NN
classifiers.

After preprocessing the image, the script loads a pre-trained model from the stored models.
In addition to specifying which architecture to use, the user can specify whether they want
to use a specific model for that architecture by providing a unique identifier associated with
each trained model, in which case that model will be loaded. Otherwise, the model from the
given architecture with the highest F1-score will be loaded. Then the actual classification is
done using the loaded model, and lastly the script outputs the probabilities for each
predicted class, e.g., “{CSV: 0.1, FMV: 0.05, Healthy: 0.8, VD: 0.05}”. The class with the
highest probability is the predicted class, which would in case of the example be “Healthy”.

Prediction is rather simple when done with any of our NN solutions: the loaded model can
immediately output the probabilities using the PyTorch-library’s internal functions. The
BOVW model requires a bit more work; it uses a ready-made predict-function from a
classifier from scikit-learn. The choice of the classifier depends on the hyperparameter
values that were optimized during the training of the model; the classifier to be used is one
of the hyperparameters. However, before the predict-function can be used, we need to
detect features from the input image. The prediction script uses the same feature detection
algorithm that was used for training the loaded model. The detected features are then used
for the classification.

# Results

The table contains the best accuracies which we have obtained with binary and
multiclass classification for plants and leafs, with each classifier. We are glad to see that the
best accuracies well exceed the majority baseline, even with our relatively small dataset. In
the binary classification for plants, ResNet18 performs the best with 97% accuracy, and in
multiclass classification, Inception V3 is the best with 83% accuracy. With leaves, ResNet18
with 95% accuracy is the best model in binary classification, and ResNet18 with 82%
accuracy is the best model also in the multiclass classification. The results for plants have
been obtained after hyperparameter tuning, but the results for leaves with default
hyperparameters, due to time constraints.

![image](https://github.com/jaakko-paavola/infected-vs-healthy-sweetpotato-image-classification/assets/7263106/6348a7a5-da5b-466c-963c-b629c917b926)

Expectedly, the binary accuracies are higher than the multiclass accuracies, since the
coinfected plants and leaves were quite easy to detect with the human eye, compared to the
individual viruses.

# Conclusion

We have built a command-line tool for classifying images of sweetpotato plants and leaves.
The tool can be used to segment images with multiple plants and leaves to a set of images
with a single plant or a leaf. We implemented four classifiers, and the tool has options for
training their parameters and tuning their hyperparameters either with the provided
dataset, or with a brand new dataset, as well as for predicting a label for an image.

The best results are significantly better than the dummy baselines. We were glad to get good
results with both the state-of-the-art neural network architectures and a more traditional
approach, bag of visual words. With more time, we would search for more hyperparameter
combinations in order to further increase the prediction accuracies.

Most of the image segmentation is automated, except with images containing overlapping
plants and leaves. If we had more time, we would build a neural network for detecting
plants and leaves from images.

To further improve the product, we would also add functionality to predict labels for
multiple images at once, so that the user could, e.g., provide a csv file with paths to the
images to be classified. The prediction script would be run for each image and the results
saved in another csv file instead of printing them out for each individual image.

For the scope of the project, a proof-of-concept command-line tool was a suitable choice,
but for follow-up work, a natural next step would be to start developing a phone application.
It would also be beneficial to improve the productization of our work. We included some
images of the results and process in the appendix.

# How to use this software?

For non-technical users we describe the full instructions in the [Wiki](https://github.com/Jakoviz/Infected-sweetpotato-classification/wiki). Please read the instructions carefully.

## Developer setup

### Data Version Control (DVC) usage

##### Install and configure DVC

1. Install DVC if you don't have it yet, e.g. with `pip install dvc` (see all installation methods [here](https://dvc.org/doc/install))
2. Install the Google Cloud Storage support to dvc with `pip install 'dvc[gs]'`
3. `git config merge.dvc.name 'DVC merge driver'`
4. `git config merge.dvc.driver 'dvc git-hook merge-driver --ancestor %O --our %A --their %B'`

Steps 3 and 4 enable DVC to automatically resolve Git conflicts (see more [here](https://dvc.org/doc/user-guide/how-to/merge-conflicts#append-only-directories))

##### Install and configure Google Cloud CLI

* Google Cloud CLI is needed for authentication during `dvc pull` and `dvc push`

1. Install gcloud CLI if you don't have it yet, e.g. with `snap install google-cloud-cli --classic` (see all installation methods [here](https://cloud.google.com/sdk/docs/install))
2. Initialize gcloud CLI with `gcloud init`
    - use the Google account that was invited to the *Sweet Potato* Google Cloud project
    - select the project `apt-hold-340700`
3. Run `gcloud beta auth application-default login` to authorize Google Auth library (DVC uses this library for authentication during push and pull)

#### Continuous usage

* `dvc pull` for getting latest changes
* `dvc add `*`local_data_directory`* to add **and** commit local changes to the data (*local_data_directory* is most likely named *data* if you pulled from Google Cloud)
* `dvc push` to push your changes to Google Cloud
* `git add data.dvc` to update the dvc file in GitHub

### Python environment

#### Setup site-packages

1. Create a virtual environment named venv with `python3 -m venv venv`
2. Active virtual environment with `source venv/bin/activate`
3. Install requirements from requirements.txt with `pip3 install -r requirements.txt`
4. After installing a new site-package, update the `requirements.txt` by running `pip3 freeze > requirements.txt`

#### Setup local environment variables

1. Create file `.env` on the folder root (`.env`-files are gitignored)
2. Add the following content to the file and fill out the variables:

```bash
DATA_FOLDER_PATH="/path/to/the/dvc/folder/root"
```

`DATA_FOLDER_PATH` is the path to the DVC `data` folder. Each image path is then relative to the `DATA_FOLDER`, something like `DATA_FOLDER_PATH + "Data/Trial_01/Dataset_01/...`

### Reading image metadata as DataFrame with Pandas

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

# References
[1]: Mercurio, Dexter & Hernandez, Alexander. (2019). Classification of Sweet Potato Variety
using Convolutional Neural Network. pp. 120-125. doi: 10.1109/ICSEngT.2019.8906329.

[2]: Mohanty, Sharada P. & Hughes, David P. & Salathé, Marcel. (2016). Using Deep Learning
for Image-Based Plant Disease Detection. 7. Frontiers in Plant Science. doi:
10.3389/fpls.2016.01419

[3] “Comparison of the OpenCV's feature detection algorithms – II.” Computer Vision Talks,
https://computer-vision-talks.com/2011-07-13-comparison-of-the-opencv-feature-detection
-algorithms/. Accessed 14 May 2022.

[4] “Bjarten/early-stopping-pytorch: Early stopping for PyTorch.” GitHub,
https://github.com/Bjarten/early-stopping-pytorch. Accessed 14 May 2022.

[5]: Goodfellow, Ian. & Bengio, Yoshua & Courville, Aaron (2016). Deep Learning. MIT Press

[6]: Jiang, Peng-Tao & Zhang, Chang-Bin & Hou, Qibin & Cheng, Ming-Ming & Wei, Yunchao.
(2021). LayerCAM: Exploring Hierarchical Class Activation Maps for Localization. In IEEE
Transactions on Image Process
