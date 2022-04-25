# %%
import os
from sklearn.preprocessing import binarize
from torch.utils.data import DataLoader
from dataloaders.csv_data_loader import CSVDataLoader
from dataloaders.gaussian_noise import GaussianNoise
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sn
import pandas as pd
import numpy as np
import click
import statistics
from models.model_factory import get_model_class
from utils.model_utils import AVAILABLE_MODELS, store_model_and_add_info_to_df
import logging
from tqdm import tqdm
import yaml

logging.basicConfig() 
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# %%

load_dotenv()
DATA_FOLDER_PATH = os.getenv("DATA_FOLDER_PATH")

# %%

@click.command()
@click.option('-m', '--model', required=True, type=click.Choice(AVAILABLE_MODELS, case_sensitive=False), help='Model architechture.')
@click.option('-d', '--dataset', type=click.Choice(['plant', 'plant_golden', 'leaf', 'leaf_golden'], case_sensitive=False), default="plant", help='Already available dataset to use to train the model. Give either -d or -csv, not both.')
@click.option('-csv', '--data-csv', type=str, help='Full file path to dataset CSV-file created during segmentation. Give either -d or -csv, not both.')
@click.option('-b', '--binary', is_flag=True, show_default=True, default=False, help='Train binary classifier instead of multiclass classifier.')
@click.option('-p', '--params-file', type=str, default="hyperparams.yaml", help='Full file path to hyperparameter-file used during the training. File must be a YAMl file and similarly structured than hyperparams.yaml.')
@click.option('-aug', '--augmentation', is_flag=True, show_default=True, default=True, help='Use data-augmentation for the training.')
@click.option('-s', '--save', is_flag=True, show_default=True, default=True, help='Save the trained model and add information to model dataframe.')
@click.option('-v', '--verbose', is_flag=True, show_default=True, default=False, help='Print verbose logs.')
def train(model, dataset, data_csv, binary, params_file, augmentation, save, verbose):
    
    if verbose:
        logger.setLevel(logging.DEBUG)

    logger.info("Reading the data")

    if (not dataset and not data_csv) or (dataset and data_csv):
        raise ValueError("You must pass either -d (name of the available dataset) or -csv (path to data-CSV)")

    if dataset:
        if dataset == 'plant':
            DATA_MASTER_PATH = os.path.join(DATA_FOLDER_PATH, "plant_data_split_master.csv")
        elif dataset == 'leaf':
            DATA_MASTER_PATH = os.path.join(DATA_FOLDER_PATH, "leaf_data.csv")
        elif dataset == 'plant_golden':
            raise NotImplementedError("Plant golden dataset not implemented yet")
        elif dataset == 'leaf_golden':
            raise NotImplementedError("Leaf golden dataset not implemented yet")
    # TODO: give dataset name when using custom CSV for storing the model
    else:
        DATA_MASTER_PATH = data_csv


    # TODO: automatize label counting from dataframe
    if binary:
        NUM_CLASSES = 2
    else:
        NUM_CLASSES = 4

    with open(params_file, "r") as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logger.error(f"Error while reading YAML: {exc}")
            raise exc

    # hyperparameters
    N_EPOCHS = params[model]['N_EPOCHS']
    BATCH_SIZE_TRAIN = params[model]['BATCH_SIZE_TRAIN']
    BATCH_SIZE_TEST = params[model]['BATCH_SIZE_TEST']
    LR = params[model]['LR']

    if augmentation:
        data_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(50),
            transforms.RandomRotation(180),
            transforms.RandomAffine(translate=(0.1, 0.1), degrees=0),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            # Values aquired from dataloaders/plant_master_dataset_stats.py
            # TODO: automatize the mean and std calculation
            transforms.Normalize(mean=[0.09872966, 0.11726899, 0.06568969],
                                std=[0.1219357, 0.14506954, 0.08257045]),
            GaussianNoise(0., 0.1),
        ])
    else:
        data_transform = None

    plant_master_dataset = CSVDataLoader(
        csv_file=DATA_MASTER_PATH,
        root_dir=DATA_FOLDER_PATH,
        image_path_col="Split masked image path",
        label_col="Label",
        transform=data_transform
    )

    train_size = int(0.85 * len(plant_master_dataset))
    test_size = len(plant_master_dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(plant_master_dataset, [train_size, test_size])

    train_plant_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True, num_workers=0)
    test_plant_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE_TEST, shuffle=False, num_workers=0)
    # %%

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = get_model_class(model, num_of_classes=NUM_CLASSES).to(device)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.75)
    loss_function = torch.nn.CrossEntropyLoss()

    # %%

    # training

    training_losses = []
    training_accuracies = []

    logger.info("Starting training cycle")

    for epoch in tqdm(range(N_EPOCHS)):
        total_train_loss = 0
        train_correct = 0
        total = 0

        for batch_num, batch in enumerate(train_plant_dataloader):
            data, target = batch['image'].to(device), batch['label'].to(device)

            # For binary classification, transform labels to one-vs-rest
            if binary:
                target = target.eq(3).type(torch.int64)

            optimizer.zero_grad()

            output = model(data)

            train_loss = loss_function(output, target)
            train_loss.backward()
            optimizer.step()
            
            pred = output.max(1, keepdim=True)[1]

            correct = pred.eq(target.view_as(pred)).sum().item()
            train_correct += correct
            total += data.shape[0]
            total_train_loss += train_loss.item()

            if batch_num == len(train_plant_dataloader) - 1:
                logger.info('Training: Epoch %d - Batch %d/%d: Loss: %.4f | Train Acc: %.3f%% (%d/%d)' % 
                    (epoch, batch_num + 1, len(train_plant_dataloader), train_loss / (batch_num + 1), 
                    100. * train_correct / total, train_correct, total))


        # Training loss average for all batches
        training_losses.append(total_train_loss / len(train_plant_dataloader))        
        training_accuracies.append((100. * train_correct / total))

    # Calculate train loss and accuracy as an average of the last min(5, N_EPOCHS) losses or accuracies
    train_loss = statistics.mean(training_losses[min(-N_EPOCHS, -5):])
    train_accuracy = statistics.mean(training_accuracies[min(-N_EPOCHS, -5):])

    logger.info("Final training score: Loss: %.4f, Accuracy: %.3f%%" % (train_loss, train_accuracy))

    plt.plot(range(N_EPOCHS), training_losses, label = "Training loss")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss')
    plt.legend()
    plt.show()

    plt.plot(range(N_EPOCHS), training_accuracies, label = "Training accuracy")
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.show()

    # %%

    # test
    test_loss = 0
    test_correct = 0
    total = 0
    y_pred = []
    y_true = []

    logger.info("Starting testing cycle")

    with torch.no_grad():
        for batch_num, batch in enumerate(test_plant_dataloader):
            data, target = batch['image'].to(device), batch['label'].to(device)

            # For binary classification, transform labels to one-vs-rest
            if binary:
                target = target.eq(3).type(torch.int64)

            output = model(data)
            test_loss += loss_function(output, target).item()

            pred = output.max(1, keepdim=True)[1]

            correct = pred.eq(target.view_as(pred)).sum().item()
            test_correct += correct
            total += data.shape[0]

            test_loss /= len(test_plant_dataloader.dataset)

            pred_list = torch.flatten(pred).cpu().numpy()
            y_pred.extend(pred_list)
            
            target_list = target.cpu().numpy()
            y_true.extend(target_list)

    test_accuracy = 100. * test_correct / total

    logger.info("Final test score: Loss: %.4f, Accuracy: %.3f%%" % (test_loss, test_accuracy))

    if binary:
        labels = ['Non-VD', 'VD']
    else:
        labels = ['CSV', 'FMV', 'Healthy', 'VD']

    # Print classification report
    cf_report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)

    precision = cf_report['weighted avg']['precision']
    recall = cf_report['weighted avg']['recall']
    f1_score = cf_report['weighted avg']['f1-score']

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)

    df_cm = pd.DataFrame(
        cf_matrix/np.sum(cf_matrix), 
        index = [i for i in labels],
        columns = [i for i in labels]
    )
    plt.figure(figsize = (12,7))

    # TODO: make seaborn to use PyQT5

    sn.heatmap(df_cm, annot=True)

    if save:
        logger.info("Saving to model")
        
        # TODO: store hyperparams to other_json
        
        store_model_and_add_info_to_df(
            model = model, 
            description = "",
            dataset = dataset,
            num_classes = NUM_CLASSES,
            precision = precision,
            recall = recall,
            train_accuracy = train_accuracy,
            train_loss = train_loss,
            validation_accuracy = None,
            validation_loss = None,
            test_accuracy = test_accuracy,
            test_loss = test_loss,
            f1_score = f1_score,
            other_json = None,
        )


if __name__ == "__main__":
    train()