# %%
import os
from sklearn.preprocessing import binarize
from torch.utils.data import DataLoader
from dataloaders.csv_data_loader import CSVDataLoader
from models.resnet import resnet18
from models.F1_score import F1Score
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
import torch
import torch.optim as optim
import torch.nn.functional as F
import pathlib
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sn
import pandas as pd
import numpy as np
import click
from models.model_factory import get_model_class
from utils.model_utils import AVAILABLE_MODELS, store_model_and_add_info_to_df
import logging
from tqdm import tqdm
import yaml

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# %%

load_dotenv()
DATA_FOLDER_PATH = os.getenv("DATA_FOLDER_PATH")

# %%

@click.command()
@click.option('-m', '--model', required=True, type=click.Choice(AVAILABLE_MODELS, case_sensitive=False), help='Model architechture.')
@click.option('-d', '--data', type=click.Choice(['plant', 'plant_golden', 'leaf', 'leaf_golden'], case_sensitive=False), default="plant", help='Dataset to use to train the model.')
@click.option('-b', '--binary', is_flag=True, show_default=True, default=False, help='Train binary classifier instead of multiclass classifier.')
@click.option('-p', '--params-file', type=str, default="hyperparams.yaml", help='Full file path to hyperparameter-file used during the training. File must be a YAMl file and similarly structured than hyperparams.yaml.')
@click.option('-aug', '--augmentation', is_flag=True, show_default=True, default=True, help='Use data-augmentation for the training.')
@click.option('-s', '--save', is_flag=True, show_default=True, default=True, help='Save the trained model and add information to model dataframe.')
@click.option('-v', '--verbose', is_flag=True, show_default=True, default=False, help='Print verbose logs.')
def train(model, data, binary, params_file, augmentation, save, verbose):
    
    if verbose:
        logger.setLevel(logging.DEBUG)

    if data == 'plant':
        DATA_MASTER_PATH = os.path.join(DATA_FOLDER_PATH, "plant_data_split_master.csv")
    elif data == 'leaf':
        DATA_MASTER_PATH = os.path.join(DATA_FOLDER_PATH, "leaf_data.csv")
    elif data == 'plant_golden':
        raise NotImplementedError("Plant golden dataset not implemented yet")
    elif data == 'leaf_golden':
        raise NotImplementedError("Leaf golden dataset not implemented yet")

    if binary:
        NUM_CLASSES = 2
    else:
        NUM_CLASSES = 4


    with open(params_file, "r") as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logger.error(f"Error while reading YAML: {exc}")


    #--- hyperparameters ---
    N_EPOCHS = params[model]['N_EPOCHS']
    BATCH_SIZE_TRAIN = params[model]['BATCH_SIZE_TRAIN']
    BATCH_SIZE_TEST = params[model]['BATCH_SIZE_TEST']
    LR = params[model]['LR']

    if augmentation:
        data_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(180),
            transforms.RandomAffine(translate=(0.1, 0.3), scale=(0.6, 0.9), degrees=180),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            # Values aquired from dataloaders/plant_master_dataset_stats.py
            # TODO: automatice the mean and std calculation
            transforms.Normalize(mean=[0.09872966, 0.11726899, 0.06568969],
                                std=[0.1219357, 0.14506954, 0.08257045])
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

    logging.info("Starting training cycle")

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
                logging.info('Training: Epoch %d - Batch %d/%d: Loss: %.4f | Train Acc: %.3f%% (%d/%d)' % 
                    (epoch, batch_num + 1, len(train_plant_dataloader), train_loss / (batch_num + 1), 
                    100. * train_correct / total, train_correct, total))


        # Training loss average for all batches
        training_losses.append(total_train_loss / len(train_plant_dataloader))        
        training_accuracies.append((100. * train_correct / total))

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

    logging.info("Starting testing cycle")

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

    logging.info("Final test score: Loss: %.4f, Accuracy: %.3f%%" % (test_loss, (100. * test_correct / total)))

    if binary:
        labels = ['Non-VD', 'VD']
    else:
        labels = ['CSV', 'FMV', 'Healthy', 'VD']

    # Print classification report
    cf_report = classification_report(y_true, y_pred, target_names=labels)
    print(cf_report)

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)

    df_cm = pd.DataFrame(
        cf_matrix/np.sum(cf_matrix), 
        index = [i for i in labels],
        columns = [i for i in labels]
    )
    plt.figure(figsize = (12,7))

    sn.heatmap(df_cm, annot=True)


    #if save:
    #    if binary:
    #        store_model_and_add_info_to_df(model)
    #    else:



if __name__ == "__main__":
    train()