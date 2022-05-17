# %%
import os
from pathlib import Path
from sklearn.preprocessing import binarize
from torch.utils.data import DataLoader
from dataloaders.csv_data_loader import CSVDataLoader
from dataloaders.gaussian_noise import GaussianNoise
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sn
import pandas as pd
import numpy as np
import click
import statistics
from models.model_factory import get_model_class
from utils.model_utils import AVAILABLE_MODELS, store_model_and_add_info_to_df, get_image_size
import logging
from tqdm import tqdm
import yaml
from dataloaders.dataset_stats import get_normalization_mean_std
from dataloaders.dataset_labels import get_dataset_labels

logging.basicConfig() 
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# %%

load_dotenv()
DATA_FOLDER_PATH = os.getenv("DATA_FOLDER_PATH")

# %%

@click.command()
@click.option('-m', '--model', required=True, type=click.Choice(AVAILABLE_MODELS, case_sensitive=False), help='Model architechture.')
@click.option('-d', '--dataset', type=click.Choice(['plant', 'plant_golden', 'leaf'], case_sensitive=False), help='Already available dataset to use to train the model. Give either -d or -csv, not both.')
@click.option('-csv', '--data-csv', type=str, help='Full file path to dataset CSV-file created during segmentation. Give either -d or -csv, not both.')
@click.option('-b', '--binary', is_flag=True, show_default=True, default=False, help='Train binary classifier instead of multiclass classifier.')
@click.option('-bl', '--binary-label', type=int, help='Binary label when dataset has more than two labels. Classification is done using one-vs-rest, where the binary label corresponds to the one compared to other labels.')
@click.option('-p', '--params-file', type=str, default="hyperparams.yaml", help='Full file path to hyperparameter-file used during the training. File must be a YAMl file and similarly structured than hyperparams.yaml.')
@click.option('-aug/-no-aug', '--augmentation/--no-augmentation', show_default=True, default=True, help='Use data-augmentation for the training.')
@click.option('-s/-nos', '--save/--no-save', show_default=True, default=True, help='Save the trained model and add information to model dataframe.')
@click.option('-v', '--verbose', is_flag=True, show_default=True, default=False, help='Print verbose logs.')
def train(model, dataset, data_csv, binary, binary_label, params_file, augmentation, save, verbose):
    
    if verbose:
        logger.setLevel(logging.DEBUG)

    logger.info("Reading the data")

    if (not dataset and not data_csv) or (dataset and data_csv):
        raise ValueError("You must pass either -d (name of the available dataset) or -csv (path to data-CSV)")

    if dataset:
        if dataset == 'plant':
            DATA_MASTER_PATH = os.path.join(DATA_FOLDER_PATH, "plant_data_split_master.csv")
        elif dataset == 'leaf':
            DATA_MASTER_PATH = os.path.join(DATA_FOLDER_PATH, "leaves_segmented_master.csv")
        elif dataset == 'plant_golden':
            DATA_MASTER_PATH = os.path.join(DATA_FOLDER_PATH, "plant_data_split_golden.csv")
        else:             
            raise ValueError(f"Dataset {dataset} not defined. Accepted values: plant, plant_golden, leaf")
        
        mean, std = get_normalization_mean_std(dataset=dataset)
    else:
        DATA_MASTER_PATH = data_csv
        mean, std = get_normalization_mean_std(datasheet=data_csv)
        # To give the dataset name when storing the model
        dataset = Path(data_csv).stem

    labels = get_dataset_labels(datasheet_path=DATA_MASTER_PATH)
    
    if binary and not binary_label and len(labels) > 2:
        raise ValueError(f"You tried to do binary classification without binary label argument. You must give also binary-label (-bl or --binary-label) argument when using binary classification and the dataset contains more than two labels. We detected {len(labels)} number of labels.")
    
    if binary:
        NUM_CLASSES = 2
        
        if len(labels) > 2:
            # Convert the label names to one-vs-rest labels
            labels = [f'Non-{labels[binary_label]}', labels[binary_label]]
    else:
        NUM_CLASSES = len(labels)

    with open(params_file, "r") as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logger.error(f"Error while reading YAML: {exc}")
            raise exc

    # hyperparameters:
    # TODO: the final form of hyperparameters.yaml
    N_EPOCHS = int(params[model]['N_EPOCHS'])
    BATCH_SIZE_TRAIN = int(params[model]['BATCH_SIZE_TRAIN'])
    BATCH_SIZE_TEST = int(params[model]['BATCH_SIZE_TEST'])
    OPTIMIZER = params[model]['OPTIMIZER']
    LR = float(params[model]['LR'])
    WEIGHT_DECAY = float(params[model]['WEIGHT_DECAY'])
    
    image_size = get_image_size(model)

    if augmentation:
        data_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(50),
            transforms.RandomRotation(180),
            transforms.RandomAffine(translate=(0.1, 0.1), degrees=0),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        data_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(50),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    master_dataset = CSVDataLoader(
        csv_file=DATA_MASTER_PATH,
        root_dir=DATA_FOLDER_PATH,
        image_path_col="Split masked image path",
        label_col="Label",
        transform=data_transform
    )

    train_size = int(0.85 * len(master_dataset))
    test_size = len(master_dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(master_dataset, [train_size, test_size])

    train_plant_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True, num_workers=0)
    test_plant_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE_TEST, shuffle=False, num_workers=0)

    # %%

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model_class = get_model_class(model, num_of_classes=NUM_CLASSES, num_heads=params[model]['NUM_HEADS'], dropout=params[model]['DROPOUT']).to(device)
    parameter_grid = {}
    parameter_grid["lr"] = LR
    parameter_grid["weight_decay"] = WEIGHT_DECAY

    if OPTIMIZER == "SGD":
        parameter_grid['dampening'] = float(params[model]['DAMPENING'])
        parameter_grid['momentum'] = float(params[model]['MOMENTUM'])
        optimizer = optim.SGD(model_class.parameters(), **parameter_grid)  
    else:
        parameter_grid['eps'] = float(params[model]['EPS'])
        if OPTIMIZER == "Adam":
            parameter_grid['betas'] = tuple(float(x) for x in params[model]['BETAS'][1:-1].replace("(", "").replace(")", "").strip().split(","))
            optimizer = optim.Adam(model_class.parameters(), **parameter_grid)
        elif OPTIMIZER == "AdamW":
            parameter_grid['betas'] = tuple(float(x) for x in params[model]['BETAS'][1:-1].replace("(", "").replace(")", "").strip().split(","))
            optimizer = optim.AdamW(model_class.parameters(), **parameter_grid)
        elif OPTIMIZER == "AdaGrad":
            parameter_grid['lr_decay'] = float(params[model]['LR_DECAY'])
            optimizer = optim.Adagrad(model_class.parameters(), **parameter_grid)
        elif OPTIMIZER == "RMSprop":
            parameter_grid['momentum'] = float(params[model]['MOMENTUM'])
            parameter_grid['alpha'] = float(params[model]['ALPHA'])
            optimizer = optim.RMSprop(model_class.parameters(), **parameter_grid)

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
                target = target.eq(binary_label).type(torch.int64)

            optimizer.zero_grad()

            output = model_class(data)
            
            if len(output) == 2:
                output = output.logits
                
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
                    (epoch, batch_num + 1, len(train_plant_dataloader), total_train_loss / (batch_num + 1), 
                    100. * train_correct / total, train_correct, total))


        # Training loss average for all batches
        training_losses.append(total_train_loss / len(train_plant_dataloader))        
        training_accuracies.append((100. * train_correct / total))

    # Calculate train loss and accuracy as an average of the last min(5, N_EPOCHS) losses or accuracies
    train_loss = statistics.mean(training_losses[-min(N_EPOCHS, 5):])
    train_accuracy = statistics.mean(training_accuracies[-min(N_EPOCHS, 5):])

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
                target = target.eq(binary_label).type(torch.int64)

            output = model_class(data)
            
            if len(output) == 2:
                output = output.logits

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

    # Print classification report
    cf_report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)

    precision = cf_report['weighted avg']['precision']
    recall = cf_report['weighted avg']['recall']
    f1_score = cf_report['weighted avg']['f1-score']

    if save:
        logger.info("Saving the model")
        
        other_json = {
            'LABELS': labels
        }
        
        model_id = store_model_and_add_info_to_df(
            model = model_class, 
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
            other_json = other_json,
        )
        
        logger.info(f"Model saved with id {model_id}")

if __name__ == "__main__":
    train()
