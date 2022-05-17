# %%
import os
from time import time, strftime, gmtime
import click
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from dataloaders.csv_data_loader import CSVDataLoader
from models.model_factory import get_model_class
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
import os
import optuna
from pytorchtools import EarlyStopping
import warnings
import logging
from utils.model_utils import AVAILABLE_MODELS, save_dataset_of_torch_model
from dataloaders.dataset_stats import get_normalization_mean_std

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# %%
load_dotenv()
DATA_FOLDER_PATH = os.getenv("DATA_FOLDER_PATH")
warnings.filterwarnings("ignore")

# %%
def compute_and_print_metrics(stage, NUM_CLASSES, epoch, total_correct, total, true_positive,
        true_negative, false_positive, false_negative, target_all_batches, pred_all_batches, batch_num, no_of_batches, loss):
    logger.info(f"{stage}: Epoch {epoch} - Batch {batch_num + 1}/{no_of_batches}: Loss: {loss} | {stage} Acc: {100. * total_correct / total} ({total_correct}/{total})")
    logger.info(f"{stage} TP: {true_positive} TN: {true_negative} FP: {false_positive} FN: {false_negative}")
    recall = true_positive / (true_positive + false_negative + 1e-10)
    precision = true_positive / (true_positive + false_positive + 1e-10)
    logger.info(f"{stage} Recall: {recall}")
    logger.info(f"{stage} Precision: {precision}")
    logger.info(f"{stage} F1: {2 * precision * recall / (precision + recall + 1e-10)}")

    f1m = f1_score(target_all_batches.detach().cpu(), pred_all_batches.detach().cpu(), average = 'macro', zero_division=1)
    f1w = f1_score(target_all_batches.detach().cpu(), pred_all_batches.detach().cpu(), average = 'weighted', zero_division=1)
    f1mi = f1_score(target_all_batches.detach().cpu(), pred_all_batches.detach().cpu(), average = 'micro', zero_division=1)

    logger.info(f"{stage} unweighted macro F1 as per sklearn: {f1m}")
    logger.info(f"{stage} weighted macro F1 as per sklearn: {f1w}")
    logger.info(f"{stage} 'global' micro F1 as per sklearn: {f1mi}")

    if (NUM_CLASSES == 2):
        f1b = f1_score(target_all_batches.detach().cpu(), pred_all_batches.detach().cpu(), average = 'binary', zero_division=1)
        logger.info(f"{stage} binary F1 as per sklearn: {f1b}")
    else:
        f1b = 0

    return recall, precision, f1m, f1w, f1mi, f1b

# %%
def evaluate_predictions(total_correct, true_positive, true_negative, false_positive, false_negative,
        target_all_batches, pred_all_batches, target, output):
    pred = output.max(1, keepdim=True)[1]
    correct = pred.eq(target.view_as(pred)).sum().item()
    total_correct += correct
    true_positive += torch.logical_and(pred.eq(target.view_as(pred)), pred).sum().item()
    true_negative += torch.logical_and(pred.eq(target.view_as(pred)), 1 - pred).sum().item()
    false_positive += torch.logical_and(pred.eq(1 - target.view_as(pred)), pred).sum().item()
    false_negative += torch.logical_and(pred.eq(1 - target.view_as(pred)), 1 - pred).sum().item()
    target_all_batches = torch.cat((target_all_batches, target.view_as(pred)), 0)
    pred_all_batches = torch.cat((pred_all_batches, pred), 0)
    return target_all_batches, pred_all_batches, true_positive, true_negative, false_positive, false_negative, total_correct

# %%
# Define an objective function to be minimized by Optuna.
def objective(trial, MODEL_NAME, NUM_CLASSES, N_EPOCHS, OPTIMIZER_SEARCH_SPACE, device, train_plant_dataloader, val_plant_dataloader, \
    FLAG_EARLYSTOPPING, EARLYSTOPPING_PATIENCE, test_plant_dataloader=None):
    if MODEL_NAME == "vision_transformer":
        num_heads = trial.suggest_categorical('num_heads', [4, 8, 16])
        dropout = trial.suggest_uniform('dropout', 0.0, 0.2)
        model = get_model_class(MODEL_NAME, num_of_classes=NUM_CLASSES, num_heads=num_heads, dropout=dropout).to(device)
    else:
        model = get_model_class(MODEL_NAME, num_of_classes=NUM_CLASSES).to(device)

    # Define hyperparameter search spaces for Optuna:
    optimizer_name = trial.suggest_categorical("optimizer", OPTIMIZER_SEARCH_SPACE)

    lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)

    parameter_grid_adam_dict = {
        "betas": trial.suggest_categorical("betas", [(0.9, 0.99), (0.95, 0.999), (0.9, 0.949)]),
        "eps": trial.suggest_uniform("eps", 1e-8, 1e-4),
        "weight_decay": trial.suggest_uniform("weight_decay", 1e-03, 0.1),
    }

    parameter_grid_adamw_dict = {
        "betas": trial.suggest_categorical("betas", [(0.9, 0.99), (0.95, 0.999), (0.9, 0.949)]),
        "eps": trial.suggest_uniform("eps", 1e-8, 1e-4),
        "weight_decay": trial.suggest_uniform("weight_decay", 1e-03, 0.1)
    }

    parameter_grid_rmsprop_dict = {
        "alpha": trial.suggest_uniform("alpha", 0.9, 0.99),
        "eps": trial.suggest_uniform("eps", 1e-8, 1e-4),
        "weight_decay": trial.suggest_uniform("weight_decay", 1e-03, 0.1),
        "momentum": trial.suggest_uniform("momentum", 0.9, 0.99)
    }

    parameter_grid_sgd_dict = {
        "momentum": trial.suggest_uniform("momentum", 0.9, 0.99),
        "weight_decay": trial.suggest_uniform("weight_decay", 1e-03, 0.1),
        "dampening": trial.suggest_uniform('dampening', 0.1, 0.2)
    }

    parameter_grid_adagrad_dict = {
        "eps": trial.suggest_uniform("eps", 1e-8, 1e-4),
        "lr_decay": trial.suggest_uniform("lr_decay", 0.0, 0.1),
        "weight_decay": trial.suggest_uniform("weight_decay", 0.0, 0.1)
    }

    if optimizer_name == "SGD":
        parameter_grid = parameter_grid_sgd_dict
    elif optimizer_name == "Adam":
        parameter_grid = parameter_grid_adam_dict
    elif optimizer_name == "AdamW":
        parameter_grid = parameter_grid_adamw_dict
    elif optimizer_name == "RMSprop":
        parameter_grid = parameter_grid_rmsprop_dict
    elif optimizer_name == "Adagrad":
        parameter_grid = parameter_grid_adagrad_dict

    # Define an optimizer
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr, **parameter_grid)

    # Define a loss function.
    loss_function = torch.nn.CrossEntropyLoss()

    # Training of the model.
    avg_training_losses = []
    training_accuracies = []
    avg_valid_losses = []
    valid_accuracies = []
    train_F1s = []
    train_unweighted_macro_F1s = []
    train_weighted_macro_F1s = []
    train_binary_F1s = []
    valid_F1s = []
    valid_unweighted_macro_F1s = []
    valid_weighted_macro_F1s = []
    valid_binary_F1s = []

    early_stopping = EarlyStopping(patience=EARLYSTOPPING_PATIENCE, verbose=True, delta=1e-4)

    for epoch in range(1, N_EPOCHS + 1):
        # Training
        print()
        model.train()
        total_train_loss = 0
        total_correct = 0
        total = 0
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        target_all_batches = torch.empty(0, dtype=torch.float).to(device)
        pred_all_batches = torch.empty(0, dtype=torch.float).to(device)

        # Training batch loop
        # batch = iter(train_plant_dataloader).next() # For debugging
        # batch_num = 1 # For debugging
        for batch_num, batch in enumerate(train_plant_dataloader):
            data, target = batch['image'].to(device), batch['label'].to(device)
            optimizer.zero_grad()
            if (NUM_CLASSES == 2):
                target = target.eq(3).type(torch.int64) # For binary classification, transform labels to one-vs-rest
            total += data.shape[0]
            output = model(data)
            if (len(output) == 2):
                output = output.logits
            train_loss = loss_function(output, target)
            train_loss.backward()
            total_train_loss += train_loss.item()
            optimizer.step()
            target_all_batches, pred_all_batches, true_positive, true_negative, false_positive, false_negative, \
                total_correct = evaluate_predictions(total_correct, true_positive, true_negative, false_positive, \
                false_negative, target_all_batches, pred_all_batches, target, output)

        recall, precision, f1m, f1w, f1mi, f1b = compute_and_print_metrics("Training", NUM_CLASSES, epoch, total_correct,
            total, true_positive, true_negative, false_positive, false_negative, target_all_batches, pred_all_batches,
            batch_num, len(train_plant_dataloader), total_train_loss / (batch_num + 1))

        train_unweighted_macro_F1s.append(f1m)
        train_weighted_macro_F1s.append(f1w)
        train_binary_F1s.append(f1b)
        train_F1s.append(2 * precision * recall / (precision + recall + 1e-10))
        avg_training_losses.append(total_train_loss / (batch_num + 1))
        training_accuracies.append(100. * total_correct / total)

        # Validation
        total_valid_loss = 0
        total_correct = 0
        total = 0
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        target_all_batches = torch.empty(0, dtype=torch.float).to(device)
        pred_all_batches = torch.empty(0, dtype=torch.float).to(device)

        model.eval()
        with torch.no_grad():
            # Validation batch loop-
            # batch = iter(val_plant_dataloader).next() # For debugging
            # batch_num = 1 # For debugging
            for batch_num, batch in enumerate(val_plant_dataloader):
                data, target = batch['image'].to(device), batch['label'].to(device)
                if (NUM_CLASSES == 2):
                    target = target.eq(3).type(torch.int64) # For binary classification, transform labels to one-vs-rest
                total += data.shape[0]
                output = model(data)
                valid_loss = loss_function(output, target)
                total_valid_loss += valid_loss.item()
                target_all_batches, pred_all_batches, true_positive, true_negative, false_positive, false_negative, \
                    total_correct = evaluate_predictions(total_correct, true_positive, true_negative, false_positive, \
                        false_negative, target_all_batches, pred_all_batches, target, output)

        recall, precision, f1m, f1w, f1mi, f1b = compute_and_print_metrics("Validation", NUM_CLASSES, epoch, total_correct,
            total, true_positive, true_negative, false_positive, false_negative, target_all_batches, pred_all_batches,
            batch_num, len(val_plant_dataloader), total_valid_loss / (batch_num + 1))

        valid_unweighted_macro_F1s.append(f1m)
        valid_weighted_macro_F1s.append(f1w)
        valid_binary_F1s.append(f1b)
        valid_F1s.append(2 * precision * recall / (precision + recall + 1e-10))
        avg_valid_losses.append(total_valid_loss / (batch_num + 1))
        valid_accuracies.append(100. * total_correct / total)

        if FLAG_EARLYSTOPPING:
            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            early_stopping(avg_valid_losses[-1], model)
            # If the loss has not decreased for {patience} number of epochs, trigger early stop
            if early_stopping.early_stop:
                logger.info("Early stop")
                break

    # Training loss and accuracy average for all batches
    plt.plot(range(1, epoch + 1), avg_training_losses, label = "Training loss")
    plt.plot(range(1, epoch + 1), avg_valid_losses, label = "Validation loss")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.ylabel('loss')
    plt.title('Loss')
    plt.legend()
    plt.show()

    plt.plot(range(1, epoch + 1), training_accuracies, label = "Training accuracy")
    plt.plot(range(1, epoch + 1), valid_accuracies, label = "Validation accuracy")
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.show()

    plt.plot(range(1, epoch + 1), train_F1s, label = "Training F1")
    plt.plot(range(1, epoch + 1), valid_F1s, label = "Validation F1")
    plt.xlabel('epoch')
    plt.ylabel('F1')
    plt.title('F1')
    plt.legend()
    plt.show()

    plt.plot(range(1, epoch + 1), train_unweighted_macro_F1s, label = "Training unweighted macro F1")
    plt.plot(range(1, epoch + 1), valid_unweighted_macro_F1s, label = "Validation unweighted macro F1")
    plt.xlabel('epoch')
    plt.ylabel('Unweighted macro F1')
    plt.title('Unweighted macro F1')
    plt.legend()
    plt.show()

    plt.plot(range(1, epoch + 1), train_weighted_macro_F1s, label = "Training weighted macro F1")
    plt.plot(range(1, epoch + 1), valid_weighted_macro_F1s, label = "Validation weighted macro F1")
    plt.xlabel('epoch')
    plt.ylabel('Weighted macro F1')
    plt.title('Weighted macro F1')
    plt.legend()
    plt.show()

    # load the last checkpoint with the best model
    model.load_state_dict(torch.load('checkpoint.pt'))

    # trial.report(early_stopping.val_loss_min, epoch)
    return early_stopping.val_loss_min

# %%
# Hyperparameter search

@click.command()
@click.option('-m', '--model', required=True, type=click.Choice(AVAILABLE_MODELS, case_sensitive=False), help='Model architechture.')
@click.option('-e', '--no_of_epochs', type=int, default=20, help='Number of epochs in training loop.')
@click.option('-t', '--no_of_trials', type=int, default=50, help='Number of hyperparamter search trials in training loop.')
@click.option('-d', '--dataset', type=click.Choice(['plant', 'plant_golden', 'leaf'], case_sensitive=False), default="plant", help='Already available dataset to use to train the model. Give either -d or -csv, not both.')
@click.option('-csv', '--data-csv', type=str, help='Full file path to dataset CSV-file created during segmentation. Give either -d or -csv, not both.')
@click.option('-b', '--binary', is_flag=True, show_default=True, default=False, help='Train binary classifier instead of multiclass classifier.')
@click.option('-aug', '--augmentation', is_flag=True, show_default=True, default=True, help='Use data-augmentation for the training.')
@click.option('-s', '--save', is_flag=True, show_default=True, default=True, help='Save the trained model and add information to model dataframe.')
@click.option('-v', '--verbose', is_flag=True, show_default=True, default=False, help='Print verbose logs.')
def search_hyperparameters(model, no_of_epochs, no_of_trials, dataset, data_csv, binary, augmentation, save, verbose):

    MODEL_NAME = model

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
        mean, std = get_normalization_mean_std(dataset=dataset)
    # TODO: give dataset name when using custom CSV for storing the model
    else:
        DATA_MASTER_PATH = data_csv
        mean, std = get_normalization_mean_std(datasheet=data_csv)


    # TODO: automatize label counting from dataframe
    if binary:
        NUM_CLASSES = 2
    else:
        NUM_CLASSES = 4


    N_EPOCHS = no_of_epochs
    N_TRIALS = no_of_trials # Number of trials for hyperparameter optimization
    BATCH_SIZE_TRAIN = 64
    BATCH_SIZE_VALID = 64
    BATCH_SIZE_TEST = 64
    FLAG_EARLYSTOPPING = True # Set to True to enable early stopping
    EARLYSTOPPING_PATIENCE = N_EPOCHS//3 # Let early stopping patience (i.e. the number of consequtive epochs with no decrease in training loss) be a third (rounded down) of the number of epochs

    if augmentation:
        data_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(50),
            transforms.RandomRotation(180),
            transforms.RandomAffine(translate=(0.1, 0.1), degrees=0),
            transforms.Resize((299, 299)) if MODEL_NAME == "inception_v3" else transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            # GaussianNoise(0., 0.1),
        ])
    else:
        data_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(50),
            transforms.Resize((299, 299)) if MODEL_NAME == "inception_v3" else transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    ## TODO: selection of the optimizer space (possible via a commandline argument)
    OPTIMIZER_SEARCH_SPACE = []
    OPTIMIZER_SEARCH_SPACE.append("Adam")
    OPTIMIZER_SEARCH_SPACE.append("AdamW")
    # OPTIMIZER_SEARCH_SPACE.append("RMSprop")
    # OPTIMIZER_SEARCH_SPACE.append("SGD")
    # OPTIMIZER_SEARCH_SPACE.append("Adagrad")

    plant_master_dataset = CSVDataLoader(
        csv_file=DATA_MASTER_PATH,
        root_dir=DATA_FOLDER_PATH,
        image_path_col="Split masked image path",
        label_col="Label",
        transform=data_transform
    )

    train_size = int(0.80 * len(plant_master_dataset))
    val_size = (len(plant_master_dataset) - train_size)//2
    test_size = len(plant_master_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(plant_master_dataset, [train_size, val_size, test_size])

    train_plant_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True, num_workers=0)
    val_plant_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE_VALID, shuffle=True, num_workers=0)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model_class = get_model_class(MODEL_NAME, num_of_classes=NUM_CLASSES).to(device)
    id, model_name, timestamp = save_dataset_of_torch_model(model_class, test_dataset, "test_dataset")

    study = optuna.create_study(direction='minimize')
    study.optimize(func=lambda trial: objective(trial, MODEL_NAME, NUM_CLASSES, N_EPOCHS, OPTIMIZER_SEARCH_SPACE, \
        device, train_plant_dataloader, val_plant_dataloader, FLAG_EARLYSTOPPING, EARLYSTOPPING_PATIENCE),\
        n_trials=N_TRIALS)

    ## TODO: how to save n best models/hyperparameter configurations, how to save the unseen test dataset,
    ## which can then be used in train.py to evaluate the model.
    df = study.trials_dataframe()
    df = df.sort_values(by=['value'], ascending=True).iloc[0:9,:]

    timestamp = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    filename = os.path.join(DATA_FOLDER_PATH, f'Top_10_hyperparameter_search_results_at_{timestamp}.csv')
    with open(filename, "w") as f:
        f.write(f"{id}-{model_name}-{timestamp}\n")

    df.to_csv(filename, mode='a', header=False)

if __name__ == "__main__":
    search_hyperparameters()
# %%
