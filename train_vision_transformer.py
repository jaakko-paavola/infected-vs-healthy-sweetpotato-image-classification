# %%
import os
from torch.utils.data import DataLoader
from dataloaders.csv_data_loader import CSVDataLoader
<<<<<<< HEAD
# from models.resnet import resnet18
=======
from models.resnet import resnet18
>>>>>>> 1a28d773e620e3cfb8e8e6d224dca239381f59ec
from models.vision_transformer import VisionTransformer
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
import torch
import torch.optim as optim
import torch.nn.functional as F
import pathlib
<<<<<<< HEAD
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
=======
from sklearn.metrics import confusion_matrix
>>>>>>> 1a28d773e620e3cfb8e8e6d224dca239381f59ec
import seaborn as sn
import pandas as pd
import numpy as np
import torchvision
from utils.image_utils import img_to_patch
<<<<<<< HEAD
import optuna
=======

>>>>>>> 1a28d773e620e3cfb8e8e6d224dca239381f59ec
# %%

load_dotenv()

DATA_FOLDER_PATH = os.getenv("DATA_FOLDER_PATH")
PLANT_SPLIT_MASTER_PATH = os.path.join(DATA_FOLDER_PATH, "plant_data_split_master.csv")

#--- hyperparameters ---
N_EPOCHS = 10
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_TEST = 64
<<<<<<< HEAD
BATCH_SIZE_VAL = 64
LR = 0.01
NUM_CLASSES = 4
N_TRIALS = 3
PATIENCE = 5
=======
LR = 0.01
NUM_CLASSES = 2
>>>>>>> 1a28d773e620e3cfb8e8e6d224dca239381f59ec

# %%

data_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(180),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    # Values aquired from dataloaders/plant_master_dataset_stats.py
    transforms.Normalize(mean=[0.09872966, 0.11726899, 0.06568969],
                         std=[0.1219357, 0.14506954, 0.08257045])
])

plant_master_dataset = CSVDataLoader(
  csv_file=PLANT_SPLIT_MASTER_PATH, 
  root_dir=DATA_FOLDER_PATH,
  image_path_col="Split masked image path",
  label_col="Label",
  transform=data_transform
)

<<<<<<< HEAD
train_size = int(0.80 * len(plant_master_dataset))
val_size = (len(plant_master_dataset)-train_size)//2 
test_size = len(plant_master_dataset) - train_size - val_size

train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(plant_master_dataset, [train_size, test_size, val_size])

train_plant_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True, num_workers=4)
val_plant_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE_VAL, shuffle=True, num_workers=4)
test_plant_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE_TEST, shuffle=False, num_workers=4)

print(len(train_dataset))
=======
train_size = int(0.85 * len(plant_master_dataset))
test_size = len(plant_master_dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(plant_master_dataset, [train_size, test_size])

train_plant_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True, num_workers=0)
test_plant_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE_TEST, shuffle=False, num_workers=0)
>>>>>>> 1a28d773e620e3cfb8e8e6d224dca239381f59ec

#%% visualize some patches
NUM_IMAGES = 4
train_images = torch.stack([train_dataset[idx]['image'] for idx in range(NUM_IMAGES)], dim=0)

img_patches = img_to_patch(train_images, patch_size=32, flatten_channels=False)

fig, ax = plt.subplots(train_images.shape[0], 1, figsize=(14,3))
fig.suptitle("Images as input sequences of patches")
for i in range(train_images.shape[0]):
    img_grid = torchvision.utils.make_grid(img_patches[i], nrow=64, normalize=True, pad_value=0.9)
    img_grid = img_grid.permute(1, 2, 0)
    ax[i].imshow(img_grid)
    ax[i].axis('off')
plt.show()
plt.close()

#%%
<<<<<<< HEAD
import torch
=======

>>>>>>> 1a28d773e620e3cfb8e8e6d224dca239381f59ec
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

<<<<<<< HEAD
device

#%% resnet18_model = resnet18(num_classes=NUM_CLASSES).to(device)
model = VisionTransformer(
    # embed_dim=256,
    # hidden_dim=512,
=======
# resnet18_model = resnet18(num_classes=NUM_CLASSES).to(device)
model = VisionTransformer(
>>>>>>> 1a28d773e620e3cfb8e8e6d224dca239381f59ec
    embed_dim=256,
    hidden_dim=512,
    num_heads=8,
    num_layers=6,
    patch_size=32,
    num_channels=3,
    num_patches=64,  # with patch size 32
<<<<<<< HEAD
    num_classes=4,
    dropout=0.0
=======
    num_classes=2,
    dropout=0.2
>>>>>>> 1a28d773e620e3cfb8e8e6d224dca239381f59ec
)

# optimizer = optim.SGD(resnet18_model.parameters(), lr=LR, momentum=0.75)
optimizer = optim.AdamW(model.parameters(), lr=3e-4)
loss_function = torch.nn.CrossEntropyLoss()

<<<<<<< HEAD
#%%
def objective(trial):
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])
    parameter_grid_adam_dict = {
        # "betas": trial.suggest_categorical("betas", [0.9, 0.999]),
        "eps": trial.suggest_uniform("eps", 1e-8, 1e-4)
    }
    parameter_grid_adamw_dict = {
        # "betas": trial.suggest_uniform("betas", 0.9, 0.99),
        "eps": trial.suggest_uniform("eps", 1e-8, 1e-4),
        "weight_decay": trial.suggest_uniform("weight_decay", 0.0, 0.1)
    }
    if optimizer_name == "Adam":
        parameter_grid = parameter_grid_adam_dict
    elif optimizer_name == "AdamW":
        parameter_grid = parameter_grid_adamw_dict
    
    num_epochs = trial.suggest_int('num_epochs', 5, 10, 20)
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)

    model = model = VisionTransformer(
        embed_dim=256,
        hidden_dim=512,
        num_heads=8,
        num_layers=6,
        patch_size=32,
        num_channels=3,
        num_patches=64,  # with patch size 32
        num_classes=2,
        dropout=0.0
    )
    model = model.to(device)
    
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=LR, **parameter_grid)

    loss_function = torch.nn.CrossEntropyLoss()

    avg_training_losses = []
    training_accuracies = []
    avg_valid_losses = []
    valid_accuracies = []
    es = 0

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        total_train_loss = 0
        train_correct = 0
        total = 0

        # Training batch loop
        for batch_num, batch in enumerate(train_plant_dataloader):
            print(batch['image'])
            print(batch['label'])
            print(batch['label'].to(device))
            data, target = batch['image'].to(device), batch['label'].to(device)
            optimizer.zero_grad()
            if NUM_CLASSES == 2:
                target = target.eq(3).type(torch.int64) # For binary classification, transform labels to one-vs-rest
            # "Inception-v3 needs an input shape of [batch_size, 3, 299, 299] instead of [..., 224, 224].
            # You could up-/resample your images to the needed size and try it again"
            # (https://discuss.pytorch.org/t/error-in-training-inception-v3/23933/8):
            # data = torch.nn.Upsample(size=(299, 299), mode='bilinear')(data)
            # data = torch.nn.functional.interpolate(data, size=(299, 299), mode='bilinear', align_corners=False)
            total += data.shape[0]
            output = model(data)
            train_loss = loss_function(output, target)
            train_loss.backward()
            total_train_loss += train_loss.item()
            optimizer.step()
            pred = output.max(1, keepdim=True)[1]
            correct = pred.eq(target.view_as(pred)).sum().item()
            train_correct += correct

            if batch_num == len(train_plant_dataloader) - 1:
                print('Training: Epoch %d - Batch %d/%d: Loss: %.4f | Train Acc: %.3f%% (%d/%d)' % 
                    (epoch, batch_num + 1, len(train_plant_dataloader), train_loss, 
                    100. * train_correct / total, train_correct, total))
                # wandb.log({"loss": total_train_loss / (batch_num + 1)})

        avg_training_losses.append(total_train_loss / (batch_num + 1))
        training_accuracies.append(100. * train_correct / total)

        total_valid_loss = 0
        valid_correct = 0
        valid_total = 0
        best_valid_loss = 0
        model.eval()
        with torch.no_grad():
            # Validation batch loop
            for batch_num, batch in enumerate(val_plant_dataloader):
                data, target = batch['image'].to(device), batch['label'].to(device)
                valid_total += data.shape[0]
                output = model(data)
                print(output)
                if NUM_CLASSES == 2:
                    target = target.eq(3).type(torch.int64) # For binary classification, transform labels to one-vs-rest
                valid_loss = loss_function(output, target)
                pred = output.max(1, keepdim=True)[1]
                correct = pred.eq(target.view_as(pred)).sum().item()
                valid_correct += correct
                total_valid_loss += valid_loss.item()

                # if batch_num == len(val_plant_dataloader) - 1:
                print('Validation: Epoch %d - Batch %d/%d: Loss: %.4f | Validation Acc: %.3f%% (%d/%d)' % 
                    (epoch, batch_num + 1, len(val_plant_dataloader), valid_loss.item(), 
                    100. * valid_correct / valid_total, valid_correct, valid_total))

        avg_valid_losses.append(total_valid_loss / (batch_num +1))
        valid_accuracies.append(100. * valid_correct / valid_total)

        if avg_valid_losses[-1] < best_valid_loss:
            best_valid_loss = avg_valid_losses[-1]
            best_valid_acc = valid_accuracies[-1]
            es = 0
        else:
            es += 1
            if es > PATIENCE:
                print("Early stopping: Validation Acc: %.3f%% (%d/%d)" % (100. * valid_correct / valid_total, valid_correct, valid_total))
                break

        # if FLAG_EARLYSTOPPING:
        #     # early_stopping needs the validation loss to check if it has decresed, 
        #     # and if it has, it will make a checkpoint of the current model
        #     early_stopping(avg_valid_losses[-1], inception_v3_model)
        #     if early_stopping.early_stop:
        #         print("Early stop")
        #         break
                
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

    # load the last checkpoint with the best model
    # model.load_state_dict(torch.load('checkpoint.pt'))

    # trial.report(early_stopping.val_loss_min, epoch)
    return best_valid_acc



# %%
# Hyperparameter search
study = optuna.create_study(direction='minimize')
study.optimize(func=objective, n_trials=N_TRIALS)
study.best_params, study.best_value

#%% training
=======
# %%

# training
>>>>>>> 1a28d773e620e3cfb8e8e6d224dca239381f59ec

training_losses = []
training_accuracies = []
train_batches = []

for epoch in range(N_EPOCHS):
    total_train_loss = 0
    train_correct = 0
    total = 0

    for batch_num, batch in enumerate(train_plant_dataloader):
        data, target = batch['image'].to(device), batch['label'].to(device)
<<<<<<< HEAD
        
        if NUM_CLASSES == 2:
            # For binary classification, transform labels to one-vs-rest
           target = target.eq(3).type(torch.int64)
=======

        # For binary classification, transform labels to one-vs-rest
        target = target.eq(3).type(torch.int64)
>>>>>>> 1a28d773e620e3cfb8e8e6d224dca239381f59ec

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

        # if batch_num == len(train_plant_dataloader) - 1:
        print('Training: Epoch %d - Batch %d/%d: Loss: %.4f | Train Acc: %.3f%% (%d/%d)' % 
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
<<<<<<< HEAD
=======

>>>>>>> 1a28d773e620e3cfb8e8e6d224dca239381f59ec
# test
test_loss = 0
test_correct = 0
total = 0

with torch.no_grad():
    for batch_num, batch in enumerate(test_plant_dataloader):
        data, target = batch['image'].to(device), batch['label'].to(device)

        if NUM_CLASSES == 2:
            # For binary classification, transform labels to one-vs-rest
            target = target.eq(3).type(torch.int64)

        output = model(data)
        test_loss += loss_function(output, target).item()

        pred = output.max(1, keepdim=True)[1]

        correct = pred.eq(target.view_as(pred)).sum().item()
        test_correct += correct
        total += data.shape[0]

        test_loss /= len(test_plant_dataloader.dataset)

print("Final test score: Loss: %.4f, Accuracy: %.3f%%" % (test_loss, (100. * test_correct / total)))

# %%

# Store the model in the current path
# CURRENT_PATH = pathlib.Path(__file__).parent.resolve()
# torch.save(resnet18_model.state_dict(), os.path.join(CURRENT_PATH, "resnet18.pt"))

# %%

y_pred = []
y_true = []

with torch.no_grad():
    for batch_num, batch in enumerate(test_plant_dataloader):
        data, target = batch['image'].to(device), batch['label'].to(device)

        # For binary classification, transform labels to one-vs-rest
        target = target.eq(3).type(torch.int64)

        output = model(data)
        output = output.max(1, keepdim=True)[1]

        output = torch.flatten(output).cpu().numpy()
        y_pred.extend(output)
        
        target = target.cpu().numpy()
        y_true.extend(target)


# Multi-class labels for confusion matrix
# labels = ('CSV', 'FMV', 'Healthy', 'VD')

# Binary labels for confusion matrix
labels = ('Non-VD', 'VD')

# Build confusion matrix
cf_matrix = confusion_matrix(y_true, y_pred)  #tn, fp, fn, tp


df_cm = pd.DataFrame(
    cf_matrix/np.sum(cf_matrix), 
    index = [i for i in labels],
    columns = [i for i in labels]
)
plt.figure(figsize = (12,7))

sn.heatmap(df_cm, annot=True)
plt.savefig('transformer.png')

print("precision", precision_score(y_true, y_pred))
print("recall", recall_score(y_true, y_pred))
print("f1", f1_score(y_true, y_pred))

# %%
