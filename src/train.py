# %%
import os
from torch.utils.data import DataLoader
from dataloaders.csv_data_loader import CSVDataLoader
from models.resnet import resnet18
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
import torch
import torch.optim as optim
import torch.nn.functional as F

# %%

load_dotenv()

DATA_FOLDER_PATH = os.getenv("DATA_FOLDER_PATH")
PLANT_SPLIT_MASTER_PATH = os.path.join(DATA_FOLDER_PATH, "plant_data_split_master.csv")

#--- hyperparameters ---
N_EPOCHS = 20
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_TEST = 64
LR = 0.0001
NUM_CLASSES = 3

# %%

data_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(180),
    transforms.Resize(224),
    transforms.ToTensor(),
    # Values aquired from dataloaders/plant_master_dataset_stats.py
    transforms.Normalize(mean=[0.2234376, 0.27598768, 0.16376022],
                         std=[0.23811504, 0.28631625, 0.18748806])
])

plant_village_dataset = CSVDataLoader(
  csv_file=PLANT_SPLIT_MASTER_PATH, 
  root_dir=DATA_FOLDER_PATH,
  image_path_col="Split masked image path",
  label_col="Label",
  transform=data_transform
)

train_size = int(0.85 * len(plant_village_dataset))
test_size = len(plant_village_dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(plant_village_dataset, [train_size, test_size])

train_plant_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True, num_workers=0)
test_plant_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE_TEST, shuffle=False, num_workers=0)
# %%

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

resnet18_model = resnet18(num_classes=NUM_CLASSES).to(device)

optimizer = optim.SGD(resnet18_model.parameters(), lr=LR, momentum=0.75)
loss_function = torch.nn.CrossEntropyLoss()

# %%

# training

training_losses = []
training_accuracies = []

for epoch in range(N_EPOCHS):
    total_train_loss = 0
    train_correct = 0
    total = 0

    for batch_num, batch in enumerate(train_plant_dataloader):
        data, target = batch['image'].to(device), batch['label'].to(device)

        optimizer.zero_grad() 

        output = resnet18_model(data)
        train_loss = loss_function(output, target)
        train_loss.backward()
        optimizer.step()
        
        pred = output.max(1, keepdim=True)[1]

        correct = pred.eq(target.view_as(pred)).sum().item()
        train_correct += correct
        total += data.shape[0]
        total_train_loss += train_loss.item()

        if batch_num == len(train_plant_dataloader) - 1:
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

# test
test_loss = 0
test_correct = 0
total = 0

with torch.no_grad():
    for batch_num, batch in enumerate(test_plant_dataloader):
        data, target = batch['image'].to(device), batch['label'].to(device)
        
        output = resnet18_model(data)
        test_loss += loss_function(output, target).item()
        pred = output.max(1, keepdim=True)[1]
        correct = pred.eq(target.view_as(pred)).sum().item()
        test_correct += correct
        total += data.shape[0]

        test_loss /= len(test_plant_dataloader.dataset)

print("Final test score: Loss: %.4f, Accuracy: %.3f%%" % (test_loss, (100. * test_correct / total)))

# %%
