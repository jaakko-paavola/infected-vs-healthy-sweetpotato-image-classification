# %%
import pickle
import os
from sklearn.model_selection import GridSearchCV
from torch.utils.data import DataLoader
from dataloaders.csv_data_loader import CSVDataLoader
from models.resnet import resnet18
from models.bag_of_words import BagOfWords
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
import torch
import torch.optim as optim
import torch.nn.functional as F
import pathlib
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import numpy as np
from itertools import product
from sklearn.model_selection import train_test_split

# %%

load_dotenv()

DATA_FOLDER_PATH = os.getenv("DATA_FOLDER_PATH")
PLANT_SPLIT_MASTER_PATH = os.path.join(DATA_FOLDER_PATH, "plant_data_split_master.csv")

#--- hyperparameters ---
N_EPOCHS = 20
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_TEST = 64
LR = 0.01
NUM_CLASSES = 2

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

train_size = int(0.85 * len(plant_master_dataset))
test_size = len(plant_master_dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(plant_master_dataset, [train_size, test_size])

train_plant_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True, num_workers=0)
test_plant_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE_TEST, shuffle=False, num_workers=0)
# # %%

# if torch.cuda.is_available():
#     device = torch.device('cuda')
# else:
#     device = torch.device('cpu')

# resnet18_model = resnet18(num_classes=NUM_CLASSES).to(device)

# optimizer = optim.SGD(resnet18_model.parameters(), lr=LR, momentum=0.75)
# loss_function = torch.nn.CrossEntropyLoss()

# # %%

# # training

# training_losses = []
# training_accuracies = []

# for epoch in range(N_EPOCHS):
#     total_train_loss = 0
#     train_correct = 0
#     total = 0

#     for batch_num, batch in enumerate(train_plant_dataloader):
#         data, target = batch['image'].to(device), batch['label'].to(device)

#         # For binary classification, transform labels to one-vs-rest
#         target = target.eq(3).type(torch.int64)

#         optimizer.zero_grad()

#         output = resnet18_model(data)

#         train_loss = loss_function(output, target)
#         train_loss.backward()
#         optimizer.step()

#         pred = output.max(1, keepdim=True)[1]

#         correct = pred.eq(target.view_as(pred)).sum().item()
#         train_correct += correct
#         total += data.shape[0]
#         total_train_loss += train_loss.item()

#         if batch_num == len(train_plant_dataloader) - 1:
#             print('Training: Epoch %d - Batch %d/%d: Loss: %.4f | Train Acc: %.3f%% (%d/%d)' %
#                   (epoch, batch_num + 1, len(train_plant_dataloader), train_loss / (batch_num + 1),
#                    100. * train_correct / total, train_correct, total))


#     # Training loss average for all batches
#     training_losses.append(total_train_loss / len(train_plant_dataloader))
#     training_accuracies.append((100. * train_correct / total))

# plt.plot(range(N_EPOCHS), training_losses, label = "Training loss")
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.title('Loss')
# plt.legend()
# plt.show()

# plt.plot(range(N_EPOCHS), training_accuracies, label = "Training accuracy")
# plt.xlabel('epoch')
# plt.ylabel('accuracy')
# plt.title('Accuracy')
# plt.legend()
# plt.show()

# # %%

# # test
# test_loss = 0
# test_correct = 0
# total = 0

# with torch.no_grad():
#     for batch_num, batch in enumerate(test_plant_dataloader):
#         data, target = batch['image'].to(device), batch['label'].to(device)

#         # For binary classification, transform labels to one-vs-rest
#         target = target.eq(3).type(torch.int64)

#         output = resnet18_model(data)
#         test_loss += loss_function(output, target).item()

#         pred = output.max(1, keepdim=True)[1]

#         correct = pred.eq(target.view_as(pred)).sum().item()
#         test_correct += correct
#         total += data.shape[0]

#         test_loss /= len(test_plant_dataloader.dataset)

# print("Final test score: Loss: %.4f, Accuracy: %.3f%%" % (test_loss, (100. * test_correct / total)))

# # %%

# # Store the model in the current path
# # CURRENT_PATH = pathlib.Path(__file__).parent.resolve()
# # torch.save(resnet18_model.state_dict(), os.path.join(CURRENT_PATH, "resnet18.pt"))

# # %%

# y_pred = []
# y_true = []

# with torch.no_grad():
#     for batch_num, batch in enumerate(test_plant_dataloader):
#         data, target = batch['image'].to(device), batch['label'].to(device)

#         # For binary classification, transform labels to one-vs-rest
#         target = target.eq(3).type(torch.int64)

#         output = resnet18_model(data)
#         output = output.max(1, keepdim=True)[1]

#         output = torch.flatten(output).cpu().numpy()
#         y_pred.extend(output)

#         target = target.cpu().numpy()
#         y_true.extend(target)


# # Multi-class labels for confusion matrix
# # labels = ('CSV', 'FMV', 'Healthy', 'VD')

# # Binary labels for confusion matrix
# labels = ('Non-VD', 'VD')

# # Build confusion matrix
# cf_matrix = confusion_matrix(y_true, y_pred)

# df_cm = pd.DataFrame(
#     cf_matrix/np.sum(cf_matrix),
#     index = [i for i in labels],
#     columns = [i for i in labels]
# )
# plt.figure(figsize = (12,7))

# sn.heatmap(df_cm, annot=True)

# %%

# %%

# %%

param_grid_SVM = {
  'C': [0.1, 1, 10, 100, 1000],
  'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
  'gamma': ['auto', 'scale', 1, 0.1, 0.01, 0.001],
}

param_grid_RandomForest = {
  'n_estimators': [100, 200, 300, 400, 500],
  'criterion': ['gini', 'entropy'],
  'max_depth': [None, 3, 5],
  'min_samples_split': [2, 4],
}

# param_grid_XGBoost = {
#     'learning_rate': [0.05, 0.3, 0.7],
#     'gamma': [1, 0, 0.1, 0.01],
#     'max_depth': [None, 3, 6],
# }

param_grid_XGBoost = {
  'learning_rate': [0.05, 0.3, 0.7],
  'gamma': [1, 0, 0.1, 0.01],
  'max_depth': [None, 3, 6],
  'min_child_weight': [0, 1, 2],
}

master_df = plant_master_dataset.df

validation_size = int(0.15 * train_size)
train_size = train_size - validation_size

train_df, test_df = train_test_split(master_df, test_size=test_size)
train_df, validation_df = train_test_split(train_df, test_size=validation_size)

# keys, values = zip(*param_grid_SVM.items())
# keys, values = zip(*param_grid_RandomForest.items())
keys, values = zip(*param_grid_XGBoost.items())
combinations = [dict(zip(keys, p)) for p in product(*values)]

# %%

bow = BagOfWords(DATA_FOLDER_PATH, 4, 'SIFT', 'XGBoost')

k = [10, 100, 200, 500]

# features = bow.detect_features(train_df)
# %%

# results = []
columns = ['accuracy', 'f1_score', 'classifier']
# columns.extend(param_grid_SVM.keys())
# columns.extend(param_grid_RandomForest.keys())
columns.extend(param_grid_XGBoost.keys())
results_df = pd.DataFrame(columns=columns)

combinations_length = len(combinations)

for k_value in k:
  features = bow.detect_features(train_df, k_value)

  for i in range(0, combinations_length):
    combination = combinations[i]
    clf = bow.fit(train_df, features['img_features'], combination)
    result = bow.predict(test_df, clf, features['k'], features['voc'], features['standard_scaler'])
    # results.append({'C': combination['C'], 'kernel': combination['kernel'], 'gamma': combination['gamma'], 'accuracy': result['accuracy'], 'f1_score': result['f1_score'], 'classifier': clf})
    # results.append({'n_estimators': combination['n_estimators'], 'criterion': combination['criterion'], 'max_depth': combination['max_depth'], 'min_samples_split': combination['min_samples_split'], 'accuracy': result['accuracy'], 'f1_score': result['f1_score'], 'classifier': clf})

    model_filename = f'training_bow_models/training_model_{i}_with_k_{k_value}.save'
    pickle.dump(clf, open(model_filename, 'wb'))

    scaler_filename = f'training_bow_scalers/training_scaler_{i}_with_k_{k_value}.save'
    pickle.dump(features['standard_scaler'], open(scaler_filename, 'wb'))

    vocabulary_filename = f'training_bow_vocabularies/training_vocabulary_{i}_with_k_{k_value}.save'
    pickle.dump(features['voc'], open(vocabulary_filename, 'wb'))

    combination.update({'accuracy': result['accuracy'], 'f1_score': result['f1_score'], 'classifier': model_filename, 'k': features['k'], 'voc': vocabulary_filename, 'standard_scaler': scaler_filename})
    result_df = pd.DataFrame(combination, index=[0])

    results_df = pd.concat([results_df, result_df], ignore_index=True)

    print(f'{i+1} / {combinations_length} done')
  print(f'done for k={k_value}')

# %%

print(len(results_df))

# results_df = pd.DataFrame(results)
best_combination = results_df.sort_values(by='f1_score', ascending=False).reset_index()[:1]

# best_C = best_combination['C'].values[0]
# best_kernel = best_combination['kernel'].values[0]
# best_gamma = best_combination['gamma'].values[0]
# best_model = best_combination['classifier'].values[0]

# best_n_estimators = best_combination['n_estimators'].values[0]
# best_criterion = best_combination['criterion'].values[0]
# best_max_depth = best_combination['max_depth'].values[0]
# best_min_samples_split = best_combination['min_samples_split'].values[0]
# best_accuracy = best_combination['accuracy'].values[0]
# best_f1_score = best_combination['f1_score'].values[0]
best_model = pickle.load(open(best_combination['classifier'].values[0], 'rb'))

# print(f'best hyperparameter values: C: {best_C}, kernel: {best_kernel}, gamma: {best_gamma}')

print(f'best hyperparameter values:')

# For SVM
# print(best_combination[['C', 'kernel', 'gamma', 'k']])

# for RandomForest
# print(best_combination[['n_estimators', 'criterion', 'max_depth', 'min_samples_split', 'k']])

# for XGBoost
# print(best_combination[['learning_rate', 'gamma', 'max_depth']]) OLD
print(best_combination[['learning_rate', 'gamma', 'max_depth', 'min_child_weight', 'k']])

k = int(best_combination['k'].values[0])
voc = pickle.load(open(best_combination['voc'].values[0], 'rb'))
standard_scaler = pickle.load(open(best_combination['standard_scaler'].values[0], 'rb'))

# print(f'n_estimators: {best_n_estimators}')
# print(f'criterion: {best_criterion}')
# print(f'max_depth: {best_max_depth}')
# print(f'min_samples_split: {best_min_samples_split}')

# actual_result = bow.predict(validation_df, best_model, features['k'], features['voc'], features['standard_scaler'])
actual_result = bow.predict(validation_df, best_model, k, voc, standard_scaler)

# %%

print(actual_result)

# # %%

# bow_RandomForest = BagOfWords(DATA_FOLDER_PATH, 4, 'SIFT', 'RandomForest')
# clf_RandomForest = bow.fit(train_df, features['img_features'])
# result_RandomForest = bow.predict(test_df, clf_RandomForest, features['k'], features['voc'], features['standard_scaler'])
# print(result_RandomForest)

# result2_RandomForest = bow.predict(validation_df, clf_RandomForest, features['k'], features['voc'], features['standard_scaler'])
# print(result2_RandomForest)

# # %%

# keys, values = zip(*param_grid_RandomForest.items())
# combinations = [dict(zip(keys, p)) for p in product(*values)]

# combo = combinations[0]

# # %%

# param_grid_RandomForest = {
#   'n_estimators': [100, 200, 300, 400, 500],
#   'criterion': ['gini', 'entropy'],
#   'max_depth': [None, 3, 5, 10],
#   'min_samples_split': [2, 4, 10],
#   'min_samples_leaf': [1, 3, 5],
#   'max_features': ['auto', 'log2', 10, 50, 100]
# }

# columns = ['accuracy', 'f1_score', 'classifier']
# columns.extend(param_grid_RandomForest.keys())
# results_df = pd.DataFrame(columns=columns)

# combo.update({'accuracy': 0.5, 'f1_score': 0.5, 'classifier': 'lolled'})

# print(combo)

# result_df = pd.DataFrame(combo, index=[0])

# result_df.head()

# # %%

# results_df.head()

# # %%

# results_df = pd.concat([results_df, result_df], ignore_index=True)

# results_df.head()

# # %%

# derp = results_df.sort_values(by='f1_score', ascending=False).reset_index()[:1]

# print(derp[['n_estimators', 'criterion', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features']])

# # %%

# bow = BagOfWords(DATA_FOLDER_PATH, 4, 'SIFT', 'XGBoost')

# features = bow.detect_features(train_df)

# clf = bow.fit(train_df, features['img_features'])

# result = bow.predict(test_df, clf, features['k'], features['voc'], features['standard_scaler'])

# actual_result = bow.predict(validation_df, clf, features['k'], features['voc'], features['standard_scaler'])

# # %%

# print(result)

# # %%

# print(actual_result)


# ################################################################################################

# %%

bow = BagOfWords(DATA_FOLDER_PATH, 4, 'ORB', 'XGBoost')

k = [10, 100, 200, 500]

# features = bow.detect_features(train_df)
# %%

# results = []
columns = ['accuracy', 'f1_score', 'classifier']
# columns.extend(param_grid_SVM.keys())
# columns.extend(param_grid_RandomForest.keys())
columns.extend(param_grid_XGBoost.keys())
results_df = pd.DataFrame(columns=columns)

combinations_length = len(combinations)

for k_value in k:
  features = bow.detect_features(train_df, k_value)

  for i in range(0, combinations_length):
    combination = combinations[i]
    clf = bow.fit(train_df, features['img_features'], combination)
    result = bow.predict(test_df, clf, features['k'], features['voc'], features['standard_scaler'])
    # results.append({'C': combination['C'], 'kernel': combination['kernel'], 'gamma': combination['gamma'], 'accuracy': result['accuracy'], 'f1_score': result['f1_score'], 'classifier': clf})
    # results.append({'n_estimators': combination['n_estimators'], 'criterion': combination['criterion'], 'max_depth': combination['max_depth'], 'min_samples_split': combination['min_samples_split'], 'accuracy': result['accuracy'], 'f1_score': result['f1_score'], 'classifier': clf})

    model_filename = f'training_bow_models/training_model_{i}_with_k_{k_value}.save'
    pickle.dump(clf, open(model_filename, 'wb'))

    scaler_filename = f'training_bow_scalers/training_scaler_{i}_with_k_{k_value}.save'
    pickle.dump(features['standard_scaler'], open(scaler_filename, 'wb'))

    vocabulary_filename = f'training_bow_vocabularies/training_vocabulary_{i}_with_k_{k_value}.save'
    pickle.dump(features['voc'], open(vocabulary_filename, 'wb'))

    combination.update({'accuracy': result['accuracy'], 'f1_score': result['f1_score'], 'classifier': model_filename, 'k': features['k'], 'voc': vocabulary_filename, 'standard_scaler': scaler_filename})
    result_df = pd.DataFrame(combination, index=[0])

    results_df = pd.concat([results_df, result_df], ignore_index=True)

    print(f'{i+1} / {combinations_length} done')
  print(f'done for k={k_value}')

# %%

print(len(results_df))

best_combination = results_df.sort_values(by='f1_score', ascending=False).reset_index()[:1]

best_model = pickle.load(open(best_combination['classifier'].values[0], 'rb'))

print(f'best hyperparameter values:')

# For SVM
# print(best_combination[['C', 'kernel', 'gamma', 'k']])

# for RandomForest
# print(best_combination[['n_estimators', 'criterion', 'max_depth', 'min_samples_split', 'k']])

# for XGBoost
# print(best_combination[['learning_rate', 'gamma', 'max_depth']]) OLD
print(best_combination[['learning_rate', 'gamma', 'max_depth', 'min_child_weight', 'k']])

k = int(best_combination['k'].values[0])
voc = pickle.load(open(best_combination['voc'].values[0], 'rb'))
standard_scaler = pickle.load(open(best_combination['standard_scaler'].values[0], 'rb'))

actual_result = bow.predict(validation_df, best_model, k, voc, standard_scaler)

# %%

print(actual_result)
