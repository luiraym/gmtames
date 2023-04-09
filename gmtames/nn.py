'''
gmtames
nn.py

Raymond Lui
13-July-2022
'''




import os
import logging
import json
import pickle
from timeit import default_timer as timer

import math
import itertools
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score

from gmtames.data import MODELLING_DATASET_TYPE_LIST




RANDOM_SEED = 20041955

FILTER_VALUE = -2

SEARCH_SPACE = {
    'n_layers': [2, 3, 4],
    'architecture': ['linear', 'top', 'bottom', 'outer', 'inner'],
    'lr': [0.001, 0.0001],
    'n_epochs': [100, 150, 200]
}

NN_ARCH = {
    'linear_2l': [683, 342],
    'linear_3l': [768, 513, 257],
    'linear_4l': [819, 615, 410, 206],
    'top_2l': [878, 586],
    'top_3l': [956, 819, 547],
    'top_4l': [991, 925, 793, 529],
    'bottom_2l': [439, 147],
    'bottom_3l': [478, 206, 69],
    'bottom_4l': [496, 232, 100, 34],
    'outer_2l': [768, 257],
    'outer_3l': [854, 513, 171],
    'outer_4l': [922, 717, 308, 103],
    'inner_2l': [615, 410],
    'inner_3l': [683, 512, 342],
    'inner_4l': [709, 552, 473, 316]
}




logger = logging.getLogger('gmtames.nn')

torch.manual_seed(RANDOM_SEED)




# BLOCK 1: NEURAL NETWORK HELPER FUNCTIONS

def checkDevice(device):
    if device.lower() == 'titanv': return 'cuda:0'
    else: logger.warning('%s is not a supported device; using CPU instead' % device); return 'cpu'


def loadModellingDataset(modelling_datasets, tasks, dataset_type, device):
    assert dataset_type in MODELLING_DATASET_TYPE_LIST, 'dataset_type must be one of "train", "val", or "test"'
    n_tasks = len(tasks)

    dataset_selected = modelling_datasets[dataset_type]
    i = dataset_selected.iloc[:, 0].values
    X = dataset_selected.iloc[:, 1:-(n_tasks)].values
    y = dataset_selected.iloc[:, -(n_tasks):].values

    X = torch.tensor(X).float().to(device)
    y = torch.tensor(y).float().to(device)

    y = torch.nan_to_num(y, nan=FILTER_VALUE)

    logger.info('%s descriptors loaded (features, instances):|%s|%s' % (dataset_type.capitalize(), X.size(dim=1), X.size(dim=0)))
    logger.info('%s endpoints loaded (tasks, instances):|%s|%s' % (dataset_type.capitalize(), y.size(dim=1), y.size(dim=0)))

    tensor_dataset = TensorDataset(X, y)
    dataloader = DataLoader(tensor_dataset, batch_size=len(tensor_dataset))

    return i, dataloader


def filterMissingLabelsWrapper(metric_func, y_true, y_pred, y_true_first):
    y_true_filtered = y_true[y_true != FILTER_VALUE]
    y_pred_filtered = y_pred[y_true != FILTER_VALUE]
    
    if y_true_first:
        metric = metric_func(y_true_filtered, y_pred_filtered)
    else:
        metric = metric_func(y_pred_filtered, y_true_filtered)

    return metric

   
def calculateMultitaskMetricWrapper(metric_func, tasks, y_true, y_pred, y_true_first=True):
    all_metrics = []

    for task_idx in range(len(tasks)):
        y_true_task = y_true[:, task_idx]
        y_pred_task = y_pred[:, task_idx]

        metric = filterMissingLabelsWrapper(metric_func, y_true_task, y_pred_task, y_true_first)
        all_metrics.append(metric)

    return all_metrics




# BLOCK 2: NEURAL NETWORK ARCHITECTURE CLASSES AND TRAINING FUNCTIONS

def trainNTaskNeuralNetwork(model, tasks, train_dataloader, n_epochs, lr):
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(n_epochs):
        for X_train, y_train in train_dataloader:
            # Forward pass
            y_pred = model.forward(X_train)
           
            # List of n task losses
            loss_list = calculateMultitaskMetricWrapper(nn.BCELoss(), tasks, y_train, y_pred, y_true_first=False)

            # Stack the n tensors in the list into one tensor containing n values
            loss_tensor = torch.stack(loss_list)

            # Sum the n values in the tensor to form a one value tensor
            total_loss = torch.sum(loss_tensor)
            
            # Backpropagation and weight update
            optimiser.zero_grad()
            total_loss.backward()
            optimiser.step()

    return model


class NTaskNeuralNetworkFromOptuna(nn.Module):
    def __init__(self, trial, n_input, tasks):
        super(NTaskNeuralNetworkFromOptuna, self).__init__()
        self.trial = trial
        self.n_input = n_input
        self.tasks = tasks

        in_features = self.n_input
        n_layers = self.trial.suggest_int('n_layers', 1, 4)
        architecture = self.trial.suggest_categorical('architecture', ['linear', 'top', 'bottom', 'outer', 'inner'])
        n_tasks = len(self.tasks)

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            out_features = NN_ARCH['{}_{}l'.format(architecture, n_layers)][i]
            self.layers.append(nn.Linear(in_features, out_features))
            in_features = out_features
        self.outputs = nn.ModuleList(nn.Linear(out_features, 1) for i in range(n_tasks))

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))

        preds = []
        for output in self.outputs:
            pred = torch.sigmoid(output(x))
            preds.append(pred)

        return torch.cat(preds, dim=1).float()


def trainNTaskNeuralNetworkFromOptuna(trial, train_dataloader, tasks, path_to_output, device):
    # Instantiate neural network architecture with Optuna trial object
    n_input = list(train_dataloader)[0][0].size(dim=1)
    model = NTaskNeuralNetworkFromOptuna(trial, n_input, tasks).to(device)
    
    # Define training parameters with Optuna trial
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    n_epochs = trial.suggest_int('n_epochs', 50, 200)

    return trainNTaskNeuralNetwork(model, tasks, train_dataloader, n_epochs, lr)


class NTaskNeuralNetworkFromDict(nn.Module):
    def __init__(self, hyperparam_dict, n_input, tasks):
        super(NTaskNeuralNetworkFromDict, self).__init__()
        self.hyperparam_dict = hyperparam_dict
        self.n_input = n_input
        self.tasks = tasks

        in_features = self.n_input
        n_layers = self.hyperparam_dict['n_layers']
        architecture = self.hyperparam_dict['architecture']
        n_tasks = len(self.tasks)

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            out_features = NN_ARCH['{}_{}l'.format(architecture, n_layers)][i]
            self.layers.append(nn.Linear(in_features, out_features))
            in_features = out_features
        self.outputs = nn.ModuleList(nn.Linear(out_features, 1) for i in range(n_tasks))

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))

        preds = []
        for output in self.outputs:
            pred = torch.sigmoid(output(x))
            preds.append(pred)

        return torch.cat(preds, dim=1).float()


def trainNTaskNeuralNetworkFromDict(hyperparam_dict, train_dataloader, tasks, path_to_output, device):
    # Instatiate neural network with hyperparameter dict
    n_input = list(train_dataloader)[0][0].size(dim=1)
    model = NTaskNeuralNetworkFromDict(hyperparam_dict, n_input, tasks).to(device)

    # Define training parameters with hyperparameter dict
    lr = hyperparam_dict['lr']
    n_epochs = hyperparam_dict['n_epochs']

    return trainNTaskNeuralNetwork(model, tasks, train_dataloader, n_epochs, lr)




# BLOCK 3: NEURAL NETWORK MODEL FUNCTIONS

def evaluateNeuralNetwork(model, dataloader_with_id, tasks, save_predictions=False, path_to_output=None):
    # Unpack dataloader_with_id
    gmtamesqsar_id = dataloader_with_id[0]
    dataloader = dataloader_with_id[1]
    
    model.eval()
    with torch.no_grad():
        for X, y_true in dataloader:
            # Make predictions
            y_pred = model.forward(X)
           
            # Detach y's from graph (no gradients), send back to CPU, and convert to numpy array
            y_pred = y_pred.detach().cpu().numpy()
            y_true = y_true.detach().cpu().numpy()

            # Save predictions
            if save_predictions:
                assert path_to_output != None, 'Need to specify path_to_output to save test predictions'
                assert gmtamesqsar_id.shape[0] == y_true.shape[0] == y_pred.shape[0], 'Dimension 0 mismatch found when checking test_id/y_true/y_pred shapes'
                
                path_to_test_predictions = path_to_output / ('test_predictions/')
                path_to_test_predictions.mkdir(exist_ok=True)
                filename_id = '_'.join(tasks)
                
                test_predictions = {'gmtamesqsar_id': gmtamesqsar_id, 'y_true': y_true, 'y_pred': y_pred}
                with open(path_to_test_predictions / ('%s_test_predictions.pkl' % filename_id), 'wb') as f: pickle.dump(test_predictions, f)

            # Compute balanced accuracy and ROC AUC
            balacc = calculateMultitaskMetricWrapper(balanced_accuracy_score, tasks, y_true, np.rint(y_pred))
            rocauc = calculateMultitaskMetricWrapper(roc_auc_score, tasks, y_true, y_pred)

    return balacc, rocauc


def saveNeuralNetwork(model, hyperparam_dict, tasks, path_to_output):
    # Define path and filename
    path_to_final_models = path_to_output / ('final_models/')
    path_to_final_models.mkdir(exist_ok=True)
    filename_id = '_'.join(tasks)

    # Save hyperparameter dictionary
    hyperparam_dict['tasks'] = tasks
    with open(path_to_final_models / ('%s_hyperparam_dict.json' % filename_id), 'w') as f: json.dump(hyperparam_dict, f)

    # Save state dictionary
    torch.save(model.state_dict(), path_to_final_models / ('%s_state_dict.pt' % filename_id))




# BLOCK 4: NEURAL NETWORK META FUNCTIONS

def computeOptunaObjective(trial, train_dataloader, val_dataloader_with_id, tasks, path_to_output, device):
    timer_start = timer()
    
    model = trainNTaskNeuralNetworkFromOptuna(trial, train_dataloader, tasks, path_to_output, device)
    balacc, rocauc = evaluateNeuralNetwork(model, val_dataloader_with_id, tasks)
    
    timer_end = timer()
    timing = timer_end - timer_start

    logger.info('Trial %s params:|%s' % (trial.number, trial.params))
    logger.info('Trial %s balacc:|%s' % (trial.number, balacc))
    logger.info('Trial %s rocauc:|%s' % (trial.number, rocauc))
    logger.info('Trial %s timings:|%s' % (trial.number, timing))

    objective = np.mean(balacc)
    logger.info('Trial %s objective:|%s' % (trial.number, objective))

    return objective


def testFinalModel(hyperparam_dict, trainval_dataloader, test_dataloader_with_id, tasks, path_to_output, device):
    model = trainNTaskNeuralNetworkFromDict(hyperparam_dict, trainval_dataloader, tasks, path_to_output, device)
    balacc, rocauc = evaluateNeuralNetwork(model, test_dataloader_with_id, tasks, save_predictions=True, path_to_output=path_to_output)
    
    logger.info('Final model test balacc:|%s' % balacc)
    logger.info('Final model test rocauc:|%s' % rocauc)

    saveNeuralNetwork(model, hyperparam_dict, tasks, path_to_output)
