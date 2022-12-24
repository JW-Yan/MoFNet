import torch
import torch.utils.data
import torch.nn as nn
from torch._C import device
from torch.utils.data import Dataset, TensorDataset, DataLoader
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.metrics import recall_score, precision_score, roc_auc_score, roc_curve
import math
import pandas as pd
import numpy as np
import scipy as sp
import random
import matplotlib.pyplot as plt
from model import MoFNetLayer, Net



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess(x, y):
    return x.float().to(device), y.int().reshape(-1, 1).to(device)

class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))

def loss_batch(model, loss_fn, xb, yb, L1REG , opt=None):
    yhat = model(xb)
    loss = loss_fn(yhat, yb.float())
    for param in model.MoFNet1.parameters():
            loss += L1REG * torch.sum(torch.abs(param))
    
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
        
    yhat_class = np.where(yhat.detach().cpu().numpy()<0.5, 0, 1)
    accuracy = accuracy_score(yb.detach().cpu().numpy(), yhat_class)

    return loss.item(), accuracy

def fit(epochs, model, loss_fn, opt, train_dl, val_dl, L1REG):
    train_loss = []
    train_accuracy = []
    val_loss = []
    val_accuracy = []
    
    for epoch in range(epochs):
        model.train()
        losses, accuracies = zip(
            *[loss_batch(model, loss_fn, xb, yb, L1REG, opt) for xb, yb in train_dl]
        )
        train_loss.append(np.mean(losses))
        train_accuracy.append(np.mean(accuracies))

        model.eval()
        with torch.no_grad():
            losses, accuracies = zip(
                *[loss_batch(model, loss_fn, xb, yb, L1REG) for xb, yb in val_dl]
            )
        val_loss.append(np.mean(losses))
        val_accuracy.append(np.mean(accuracies))
        
        if (epoch % 10 == 0):
            print("epoch %s" %epoch, np.mean(losses),np.mean(train_accuracy), \
                  np.mean(accuracies))
    
    return train_loss, train_accuracy, val_loss, val_accuracy


def training(LR, L1REG, L2REG,H1, H2, H3,D_in,T1, epochs, n_seed,D_out, dout_ratio, X, y, adj1, adj2, data_path):
    """
    Training function performs training of MofNet based on the input hyperparameters
    :param learning_rate LR: The learning rate to use for gradient descent
    :param L1REG: L1 Regularization
    :param L2REG: L2 Regularization
    :param H2: Size of Fully connected Hidden Layer
    :param H3: Size of Fully connected Hidden Layer
    :param epochs: Number of epochs
    :param dout_ratio: Dropout ratio
    :return test_accuracy, auc_score, specificity, recall, precision, f1, X_test , best_model_path
      returns to the main function to extract weights
    """

    np.random.seed(n_seed)
    torch.manual_seed(n_seed)
    random.seed(n_seed)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
						random_state=n_seed, stratify=y)
    
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.2, 
						random_state=n_seed, stratify=y_test)
						

    #Loading the data
    X_train = np.array(X_train)
    X_val = np.array(X_val)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)

    adj1 = np.array(adj1)

    adj2 = np.array(adj2)

	
    X_train, y_train, X_val, y_val = map(torch.tensor,(X_train, y_train, X_val, y_val))

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)

    train_dl = DataLoader(dataset=train_ds, batch_size=30) 
    val_dl = DataLoader(dataset=val_ds) 

    train_dl = WrappedDataLoader(train_dl, preprocess)
    val_dl = WrappedDataLoader(val_dl, preprocess)

    loss = nn.BCELoss()
    opt = torch.optim.Adam

    adj_1 = torch.from_numpy(adj1).float().to(device)
    adj_2 = torch.from_numpy(adj2).float().to(device)

    model = Net(adj_1, adj_2, D_in, T1, H1, H2, H3, D_out, dout_ratio).to(device)

    opt = opt(model.parameters(), lr=LR, weight_decay=L2REG, \
            betas=(0.9, 0.999), amsgrad=True)

    train_loss, train_accuracy, val_loss, val_accuracy = fit(epochs, \
            model, loss, opt, train_dl, val_dl, L1REG)


    print('Training process has finished. Saving trained model.')
    print('Starting testing')

    best_model_path = data_path+"saved_model.pth"
    torch.save(model, best_model_path)


    with torch.no_grad():
        x_tensor_test = torch.from_numpy(X_test).float().to(device)
        model.eval()
        yhat = model(x_tensor_test)
        y_hat_class = np.where(yhat.cpu().numpy()<0.5, 0, 1)
        test_accuracy = accuracy_score(y_test.reshape(-1,1), y_hat_class)
        f1 = f1_score(y_test.reshape(-1,1), y_hat_class)
        recall = recall_score(y_test.reshape(-1,1), y_hat_class)
        precision = precision_score(y_test.reshape(-1,1), y_hat_class)
        fpr, tpr, threshold = roc_curve(y_test.reshape(-1,1), y_hat_class)
        auc_score = roc_auc_score(y_test.reshape(-1,1), y_hat_class)
        tn, fp, fn, tp = confusion_matrix(y_test.reshape(-1,1), y_hat_class).ravel()
        specificity = tn / (tn+fp)

        print('Accuracy: %d %%' % (100.0 * test_accuracy))
        print("auc_score:",auc_score)
        print("specificity:",specificity)
        print("Recall:",recall)
        print("Precision:",precision)
        print("F1 score:",f1)
        print('--------------------------------')
         
    return test_accuracy, auc_score, specificity, recall, precision, f1, X_test , best_model_path #layer_class
