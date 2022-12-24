"""
Usage:
    Example running in command line with required command arguments:
    	python mofnet_main.py /the_path_to_your_dataset/
"""

import sys
import pandas as pd
import utils
from train import training
import torch
from utils import Layer_importance

# get the list of all users actions
data_path = sys.argv[1]
print('User input data path is:', data_path)

if data_path[-1] != '/':
    print('Please add / at the end of each path.\n')
    sys.exit(1) # abort

if len(sys.argv) < 1:
    print('You might want to use the following format:')
    print('python main.py data_path')
    sys.exit(1) # abort because of no enough arguments

# load the input data
X = pd.read_csv(data_path + 'X.csv')
print(X.shape)
y = pd.read_csv(data_path + 'y.csv')
print(y.shape)
adj1_tmp = pd.read_csv(data_path + 'adj1.csv')
adj1 = adj1_tmp.set_index('probe')
adj2_tmp = pd.read_csv(data_path + 'adj2.csv')
adj2 = adj2_tmp.set_index('probe')

# pre-set hyper-parameters
LR, L1REG, L2REG = 0.0006, 0.005, 0.0005
epochs, n_seed, dout_ratio = 10, 66, 0.5

# pre-set Network structure
D_in, T1, H1, H2, H3, D_out = 743+822, 743, 186, 96, 16, 1 

test_accuracy, auc_score, specificity, recall, precision, f1, X_test, saved_model = training( LR, 
L1REG, L2REG,H1, H2, H3, D_in, T1, epochs, n_seed,D_out, dout_ratio, X, y, adj1, adj2, data_path)


print('Saved_model variable:', saved_model)
model = torch.load(saved_model, map_location=torch.device('cpu'))

layer_weight = Layer_importance(adj1, adj2, model, X_test)

layer_weight.layer1()

layer_weight.layer2()
