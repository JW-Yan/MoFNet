
from captum.attr import IntegratedGradients
from captum.attr import LayerConductance
from captum.attr import NeuronConductance
import torch
import numpy as np


#Loading your Model
class Layer_importance:
  """
  Training function performs training of MofNet based on the input hyperparameters.
  :param learning_rate LR: The learning rate to use for gradient descent
  :param L1REG: L1 Regularization
  :param L2REG: L2 Regularization
  :param H2: Size of Fully connected Hidden Layer
  :param H3: Size of Fully connected Hidden Layer
  :param epochs: Number of epochs
  :param dout_ratio: Dropout ratio
  :return test_accuracy, auc_score, specificity, recall, precision, f1, X_test , best_model_path,
    returns to the main function to extract weights
  """
  def __init__(self, adj_1, adj_2, model, X_test):
    self.adj_1 = adj_1
    self.adj_2 = adj_2
    self.model = model
    self.X_test = X_test

  def layer1(self):
    input_feature_name = list(self.adj_1.index) # for layer 1 adj1 and for layer 2 adj2
    self.model.adj_1 = self.model.adj_1.cpu() #change adj_1 -> adj_2 for layer 2
    intergrated = IntegratedGradients(self.model.cpu())
    test_input = torch.from_numpy(self.X_test).type(torch.FloatTensor)
    attr, delta = intergrated.attribute(test_input, return_convergence_delta=True)
    attr = attr.detach().numpy()

    importances = dict(zip(input_feature_name, np.mean(abs(attr), axis=0)))

    outFile = 'layer1_importance_of_input_feature.csv'

    print('Iuput feature importance {}'.format(outFile))
    with open(outFile, 'w') as f:
        for key in importances.keys():
            f.write("%s,%s\n"%(key,importances[key]))

    cond = LayerConductance(self.model, self.model.MoFNet1) #for layer 2 change it to MoFNet2

    cond_vals = cond.attribute(test_input)
    cond_vals = cond_vals.detach().numpy()

    importances_layer1 = dict(zip(self.adj_1.columns.tolist(), \
                np.mean(abs(cond_vals), axis=0))) #change adj1 -> adj2 for layer 2

    outFile = 'layer1_node_importance.csv'

    print('Transparent layer node importance {}'.format(outFile))
    with open(outFile, 'w') as f:
        for key in importances_layer1.keys():
            f.write("%s,%s\n"%(key,importances_layer1[key]))

    # layer 1
    neuron_cond = NeuronConductance(self.model, self.model.MoFNet1) 
    outFile = 'layer1_connection_weights.csv'
    with open(outFile, 'w') as f:
        print('Connection weights')

        for idx in self.adj_1.columns.tolist(): 
            neuron_cond_vals = neuron_cond.attribute(test_input, \
                neuron_selector=self.adj_1.columns.tolist().index(idx)) 
            importances_neuron = dict(zip(input_feature_name, \
                abs(neuron_cond_vals.mean(dim=0).detach().numpy())))
            importances_neuron = {key:val for key, val in \
                                importances_neuron.items() if val != 0}
            for key in importances_neuron.keys():
                f.write("%s,%s,%s\n"%(idx,key,importances_neuron[key]))

  def layer2(self):
    input_feature_name = list(self.adj_2.index) # for layer 1 adj1 and for layer 2 adj2
    self.model.adj_2 = self.model.adj_2.cpu() #change adj_1 -> adj_2 for layer 2
    intergrated = IntegratedGradients(self.model.cpu())
    test_input = torch.from_numpy(self.X_test).type(torch.FloatTensor)
    attr, delta = intergrated.attribute(test_input, return_convergence_delta=True)
    attr = attr.detach().numpy()

    importances = dict(zip(input_feature_name, np.mean(abs(attr), axis=0)))

    outFile = 'layer2_importance_of_input_feature.csv'

    print('Iuput feature importance {}'.format(outFile))
    with open(outFile, 'w') as f:
        for key in importances.keys():
            f.write("%s,%s\n"%(key,importances[key]))

    cond = LayerConductance(self.model, self.model.MoFNet2)

    cond_vals = cond.attribute(test_input)
    cond_vals = cond_vals.detach().numpy()

    importances_layer1 = dict(zip(self.adj_2.columns.tolist(), \
                np.mean(abs(cond_vals), axis=0))) #change adj1 -> adj2 for layer 2

    outFile = 'layer2_node_importance.csv'

    print('Transparent layer node importance {}'.format(outFile))
    with open(outFile, 'w') as f:
        for key in importances_layer1.keys():
            f.write("%s,%s\n"%(key,importances_layer1[key]))

    # layer 2
    neuron_cond = NeuronConductance(self.model, self.model.MoFNet2) 
    outFile = 'layer2_connection_weights.csv'
    with open(outFile, 'w') as f:
        print('Connection weights')
         
        for idx in self.adj_2.columns.tolist(): 
            neuron_cond_vals = neuron_cond.attribute(test_input, \
                neuron_selector=self.adj_2.columns.tolist().index(idx)) 
            importances_neuron = dict(zip(input_feature_name, \
                abs(neuron_cond_vals.mean(dim=0).detach().numpy())))
            importances_neuron = {key:val for key, val in \
                                importances_neuron.items() if val != 0}
            for key in importances_neuron.keys():
                f.write("%s,%s,%s\n"%(idx,key,importances_neuron[key]))
