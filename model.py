
import torch
import torch.nn as nn
import math

class MoFNetLayer(nn.Module):
    def __init__(self, in_dims, out_dims, bias=True):
        super(MoFNetLayer, self).__init__()
        self.in_dims = in_dims
        self.in_dims = out_dims
        self.weight = nn.Parameter(torch.Tensor(out_dims, in_dims))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_dims))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input, adj):
        return input.matmul(self.weight.t() * adj) + self.bias

    def extra_repr(self):
        return 'in_dims={}, out_dims={}, bias={}'.format(
            self.in_dims, self.out_dims, self.bias is not None
        )

class Net(nn.Module):
    def __init__(self, adj_1, adj_2, D_in, T1, H1, H2, H3, D_out,dout_ratio):
        super(Net, self).__init__()
        self.adj_1 = adj_1  # only gene expression with snp
        self.adj_2 = adj_2  # only gene expression with protein expression 
        self.MoFNet1 = MoFNetLayer(D_in, T1)
        self.MoFNet2 = MoFNetLayer(T1+H1, H1)
        self.dropout1 = torch.nn.Dropout(dout_ratio)
        self.linear2 = torch.nn.Linear(H1, H2)
        self.linear3 = torch.nn.Linear(H2, H3)
        self.linear4 = torch.nn.Linear(H3, D_out)

    def forward(self, x):
        t1 = self.MoFNet1(x[:, 186:1751], self.adj_1).relu()
        x_2 = torch.cat((x[:, 0:186], t1),1)
        h1 = self.MoFNet2(x_2, self.adj_2).relu()
        h1 = self.dropout1(h1)
        h2 = self.linear2(h1).relu()
        h2 = self.dropout1(h2)
        h3 = self.linear3(h2).relu()
        y_pred = self.linear4(h3).sigmoid()
        return y_pred
