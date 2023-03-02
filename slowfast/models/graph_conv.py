import math
from unittest import skip

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import torch.nn as nn
from .graph import Graph
class GraphConvolution(Module):
    """
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        #if input.size()[0]==4:
        #nput = torch.cat([input,input],dim=0)
        #print(input.size())
        support = torch.matmul(input, self.weight)
        #adj = adj.expand(input.size()[0],-1,-1)
        #print(support.size())
        output = torch.matmul(support.permute(0,2,1),adj).permute(0,2,1)
        #print(output.size())
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, outfeat, dropout=0):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, outfeat)
        self.dropout = dropout
        self.Graph = Graph()
    def forward(self, x):
        adj = torch.Tensor(self.Graph.A).cuda()
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        #print(x.size())
        return F.log_softmax(x, dim=1)