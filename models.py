import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
    '''
    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
    '''
    
    def forward(self, x, adj, split_data_indexs = None):
        if split_data_indexs != None:
            x = F.relu(self.gc1(x, adj[split_data_indexs[0]]))        
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.gc2(x, adj[split_data_indexs[1]][:,split_data_indexs[0]]) 
        else:
            x = F.relu(self.gc1(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

