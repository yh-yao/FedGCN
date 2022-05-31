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
    
    
    
    
class MultiLayerGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, NumLayers):
        super(MultiLayerGCN, self).__init__()
        
        self.dropout = dropout
        self.gcns = nn.ModuleList()
        self.num_layers = NumLayers
        #first layer
        if NumLayers <= 1:
            self.gcns.append(GraphConvolution(nfeat, nclass))
        else:
            self.gcns.append(GraphConvolution(nfeat, nhid))
            #remaining layers                 
            for i in range(1, NumLayers-1):
                self.gcns.append(GraphConvolution(nhid, nhid))
            self.gcns.append(GraphConvolution(nhid, nclass))
                          
        
    
    def forward(self, x, adj):
        if self.num_layers > 1:
            for i in range(self.num_layers-1):                     
                x = F.relu(self.gcns[i](x, adj))
                x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcns[-1](x, adj)
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
    '''

