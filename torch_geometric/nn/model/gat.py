import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    def __init__(self,num_features,num_classes,hidden_size=8,num_heads=8,dropout=0.3):
        super(GAT, self).__init__()
        self.att1 = GATConv(num_features, hidden_size, heads=num_heads, dropout=dropout)
        self.att2 = GATConv(hidden_size * num_heads, num_classes, dropout=dropout)
        self.dropout = dropout
    def forward(self,data):
        x = F.dropout(data.x, p=self.dropout, training=self.training)
        x = F.elu(self.att1(x, data.edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.att2(x, data.edge_index)
        return F.log_softmax(x, dim=1)