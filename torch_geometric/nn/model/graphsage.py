import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import SAGEPoolAggr
class GraphSage(torch.nn.Module):
    def __init__(self,num_features,hidden_size,num_classes,dropout=0.3):
        super(GraphSage, self).__init__()
        self.age = SAGEPoolAggr(num_features, hidden_size)
        self.conv1 = SAGEConv(self.age, hidden_size)
        self.conv2 = SAGEConv(self.age, num_classes)


    def forward(self,data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, data.edge_index)
        return F.log_softmax(x, dim=1)