import torch
import torch.nn.functional as F
from torch_geometric.nn import SplineConv



class SplineCnn(torch.nn.Module):
    def __init__(self,num_features,hidden_size,num_classes,dim=1,kernel_size=2):
        super(SplineCnn, self).__init__()
        self.conv1 = SplineConv(num_features, hidden_size, dim=dim, kernel_size=kernel_size)
        self.conv2 = SplineConv(16, num_classes, dim=dim, kernel_size=kernel_size)

    def forward(self,data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.dropout(x, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        return F.log_softmax(x, dim=1)
