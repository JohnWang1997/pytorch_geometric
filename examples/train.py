import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCN



def train(data,model,optimizer):
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()

def test(data,model,optimizer):
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

def main():
    dataset = 'Cora'
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
    data = Planetoid(path, dataset, T.NormalizeFeatures())[0]
    model = GCN(data.num_features,16,data.num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, data = model.to(device), data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-3)
    for epoch in range(1, 201):
        train(data,model,optimizer)
        log = 'Epoch: {:03d}, Train: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, *test(data,model,optimizer)))

if __name__ == "__main__":
    main()