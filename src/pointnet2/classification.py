import argparse
import os.path as osp
import time
from tqdm import tqdm
import pickle
import numpy as np

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.nn import PointConv, radius_graph, fps, global_max_pool
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=50)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--dataset', type=str, default='ModelNet10', choices=['ModelNet10', 'ModelNet40'])
parser.add_argument('--num_points', type=int, default=1024)
args = parser.parse_args()


class Net(torch.nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()

        nn = Seq(Lin(3, 64), ReLU(), Lin(64, 64))
        self.conv1 = PointConv(local_nn=nn)

        nn = Seq(Lin(67, 128), ReLU(), Lin(128, 128))
        self.conv2 = PointConv(local_nn=nn)

        nn = Seq(Lin(131, 256), ReLU(), Lin(256, 256))
        self.conv3 = PointConv(local_nn=nn)

        self.lin1 = Lin(256, 256)
        self.lin2 = Lin(256, 256)
        self.lin3 = Lin(256, num_classes)

    def forward(self, pos, batch):
        radius = 0.2
        edge_index = radius_graph(pos, r=radius, batch=batch)
        x = F.relu(self.conv1(None, pos, edge_index))

        idx = fps(pos, batch, ratio=0.5)
        x, pos, batch = x[idx], pos[idx], batch[idx]

        radius = 0.4
        edge_index = radius_graph(pos, r=radius, batch=batch)
        x = F.relu(self.conv2(x, pos, edge_index))

        idx = fps(pos, batch, ratio=0.25)
        x, pos, batch = x[idx], pos[idx], batch[idx]

        radius = 1
        edge_index = radius_graph(pos, r=radius, batch=batch)
        x = F.relu(self.conv3(x, pos, edge_index))

        x = global_max_pool(x, batch)

        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        return F.log_softmax(x, dim=-1)


def get_dataset(num_points):
    name = args.dataset
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', name)
    pre_transform = T.NormalizeScale()
    transform = T.SamplePoints(num_points)

    train_dataset = ModelNet(
        path,
        name=name[-2:],  ## '10' or '40'
        train=True,
        transform=transform,
        pre_transform=pre_transform)
    test_dataset = ModelNet(
        path,
        name=name[-2:],   ## '10' or '40'
        train=False,
        transform=transform,
        pre_transform=pre_transform)

    return train_dataset, test_dataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run(train_dataset, test_dataset, model, epochs, batch_size, lr,
        lr_decay_factor, lr_decay_step_size, weight_decay):

    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    train_loss = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(1, epochs + 1):
        print("epoch", epoch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        epoch_loss, train_acc = train(model, optimizer, train_loader, device)
        test_acc = test(model, test_loader, device)

        train_loss.append(epoch_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()

        print('Epoch: {:03d}, Test: {:.4f}, Duration: {:.2f}'.format(
            epoch, test_acc, t_end - t_start))

        if epoch % lr_decay_step_size == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_decay_factor * param_group['lr']

    np.save('results/train_loss_'+args.dataset+'_'+str(args.num_points)+'.npy', np.array(train_loss))
    np.save('results/train_accuracy_'+args.dataset+'_'+str(args.num_points)+'.npy', np.array(train_accuracies))
    np.save('results/test_accuracy_'+args.dataset+'_'+str(args.num_points)+'.npy', np.array(test_accuracies))


def train(model, optimizer, train_loader, device):
    model.train()
    epoch_loss = 0.0
    correct = 0
    for data in tqdm(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data.pos, data.batch)
        correct += out.max(1)[1].eq(data.y).sum().item()
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss/len(train_loader), correct/len(train_loader.dataset)



def test(model, test_loader, device):
    model.eval()

    correct = 0
    for data in test_loader:
        data = data.to(device)
        pred = model(data.pos, data.batch).max(1)[1]
        correct += pred.eq(data.y).sum().item()
    test_acc = correct / len(test_loader.dataset)

    return test_acc


train_dataset, test_dataset = get_dataset(num_points=args.num_points)
model = Net(train_dataset.num_classes)
run(train_dataset, test_dataset, model, args.epochs, args.batch_size, args.lr,
    args.lr_decay_factor, args.lr_decay_step_size, args.weight_decay)

