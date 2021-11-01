import os.path as osp
import time
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
from torch_geometric.datasets import ShapeNet
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.nn import knn_interpolate
from torch_geometric.utils import intersection_and_union as i_and_u

import argparse
from torch.nn import Sequential as Seq, Linear as Lin, BatchNorm1d as BN, ReLU
from torch_geometric.nn import PointConv, radius_graph, fps, global_max_pool, radius


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=12)
parser.add_argument('--lr', type=float, default=0.001)
args = parser.parse_args()


category = ["Airplane", "Bag" , "Cap", "Car", "Chair", "Earphone", "Guitar", "Knife"]  # Pass in None to train on all categories.
path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'ShapeNet')
transform = T.Compose([
    T.RandomTranslate(0.01),
    T.RandomRotate(15, axis=0),
    T.RandomRotate(15, axis=1),
    T.RandomRotate(15, axis=2)
])
pre_transform = T.NormalizeScale()
train_dataset = ShapeNet(path, category, split='trainval', transform=transform,
                         pre_transform=pre_transform)
test_dataset = ShapeNet(path, category, split='test',
                        pre_transform=pre_transform)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                          num_workers=6)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                         num_workers=6)



class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv((x, x[idx]), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super(GlobalSAModule, self).__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])


class FPModule(torch.nn.Module):
    def __init__(self, k, nn):
        super(FPModule, self).__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip


class Net(torch.nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()

        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(0.2, 0.2, MLP([3 + 3, 64, 64]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([64 + 3, 128, 128]))
        self.sa3_module = GlobalSAModule(MLP([128 + 3, 256, 256]))

        self.fp3_module = FPModule(1, MLP([256 + 128, 128, 128]))
        self.fp2_module = FPModule(3, MLP([128 + 64, 128, 64]))
        self.fp1_module = FPModule(3, MLP([64 + 3, 64, 64]))

        self.lin1 = torch.nn.Linear(64, 64)
        self.lin2 = torch.nn.Linear(64, 64)
        self.lin3 = torch.nn.Linear(64, num_classes)

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)

        fp3_out = self.fp3_module(*sa3_out, *sa2_out)
        fp2_out = self.fp2_module(*fp3_out, *sa1_out)
        x, _, _ = self.fp1_module(*fp2_out, *sa0_out)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        return F.log_softmax(x, dim=-1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(train_dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def train(model):
    model.train()

    total_loss = correct_nodes = total_nodes = 0
    for data in tqdm(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct_nodes += out.argmax(dim=1).eq(data.y).sum().item()
        total_nodes += data.num_nodes

    return (total_loss/len(train_loader), correct_nodes/total_nodes)


@torch.no_grad()
def test(loader):
    model.eval()

    correct_nodes = total_nodes = 0

    y_mask = loader.dataset.y_mask
    ious = [[] for _ in range(len(loader.dataset.categories))]

    for data in loader:
        data = data.to(device)
        pred = model(data).argmax(dim=1)
        correct_nodes += pred.eq(data.y).sum().item()
        total_nodes += data.num_nodes

        i, u = i_and_u(pred, data.y, loader.dataset.num_classes, data.batch)
        iou = i.cpu().to(torch.float) / u.cpu().to(torch.float)
        iou[torch.isnan(iou)] = 1

        # Find and filter the relevant classes for each category.
        for iou, category in zip(iou.unbind(), data.category.unbind()):
            ious[category.item()].append(iou[y_mask[category]])

    # Compute mean IoU.
    ious = [torch.stack(iou).mean(0).mean(0) for iou in ious]
    return torch.tensor(ious).mean().item(), correct_nodes/total_nodes


train_loss = []
train_accuracies = []
test_accuracies = []
test_ious = []

for epoch in range(1, args.epochs+1):
    print("epoch", epoch)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t_start = time.perf_counter()

    epoch_loss, train_acc = train(model)
    iou, test_acc = test(test_loader)

    train_loss.append(epoch_loss)
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
    test_ious.append(iou)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t_end = time.perf_counter()

    print('Epoch: {:02d}, Test IoU: {:.4f}, Duration: {:.2f}'.format(epoch, iou, t_end - t_start))


np.save('results/train_loss_ShapeNet.npy', np.array(train_loss))
np.save('results/train_accuracy_ShapeNet.npy', np.array(train_accuracies))
np.save('results/test_accuracy_ShapeNet.npy', np.array(test_accuracies))
np.save('results/test_iou_ShapeNet.npy', np.array(test_ious))