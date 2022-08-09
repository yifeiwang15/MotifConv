"""
This part defines gcn model to get accuracy 
"""

import dgl
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GraphConv
from dataset import train_test_split


# define GCN model, two layers
class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        g.ndata['x'] = h
        return dgl.mean_nodes(g, 'x')


def train():
    # get train and test dataloader
    train_dataloader, test_dataloader = train_test_split()
    # define model parameter
    model = GCN(1, 3, 5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1000):
        for batched_graph, labels in train_dataloader:
            pred = model(batched_graph, batched_graph.ndata['x'].float())
            loss = F.cross_entropy(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model, test_dataloader


def test():
    model, test_dataloader = train()
    num_correct = 0
    num_tests = 0

    correct_count = [0, 0, 0, 0, 0]
    count_total = [0, 0, 0, 0, 0]

    for batched_graph, labels in test_dataloader:
        pred = model(batched_graph, batched_graph.ndata['x'].float())
        num_correct += (pred.argmax(1) == labels).sum().item()
        num_tests += len(labels)

        # count each label number
        for i in labels:
            count_total[i.item()] += 1

        for i in (pred.argmax(1) == labels).nonzero().view(-1):
            correct_count[labels[i].item()] += 1

    accuracy = num_correct / num_tests
    return accuracy, correct_count, count_total


accuracy_list = []
corr_array, total_array = np.array([0, 0, 0, 0, 0]), np.array([0, 0, 0, 0, 0])
for i in range(5):

    a, b, c = test()
    accuracy_list.append(a)
    b, c = np.array(b), np.array(c)
    corr_array += b
    total_array += c

print(accuracy_list)
print(corr_array)
print(total_array)


# accuracy: 0.198
# 0: 0.44, 1: 0.15, 2: 0.11, 3:0.16, 4: 0.11

# [0.22, 0.25, 0.22, 0.17, 0.32]
# [27 13 16 60  2]
# [ 96  88  98 112 106]

# [0.2, 0.23, 0.16, 0.14, 0.19]
# [78 14  0  0  0]
# [100  95  98 100 107]

# [0.16, 0.38, 0.2, 0.22, 0.19]
# [27 17 20 38 13]
# [110 105 101  93  91]

# [0.21, 0.26, 0.23, 0.26, 0.41]
# [28 27 21 55  6]
# [ 92 100 116  90 102]


# [0.31, 0.41, 0.36, 0.29, 0.23]
# [25 21 27 78  9]
# [ 89 104  96 102 109]

# [0.28, 0.17, 0.16, 0.17, 0.3]
# [ 4 13 47 39  5]
# [107 107  91  95 100]