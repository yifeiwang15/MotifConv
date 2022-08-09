import dgl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

from dataset import train_test_split
from gcn import GCN

train_dataloader, test_dataloader = train_test_split()


def train():
    model = GCN(1, 2, 5, 2, F.relu, 0.2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    running_loss = 0
    for epoch in range(1000):
        for batched_graph, labels in train_dataloader:
            predict = model(batched_graph, batched_graph.ndata['x'])

            loss = criterion(predict, labels)
            running_loss += loss.item()

            # back
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def test():
    model = train()
    num_correct = 0
    num_tests = 0

    correct_count = [0, 0, 0, 0, 0]
    count_total = [0, 0, 0, 0, 0]

    for batched_graph, labels in test_dataloader:
        pred = model(batched_graph, batched_graph.ndata['x'])
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


test()