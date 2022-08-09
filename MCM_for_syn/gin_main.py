import sys
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from gin_model import GIN
from dataset import train_test_split

train_dataloader, test_dataloader = train_test_split()


def train():
    model = GIN(1, 1, 1, 3, 5, 0.5, 0.01, 'mean', 'mean')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # total_iter = len(train_dataloader)
    for epoch in range(200):
        running_loss = 0
        for batched_graph, labels in train_dataloader:
            predict = model(batched_graph, batched_graph.ndata['x'])

            loss = criterion(predict, labels)
            running_loss += loss.item()
            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # print(running_loss / total_iter)
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

# hidden dimension: 3
# [0.27, 0.23, 0.22, 0.19, 0.2]
# [17  7 19 57 11]
# [ 85 150  60 115  90]

# [0.17, 0.25, 0.21, 0.27, 0.22]
# [13 34  1 43 21]
# [120  85  95 105  95]

# [0.26, 0.22, 0.2, 0.22, 0.19]
# [16 24  4 55 10]
# [ 70  75 140 115 100]

# [0.22, 0.18, 0.27, 0.18, 0.23]
# [ 4 37 26 32  9]
# [125 100 110  65 100]

# [0.27, 0.2, 0.21, 0.16, 0.16]
# [17 24  9 31 19]
# [110  95 110  65 120]

# [0.25, 0.24, 0.24, 0.26, 0.22]
# [ 2 46 11 48 14]
# [110  85 120 105  80]

