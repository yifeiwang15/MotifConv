"""
This part is used to preprocess the dataset and lebel each datapoint
"""

import torch
import dgl
import numpy as np
import networkx as nx
from dgl.data import DGLDataset
import matplotlib.pyplot as plt
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler


def train_test_split():
    # get syntheticDataset()
    dataset = SyntheticDataset()
    num_examples = len(dataset)
    # train : test = 4 : 1
    num_train = int(num_examples * 0.8)
    dataset_list = list(range(num_examples))
    np.random.shuffle(dataset_list)
    train_sampler = SubsetRandomSampler(dataset_list[:num_train])
    test_sampler = SubsetRandomSampler(dataset_list[num_train:])

    train_dataloader = GraphDataLoader(dataset, sampler=train_sampler, batch_size=5, drop_last=False)
    test_dataloader = GraphDataLoader(dataset, sampler=test_sampler, batch_size=5, drop_last=False)

    return train_dataloader, test_dataloader


class SyntheticDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='synthetic')

    def process(self):
        syn_data = torch.load('syn-data.pt')
        self.graphs, self.labels = [], []

        for i in range(len(syn_data)):
            u, d = syn_data[i][2]
            g = dgl.graph((u, d))
            g.ndata['x'] = syn_data[i][0]
            g.edata['x'] = syn_data[i][1]

            self.graphs.append(g)
            self.labels.append(i // 100)

        # covert label list to tensor for saving
        self.labels = torch.LongTensor(self.labels)

    def __getitem__(self, item):
        return self.graphs[item], self.labels[item]

    def __len__(self):
        return len(self.graphs)

# dataset = SyntheticDataset()
# graph, label = dataset[98]
# print(graph.ndata['x'])
# print(label)
# nx_G = graph.to_networkx().to_undirected()
# pos = nx.kamada_kawai_layout(nx_G)
# nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])
# plt.show()