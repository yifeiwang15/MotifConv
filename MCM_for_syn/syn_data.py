import torch
import numpy as np
from torch.utils.data import DataLoader
import networkx as nx
import random
import copy
import networkx as nx
import numpy as np
import torch
from torch.utils.data import DataLoader



class SynGraphBuilder:
    """Construct a graph as a combination of one basis graph
    and several candidate graphs.
    """

    def __init__(self, basis_graph: nx.Graph, noise_rate=0.1, add_nodes_rate=0.3):
        """
        Nodes in graphs should have consecutive IDs starting from 0. (We've included a preprocesssing step.)

        Parameters
        ----------
        basisGraph     : the basis.
        CandidateGraphs: list of candidate graphs.
        """

        self.basis_graph = basis_graph
        self.noise_rate = noise_rate
        self.add_nodes_rate = add_nodes_rate

    def build_graph(self):
        n_basis = self.basis_graph.number_of_nodes()
        g = copy.deepcopy(self.basis_graph)

        # Randomly add nodes
        for i in range(self.basis_graph.number_of_nodes()):
            if random.random() <= self.add_nodes_rate:
                g.add_nodes_from([(n_basis, {'x': 0.5+random.random()})])
                g.add_edges_from([(i, n_basis, {'dist': 1.0})])
                n_basis += 1

        # add random noise to node and edge attrs
        for i in range(g.number_of_nodes()):
            g.nodes[i]['x'] = g.nodes[i]['x'] + np.random.normal(0, self.noise_rate)

        # for e in g.edges():
        #     g.edges[e]['dist'] = max(g.edges[e]['dist'] + np.random.normal(0, self.noise_rate), 0)

        return self.convert_to_matrices(g)

    def convert_to_matrices(self, g: nx.Graph):
        g = g.to_directed()  # add reverse edges
        # node matrix
        # node attr: atom one hot encoding
        atoms = nx.get_node_attributes(g, 'x').values()
        num = len(atoms)
        M_n = torch.zeros([num, 1])
        for i in range(num):
            M_n[i, :] = torch.tensor(g.nodes[i]['x'])

        # edge matrix
        # edge attr: dist only
        dists = nx.get_edge_attributes(g, 'dist').values()
        M_e = torch.tensor(list(dists)).unsqueeze(-1)
        dic_node_tmp = dict(zip(g.nodes, range(len(g.nodes))))
        E = list(map(lambda s: [dic_node_tmp[s[0]], dic_node_tmp[s[1]]], g.edges))
        E = torch.transpose(torch.tensor(E, dtype=torch.long), 0, 1)
        return M_n, M_e, E


class SynBinaryDataset(torch.utils.data.Dataset):

    def __init__(self, template_1, template_2, num=100, noise_rate=0.1, add_nodes_rate=0.3):
        super(SynBinaryDataset, self).__init__()

        self.builder_1 = SynGraphBuilder(template_1, noise_rate=noise_rate, add_nodes_rate=add_nodes_rate)
        self.builder_2 = SynGraphBuilder(template_2, noise_rate=noise_rate, add_nodes_rate=add_nodes_rate)
        self.num = num
        self.graphs = []
        labels = [0] * self.num + [1] * self.num
        self.labels = torch.LongTensor(labels)
        self.process()

    def process(self):
        """Sampling graphs from each builder."""

        for i in range(self.num):
            new_g = self.builder_1.build_graph()
            self.graphs.append(new_g)

        for i in range(self.num):
            new_g = self.builder_2.build_graph()
            self.graphs.append(new_g)

    @staticmethod
    def collate_fn(x):
        """collate_fn for DataLoader"""
        return [x[i][0] for i in range(len(x))], torch.cat([x[i][1].unsqueeze(0) for i in range(len(x))])

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return 2 * self.num


class SynMultiDataset(torch.utils.data.Dataset):

    def __init__(self, template_list, num=100, noise_rate=0.1, add_nodes_rate=0.3):
        super(SynMultiDataset, self).__init__()

        self.num = num
        self.builder = {}
        labels = []
        for i in range(len(template_list)):
            self.builder[i] = SynGraphBuilder(template_list[i], noise_rate=noise_rate, add_nodes_rate=add_nodes_rate)
            labels = labels + [i] * self.num

        self.graphs = []
        self.labels = torch.LongTensor(labels)
        self.process()

    def process(self):
        """Sampling graphs from each builder."""
        for i in range(len(self.builder)):
            for n in range(self.num):
                new_g = self.builder[i].build_graph()
                self.graphs.append(new_g)


    @staticmethod
    def collate_fn(x):
        """collate_fn for DataLoader"""
        return [x[i][0] for i in range(len(x))], torch.cat([x[i][1].unsqueeze(0) for i in range(len(x))])

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.builder) * self.num


if __name__ == '__main__':
    NUM = 20  # number of samples from each template
    # generate 5 templates
    g1 = nx.Graph()
    g1.add_nodes_from([(0, {'x': 1.5}),
                       (1, {'x': 0.5}),
                       (2, {'x': 0.2}),
                       (3, {'x': 1.3}),
                       ])
    g1.add_edges_from([(0, 1, {'dist': 1.0}),
                       (0, 2, {'dist': 1.41}),
                       (0, 3, {'dist': 1.0}),
                       (1, 2, {'dist': 1.0}),
                       (1, 3, {'dist': 1.41}),
                       (2, 3, {'dist': 1.0})
                       ])

    g2 = nx.Graph()
    g2.add_nodes_from([(0, {'x': 1.5}),
                       (1, {'x': 0.5}),
                       (2, {'x': 0.2}),
                       (3, {'x': 1.3}),
                       ])
    g2.add_edges_from([(0, 1, {'dist': 1.0}),
                       (0, 2, {'dist': 1.0}),
                       (0, 3, {'dist': 1.0}),
                       (1, 2, {'dist': 1.0}),
                       ])

    g3 = nx.Graph()
    g3.add_nodes_from([(0, {'x': 0.5}),
                       (1, {'x': 1.5}),
                       (2, {'x': 0.2}),
                       (3, {'x': 1.3}),
                       ])
    g3.add_edges_from([(0, 1, {'dist': 1.0}),
                       (0, 2, {'dist': 1.0}),
                       (0, 3, {'dist': 1.0}),
                       (1, 3, {'dist': 1.0}),
                       (2, 3, {'dist': 1.0})
                       ])

    g4 = nx.Graph()
    g4.add_nodes_from([(0, {'x': 1.0}),
                       (1, {'x': 0.5}),
                       (2, {'x': 0.5}),
                       (3, {'x': 1.3}),
                       ])
    g4.add_edges_from([(0, 1, {'dist': 1.0}),
                       (0, 2, {'dist': 1.41}),
                       (0, 3, {'dist': 1.0}),
                       (1, 2, {'dist': 1.0}),
                       (1, 3, {'dist': 1.0}),
                       (2, 3, {'dist': 1.0})
                       ])

    g5 = nx.Graph()
    g5.add_nodes_from([(0, {'x': 1.5}),
                       (1, {'x': 0.5}),
                       (2, {'x': 0.2}),
                       (3, {'x': 1.3}),
                       ])
    g5.add_edges_from([(0, 1, {'dist': 1.0}),
                       (0, 2, {'dist': 1.41}),
                       (0, 3, {'dist': 1.0}),
                       (1, 2, {'dist': 1.41}),
                       ])

    g_dic = {1: g1, 2: g2, 3: g3, 4: g4, 5: g5}
    g_list = []
    for idx in range(5):
        builder = SynGraphBuilder(g_dic[idx + 1], noise_rate=0.1, add_nodes_rate=0.1)
        for _ in range(NUM):
            g = builder.build_graph()
            g_list.append(g)
    print(len(g_list))
