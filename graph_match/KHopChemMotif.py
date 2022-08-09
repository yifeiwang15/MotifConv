import numpy as np
import networkx as nx
import sys
import torch
from typing import Optional

class Gdata(object):
    def __init__(self, g_matrices):
        self.x = g_matrices[0]
        self.edge_attr = g_matrices[1]
        self.edge_index = g_matrices[2]
        self.map_center = None if len(g_matrices) == 3 else g_matrices[3]
        
def maybe_num_nodes(index: torch.Tensor,
                    num_nodes: Optional[int] = None) -> int:
    return int(index.max()) + 1 if num_nodes is None else num_nodes

def convert_g_to_matrices(data):
    M_n = data['x'][:,[0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11]]
    E  = data['edge_index']
    dists = torch.norm(data['pos'][E[0, :]] - data['pos'][E[1, :]], 2, 1).unsqueeze(1) # l2 norm
    M_e = torch.cat([data['edge_attr'], dists], dim=1)
    return M_n, M_e, E

def k_hop_subgraph(node_idx, num_hops, edge_index, relabel_nodes=False,
                   num_nodes=None, flow='source_to_target'):
    r"""Modified from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/subgraph.html#k_hop_subgraph
        
        Computes the :math:`k`-hop subgraph of :obj:`edge_index` around node
        :attr:`node_idx`.
        It returns (1) the nodes involved in the subgraph, (2) the filtered
        :obj:`edge_index` connectivity, (3) the mapping from node indices in
        :obj:`node_idx` to their new location (4) the edge mask indicating
        which edges were preserved, and (5) the mapped index of the central node.

        Args:
            node_idx (int, list, tuple or :obj:`torch.Tensor`): The central
                node(s).
            num_hops: (int): The number of hops :math:`k`.
            edge_index (LongTensor): The edge indices.
            relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
                :obj:`edge_index` will be relabeled to hold consecutive indices
                starting from zero. (default: :obj:`False`)
            num_nodes (int, optional): The number of nodes, *i.e.*
                :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
            flow (string, optional): The flow direction of :math:`k`-hop
                aggregation (:obj:`"source_to_target"` or
                :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)

        :rtype: (:class:`LongTensor`, :class:`LongTensor`, :class:`LongTensor`,
                 :class:`BoolTensor`)
        """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    assert num_hops >= 1
    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor([node_idx], device=row.device).flatten()
    else:
        node_idx = node_idx.to(row.device)

    subsets = [node_idx]

    for _ in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        subsets.append(col[edge_mask])

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    center = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes, ), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]

    return subset, edge_index, center, edge_mask, inv


if __name__ == '__main__':
    # extract k-hop motifs
    filename = "qm9.pt" # sys.argv[1]
    k_hop = 1 if len(sys.argv) < 3 else int(sys.argv[2])
    dataset = torch.load(filename)
    motifs = []
    for i, raw_g in enumerate(dataset):
        g = Gdata(convert_g_to_matrices(raw_g))
        for i in range(g.x.shape[0]):
            subset, sub_edge_index, map_center, mask, _ = k_hop_subgraph(i, k_hop, g.edge_index, relabel_nodes=True)
            sub_edge_attr = g.edge_attr[mask]
            sub_x = g.x[subset]
            if subset.shape[0] > 2:
                motifs.append((sub_x, sub_edge_attr, sub_edge_index))
        print(len(motifs))
    torch.save(motifs, 'motifs.pt')
