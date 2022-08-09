import torch


def convert_g_to_matrices(data):
    M_n = data['x'][:, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11]]
    E = data['edge_index']
    dists = torch.norm(data['pos'][E[0, :]] - data['pos']
                       [E[1, :]], 2, 1).unsqueeze(1)  # l2 norm
    M_e = torch.cat([data['edge_attr'], dists], dim=1)
    # E = E.to(torch.int32)
    return M_n, M_e, E


def load(path, preprocessed=False):
    data_list = torch.load(path)
    if preprocessed:
        # The case where graphs are saved in matrices format.
        return [(i, data) for i, data in enumerate(data_list)]
    else:
        return [(i, convert_g_to_matrices(data)) for i, data in enumerate(data_list)]

def load_raw(path):
    data_list = torch.load(path)
    return [(i, data) for i, data in enumerate(data_list)]
