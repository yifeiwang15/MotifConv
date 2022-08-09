import torch


class Graph:
    __slots__ = ['M_n', 'M_e', 'E', 'nNode', 'nEdge', 'idx']
    M_n: torch.Tensor
    M_e: torch.Tensor
    E: torch.Tensor

    def __init__(self, M_n, M_e, E):
        self.nNode = M_n.shape[0]
        self.nEdge = M_e.shape[0]
        # self.nodes = M_n.contiguous().cuda()
        assert self.nEdge == E.shape[1]

        self.M_n = M_n.contiguous().cuda()
        self.M_e = M_e.contiguous().cuda()
        self.E = E.contiguous().cuda()

    def set_idx(self, idx):
        self.idx = idx

        # temp = [(E[0][e], E[1][e], M_e[e]) for e in range(self.nEdge)]
        # temp = sort(temp, key=lambda x: x[0])

        # self.idxptr = torch.empty(self.nNode + 1, dtype=torch.int32)
        # self.idxptr[0] = 0
        # for i in range(self.nNode):
        #     self.idxptr[i + 1] = sum(1 for j in temp if temp[j] == i)
        # assert self.idxptr[-1] == self.nEdge
        # self.idxptr = self.idxptr.cuda()
