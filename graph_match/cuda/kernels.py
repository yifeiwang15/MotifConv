import torch
from torch.utils.cpp_extension import load
import os
cuda_nodeSim = load(name="nodeSim", extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "--use_fast_math"],
                    sources=["cuda/nodeSim/nodeSim.cpp",
                             "cuda/nodeSim/nodeSim.cu"], build_directory = "./cuda/nodeSim",
                    verbose=False)

cuda_updateM = load(name="updateM", extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "--use_fast_math"],
                    sources=["cuda/updateM/updateM.cpp",
                             "cuda/updateM/updateM.cu"],build_directory = "./cuda/updateM",
                    verbose=False)

cuda_matchScore = load(name="matchScore", extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "--use_fast_math"],
                       sources=["cuda/matchScore/matchScore.cpp",
                                "cuda/matchScore/matchScore.cu"],build_directory = "./cuda/matchScore",
                       verbose=False)
cuda_toHardAssign = load(name="toHardAssign", extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "--use_fast_math"],
                    sources=["cuda/toHardAssign/toHardAssign.cpp",
                             "cuda/toHardAssign/toHardAssign.cu"], build_directory = "./cuda/toHardAssign",
                    verbose=False)


from math import sqrt


def triu_k2ij(n, k) -> int:
    """
    n: the size of array. One side of the triangle.
    k: the linear index of the element in the upper triangle.
    """
    i = n - 2 - int(sqrt(-8 * k + 4 * n * (n - 1) - 7) / 2.0 - 0.5)
    j = k + i + 1 - n * (n - 1) / 2 + (n - i) * ((n - i) - 1) / 2
    return int(i), int(j)


def triu_ij2k(n, i, j) -> int:
    """
    n: the size of array. One side of the triangle.
    i,j: the coordinates of the element in the upper triangle.
    """
    k = (n * (n - 1) / 2) - (n - i) * ((n - i) - 1) / 2 + j - i - 1
    return int(k)


def nodeSim(*args):
    out = []
    if len(args) == 3:
        M_N1, M_N2, mask = args
        for i in range(len(M_N1)):
            for j in range(len(M_N2)):
                out.append(torch.full(
                    (M_N1[i].size(0) + 1, M_N2[j].size(0) + 1), 1e-3, device="cuda"))
        # leave the space for slack node, node 0 is the slack node
        views = [t[1:, 1:] for t in out]
        cuda_nodeSim.torch_launch_nodeSim(views, M_N1, M_N2, mask)
    elif len(args) == 2:
        M_N, mask = args
        npairs = len(M_N) * (len(M_N) - 1) // 2
        for k in range(npairs):
            i, j = triu_k2ij(len(M_N), k)
            out.append(torch.full(
                (M_N[i].size(0) + 1, M_N[j].size(0) + 1), 1e-3, device="cuda"))
        views = [t[1:, 1:] for t in out]
        cuda_nodeSim.torch_launch_nodeSimSelf(views, M_N, mask)
    else:
        raise ValueError("Wrong number of arguments")
    return out


def updateM(M, M_E, E, S, mask, alpha, beta_0, beta_f, beta_r, I_0):
    out = [torch.empty_like(t) for t in M]
    if len(M_E) == 2 and len(E) == 2:
        M_E1, M_E2 = M_E
        E1, E2 = E
        cuda_updateM.torch_launch_updateM(
            out, M, M_E1, M_E2, E1, E2, S, mask, alpha, beta_0, beta_f, beta_r, I_0)
    elif len(M_E) == 1 and len(E) == 1:
        M_E_ = M_E[0]
        E_ = E[0]
        cuda_updateM.torch_launch_updateMSelf(
            out, M, M_E_, E_, S, mask, alpha, beta_0, beta_f, beta_r, I_0)
    else:
        raise ValueError("Wrong number of arguments")
    return out


def matchScore(M, M_E, E, S, mask, alpha):
    if len(M_E) == 2 and len(E) == 2:
        M_E1, M_E2 = M_E
        E1, E2 = E
        out = torch.empty((len(E1), len(E2)), device="cuda")
        cuda_matchScore.torch_launch_matchScore(
            out, M, M_E1, M_E2, E1, E2, S, mask, alpha)
    elif len(M_E) == 1 and len(E) == 1:
        M_E_ = M_E[0]
        E_ = E[0]
        out = torch.empty((len(M), ), device="cuda")
        cuda_matchScore.torch_launch_matchScoreSelf(
            out, M, M_E_, E_, S, mask, alpha)
    else:
        raise ValueError("Wrong number of arguments")
    return out

def toHardAssign(M):
    assert isinstance(M, list)
    return cuda_toHardAssign.torch_launch_toHardAssign(M)
