#include "nodeSim.h"
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <torch/extension.h>

void torch_launch_nodeSim(Tensor_lst_t &S, const Tensor_lst_t &G1,
                          const Tensor_lst_t &G2, const torch::Tensor &mask) {
    launch_nodeSim(S, G1, G2, mask);
}

void torch_launch_nodeSimSelf(Tensor_lst_t &S, const Tensor_lst_t &M_n,
                              const torch::Tensor &mask) {
    launch_nodeSimSelf(S, M_n, mask);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_nodeSim", &torch_launch_nodeSim,
          "nodeSim kernel warpper");

    m.def("torch_launch_nodeSimSelf", &torch_launch_nodeSimSelf,
          "nodeSimSelf kernel warpper");
}