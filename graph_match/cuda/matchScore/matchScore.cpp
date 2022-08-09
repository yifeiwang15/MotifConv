#include "matchScore.h"
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <torch/extension.h>

void torch_launch_matchScore(torch::Tensor &out, const Tensor_lst_t &M,
                             const Tensor_lst_t &M_E1, const Tensor_lst_t &M_E2,
                             const Tensor_lst_t &E1, const Tensor_lst_t &E2,
                             const Tensor_lst_t &S, const torch::Tensor &mask,
                             const float alpha) {
    launch_matchScore(out, M, M_E1, M_E2, E1, E2, S, mask, alpha);
}

void torch_launch_matchScoreSelf(torch::Tensor &out, const Tensor_lst_t &M,
                                 const Tensor_lst_t &M_e, const Tensor_lst_t &E,
                                 const Tensor_lst_t &S,
                                 const torch::Tensor &mask,
                                 const data_t alpha) {
    launch_matchScoreSelf(out, M, M_e, E, S, mask, alpha);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_matchScore", &torch_launch_matchScore,
          "matchScore kernel warpper");

    m.def("torch_launch_matchScoreSelf", &torch_launch_matchScoreSelf,
          "matchScoreSelf kernel warpper");
}