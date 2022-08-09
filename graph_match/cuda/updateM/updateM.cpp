#include "updateM.h"
#include <torch/extension.h>
void torch_launch_updateM(Tensor_lst_t &M_out, const Tensor_lst_t &M,
                          const Tensor_lst_t &M_E1, const Tensor_lst_t &M_E2,
                          const Tensor_lst_t &E1, const Tensor_lst_t &E2,
                          const Tensor_lst_t &S, const torch::Tensor &mask,
                          const float alpha, const float beta_0,
                          const float beta_f, const float beta_r,
                          const int I_0) {

    launch_updateM(M_out, M, M_E1, M_E2, E1, E2, S, mask, alpha, beta_0, beta_f,
                   beta_r, I_0);
}

void torch_launch_updateMSelf(Tensor_lst_t &M_out, const Tensor_lst_t &M,
                              const Tensor_lst_t &M_e, const Tensor_lst_t &E,
                              const Tensor_lst_t &S, const torch::Tensor &mask,
                              const float alpha, const float beta_0,
                              const float beta_f, const float beta_r,
                              const int I_0) {

    launch_updateMSelf(M_out, M, M_e, E, S, mask, alpha, beta_0, beta_f, beta_r, I_0);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_updateM", &torch_launch_updateM,
          "updateM kernel warpper");
    m.def("torch_launch_updateMSelf", &torch_launch_updateMSelf,
          "updateMSelf kernel warpper");
}