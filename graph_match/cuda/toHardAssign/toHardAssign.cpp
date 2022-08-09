#include "toHardAssign.h"
#include <torch/extension.h>
void torch_launch_toHardAssign(Tensor_lst_t &M) { launch_toHardAssign(M); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_toHardAssign", &torch_launch_toHardAssign,
          "toHardAssign kernel warpper");
}