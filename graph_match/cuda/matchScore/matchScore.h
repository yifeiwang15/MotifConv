#pragma once
#include "../helper.h"
#include <stdint.h>

void launch_matchScore(torch::Tensor &out, const Tensor_lst_t &M,
                       const Tensor_lst_t &M_E1, const Tensor_lst_t &M_E2,
                       const Tensor_lst_t &E1, const Tensor_lst_t &E2,
                       const Tensor_lst_t &S, const torch::Tensor &mask,
                       const float alpha);

void launch_matchScoreSelf(torch::Tensor &out, const Tensor_lst_t &M,
                           const Tensor_lst_t &M_e, const Tensor_lst_t &E,
                           const Tensor_lst_t &S, const torch::Tensor &mask,
                           const data_t alpha);