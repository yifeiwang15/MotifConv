#pragma once
#include "../helper.h"
#include <stdint.h>

void launch_nodeSim(Tensor_lst_t &S, const Tensor_lst_t &G1,
                    const Tensor_lst_t &G2, const torch::Tensor &mask);

void launch_nodeSimSelf(Tensor_lst_t &S, const Tensor_lst_t &M_n, const torch::Tensor &mask);