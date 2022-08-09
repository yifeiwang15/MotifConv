#pragma once
#include "../helper.h"
#include <stdint.h>
void launch_updateM(Tensor_lst_t &M_out, const Tensor_lst_t &M,
                    const Tensor_lst_t &M_E1, const Tensor_lst_t &M_E2,
                    const Tensor_lst_t &E1, const Tensor_lst_t &E2,
                    const Tensor_lst_t &S, const torch::Tensor &mask,
                    const float alpha, const float beta_0, const float beta_f,
                    const float beta_r, const int I_0);
void launch_updateMSelf(Tensor_lst_t &M_out, const Tensor_lst_t &M,
                        const Tensor_lst_t &M_e, const Tensor_lst_t &E,
                        const Tensor_lst_t &S, const torch::Tensor &mask,
                        const float alpha, const float beta_0,
                        const float beta_f, const float beta_r, const int I_0);
