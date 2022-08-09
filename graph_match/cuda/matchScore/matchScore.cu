#include "../similarity.h"
#include "matchScore.h"
#include <initializer_list>
using F_accessor_t =
    torch::PackedTensorAccessor32<data_t, 2, torch::RestrictPtrTraits>;
using I_accessor_t =
    torch::PackedTensorAccessor32<index_t, 2, torch::RestrictPtrTraits>;

template <int num_thd, typename T>
__device__ T sum_all_thd(T data) { // only warp 0 has the result
    static_assert(num_thd <= 1024, "num_thd should be less than 1024");
    const auto warp_id = threadIdx.x >> 5;
    const auto lane_id = threadIdx.x % 32;
    constexpr auto num_warp = num_thd >> 5;
    __shared__ T smem[num_warp];
    const auto temp_warp = warp_sum(data);
    if (lane_id == 0)
        smem[warp_id] = temp_warp;
    __syncthreads();
    T temp = (T)0;
    if (warp_id == 0) {
        if (lane_id < num_warp)
            temp = smem[lane_id];
        __syncwarp();
    }
    return warp_sum(temp);
}

template <int num_thd>
__device__ data_t first_term(const F_accessor_t &M, const F_accessor_t &M_E1,
                             const F_accessor_t &M_E2, const I_accessor_t &E1,
                             const I_accessor_t &E2) {
    const auto N1 = M.size(0);
    const auto N2 = M.size(1);
    data_t temp = 0;
    for (int _i = threadIdx.x; _i < N1 * N2 * N1 * N2; _i += num_thd) {
        // _i = a*N2*N1*N2 + i*N1*N2 + b*N2 + j
        const auto a = _i / (N2 * N1 * N2);
        const auto i = (_i - (a * N2 * N1 * N2)) / (N1 * N2);
        const auto b = (_i - (a * N2 * N1 * N2) - (i * N1 * N2)) / N2;
        const auto j = _i % N2;
        temp += __ldg(&M[a][i]) * __ldg(&M[b][j]) *
                C(a, i, b, j, M_E1, M_E2, E1, E2);
    }
    const auto temp1 = sum_all_thd<num_thd>(temp);
    const auto e1 = E1.size(1);
    const auto e2 = E2.size(1);
    const auto norm = 1.0f / sqrtf(e1 * e2);
    return -0.5 * temp1 * norm;
}

template <int num_thd>
__device__ data_t second_term(const F_accessor_t &M, const F_accessor_t &S,
                              const data_t alpha) {
    const auto N1 = S.size(0);
    const auto N2 = S.size(1);
    data_t temp = 0;
    for (int _i = threadIdx.x; _i < N1 * N2; _i += blockDim.x) {
        const auto a = _i / N2;
        const auto i = _i % N2;
        temp += __ldg(&M[a][i]) * __ldg(&S[a][i]);
    }
    const auto temp1 = sum_all_thd<num_thd>(temp);
    const auto norm = 1.0f / sqrtf(N1 * N2);
    return -1.0f * alpha * temp1 * norm;
}

template <int num_thd>
__global__ void __launch_bounds__(1024)
    __kernel_matchScore(F_accessor_t out, const F_accessor_t *__restrict__ _M,
                        const F_accessor_t *__restrict__ _M_E1,
                        const F_accessor_t *__restrict__ _M_E2,
                        const I_accessor_t *__restrict__ _E1,
                        const I_accessor_t *__restrict__ _E2,
                        const F_accessor_t *__restrict__ _S,
                        const I_accessor_t mask, const data_t alpha,
                        const size_t num_G2) {
    const auto g1_id = blockIdx.x;
    const auto g2_id = blockIdx.y;
    if (mask[g1_id][g2_id] == 0)
        return;
    const auto &M = _M[g1_id * num_G2 + g2_id];
    const auto &M_e1 = _M_E1[g1_id];
    const auto &M_e2 = _M_E2[g2_id];
    const auto &E1 = _E1[g1_id];
    const auto &E2 = _E2[g2_id];
    const auto &S = _S[g1_id * num_G2 + g2_id];
    auto res = first_term<num_thd>(M, M_e1, M_e2, E1, E2);
    res += second_term<num_thd>(M, S, alpha);
    if (threadIdx.x == 0)
#if (CUDA_VERSION >= 11000)
        __stcs(&out[g1_id][g2_id], res);
#else
        out[g1_id][g2_id] = res;
#endif
}

template <int num_thd>
__global__ void __launch_bounds__(1024) __kernel_matchScoreSelf(
    torch::PackedTensorAccessor32<data_t, 1, torch::RestrictPtrTraits> out,
    const F_accessor_t *__restrict__ _M, const F_accessor_t *__restrict__ _M_E,
    const I_accessor_t *__restrict__ _E, const F_accessor_t *__restrict__ _S,
    const I_accessor_t mask, const data_t alpha, const size_t N) {
    for (int idx = blockIdx.x; idx < N * (N - 1) / 2; idx += gridDim.x) {
        const auto &M = _M[idx];
        int g1_id, g2_id;
        triu_k2ij(idx, N, g1_id, g2_id);
        if (mask[g1_id][g2_id] == 0)
            continue;
        const auto &M_e1 = _M_E[g1_id];
        const auto &M_e2 = _M_E[g2_id];
        const auto &E1 = _E[g1_id];
        const auto &E2 = _E[g2_id];
        const auto &S = _S[idx];
        auto res = first_term<num_thd>(M, M_e1, M_e2, E1, E2);
        res += second_term<num_thd>(M, S, alpha);
        if (threadIdx.x == 0)
#if (CUDA_VERSION >= 11000)
            __stcs(&out[idx], res);
#else
            out[idx] = res;
#endif
    }
}

void launch_matchScore(torch::Tensor &out, const Tensor_lst_t &M,
                       const Tensor_lst_t &M_E1, const Tensor_lst_t &M_E2,
                       const Tensor_lst_t &E1, const Tensor_lst_t &E2,
                       const Tensor_lst_t &S, const torch::Tensor &mask,
                       const float alpha) {
    constexpr int num_thd = 1024;
    auto F_dptr = to_accessor_lst<data_t, 2, torch::RestrictPtrTraits>(
        {M, M_E1, M_E2, S});
    auto I_dptr =
        to_accessor_lst<index_t, 2, torch::RestrictPtrTraits>({E1, E2});
    auto mask_accessor =
        mask.packed_accessor32<index_t, 2, torch::RestrictPtrTraits>();

    __kernel_matchScore<num_thd><<<dim3(E1.size(), E2.size()), num_thd>>>(
        out.packed_accessor32<data_t, 2, torch::RestrictPtrTraits>(), F_dptr[0],
        F_dptr[1], F_dptr[2], I_dptr[0], I_dptr[1], F_dptr[3], mask_accessor,
        alpha, E2.size());

    gpuErrchk(cudaFree(F_dptr[0]));
    gpuErrchk(cudaFree(I_dptr[0]));
}

void launch_matchScoreSelf(torch::Tensor &out, const Tensor_lst_t &M,
                           const Tensor_lst_t &M_e, const Tensor_lst_t &E,
                           const Tensor_lst_t &S, const torch::Tensor &mask,
                           const data_t alpha) {
    constexpr int num_thd = 1024;
    auto F_dptr =
        to_accessor_lst<data_t, 2, torch::RestrictPtrTraits>({M, M_e, S});
    auto I_dptr = to_accessor_lst<index_t, 2, torch::RestrictPtrTraits>({E});
    int nthd, nblk;
    cudaOccupancyMaxPotentialBlockSize(&nblk, &nthd,
                                       __kernel_matchScoreSelf<num_thd>);
    auto out_accessor =
        out.packed_accessor32<data_t, 1, torch::RestrictPtrTraits>();
    auto mask_accessor =
        mask.packed_accessor32<index_t, 2, torch::RestrictPtrTraits>();

    __kernel_matchScoreSelf<num_thd>
        <<<nblk, num_thd>>>(out_accessor, F_dptr[0], F_dptr[1], I_dptr[0],
                            F_dptr[2], mask_accessor, alpha, E.size());
    gpuErrchk(cudaFree(F_dptr[0]));
    gpuErrchk(cudaFree(I_dptr[0]));
}
