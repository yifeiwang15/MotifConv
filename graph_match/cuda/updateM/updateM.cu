#include "../similarity.h"
#include "updateM.h"
#include <initializer_list>
using F_accessor_t =
    torch::PackedTensorAccessor32<data_t, 2, torch::RestrictPtrTraits>;
using I_accessor_t =
    torch::PackedTensorAccessor32<index_t, 2, torch::RestrictPtrTraits>;

template <typename accessor_t>
__device__ void normalize_row(accessor_t &M, const int nrow, const int ncol) {
    // call by a block
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x % 32;
    const int num_warp = blockDim.x >> 5;
    // each warp process a row
    for (int r = warp_id; r < nrow; r += num_warp) {
        data_t val = 0;
        for (int c = lane_id; c < ncol; c += 32) {
            auto temp = M[r][c];
            val += temp * temp;        
        }
        auto sum = warp_sum(val);
        const auto one_over_sum = 1.0f / sqrtf(sum);
        for (int c = lane_id; c < ncol; c += 32) {
            M[r][c] *= one_over_sum;
        }
    }
}

template <typename accessor_t>
__device__ void normalize_col(accessor_t &M, const int nrow, const int ncol) {
    // call by a block
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x % 32;
    const int num_warp = blockDim.x >> 5;
    // each warp process a column
    for (int c = warp_id; c < ncol; c += num_warp) {
        data_t val = 0;
        for (int r = lane_id; r < nrow; r += 32) {
            auto temp = M[r][c];
            val += temp * temp;
        }
        auto sum = warp_sum(val);
        const auto one_over_sum = 1.0f / sqrtf(sum);
        for (int r = lane_id; r < nrow; r += 32) {
            M[r][c] *= one_over_sum;
        }
    }
}

// The similarity matrix M has N1 rows and N2 columns.
// M has num_G1 * num_G2 matrices in total
// M[g1 * num_G2 + g2] represents the similarity between g1 and g2
template <int num_thd>
__global__ void __launch_bounds__(1024)
    updateM_kernel(F_accessor_t *__restrict__ _M_out,
                   const F_accessor_t *__restrict__ _M,
                   const F_accessor_t *__restrict__ _M_E1,
                   const F_accessor_t *__restrict__ _M_E2,
                   const I_accessor_t *__restrict__ _E1,
                   const I_accessor_t *__restrict__ _E2,
                   const F_accessor_t *__restrict__ _S, const I_accessor_t mask,
                   const data_t alpha, const data_t beta, const size_t num_G1,
                   const size_t num_G2) {
    const auto warp_id = threadIdx.x >> 5;
    const auto lane_id = threadIdx.x % 32;
    constexpr auto num_warp = num_thd >> 5;
    for (int idx = blockIdx.x; idx < num_G1 * num_G2; idx += gridDim.x) {
        const auto g1_id = idx / num_G2;
        const auto g2_id = idx % num_G2;
        if (mask[g1_id][g2_id] == 0)
            continue;
        auto dst_M = _M_out[idx];
        const auto src_M = _M[idx];
        const auto &M_e1 = _M_E1[g1_id];
        const auto &M_e2 = _M_E2[g2_id];
        const auto &E1 = _E1[g1_id];
        const auto &E2 = _E2[g2_id];
        const auto &S = _S[idx];
        const auto N1 = S.size(0);
        const auto N2 = S.size(1);
        const data_t scale_C = 0.5f / (2 * min(N1, N2));

        // a warp update a element of M
        for (int _i = warp_id; _i < N1 * N2; _i += num_warp) {
            const int a = _i / N2; // node in graph 1
            const int i = _i % N2; // node in graph 2
            // if node a or node i are slack nodes, then skip
            // because their related edges are not in the graph
            if (a == 0 || i == 0) {
                if (lane_id == 0)
                    dst_M[a][i] = __expf(beta * alpha * S[a][i]);
                continue;
            }
            data_t val = 0;
            for (int _j = lane_id; _j < N1 * N2; _j += 32) {
                const int b = _j / N2; // node in graph 1
                const int j = _j % N2; // node in graph 2
                // skip if node b or node j are slack node
                if (b != 0 && j != 0)
                    val += C(a - 1, i - 1, b - 1, j - 1, M_e1, M_e2, E1, E2) *
                           __ldg(&src_M[b][j]);
            }
            val *= scale_C; // prevent overflow
            __syncwarp();
            const data_t sum = warp_sum(val);
            if (lane_id == 0) {
                dst_M[a][i] = expf((alpha * __ldg(&S[a][i]) + sum) * beta);
            }
        }
        // normalize M by row
        __syncthreads();
        normalize_row(dst_M, N1, N2);
        // normalize M by col
        __syncthreads();
        normalize_col(dst_M, N1, N2);
    }
}

// The similarity matrix M has N1 rows and N2 columns.
// M has num_G1 * num_G2 matrices in total
// M[g1 * num_G2 + g2] represents the similarity between g1 and g2
__global__ void __launch_bounds__(1024) updateMSelf_kernel(
    F_accessor_t *__restrict__ _M_out, const F_accessor_t *__restrict__ _M,
    const F_accessor_t *__restrict__ _M_e, const I_accessor_t *__restrict__ _E,
    const F_accessor_t *__restrict__ _S, const I_accessor_t mask,
    const data_t alpha, const data_t beta, const size_t N) {
    const auto warp_id = threadIdx.x >> 5;
    const auto lane_id = threadIdx.x % 32;
    const auto num_warp = blockDim.x >> 5;
    for (int M_idx = blockIdx.x; M_idx < N * (N - 1) / 2; M_idx += gridDim.x) {
        auto dst_M = _M_out[M_idx];
        const auto src_M = _M[M_idx];
        int g1_id, g2_id;
        triu_k2ij(M_idx, N, g1_id, g2_id);
        if (mask[g1_id][g2_id] == 0)
            continue;
        const auto &M_e1 = _M_e[g1_id];
        const auto &M_e2 = _M_e[g2_id];
        const auto &E1 = _E[g1_id];
        const auto &E2 = _E[g2_id];
        const auto &S = _S[M_idx];
        const auto N1 = S.size(0);
        const auto N2 = S.size(1);
        const data_t scale_C = 0.5f / (2 * min(N1, N2));
        // a warp update a element of M
        for (int _i = warp_id; _i < N1 * N2; _i += num_warp) {
            const int a = _i / N2; // node in graph 1
            const int i = _i % N2; // node in graph 2
            // if node a or node i are slack nodes, then skip
            // because their related edges are not in the graph
            if (a == 0 || i == 0) {
                if (lane_id == 0)
                    dst_M[a][i] = __expf(beta * alpha * S[a][i]);
                continue;
            }
            data_t val = 0;
            for (int _j = lane_id; _j < N1 * N2; _j += 32) {
                const int b = _j / N2; // node in graph 1
                const int j = _j % N2; // node in graph 2
                // skip if node b or node j are slack node
                if (b == 0 || j == 0)
                    continue;
                val += C(a - 1, i - 1, b - 1, j - 1, M_e1, M_e2, E1, E2) *
                       __ldg(&src_M[b][j]);
            }
            val *= scale_C; // prevent overflow
            __syncwarp();
            const data_t sum = warp_sum(val);
            if (lane_id == 0) {
                dst_M[a][i] = expf((alpha * __ldg(&S[a][i]) + sum) * beta);
            }
            __syncwarp();
        }
        __syncthreads();
        // normalize M by row
        normalize_row(dst_M, N1, N2);
        __syncthreads();
        // normalize M by col
        normalize_col(dst_M, N1, N2);
    }
}

void launch_updateMSelf(Tensor_lst_t &M_out, const Tensor_lst_t &M,
                        const Tensor_lst_t &M_e, const Tensor_lst_t &E,
                        const Tensor_lst_t &S, const torch::Tensor &mask,
                        const data_t alpha, const data_t beta_0,
                        const data_t beta_f, const data_t beta_r,
                        const int I_0) {
    constexpr int num_thd = 1024;
    int nthd, nblk;
    cudaOccupancyMaxPotentialBlockSize(&nblk, &nthd, updateMSelf_kernel);

    auto F_dptr = to_accessor_lst<data_t, 2, torch::RestrictPtrTraits>(
        {M_out, M, M_e, S});
    auto I_dptr = to_accessor_lst<index_t, 2, torch::RestrictPtrTraits>({E});
    auto mask_accessor =
        mask.packed_accessor32<index_t, 2, torch::RestrictPtrTraits>();
    data_t beta = beta_0;
    F_accessor_t *out = F_dptr[0];
    F_accessor_t *in = F_dptr[1];
    int it = 0;
    while (beta <= beta_f && it < I_0) {
        it++;
        updateMSelf_kernel<<<nblk * 2, num_thd>>>(out, in, F_dptr[2], I_dptr[0],
                                                  F_dptr[3], mask_accessor,
                                                  alpha, beta, E.size());
        beta *= beta_r;
        std::swap(out, in);
    }
    if (out == F_dptr[0]) {
        for (int i = 0; i < M_out.size(); i++) {
            M_out[i].copy_(M[i], true);
        }
    }
    gpuErrchk(cudaFree(F_dptr[0]));
    gpuErrchk(cudaFree(I_dptr[0]));
}

void launch_updateM(Tensor_lst_t &M_out, const Tensor_lst_t &M,
                    const Tensor_lst_t &M_E1, const Tensor_lst_t &M_E2,
                    const Tensor_lst_t &E1, const Tensor_lst_t &E2,
                    const Tensor_lst_t &S, const torch::Tensor &mask,
                    const data_t alpha, const data_t beta_0,
                    const data_t beta_f, const data_t beta_r, const int I_0) {
    constexpr int num_thd = 1024;
    auto F_dptr = to_accessor_lst<data_t, 2, torch::RestrictPtrTraits>(
        {M_out, M, M_E1, M_E2, S});
    auto I_dptr =
        to_accessor_lst<index_t, 2, torch::RestrictPtrTraits>({E1, E2});
    int nthd, nblk;
    cudaOccupancyMaxPotentialBlockSize(&nblk, &nthd, updateM_kernel<num_thd>);
    auto mask_accessor =
        mask.packed_accessor32<index_t, 2, torch::RestrictPtrTraits>();
    data_t beta = beta_0;
    F_accessor_t *out = F_dptr[0];
    F_accessor_t *in = F_dptr[1];
    int it = 0;
    while (beta <= beta_f && it < I_0) {
        it++;
        updateM_kernel<num_thd><<<nblk, num_thd>>>(
            out, in, F_dptr[2], F_dptr[3], I_dptr[0], I_dptr[1], F_dptr[4],
            mask_accessor, alpha, beta, E1.size(), E2.size());
        beta *= beta_r;
        std::swap(out, in);
    }
    if (out == F_dptr[0]) {
        for (int i = 0; i < M_out.size(); i++) {
            M_out[i].copy_(M[i], true);
        }
    }
    gpuErrchk(cudaFree(F_dptr[0]));
    gpuErrchk(cudaFree(I_dptr[0]));
}
