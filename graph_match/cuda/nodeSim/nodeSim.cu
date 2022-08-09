#include "../similarity.h"
#include "nodeSim.h"
#include <initializer_list>
using accessor_t =
    torch::PackedTensorAccessor32<data_t, 2, torch::RestrictPtrTraits>;

using I_accessor_t =
    torch::PackedTensorAccessor32<index_t, 2, torch::RestrictPtrTraits>;

// The similarity matrix has N1 rows and N2 columns.
// S has num_G1 * num_G2 matrices in total
// S[g1 * num_G2 + g2] represents the similarity between g1 and g2
__global__ void __launch_bounds__(1024)
    nodeSim_kernel(accessor_t *__restrict__ S,
                   const accessor_t *__restrict__ G1,
                   const accessor_t *__restrict__ G2, const I_accessor_t mask,
                   const size_t num_G1, const size_t num_G2) {
    for (int idx = blockIdx.x; idx < num_G1 * num_G2; idx += gridDim.x) {
        const int g1_id = idx / num_G2;
        const int g2_id = idx % num_G2;
        if (mask[g1_id][g2_id] == 0)
            continue;
        const auto &g1 = G1[g1_id];
        const auto &g2 = G2[g2_id];
        auto &target_S = S[idx];
        const auto N1 = g1.size(0);
        const auto N2 = g2.size(0);
        const auto D = g1.size(1);
        for (auto i = threadIdx.x; i < N1 * N2; i += blockDim.x) {
            const int n1 = i / N2;
            const int n2 = i % N2;
#if (CUDA_VERSION >= 11000)
            __stcs(&target_S[n1][n2], nodeSim(g1[n1], g2[n2], D));
#else
            target_S[n1][n2] = nodeSim(g1[n1], g2[n2], D);
#endif
        }
    }
}

void launch_nodeSim(Tensor_lst_t &S, const Tensor_lst_t &G1,
                    const Tensor_lst_t &G2, const torch::Tensor &mask) {
    auto ptr_vec =
        to_accessor_lst<data_t, 2, torch::RestrictPtrTraits>({G1, G2, S});
    auto mask_accessor =
        mask.packed_accessor32<index_t, 2, torch::RestrictPtrTraits>();

    accessor_t *G1_ptr = ptr_vec[0], *G2_ptr = ptr_vec[1], *S_ptr = ptr_vec[2];
    int nthd, nblk;
    cudaOccupancyMaxPotentialBlockSize(&nblk, &nthd, nodeSim_kernel);
    nodeSim_kernel<<<nblk, 128>>>(S_ptr, G1_ptr, G2_ptr, mask_accessor,
                                  G1.size(), G2.size());

    cudaFree(ptr_vec[0]);
}

// The similarity matrix has N1 rows and N2 columns.
// S has N * (N-1) / 2 matrices in total
// Refer to triu_k2ij or triu_ij2k for the indexing
__global__ void __launch_bounds__(1024)
    nodeSimSelf_kernel(accessor_t *__restrict__ S,
                       const accessor_t *__restrict__ M_n,
                       const I_accessor_t mask, const size_t N) {
    for (int S_idx = blockIdx.x; S_idx < N * (N - 1) / 2; S_idx += gridDim.x) {
        int g1_id, g2_id;
        triu_k2ij(S_idx, N, g1_id, g2_id);
        if (mask[g1_id][g2_id] == 0)
            continue;
        const auto &g1 = M_n[g1_id];
        const auto &g2 = M_n[g2_id];
        const auto N1 = g1.size(0);
        const auto N2 = g2.size(0);
        const auto D = g1.size(1);
        auto &target_S = S[S_idx];
        for (auto i = threadIdx.x; i < N1 * N2; i += blockDim.x) {
            const int n1 = i / N2;
            const int n2 = i % N2;
#if (CUDA_VERSION >= 11000)
            __stcs(&target_S[n1][n2], nodeSim(g1[n1], g2[n2], D));
#else
            target_S[n1][n2] = nodeSim(g1[n1], g2[n2], D);
#endif
        }
    }
}

void launch_nodeSimSelf(Tensor_lst_t &S, const Tensor_lst_t &M_n,
                        const torch::Tensor &mask) {
    auto ptr_vec =
        to_accessor_lst<data_t, 2, torch::RestrictPtrTraits>({M_n, S});

    accessor_t *M_n_ptr = ptr_vec[0], *S_ptr = ptr_vec[1];
    auto mask_accessor =
        mask.packed_accessor32<index_t, 2, torch::RestrictPtrTraits>();

    int nthd, nblk;
    cudaOccupancyMaxPotentialBlockSize(&nblk, &nthd, nodeSimSelf_kernel);
    nodeSimSelf_kernel<<<nblk, 128>>>(S_ptr, M_n_ptr, mask_accessor,
                                      M_n.size());

    cudaFree(ptr_vec[0]);
}
