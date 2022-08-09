#pragma once
#include <cstdio>
#include <initializer_list>
#include <torch/extension.h>
#include <vector>
#include <cuda.h>
using Tensor_lst_t = std::vector<torch::Tensor>;
using index_t = int64_t;
using data_t = float;

constexpr int MAX_NODES = 32;

#if defined(__CUDACC__)
#ifndef CUDA_VERSION
#error CUDA_VERSION Undefined!
#endif
#define gpuErrchk(ans)                                                         \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = false) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
                line);
        if (abort)
            exit(code);
    }
}

template <typename T, int N, template <typename U> typename Trait>
auto to_accessor_lst(std::vector<Tensor_lst_t> lsts) {
    using accessor_t = torch::PackedTensorAccessor32<T, N, Trait>;
    accessor_t *dev_ptr;
    size_t _size = 0;
    for (auto lst : lsts)
        _size += lst.size();
    gpuErrchk(cudaMalloc(&dev_ptr, sizeof(accessor_t) * _size));
    std::vector<accessor_t> accessor_vec;
    accessor_vec.reserve(_size);
    for (auto lst : lsts) {
        for (auto &t : lst) {
            accessor_vec.push_back(t.packed_accessor32<T, N, Trait>());
        }
    }
    gpuErrchk(cudaMemcpy(dev_ptr, accessor_vec.data(),
                         sizeof(accessor_t) * accessor_vec.size(),
                         cudaMemcpyHostToDevice));

    // return dev_ptr;

    std::vector<accessor_t *> res;
    size_t offset = 0;
    for (auto lst : lsts) {
        accessor_t *ptr = &dev_ptr[offset];
        res.push_back(ptr);
        offset += lst.size();
    }
    return res;
}

template <typename T> __device__ T warp_sum(T val) {
    for (int i = 1; i < 32; i *= 2)
        val += __shfl_xor_sync(0xffffffff, val, i);
    return val;
}

__device__ void triu_k2ij(int k, int n, int &i, int &j) {
    i = n - 2 -
        __float2int_rd(sqrtf(float(-8 * k + 4 * n * (n - 1) - 7)) * 0.5 - 0.5);
    j = k + i + 1 - ((n * (n - 1)) / 2) + (((n - i) * ((n - i) - 1)) / 2);
}

__device__ void triu_ij2k(int &k, int n, int i, int j) {
    k = (n * (n - 1) / 2) - (n - i) * ((n - i) - 1) / 2 + j - i - 1;
}

#define SIM_FUNC __device__ __inline__
template <typename accessor_t>
SIM_FUNC data_t nodeSim(const accessor_t &v1, const accessor_t &v2,
                        const size_t len);
template <typename accessor_t>
SIM_FUNC data_t edgeSim(const accessor_t &v1, const accessor_t &v2,
                        const size_t len);

template <typename accessor_t>
__device__ index_t get_edge(const index_t a, const index_t b,
                            const accessor_t &E) {
    for (int i = 0; i < E.size(1); ++i) {
        if (__ldg(&E[0][i]) == a && __ldg(&E[1][i]) == b)
            return i;
    }
    return -1;
}

template <typename F_accessor_t, typename I_accessor_t>
__device__ data_t C(const index_t a, const index_t i, const index_t b,
                    const index_t j, const F_accessor_t &M_e1,
                    const F_accessor_t &M_e2, const I_accessor_t &E1,
                    const I_accessor_t &E2) {
    // call by a thread
    const index_t e1_idx = get_edge(a, b, E1);
    const index_t e2_idx = get_edge(i, j, E2);
    data_t val = 0;
    if (e1_idx != -1 && e2_idx != -1) {
        const auto D = M_e1.size(1);
        val = edgeSim(M_e1[e1_idx], M_e2[e2_idx], D);
    }
    return val;
}

#endif