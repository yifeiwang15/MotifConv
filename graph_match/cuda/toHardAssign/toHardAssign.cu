#include "toHardAssign.h"
#include <initializer_list>
using accessor_t =
    torch::PackedTensorAccessor32<data_t, 2, torch::RestrictPtrTraits>;

struct ele_t {
    int r;
    int c;
    data_t val;
};

__device__ ele_t block2Dargmax(data_t M[][MAX_NODES], int N1, int N2) {
    ele_t temp{0, 0, -1e5};
    for (int i = threadIdx.x; i < N1 * N2; i += blockDim.x) {
        int r = i / N2;
        int c = i % N2;
        auto val = M[r][c];
        if (temp.val < val)
            temp = {r, c, val};
    }
    for (int i = 1; i < 32; i *= 2) {
        int r = __shfl_xor_sync(0xffffffff, temp.r, i);
        int c = __shfl_xor_sync(0xffffffff, temp.c, i);
        auto val = __shfl_xor_sync(0xffffffff, temp.val, i);
        if (temp.val < val)
            temp = {r, c, val};
    }
    __shared__ ele_t smem[32];
    int lane_id = threadIdx.x % 32;
    if (lane_id == 0)
        smem[threadIdx.x / 32] = temp;
    __syncthreads();
    int num_warp = blockDim.x / 32;
    ele_t res = (lane_id < num_warp) ? smem[lane_id] : ele_t{0, 0, -1e5};
    for (int i = 1; i < 32; i *= 2) {
        int r = __shfl_xor_sync(0xffffffff, res.r, i);
        int c = __shfl_xor_sync(0xffffffff, res.c, i);
        auto val = __shfl_xor_sync(0xffffffff, res.val, i);
        if (res.val < val)
            res = {r, c, val};
    }
    return res;
}

__global__ void __kernel_toHardAssign(accessor_t *M_lst, int size) {
    __shared__ data_t smem[MAX_NODES][MAX_NODES];
    for (int _i = blockIdx.x; _i < size; _i += gridDim.x) {
        auto M = M_lst[_i];
        int N1 = M.size(0);
        int N2 = M.size(1);
        assert(N1 <= MAX_NODES && N2 <= MAX_NODES);
        for (int i = threadIdx.x; i < N1 * N2; i += blockDim.x) {
            int r = i / N2;
            int c = i % N2;
            smem[r][c] = M[r][c];
            M[r][c] = 0;
        }
        __syncthreads();
        for (int rep = 0; rep < min(N1, N2); ++rep) {
            auto k = block2Dargmax(smem, N1, N2);
            for (int i = threadIdx.x; i < N1 * N2; i += blockDim.x) {
                int r = i / N2;
                int c = i % N2;
                if (r == k.r || c == k.c)
                    smem[r][c] = 0;
            }
            if (threadIdx.x == 0) {
                M[k.r][k.c] = 1;
            }
            __syncthreads();
        }
    }
}

void launch_toHardAssign(Tensor_lst_t &M) {
    constexpr int num_thd = 128;
    auto M_dptr = to_accessor_lst<data_t, 2, torch::RestrictPtrTraits>({M});
    __kernel_toHardAssign<<<M.size(), num_thd>>>(M_dptr[0], M.size());
    gpuErrchk(cudaFree(M_dptr[0]));
}
