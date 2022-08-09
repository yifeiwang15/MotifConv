#pragma once
#include "helper.h"

/*
 * Compute the similarity metric between two vectors v1 and v2
 * Access to v1 and v2 is via v1[i] and v2[i]
 * For math functions: refer to the cuda math library:
 * https://docs.nvidia.com/cuda/cuda-math-api/modules.html
 */

// Similarity measurements for synthetic dataset
// template <typename accessor_t>
// SIM_FUNC float nodeSim(const accessor_t& v1, const accessor_t& v2, const
// size_t len) {
//     float sum = 0.0f;
//     for (size_t i = 0; i < len; i++) {
//         sum += (v1[i] - v2[i]) * (v1[i] - v2[i]);
//     }
//     return expf(-1.0 * sum);
// }

// template <typename accessor_t>
// SIM_FUNC float edgeSim(const accessor_t& v1, const accessor_t& v2, const
// size_t len) {
//     float sum = 0.0f;
//     for (size_t i = 0; i < len; i++) {
//         sum += (v1[i] - v2[i]) * (v1[i] - v2[i]);
//     }
//     return expf(-3.14 * sum);
// }

 Similarity measurements for QM9 dataset
 template <typename accessor_t>
 SIM_FUNC data_t nodeSim(const accessor_t &v1, const accessor_t &v2,
                        const size_t len) {
    data_t sum = (data_t)0;
    for (int i = 0; i < 5; i++) {
    sum += v1[i] * v2[i];
    }
    return sum;
 }

 template <typename accessor_t>
 SIM_FUNC data_t edgeSim(const accessor_t &v1, const accessor_t &v2,
                        const size_t len) {
    for (int i = 0; i < 4; i++) {
        if (fabsf(v1[i] - v2[i]) > 1e-6)
            return 1e-4;
    }
    data_t sum = (data_t)0;
    for (int i = 4; i < len; i++) {
       sum += (v1[i] - v2[i]) * (v1[i] - v2[i]);
    }
    return __expf(-2.0 * sum);

 }
