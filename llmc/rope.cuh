/*
RoPE kernels and launchers: rotary positional embeddings for Q/K
*/
#include <assert.h>
#include "cuda_common.h"
#include "cuda_utils.cuh"

// RoPE: apply rotary embeddings in-place to fused QKV buffer (only Q and K)
// fused_qkv shape per token: [3*C] with segments [Q(0..C-1), K(C..2C-1), V(2C..3C-1)]
// C = NH * HS, HS must be even. Applies pairwise rotation across last dim of Q,K per head.
__global__ void rope_apply_fused_qk_kernel(floatX* fused_qkv,
                                           int B, int T, int C, int NH, float theta) {
    int HS = C / NH;
    if ((HS & 1) != 0) return; // require even head size
    int bt = blockIdx.x; // one block per (B*T)
    int lane = threadIdx.x;
    if (bt >= B * T) return;
    int b = bt / T;
    int t = bt % T;
    floatX* base = fused_qkv + (size_t)(b * T + t) * 3 * C;
    // iterate heads and pairs within head
    for (int h = lane; h < NH; h += blockDim.x) {
        int head_offset = h * HS;
        // rotate Q and K segments for this head
        floatX* q = base + 0 * C + head_offset;
        floatX* k = base + 1 * C + head_offset;
        for (int m = 0; m < HS/2; ++m) {
            int i0 = 2*m;
            int i1 = i0 + 1;
            float angle = t * powf(theta, -2.0f * (float)m / (float)HS);
            float cs = cosf(angle);
            float sn = sinf(angle);
            // Q rotation
            float q0 = (float)q[i0];
            float q1 = (float)q[i1];
            float q0r = q0 * cs - q1 * sn;
            float q1r = q1 * cs + q0 * sn;
            q[i0] = (floatX)q0r;
            q[i1] = (floatX)q1r;
            // K rotation
            float k0 = (float)k[i0];
            float k1 = (float)k[i1];
            float k0r = k0 * cs - k1 * sn;
            float k1r = k1 * cs + k0 * sn;
            k[i0] = (floatX)k0r;
            k[i1] = (floatX)k1r;
        }
    }
}

// Inverse rotation for gradients fused dQKV (only dQ and dK)
__global__ void rope_apply_fused_qk_backward_kernel(floatX* dfused_qkv,
                                                    int B, int T, int C, int NH, float theta) {
    int HS = C / NH;
    if ((HS & 1) != 0) return;
    int bt = blockIdx.x;
    int lane = threadIdx.x;
    if (bt >= B * T) return;
    int b = bt / T;
    int t = bt % T;
    floatX* base = dfused_qkv + (size_t)(b * T + t) * 3 * C;
    for (int h = lane; h < NH; h += blockDim.x) {
        int head_offset = h * HS;
        floatX* dq = base + 0 * C + head_offset;
        floatX* dk = base + 1 * C + head_offset;
        for (int m = 0; m < HS/2; ++m) {
            int i0 = 2*m;
            int i1 = i0 + 1;
            float angle = t * powf(theta, -2.0f * (float)m / (float)HS);
            float cs = cosf(angle);
            float sn = sinf(angle);
            // inverse rotate gradients: g' = R(-angle) g
            float gq0 = (float)dq[i0];
            float gq1 = (float)dq[i1];
            float gq0r = gq0 * cs + gq1 * sn;
            float gq1r = gq1 * cs - gq0 * sn;
            dq[i0] = (floatX)gq0r;
            dq[i1] = (floatX)gq1r;

            float gk0 = (float)dk[i0];
            float gk1 = (float)dk[i1];
            float gk0r = gk0 * cs + gk1 * sn;
            float gk1r = gk1 * cs - gk0 * sn;
            dk[i0] = (floatX)gk0r;
            dk[i1] = (floatX)gk1r;
        }
    }
}

inline void apply_rope_inplace_fused_qk(floatX* fused_qkv,
                                        int B, int T, int C, int NH, float theta,
                                        cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 128;
    int grid = B * T;
    rope_apply_fused_qk_kernel<<<grid, block_size, 0, stream>>>(fused_qkv, B, T, C, NH, theta);
    cudaCheck(cudaGetLastError());
}

inline void apply_rope_inplace_fused_qk_backward(floatX* dfused_qkv,
                                                 int B, int T, int C, int NH, float theta,
                                                 cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 128;
    int grid = B * T;
    rope_apply_fused_qk_backward_kernel<<<grid, block_size, 0, stream>>>(dfused_qkv, B, T, C, NH, theta);
    cudaCheck(cudaGetLastError());
}

