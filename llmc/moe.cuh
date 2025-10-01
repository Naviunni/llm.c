/*
Minimal Mixture-of-Experts helpers: row-wise softmax, row scaling, accumulation, and row-wise dot.
These utilities support dense MoE routing over E experts per (B*T) row.
*/
#pragma once
#include <assert.h>
#include "cuda_common.h"

// Row-wise softmax over E columns for N rows
__global__ void softmax_rows_forward_kernel(floatX* out, const floatX* logits, int N, int E) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;
    const floatX* in = logits + row * E;
    floatX* o = out + row * E;
    // compute max for numerical stability
    float m = -3.402823466e+38f; // ~-FLT_MAX
    for (int j = 0; j < E; ++j) {
        float v = (float)in[j];
        if (v > m) m = v;
    }
    // compute denominator
    float sum = 0.f;
    for (int j = 0; j < E; ++j) {
        sum += expf((float)in[j] - m);
    }
    float inv = 1.f / sum;
    for (int j = 0; j < E; ++j) {
        o[j] = (floatX)(expf((float)in[j] - m) * inv);
    }
}

inline void softmax_rows_forward(floatX* out, const floatX* logits, int N, int E, cudaStream_t stream) {
    int block = 256;
    int grid = (N + block - 1) / block;
    softmax_rows_forward_kernel<<<grid, block, 0, stream>>>(out, logits, N, E);
    cudaCheck(cudaGetLastError());
}

// In-place softmax backward: given dP (input) and P (probs), produce dZ = J^T * dP into dP
__global__ void softmax_rows_backward_inplace_kernel(floatX* dprobs, const floatX* probs, int N, int E) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;
    floatX* dp = dprobs + row * E;
    const floatX* p = probs + row * E;
    float dot = 0.f;
    for (int j = 0; j < E; ++j) {
        dot += (float)dp[j] * (float)p[j];
    }
    for (int j = 0; j < E; ++j) {
        float pj = (float)p[j];
        float d = (float)dp[j] - dot;
        dp[j] = (floatX)(pj * d);
    }
}

inline void softmax_rows_backward_inplace(floatX* dprobs, const floatX* probs, int N, int E, cudaStream_t stream) {
    int block = 256;
    int grid = (N + block - 1) / block;
    softmax_rows_backward_inplace_kernel<<<grid, block, 0, stream>>>(dprobs, probs, N, E);
    cudaCheck(cudaGetLastError());
}

// out[row, :] += scale[row] * inp[row, :], row-major with row size C
__global__ void add_scaled_rows_kernel(floatX* out, const floatX* inp, const floatX* scale, int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C;
    if (idx >= total) return;
    int row = idx / C;
    float s = (float)scale[row];
    out[idx] = (floatX)((float)out[idx] + s * (float)inp[idx]);
}

inline void add_scaled_rows(floatX* out, const floatX* inp, const floatX* scale, int N, int C, cudaStream_t stream) {
    int block = 256;
    int grid = (N * C + block - 1) / block;
    add_scaled_rows_kernel<<<grid, block, 0, stream>>>(out, inp, scale, N, C);
    cudaCheck(cudaGetLastError());
}

// out[row, :] += scale2d[row*E + e] * inp[row, :]
__global__ void add_scaled_rows_col_kernel(floatX* out, const floatX* inp, const floatX* scale2d, int E, int e, int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C;
    if (idx >= total) return;
    int row = idx / C;
    float s = (float)scale2d[row * E + e];
    out[idx] = (floatX)((float)out[idx] + s * (float)inp[idx]);
}

inline void add_scaled_rows_col(floatX* out, const floatX* inp, const floatX* scale2d, int E, int e, int N, int C, cudaStream_t stream) {
    int block = 256;
    int grid = (N * C + block - 1) / block;
    add_scaled_rows_col_kernel<<<grid, block, 0, stream>>>(out, inp, scale2d, E, e, N, C);
    cudaCheck(cudaGetLastError());
}

// out[row, :] = scale[row] * inp[row, :]
__global__ void scale_rows_kernel(floatX* out, const floatX* inp, const floatX* scale, int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C;
    if (idx >= total) return;
    int row = idx / C;
    float s = (float)scale[row];
    out[idx] = (floatX)(s * (float)inp[idx]);
}

inline void scale_rows(floatX* out, const floatX* inp, const floatX* scale, int N, int C, cudaStream_t stream) {
    int block = 256;
    int grid = (N * C + block - 1) / block;
    scale_rows_kernel<<<grid, block, 0, stream>>>(out, inp, scale, N, C);
    cudaCheck(cudaGetLastError());
}

// out[row, :] = scale2d[row*E + e] * inp[row, :]
__global__ void scale_rows_col_kernel(floatX* out, const floatX* inp, const floatX* scale2d, int E, int e, int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C;
    if (idx >= total) return;
    int row = idx / C;
    float s = (float)scale2d[row * E + e];
    out[idx] = (floatX)(s * (float)inp[idx]);
}

inline void scale_rows_col(floatX* out, const floatX* inp, const floatX* scale2d, int E, int e, int N, int C, cudaStream_t stream) {
    int block = 256;
    int grid = (N * C + block - 1) / block;
    scale_rows_col_kernel<<<grid, block, 0, stream>>>(out, inp, scale2d, E, e, N, C);
    cudaCheck(cudaGetLastError());
}

// row-wise dot product: out[row] = sum_k a[row,k] * b[row,k]
__global__ void row_dot_kernel(floatX* out, const floatX* a, const floatX* b, int N, int C) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;
    const floatX* ap = a + row * C;
    const floatX* bp = b + row * C;
    float acc = 0.f;
    for (int k = 0; k < C; ++k) {
        acc += (float)ap[k] * (float)bp[k];
    }
    out[row] = (floatX)acc;
}

inline void row_dot(floatX* out, const floatX* a, const floatX* b, int N, int C, cudaStream_t stream) {
    int block = 256;
    int grid = (N + block - 1) / block;
    row_dot_kernel<<<grid, block, 0, stream>>>(out, a, b, N, C);
    cudaCheck(cudaGetLastError());
}

// dgate2d[row*E + e] += dot(a[row,:], b[row,:])
__global__ void row_dot_add_col_kernel(floatX* dgate2d, const floatX* a, const floatX* b, int E, int e, int N, int C) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;
    const floatX* ap = a + row * C;
    const floatX* bp = b + row * C;
    float acc = 0.f;
    for (int k = 0; k < C; ++k) {
        acc += (float)ap[k] * (float)bp[k];
    }
    dgate2d[row * E + e] = (floatX)((float)dgate2d[row * E + e] + acc);
}

inline void row_dot_add_col(floatX* dgate2d, const floatX* a, const floatX* b, int E, int e, int N, int C, cudaStream_t stream) {
    int block = 256;
    int grid = (N + block - 1) / block;
    row_dot_add_col_kernel<<<grid, block, 0, stream>>>(dgate2d, a, b, E, e, N, C);
    cudaCheck(cudaGetLastError());
}

// out += src elementwise for N elements
__global__ void add_inplace_kernel(floatX* out, const floatX* src, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = (floatX)((float)out[i] + (float)src[i]);
}

inline void add_inplace(floatX* out, const floatX* src, int N, cudaStream_t stream) {
    int block = 256;
    int grid = (N + block - 1) / block;
    add_inplace_kernel<<<grid, block, 0, stream>>>(out, src, N);
    cudaCheck(cudaGetLastError());
}
