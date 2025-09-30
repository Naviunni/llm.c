// Simple row-wise softmax and row-wise axpy helpers for MoE gating
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_common.h"

// out: (M, E) softmax over E for each row; templates on storage type T but computes in float
template<typename T>
__global__ void moe_row_softmax_fwd_kernel(T* out, const T* inp, int M, int E) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;
    const T* x = inp + (size_t)row * E;
    T* y = out + (size_t)row * E;
    // find max
    float m = -INFINITY;
    for (int j = 0; j < E; ++j) m = fmaxf(m, (float)x[j]);
    // exp and sum
    float s = 0.f;
    for (int j = 0; j < E; ++j) { float v = __expf((float)x[j] - m); y[j] = (T)v; s += v; }
    float invs = 1.f / s;
    for (int j = 0; j < E; ++j) y[j] = (T)((float)y[j] * invs);
}

// dinp: (M,E) gradient wrt logits given dout wrt softmax output and softmax output out
template<typename T>
__global__ void moe_row_softmax_bwd_kernel(T* dinp, const T* out, const T* dout, int M, int E) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;
    const T* y = out + (size_t)row * E;
    const T* dy = dout + (size_t)row * E;
    T* dz = dinp + (size_t)row * E;
    // compute dot(dy, y)
    float dot = 0.f;
    for (int j = 0; j < E; ++j) dot += (float)dy[j] * (float)y[j];
    for (int j = 0; j < E; ++j) {
        dz[j] = (T)(((float)dy[j] - dot) * (float)y[j]);
    }
}

// Y += diag(g) * X; X,Y shape (M,C), g shape (M)
template<typename T>
__global__ void moe_axpy_rowwise_kernel(T* Y, const T* X, const T* g, int M, int C, int stride_g) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = M * C;
    if (idx >= N) return;
    int row = idx / C;
    T scale = (T)g[(size_t)row * stride_g];
    Y[idx] += scale * X[idx];
}

template<typename T>
inline void moe_row_softmax_forward(T* out, const T* inp, int M, int E, cudaStream_t stream) {
    int threads = 256;
    int blocks = (M + threads - 1) / threads;
    moe_row_softmax_fwd_kernel<T><<<blocks, threads, 0, stream>>>(out, inp, M, E);
    cudaCheck(cudaGetLastError());
}

template<typename T>
inline void moe_row_softmax_backward(T* dinp, const T* out, const T* dout, int M, int E, cudaStream_t stream) {
    int threads = 256;
    int blocks = (M + threads - 1) / threads;
    moe_row_softmax_bwd_kernel<T><<<blocks, threads, 0, stream>>>(dinp, out, dout, M, E);
    cudaCheck(cudaGetLastError());
}

template<typename T>
inline void moe_axpy_rowwise(T* Y, const T* X, const T* g, int M, int C, int stride_g, cudaStream_t stream) {
    int threads = 256;
    int blocks = ((long long)M * C + threads - 1) / threads;
    moe_axpy_rowwise_kernel<<<blocks, threads, 0, stream>>>(Y, X, g, M, C, stride_g);
    cudaCheck(cudaGetLastError());
}

// g[row] += sum_c A[row,c] * B[row,c]
template<typename T>
__global__ void moe_rowwise_dot_accum_kernel(T* g, const T* A, const T* B, int M, int C, int stride_g) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;
    const T* a = A + (size_t)row * C;
    const T* b = B + (size_t)row * C;
    float sum = 0.f;
    for (int j = 0; j < C; ++j) sum += (float)a[j] * (float)b[j];
    size_t idx = (size_t)row * stride_g;
    g[idx] = (T)((float)g[idx] + sum);
}

template<typename T>
inline void moe_rowwise_dot_accum(T* g, const T* A, const T* B, int M, int C, int stride_g, cudaStream_t stream) {
    int threads = 256;
    int blocks = (M + threads - 1) / threads;
    moe_rowwise_dot_accum_kernel<T><<<blocks, threads, 0, stream>>>(g, A, B, M, C, stride_g);
    cudaCheck(cudaGetLastError());
}
