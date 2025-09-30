/*
Attention, as a fallback when we do not use the Flash Attention from cuDNN
*/
#include <assert.h>
// llmc internal imports
#include "cuda_common.h"
#include "cuda_utils.cuh"
#include "cublas_common.h"

// ----------------------------------------------------------------------------
// CUDA kernels

// inputs floatX, outputs FP32 (for current FP32-only activation path for this WIP)
__global__ void permute_kernel(floatX* q, floatX* k, floatX* v,
                               const floatX* inp,
                               int B, int N, int NH, int d) {
    // okay so now, this kernel wants Q,K,V to all be of shape (B, NH, N, d)
    // but instead, we have a single tensor QKV (inp) of shape (B, N, 3, NH, d)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * NH * N * d) { return; }

    // Q[b][nh_][n][d_] = inp[b][n][0][nh_][d_]
    int b = idx / (NH * N * d);
    int rest = idx % (NH * N * d);
    int nh_ = rest / (N * d);
    rest = rest % (N * d);
    int n = rest / d;
    int d_ = rest % d;
    int inp_idx = (b * N * 3 * NH * d) + (n * 3 * NH * d) + (0 * NH * d) + (nh_ * d) + d_;
    q[idx] = __ldcs(&inp[inp_idx]);
    k[idx] = __ldcs(&inp[inp_idx + NH * d]);
    v[idx] = __ldcs(&inp[inp_idx + 2 * (NH * d)]);
}

__global__ void permute_kernel_backward(floatX* dinp,
                                        const floatX* dq, const floatX* dk, const floatX* dv,
                                        int B, int N, int NH, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * NH * N * d) { return; }

    int b = idx / (NH * N * d);
    int rest = idx % (NH * N * d);
    int nh_ = rest / (N * d);
    rest = rest % (N * d);
    int n = rest / d;
    int d_ = rest % d;

    int inp_idx = (b * N * 3 * NH * d) + (n * 3 * NH * d) + (0 * NH * d) + (nh_ * d) + d_;
    dinp[inp_idx] = dq[idx];
    dinp[inp_idx + NH * d] = dk[idx];
    dinp[inp_idx + 2 * (NH * d)] = dv[idx];
}

__global__ void unpermute_kernel(floatX* inp, floatX *out, int B, int N, int NH, int d) {
   // out has shape (B, nh, N, d) but we need to unpermute it to (B, N, nh, d)

    int idx = (blockIdx.x * blockDim.x + threadIdx.x);
    // out[b][n][nh_][d_] <- inp[b][nh_][n][d_]
    if (idx >= B * NH * N * d) { return; }

    int b = idx / (NH * N * d);
    int rest = idx % (NH * N * d);
    int nh_ = rest / (N * d);
    rest = rest % (N * d);
    int n = rest / d;
    int d_ = rest % d;
    int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
    out[other_idx] = __ldcs(&inp[idx]);
}

__global__ void unpermute_kernel_backward(floatX* dinp, const floatX *dout, int B, int N, int NH, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * NH * N * d) { return; }

    int b = idx / (NH * N * d);
    int rest = idx % (NH * N * d);
    int nh_ = rest / (N * d);
    rest = rest % (N * d);
    int n = rest / d;
    int d_ = rest % d;
    int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
    dinp[idx] = (floatX)dout[other_idx];
}

__global__ void softmax_forward_kernel5(floatX* out, float inv_temperature, const floatX* inp, int N, int T) {
    // inp, out shape: (N, T, T), where N = B * NH
    // fuses the multiplication by scale inside attention
    // directly autoregressive, so we only compute the lower triangular part
    // uses the online softmax algorithm
    assert(T % 4  == 0);
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    // micro-optimization: we iterate backwards so that
    // after the softmax backward operation completes, the cache retains the
    // part of the matrix close to the upper left corner, which benefits the
    // matmul operation that immediately follows.
    // int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank(); // forward order
    int idx = (gridDim.x - blockIdx.x - 1) * num_warps + warp_id; // backward order
    if(idx >= N * T) {
        return;
    }
    int own_pos = idx % T;
    int pos_by_4 = own_pos / 4;

    // one row of inp, i.e. inp[idx, :] of shape (T,)
    const floatX* x = inp + idx * T;

    // not INF, so we don't get NaNs accidentally when subtracting two values.
    const float flt_max = 340282346638528859811704183484516925440.0f; // to avoid including float.h
    float maxval = -flt_max;
    float sumval = 0.0f;

    const floatX* x_aligned = reinterpret_cast<const floatX*>(__builtin_assume_aligned(x, 16));
    for (int i = lane_id; i < pos_by_4; i += WARP_SIZE) {
        float regarray[4];
        for (int k = 0; k < 4; ++k) {
            regarray[k] = (float)x_aligned[4*i + k];
        }
        float old_maxval = maxval;
        for(int k = 0; k < 4; ++k) {
            maxval = fmaxf(maxval, regarray[k]);
        }
        sumval *= expf(inv_temperature * (old_maxval - maxval));
        for(int k = 0; k < 4; ++k) {
            sumval += expf(inv_temperature * (regarray[k] - maxval));
        }
    }

    if(4*pos_by_4 + lane_id <= own_pos) {
        float old_maxval = maxval;
        maxval = fmaxf(maxval, (float)x[4*pos_by_4 + lane_id]);
        sumval *= expf(inv_temperature * (old_maxval - maxval));
        sumval += expf(inv_temperature * ((float)x[4*pos_by_4 + lane_id] - maxval));
    }

    float global_maxval = warpReduceMax(maxval);
    sumval *= expf(inv_temperature * (maxval - global_maxval));

    float sum = warpReduceSum(sumval);
    float norm = 1.f / sum;

    // divide the whole row by the sum
    for (int i = lane_id; i <= own_pos; i += WARP_SIZE) {
        // recalculation is faster than doing the round-trip through memory.
        float ev = expf(inv_temperature * ((float)__ldcs(x + i) - global_maxval));
        __stcs(out + idx * T + i, (floatX)(ev * norm));
    }
}

__global__ void softmax_autoregressive_backward_inplace_kernel(floatX* datt, const floatX* att,
                                                               int B, int T, int C, float scale) {
    constexpr const int BlockSize = 256;
    constexpr int T_per_block = 4;

    // go through blocks in reverse order, so the slowest block starts first
    int t0 = T - 1 - T_per_block*blockIdx.x;
    int idx = blockIdx.y;

    att += idx * T * T;
    datt += idx * T * T;

    for(int to = 0; to < T_per_block; ++to) {
        int t = t0 - to;
        if(t < 0) return;
        const floatX* att_bth = att + t * T;
        const floatX* datt_bth = datt + t * T;
        floatX* dpreatt_bth = datt + t * T;

        float local_sum = 0;
        for (int t2 = threadIdx.x; t2 <= t; t2 += BlockSize) {
            local_sum += (float)att_bth[t2] * (float)datt_bth[t2];
        }

        local_sum = blockReduce<warpReduceSum>(local_sum);

        for (int t3 = threadIdx.x; t3 < T; t3 += BlockSize) {
            // don't touch the cache. Some parts will still be here from the previous loop, and
            // we want to exploit those.
            if(t3 <= t) {
                float acc = (float) __ldcs(att_bth + t3) * ((float) __ldcs(datt_bth + t3) - local_sum);
                __stcs(dpreatt_bth + t3, (floatX) (scale * acc));
            } else {
                // explicitly set non-causal elements to zero
                __stcs(dpreatt_bth + t3, (floatX)0.f);
            }
        }
    }
}

// ----------------------------------------------------------------------------
// kernel launchers

__global__ void rope_apply_qk_kernel(floatX* q, floatX* k, int B, int T, int NH, int HS, float theta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half = HS / 2;
    int total = B * NH * T * half;
    if (idx >= total) return;
    int d = idx % half; int tmp = idx / half;
    int t = tmp % T; tmp /= T; int nh = tmp % NH; int b = tmp / NH;
    floatX* q_ptr = q + (((b * NH + nh) * T + t) * HS);
    floatX* k_ptr = k + (((b * NH + nh) * T + t) * HS);
    float inv_freq = powf(theta, -(float)d / (float)half);
    float angle = (float)t * inv_freq;
    float c = cosf(angle);
    float s = sinf(angle);
    int even = 2*d;
    int odd  = 2*d + 1;
    float q_e = (float)q_ptr[even];
    float q_o = (float)q_ptr[odd];
    float k_e = (float)k_ptr[even];
    float k_o = (float)k_ptr[odd];
    q_ptr[even] = (floatX)(q_e * c - q_o * s);
    q_ptr[odd]  = (floatX)(q_e * s + q_o * c);
    k_ptr[even] = (floatX)(k_e * c - k_o * s);
    k_ptr[odd]  = (floatX)(k_e * s + k_o * c);
}

__global__ void rope_apply_qk_backward_kernel(floatX* dq, floatX* dk, int B, int T, int NH, int HS, float theta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half = HS / 2;
    int total = B * NH * T * half;
    if (idx >= total) return;
    int d = idx % half; int tmp = idx / half;
    int t = tmp % T; tmp /= T; int nh = tmp % NH; int b = tmp / NH;
    floatX* dq_ptr = dq + (((b * NH + nh) * T + t) * HS);
    floatX* dk_ptr = dk + (((b * NH + nh) * T + t) * HS);
    float inv_freq = powf(theta, -(float)d / (float)half);
    float angle = (float)t * inv_freq;
    float c = cosf(angle);
    float s = sinf(angle);
    int even = 2*d;
    int odd  = 2*d + 1;
    float dqe = (float)dq_ptr[even];
    float dqo = (float)dq_ptr[odd];
    float dke = (float)dk_ptr[even];
    float dko = (float)dk_ptr[odd];
    dq_ptr[even] = (floatX)(dqe * c + dqo * s);
    dq_ptr[odd]  = (floatX)(-dqe * s + dqo * c);
    dk_ptr[even] = (floatX)(dke * c + dko * s);
    dk_ptr[odd]  = (floatX)(-dke * s + dko * c);
}

inline void apply_rope_qk(floatX* q, floatX* k, int B, int T, int C, int NH, float theta, cudaStream_t stream) {
    int HS = C / NH;
    if (HS % 2 != 0) { fprintf(stderr, "RoPE requires even head size.\n"); exit(EXIT_FAILURE);} 
    int total = B * NH * T * (HS/2);
    int block = 256;
    int grid = CEIL_DIV(total, block);
    rope_apply_qk_kernel<<<grid, block, 0, stream>>>(q, k, B, T, NH, HS, theta);
    cudaCheck(cudaGetLastError());
}

inline void apply_rope_qk_backward(floatX* dq, floatX* dk, int B, int T, int C, int NH, float theta, cudaStream_t stream) {
    int HS = C / NH;
    if (HS % 2 != 0) { fprintf(stderr, "RoPE requires even head size.\n"); exit(EXIT_FAILURE);} 
    int total = B * NH * T * (HS/2);
    int block = 256;
    int grid = CEIL_DIV(total, block);
    rope_apply_qk_backward_kernel<<<grid, block, 0, stream>>>(dq, dk, B, T, NH, HS, theta);
    cudaCheck(cudaGetLastError());
}

__global__ void rope_apply_qkvr_inplace_kernel(floatX* qkvr, int B, int T, int C, int NH, int HS, float theta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half = HS / 2;
    int total = B * T * NH * half;
    if (idx >= total) return;
    int d = idx % half; int tmp = idx / half;
    int nh = tmp % NH; tmp /= NH; int t = tmp % T; int b = tmp / T;
    float inv_freq = powf(theta, -(float)d / (float)half);
    float angle = (float)t * inv_freq;
    float c = cosf(angle);
    float s = sinf(angle);
    int even = 2*d;
    int odd  = 2*d + 1;
    size_t base = ((size_t)b * T + t) * (size_t)(3 * C);
    floatX* q = qkvr + base + 0 * C + nh * HS;
    floatX* k = qkvr + base + 1 * C + nh * HS;
    float q_e = (float)q[even]; float q_o = (float)q[odd];
    float k_e = (float)k[even]; float k_o = (float)k[odd];
    q[even] = (floatX)(q_e * c - q_o * s);
    q[odd]  = (floatX)(q_e * s + q_o * c);
    k[even] = (floatX)(k_e * c - k_o * s);
    k[odd]  = (floatX)(k_e * s + k_o * c);
}

__global__ void rope_apply_qkvr_backward_inplace_kernel(floatX* dqkvr, int B, int T, int C, int NH, int HS, float theta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half = HS / 2;
    int total = B * T * NH * half;
    if (idx >= total) return;
    int d = idx % half; int tmp = idx / half;
    int nh = tmp % NH; tmp /= NH; int t = tmp % T; int b = tmp / T;
    float inv_freq = powf(theta, -(float)d / (float)half);
    float angle = (float)t * inv_freq;
    float c = cosf(angle);
    float s = sinf(angle);
    int even = 2*d;
    int odd  = 2*d + 1;
    size_t base = ((size_t)b * T + t) * (size_t)(3 * C);
    floatX* dq = dqkvr + base + 0 * C + nh * HS;
    floatX* dk = dqkvr + base + 1 * C + nh * HS;
    float dqe = (float)dq[even]; float dqo = (float)dq[odd];
    float dke = (float)dk[even]; float dko = (float)dk[odd];
    dq[even] = (floatX)(dqe * c + dqo * s);
    dq[odd]  = (floatX)(-dqe * s + dqo * c);
    dk[even] = (floatX)(dke * c + dko * s);
    dk[odd]  = (floatX)(-dke * s + dko * c);
}

inline void apply_rope_qkvr_inplace(floatX* qkvr, int B, int T, int C, int NH, float theta, cudaStream_t stream) {
    int HS = C / NH; if (HS % 2 != 0) { fprintf(stderr, "RoPE requires even head size.\n"); exit(EXIT_FAILURE);} 
    int total = B * T * NH * (HS/2);
    int block = 256; int grid = CEIL_DIV(total, block);
    rope_apply_qkvr_inplace_kernel<<<grid, block, 0, stream>>>(qkvr, B, T, C, NH, HS, theta);
    cudaCheck(cudaGetLastError());
}

inline void apply_rope_qkvr_backward_inplace(floatX* dqkvr, int B, int T, int C, int NH, float theta, cudaStream_t stream) {
    int HS = C / NH; if (HS % 2 != 0) { fprintf(stderr, "RoPE requires even head size.\n"); exit(EXIT_FAILURE);} 
    int total = B * T * NH * (HS/2);
    int block = 256; int grid = CEIL_DIV(total, block);
    rope_apply_qkvr_backward_inplace_kernel<<<grid, block, 0, stream>>>(dqkvr, B, T, C, NH, HS, theta);
    cudaCheck(cudaGetLastError());
}

void attention_forward(floatX* out, floatX* qkvr, floatX* att,
                       floatX* inp,
                       int B, int T, int C, int NH, cudaStream_t stream,
                       bool use_rope=false, float rope_theta=10000.0f) {
    NVTX_RANGE_FN();
    // Note: `inp` is not needed for backward pass, so we re-use it as a scratch buffer.
    // Its contents will be overwritten by this function.
    const int block_size = 256;

    // inp is (B, T, 3C) QKV
    // preatt, att are (B, NH, T, T)
    // output is (B, T, C)
    const int HS = C / NH; // head size

    // permute and separate inp from (B, T, 3, NH, HS) to 3X (B, NH, T, HS)
    floatX *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    int total_threads = B * NH * T * HS;
    int num_blocks = CEIL_DIV(total_threads, block_size);
    permute_kernel<<<num_blocks, block_size, 0, stream>>>(q, k, v, inp, B, T, NH, HS);
    if (use_rope) { apply_rope_qk(q, k, B, T, C, NH, rope_theta, stream); }

    floatX* preatt = inp; // reuse inp as scratch buffer
    matmul_cublaslt(preatt, k, q, nullptr, T, T, HS, stream, true, false, B * NH, T * HS, T * HS, T * T);

    // multiply all elements of preatt elementwise by scale
    float scale = 1.f / sqrtf(HS);
    int grid_size = CEIL_DIV(B * NH * T * WARP_SIZE, block_size);
    softmax_forward_kernel5<<<grid_size, block_size, 0, stream>>>(att, scale, preatt, B * NH, T);

    // new approach: first cuBLAS another batched matmul
    floatX* vaccum = inp;
    // y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
    matmul_cublaslt(vaccum, v, att, nullptr, HS, T, T, stream, false, false, B * NH, T * HS, T * T, T * HS);

    // now unpermute
    // y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
    num_blocks = CEIL_DIV(B * T * C, block_size);
    unpermute_kernel<<<num_blocks, block_size, 0, stream>>>(vaccum, out, B, T, NH, HS);
    cudaCheck(cudaGetLastError());
}

// the sequence of transformations in this compound op is:
// inp (B,T,3C) -> qkvr (B,T,3C) -> preatt (B,NH,T,T) -> att (B,NH,T,T) -> vaccum (B,T,C) -> out (B,T,C)
void attention_backward(floatX* dqkvr_out, floatX* scratch,
                        const floatX* dout,
                        const floatX* qkvr, floatX* att,
                        int B, int T, int C, int NH, cudaStream_t stream,
                        bool use_rope=false, float rope_theta=10000.0f) {
    NVTX_RANGE_FN();
    const int block_size = 256;
    const int HS = C / NH; // head size

    // unpack convenience pointers into q, k, v
    const floatX *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    floatX *dq, *dk, *dv;
    dq = dqkvr_out + 0 * B * T * C;
    dk = dqkvr_out + 1 * B * T * C;
    dv = dqkvr_out + 2 * B * T * C;

    // backward through the unpermute operation
    int num_blocks = CEIL_DIV(B * T * C, block_size);
    unpermute_kernel_backward<<<num_blocks, block_size, 0, stream>>>(scratch, dout, B, T, NH, HS);
    // backward into pre-attention (dpreatt) and dv using att buffer as workspace
    floatX* dpreatt = att; // reuse attention buffer for dpreatt (in-place ok)
    matmul_cublaslt(dpreatt, v, scratch, nullptr, T, T, HS, stream, true, false, B * NH, T * HS, T * HS, T * T);
    // backward into dv
    matmul_cublaslt(dv, scratch, att, nullptr, HS, T, T, stream, false, true, B * NH, T * HS, T * T, T * HS);
    const float scale = 1.0f / sqrtf((float)HS);
    // backward into preatt. this is an in-place operation on the att buffer
    softmax_autoregressive_backward_inplace_kernel<<<dim3(T / 4, B * NH), 256>>>(dpreatt, att, B, T, C, scale);
    // backward into q
    matmul_cublaslt(dq, k, dpreatt, nullptr, HS, T, T, stream, false, false, B * NH, T * HS, T * T, T * HS);
    // backward into k
    matmul_cublaslt(dk, q, dpreatt, nullptr, HS, T, T, stream, false, true, B * NH, T * HS, T * T, T * HS);
    // undo RoPE rotation for dq, dk before permuting back
    if (use_rope) { apply_rope_qk_backward(dq, dk, B, T, C, NH, rope_theta, stream); }
    // write gradients wrt QKV back to packed (B,T,3C)
    num_blocks = CEIL_DIV(B * NH * T * HS, block_size);
    permute_kernel_backward<<<num_blocks, block_size, 0, stream>>>(dqkvr_out, dq, dk, dv, B, T, NH, HS);
    cudaCheck(cudaGetLastError());
}
