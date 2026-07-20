#include "exl3.cuh"
#include <cstdint>

#define EXL3_HAD128_SCALE (1.0f/11.313708498984761f) // 1/sqrt(128)

// decode one EXL3 codebook value (port of codebook.cuh)
template <int cb>
static __device__ __forceinline__ float exl3_decode_cb(uint32_t w) {
    uint32_t x;
    if constexpr (cb == 0) {
        x = w*89226354u + 64248484u;
    } else if constexpr (cb == 1) {
        x = w*0xCBAC1FEDu;
    } else {
        x = w*0x83DCD12Du;
        const uint32_t s = __dp4a(x, 0x01010101u, 0x6400u);
        const float h = __half2float(__ushort_as_half((unsigned short)(s & 0xffff)));
        return h*__half2float(__ushort_as_half(0x1eee)) + __half2float(__ushort_as_half(0xc931));
    }
    x = (x & 0x8fff8fffu) ^ 0x3b603b60u;
    return __half2float(__ushort_as_half((unsigned short)(x & 0xffff))) +
           __half2float(__ushort_as_half((unsigned short)(x >> 16)));
}

// extract the 16-bit codebook index of weight i in a 256-weight tile (port of unpack_trellis_kernel)
template <int K>
static __device__ __forceinline__ uint32_t exl3_window(const uint32_t * words, int i) {
    const int nwords = 8*K;
    const int b2 = (i + 1)*K + 256*K;
    const int i0 = (b2 - 16)/32;
    const int i1 = (b2 - 1)/32;
    const int s  = (i1 + 1)*32 - b2;
    const uint32_t a = words[i0 % nwords];
    const uint32_t b = words[i1 % nwords];
    return (uint32_t)(((((uint64_t) a) << 32) | b) >> s) & 0xffff;
}

// xh = had128(x (*) suh), one block per 128-element block of one row
static __global__ void exl3_had_in_kernel(
        const float * __restrict__ x, const half * __restrict__ suh, float * __restrict__ xh,
        const int64_t k,
        const int64_t nb1, const int64_t nb2, const int64_t nb3,
        const int64_t ne1, const int64_t ne2) {
    __shared__ float s[128];

    const int tid = threadIdx.x;

    const int64_t row = blockIdx.y;
    const int64_t r1  = row % ne1;
    const int64_t r2  = (row/ne1) % ne2;
    const int64_t r3  = row/(ne1*ne2);

    const float * xr   = (const float *) ((const char *) x + r1*nb1 + r2*nb2 + r3*nb3) + blockIdx.x*128;
    const half  * suhb = suh + blockIdx.x*128;

    s[tid] = xr[tid]*__half2float(suhb[tid]);
    __syncthreads();

    for (int h = 1; h < 128; h *= 2) {
        if ((tid & h) == 0) {
            const float a = s[tid];
            const float b = s[tid + h];
            s[tid]     = a + b;
            s[tid + h] = a - b;
        }
        __syncthreads();
    }

    xh[row*k + blockIdx.x*128 + tid] = s[tid]*EXL3_HAD128_SCALE;
}

// y = had128(xh @ W_dec) (*) svh [+ bias], one block per 128 output columns, rows looped inside
template <int K, int cb>
static __global__ void exl3_gemv_kernel(
        const uint32_t * __restrict__ trellis, const float * __restrict__ xh,
        const half * __restrict__ svh, const half * __restrict__ bias, float * __restrict__ dst,
        const int64_t k, const int64_t n, const int64_t rows,
        const int64_t nb1, const int64_t nb2, const int64_t nb3,
        const int64_t ne1, const int64_t ne2) {
    extern __shared__ uint32_t sh[];
    uint32_t * sh_trellis = sh;                          // 8 warps * 64*K words
    float    * sh_red     = (float *) (sh + 8*64*K);     // 8*128 floats
    float    * sh_had     = sh_red + 8*128;              // 128 floats

    const int warp = threadIdx.x/32;
    const int lane = threadIdx.x%32;

    const int64_t nb    = n/16; // n-tiles per k-row
    const int64_t krows = k/16;

    uint32_t * sh_w = sh_trellis + warp*64*K;

    // tile position of packed weight i = lane*8 + j: r = r0 + {0,1,8,9}[j%4], c = c0 + (j >= 4 ? 8 : 0)
    const int c0 = lane/4;
    const int r0 = (lane%4)*2;

    for (int64_t row = 0; row < rows; ++row) {
        float acc[8][2] = {};

        const float * xh_row = xh + row*k;

        for (int64_t krow = warp; krow < krows; krow += 8) {
            // stage the 8 n-adjacent tiles of this k-row (contiguous) into shared
            const uint32_t * src = trellis + (krow*nb + blockIdx.x*8)*(8*K);
            for (int i = lane; i < 64*K; i += 32) {
                sh_w[i] = __ldcs(src + i);
            }
            __syncwarp();

            const float * xhb = xh_row + krow*16;
            const float x0 = xhb[r0 + 0];
            const float x1 = xhb[r0 + 1];
            const float x2 = xhb[r0 + 8];
            const float x3 = xhb[r0 + 9];

#pragma unroll
            for (int t = 0; t < 8; ++t) {
                const uint32_t * words = sh_w + t*8*K;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float w = exl3_decode_cb<cb>(exl3_window<K>(words, lane*8 + j));
                    const float xv = j%4 == 0 ? x0 : j%4 == 1 ? x1 : j%4 == 2 ? x2 : x3;
                    acc[t][j/4] += w*xv;
                }
            }
            __syncwarp();
        }

        // reduce across the 4 lanes of each quad
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            acc[t][0] += __shfl_xor_sync(0xffffffffu, acc[t][0], 1);
            acc[t][0] += __shfl_xor_sync(0xffffffffu, acc[t][0], 2);
            acc[t][1] += __shfl_xor_sync(0xffffffffu, acc[t][1], 1);
            acc[t][1] += __shfl_xor_sync(0xffffffffu, acc[t][1], 2);
        }
        if (lane%4 == 0) {
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                sh_red[warp*128 + t*16 + c0]     = acc[t][0];
                sh_red[warp*128 + t*16 + c0 + 8] = acc[t][1];
            }
        }
        __syncthreads();

        const int col = threadIdx.x;
        if (col < 128) {
            float v = 0.0f;
#pragma unroll
            for (int w = 0; w < 8; ++w) {
                v += sh_red[w*128 + col];
            }
            sh_had[col] = v;
        }
        __syncthreads();

        // output hadamard
        for (int h = 1; h < 128; h *= 2) {
            if (col < 128 && (col & h) == 0) {
                const float a = sh_had[col];
                const float b = sh_had[col + h];
                sh_had[col]     = a + b;
                sh_had[col + h] = a - b;
            }
            __syncthreads();
        }

        if (col < 128) {
            const int64_t out_col = blockIdx.x*128 + col;
            float v = sh_had[col]*EXL3_HAD128_SCALE*__half2float(svh[out_col]);
            if (bias) {
                v += __half2float(bias[out_col]);
            }
            const int64_t r1 = row % ne1;
            const int64_t r2 = (row/ne1) % ne2;
            const int64_t r3 = row/(ne1*ne2);
            float * dr = (float *) ((char *) dst + r1*nb1 + r2*nb2 + r3*nb3);
            dr[out_col] = v;
        }
        __syncthreads(); // protect sh_red/sh_had reuse on the next row
    }
}

typedef void (* exl3_gemv_kernel_t)(
        const uint32_t *, const float *, const half *, const half *, float *,
        int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);

#define EXL3_GEMV_ROW(K) { exl3_gemv_kernel<K, 0>, exl3_gemv_kernel<K, 1>, exl3_gemv_kernel<K, 2> }

static const exl3_gemv_kernel_t exl3_gemv_kernels[9][3] = {
    { nullptr, nullptr, nullptr },
    EXL3_GEMV_ROW(1), EXL3_GEMV_ROW(2), EXL3_GEMV_ROW(3), EXL3_GEMV_ROW(4),
    EXL3_GEMV_ROW(5), EXL3_GEMV_ROW(6), EXL3_GEMV_ROW(7), EXL3_GEMV_ROW(8),
};

void ggml_cuda_op_exl3_matmul(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0]; // x, F32 [k, rows...]
    const ggml_tensor * src1 = dst->src[1]; // trellis, I16 [16*K, n/16, k/16]
    const ggml_tensor * src2 = dst->src[2]; // suh, F16 [k]
    const ggml_tensor * src3 = dst->src[3]; // svh, F16 [n]
    const ggml_tensor * src4 = dst->src[4]; // bias, F16 [n] or NULL

    const int K  = ggml_get_op_params_i32(dst, 0);
    const int cb = ggml_get_op_params_i32(dst, 1);

    const int64_t k    = src1->ne[2]*16;
    const int64_t n    = src1->ne[1]*16;
    const int64_t rows = src0->ne[1]*src0->ne[2]*src0->ne[3];

    GGML_ASSERT(src0->type == GGML_TYPE_F32 && src0->nb[0] == sizeof(float));
    GGML_ASSERT(dst->nb[0] == sizeof(float));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(k % 128 == 0 && n % 128 == 0);
    GGML_ASSERT(K >= 1 && K <= 8 && cb >= 0 && cb <= 2);

    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<float> xh(ctx.pool(), (size_t) rows*k);

    exl3_had_in_kernel<<<dim3(k/128, rows), 128, 0, stream>>>(
        (const float *) src0->data, (const half *) src2->data, xh.get(), k,
        src0->nb[1], src0->nb[2], src0->nb[3], src0->ne[1], src0->ne[2]);

    const int shmem = 8*64*K*sizeof(uint32_t) + (8*128 + 128)*sizeof(float);
    exl3_gemv_kernels[K][cb]<<<dim3(n/128), 256, shmem, stream>>>(
        (const uint32_t *) src1->data, xh.get(), (const half *) src3->data,
        src4 ? (const half *) src4->data : nullptr, (float *) dst->data,
        k, n, rows,
        dst->nb[1], dst->nb[2], dst->nb[3], dst->ne[1], dst->ne[2]);
}
