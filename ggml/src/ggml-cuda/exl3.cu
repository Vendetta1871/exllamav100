#include "exl3.cuh"
#include <cstdint>
#include <mma.h>

namespace wmma = nvcuda::wmma;

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
// ids != nullptr: row r maps to (u = r % ne1, t = r / ne1) and uses expert e = ids[u, t];
// the x rows are broadcast over the expert slots (x_ne1 = x->ne[1], may be 1)
static __global__ void exl3_had_in_kernel(
        const float * __restrict__ x, const half * __restrict__ suh, float * __restrict__ xh,
        const int64_t k,
        const int64_t nb1, const int64_t nb2, const int64_t nb3,
        const int64_t ne1, const int64_t ne2,
        const int32_t * __restrict__ ids, const int64_t ids_s1,
        const int64_t x_ne1) {
    __shared__ float s[128];

    const int tid = threadIdx.x;

    const int64_t row = blockIdx.y;
    const int64_t r1  = row % ne1;
    const int64_t r2  = (row/ne1) % ne2;
    const int64_t r3  = row/(ne1*ne2);

    const int64_t e = ids ? ids[r1 + r2*ids_s1] : 0;

    const float * xr   = (const float *) ((const char *) x + (r1 % x_ne1)*nb1 + r2*nb2 + r3*nb3) + blockIdx.x*128;
    const half  * suhb = suh + e*k + blockIdx.x*128;

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
// ids != nullptr: row r maps to (u = r % ne1, t = r / ne1) and uses expert e = ids[u, t]
template <int K, int cb>
static __global__ void exl3_gemv_kernel(
        const uint32_t * __restrict__ trellis, const float * __restrict__ xh,
        const half * __restrict__ svh, const half * __restrict__ bias, float * __restrict__ dst,
        const int64_t k, const int64_t n, const int64_t rows,
        const int64_t nb1, const int64_t nb2, const int64_t nb3,
        const int64_t ne1, const int64_t ne2,
        const int32_t * __restrict__ ids, const int64_t ids_s1) {
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

        int64_t e = 0;
        if (ids) {
            e = ids[row%ne1 + (row/ne1)*ids_s1];
        }
        const uint32_t * trellis_e = trellis + e*krows*nb*(8*K);
        const half     * svh_e     = svh + e*n;

        for (int64_t krow = warp; krow < krows; krow += 8) {
            // stage the 8 n-adjacent tiles of this k-row (contiguous) into shared
            const uint32_t * src = trellis_e + (krow*nb + blockIdx.x*8)*(8*K);
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
            float v = sh_had[col]*EXL3_HAD128_SCALE*__half2float(svh_e[out_col]);
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

// partial sums part[s][row][*] = xh @ W_dec over one slice of the k-tiles (split-k path,
// the output hadamard is deferred to exl3_epilogue_kernel)
template <int K, int cb>
static __global__ void exl3_gemv_splitk_kernel(
        const uint32_t * __restrict__ trellis, const float * __restrict__ xh,
        float * __restrict__ part,
        const int64_t k, const int64_t n, const int64_t rows,
        const int64_t ne1,
        const int32_t * __restrict__ ids, const int64_t ids_s1) {
    extern __shared__ uint32_t sh[];
    uint32_t * sh_trellis = sh;                          // 8 warps * 64*K words
    float    * sh_red     = (float *) (sh + 8*64*K);     // 8*128 floats

    const int warp = threadIdx.x/32;
    const int lane = threadIdx.x%32;

    const int64_t nb    = n/16; // n-tiles per k-row
    const int64_t krows = k/16;
    const int64_t kps   = (krows + gridDim.y - 1)/gridDim.y; // k-tiles per split
    const int64_t k0    = blockIdx.y*kps;
    const int64_t k1    = min(krows, k0 + kps);

    uint32_t * sh_w = sh_trellis + warp*64*K;

    // tile position of packed weight i = lane*8 + j: r = r0 + {0,1,8,9}[j%4], c = c0 + (j >= 4 ? 8 : 0)
    const int c0 = lane/4;
    const int r0 = (lane%4)*2;

    for (int64_t row = 0; row < rows; ++row) {
        float acc[8][2] = {};

        const float * xh_row = xh + row*k;

        int64_t e = 0;
        if (ids) {
            e = ids[row%ne1 + (row/ne1)*ids_s1];
        }
        const uint32_t * trellis_e = trellis + e*krows*nb*(8*K);

        // prefetch pipeline: pf holds the next k-row's packed words (LDGs issued while decoding)
        constexpr int PF = (16*K + 31)/32; // uint4 per lane per k-row
        uint4 pf[PF];

        for (int64_t krow = k0 + warp; krow < k1; krow += 8) {
            uint4 * dst4 = (uint4 *) sh_w;
            const uint4 * src = (const uint4 *) (trellis_e + (krow*nb + blockIdx.x*8)*(8*K));
            if (krow == k0 + warp) {
#pragma unroll
                for (int i = 0; i < PF; ++i) {
                    const int idx = lane + i*32;
                    if (idx < 16*K) {
                        pf[i] = __ldcs(src + idx);
                    }
                }
            }
#pragma unroll
            for (int i = 0; i < PF; ++i) {
                const int idx = lane + i*32;
                if (idx < 16*K) {
                    dst4[idx] = pf[i];
                }
            }
            if (krow + 8 < k1) {
                const uint4 * src1 = src + 8*nb*(8*K)/4;
#pragma unroll
                for (int i = 0; i < PF; ++i) {
                    const int idx = lane + i*32;
                    if (idx < 16*K) {
                        pf[i] = __ldcs(src1 + idx);
                    }
                }
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
            part[(blockIdx.y*rows + row)*n + blockIdx.x*128 + col] = v;
        }
        __syncthreads(); // protect sh_red reuse on the next row
    }
}

// split-k epilogue: y = had128(sum_s part[s]) (*) svh [+ bias], one block per 128 output columns
static __global__ void exl3_epilogue_kernel(
        const float * __restrict__ part,
        const half * __restrict__ svh, const half * __restrict__ bias, float * __restrict__ dst,
        const int64_t n, const int64_t rows, const int S,
        const int64_t nb1, const int64_t nb2, const int64_t nb3,
        const int64_t ne1, const int64_t ne2,
        const int32_t * __restrict__ ids, const int64_t ids_s1) {
    __shared__ float sh_had[128];

    const int tid = threadIdx.x;
    const int64_t row = blockIdx.y;

    const int64_t out_col = blockIdx.x*128 + tid;

    float v = 0.0f;
    for (int s = 0; s < S; ++s) {
        v += part[(s*rows + row)*n + out_col];
    }
    sh_had[tid] = v;
    __syncthreads();

    // output hadamard
    for (int h = 1; h < 128; h *= 2) {
        if ((tid & h) == 0) {
            const float a = sh_had[tid];
            const float b = sh_had[tid + h];
            sh_had[tid]     = a + b;
            sh_had[tid + h] = a - b;
        }
        __syncthreads();
    }

    int64_t e = 0;
    if (ids) {
        e = ids[row%ne1 + (row/ne1)*ids_s1];
    }
    v = sh_had[tid]*EXL3_HAD128_SCALE*__half2float(svh[e*n + out_col]);
    if (bias) {
        v += __half2float(bias[out_col]);
    }

    const int64_t r1 = row % ne1;
    const int64_t r2 = (row/ne1) % ne2;
    const int64_t r3 = row/(ne1*ne2);
    float * dr = (float *) ((char *) dst + r1*nb1 + r2*nb2 + r3*nb3);
    dr[out_col] = v;
}

// y = had128(xh @ W_dec) (*) svh [+ bias] for M_TILE rows per block, Volta tensor cores
// block: M_TILE rows x 128 output columns (8 n-tiles); warp w handles n-tile w
template <int K, int cb, int M_TILE>
static __global__ void exl3_gemm_kernel(
        const uint32_t * __restrict__ trellis, const float * __restrict__ xh,
        const half * __restrict__ svh, const half * __restrict__ bias, float * __restrict__ dst,
        const int64_t k, const int64_t n, const int64_t rows,
        const int64_t nb1, const int64_t nb2, const int64_t nb3,
        const int64_t ne1, const int64_t ne2,
        const int32_t * __restrict__ ids, const int64_t ids_s1) {
    GGML_UNUSED(ids); // dense only, the ID op always takes the gemv path
    GGML_UNUSED(ids_s1);
    extern __shared__ __align__(32) unsigned char shm[];
    half     * a_sh = (half *) shm;                    // M_TILE*16 halfs, row-major
    half     * b_sh = a_sh + M_TILE*16;                // 8*256 halfs, per-warp col-major tile
    uint32_t * pack = (uint32_t *) (b_sh + 8*256);     // 8*8*K words, per-warp packed tile
    float    * y_sh = (float *) (pack + 8*8*K);        // M_TILE*128 floats, row-major

    const int warp = threadIdx.x/32;
    const int lane = threadIdx.x%32;
    const int tid  = threadIdx.x;

    const int64_t nb    = n/16; // n-tiles per k-row
    const int64_t krows = k/16;
    const int64_t row0  = blockIdx.y*M_TILE;

    uint32_t * pack_w = pack + warp*8*K;
    half     * b_w    = b_sh + warp*256;

    // tile position of packed weight i = lane*8 + j: r = r0 + {0,1,8,9}[j%4], c = c0 + (j >= 4 ? 8 : 0)
    const int c0 = lane/4;
    const int r0 = (lane%4)*2;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[M_TILE/16];
#pragma unroll
    for (int mf = 0; mf < M_TILE/16; ++mf) {
        wmma::fill_fragment(acc[mf], 0.0f);
    }

    for (int64_t kt = 0; kt < krows; ++kt) {
        // stage the A tile (M_TILE x 16) from the F32 xh workspace as fp16, zero-pad edge rows
#pragma unroll
        for (int idx = tid; idx < M_TILE*16; idx += 256) {
            const int64_t r = row0 + idx/16;
            a_sh[idx] = r < rows ? __float2half(xh[r*k + kt*16 + idx%16]) : __float2half(0.0f);
        }

        // stage and decode this warp's 16x16 B tile
        const uint32_t * src = trellis + (kt*nb + blockIdx.x*8 + warp)*(8*K);
#pragma unroll
        for (int i = lane; i < 8*K; i += 32) {
            pack_w[i] = __ldcs(src + i);
        }
        __syncwarp();
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float w = exl3_decode_cb<cb>(exl3_window<K>(pack_w, lane*8 + j));
            b_w[(c0 + (j >= 4 ? 8 : 0))*16 + r0 + (j%4 == 0 ? 0 : j%4 == 1 ? 1 : j%4 == 2 ? 8 : 9)] = __float2half(w);
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
        wmma::load_matrix_sync(b_frag, b_w, 16);
#pragma unroll
        for (int mf = 0; mf < M_TILE/16; ++mf) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
            wmma::load_matrix_sync(a_frag, a_sh + mf*16*16, 16);
            wmma::mma_sync(acc[mf], a_frag, b_frag, acc[mf]);
        }
        __syncthreads();
    }

    // store C fragments to the shared y tile (M_TILE x 128)
#pragma unroll
    for (int mf = 0; mf < M_TILE/16; ++mf) {
        wmma::store_matrix_sync(y_sh + mf*16*128 + warp*16, acc[mf], 128, wmma::mem_row_major);
    }
    __syncthreads();

    // output hadamard + svh + bias per row; each warp handles rows warp, warp+8, ...
    const int64_t nrows_blk = min((int64_t) M_TILE, rows - row0);
    for (int64_t r = warp; r < nrows_blk; r += 8) {
        float * yr = y_sh + r*128;
        for (int h = 1; h < 128; h *= 2) {
            for (int p = lane; p < 64; p += 32) {
                const int i = (p/h)*2*h + p%h;
                const float a = yr[i];
                const float b = yr[i + h];
                yr[i]     = a + b;
                yr[i + h] = a - b;
            }
            __syncwarp();
        }

        const int64_t row = row0 + r;
        const int64_t r1 = row % ne1;
        const int64_t r2 = (row/ne1) % ne2;
        const int64_t r3 = row/(ne1*ne2);
        float * dr = (float *) ((char *) dst + r1*nb1 + r2*nb2 + r3*nb3) + blockIdx.x*128;
        for (int col = lane; col < 128; col += 32) {
            const int64_t out_col = blockIdx.x*128 + col;
            float v = yr[col]*EXL3_HAD128_SCALE*__half2float(svh[out_col]);
            if (bias) {
                v += __half2float(bias[out_col]);
            }
            dr[col] = v;
        }
        __syncwarp();
    }
}

// expert -> row-list mapping for the GEMM-ID path (single block): counting sort of the
// (u,t) rows by expert id, plus a work list of (expert, row-block start) pairs
static __global__ void exl3_id_map_kernel(
        const int32_t * __restrict__ ids, const int64_t ids_s1,
        const int neu, const int64_t rows, const int n_expert,
        int32_t * __restrict__ wcount, int32_t * __restrict__ offs,
        int32_t * __restrict__ sorted, int2 * __restrict__ work) {
    extern __shared__ int32_t sm[]; // counts/cursor[n_expert], woff[n_expert+1]
    int32_t * cnt  = sm;
    int32_t * woff = sm + n_expert;

    const int tid = threadIdx.x;

    for (int e = tid; e < n_expert; e += blockDim.x) {
        cnt[e] = 0;
    }
    __syncthreads();

    for (int64_t r = tid; r < rows; r += blockDim.x) {
        const int e = ids[(int)(r%neu) + (int)(r/neu)*ids_s1];
        atomicAdd(&cnt[e], 1);
    }
    __syncthreads();

    if (tid == 0) {
        int acc = 0, wacc = 0;
        for (int e = 0; e < n_expert; ++e) {
            const int c = cnt[e];
            offs[e] = acc;
            cnt[e]  = acc; // reuse as scatter cursor
            woff[e] = wacc;
            acc  += c;
            wacc += (c + 15)/16;
        }
        offs[n_expert] = acc;
        woff[n_expert] = wacc;
        wcount[0] = wacc;
    }
    __syncthreads();

    for (int64_t r = tid; r < rows; r += blockDim.x) {
        const int e = ids[(int)(r%neu) + (int)(r/neu)*ids_s1];
        sorted[atomicAdd(&cnt[e], 1)] = (int32_t) r;
    }
    __syncthreads();

    for (int e = tid; e < n_expert; e += blockDim.x) {
        const int nb_e = (offs[e+1] - offs[e] + 15)/16;
        for (int b = 0; b < nb_e; ++b) {
            work[woff[e] + b] = make_int2(e, offs[e] + b*16);
        }
    }
}

// y = had128(xh @ W_dec[e]) (*) svh[e] for 16 gathered rows per block, Volta tensor cores
// grid: (n/128, work items); block gathers its 16 rows of expert e via the sorted row list
template <int K, int cb>
static __global__ void exl3_gemm_id_kernel(
        const uint32_t * __restrict__ trellis, const float * __restrict__ xh,
        const half * __restrict__ svh, float * __restrict__ dst,
        const int32_t * __restrict__ wcount, const int32_t * __restrict__ offs,
        const int32_t * __restrict__ sorted, const int2 * __restrict__ work,
        const int64_t k, const int64_t n,
        const int64_t nb1, const int64_t nb2, const int64_t nb3,
        const int64_t ne1, const int64_t ne2) {
    if ((int) blockIdx.y >= wcount[0]) {
        return;
    }

    extern __shared__ __align__(32) unsigned char shm[];
    half     * a_sh   = (half *) shm;                  // 16*16 halfs, row-major
    half     * b_sh   = a_sh + 16*16;                  // 8*256 halfs, per-warp col-major tile
    uint32_t * pack   = (uint32_t *) (b_sh + 8*256);   // 8*8*K words, per-warp packed tile
    float    * y_sh   = (float *) (pack + 8*8*K);      // 16*128 floats, row-major
    int32_t  * rows_sh = (int32_t *) (y_sh + 16*128);  // 16 row ids

    const int warp = threadIdx.x/32;
    const int lane = threadIdx.x%32;
    const int tid  = threadIdx.x;

    const int64_t nb    = n/16; // n-tiles per k-row
    const int64_t krows = k/16;

    const int2  wi    = work[blockIdx.y];
    const int   e     = wi.x;
    const int   start = wi.y;
    const int   end   = min(start + 16, offs[e+1]);

    const uint32_t * trellis_e = trellis + (int64_t) e*krows*nb*(8*K);
    const half     * svh_e     = svh + (int64_t) e*n;

    if (tid < 16) {
        rows_sh[tid] = start + tid < end ? sorted[start + tid] : -1;
    }
    __syncthreads();

    uint32_t * pack_w = pack + warp*8*K;
    half     * b_w    = b_sh + warp*256;

    // tile position of packed weight i = lane*8 + j: r = r0 + {0,1,8,9}[j%4], c = c0 + (j >= 4 ? 8 : 0)
    const int c0 = lane/4;
    const int r0 = (lane%4)*2;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    for (int64_t kt = 0; kt < krows; ++kt) {
        // gather the A tile (16 x 16) from the F32 xh workspace as fp16, zero-pad empty rows
#pragma unroll
        for (int idx = tid; idx < 16*16; idx += 256) {
            const int rr = rows_sh[idx/16];
            a_sh[idx] = rr >= 0 ? __float2half(xh[(int64_t) rr*k + kt*16 + idx%16]) : __float2half(0.0f);
        }

        // stage and decode this warp's 16x16 B tile
        const uint32_t * src = trellis_e + (kt*nb + blockIdx.x*8 + warp)*(8*K);
#pragma unroll
        for (int i = lane; i < 8*K; i += 32) {
            pack_w[i] = __ldcs(src + i);
        }
        __syncwarp();
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float w = exl3_decode_cb<cb>(exl3_window<K>(pack_w, lane*8 + j));
            b_w[(c0 + (j >= 4 ? 8 : 0))*16 + r0 + (j%4 == 0 ? 0 : j%4 == 1 ? 1 : j%4 == 2 ? 8 : 9)] = __float2half(w);
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
        wmma::load_matrix_sync(a_frag, a_sh, 16);
        wmma::load_matrix_sync(b_frag, b_w, 16);
        wmma::mma_sync(acc, a_frag, b_frag, acc);
        __syncthreads();
    }

    // store the C fragment to the shared y tile (16 x 128)
    wmma::store_matrix_sync(y_sh + warp*16, acc, 128, wmma::mem_row_major);
    __syncthreads();

    // output hadamard + svh per row, scatter back by row id; warp w handles rows w, w+8, ...
    for (int r = warp; r < end - start; r += 8) {
        float * yr = y_sh + r*128;
        for (int h = 1; h < 128; h *= 2) {
            for (int p = lane; p < 64; p += 32) {
                const int i = (p/h)*2*h + p%h;
                const float a = yr[i];
                const float b = yr[i + h];
                yr[i]     = a + b;
                yr[i + h] = a - b;
            }
            __syncwarp();
        }

        const int64_t row = rows_sh[r];
        const int64_t r1 = row % ne1;
        const int64_t r2 = (row/ne1) % ne2;
        const int64_t r3 = row/(ne1*ne2);
        float * dr = (float *) ((char *) dst + r1*nb1 + r2*nb2 + r3*nb3) + blockIdx.x*128;
        for (int col = lane; col < 128; col += 32) {
            const int64_t out_col = blockIdx.x*128 + col;
            dr[col] = yr[col]*EXL3_HAD128_SCALE*__half2float(svh_e[out_col]);
        }
        __syncwarp();
    }
}

typedef void (* exl3_gemv_kernel_t)(
        const uint32_t *, const float *, const half *, const half *, float *,
        int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
        const int32_t *, int64_t);

#define EXL3_GEMV_ROW(K) { exl3_gemv_kernel<K, 0>, exl3_gemv_kernel<K, 1>, exl3_gemv_kernel<K, 2> }

static const exl3_gemv_kernel_t exl3_gemv_kernels[9][3] = {
    { nullptr, nullptr, nullptr },
    EXL3_GEMV_ROW(1), EXL3_GEMV_ROW(2), EXL3_GEMV_ROW(3), EXL3_GEMV_ROW(4),
    EXL3_GEMV_ROW(5), EXL3_GEMV_ROW(6), EXL3_GEMV_ROW(7), EXL3_GEMV_ROW(8),
};

#define EXL3_M_TILE 16

#define EXL3_GEMM_ROW(K) { exl3_gemm_kernel<K, 0, EXL3_M_TILE>, exl3_gemm_kernel<K, 1, EXL3_M_TILE>, exl3_gemm_kernel<K, 2, EXL3_M_TILE> }

static const exl3_gemv_kernel_t exl3_gemm_kernels[9][3] = {
    { nullptr, nullptr, nullptr },
    EXL3_GEMM_ROW(1), EXL3_GEMM_ROW(2), EXL3_GEMM_ROW(3), EXL3_GEMM_ROW(4),
    EXL3_GEMM_ROW(5), EXL3_GEMM_ROW(6), EXL3_GEMM_ROW(7), EXL3_GEMM_ROW(8),
};

typedef void (* exl3_gemv_splitk_kernel_t)(
        const uint32_t *, const float *, float *,
        int64_t, int64_t, int64_t, int64_t,
        const int32_t *, int64_t);

#define EXL3_SPLITK_ROW(K) { exl3_gemv_splitk_kernel<K, 0>, exl3_gemv_splitk_kernel<K, 1>, exl3_gemv_splitk_kernel<K, 2> }

static const exl3_gemv_splitk_kernel_t exl3_gemv_splitk_kernels[9][3] = {
    { nullptr, nullptr, nullptr },
    EXL3_SPLITK_ROW(1), EXL3_SPLITK_ROW(2), EXL3_SPLITK_ROW(3), EXL3_SPLITK_ROW(4),
    EXL3_SPLITK_ROW(5), EXL3_SPLITK_ROW(6), EXL3_SPLITK_ROW(7), EXL3_SPLITK_ROW(8),
};

typedef void (* exl3_gemm_id_kernel_t)(
        const uint32_t *, const float *, const half *, float *,
        const int32_t *, const int32_t *, const int32_t *, const int2 *,
        int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);

#define EXL3_GEMM_ID_ROW(K) { exl3_gemm_id_kernel<K, 0>, exl3_gemm_id_kernel<K, 1>, exl3_gemm_id_kernel<K, 2> }

static const exl3_gemm_id_kernel_t exl3_gemm_id_kernels[9][3] = {
    { nullptr, nullptr, nullptr },
    EXL3_GEMM_ID_ROW(1), EXL3_GEMM_ID_ROW(2), EXL3_GEMM_ID_ROW(3), EXL3_GEMM_ID_ROW(4),
    EXL3_GEMM_ID_ROW(5), EXL3_GEMM_ID_ROW(6), EXL3_GEMM_ID_ROW(7), EXL3_GEMM_ID_ROW(8),
};

// split factor for the split-k gemv path: aim for ~240 blocks, at least 8 k-tiles per split
static int exl3_gemv_splits(const int64_t n, const int64_t k, const int64_t rows) {
    if (rows > 8) {
        return 1;
    }
    const int64_t nb128 = n/128;
    const int64_t krows = k/16;
    int S = (int) MIN((240 + nb128 - 1)/nb128, krows/8);
    S = MIN(S, 32);
    return MAX(S, 1);
}

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
        src0->nb[1], src0->nb[2], src0->nb[3], src0->ne[1], src0->ne[2],
        nullptr, 0, src0->ne[1]);

    if (rows > 8) {
        const int shmem = EXL3_M_TILE*16*sizeof(half) + 8*256*sizeof(half) + 8*8*K*sizeof(uint32_t) + EXL3_M_TILE*128*sizeof(float);
        const int m_tiles = (rows + EXL3_M_TILE - 1)/EXL3_M_TILE;
        exl3_gemm_kernels[K][cb]<<<dim3(n/128, m_tiles), 256, shmem, stream>>>(
            (const uint32_t *) src1->data, xh.get(), (const half *) src3->data,
            src4 ? (const half *) src4->data : nullptr, (float *) dst->data,
            k, n, rows,
            dst->nb[1], dst->nb[2], dst->nb[3], dst->ne[1], dst->ne[2],
            nullptr, 0);
        return;
    }

    const int S = exl3_gemv_splits(n, k, rows);
    if (S > 1) {
        ggml_cuda_pool_alloc<float> part(ctx.pool(), (size_t) S*rows*n);
        const int shmem = 8*64*K*sizeof(uint32_t) + 8*128*sizeof(float);
        exl3_gemv_splitk_kernels[K][cb]<<<dim3(n/128, S), 256, shmem, stream>>>(
            (const uint32_t *) src1->data, xh.get(), part.get(),
            k, n, rows, dst->ne[1], nullptr, 0);
        exl3_epilogue_kernel<<<dim3(n/128, rows), 128, 0, stream>>>(
            part.get(), (const half *) src3->data, src4 ? (const half *) src4->data : nullptr,
            (float *) dst->data, n, rows, S,
            dst->nb[1], dst->nb[2], dst->nb[3], dst->ne[1], dst->ne[2],
            nullptr, 0);
        return;
    }

    const int shmem = 8*64*K*sizeof(uint32_t) + (8*128 + 128)*sizeof(float);
    exl3_gemv_kernels[K][cb]<<<dim3(n/128), 256, shmem, stream>>>(
        (const uint32_t *) src1->data, xh.get(), (const half *) src3->data,
        src4 ? (const half *) src4->data : nullptr, (float *) dst->data,
        k, n, rows,
        dst->nb[1], dst->nb[2], dst->nb[3], dst->ne[1], dst->ne[2],
        nullptr, 0);
}

void ggml_cuda_op_exl3_matmul_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0]; // x, F32 [k, n_expert_used, n_tokens]
    const ggml_tensor * src1 = dst->src[1]; // trellis, I16 [16*K, n/16, k/16, n_expert]
    const ggml_tensor * src2 = dst->src[2]; // ids, I32 [n_expert_used, n_tokens]
    const ggml_tensor * src3 = dst->src[3]; // suh, F16 [k, n_expert]
    const ggml_tensor * src4 = dst->src[4]; // svh, F16 [n, n_expert]

    const int K  = ggml_get_op_params_i32(dst, 0);
    const int cb = ggml_get_op_params_i32(dst, 1);

    const int64_t k    = src1->ne[2]*16;
    const int64_t n    = src1->ne[1]*16;
    const int64_t rows = src2->ne[0]*src2->ne[1]; // output rows: n_expert_used x n_tokens (x rows broadcast)

    GGML_ASSERT(src0->type == GGML_TYPE_F32 && src0->nb[0] == sizeof(float));
    GGML_ASSERT(dst->nb[0] == sizeof(float));
    GGML_ASSERT(ggml_is_contiguous(src1) && src2->nb[0] == sizeof(int32_t));
    GGML_ASSERT(ggml_is_contiguous(src3) && ggml_is_contiguous(src4));
    GGML_ASSERT(k % 128 == 0 && n % 128 == 0);
    GGML_ASSERT(K >= 1 && K <= 8 && cb >= 0 && cb <= 2);

    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<float> xh(ctx.pool(), (size_t) rows*k);

    exl3_had_in_kernel<<<dim3(k/128, rows), 128, 0, stream>>>(
        (const float *) src0->data, (const half *) src3->data, xh.get(), k,
        src0->nb[1], src0->nb[2], src0->nb[3], src2->ne[0], src2->ne[1],
        (const int32_t *) src2->data, src2->nb[1]/sizeof(int32_t), src0->ne[1]);

    // rows > 8: group rows by expert and use the wmma GEMM path; rows <= 8 stay on
    // the gemv/split-k path (decode hot path)
    if (rows > 8) {
        const int64_t n_expert = src1->ne[3];
        const int64_t w_max    = n_expert + (rows + 15)/16;

        // workspace: work[2*w_max] | wcount[1] | offs[n_expert+1] | sorted[rows]
        ggml_cuda_pool_alloc<int32_t> ws(ctx.pool(), (size_t) 2*w_max + 1 + (n_expert + 1) + rows);
        int2     * work   = (int2 *) ws.get();
        int32_t  * wcount = ws.get() + 2*w_max;
        int32_t  * offs   = wcount + 1;
        int32_t  * sorted = offs + n_expert + 1;

        const int map_smem = (2*n_expert + 1)*sizeof(int32_t);
        exl3_id_map_kernel<<<1, 256, map_smem, stream>>>(
            (const int32_t *) src2->data, src2->nb[1]/sizeof(int32_t),
            (int) src2->ne[0], rows, (int) n_expert, wcount, offs, sorted, work);

        const int shmem = 16*16*sizeof(half) + 8*256*sizeof(half) + 8*8*K*sizeof(uint32_t)
                        + 16*128*sizeof(float) + 16*sizeof(int32_t);
        exl3_gemm_id_kernels[K][cb]<<<dim3(n/128, w_max), 256, shmem, stream>>>(
            (const uint32_t *) src1->data, xh.get(), (const half *) src4->data, (float *) dst->data,
            wcount, offs, sorted, work,
            k, n,
            dst->nb[1], dst->nb[2], dst->nb[3], dst->ne[1], dst->ne[2]);
        return;
    }

    const int S = exl3_gemv_splits(n, k, rows);
    if (S > 1) {
        ggml_cuda_pool_alloc<float> part(ctx.pool(), (size_t) S*rows*n);
        const int shmem = 8*64*K*sizeof(uint32_t) + 8*128*sizeof(float);
        exl3_gemv_splitk_kernels[K][cb]<<<dim3(n/128, S), 256, shmem, stream>>>(
            (const uint32_t *) src1->data, xh.get(), part.get(),
            k, n, rows, dst->ne[1],
            (const int32_t *) src2->data, src2->nb[1]/sizeof(int32_t));
        exl3_epilogue_kernel<<<dim3(n/128, rows), 128, 0, stream>>>(
            part.get(), (const half *) src4->data, nullptr,
            (float *) dst->data, n, rows, S,
            dst->nb[1], dst->nb[2], dst->nb[3], dst->ne[1], dst->ne[2],
            (const int32_t *) src2->data, src2->nb[1]/sizeof(int32_t));
        return;
    }

    const int shmem = 8*64*K*sizeof(uint32_t) + (8*128 + 128)*sizeof(float);
    exl3_gemv_kernels[K][cb]<<<dim3(n/128), 256, shmem, stream>>>(
        (const uint32_t *) src1->data, xh.get(), (const half *) src4->data,
        nullptr, (float *) dst->data,
        k, n, rows,
        dst->nb[1], dst->nb[2], dst->nb[3], dst->ne[1], dst->ne[2],
        (const int32_t *) src2->data, src2->nb[1]/sizeof(int32_t));
}
