#include<cuda_runtime.h>
#include"macro.h"
#include<cstdint>
#include<cuda/barrier>
#include<cuda/pipeline>
#include<cuda.h>
#include"wgmma_utils.cuh"

namespace K11 {
namespace cde = cuda::device::experimental;

__device__ static __forceinline__ void init_barrier(uint64_t* bar, int thread_count, int transaction_count) {
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar)); 
    asm volatile (
        "mbarrier.init.shared::cta.b64 [%0], %1;\n"
        :: "r"(bar_ptr), "r"(thread_count+transaction_count)
    );
}

__device__ static __forceinline__ void expect_bytes(uint64_t* bar, uint32_t bytes) {
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar)); 
    asm volatile ("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
        :: "r"(bar_ptr), "r"(bytes));
}

__device__ static inline void load_async(int8_t *dst, void const* const src_tma_map, uint64_t* bar, int global_col_idx, int global_row_idx) {
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(src_tma_map);
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    uint32_t dst_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(dst));

    __asm__ __volatile__ (
        "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%3, %4, 0, 0, 0}], [%2];"
        :
        : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr), "r"(global_col_idx), "r"(global_row_idx)
        : "memory"
    );
}

__device__ static __forceinline__ void wait(uint64_t* bar, int kPhaseBit) {
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar)); 
    asm volatile (
        "{\n"
        ".reg .pred                P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "@P1                       bra.uni DONE;\n"
        "bra.uni                   LAB_WAIT;\n"
        "DONE:\n"
        "}\n"
        :: "r"(mbar_ptr),
        "r"(kPhaseBit)
    );
}

__device__ static __forceinline__ void arrive(uint64_t* bar, uint32_t count=1) {
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar)); 
    asm volatile (
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;\n"
        :
        : "r"(mbar_ptr), "r"(count)
        : "memory"
    );
}

// 默认BN与WGMMA_N相同
template<
    const int BM,
    const int BK,
    const int WGMMA_M,
    const int WGMMA_N,
    const int WGMMA_K,
    const int THREADS_NUM,
    const int QSIZE,
    const int BN = WGMMA_N
>
__global__ void __launch_bounds__(THREADS_NUM) gemm_wgmma(int M, int N, int K, CUtensorMap* tensorMapA, CUtensorMap* tensorMapB, int* C) {
    extern __shared__ __align__(128) int8_t smem[];
    int8_t* As = smem;
    int8_t* Bs = smem+BM*BK*QSIZE;

    const int BKITER = K/BK;
    const int BLOCK_TILE_COL = blockIdx.x%(N/BN);
    const int BLOCK_TILE_ROW = blockIdx.x/(N/BN);
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ __align__(8) uint64_t bar[QSIZE];

    if(threadIdx.x == 0) {
        for(int i = 0; i < QSIZE; i++) {
            init_barrier(&bar[i], 0, 1);
        }
    }
    __syncthreads();

    if(threadIdx.x == 0) {
        expect_bytes(&bar[0], (BK*BM+BK*BN)*sizeof(int8_t));
        load_async(As, tensorMapA, &bar[0], 0, BLOCK_TILE_ROW*BM);
        load_async(Bs, tensorMapB, &bar[0], 0, BLOCK_TILE_COL*BN);
    }

    int q_idx = 0, phase = 1;

    int d[BM/WGMMA_M*WGMMA_N/2];
    memset(d, 0, sizeof(d));

    for(int k_iter = 0; k_iter < BKITER; k_iter++) {
        if(q_idx == 0) phase ^= 1;
        if(k_iter+1 < BKITER) {
            int next_idx = q_idx^1;
            if(threadIdx.x == 0) {
                expect_bytes(&bar[next_idx], (BK*BM+BK*BN)*sizeof(int8_t));
                load_async(&As[next_idx*BK*BM], tensorMapA, &bar[next_idx], (k_iter+1)*BK, BLOCK_TILE_ROW*BM);
                load_async(&Bs[next_idx*BK*BN], tensorMapB, &bar[next_idx], (k_iter+1)*BK, BLOCK_TILE_COL*BN);
            }
        }
        wait(&bar[q_idx], phase);
        warpgroup_arrive();
        #pragma unroll
        for(int m_it = 0; m_it < BM/WGMMA_M; m_it++) {
            int8_t* As_ptr = As+m_it*WGMMA_M*BK+q_idx*BK*BM;
            #pragma unroll
            for(int k_it = 0; k_it < BK/WGMMA_K; k_it++) {
                wgmma<WGMMA_N, 1>(&d[m_it*WGMMA_N/2], &As_ptr[k_it*WGMMA_K], &Bs[q_idx*BK*BN+k_it*WGMMA_K]);
            }
        }
        warpgroup_commit_batch();
        warpgroup_wait<0>(); // 0表示等待所有任务完成
        q_idx ^= 1;
    }

    const int tid = threadIdx.x%128;
    const int lane = tid%32, warp = tid/32;
    const int row = warp*16+lane/4, col = (lane%4)*2;
    int* C_ptr = C+BLOCK_TILE_ROW*BM*N+BLOCK_TILE_COL*BN;
    int* d_ptr = d;

    for(int m_it = 0; m_it < BM/WGMMA_M; m_it++) {
        #pragma unroll
        for(int i = 0; i < WGMMA_N/8; i++) {
            GET_INT2(&C_ptr[row*N+col+i*8]) = GET_INT2(&d_ptr[i*4]);
            GET_INT2(&C_ptr[(row+8)*N+col+i*8]) = GET_INT2(&d_ptr[i*4+2]);
        }
        C_ptr += WGMMA_M*N;
        d_ptr += WGMMA_N/2;
    }
}

CUtensorMap *d_tma_map_A = 0;
CUtensorMap *d_tma_map_B = 0;
int _prev_m = 0, _prev_n = 0, _prev_k = 0;

void run_kernel_11(int M, int N, int K, int8_t* A, int8_t* B, int* C) {
    constexpr int BM = 128, BN = 128, BK = 128;
    const int WGMMA_M = 64, WGMMA_N = 128, WGMMA_K = 32;
    static_assert(BK%128 == 0); // Swizzle要求行跨度必须为Swizzle尺寸的倍数
    constexpr int THREADS_NUM = 128;
    const int QSIZE = 2;

    if(!d_tma_map_A || M != _prev_m || N != _prev_n || K != _prev_k) {
        d_tma_map_A = allocate_tensor_map<BM, BK>(A, M, K);
        d_tma_map_B = allocate_tensor_map<BN, BK>(B, N, K);
        _prev_m = M;
        _prev_n = N;
        _prev_k = K;
    }

    auto* kernel = gemm_wgmma<BM, BK, WGMMA_M, WGMMA_N, WGMMA_K, THREADS_NUM, QSIZE>;
    const int SMEM_SIZE = (BM+BN)*BK*QSIZE*sizeof(int8_t);

    CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));

    kernel<<<M*N/BN/BM, THREADS_NUM, SMEM_SIZE>>>(M, N, K, d_tma_map_A, d_tma_map_B, C);
}
}

using K11::run_kernel_11;