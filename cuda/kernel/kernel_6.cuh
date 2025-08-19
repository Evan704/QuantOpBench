// 增大分块大小
#include<cuda_runtime.h>
#include"macro.h"
#include<cstdint>
#include<cuda/barrier>
#include<cuda/pipeline>
#include<cuda.h>
#include"wgmma_utils.cuh"

namespace K6 {
using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

// 默认BN与WGMMA_N相同
template<
    const int BM,
    const int BK,
    const int WGMMA_M,
    const int WGMMA_N,
    const int WGMMA_K,
    const int THREADS_NUM,
    const bool DBG,
    const int BN = WGMMA_N
>
__global__ void __launch_bounds__(THREADS_NUM) gemm_wgmma(int M, int N, int K, CUtensorMap* tensorMapA, CUtensorMap* tensorMapB, int* C, int* dbg) {
    __shared__ alignas(128) int8_t As[BM*BK];
    __shared__ alignas(128) int8_t Bs[BN*BK];

    // 每次WGMMA存储WGMMA_N*4/8个结果
    int d[BM/WGMMA_M*WGMMA_N/2];
    memset(d, 0, sizeof(d));

    const int BKITER = K/BK;
    const int BLOCK_TILE_COL = blockIdx.x%(N/BN);
    const int BLOCK_TILE_ROW = blockIdx.x/(N/BN);
    __shared__ barrier barA, barB;

    if(threadIdx.x == 0) {
        init(&barA, THREADS_NUM);
        init(&barB, THREADS_NUM);
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    barrier::arrival_token tokenA, tokenB;
    int sumLoad = 0, cntLoad = 0;
    int sumCompute = 0, cntCompute = 0;
    int sumStore = 0, cntStore = 0;
    for(int bkIt = 0; bkIt < BKITER; bkIt++) {
        clock_t start = clock();
        if(threadIdx.x == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(&As[0], tensorMapA, bkIt*BK, BLOCK_TILE_ROW*BM, barA);
            tokenA = cuda::device::barrier_arrive_tx(barA, 1, sizeof(As));
            cde::cp_async_bulk_tensor_2d_global_to_shared(&Bs[0], tensorMapB, bkIt*BK, BLOCK_TILE_COL*BN, barB);
            tokenB = cuda::device::barrier_arrive_tx(barB, 1, sizeof(Bs));
        }
        else {
            tokenA = barA.arrive();
            tokenB = barB.arrive();
        }
        barA.wait(std::move(tokenA));
        barB.wait(std::move(tokenB));
        __syncthreads();
        if constexpr(DBG) {
            sumLoad += clock() - start;
            cntLoad++;
            start = clock();
        }

        warpgroup_arrive();
        #pragma unroll
        for(int m_it = 0; m_it < BM/WGMMA_M; m_it++) {
            int8_t* As_ptr = As+m_it*WGMMA_M*BK;
            #pragma unroll
            for(int k_it = 0; k_it < BK/WGMMA_K; k_it++) {
                wgmma<WGMMA_N, 1>(&d[m_it*WGMMA_N/2], &As_ptr[k_it*WGMMA_K], &Bs[k_it*WGMMA_K]);
            }
        }
        warpgroup_commit_batch();
        warpgroup_wait<0>(); // 0表示等待所有任务完成
        if constexpr (DBG) {
            sumCompute += clock() - start;
            cntCompute++;
        }
    }

    const int lane = threadIdx.x%32, warp = threadIdx.x/32;
    const int row = warp*16+lane/4, col = (lane%4)*2;
    int* C_ptr = C+BLOCK_TILE_ROW*BM*N+BLOCK_TILE_COL*BN;
    int* d_ptr = d;

    clock_t start = clock();
    for(int m_it = 0; m_it < BM/WGMMA_M; m_it++) {
        #pragma unroll
        for(int i = 0; i < WGMMA_N/8; i++) {
            GET_INT2(&C_ptr[row*N+col+i*8]) = GET_INT2(&d_ptr[i*4]);
            GET_INT2(&C_ptr[(row+8)*N+col+i*8]) = GET_INT2(&d_ptr[i*4+2]);
        }
        C_ptr += WGMMA_M*N;
        d_ptr += WGMMA_N/2;
    }
    if constexpr (DBG) {
        sumStore += clock() - start;
        cntStore++;
        if (threadIdx.x == 63) {
            int i = blockIdx.x*6;
            dbg[i] = sumLoad; dbg[i+1] = cntLoad;
            dbg[i+2] = sumCompute; dbg[i+3] = cntCompute;
            dbg[i+4] = sumStore; dbg[i+5] = cntStore;
        }
    }
}

CUtensorMap *d_tma_map_A = 0;
CUtensorMap *d_tma_map_B = 0;
int _prev_m = 0, _prev_n = 0, _prev_k = 0;

void run_kernel_6(int M, int N, int K, int8_t* A, int8_t* B, int* C, int* dbg) {
    constexpr int BM = 128, BN = 128, BK = 128;
    static_assert(BK%128 == 0); // Swizzle要求行跨度必须为Swizzle尺寸的倍数
    constexpr int THREADS_NUM = 128;

    if(!d_tma_map_A || M != _prev_m || N != _prev_n || K != _prev_k) {
        d_tma_map_A = allocate_tensor_map<BM, BK, 2>(A, M, K);
        d_tma_map_B = allocate_tensor_map<BN, BK, 2>(B, N, K);
        _prev_m = M;
        _prev_n = N;
        _prev_k = K;
    }

    if(dbg != nullptr) {
        gemm_wgmma<BM, BK, 64, 128, 32, THREADS_NUM, true>
        <<<M*N/BN/BM, THREADS_NUM>>>(M, N, K, d_tma_map_A, d_tma_map_B, C, dbg);
    }
    else {
        gemm_wgmma<BM, BK, 64, 128, 32, THREADS_NUM, false>
        <<<M*N/BN/BM, THREADS_NUM>>>(M, N, K, d_tma_map_A, d_tma_map_B, C, dbg);
    }
}
}

using K6::run_kernel_6;