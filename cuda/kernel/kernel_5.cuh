#include<cuda_runtime.h>
#include"macro.h"
#include<cstdint>
#include<cuda/barrier>
#include<cuda/pipeline>
#include<cuda.h>
#include"wgmma_utils.cuh"

namespace K5 {
using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

// 默认分块大小等于WGMMA大小
template<
    const int BK,
    const int WGMMA_M,
    const int WGMMA_N,
    const int WGMMA_K,
    const int THREADS_NUM,
    const int BM = WGMMA_M,
    const int BN = WGMMA_N
>
__global__ void gemm_wgmma(int M, int N, int K, CUtensorMap* tensorMapA, CUtensorMap* tensorMapB, int* C) {
    __shared__ alignas(128) int8_t As[BM*BK];
    __shared__ alignas(128) int8_t Bs[BN*BK];

    int d[WGMMA_N/2];
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
    for(int bkIt = 0; bkIt < BKITER; bkIt++) {
        if(threadIdx.x == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(As, tensorMapA, bkIt*BK, BLOCK_TILE_ROW*BM, barA);
            tokenA = cuda::device::barrier_arrive_tx(barA, 1, sizeof(As));
            cde::cp_async_bulk_tensor_2d_global_to_shared(Bs, tensorMapB, bkIt*BK, BLOCK_TILE_COL*BN, barB);
            tokenB = cuda::device::barrier_arrive_tx(barB, 1, sizeof(Bs));
        }
        else {
            tokenA = barA.arrive();
            tokenB = barB.arrive();
        }
        barA.wait(std::move(tokenA));
        barB.wait(std::move(tokenB));
        __syncthreads();

        warpgroup_arrive();
        #pragma unroll
        for(int k_inner = 0; k_inner < BK/WGMMA_K; k_inner++) {
            wgmma<WGMMA_N, 1>(d, &As[k_inner*WGMMA_K], &Bs[k_inner*WGMMA_K]);
        }
        warpgroup_commit_batch();
        warpgroup_wait<0>(); // 0表示等待所有任务完成
    }

    int lane = threadIdx.x%32, warp = threadIdx.x/32;
    int row = warp*16+lane/4, col = (lane%4)*2;
    int* C_ptr = C+BLOCK_TILE_ROW*BM*N+BLOCK_TILE_COL*BN;

    for(int i = 0; i < WGMMA_N/8; i++) {
        GET_INT2(&C_ptr[row*N+col+i*8]) = GET_INT2(&d[i*4]);
        GET_INT2(&C_ptr[(row+8)*N+col+i*8]) = GET_INT2(&d[i*4+2]);
    }
}

CUtensorMap *d_tma_map_A = 0;
CUtensorMap *d_tma_map_B = 0;
int _prev_m = 0, _prev_n = 0, _prev_k = 0;

void run_kernel_5(int M, int N, int K, int8_t* A, int8_t* B, int* C) {
    constexpr int BM = 64, BN = 64, BK = 128;
    static_assert(BK%128 == 0); // Swizzle要求行跨度必须为Swizzle尺寸的倍数
    constexpr int THREADS_NUM = 128;
    if(!d_tma_map_A || M != _prev_m || N != _prev_n || K != _prev_k) {
        d_tma_map_A = allocate_tensor_map<BM, BK>(A, M, K);
        d_tma_map_B = allocate_tensor_map<BN, BK>(B, N, K);
        _prev_m = M;
        _prev_n = N;
        _prev_k = K;
    }

    gemm_wgmma<BK, 64, 64, 32, THREADS_NUM>
    <<<M*N/BN/BM, THREADS_NUM>>>(M, N, K, d_tma_map_A, d_tma_map_B, C);
}
}

using K5::run_kernel_5;