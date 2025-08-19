// 尝试增加线程数，降低每个线程的寄存器使用量
#include<cuda_runtime.h>
#include"macro.h"
#include<cstdint>
#include<cuda/barrier>
#include<cuda/pipeline>
#include<cuda.h>
#include"wgmma_utils.cuh"

namespace K9 {
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
    const int QSIZE,
    const int BN = WGMMA_N
>
__global__ void __launch_bounds__(THREADS_NUM) gemm_wgmma(int M, int N, int K, CUtensorMap* tensorMapA, CUtensorMap* tensorMapB, int* C) {
    extern __shared__ __align__(128) int8_t smem[];
    int8_t* As = smem;
    int8_t* Bs = smem+BM*BK*QSIZE;

    const int BKITER = K/BK;
    const int BLOCK_TILE_COL = blockIdx.x%((N+BN-1)/BN);
    const int BLOCK_TILE_ROW = blockIdx.x/((N+BN-1)/BN);
    const int wg_idx = threadIdx.x/128;
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar[QSIZE];

    if(threadIdx.x == 0) {
        for(int i = 0; i < QSIZE; i++) {
            init(&bar[i], THREADS_NUM);
        }
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    barrier::arrival_token token[2];

    if(threadIdx.x == 0) {
        cde::cp_async_bulk_tensor_2d_global_to_shared(As, tensorMapA, 0, BLOCK_TILE_ROW*BM, bar[0]);
        cde::cp_async_bulk_tensor_2d_global_to_shared(Bs, tensorMapB, 0, BLOCK_TILE_COL*BN, bar[0]);
        token[0] = cuda::device::barrier_arrive_tx(bar[0], 1, (BK*BM+BK*BN)*sizeof(int8_t));
    }
    else token[0] = bar[0].arrive();

    int q_idx = 0;

    int d[WGMMA_N/2];
    memset(d, 0, sizeof(d));

    for(int k_iter = 0; k_iter < BKITER; k_iter++) {
        if(k_iter+1 < BKITER) {
            int next_idx = q_idx^1;
            if(threadIdx.x == 0) {
                cde::cp_async_bulk_tensor_2d_global_to_shared(&As[next_idx*BK*BM], tensorMapA, (k_iter+1)*BK, BLOCK_TILE_ROW*BM, bar[next_idx]);
                cde::cp_async_bulk_tensor_2d_global_to_shared(&Bs[next_idx*BK*BN], tensorMapB, (k_iter+1)*BK, BLOCK_TILE_COL*BN, bar[next_idx]);
                token[next_idx] = cuda::device::barrier_arrive_tx(bar[next_idx], 1, (BK*BM+BK*BN)*sizeof(int8_t));
            }
            else token[next_idx] = bar[next_idx].arrive();
        }
        bar[q_idx].wait(std::move(token[q_idx]));
        warpgroup_arrive();
        int8_t* As_ptr = As+wg_idx*WGMMA_M*BK+q_idx*BK*BM;
        #pragma unroll
        for(int k_it = 0; k_it < BK/WGMMA_K; k_it++) {
            wgmma<WGMMA_N, 1>(d, &As_ptr[k_it*WGMMA_K], &Bs[q_idx*BK*BN+k_it*WGMMA_K]);
        }
        warpgroup_commit_batch();
        warpgroup_wait<0>(); // 0表示等待所有任务完成
        q_idx ^= 1;
    }

    const int tid = threadIdx.x%128;
    const int lane = tid%32, warp = tid/32;
    const int row = warp*16+lane/4, col = (lane%4)*2;
    int* C_ptr = C+BLOCK_TILE_ROW*BM*N+BLOCK_TILE_COL*BN+wg_idx*WGMMA_M*N;
    const int WGMMA_N_ITER = (BLOCK_TILE_COL == (N+BN-1)/BN-1) ? (N-(N/BN)*BN)/8 : WGMMA_N/8;

    #pragma unroll
    for(int i = 0; i < WGMMA_N_ITER; i++) {
        GET_INT2(&C_ptr[row*N+col+i*8]) = GET_INT2(&d[i*4]);
        GET_INT2(&C_ptr[(row+8)*N+col+i*8]) = GET_INT2(&d[i*4+2]);
    }
}

CUtensorMap *d_tma_map_A = 0;
CUtensorMap *d_tma_map_B = 0;
int _prev_m = 0, _prev_n = 0, _prev_k = 0;

void run_kernel_9(int M, int N, int K, int8_t* A, int8_t* B, int* C) {
    constexpr int BM = 128, BN = 224, BK = 128;
    constexpr int WGMMA_M = 64, WGMMA_N = 224, WGMMA_K = 32;
    static_assert(BK%128 == 0); // Swizzle要求行跨度必须为Swizzle尺寸的倍数
    constexpr int THREADS_NUM = 128*2;
    const int QSIZE = 2;

    const int N_padded = ((N+BN-1)/BN)*BN;

    if(!d_tma_map_A || M != _prev_m || N_padded != _prev_n || K != _prev_k) {
        d_tma_map_A = allocate_tensor_map<BM, BK, 2>(A, M, K);
        d_tma_map_B = allocate_tensor_map<BN, BK, 2>(B, N_padded, K);
        _prev_m = M;
        _prev_n = N_padded;
        _prev_k = K;
    }

    auto* kernel = gemm_wgmma<BM, BK, WGMMA_M, WGMMA_N, WGMMA_K, THREADS_NUM, QSIZE>;
    const int SMEM_SIZE = (BM+BN)*BK*QSIZE*sizeof(int8_t);

    CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));

    kernel<<<M*N_padded/BN/BM, THREADS_NUM, SMEM_SIZE>>>(M, N, K, d_tma_map_A, d_tma_map_B, C);
}
}

using K9::run_kernel_9;