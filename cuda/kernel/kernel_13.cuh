#include<cuda_runtime.h>
#include"macro.h"
#include<cstdint>
#include<cuda/barrier>
#include<cuda/pipeline>
#include<cuda.h>
#include"wgmma_utils.cuh"
#include"barrier_utils.cuh"

namespace K13 {
namespace cde = cuda::device::experimental;

template <uint32_t RegCount>
__device__ void warpgroup_reg_alloc() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}

template <uint32_t RegCount>
__device__ void warpgroup_reg_dealloc() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}

// 默认BN与WGMMA_N相同
template<
    const int GM,
    const int GN,
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

    const int GMITER = M/BM/GM, GNITER = N/BN/GN;

    const int BKITER = K/BK;
    const int BLOCK_TILE_COL = blockIdx.x%GN;
    const int BLOCK_TILE_ROW = blockIdx.x/GN;
    __shared__ __align__(8) uint64_t full[QSIZE], empty[QSIZE];

    if(threadIdx.x == 0) {
        for(int i = 0; i < QSIZE; i++) {
            init_barrier(&full[i], 0, 1);
            init_barrier(&empty[i], 0, 1);
        }
    }
    __syncthreads();

    const int wg_idx = threadIdx.x/128;

    if(wg_idx == 0) {
        // Producer
        // warpgroup_reg_dealloc<32>();
        if(threadIdx.x == 0) {
            int q_idx = 0, phase = 0;
            for(int gm_it = 0; gm_it < GMITER; gm_it++) {
                for(int gn_it = 0; gn_it < GNITER; gn_it++) {
                    for(int bk_it = 0; bk_it < BKITER; bk_it++) {
                        if(q_idx == QSIZE) {
                            q_idx = 0;
                            phase ^= 1;
                        }
                        wait(&empty[q_idx], phase);
                        expect_bytes(&full[q_idx], (BM*BK+BN*BK)*sizeof(int8_t));
                        load_async(&As[q_idx*BK*BM], tensorMapA, &full[q_idx], bk_it*BK, (BLOCK_TILE_ROW+gm_it*GM)*BM);
                        load_async(&Bs[q_idx*BK*BN], tensorMapB, &full[q_idx], bk_it*BK, (BLOCK_TILE_COL+gn_it*GN)*BN);
                        q_idx++;
                    }
                }
            }
        }
    }
    else {
        // Consumer
        // warpgroup_reg_alloc<256>();
        int d[BM/WGMMA_M*WGMMA_N/2];
        const int tid = threadIdx.x-128;
        for(int i = 0; i < QSIZE; i++) {
            if(tid == 0) arrive(&empty[i], 1);
        }
        int q_idx = 0, phase = 0;
        for(int gm_it = 0; gm_it < GMITER; gm_it++) {
            for(int gn_it = 0; gn_it < GNITER; gn_it++) {
                memset(d, 0, sizeof(d));
                for(int bk_it = 0; bk_it < BKITER; bk_it++) {
                    if(q_idx == QSIZE) {
                        q_idx = 0;
                        phase ^= 1;
                    }
                    wait(&full[q_idx], phase);
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
                    warpgroup_wait<0>();
                    if(tid == 0) arrive(&empty[q_idx], 1);
                    q_idx++;
                }
                const int lane = tid%32, warp = tid/32;
                const int row = warp*16+lane/4, col = (lane%4)*2;
                int* C_ptr = C+(BLOCK_TILE_ROW+gm_it*GM)*BM*N+(BLOCK_TILE_COL+gn_it*GN)*BN;
                int* d_ptr = d;

                #pragma unroll
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
        }
    }
}

CUtensorMap *d_tma_map_A = 0;
CUtensorMap *d_tma_map_B = 0;
int _prev_m = 0, _prev_n = 0, _prev_k = 0;

void run_kernel_13(int M, int N, int K, int8_t* A, int8_t* B, int* C) {
    constexpr int BM = 128, BN = 128, BK = 128;
    const int WGMMA_M = 64, WGMMA_N = 128, WGMMA_K = 32;
    static_assert(BK%128 == 0); // Swizzle要求行跨度必须为Swizzle尺寸的倍数
    constexpr int THREADS_NUM = 128*2;
    constexpr int QSIZE = 5;
    constexpr int SM_NUM = 128;
    constexpr int GM = 16;
    constexpr int GN = SM_NUM/GM;

    if(!d_tma_map_A || M != _prev_m || N != _prev_n || K != _prev_k) {
        d_tma_map_A = allocate_tensor_map<BM, BK, 5>(A, M, K);
        d_tma_map_B = allocate_tensor_map<BN, BK, 5>(B, N, K);
        _prev_m = M;
        _prev_n = N;
        _prev_k = K;
    }

    auto* kernel = gemm_wgmma<GM, GN, BM, BK, WGMMA_M, WGMMA_N, WGMMA_K, THREADS_NUM, QSIZE>;
    const int SMEM_SIZE = (BM+BN)*BK*QSIZE*sizeof(int8_t);

    CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));

    kernel<<<SM_NUM, THREADS_NUM, SMEM_SIZE>>>(M, N, K, d_tma_map_A, d_tma_map_B, C);
}
}

using K13::run_kernel_13;