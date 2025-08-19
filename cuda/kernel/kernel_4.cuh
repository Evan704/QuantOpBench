// warp tile
// 每个线程块负责BM*BN，每个warp负责WM*WN，每个线程负责TM*TN
#include<cuda_runtime.h>
#include"macro.h"
#include<iostream>

template<
    const int BM,
    const int BN,
    const int BK,
    const int WM,
    const int WN,
    const int WNITER,
    const int TM,
    const int TN,
    const int THREADS_NUM
>
__global__ void gemm_warp_tile(int M, int N, int K, int8_t* A, int8_t* B, int* C) {
    const int bx = blockIdx.x, by = blockIdx.y;

    // 每个warp负责WM*WN大小矩阵
    const int warpIdx = threadIdx.x/32;
    const int warpCol = warpIdx%(BN/WN);
    const int warpRow = warpIdx/(BN/WN);

    // WSUBM*WSUBN = 32*TM*TN
    // WM = WSUBM*WMITER, WN = WSUBN*WNITER
    constexpr int WMITER = (WM*WN)/(32*TM*TN)/WNITER;
    constexpr int WSUBM = WM/WMITER;
    constexpr int WSUBN = WN/WNITER;

    const int threadIdxInWarp = threadIdx.x%32;
    const int threadColInWarp = threadIdxInWarp%(WSUBN/TN);
    const int threadRowInWarp = threadIdxInWarp/(WSUBN/TN);

    __shared__ int8_t As[BK*BM];
    __shared__ int8_t Bs[BK*BN];

    A += by*BM*K;
    B += bx*BN*K;
    C += (by*BM+warpRow*WM)*N+bx*BN+warpCol*WN;

    // 用INT4加载，一次加载16个int8
    const int innerRowA = threadIdx.x/(BK/16);
    const int innerColA = threadIdx.x%(BK/16);
    constexpr int rowStrideA = (THREADS_NUM*16)/BK;
    // B以列主序存储
    const int innerRowB = threadIdx.x/(BK/16);
    const int innerColB = threadIdx.x%(BK/16);
    constexpr int rowStrideB = (THREADS_NUM*16)/BK;

    // 每个线程处理WMITER*WNITER个小矩阵块, 每块TM*TN
    int threadResults[WMITER*WNITER*TM*TN] = {0};

    // 每个线程储存WMITER个TM长度的片段, ...
    int8_t regM[WMITER*TM], regN[WNITER*TN];

    #pragma unroll
    for(int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // 在Smem中A和B均以K主序存储
        // Load A from gmem to smem
        #pragma unroll
        for(int i = 0; i+rowStrideA <= BM; i += rowStrideA) {
            int8_t tmp[16];
            GET_INT4(&tmp) = GET_INT4(&A[(innerRowA+i)*K+innerColA*16]);
            #pragma unroll
            for(int j = 0; j < 16; j++) {
                As[(innerColA+j)*BM+innerRowA+i] = tmp[j];
            }
        }

        // Load B from gmem to smem
        #pragma unroll
        for(int i = 0; i+rowStrideB <= BN; i += rowStrideB) {
            // GET_INT4(&Bs[(innerRowB+i)*BN+innerColB*16]) = GET_INT4(&B[(innerRowB+i)*N+innerColB*16]);
            int8_t tmp[16];
            GET_INT4(&tmp) = GET_INT4(&B[(innerRowB+i)*K+innerColB*16]);
            #pragma unroll
            for(int j = 0; j < 16; j++) {
                Bs[(innerColB+j)*BN+innerRowB+i] = tmp[j];
            }
        }

        __syncthreads();

        // 对每个Block Tile, 沿着K维度一行一行步进
        #pragma unroll
        for(int k = 0; k < BK; k++) {
            #pragma unroll
            for(int wSubRowIdx = 0; wSubRowIdx < WMITER; wSubRowIdx++) {
                #pragma unroll
                for(int i = 0; i < TM; i++) {
                    regM[wSubRowIdx*TM+i] = As[k*BM+warpRow*WM+wSubRowIdx*WSUBM+threadRowInWarp*TM+i];
                }
            }

            #pragma unroll
            for(int wSubColIdx = 0; wSubColIdx < WNITER; wSubColIdx++) {
                #pragma unroll
                for(int i = 0; i < TN; i++) {
                    regN[wSubColIdx*TN+i] = Bs[k*BN+warpCol*WN+wSubColIdx*WSUBN+threadColInWarp*TN+i];
                }
            }

            //calc
            #pragma unroll
            for(int wSubRowIdx = 0; wSubRowIdx < WMITER; wSubRowIdx++) {
                #pragma unroll
                for(int wSubColIdx = 0; wSubColIdx < WNITER; wSubColIdx++) {
                    #pragma unroll
                    for(int i = 0; i < TM; i++) {
                        #pragma unroll
                        for(int j = 0; j < TN; j++) {
                            threadResults[wSubRowIdx*WNITER*TM*TN+wSubColIdx*TM*TN+i*TN+j] += (int)regM[wSubRowIdx*TM+i]*(int)regN[wSubColIdx*TN+j];
                        }
                    }
                }
            }
        }

        A += BK;
        B += BK;

        __syncthreads();
    }

    #pragma unroll
    for(int wSubRowIdx = 0; wSubRowIdx < WMITER; wSubRowIdx++) {
        #pragma unroll
        for(int wSubColIdx = 0; wSubColIdx < WNITER; wSubColIdx++) {
            int* C_ptr = C+(wSubRowIdx*WSUBM)*N+wSubColIdx*WSUBN;
            #pragma unroll
            for(int i = 0; i < TM; i++) {
                #pragma unroll
                for(int j = 0; j < TN; j += 4) {
                    // 用int4搬运, 一次搬运4个int
                    GET_INT4(&C_ptr[(threadRowInWarp*TM+i)*N+threadColInWarp*TN+j]) = GET_INT4(&threadResults[wSubRowIdx*WNITER*TM*TN+wSubColIdx*TM*TN+i*TN+j]);
                }
            }
        }
    }
}

void run_kernel_4(int M, int N, int K, int8_t* A, int8_t* B, int* C) {
    const int THREADS_NUM = 128;
    const int BM = 128, BN = 128, BK = 16;
    const int WN = 64, WM = 64;
    const int WNITER = 4;
    const int TM = 8, TN = 4;

    constexpr int WARPS_NUM = THREADS_NUM/32;
    static_assert((BN%WN == 0) && (BM%WM == 0));
    static_assert(BN*BM/WN/WM == WARPS_NUM);

    dim3 block_shape(THREADS_NUM);
    dim3 grid_shape(N/BN, M/BM);
    gemm_warp_tile<BM, BN, BK, WM, WN, WNITER, TM, TN, THREADS_NUM>
    <<<grid_shape, block_shape>>>(M, N, K, A, B, C);
}