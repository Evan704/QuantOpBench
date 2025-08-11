// tile
#include<cuda_runtime.h>
#include"macro.h"

template<
    const int BM,
    const int BN,
    const int BK,
    const int RM,
    const int RN,
    const int THREADS_PER_BLOCK = BM/RM*BN/RN,
    const int THREADS_PER_ROW = BK/16,
    const int ROW_STRIDE = THREADS_PER_BLOCK/THREADS_PER_ROW
>
__global__ void gemm_tile(int M, int N, int K, int8_t* A, int8_t* B, int* C) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y, tid = ty*blockDim.x+tx;

    __shared__ __align__(128) int8_t sh_A[BM*BK];
    __shared__ __align__(128) int8_t sh_B[BN*BK];

    int acc[RM*RN] = {0};

    int8_t A_frag[RM], B_frag[RN];

    const int TILE_ROW_START = tid/THREADS_PER_ROW;
    const int COL = tid%THREADS_PER_ROW*16;

    int8_t* A_tile_ptr = A+(TILE_ROW_START+by*BM)*K+COL;
    int8_t* B_tile_ptr = B+(TILE_ROW_START+bx*BN)*K+COL;

    for(int k = 0; k < K; k += BK) {
        // load A from gmem to smem
        #pragma unroll
        for(int i = 0; i < BM; i += ROW_STRIDE) {
            GET_INT4(&sh_A[(TILE_ROW_START+i)*BK+COL]) = GET_INT4(&A_tile_ptr[i*K+k]);
        }
        // load B from gmem to smem
        #pragma unroll
        for(int i = 0; i < BN; i += ROW_STRIDE) {
            GET_INT4(&sh_B[(TILE_ROW_START+i)*BK+COL]) = GET_INT4(&B_tile_ptr[i*K+k]);
        }
        __syncthreads();
        
        // load data from smem to reg
        for(int i = 0; i < BK; i++) {
            #pragma unroll
            for(int j = 0; j < RM; j++) {
                A_frag[j] = sh_A[(ty*RM+j)*BK+i];
            }
            #pragma unroll
            for(int j = 0; j < RN; j++) {
                B_frag[j] = sh_B[(tx*RN+j)*BK+i];
            }

            // calc
            #pragma unroll
            for(int y = 0; y < RM; y++) {
                #pragma unroll
                for(int x = 0; x < RN; x++) {
                    acc[y*RN+x] += (int)A_frag[y]*(int)B_frag[x];
                }
            }
        }
    }

    // write back to gmem
    #pragma unroll
    for(int y = 0; y < RM; y++) {
        #pragma unroll
        for(int x = 0; x < RN; x += 4) {
            GET_INT4(&C[(BM*by+RM*ty+y)*N+BN*bx+RN*tx+x]) = GET_INT4(&acc[y*RN+x]);
        }
    }
}

void run_kernel_1(int M, int N, int K, int8_t* A, int8_t* B, int* C) {
    const int BM = 256, BN = 256, BK = 16;
    const int RM = 16, RN = 16;
    dim3 block_shape(BN/RN, BM/RM);
    dim3 grid_shape(N/BN, M/BM);
    gemm_tile<BM, BN, BK, RM, RN><<<grid_shape, block_shape>>>(M, N, K, A, B, C);
}