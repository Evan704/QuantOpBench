// tile_trans
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
__global__ void gemm_tile_trans(int M, int N, int K, int8_t* A, int8_t* B, int* C) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y, tid = ty*blockDim.x+tx;

    __shared__ __align__(128) int8_t sh_A[BK][BM];
    __shared__ __align__(128) int8_t sh_B[BK][BN];

    int acc[RM][RN] = {0};

    int8_t A_frag[RM], B_frag[RN];

    const int TILE_ROW_START = tid/THREADS_PER_ROW;
    const int COL = tid%THREADS_PER_ROW*16;

    int8_t* A_tile_ptr = A+(TILE_ROW_START+by*BM)*K+COL;
    int8_t* B_tile_ptr = B+(TILE_ROW_START+bx*BN)*K+COL;

    int8_t ldg[16];

    #pragma unroll
    for(int k = 0; k < K; k += BK) {
        // load A from gmem to smem
        #pragma unroll
        for(int i = 0; i < BM; i += ROW_STRIDE) {
            GET_INT4(&ldg) = GET_INT4(&A_tile_ptr[i*K+k]);
            #pragma unroll
            for(int j = 0; j < 16; j++) {
                sh_A[COL+j][TILE_ROW_START+i] = ldg[j];
            }
        }
        // load B from gmem to smem
        #pragma unroll
        for(int i = 0; i < BN; i += ROW_STRIDE) {
            GET_INT4(&ldg) = GET_INT4(&B_tile_ptr[i*K+k]);
            #pragma unroll
            for(int j = 0; j < 16; j++) {
                sh_B[COL+j][TILE_ROW_START+i] = ldg[j];
            }
        }
        __syncthreads();
        
        // load data from smem to reg
        for(int i = 0; i < BK; i++) {
            #pragma unroll
            for(int j = 0; j < RM; j += 16) {
                GET_INT4(&A_frag[j]) = GET_INT4(&sh_A[i][ty*RM+j]);
            }
            #pragma unroll
            for(int j = 0; j < RM; j += 16) {
                GET_INT4(&B_frag[j]) = GET_INT4(&sh_B[i][tx*RN+j]);
            }

            // calc
            #pragma unroll
            for(int y = 0; y < RM; y++) {
                #pragma unroll
                for(int x = 0; x < RN; x++) {
                    acc[y][x] += (int)A_frag[y]*(int)B_frag[x];
                }
            }
        }
        __syncthreads();
    }

    // write back to gmem
    int* C_thread_ptr = &C[(BM*by+RM*ty)*N+BN*bx+RN*tx];
    #pragma unroll
    for(int y = 0; y < RM; y++) {
        #pragma unroll
        for(int x = 0; x < RN; x += 4) {
            GET_INT4(&C_thread_ptr[y*N+x]) = GET_INT4(&acc[y][x]);
        }
    }
} // 11.44 tflops

void run_kernel_2(int M, int N, int K, int8_t* A, int8_t* B, int* C) {
    const int BM = 256, BN = 256, BK = 16;
    const int RM = 16, RN = 16;
    dim3 block_shape(BN/RN, BM/RM);
    dim3 grid_shape(N/BN, M/BM);
    gemm_tile_trans<BM, BN, BK, RM, RN><<<grid_shape, block_shape>>>(M, N, K, A, B, C);
}