// mma
// 使用PTX代替部分CUDA代码
#include<cuda_runtime.h>
#include"macro.h"

template<
    const int BM,
    const int BN,
    const int BK
>
__global__ void gemm_mma(int M, int N, int K, int8_t* A, int8_t* B, int* C) {
    __shared__ __align__(128) int8_t sh_A[BM][BK];
    __shared__ __align__(128) int8_t sh_B[BN][BK];

    // 每个寄存器含4个int8元素, A有16*32=512个元素, 需要512/4/32=4个寄存器, B同理
    u_int A_frag[4], B_frag[2];
    int C_frag[4] = {0};

    int8_t* A_tile_ptr = A+blockIdx.y*BM*K;
    int8_t* B_tile_ptr = B+blockIdx.x*BN*K;

    #pragma unroll
    for(int k = 0; k < K; k += BK) {
        // Gmem to Smem
        // 16*32/32=16, 每个线程加载16个A元素
        #pragma unroll
        for(int i = 0; i < 16; i++) {
            int row = i, col = threadIdx.x;
            sh_A[row][col] = A_tile_ptr[row*K+k+col];
        }
        // 8*32/32=32, 每个线程加载8个B元素，假设主机中B以列主序存储
        #pragma unroll
        for(int i = 0; i < 8; i++) {
            int row = i, col = threadIdx.x;
            sh_B[row][col] = B_tile_ptr[row*K+k+col];
        }

        __asm__ __volatile__(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
            : "=r"(A_frag[0]), "=r"(A_frag[1]), "=r"(A_frag[2]), "=r"(A_frag[3])
            : "l"(__cvta_generic_to_shared((void*)((uintptr_t)sh_A+(threadIdx.x%16)*32+(threadIdx.x/16)*16)))
        );
        __asm__ __volatile__(
            "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
            : "=r"(B_frag[0]), "=r"(B_frag[1])
            : "l"(__cvta_generic_to_shared((void*)((uintptr_t)sh_B+(threadIdx.x%8)*32+((threadIdx.x/8)%2)*16)))
        );

        __asm__ __volatile__(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0, %1, %2, %3}, "      // D
            "{%4, %5, %6, %7}, "      // A
            "{%8, %9}, "              // B
            "{%0, %1, %2, %3};"       // C
            : "+r"(C_frag[0]), "+r"(C_frag[1]), "+r"(C_frag[2]), "+r"(C_frag[3])
            : "r"(A_frag[0]), "r"(A_frag[1]), "r"(A_frag[2]), "r"(A_frag[3]),
            "r"(B_frag[0]), "r"(B_frag[1])
            : "memory"
        );

        __syncthreads();
    }

    int* C_ptr = C+blockIdx.y*BM*N+blockIdx.x*BN+(threadIdx.x%4)*2+(threadIdx.x/4)*N;
    *(reinterpret_cast<int2*>(C_ptr)) = make_int2(C_frag[0], C_frag[1]);
    *(reinterpret_cast<int2*>(C_ptr+8*N)) = make_int2(C_frag[2], C_frag[3]);
}

void run_kernel_3(int M, int N, int K, int8_t* A, int8_t* B, int* C) {
    const int BM = 16, BN = 8, BK = 32;
    dim3 block_shape(32);
    dim3 grid_shape(N/BN, M/BM);
    gemm_mma<BM, BN, BK><<<grid_shape, block_shape>>>(M, N, K, A, B, C);
}