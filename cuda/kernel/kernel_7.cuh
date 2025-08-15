#include<cuda_runtime.h>
#include"macro.h"
#include<cstdint>
#include<cuda/barrier>
#include<cuda/pipeline>
#include<cuda.h>

namespace K7 {
using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

// Copied from https://github.com/pranjalssh/fast.cu
__device__ static inline uint64_t matrix_descriptor_encode(uint64_t x) { return (((x) & 0x3FFFF) >> 0x4); }

__device__ uint64_t make_smem_desc(int8_t* ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = 0x0000000000000000;
    desc |= matrix_descriptor_encode(addr);
    desc |= matrix_descriptor_encode((uint64_t)16) << 16;
    desc |= matrix_descriptor_encode((uint64_t)1024) << 32;
    desc |= 1llu << 62; // 128B swizzle
    return desc;
}

__device__ void warpgroup_arrive() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ void warpgroup_commit_batch() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int N>
__device__ void warpgroup_wait() {
    static_assert(N >= 0 && N <= 7, "WGMMA wait: N must be in range [0, 7]");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}

template<const int ScaleD>
__device__ void wgmma_m64n64k32(int* d, int8_t* As, int8_t* Bs) {
    uint64_t desc_A = make_smem_desc(As);
    uint64_t desc_B = make_smem_desc(Bs);
    __asm__ __volatile__(
        "wgmma.mma_async.sync.aligned.m64n64k32.s32.s8.s8 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31},"
        " %32,"
        " %33,"
        " %34;"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), 
        "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]), 
        "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), 
        "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]), 
        "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), 
        "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]), 
        "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), 
        "+r"(d[28]), "+r"(d[29]), "+r"(d[30]), "+r"(d[31])
        : "l"(desc_A), "l"(desc_B), "n"((int32_t)(ScaleD))
    );
}

template<const int ScaleD>
__device__ void wgmma_m64n128k32(int* d, int8_t* As, int8_t* Bs) {
    uint64_t desc_A = make_smem_desc(As);
    uint64_t desc_B = make_smem_desc(Bs);
    __asm__ __volatile__(
        "wgmma.mma_async.sync.aligned.m64n128k32.s32.s8.s8 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63},"
        " %64,"
        " %65,"
        " %66;"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), 
        "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]), 
        "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), 
        "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]), 
        "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), 
        "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]), 
        "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), 
        "+r"(d[28]), "+r"(d[29]), "+r"(d[30]), "+r"(d[31]),
        "+r"(d[32]), "+r"(d[33]), "+r"(d[34]), "+r"(d[35]), 
        "+r"(d[36]), "+r"(d[37]), "+r"(d[38]), "+r"(d[39]), 
        "+r"(d[40]), "+r"(d[41]), "+r"(d[42]), "+r"(d[43]), 
        "+r"(d[44]), "+r"(d[45]), "+r"(d[46]), "+r"(d[47]), 
        "+r"(d[48]), "+r"(d[49]), "+r"(d[50]), "+r"(d[51]), 
        "+r"(d[52]), "+r"(d[53]), "+r"(d[54]), "+r"(d[55]), 
        "+r"(d[56]), "+r"(d[57]), "+r"(d[58]), "+r"(d[59]), 
        "+r"(d[60]), "+r"(d[61]), "+r"(d[62]), "+r"(d[63])
        : "l"(desc_A), "l"(desc_B), "n"((int32_t)(ScaleD))
    );
}

template<const int WGMMA_N, const int ScaleD>
__device__ void wgmma(int* d, int8_t* As, int8_t* Bs) {
    static_assert(WGMMA_N == 64 || WGMMA_N == 128);
    if constexpr(WGMMA_N == 64) {
        wgmma_m64n64k32<ScaleD>(d, As, Bs);
    }
    else if constexpr(WGMMA_N == 128) {
        wgmma_m64n128k32<ScaleD>(d, As, Bs);
    }
}

// 默认BN与WGMMA_N相同
// 一个producer一个consumer
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
    __shared__ barrier full[QSIZE], empty[QSIZE];

    if(threadIdx.x == 0) {
        for(int i = 0; i < QSIZE; i++) {
            init(&full[i], 128+1);
            init(&empty[i], 128+1);
        }
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    const int wg_idx = threadIdx.x/128;

    if(wg_idx == 0) {
        // Producer
        if(threadIdx.x == 0) {
            int q_idx = 0;
            for(int k_iter = 0; k_iter < BKITER; k_iter++) {
                if(q_idx == QSIZE) q_idx = 0;
                empty[q_idx].wait(empty[q_idx].arrive());
                cde::cp_async_bulk_tensor_2d_global_to_shared(&As[q_idx*BK*BM], tensorMapA, k_iter*BK, BLOCK_TILE_ROW*BM, full[q_idx]);
                cde::cp_async_bulk_tensor_2d_global_to_shared(&Bs[q_idx*BK*BN], tensorMapB, k_iter*BK, BLOCK_TILE_COL*BN, full[q_idx]);
                cuda::device::barrier_arrive_tx(full[q_idx], 1, (BK*BM+BK*BN)*sizeof(int8_t));
                q_idx++;
            }
        }
    }
    else {
        // Consumer
        // Initialize
        for(int i = 0; i < QSIZE; i++) {
            empty[i].arrive();
        }

        // 每次WGMMA存储WGMMA_N*4/8个结果
        int d[BM/WGMMA_M*WGMMA_N/2];
        memset(d, 0, sizeof(d));

        int q_idx = 0;

        for(int k_iter = 0; k_iter < BKITER; k_iter++) {
            if(q_idx == QSIZE) q_idx = 0;
            full[q_idx].wait(full[q_idx].arrive());
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
            empty[q_idx].arrive();
            q_idx++;
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
            C_ptr += WGMMA_M*BK;
            d_ptr += WGMMA_M*WGMMA_N/2;
        }
    }
}

CUtensorMap *d_tma_map_A = 0;
CUtensorMap *d_tma_map_B = 0;
int _prev_m = 0, _prev_n = 0, _prev_k = 0;

template<const int BH, const int BW>
void create_tensor_map(CUtensorMap* tma_map, int8_t* src, int height, int width) {
    uint64_t globalDim[5] = {(uint64_t)width, (uint64_t)height, 1, 1, 1};
    uint64_t globalStride[5] = {sizeof(int8_t), sizeof(int8_t)*width, 0, 0, 0};
    uint32_t boxDim[5] = {(uint32_t)BW, (uint32_t)BH, 1, 1, 1};
    uint32_t boxStride[5] = {1, 1, 1, 1, 1};
    CUresult result = cuTensorMapEncodeTiled(
        tma_map,
        CU_TENSOR_MAP_DATA_TYPE_UINT8,
        2, (void*)src, globalDim, globalStride+1, boxDim, boxStride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    assert(result == CUDA_SUCCESS);
}

template<const int BH, const int BW>
__host__ CUtensorMap* allocate_tensor_map(int8_t* src, int height, int width) {
    CUtensorMap* d_tma_map;
    cudaMalloc(&d_tma_map, sizeof(CUtensorMap));
    CUtensorMap h_tma_map;
    create_tensor_map<BH, BW>(&h_tma_map, src, height, width);
    cudaMemcpy(d_tma_map, &h_tma_map, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
    return d_tma_map;
}

void run_kernel_7(int M, int N, int K, int8_t* A, int8_t* B, int* C) {
    constexpr int BM = 128, BN = 128, BK = 128;
    const int WGMMA_M = 64, WGMMA_N = 128, WGMMA_K = 32;
    static_assert(BK%128 == 0); // Swizzle要求行跨度必须为Swizzle尺寸的倍数
    constexpr int THREADS_NUM = 128*2;
    const int QSIZE = 5;

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

using K7::run_kernel_7;