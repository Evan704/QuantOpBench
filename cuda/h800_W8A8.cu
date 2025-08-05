__device__ __inline__ dim3 rasterization2DColumn(const int panel_width) {
    const auto baseBlockIdx = blockIdx.x + gridDim.x *blockIdx.y;
    const auto totalPanel = (gridDim.x * gridDim.y +panel_width * gridDim.x - 1) / (panel_width * gridDim.x);
    const auto totalBlock = gridDim.x * gridDim.y;
    const auto panelIdx = baseBlockIdx / (panel_width *gridDim.x);
    const auto strideLd = panelIdx + 1 < totalPanel ?panel_width : (totalBlock - panelIdx * (panel_width *gridDim.x)) / gridDim.x;
    const auto bx = (panelIdx & 1) ? gridDim.x -(baseBlockIdx - panelIdx * panel_width * gridDim.x) /strideLd - 1 : (baseBlockIdx - panelIdx * panel_width *gridDim.x) / strideLd;
    const auto by = (baseBlockIdx - panelIdx * panel_width *gridDim.x) % strideLd + panelIdx * panel_width;
    const auto bz = blockIdx.z;

    dim3 blockIdx(bx, by, bz);
    return blockIdx;
}
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610)
#include <sm_61_intrinsics.h>


#if defined(__CUDACC_RTC__)
#define __SM_61_INTRINSICS_DECL__ __device__
#else /* !__CUDACC_RTC__ */
#define __SM_61_INTRINSICS_DECL__ static __device__ __inline__
#endif /* __CUDACC_RTC__ */

#ifndef __CUDA_ARCH__
#define __DEF_IF_HOST { }
#else  /* !__CUDA_ARCH__ */
#define __DEF_IF_HOST ;
#endif /* __CUDA_ARCH__ */

__SM_61_INTRINSICS_DECL__ int __dp4a(unsigned int srcA, int srcB, int c) __DEF_IF_HOST
__SM_61_INTRINSICS_DECL__ int __dp4a(int srcA, unsigned int srcB, int c) __DEF_IF_HOST

#undef __DEF_IF_HOST

#if !defined(__CUDACC_RTC__) && defined(__CUDA_ARCH__)
__SM_61_INTRINSICS_DECL__ int __dp4a(unsigned int srcA, int srcB, int c) {
    int ret;
    asm volatile ("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(ret) : "r"(srcA), "r"(srcB), "r"(c));
    return ret;
}

__SM_61_INTRINSICS_DECL__ int __dp4a(int srcA, unsigned int srcB, int c) {
    int ret;
    asm volatile ("dp4a.s32.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(srcA), "r"(srcB), "r"(c));
    return ret;
}
#endif /* !__CUDACC_RTC__ && defined(__CUDA_ARCH__) */

#undef __SM_61_INTRINSICS_DECL__

#endif
__forceinline__ __device__ unsigned int
cast_smem_ptr_to_int(const void* const smem_ptr)
{
  unsigned int smem_int;
  asm volatile ("{ .reg .u64 smem_int; cvta.to.shared.u64 smem_int, %1; cvt.u32.u64 %0, smem_int; }"
    : "=r"(smem_int) : "l"(smem_ptr));
  return smem_int;
}

#if (((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || \
     (__CUDACC_VER_MAJOR__ > 11))
#define TVM_ENABLE_L2_PREFETCH 1
#else
#define TVM_ENABLE_L2_PREFETCH 0
#endif

#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
  #define int64_t long long
  #define uint64_t unsigned long long
#endif

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 800)
#define TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST 1
#else
#define TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST 0
#endif
extern "C" __global__ void __launch_bounds__(128) matmul_m4096n4096k4096_i8xi8_tcx128x64x64w64x32_kernel(signed char* __restrict__ A, signed char* __restrict__ B, int* __restrict__ C);
extern "C" __global__ void __launch_bounds__(128) matmul_m4096n4096k4096_i8xi8_tcx128x64x64w64x32_kernel(signed char* __restrict__ A, signed char* __restrict__ B, int* __restrict__ C) {
  extern __shared__ uchar buf_dyn_shmem[];
  int C_reindex_shared_dyn_warp[64];
  signed char A_reindex_reindex_shared_dyn_warp[64];
  signed char B_reindex_reindex_shared_dyn_warp[32];
  for (int var = 0; var < 1; ++var) {

    const dim3 blockIdx = rasterization2DColumn(11);
    for (int ax1_0_3_init = 0; ax1_0_3_init < 4; ++ax1_0_3_init) {
      for (int ax2_0_3_init = 0; ax2_0_3_init < 2; ++ax2_0_3_init) {
        for (int i = 0; i < 8; ++i) {
C_reindex_shared_dyn_warp[((ax1_0_3_init * 16) + (ax2_0_3_init * 8)) + i] = 0.0;}
;
      }
    }
    for (int ax3_0_0 = 0; ax3_0_0 < 64; ++ax3_0_0) {
      __syncthreads();
      #pragma unroll
      for (int ax0_ax1_ax2_ax3_ax4_fused_2 = 0; ax0_ax1_ax2_ax3_ax4_fused_2 < 4; ++ax0_ax1_ax2_ax3_ax4_fused_2) {
        *(int4*)(((signed char*)buf_dyn_shmem) + (((((((int)threadIdx.y) * 4096) + (((int)threadIdx.z) * 2048)) + (ax0_ax1_ax2_ax3_ax4_fused_2 * 512)) + (((int)threadIdx.x) * 16)) + 4096)) = *(int4*)(A + (((((((((int)blockIdx.y) * 524288) + (((int)threadIdx.y) * 262144)) + (((int)threadIdx.z) * 131072)) + ((ax0_ax1_ax2_ax3_ax4_fused_2 >> 1) * 65536)) + (ax3_0_0 * 1024)) + ((ax0_ax1_ax2_ax3_ax4_fused_2 & 1) * 512)) + (((int)threadIdx.x) * 16)));
      }
      #pragma unroll
      for (int ax0_ax1_ax2_ax3_ax4_fused_2_1 = 0; ax0_ax1_ax2_ax3_ax4_fused_2_1 < 2; ++ax0_ax1_ax2_ax3_ax4_fused_2_1) {
        *(int4*)(((signed char*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 2048) + (((int)threadIdx.z) * 1024)) + (ax0_ax1_ax2_ax3_ax4_fused_2_1 * 512)) + (((int)threadIdx.x) * 16))) = *(int4*)(B + ((((((((int)blockIdx.x) * 262144) + (((int)threadIdx.y) * 131072)) + (((int)threadIdx.z) * 65536)) + (ax3_0_0 * 1024)) + (ax0_ax1_ax2_ax3_ax4_fused_2_1 * 512)) + (((int)threadIdx.x) * 16)));
      }
      __syncthreads();
      for (int ax3_0_1 = 0; ax3_0_1 < 2; ++ax3_0_1) {
        for (int ax0 = 0; ax0 < 4; ++ax0) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)((&(((signed char*)buf_dyn_shmem)[((((((int)threadIdx.y) * 4096) + (ax0 * 1024)) + (ax3_0_1 * 512)) + 4096)])) + (((int)threadIdx.x) * 16))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(((signed char*)buf_dyn_shmem)[((((((int)threadIdx.y) * 4096) + (ax0 * 1024)) + (ax3_0_1 * 512)) + 4096)])) + (((int)threadIdx.x) * 16)))
    );
#endif
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_reindex_shared_dyn_warp + (ax0 * 16)))[0]), "=r"(((unsigned *)(A_reindex_reindex_shared_dyn_warp + (ax0 * 16)))[1]), "=r"(((unsigned *)(A_reindex_reindex_shared_dyn_warp + (ax0 * 16)))[2]), "=r"(((unsigned *)(A_reindex_reindex_shared_dyn_warp + (ax0 * 16)))[3])
      : "r"(addr)
    );
  }
        }
        for (int ax0_1 = 0; ax0_1 < 2; ++ax0_1) {
          *(int4*)(B_reindex_reindex_shared_dyn_warp + (ax0_1 * 16)) = *(int4*)(((signed char*)buf_dyn_shmem) + ((((((int)threadIdx.z) * 2048) + (ax0_1 * 1024)) + (ax3_0_1 * 512)) + (((int)threadIdx.x) * 16)));
        }
        for (int ax1_0_3 = 0; ax1_0_3 < 4; ++ax1_0_3) {
          for (int ax2_0_3 = 0; ax2_0_3 < 2; ++ax2_0_3) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=r"(((int *)(C_reindex_shared_dyn_warp + ((ax1_0_3 * 16) + (ax2_0_3 * 8))))[0]), "=r"(((int *)(C_reindex_shared_dyn_warp + ((ax1_0_3 * 16) + (ax2_0_3 * 8))))[1]), "=r"(((int *)(C_reindex_shared_dyn_warp + ((ax1_0_3 * 16) + (ax2_0_3 * 8))))[2]), "=r"(((int *)(C_reindex_shared_dyn_warp + ((ax1_0_3 * 16) + (ax2_0_3 * 8))))[3])
      : "r"(((unsigned *)((signed char*)A_reindex_reindex_shared_dyn_warp + (ax1_0_3 * 16)))[0]), "r"(((unsigned *)((signed char*)A_reindex_reindex_shared_dyn_warp + (ax1_0_3 * 16)))[1]), "r"(((unsigned *)((signed char*)A_reindex_reindex_shared_dyn_warp + (ax1_0_3 * 16)))[2]), "r"(((unsigned *)((signed char*)A_reindex_reindex_shared_dyn_warp + (ax1_0_3 * 16)))[3]), "r"(((unsigned *)((signed char*)B_reindex_reindex_shared_dyn_warp + (ax2_0_3 * 16)))[0]), "r"(((unsigned *)((signed char*)B_reindex_reindex_shared_dyn_warp + (ax2_0_3 * 16)))[1]), "r"(((int *)(C_reindex_shared_dyn_warp + ((ax1_0_3 * 16) + (ax2_0_3 * 8))))[0]), "r"(((int *)(C_reindex_shared_dyn_warp + ((ax1_0_3 * 16) + (ax2_0_3 * 8))))[1]), "r"(((int *)(C_reindex_shared_dyn_warp + ((ax1_0_3 * 16) + (ax2_0_3 * 8))))[2]), "r"(((int *)(C_reindex_shared_dyn_warp + ((ax1_0_3 * 16) + (ax2_0_3 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=r"(((int *)(C_reindex_shared_dyn_warp + (((ax1_0_3 * 16) + (ax2_0_3 * 8)) + 4)))[0]), "=r"(((int *)(C_reindex_shared_dyn_warp + (((ax1_0_3 * 16) + (ax2_0_3 * 8)) + 4)))[1]), "=r"(((int *)(C_reindex_shared_dyn_warp + (((ax1_0_3 * 16) + (ax2_0_3 * 8)) + 4)))[2]), "=r"(((int *)(C_reindex_shared_dyn_warp + (((ax1_0_3 * 16) + (ax2_0_3 * 8)) + 4)))[3])
      : "r"(((unsigned *)((signed char*)A_reindex_reindex_shared_dyn_warp + (ax1_0_3 * 16)))[0]), "r"(((unsigned *)((signed char*)A_reindex_reindex_shared_dyn_warp + (ax1_0_3 * 16)))[1]), "r"(((unsigned *)((signed char*)A_reindex_reindex_shared_dyn_warp + (ax1_0_3 * 16)))[2]), "r"(((unsigned *)((signed char*)A_reindex_reindex_shared_dyn_warp + (ax1_0_3 * 16)))[3]), "r"(((unsigned *)((signed char*)B_reindex_reindex_shared_dyn_warp + ((ax2_0_3 * 16) + 8)))[0]), "r"(((unsigned *)((signed char*)B_reindex_reindex_shared_dyn_warp + ((ax2_0_3 * 16) + 8)))[1]), "r"(((int *)(C_reindex_shared_dyn_warp + (((ax1_0_3 * 16) + (ax2_0_3 * 8)) + 4)))[0]), "r"(((int *)(C_reindex_shared_dyn_warp + (((ax1_0_3 * 16) + (ax2_0_3 * 8)) + 4)))[1]), "r"(((int *)(C_reindex_shared_dyn_warp + (((ax1_0_3 * 16) + (ax2_0_3 * 8)) + 4)))[2]), "r"(((int *)(C_reindex_shared_dyn_warp + (((ax1_0_3 * 16) + (ax2_0_3 * 8)) + 4)))[3]));
  }
          }
        }
      }
    }
    __syncthreads();
    for (int ax0_0 = 0; ax0_0 < 4; ++ax0_0) {
      for (int ax1_0 = 0; ax1_0 < 2; ++ax1_0) {
        for (int local_id = 0; local_id < 8; ++local_id) {
(&(((int*)buf_dyn_shmem)[(((((((int)threadIdx.y) * 4096) + (ax0_0 * 1024)) + (((int)threadIdx.z) * 32)) + (ax1_0 * 16)) + 3072)]))[((((((local_id % 4) / 2) * 8) + (threadIdx.x / 4)) * 64) + ((((local_id / 4) * 8) + ((threadIdx.x % 4) * 2)) + (local_id % 2)))] = C_reindex_shared_dyn_warp[((ax0_0 * 16) + (ax1_0 * 8)) + local_id];
}
;
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int ax0_ax1_ax2_fused_0 = 0; ax0_ax1_ax2_fused_0 < 32; ++ax0_ax1_ax2_fused_0) {
    *(int4*)(C + ((((((((int)blockIdx.y) * 524288) + (((int)threadIdx.y) * 262144)) + (ax0_ax1_ax2_fused_0 * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 15) * 4))) = *(int4*)(((int*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 4096) + (ax0_ax1_ax2_fused_0 * 128)) + (((int)threadIdx.x) * 4)) + 3072));
  }
}

extern "C" void init() {

    cudaFuncSetAttribute(matmul_m4096n4096k4096_i8xi8_tcx128x64x64w64x32_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 45056);

}

extern "C" void call(int8_t* __restrict__ A, int8_t* __restrict__ B, int* __restrict__ C, cudaStream_t stream=cudaStreamDefault) {
    matmul_m4096n4096k4096_i8xi8_tcx128x64x64w64x32_kernel<<<dim3(64, 32, 1), dim3(32, 2, 2), 45056, stream>>>(A, B, C);
}