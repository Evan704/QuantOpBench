#pragma once
#include<cstdint>

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

template<const int ScaleD>
__device__ void wgmma_m64n224k32(int* d, int8_t* As, int8_t* Bs) {
    uint64_t desc_A = make_smem_desc(As);
    uint64_t desc_B = make_smem_desc(Bs);
    __asm__ __volatile__(
        "wgmma.mma_async.sync.aligned.m64n224k32.s32.s8.s8 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95,  "
        " %96,  %97,  %98,  %99,  %100, %101, %102, %103,  "
        " %104, %105, %106, %107, %108, %109, %110, %111},  "
        " %112,"
        " %113,"
        " %114;"
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
        "+r"(d[60]), "+r"(d[61]), "+r"(d[62]), "+r"(d[63]),
        "+r"(d[64]), "+r"(d[65]), "+r"(d[66]), "+r"(d[67]), 
        "+r"(d[68]), "+r"(d[69]), "+r"(d[70]), "+r"(d[71]), 
        "+r"(d[72]), "+r"(d[73]), "+r"(d[74]), "+r"(d[75]), 
        "+r"(d[76]), "+r"(d[77]), "+r"(d[78]), "+r"(d[79]), 
        "+r"(d[80]), "+r"(d[81]), "+r"(d[82]), "+r"(d[83]), 
        "+r"(d[84]), "+r"(d[85]), "+r"(d[86]), "+r"(d[87]), 
        "+r"(d[88]), "+r"(d[89]), "+r"(d[90]), "+r"(d[91]), 
        "+r"(d[92]), "+r"(d[93]), "+r"(d[94]), "+r"(d[95]),
        "+r"(d[96]), "+r"(d[97]), "+r"(d[98]), "+r"(d[99]), 
        "+r"(d[100]), "+r"(d[101]), "+r"(d[102]), "+r"(d[103]), 
        "+r"(d[104]), "+r"(d[105]), "+r"(d[106]), "+r"(d[107]),
        "+r"(d[108]), "+r"(d[109]), "+r"(d[110]), "+r"(d[111])
        : "l"(desc_A), "l"(desc_B), "n"((int32_t)(ScaleD))
    );
}

template<const int WGMMA_N, const int ScaleD>
__device__ void wgmma(int* d, int8_t* As, int8_t* Bs) {
    static_assert(WGMMA_N == 64 || WGMMA_N == 128 || WGMMA_N == 224);
    if constexpr(WGMMA_N == 64) {
        wgmma_m64n64k32<ScaleD>(d, As, Bs);
    }
    else if constexpr(WGMMA_N == 128) {
        wgmma_m64n128k32<ScaleD>(d, As, Bs);
    }
    else if constexpr(WGMMA_N == 224) {
        wgmma_m64n224k32<ScaleD>(d, As, Bs);
    }
}

template<const int BH, const int BW>
void create_tensor_map(CUtensorMap* tma_map, int8_t* src, int height, int width) {
    uint64_t globalDim[5] = {(uint64_t)width, (uint64_t)height, 1, 1, 1};
    uint64_t globalStride[5] = {sizeof(int8_t), sizeof(int8_t)*width, 0, 0, 0};
    uint32_t boxDim[5] = {(uint32_t)BW, (uint32_t)BH, 1, 1, 1};
    uint32_t boxStride[5] = {1, 1, 1, 1, 1};
    CUresult result = cuTensorMapEncodeTiled(
        tma_map,
        CU_TENSOR_MAP_DATA_TYPE_UINT8,
        5, (void*)src, globalDim, globalStride+1, boxDim, boxStride,
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