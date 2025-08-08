#include<cuda_runtime.h>
#include<vector>
#include<cstdint>
#include<random>
#include<iostream>
#include<assert.h>

#define M_TILE 16
#define N_TILE 8
#define K_TILE 32
#define THREADS_PER_BLOCK 32

#define CUDA_CHECK(call)                                         \
    do {                                                         \
        cudaError_t err = call;                                  \
        if (err != cudaSuccess) {                                \
            fprintf(stderr, "CUDA error at %s %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    } while (0)

constexpr int M = 4096;
constexpr int N = 4096;
constexpr int K = 4096;

// mma m16n8k32
__global__ void gemm_mma(int8_t* A, int8_t* B, int* C) {
    __shared__ __align__(128) int8_t sh_A[M_TILE][K_TILE];
    __shared__ __align__(128) int8_t sh_B[N_TILE][K_TILE];

    // 每个寄存器含4个int8元素, A有16*32=512个元素, 需要512/4/32=4个寄存器, B同理
    u_int A_frag[4], B_frag[2];
    int C_frag[4] = {0};

    int8_t* A_tile_ptr = A+blockIdx.y*M_TILE*K;
    int8_t* B_tile_ptr = B+blockIdx.x*N_TILE*K;
    
    for(int k = 0; k < K; k += K_TILE) {
        // Gmem to Smem
        // 16*32/32=16, 每个线程加载16个A元素
        for(int i = 0; i < 16; i++) {
            int row = i, col = threadIdx.x;
            sh_A[row][col] = A_tile_ptr[row*K+k+col];
        }
        // 8*32/32=32, 每个线程加载8个B元素，假设主机中B以列主序存储
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
        
        __syncthreads();

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
    }

    int* C_ptr = C+blockIdx.y*M_TILE*N+blockIdx.x*N_TILE+(threadIdx.x%4)*2+(threadIdx.x/4)*N;
    *(reinterpret_cast<int2*>(C_ptr)) = make_int2(C_frag[0], C_frag[1]);
    *(reinterpret_cast<int2*>(C_ptr+8*N)) = make_int2(C_frag[2], C_frag[3]);
}

void gemm_cpu_reference(const int8_t* A, const int8_t* B, int* C) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            int32_t acc = 0;
            for (int k = 0; k < K; ++k) {
                acc += static_cast<int32_t>(A[m*K+k])*static_cast<int32_t>(B[n*K+k]);
            }
            C[m*N+n] = acc;
        }
    }
}

bool verify_result(const int* gpu_C, const int* cpu_C, int M, int N) {
    for (int i = 0; i < M * N; ++i) {
        if (gpu_C[i] != cpu_C[i]) {
            std::cerr << "Verification FAILED at index " << i << "!" << std::endl;
            std::cerr << "GPU result: " << gpu_C[i] << ", CPU reference: " << cpu_C[i] << std::endl;
            return false;
        }
    }
    std::cout << "Verification PASSED!" << std::endl;
    return true;
}

int main() {
    // Initialize the matrix(W8A8)
    std::vector<int8_t> h_A(M * K);
    std::vector<int8_t> h_B(K * N);
    std::vector<int> h_C(M * N);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(-8, 7);

    for (size_t i = 0; i < h_A.size(); ++i) h_A[i] = static_cast<int8_t>(distrib(gen));
    for (size_t i = 0; i < h_B.size(); ++i) h_B[i] = static_cast<int8_t>(distrib(gen));

    int8_t *d_A, *d_B;
    int *d_C;

    CUDA_CHECK(cudaMalloc(&d_A, h_A.size() * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_B, h_B.size() * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_C, h_C.size() * sizeof(int)));

    // Copy the data to GPU
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(int8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(int8_t), cudaMemcpyHostToDevice));


    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    dim3 threadsPerBlock(THREADS_PER_BLOCK);
    dim3 numBlocks(N / N_TILE, M / M_TILE);
    // gemm_mma<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C);
    // CUDA_CHECK(cudaDeviceSynchronize());
    // CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(int), cudaMemcpyDeviceToHost));
    // std::vector<int> h_C_cpu(M * N);
    // gemm_cpu_reference(h_A.data(), h_B.data(), h_C_cpu.data());
    // assert(verify_result(h_C.data(), h_C_cpu.data(), M, N));

    const int num_runs = 100;

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < num_runs; ++i) {
        gemm_mma<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C);
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Run times: " << num_runs << std::endl;
    std::cout << "Total time: " << milliseconds << " ms" << std::endl;
    std::cout << "Latency: " << milliseconds / num_runs << " ms" << std::endl;
    // std::cout << "TFLOPS: " << 2*M*N*K/milliseconds/1e12 << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}