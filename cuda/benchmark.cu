#include<cuda_runtime.h>
#include<vector>
#include<cstdint>
#include<random>
#include<iostream>
#include<assert.h>

#include"kernel/kernel_1.cuh"
#include"kernel/kernel_2.cuh"
#include"kernel/kernel_3.cuh"
#include"kernel/kernel_4.cuh"

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
    for (int i = 0; i < N; ++i) {
        if (gpu_C[i] != cpu_C[i]) {
            std::cerr << "Verification FAILED at index " << i << "!" << std::endl;
            std::cerr << "GPU result: " << gpu_C[i] << ", CPU reference: " << cpu_C[i] << std::endl;
            return false;
        }
    }

    std::cout << "Verification PASSED!" << std::endl;
    return true;
}

void run_kernel(int num, int8_t* A, int8_t* B, int* C) {
    switch(num) {
        case 1:
            run_kernel_1(M, N, K, A, B, C);
            break;
        case 2:
            run_kernel_2(M, N, K, A, B, C);
            break;
        case 3:
            run_kernel_3(M, N, K, A, B, C);
            break;
        case 4:
            run_kernel_4(M, N, K, A, B, C);
            break;
    }
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

    
    const int num_runs = 100;
    const int warm_up = 10;


    for(int kernel = 1; kernel <= 4; kernel++) {
        std::cout << "Kernel " << kernel << ":" << std::endl;

        // test
        run_kernel(kernel, d_A, d_B, d_C);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "An error occurred during kernel execution: %s\n", cudaGetErrorString(err));
            continue;
        }

        // warm up
        for (int i = 0; i < warm_up; ++i) {
            run_kernel_1(M, N, K, d_A, d_B, d_C);
        }

        CUDA_CHECK(cudaEventRecord(start));

        for (int i = 0; i < num_runs; ++i) {
            run_kernel(kernel, d_A, d_B, d_C);
        }

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        double latency = milliseconds/num_runs;

        long long flops = (long long)M*(long long)N*(long long)K*2;
        double tflops = (double)flops/latency/1e9;
        
        std::cout << "----------------------------------------" << std::endl;
        std::cout << "Latency: " << latency << " ms" << std::endl;
        std::cout << "TFLOPS: " << tflops << std::endl;
        std::cout << "----------------------------------------" << std::endl << std::endl;
    }
    
    return 0;
}