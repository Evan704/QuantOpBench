#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                    \
do {                                                                        \
    cudaError_t err = call;                                                 \
    if (err != cudaSuccess) {                                               \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n",                        \
                cudaGetErrorString(err), __FILE__, __LINE__);               \
        exit(EXIT_FAILURE);                                                 \
    }                                                                       \
} while (0)

extern "C" void init();
extern "C" void call(int8_t* A, int8_t* B, int* C, cudaStream_t stream);

int main() {
    const int M = 4096;
    const int N = 4096;
    const int K = 4096;

    // Initialize the matrix(W8A8)
    std::vector<int8_t> h_A(M * K);
    std::vector<int8_t> h_B(K * N);
    std::vector<int> h_C(M * N);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(-128, 127);

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

    init();

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    call(d_A, d_B, d_C, 0); // warm up
    CUDA_CHECK(cudaDeviceSynchronize());

    const int num_runs = 50;

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < num_runs; ++i) {
        call(d_A, d_B, d_C, 0);
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Run times: " << num_runs << std::endl;
    std::cout << "Total time: " << milliseconds << " ms" << std::endl;
    std::cout << "Latency: " << milliseconds / num_runs << " ms" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
// nvcc -gencode=arch=compute_90,code=sm_90 h800_W8A8.cu main.cpp -o test