#include<cuda_runtime.h>
#include<vector>
#include<cstdint>
#include<random>
#include<iostream>
#include<assert.h>
#include<cublas_v2.h>
#include<iomanip>

#include"kernel/macro.h"
#include"kernel/kernel_0.cuh"
#include"kernel/kernel_1.cuh"
#include"kernel/kernel_2.cuh"
#include"kernel/kernel_3.cuh"
#include"kernel/kernel_4.cuh"
#include"kernel/kernel_5.cuh"
#include"kernel/kernel_6.cuh"
#include"kernel/kernel_7.cuh"
#include"kernel/kernel_8.cuh"
#include"kernel/kernel_9.cuh"

// constexpr int M = 4;
// constexpr int N = 4;
// constexpr int K = 4;
constexpr int M = 4096;
constexpr int N = 4096;
constexpr int K = 4096;

void gemm_cpu(int8_t* A, int8_t* B, int* C) {
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            C[i*N+j] = 0;
            for(int k = 0; k < K; k++) {
                C[i*N+j] += (int)A[i*K+k]*(int)B[j*K+k];
            }
        }
    }
}

bool verify_result(const int* C, const int* C_ref, int M, int N) {
    for (int i = 0; i < N; ++i) {
        if (C[i] != C_ref[i]) {
            std::cout << "Verification FAILED at index " << i << "!" << std::endl;
            std::cout << "result: " << C[i] << ", reference: " << C_ref[i] << std::endl;
            return false;
        }
    }

    std::cout << "Verification PASSED!" << std::endl << std::endl;
    return true;
}

cublasHandle_t cublas_handle;
void runCublasGemm(int M, int N, int K, int8_t *A, int8_t *B, int *C) {
  int alpha = 1, beta = 0;
  // C(column major) = A(row major) * B(column major)
  cublasStatus_t status = cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_8I,
    K, A, CUDA_R_8I, K, &beta, C, CUDA_R_32I, N, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);

  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "CUBLAS error: " << status << std::endl;
    exit(1);
  }
}

void run_kernel(int num, int8_t* A, int8_t* B, int* C, bool dbg) {
    switch(num) {
        case 0:
            runCublasGemm(M, N, K, A, B, C);
            // run_kernel_0(A, B, C);
            break;
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
        case 5:
            run_kernel_5(M, N, K, A, B, C);
            break;
        case 6:
            if(dbg) {
                int dbg[M/128*N/128*6];
                memset(dbg, 0, sizeof(int)*M/128*N/128*6);
                int* d_dbg;
                CHECK_CUDA(cudaMalloc(&d_dbg, M/128*N/128*6*sizeof(int)));
                CHECK_CUDA(cudaMemcpy(d_dbg, dbg, M/128*N/128*6*sizeof(int), cudaMemcpyHostToDevice));
                run_kernel_6(M, N, K, A, B, C, d_dbg);
                CHECK_CUDA(cudaMemcpy(dbg, d_dbg, M/128*N/128*6*sizeof(int), cudaMemcpyDeviceToHost));
                int sumLoad = 0, cntLoad = 0;
                int sumCompute = 0, cntCompute = 0;
                int sumStore = 0, cntStore = 0;
                for(int i = 0; i < M/128*N/128*6; i += 6) {
                    sumLoad += dbg[i]; cntLoad += dbg[i+1];
                    sumCompute += dbg[i+2]; cntCompute += dbg[i+3];
                    sumStore += dbg[i+4]; cntStore += dbg[i+5];
                }
                printf("Load: %.2f\nCompute: %.2f\nStore: %.2f\n", (float)sumLoad/cntLoad, (float)sumCompute/cntCompute, (float)sumStore/cntStore);
            }
            else run_kernel_6(M, N, K, A, B, C, nullptr);
            break;
        case 7:
            run_kernel_7(M, N, K, A, B, C);
            break;
        case 8:
            run_kernel_8(M, N, K, A, B, C);
            break;
        case 9:
            run_kernel_9(M, N, K, A, B, C);
            break;
    }
}

int main() {
    cublasCreate_v2(&cublas_handle);

    // Initialize the matrix(W8A8)
    // A is row major, B is col major
    std::vector<int8_t> h_A(M * K);
    std::vector<int8_t> h_B(K * N);
    std::vector<int> h_C(M * N), h_C_ref(M*N);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(-8, 7);

    for (size_t i = 0; i < h_A.size(); ++i) h_A[i] = static_cast<int8_t>(distrib(gen));
    for (size_t i = 0; i < h_B.size(); ++i) h_B[i] = static_cast<int8_t>(distrib(gen));

    int8_t *d_A, *d_B;
    int *d_C;

    CHECK_CUDA(cudaMalloc(&d_A, h_A.size() * sizeof(int8_t)));
    CHECK_CUDA(cudaMalloc(&d_B, h_B.size() * sizeof(int8_t)));
    CHECK_CUDA(cudaMalloc(&d_C, h_C.size() * sizeof(int)));

    // Copy the data to GPU
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(int8_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(int8_t), cudaMemcpyHostToDevice));

    // K0::init(M, N, K, d_A, d_B, d_C);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    const int num_runs = 100;
    const int warm_up = 10;

    // run_kernel(0, d_A, d_B, d_C, false);
    runCublasGemm(M, N, K, d_A, d_B, d_C);
    CHECK_CUDA(cudaMemcpy(h_C_ref.data(), d_C, h_C_ref.size()*sizeof(int), cudaMemcpyDeviceToHost));

    double cublas_tops;

    for(int kernel = 0; kernel <= 9; kernel++) {
        std::cout << "Kernel " << kernel << ":" << std::endl;

        // test
        if(kernel > 0) {
            for(int i = 0; i < M*N; i++) h_C[i] = 0;
            CHECK_CUDA(cudaMemcpy(d_C, h_C.data(), h_C.size()*sizeof(int), cudaMemcpyHostToDevice));
            run_kernel(kernel, d_A, d_B, d_C, true);
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, h_C.size()*sizeof(int), cudaMemcpyDeviceToHost));
            if(!verify_result(h_C.data(), h_C_ref.data(), M, N)) continue;
        }

        // warm up
        for (int i = 0; i < warm_up; ++i) {
            run_kernel(0, d_A, d_B, d_C, false);
        }

        CHECK_CUDA(cudaEventRecord(start));

        for (int i = 0; i < num_runs; ++i) {
            run_kernel(kernel, d_A, d_B, d_C, false);
        }

        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float milliseconds = 0;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        double latency = milliseconds/num_runs;

        long long ops = (long long)M*(long long)N*(long long)K*2;
        double tops = (double)ops/latency/1e9;

        if(kernel == 0) cublas_tops = tops;
        
        std::cout << "----------------------------------------" << std::endl;
        std::cout << "Latency: " << latency << " ms" << std::endl;
        std::cout << "TOPS: " << tops << std::endl;
        printf("Performance: %.2f%% cuBLAS\n", (double)tops/cublas_tops*100);
        std::cout << "----------------------------------------" << std::endl << std::endl;
    }
    
    return 0;
}