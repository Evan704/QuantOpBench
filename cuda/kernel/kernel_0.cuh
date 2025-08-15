#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <iostream>
#include"macro.h"

namespace K0 {
cublasLtHandle_t ltHandle;
cublasLtMatrixLayout_t matA_desc, matB_desc, matC_desc;
cublasLtMatmulDesc_t op_desc;
cublasOperation_t transa = CUBLAS_OP_N;
cublasOperation_t transb = CUBLAS_OP_T;
int alpha = 1;
int beta = 0;
cublasLtMatmulPreference_t preference;
size_t workspace_size = 100 * 1024 * 1024; // 100 MB workspace
void *workspace;
const int HEURISTIC_NUM = 10;
cublasLtMatmulHeuristicResult_t heuristic_result[HEURISTIC_NUM];
int best_result = -1;
int returned_results = 0;

void init(int M, int N, int K, int8_t* A, int8_t* B, int* C) {
    CHECK_CUBLAS(cublasLtCreate(&ltHandle));

    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&matA_desc, CUDA_R_8I, M, K, M));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&matB_desc, CUDA_R_8I, N, K, N));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&matC_desc, CUDA_R_32I, M, N, M));

    CHECK_CUBLAS(cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32I, CUDA_R_32I));

    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));

    CHECK_CUDA(cudaMalloc(&workspace, workspace_size));
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size)));
    
    CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(ltHandle, op_desc, matA_desc, matB_desc, matC_desc, matC_desc, preference, HEURISTIC_NUM, heuristic_result, &returned_results));

    if (returned_results == 0) {
        std::cerr << "No cuBLASLt algorithm found!" << std::endl;
        exit(EXIT_FAILURE);
    }

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    float min_milliseconds = -1, milliseconds;
    for(int i = 0; i < HEURISTIC_NUM; i++) {
        CHECK_CUDA(cudaEventRecord(start));
        CHECK_CUBLAS(cublasLtMatmul(ltHandle, op_desc, &alpha, A, matA_desc, B, matB_desc, &beta, C, matC_desc, C, matC_desc,
                                &heuristic_result[i].algo, workspace, workspace_size, 0));
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        if(min_milliseconds < 0 || milliseconds < min_milliseconds) {
            min_milliseconds = milliseconds;
            best_result = i;
        }
    }
}

void run_kernel_0(int8_t* A, int8_t* B, int* C) {
    CHECK_CUBLAS(cublasLtMatmul(ltHandle, op_desc, &alpha, A, matA_desc, B, matB_desc, &beta, C, matC_desc, C, matC_desc,
                                &heuristic_result[best_result].algo, workspace, workspace_size, 0));
}
}

using K0::run_kernel_0;