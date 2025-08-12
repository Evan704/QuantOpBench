#pragma once

#define GET_INT4(pointer) (*(reinterpret_cast<int4*>(pointer)))
#define GET_INT2(pointer) (*(reinterpret_cast<int2*>(pointer)))

#define CHECK_CUDA(func)                                                       \
    do {                                                                       \
        cudaError_t err = (func);                                              \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
                      << " code=" << err << " (" << cudaGetErrorString(err)     \
                      << ")" << std::endl;                                     \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)
#define CHECK_CUBLAS(func)                                                     \
    do {                                                                       \
        cublasStatus_t status = (func);                                        \
        if (status != CUBLAS_STATUS_SUCCESS) {                                 \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__     \
                      << " code=" << status << std::endl;                      \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)