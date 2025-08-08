#include<cuda_runtime.h>
#include<vector>
#include<cstdint>
#include<cstdio>
#include<iostream>
#include<random>
#include<iomanip>
#include<assert.h>

constexpr int M = 16;
constexpr int N = 8;
constexpr int K = 32;

__global__ void row_major_test(int8_t* A) {
    __shared__ int8_t sh[16][32];
    for(int i = 0; i < 16; i++) {
        sh[i][threadIdx.x] = A[i*32+threadIdx.x];
    }
    u_int32_t A_frag[4];
    __asm__ __volatile__(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(A_frag[0]), "=r"(A_frag[1]), "=r"(A_frag[2]), "=r"(A_frag[3])
        : "l"(__cvta_generic_to_shared((void*)((uintptr_t)sh+(threadIdx.x%16)*32+(threadIdx.x/16)*16)))
        : "memory"
    );
    if(threadIdx.x == 0) {
        int8_t* data = reinterpret_cast<int8_t*>(&A_frag);
        for(int i = 0; i < 16; i++) {
            printf("Lane %d, Data %d: %d\n", threadIdx.x, i, data[i]);
        }
    }
}

__global__ void col_major_test(int8_t* B) {
    __shared__ int8_t sh[8][32];
    for(int i = 0; i < 8; i++) {
        sh[i][threadIdx.x] = B[i*32+threadIdx.x];
    }
    u_int32_t B_frag[2];
    __asm__ __volatile__(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
        : "=r"(B_frag[0]), "=r"(B_frag[1])
        : "l"(__cvta_generic_to_shared((void*)((uintptr_t)sh+(threadIdx.x%8)*32+((threadIdx.x/8)%2)*16)))
        : "memory"
    );
    // if(threadIdx.x == 0) {
        int8_t* data = reinterpret_cast<int8_t*>(&B_frag);
        for(int i = 0; i < 8; i++) {
            printf("Lane %d, Data %d: %d\n", threadIdx.x, i, data[i]);
        }
    // }
}

__global__ void mma_test(int8_t* A, int8_t* B, int* C) {
    __shared__ int8_t sh_A[16][32];
    for(int i = 0; i < 16; i++) {
        sh_A[i][threadIdx.x] = A[i*32+threadIdx.x];
    }
    u_int32_t A_frag[4];
    __asm__ __volatile__(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(A_frag[0]), "=r"(A_frag[1]), "=r"(A_frag[2]), "=r"(A_frag[3])
        : "l"(__cvta_generic_to_shared((void*)((uintptr_t)sh_A+(threadIdx.x%16)*32+(threadIdx.x/16)*16)))
    );

    __shared__ int8_t sh_B[8][32];
    for(int i = 0; i < 8; i++) {
        sh_B[i][threadIdx.x] = B[i*32+threadIdx.x];
    }
    u_int32_t B_frag[2];
    __asm__ __volatile__(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
        : "=r"(B_frag[0]), "=r"(B_frag[1])
        : "l"(__cvta_generic_to_shared((void*)((uintptr_t)sh_B+(threadIdx.x%8)*32+((threadIdx.x/8)%2)*16)))
    );

    int C_frag[4] = {0};
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
    
    int *C_ptr = C+threadIdx.x*2;
    *(reinterpret_cast<int2*>(C_ptr)) = make_int2(C_frag[0], C_frag[1]);
    *(reinterpret_cast<int2*>(C_ptr+64)) = make_int2(C_frag[2], C_frag[3]);
}

__global__ void r2g_test(int* C) {
    int C_frag[4];
    for(int i = 0; i < 2; i++) {
        C_frag[i] = threadIdx.x*2+i;
    }
    for(int i = 2; i < 4; i++) {
        C_frag[i] = threadIdx.x*2+i+62;
    }
    int *C_ptr = C+threadIdx.x*2;
    *(reinterpret_cast<int2*>(C_ptr)) = make_int2(C_frag[0], C_frag[1]);
    *(reinterpret_cast<int2*>(C_ptr+64)) = make_int2(C_frag[2], C_frag[3]);
}

void gemm_cpu_reference(const int8_t* A, const int8_t* B, int32_t* C) {
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

bool verify_result(const int32_t* gpu_C, const int32_t* cpu_C) {
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
    std::vector<int8_t> h_A(16*32);
    std::vector<int8_t> h_B(8*32);
    std::vector<int> h_C(16*8);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(-8, 7);
    for(int i = 0; i < h_A.size(); i++) h_A[i] = i%128;
    for(int i = 0; i < h_B.size(); i++) h_B[i] = i%128;
    // for(int i = 0; i < h_A.size(); i++) h_A[i] = static_cast<int8_t>(distrib(gen));
    // for(int i = 0; i < h_B.size(); i++) h_B[i] = static_cast<int8_t>(distrib(gen));
    int8_t *d_A, *d_B;
    int *d_C;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    int run_times = 1000;
    printf("Ready to start!\n");

    cudaMalloc(&d_A, h_A.size()*sizeof(int8_t));
    cudaMemcpy(d_A, h_A.data(), h_A.size()*sizeof(int8_t), cudaMemcpyHostToDevice);

    // Test row major
    // printf("Testing row major!\n");
    // cudaEventRecord(start);
    // for(int i = 0; i < run_times; i++) {
    //     row_major_test<<<1, 32>>>(d_A);
    // }
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // std::cout << "Latency: " << milliseconds / run_times << " ms" << std::endl;

    cudaMalloc(&d_B, h_B.size()*sizeof(int8_t));
    cudaMemcpy(d_B, h_B.data(), h_B.size()*sizeof(int8_t), cudaMemcpyHostToDevice);

    // Test col major
    // printf("Testing col major!\n");
    // col_major_test<<<1, 32>>>(d_B);
    // cudaDeviceSynchronize();
    // cudaEventRecord(start);
    // for(int i = 0; i < run_times; i++) {
    //     col_major_test<<<1, 32>>>(d_B);
    // }
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // std::cout << "Latency: " << milliseconds / run_times << " ms" << std::endl;

    cudaMalloc(&d_C, h_C.size() * sizeof(int));

    // Test r2g
    // printf("Testing r2g!\n");
    // r2g_test<<<1, 32>>>(d_C);
    // cudaMemcpy(h_C.data(), d_C, M * N * sizeof(int), cudaMemcpyDeviceToHost);
    // for(int i = 0; i < 16; i++) {
    //     for(int j = 0; j < 8; j++) {
    //         std::cout << h_C[i*8+j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // Test mma
    printf("Testing mma!\n");
    mma_test<<<1, 32>>>(d_A, d_B, d_C);
    cudaMemcpy(h_C.data(), d_C, M * N * sizeof(int), cudaMemcpyDeviceToHost);
    std::vector<int> h_C_cpu(M * N);
    gemm_cpu_reference(h_A.data(), h_B.data(), h_C_cpu.data());
    // std::cout << "GPU Result:" << std::endl;
    // for(int i = 0; i < 16; i++) {
    //     for(int j = 0; j < 8; j++) {
    //         std::cout << std::setw(4) << h_C[i*8+j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << "CPU Result:" << std::endl;
    // for(int i = 0; i < 16; i++) {
    //     for(int j = 0; j < 8; j++) {
    //         std::cout << std::setw(4) << h_C_cpu[i*8+j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    assert(verify_result(h_C.data(), h_C_cpu.data()));

    cudaEventRecord(start);
    for(int i = 0; i < run_times; i++) {
        mma_test<<<1, 32>>>(d_A, d_B, d_C);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Latency: " << milliseconds / run_times << " ms" << std::endl;
}