import bitblas
import sys

# bitblas.set_log_level("ERROR")

import os

print("--- 子进程收到的环境变量 ---")
print(f"PATH: {os.getenv('PATH')}")
print(f"LD_LIBRARY_PATH: {os.getenv('LD_LIBRARY_PATH')}")
print(f"CUDA_HOME: {os.getenv('CUDA_HOME')}")
print("--------------------------\n")

def get_bitblas_operator(
        gpu_id: str, m: int, n: int, k: int, W_dtype: str, A_dtype: str, out_dtype: str
    ) -> bitblas.Matmul:
    """
    根据精度字符串，返回实际的 BitBLAS 算子。
    """
    
    accum_dtype = "int32" if W_dtype == "int8" and A_dtype == "int8" else "float16"
    matmul_config = bitblas.MatmulConfig(
        M=m,  # M dimension
        N=n,  # N dimension
        K=k,  # K dimension
        A_dtype=A_dtype,  # activation A dtype
        W_dtype=W_dtype,  # weight W dtype
        accum_dtype=accum_dtype,  # accumulation dtype
        out_dtype=out_dtype,  # output dtype
        layout="nt",  # matrix layout, "nt" indicates the layout of A is non-transpose and the layout of W is transpose
        with_bias=False,  # bias
        # configs for weight only quantization
        group_size=None,  # setting for grouped quantization
        with_scaling=False,  # setting for scaling factor
        with_zeros=False,  # setting for zeros
        zeros_mode=None,  # setting for how to calculating zeros
    )
    if gpu_id == "nvidia/nvidia-h100":
        operator = bitblas.Matmul(config=matmul_config, backend="tir")
    else:
        operator = bitblas.Matmul(config=matmul_config, target=gpu_id)
    return operator

if __name__ == "__main__":
    gpu_id, m, n, k, W_dtype, A_dtype, out_dtype = sys.argv[1:]
    m, n, k = [int(m), int(n), int(k)]

    operator = get_bitblas_operator(gpu_id, m, n, k, W_dtype, A_dtype, out_dtype)

    avg_time_ms = operator.profile_latency()
    
    print(avg_time_ms)