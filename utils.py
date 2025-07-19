import torch

def get_precision(W_dtype: str, A_dtype: str, out_dtype: str) -> str:
    """精度标识"""
    return W_dtype+"_"+A_dtype+"_"+out_dtype

def get_gpu_name() -> str:
    """获取当前 CUDA 设备的名称"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires a GPU.")
    return torch.cuda.get_device_name(0)

def get_bytes(dtype: str) -> float:
    if dtype == "float16":
        return 2.0
    elif dtype == "int8":
        return 1.0
    elif dtype == "int4":
        return 0.5
    elif dtype == "int32" or dtype == "float32":
        return 4.0
    else:
        raise NotImplementedError(f"Invalid dtype: {dtype}")

def calculate_tflops(m: int, n: int, k: int, avg_time_ms: float) -> float:
    """
    计算 TFLOPS 
    对于矩阵乘法, FLOPs 约为 2 * M * N * K
    """
    if avg_time_ms == 0:
        return 0.0
    flops = 2 * m * n * k
    tflops = flops / (avg_time_ms / 1000) / 1e12
    return tflops

def calculate_gbps(m: int, n: int, k: int, W_dtype: str, A_dtype: str, out_dtype: str, avg_time_ms: float) -> float:
    """
    计算内存带宽 (GB/s)。
    带宽 = (读取A + 读取W + 写入O) / 时间
    """
    if avg_time_ms == 0:
        return 0.0
    w_bytes = get_bytes(W_dtype)
    a_bytes = get_bytes(A_dtype)
    o_bytes = get_bytes(out_dtype)
    
    total_bytes = (m * k * a_bytes) + (k * n * w_bytes) + (m * n * o_bytes)
    gbps = total_bytes / (avg_time_ms / 1000) / 1e9
    return gbps