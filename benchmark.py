import torch
import yaml
import pandas as pd
import argparse
from typing import Tuple, Dict, Any, List
import bitblas
import traceback

# bitblas.set_log_level("Debug")

# -----------------------------------------------------------------------------
# 辅助函数
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# 核心测试逻辑
# -----------------------------------------------------------------------------

def get_bitblas_operator(
        gpu_id: str, m_values: List[int], n: int, k: int, W_dtype: str, A_dtype: str, out_dtype: str
    ) -> bitblas.Matmul:
    """
    根据精度字符串，返回实际的 BitBLAS 算子。
    """
    
    print(f"INFO: Attempting to get BitBLAS operator for precision '{get_precision(W_dtype, A_dtype, out_dtype)}'.")
    accum_dtype = "int32" if W_dtype == "int8" and A_dtype == "int8" else "float32"
    matmul_config = bitblas.MatmulConfig(
        M=m_values,  # M dimension
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
    backend = "tir" if gpu_id == "nvidia/nvidia-h100" else "tl"
    operator = bitblas.Matmul(config=matmul_config, backend=backend)
    return operator

def run_benchmark(
    m: int, n: int, k: int, W_dtype: str, A_dtype: str, out_dtype: str, bitblas_op: bitblas.Matmul
) -> Dict[str, Any]:
    """
    为给定的配置运行基准测试。
    """
    result = {
        "M": m, "N": n, "K": k, "Precision": get_precision(W_dtype, A_dtype, out_dtype),
        "Time_ms": -1.0, "TFLOPS": -1.0, "GB/s": -1.0
    }
    
    try:
        avg_time_ms = bitblas_op.profile_latency(
            dynamic_symbolic_constraints={"m": m}
        )
        
        # 计算性能指标
        tflops = calculate_tflops(m, n, k, avg_time_ms)
        gbps = calculate_gbps(m, n, k, W_dtype, A_dtype, out_dtype, avg_time_ms)
        
        result.update({
            "Time_ms": round(avg_time_ms, 5),
            "TFLOPS/TOPS": round(tflops, 3),
            "GB/s": round(gbps, 3)
        })

        print("Successfully finished the test!")
        print(f"Time_ms: {round(avg_time_ms, 5)}")
        print(f"TFLOPS/TOPS: {round(tflops, 3)}")
        print(f"GB/s: {round(gbps, 3)}\n")
    except Exception as e:
        print(f"ERROR running benchmark for M={m}, N={n}, K={k}, Precision={get_precision(W_dtype, A_dtype, out_dtype)}: {e}")
        traceback.print_exc() 
        # 在 result 中记录错误信息
        result['Error'] = str(e)
        
    return result

# -----------------------------------------------------------------------------
# 主执行流程
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="BitBLAS Performance Benchmark Framework")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to the config YAML file."
    )
    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 检查 GPU 环境
    gpu_name = get_gpu_name()
    gpu_id = ""
    in_list = False
    print(f"Detected GPU: {gpu_name}")
    for target, id in config['target_gpus'].items():
        if target in gpu_name:
            gpu_id = id
            gpu_name = target
            in_list = True
            break
    if not in_list:
        print(f"{gpu_name} is not in the target list. Continue anyway.")
    
    # 准备存储结果
    part_results = []
    
    # 开始迭代测试
    for model_name, layers in config['models'].items():
        for (W_dtype, A_dtype, out_dtype) in config['precisions']:
            for layer_name, (n, k) in layers.items():
                m_values = config['M']
                
                # 为一系列 M 获取 BitBLAS 算子
                bitblas_op = get_bitblas_operator(gpu_id, m_values, n, k, W_dtype, A_dtype, out_dtype)
                print("Succesfully get the operator!")

                for m in m_values:
                    print(f"--------- Running Test ---------")
                    print(f"Model: {model_name}, Layer: {layer_name}")
                    print(f"Shape (M, N, K): ({m}, {n}, {k})")
                    print(f"Precision (W_dtype, A_dtype, out_dtype): ({W_dtype}, {A_dtype}, {out_dtype})")
                    
                    # 执行测试
                    perf_data = run_benchmark(m, n, k, W_dtype, A_dtype, out_dtype, bitblas_op)
                    
                    # 补充元数据
                    perf_data['GPU'] = gpu_name
                    perf_data['Model'] = model_name
                    perf_data['Layer_Name'] = layer_name
                
                    part_results.append(perf_data)

            # 为每个模型的特定精度及时保存，避免崩溃
            df = pd.DataFrame(part_results)
            # 重新排列列的顺序，使其更易读
            cols_order = [
                'GPU', 'Model', 'Precision', 'Layer_Name',
                'M', 'N', 'K', 'Time_ms', 'TFLOPS/TOPS', 'GB/s', 'Error'
            ]
            # 过滤掉不存在的列
            df_cols = [col for col in cols_order if col in df.columns]
            df = df[df_cols]

            file_name = gpu_name + '-' + model_name + '-' + get_precision(W_dtype, A_dtype, out_dtype) + ".csv"
            path = "data/" + file_name
            print(f"--------- Benchmark Results For {file_name}---------")
            print(df.to_string())
        
            df.to_csv(path, index=False)
            print(f"Results saved to {path}\n")

            part_results = []

if __name__ == "__main__":
    main()