import torch
import yaml
import pandas as pd
import argparse
from typing import Tuple, Dict, Any, List
import bitblas
import traceback

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

def get_bitblas_operator(m: int, n: int, k: int, W_dtype: str, A_dtype: str, out_dtype: str):
    """
    根据精度字符串，返回实际的 BitBLAS 算子。
    """
    
    print(f"INFO: Attempting to get BitBLAS operator for precision '{get_precision(W_dtype, A_dtype, out_dtype)}'.")
    
    try:
        matmul_config = bitblas.MatmulConfig(
            M=m,  # M dimension
            N=n,  # N dimension
            K=k,  # K dimension
            A_dtype=A_dtype,  # activation A dtype
            W_dtype=W_dtype,  # weight W dtype
            accum_dtype="float16",  # accumulation dtype
            out_dtype=out_dtype,  # output dtype
            layout="nt",  # matrix layout, "nt" indicates the layout of A is non-transpose and the layout of W is transpose
            with_bias=False,  # bias
            # configs for weight only quantization
            group_size=None,  # setting for grouped quantization
            with_scaling=False,  # setting for scaling factor
            with_zeros=False,  # setting for zeros
            zeros_mode=None,  # setting for how to calculating zeros
        )
        operator = bitblas.Matmul(config=matmul_config, enable_tuning=False)
        if operator is None:
            raise RuntimeError(f"bitblas.Matmul return None. Failed to find a suitable kernel for "
                               f"M={m}, N={n}, K={k}, Precision={get_precision(W_dtype, A_dtype, out_dtype)}.")
        return operator
    except Exception as e:
        raise RuntimeError(f"ERROR getting bitblas operator for M={m}, N={n}, K={k}, Precision={get_precision(W_dtype, A_dtype, out_dtype)}: {e}")

def get_random_matrix(row: int, col: int, dtype: str, bitblas_op) -> torch.Tensor:
    if dtype == "float16":
        return torch.rand((row, col), dtype=torch.float16).cuda()
    elif dtype == "int8":
        return torch.randint(-8, 8, (row, col), dtype=torch.int8).cuda()
    elif dtype == "int4":
        tensor = torch.randint(-8, 8, (row, col), dtype=torch.int8).cuda()
        return bitblas_op.tranform_weight(tensor)
    else:
        raise NotImplementedError(f"Invalid dtype: {dtype}")

def run_benchmark(
    m: int, n: int, k: int, W_dtype: str, A_dtype: str, out_dtype: str, settings: Dict[str, Any]
) -> Dict[str, Any]:
    """
    为给定的配置运行单次基准测试。
    """
    result = {
        "M": m, "N": n, "K": k, "Precision": get_precision(W_dtype, A_dtype, out_dtype),
        "Time_ms": -1.0, "TFLOPS": -1.0, "GB/s": -1.0
    }
    
    try:
        # 1. 获取 BitBLAS 算子
        bitblas_op = get_bitblas_operator(m, n, k, W_dtype, A_dtype, out_dtype)

        # 2. 创建输入张量
        # bitblas.Matmul 默认 W 转置后进行矩阵乘法
        A = get_random_matrix(m, k, A_dtype, bitblas_op)
        W = get_random_matrix(n, k, W_dtype, bitblas_op)

        # 3. 预热 GPU
        for _ in range(settings['warmup_iterations']):
            _ = bitblas_op(A, W)
        torch.cuda.synchronize()

        # 4. 精确计时
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(settings['test_iterations']):
            _ = bitblas_op(A, W)
        end_event.record()
        
        torch.cuda.synchronize()
        
        avg_time_ms = start_event.elapsed_time(end_event) / settings['test_iterations']
        
        # 5. 计算性能指标
        tflops = calculate_tflops(m, n, k, avg_time_ms)
        gbps = calculate_gbps(m, n, k, W_dtype, A_dtype, out_dtype, avg_time_ms)
        
        result.update({
            "Time_ms": round(avg_time_ms, 5),
            "TFLOPS": round(tflops, 3),
            "GB/s": round(gbps, 3)
        })
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
    parser.add_argument(
        "--output", type=str, default="benchmark_results.csv", help="Path to save the results CSV file."
    )
    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 检查 GPU 环境
    gpu_name = get_gpu_name()
    print(f"Detected GPU: {gpu_name}")
    if not any(target in gpu_name for target in config['target_gpus']):
        print(f"Warning: GPU '{gpu_name}' is not in the target list. Continuing anyway.")

    # 准备存储结果
    all_results = []
    
    # 开始迭代测试
    prefill_sequence_length = config['sequence_length']
    for model_name, layers in config['models'].items():
        for (W_dtype, A_dtype, out_dtype) in config['precisions']:
            for layer_name, (n, k) in layers.items():
                for batch_size in config['batch_sizes']:
                    for is_decode in range(2):
                        sequence_length = 1 if is_decode else prefill_sequence_length
                        state = "Decode" if is_decode else "Prefill"

                        # 计算实际的 M 维度
                        m = batch_size * sequence_length
                        
                        print(f"--- Running Test ---")
                        print(f"Model: {model_name}, Layer: {layer_name}, Batch Size: {batch_size}, State: {state}")
                        print(f"Shape (M, N, K): ({m}, {n}, {k})")
                        print(f"Precision (W_dtype, A_dtype, out_dtype): ({W_dtype}, {A_dtype}, {out_dtype})")
                        
                        # 执行测试
                        perf_data = run_benchmark(m, n, k, W_dtype, A_dtype, out_dtype, config['test_settings'])
                        
                        # 补充元数据
                        perf_data['GPU'] = gpu_name
                        perf_data['Model'] = model_name
                        perf_data['Layer_Name'] = layer_name
                        perf_data['State'] = state
                        perf_data['Batch_Size'] = batch_size
                        
                        all_results.append(perf_data)
    
    # 将结果转换为 DataFrame 并保存
    if all_results:
        df = pd.DataFrame(all_results)
        # 重新排列列的顺序，使其更易读
        cols_order = [
            'GPU', 'Model', 'Precision', 'Layer_Name', 'State', 'Batch_Size',
            'M', 'N', 'K', 'Time_ms', 'TFLOPS', 'GB/s', 'Error'
        ]
        # 过滤掉不存在的列
        df_cols = [col for col in cols_order if col in df.columns]
        df = df[df_cols]

        print("\n--- Benchmark Results ---")
        print(df.to_string())
        
        # 保存到 CSV
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")
    else:
        print("No benchmarks were run.")

if __name__ == "__main__":
    main()