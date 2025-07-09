import torch
import yaml
import pandas as pd
import time
import argparse
from typing import Tuple, Dict, Any, List
import bitblas

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

def get_bytes_per_element(precision_str: str) -> Tuple[float, float, float]:
    """
    根据精度字符串估算每个矩阵元素的字节数。
    你需要根据 BitBLAS 的实际情况调整此函数。
    返回 (A_bytes, W_bytes, C_bytes)
    """
    # 这是一个示例实现，你需要确认 BitBLAS 的具体数据类型
    # FP16 = 2 bytes, INT8 = 1 byte, INT4 = 0.5 bytes
    parts = precision_str.split('_')
    # 例如 "W4A16_FP16" -> A=FP16, W=INT4, C=FP16
    precision_map = {
        'FP32': 4,
        'FP16': 2,
        'BF16': 2,
        'A16': 2, # Assuming A16 is FP16
        'W8': 1,
        'A8': 1,
        'W4': 0.5,
    }
    
    # 示例逻辑，需要你根据实际情况修改
    # 格式假定为 W{weight_prec}A{activation_prec}_{output_prec} 或 Aprec_Wprec_Cprec
    if "FP16_FP16_FP16" in precision_str:
        return 2.0, 2.0, 2.0
    elif "W4A16" in precision_str:
        return 2.0, 0.5, 2.0 # Activation=FP16, Weight=INT4, Output=FP16
    elif "W8A8" in precision_str:
        return 1.0, 1.0, 2.0 # Activation=INT8, Weight=INT8, Output=FP16
    else:
        # 默认返回一个基准值，并打印警告
        print(f"Warning: Precision '{precision_str}' not fully recognized. Using default byte sizes (2,2,2).")
        return 2.0, 2.0, 2.0

def calculate_tflops(m: int, n: int, k: int, avg_time_ms: float) -> float:
    """
    计算 TFLOPS (每秒万亿次浮点运算)。
    对于矩阵乘法，FLOPs 约为 2 * M * N * K。
    """
    if avg_time_ms == 0:
        return 0.0
    flops = 2 * m * n * k
    tflops = flops / (avg_time_ms / 1000) / 1e12
    return tflops

def calculate_gbps(m: int, n: int, k: int, precision: str, avg_time_ms: float) -> float:
    """
    计算内存带宽 (GB/s)。
    带宽 = (读取A + 读取W + 写入C) / 时间
    """
    if avg_time_ms == 0:
        return 0.0
    a_bytes, w_bytes, c_bytes = get_bytes_per_element(precision)
    
    total_bytes = (m * k * a_bytes) + (k * n * w_bytes) + (m * n * c_bytes)
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
    except Exception as e:
        raise RuntimeError(f"ERROR getting bitblas operator for M={m}, N={n}, K={k}, Precision={get_precision(W_dtype, A_dtype, out_dtype)}: {e}")
    return bitblas.Matmul(config=matmul_config)

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

        # 2. !!! TODO: 创建输入张量 !!!
        # 你需要根据 BitBLAS 的要求创建正确的 dtype 和 layout 的张量。
        # 例如，权重可能是预量化和打包的。
        # 这里的创建是示例性的。
        a_bytes, w_bytes, _ = get_bytes_per_element(W_dtype, A_dtype, out_dtype)

        # 模拟激活张量
        a_dtype = torch.float16 if a_bytes == 2 else torch.int8
        A = torch.randn(m, k, device="cuda", dtype=a_dtype)

        # 模拟权重张量 (可能是量化的)
        w_dtype = torch.float16
        if w_bytes == 1:
            w_dtype = torch.int8
        elif w_bytes == 0.5:
            # 对于 INT4，通常用 INT8 存储，需要特殊处理
            # 这是一个简化，真实情况可能更复杂
            W = torch.randint(-8, 7, (k, n), device="cuda", dtype=torch.int8)
        else:
            W = torch.randn(k, n, device="cuda", dtype=w_dtype)

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
        
        # 计算平均时间 (毫秒)
        avg_time_ms = start_event.elapsed_time(end_event) / settings['test_iterations']
        
        # 5. 计算性能指标
        tflops = calculate_tflops(m, n, k, avg_time_ms)
        gbps = calculate_gbps(m, n, k, W_dtype, A_dtype, out_dtype, avg_time_ms)
        
        result.update({
            "Time_ms": round(avg_time_ms, 5),
            "TFLOPS": round(tflops, 2),
            "GB/s": round(gbps, 2)
        })

    except Exception as e:
        print(f"ERROR running benchmark for M={m}, N={n}, K={k}, Precision={get_precision(W_dtype, A_dtype, out_dtype)}: {e}")
        # 可以在 result 中记录错误信息
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
    prefill_sequence_length = config['prefill_sequence_length']
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