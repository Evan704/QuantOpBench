import torch
import yaml
import pandas as pd
import argparse
from typing import Tuple, Dict, Any, List
import bitblas
import traceback
from utils import get_precision, get_gpu_name, calculate_gbps, calculate_tflops

# bitblas.set_log_level("Debug")

dtype_dict = {
    "float16": torch.float16,
    "int8": torch.int8
}

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
    if gpu_id == "nvidia/nvidia-h100":
        operator = bitblas.Matmul(config=matmul_config, backend="tir")
    else:
        operator = bitblas.Matmul(config=matmul_config, target=gpu_id)
    return operator

def run_benchmark_bitblas(
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

def run_benchmark_torch(m: int, n: int, k: int, W_dtype: str, A_dtype: str, out_dtype: str):
    result = {
        "M": m, "N": n, "K": k, "Precision": get_precision(W_dtype, A_dtype, out_dtype),
        "Time_ms": -1.0, "TFLOPS": -1.0, "GB/s": -1.0
    }

    device = torch.device("cuda")

    if W_dtype == "float16" and A_dtype == "float16":
        a = torch.randn(m, k, device=device, dtype=torch.float16)
        b = torch.randn(k, n, device=device, dtype=torch.float16)
    else:
        a = torch.randint(-10, 10, (m, k), dtype=torch.int8)
        b = torch.randint(-10, 10, (k, n), dtype=torch.int8)
        
    n_warmup = 20
    n_repeat = 100

    for _ in range(n_warmup):
        torch.matmul(a, b)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(n_repeat):
        torch.matmul(a, b)
    end_event.record()
    torch.cuda.synchronize()

    total_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = total_time_ms / n_repeat

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
    for op_name in config['operators']:
        for model_name, layers in config['models'].items():
            for (W_dtype, A_dtype, out_dtype) in config['precisions']:
                if W_dtype == "int4" and op_name == "torch":
                    continue
                for layer_name, (n, k) in layers.items():
                    m_values = config['M']
                    
                    # 为一系列 M 获取 BitBLAS 算子
                    if op_name == "bitblas":
                        operator = get_bitblas_operator(gpu_id, m_values, n, k, W_dtype, A_dtype, out_dtype)
                        print("Succesfully get the operator!")

                    for m in m_values:
                        print(f"--------- Running Test ---------")
                        print(f"Model: {model_name}, Layer: {layer_name}")
                        print(f"Shape (M, N, K): ({m}, {n}, {k})")
                        print(f"Precision (W_dtype, A_dtype, out_dtype): ({W_dtype}, {A_dtype}, {out_dtype})")
                        
                        # 执行测试
                        if op_name == "torch":
                            perf_data = run_benchmark_torch(m, n, k, W_dtype, A_dtype, out_dtype)
                        else:
                            perf_data = run_benchmark_bitblas(m, n, k, W_dtype, A_dtype, out_dtype, operator)
                        
                        # 补充元数据
                        perf_data['Operator'] = op_name
                        perf_data['GPU'] = gpu_name
                        perf_data['Model'] = model_name
                        perf_data['Layer_Name'] = layer_name
                    
                        part_results.append(perf_data)

                # 为每个模型的特定精度及时保存，避免崩溃
                df = pd.DataFrame(part_results)
                # 重新排列列的顺序，使其更易读
                cols_order = [
                    'Operator', 'GPU', 'Model', 'Precision', 'Layer_Name',
                    'M', 'N', 'K', 'Time_ms', 'TFLOPS/TOPS', 'GB/s', 'Error'
                ]
                # 过滤掉不存在的列
                df_cols = [col for col in cols_order if col in df.columns]
                df = df[df_cols]

                file_name = f"{op_name}-{gpu_name}-{model_name}-{get_precision(W_dtype, A_dtype, out_dtype)}.csv"
                path = "data/" + file_name
                print(f"--------- Benchmark Results For {file_name}---------")
                print(df.to_string())
            
                df.to_csv(path, index=False)
                print(f"Results saved to {path}\n")

                part_results = []

if __name__ == "__main__":
    main()