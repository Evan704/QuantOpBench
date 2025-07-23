import torch
import yaml
import pandas as pd
import argparse
import bitblas
from utils import get_precision, get_gpu_name, calculate_gbps, calculate_tflops
import subprocess
import os

env = os.environ.copy()

cuda_bin_path = "/home/fit/zhaijdclass/WORK/miniconda3/envs/cuda124/bin"  # nvcc 所在的目录
cuda_lib_path = "/home/fit/zhaijdclass/WORK/miniconda3/envs/cuda124/lib64" # CUDA 库所在的目录

# 修复 PATH
# 确保 CUDA 的 bin 目录在 PATH 的最前面
env['PATH'] = f"{cuda_bin_path}:{env.get('PATH', '')}"

# 修复 LD_LIBRARY_PATH
env['LD_LIBRARY_PATH'] = f"{cuda_lib_path}:{env.get('LD_LIBRARY_PATH', '')}"

# (可选) 设置 CUDA_HOME
env['CUDA_HOME'] = "/home/fit/zhaijdclass/WORK/miniconda3/envs/cuda124"

# bitblas.set_log_level("Debug")

dtype_dict = {
    "float16": torch.float16,
    "int8": torch.int8
}

# -----------------------------------------------------------------------------
# 核心测试逻辑
# -----------------------------------------------------------------------------

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

def run_benchmark_torch(m: int, n: int, k: int, W_dtype: str, A_dtype: str, out_dtype: str):
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

    del a, b

    total_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = total_time_ms / n_repeat

    return avg_time_ms

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
                if W_dtype != "float16" and op_name == "torch":
                    continue
                for layer_name, (n, k) in layers.items():
                    for m in config['M']:
                        print(f"--------- Running Test ---------")
                        print(f"Model: {model_name}, Layer: {layer_name}")
                        print(f"Shape (M, N, K): ({m}, {n}, {k})")
                        print(f"Precision (W_dtype, A_dtype, out_dtype): ({W_dtype}, {A_dtype}, {out_dtype})")
                        
                        # 执行测试
                        if op_name == "torch":
                            avg_time_ms = run_benchmark_torch(m, n, k, W_dtype, A_dtype, out_dtype)
                        else:
                            operator = get_bitblas_operator(gpu_id, m, n, k, W_dtype, A_dtype, out_dtype)

                            avg_time_ms = operator.profile_latency()

                            tflops = calculate_tflops(m, n, k, avg_time_ms)
                            gbps = calculate_gbps(m, n, k, W_dtype, A_dtype, out_dtype, avg_time_ms)
                            
                            perf_data = {
                                "M": m, "N": n, "K": k, "Precision": get_precision(W_dtype, A_dtype, out_dtype),
                                "Time_ms": round(avg_time_ms, 5),
                                "TFLOPS/TOPS": round(tflops, 3),
                                "GB/s": round(gbps, 3)
                            }

                            print("Successfully finished the test!")
                            print(f"Time_ms: {round(avg_time_ms, 5)}")
                            print(f"TFLOPS/TOPS: {round(tflops, 3)}")
                            print(f"GB/s: {round(gbps, 3)}\n")
                        
                        # 补充元数据
                        perf_data['Operator'] = op_name
                        perf_data['GPU'] = gpu_name
                        perf_data['Model'] = model_name
                        perf_data['Layer_Name'] = layer_name
                    
                        part_results.append(perf_data)

                    # 为每个模型的特定精度保存
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
                
                    file_exists = os.path.isfile(path)
                    df.to_csv(path, mode='a', header=not file_exists, index=False)
                    print(f"Results saved to {path}\n")

                    part_results = []

if __name__ == "__main__":
    main()