import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import argparse
import inspect

precision_dict = {
    "float16_float16_float16": "W16A16",
    "int4_float16_float16": "W4A16",
    "int8_int8_int32": "W8A8"
}
COMMAND_REGISTRY = []

def get_file_name(gpu: str, model:str, precision: str) -> str:
    return f"{gpu}-{model}-{precision}"

def register_task(name: str):
    """用于将函数注册为一个命令行命令"""
    def decorator(func):
        help_text = inspect.getdoc(func) or "No help text exist."
        COMMAND_REGISTRY.append((name, func, help_text))
        return func
    return decorator

@register_task("p1")
def plot_tflops_vs_m(df: pd.DataFrame, config):
    """绘制相同GPU、相同精度条件下不同矩阵大小时的TFLOPS"""
    img_path = config['img_path']

    for gpu in config['target_gpus']:
        for precision in config['precisions']:
            part_df = df[(df["Precision"] == precision) & (df["GPU"] == gpu)]
            part_df = part_df.sort_values(by='M')
            part_df['M'] = part_df['M'].astype(str)

            plt.figure(figsize=(12, 7))
            plot = sns.lineplot(
                data=part_df,
                x="M",
                y="TFLOPS/TOPS",
                hue="Model",
                style="Layer_Name",
                markers=True
            )

            plot.set_title(fr'TFLOPS for different M on {gpu} of {precision_dict[precision]}')
            plt.xticks(rotation=45)

            img_name = f'{img_path}{gpu}-{precision_dict[precision]}-tflops-for-different-M.png'
            plt.savefig(img_name, dpi=300)
            print(f"image saved at {img_name}!")

@register_task("p2")
def plot_gbs_vs_m(df: pd.DataFrame, config):
    """绘制相同GPU、相同精度条件下不同矩阵大小时的GB/s"""
    img_path = config['img_path']

    for gpu in config['target_gpus']:
        for precision in config['precisions']:
            part_df = df[(df["Precision"] == precision) & (df["GPU"] == gpu)]
            part_df = part_df.sort_values(by='M')
            part_df['M'] = part_df['M'].astype(str)

            plt.figure(figsize=(12, 7))
            plot = sns.lineplot(
                data=part_df,
                x="M",
                y="GB/s",
                hue="Model",
                style="Layer_Name",
                markers=True
            )

            plot.set_title(fr'GB/s for different M on {gpu} of {precision_dict[precision]}')
            plt.xticks(rotation=45)

            img_name = f'{img_path}{gpu}-{precision_dict[precision]}-gbs-for-different-M.png'
            plt.savefig(img_name, dpi=300)
            print(f"image saved at {img_name}!")

@register_task("p3")
def plot_ltc_vs_m(df: pd.DataFrame, config):
    """绘制相同GPU、相同精度条件下不同矩阵大小时的Latency"""
    img_path = config['img_path']

    for gpu in config['target_gpus']:
        for precision in config['precisions']:
            part_df = df[(df["Precision"] == precision) & (df["GPU"] == gpu)]
            part_df = part_df.sort_values(by='M')
            part_df['M'] = part_df['M'].astype(str)

            plt.figure(figsize=(12, 7))
            plot = sns.lineplot(
                data=part_df,
                x="M",
                y="Time_ms",
                hue="Model",
                style="Layer_Name",
                markers=True
            )

            plot.set_title(fr'Latency for different M on {gpu} of {precision_dict[precision]}')
            plt.xticks(rotation=45)
            # 将纵轴取对数
            plot.set_yscale('log')

            img_name = f'{img_path}{gpu}-{precision_dict[precision]}-ltc-for-different-M.png'
            plt.savefig(img_name, dpi=300)
            print(f"image saved at {img_name}!")

# 在运行时需添加参数，指定需要绘图的类型
def main():
    parser = argparse.ArgumentParser(description="Data Analysis")
    parser.add_argument(
        "--config", type=str, default="plot_config.yaml", help="Path to the config YAML file."
    )
    # 注册命令
    for name, _, help_text in COMMAND_REGISTRY:
        parser.add_argument(
            f"--{name}", action="store_true", help=help_text
        )

    args = parser.parse_args()

    sns.set_theme(style="whitegrid")

    # 配置载入
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    data_path = config['data_path']

    # 导入数据
    df = pd.DataFrame()

    for gpu in config['target_gpus']:
        for model in config['models']:
            for precision in config['precisions']:
                part_df = pd.read_csv(f"{data_path}{get_file_name(gpu, model, precision)}.csv")
                df = pd.concat([df, part_df])

    # 绘图
    executed = False
    for name, func, _ in COMMAND_REGISTRY:
        if getattr(args, name):
            func(df, config)
            executed = True
    if not executed:
        print("No command executed.")

if __name__ == "__main__":
    main()