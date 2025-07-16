import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import argparse

precision_dict = {
    "float16_float16_float16": "W16A16",
    "int4_float16_float16": "W4A16",
    "int8_int8_int32": "W8A8"
}

def get_file_name(gpu: str, model:str, precision: str) -> str:
    return f"{gpu}-{model}-{precision}"

def main():
    parser = argparse.ArgumentParser(description="Data Analysis")
    parser.add_argument(
        "--config", type=str, default="plot_config.yaml", help="Path to the config YAML file."
    )
    args = parser.parse_args()

    sns.set_theme(style="whitegrid")

    # 配置载入
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    data_path = config['data_path']
    img_path = config['img_path']

    # 导入数据
    df = pd.DataFrame()

    for gpu in config['target_gpus']:
        for model in config['models']:
            for precision in config['precisions']:
                part_df = pd.read_csv(f"{data_path}{get_file_name(gpu, model, precision)}.csv")
                df = pd.concat([df, part_df])

    # 绘制相同GPU、相同精度条件下不同矩阵大小时的TFLOPS
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

    # 绘制相同GPU、相同精度条件下不同矩阵大小时的GB/s
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

if __name__ == "__main__":
    main()