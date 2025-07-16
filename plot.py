import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

df = pd.read_csv('data/A800-Llama-3-8B-float16_float16_float16.csv')

df["Model-Layer"] = df["Model"] + "-" + df["Layer_Name"]

df = df[df["Precision"] == "float16_float16_float16"]
df = df.sort_values(by='M')
df['M'] = df['M'].astype(str)

plt.figure(figsize=(12, 7))
plot = sns.lineplot(
    data=df,
    x="M",
    y="TFLOPS/TOPS",
    hue="Model",
    style="Layer_Name",
    sort=False
)

plot.set_title('TFLOPS for different M')
plt.xticks(rotation=45)
plt.savefig('img/tflops-for-different-M.png', dpi=300)
print("image saved!")