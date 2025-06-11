import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data (assumes tab-separated values)
df = pd.read_csv("QTT-ZERO-SHOT - Binary_all.csv")  # Change to your actual file path

# Compute group statistics
summary = df.groupby("TIME_BUDGET").agg({
    "Zero-Shot": ["mean", "std"],
    "AutoGluon" : ["mean", "std"],
    "QTT_Scores": ["mean", "std"]
})

# Flatten column names
summary.columns = ['_'.join(col) for col in summary.columns]
summary.reset_index(inplace=True)

# Extract values
time_budgets = summary["TIME_BUDGET"]
x = np.arange(len(time_budgets))
bar_width = 0.25

# Means and stds
zero_means = summary["Zero-Shot_mean"]
zero_stds = summary["Zero-Shot_std"]

auto_means = summary["AutoGluon_mean"]
auto_stds = summary["AutoGluon_std"]

qtt_means = summary["QTT_Scores_mean"]
qtt_stds = summary["QTT_Scores_std"]

error_config = dict(ecolor='gray', capsize=3, linewidth=1)

plt.figure(figsize=(6, 3))

# Plot Zero-shot with error bars
plt.bar(x- bar_width, zero_means, bar_width,
        label='Zero-shot',
        color='#C7C8CC',
        yerr=zero_stds,
        **error_config)

# Plot QTT with error bars
plt.bar(x, auto_means, auto_stds,
        label='AutoGluon',
        color='#9AA6B2',
        yerr=qtt_stds,
        **error_config)

# Plot QTT with error bars
plt.bar(x + bar_width, qtt_means, bar_width,
        label='QTT-SEG',
        color='#b4d4ff',
        yerr=qtt_stds,
        **error_config)

# Customize plot
plt.xticks(x, time_budgets)
plt.xlabel('TIME BUDGET (s)')
plt.ylabel('AVG IOU')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=3)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

plt.savefig("Binary_plot_pastel.png")
plt.close()
