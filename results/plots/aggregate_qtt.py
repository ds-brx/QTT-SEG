import os
import pandas as pd
import re

# Set your directory containing result CSV files
results_dir = "qtt_multiclass_results"

# Specify the list of datasets to include
target_datasets = {'human_parsing', 'US', 'terrain', 'golf', 'cholec'}  # use a set for fast lookup

summary_rows = []

for fname in os.listdir(results_dir):
    if not fname.endswith("_results.csv"):
        continue

    match = re.match(r"(.*)_(\d+)_results\.csv", fname)
    if not match:
        print(f"Skipping unexpected filename format: {fname}")
        continue

    dataset = match.group(1)
    time_budget = int(match.group(2))

    # Skip datasets not in the target list
    if dataset not in target_datasets:
        continue

    file_path = os.path.join(results_dir, fname)
    df = pd.read_csv(file_path)

    if 'QTT_TUNING_SCORE' not in df.columns:
        print(f"Skipping {fname} — no QTT_TUNING_SCORE column.")
        continue

    mean_score = df['QTT_TUNING_SCORE'].mean()
    std_score = df['QTT_TUNING_SCORE'].std()

    summary_rows.append({
        "dataset": dataset,
        "time_budget": time_budget,
        "mean_score": mean_score,
        "std_score": std_score
    })

summary_df = pd.DataFrame(summary_rows)
summary_df = summary_df.sort_values(by=["dataset", "time_budget"])

# Save to CSV or display
summary_df.to_csv("filtered_qtt_score_summary.csv", index=False)
print(summary_df)
