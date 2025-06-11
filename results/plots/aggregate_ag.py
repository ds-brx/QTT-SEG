import os
import pandas as pd

# Directory containing result CSVs
results_dir = "benchmarks/ag_results"

# Datasets you want to include
target_datasets = {"polyp", "cardiac", "covid", "lesion", "leaf", "chest", "eyes", "fiber"}

# Store all data across files
all_data = []

# Read all matching CSV files
for fname in os.listdir(results_dir):
    if not fname.endswith(".csv"):
        continue

    file_path = os.path.join(results_dir, fname)
    df = pd.read_csv(file_path)

    # Optional: check if required columns exist
    if not {"dataset_name", "time_budget", "score"}.issubset(df.columns):
        print(f"Skipping {fname} — missing required columns.")
        continue

    # Filter to only relevant datasets
    df = df[df["dataset_name"].isin(target_datasets)]

    all_data.append(df)

# Concatenate all filtered data
if all_data:
    full_df = pd.concat(all_data, ignore_index=True)

    # Group and aggregate
    summary = (
        full_df.groupby(["dataset_name", "time_budget"])["score"]
        .agg(mean_score="mean", std_score="std", count="count")
        .reset_index()
        .sort_values(["dataset_name", "time_budget"])
    )

    # Save and display
    summary.to_csv("aggregated_scores_ag.csv", index=False)
    print(summary)
else:
    print("No data matched the specified datasets.")
