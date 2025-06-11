import pandas as pd

# Load the CSV
df = pd.read_csv("zero_shot_results_binary.csv", header=0)

# Group by dataset and compute mean and std
summary = df.groupby("DATASET")["ZERO-SHOT"].agg(["mean", "std"]).reset_index()

# Optionally round for cleaner output
summary = summary.round(6)

# Save to file
summary.to_csv("zero_shot_score_summary.csv", index=False)

# Print summary
print(summary)