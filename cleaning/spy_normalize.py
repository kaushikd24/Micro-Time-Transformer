import pandas as pd
import numpy as np

# Load full dataset and normalization stats
df = pd.read_csv("data/sp500_scalper_with_rtg.csv", parse_dates=["Date"])
df = df.sort_values("Date")

# Load train mean/std
stats = np.load("prepared_datasets/normalization_stats.npz")
mean = stats["mean"]
std = stats["std"]
state_features = stats["columns"]

# Filter to dates AFTER training split
df = df[df["Date"] >= "2023-01-01"].reset_index(drop=True)

# Normalize using train stats
df_norm = df.copy()
df_norm[state_features] = (df_norm[state_features] - mean) / std

# Save to CSV
df_norm.to_csv("data/normalized_sp500_post2023.csv", index=False)

print("Saved: normalized_sp500_post2023.csv")
