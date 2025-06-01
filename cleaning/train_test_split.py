import pandas as pd
import numpy as np
import torch
import os
from torch.serialization import add_safe_globals

def create_sequence_tensors(df, state_features, context_len):
    states, prev_actions, action_targets, rtgs, timesteps, dates = [], [], [], [], [], []

    for i in range(context_len, len(df)):
        state_seq = df[state_features].iloc[i - context_len:i].values

        # Shifted action sequence for model input
        prev_action_seq = df["action"].shift(1).fillna(0).iloc[i - context_len:i].values
        action_target_seq = df["action"].iloc[i - context_len:i].values

        rtg_seq = [df["rtg"].iloc[i]] + [0] * (context_len - 1)
        timestep_seq = list(range(context_len))
        date = df["Date"].iloc[i]  # Use last date in each sequence

        states.append(state_seq)
        prev_actions.append(prev_action_seq)
        action_targets.append(action_target_seq)
        rtgs.append(rtg_seq)
        timesteps.append(timestep_seq)
        dates.append(date)

    return (
        torch.tensor(states, dtype=torch.float32),
        torch.tensor(prev_actions, dtype=torch.long),
        torch.tensor(action_targets, dtype=torch.long),
        torch.tensor(rtgs, dtype=torch.float32),
        torch.tensor(timesteps, dtype=torch.long),
        dates
    )

def prepare_sequence_dataset_by_date(
    input_csv="data/sp500_scalper_with_rtg.csv",
    split_date="2023-01-01",
    context_len=10,
    save_dir="prepared_datasets"
):
    os.makedirs(save_dir, exist_ok=True)

    # Load + sort
    df = pd.read_csv(input_csv, parse_dates=["Date"])
    df = df.sort_values("Date")

    # Drop bad rows
    df = df.dropna(subset=["rtg", "reward"])

    # Optional: clip extreme RTG values
    df["rtg"] = df["rtg"].clip(lower=-1000, upper=1000)

    # Define state features
    exclude = {"Date", "action", "reward", "rtg", "position"}
    state_features = [col for col in df.columns if col not in exclude]

    # Split
    train_df = df[df["Date"] < split_date].reset_index(drop=True)
    val_df = df[df["Date"] >= split_date].reset_index(drop=True)

    print(f"Splitting at {split_date}")
    print(f"Train: {len(train_df)} rows | Val: {len(val_df)} rows")

    # Normalize using train stats
    mean = train_df[state_features].mean()
    std = train_df[state_features].std()
    std[std == 0] = 1e-8

    train_df[state_features] = (train_df[state_features] - mean) / std
    val_df[state_features] = (val_df[state_features] - mean) / std

    # Tensorize
    train_data = create_sequence_tensors(train_df, state_features, context_len)
    val_data = create_sequence_tensors(val_df, state_features, context_len)

    # Save train.pt
    torch.save(
        {
            "states": train_data[0],
            "actions": train_data[1],          # shifted prev actions (model input)
            "action_targets": train_data[2],   # actual next actions (targets)
            "rtgs": train_data[3],
            "timesteps": train_data[4],
            "dates": train_data[5],
        },
        os.path.join(save_dir, "train.pt")
    )

    # Save val.pt
    torch.save(
        {
            "states": val_data[0],
            "actions": val_data[1],
            "action_targets": val_data[2],
            "rtgs": val_data[3],
            "timesteps": val_data[4],
            "dates": val_data[5],
        },
        os.path.join(save_dir, "val.pt")
    )

    # Save normalization stats
    np.savez(os.path.join(save_dir, "normalization_stats.npz"),
             mean=mean.values, std=std.values, columns=np.array(state_features))

    print("Dataset saved:")
    print(f" - {save_dir}/train.pt")
    print(f" - {save_dir}/val.pt")
    print(f" - {save_dir}/normalization_stats.npz")
    print(f"Features: {state_features}")


if __name__ == "__main__":
    prepare_sequence_dataset_by_date()

# Allow pandas Timestamp unpickling
add_safe_globals([pd.Timestamp])

# Now load safely
train_data = torch.load("prepared_datasets/train.pt", weights_only=False)
val_data = torch.load("prepared_datasets/val.pt", weights_only=False)

# Check the date ranges
print("Train Date Range:", min(train_data["dates"]), "to", max(train_data["dates"]))
print("Val Date Range:  ", min(val_data["dates"]), "to", max(val_data["dates"]))
