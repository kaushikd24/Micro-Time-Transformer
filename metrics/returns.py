import torch
import pandas as pd
import numpy as np
from datetime import timedelta
from src.model import DecisionTransformer

# --- Model loading (assumes class already defined) ---
model = DecisionTransformer(
    state_dim=18,
    action_dim=3,
    embed_dim=256,
    context_length=10,
    num_classes=3,
    n_layers=4,
    n_heads=4
)
model.load_state_dict(torch.load("models/model_epoch_5.pt", map_location="cpu"))
model.eval()

# --- Load normalized data ---
df = pd.read_csv("data/normalized_sp500_post2023.csv", parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Your feature list (same as training)
exclude = {"Date", "action", "reward", "rtg", "position"}
state_features = [col for col in df.columns if col not in exclude]
 # <--- insert actual feature column names here

def get_state_sequence(start_date, context_len=10):
    idx = df.index[df["Date"] == pd.to_datetime(start_date)][0]
    assert idx >= context_len, "Not enough history before start date"
    state_window = df[state_features].iloc[idx - context_len:idx].astype(np.float32).values


    date_window = df["Date"].iloc[idx - context_len:idx].tolist()
    return torch.tensor(state_window, dtype=torch.float32).unsqueeze(0), idx, date_window


def run_extended_goal_rollout(start_date, target_return=0.05, horizon=150):
    start_idx = df.index[df["Date"] == pd.to_datetime(start_date)][0]
    assert start_idx >= 10, "Need 10 days of context before start_date"

    rtg_remaining = target_return
    achieved = 0.0
    position = 0
    entry_price = None

    actions = []
    dates = []
    returns = []

    for step in range(horizon):
        idx = start_idx + step
        if idx + 1 >= len(df):
            break  # stop if out of bounds

        # Context window
        state_window = df[state_features].iloc[idx - 10:idx].astype(np.float32).values
        state_tensor = torch.tensor(state_window, dtype=torch.float32).unsqueeze(0)

        rtg_tensor = torch.tensor([[rtg_remaining] + [0.0] * 9], dtype=torch.float32).unsqueeze(-1)
        action_seq = torch.zeros((1, 10), dtype=torch.long)
        action_seq[0, 0] = 1  # HOLD seed
        timestep_seq = torch.arange(10).unsqueeze(0)

        with torch.no_grad():
            logits = model(rtg_tensor, state_tensor, action_seq, timestep_seq)
            action = torch.argmax(logits[:, 0, :], dim=-1).item()

        date = df["Date"].iloc[idx]
        close = df["Close"].iloc[idx]
        next_close = df["Close"].iloc[idx + 1]

        dates.append(date)
        actions.append(["SELL", "HOLD", "BUY"][action])

        # Simulate trade logic
        if action == 2 and position == 0:  # BUY
            position = 1
            entry_price = close
            returns.append(0.0)
        elif action == 0 and position == 1:  # SELL
            daily_return = (close - entry_price) / entry_price
            returns.append(daily_return)
            achieved += daily_return
            rtg_remaining = max(0.0, target_return - achieved)
            position = 0
            entry_price = None
        else:
            returns.append(0.0)

    if position == 1:
        final_return = (df["Close"].iloc[idx] - entry_price) / entry_price
        returns[-1] = final_return
        achieved += final_return

    return {
        "start_date": start_date,
        "target_return": target_return,
        "achieved_return": achieved,
        "dates": dates,
        "actions": actions,
        "daily_returns": returns
    }
    
result = run_extended_goal_rollout("2023-01-18", target_return=0.05, horizon=150)
for d, a, r in zip(result["dates"], result["actions"], result["daily_returns"]):
    print(f"{d.date()} — {a} — return: {r:.4f}")

print(f"\nTarget Return: {result['target_return']*100:.2f}%")
print(f"Achieved Return: {result['achieved_return']*100:.2f}%")
