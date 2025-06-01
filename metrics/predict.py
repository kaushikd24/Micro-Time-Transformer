import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.eval import pred_actions, true_actions

# Load your original CSV
df = pd.read_csv("data/sp500_scalper_with_rtg.csv", parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Trim to match number of predictions
df_eval = df.iloc[-len(pred_actions):].copy().reset_index(drop=True)

# Add predictions
df_eval["predicted_action"] = pred_actions  # 0: SELL, 1: HOLD, 2: BUY

# Compute next-day return
df_eval["next_day_return"] = df_eval["Close"].shift(-1) / df_eval["Close"] - 1

# Create cumulative return curves per class
results = {}
for label, name in zip([0, 1, 2], ["SELL", "HOLD", "BUY"]):
    mask = df_eval["predicted_action"] == label
    returns = df_eval.loc[mask, "next_day_return"].fillna(0)
    cum_returns = (1 + returns).cumprod()
    results[name] = cum_returns.reset_index(drop=True)

# Plot
plt.figure(figsize=(10, 5))
for name, series in results.items():
    plt.plot(series, label=name)

plt.title("Cumulative Returns by Predicted Action")
plt.xlabel("Trade Index")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
