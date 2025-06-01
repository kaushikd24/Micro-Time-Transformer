import torch
from sklearn.metrics import classification_report
from src.model import DecisionTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load val set
val_data = torch.load("prepared_datasets/val.pt", weights_only=False)
states = val_data["states"].to(torch.float32).to(device)       # [N, 10, 18]
rtgs = val_data["rtgs"].unsqueeze(-1).to(torch.float32).to(device)  # [N, 10, 1]
timesteps = val_data["timesteps"].to(torch.long).to(device)    # [N, 10]
true_actions = val_data["action_targets"] + 1  # remap to [0, 1, 2]
true_actions = true_actions.view(-1).cpu().numpy()

# Load model
model = DecisionTransformer(
    state_dim=18,
    action_dim=3,
    embed_dim=256,
    context_length=10,
    num_classes=3,
    n_layers=4,
    n_heads=4
).to(device)

model.load_state_dict(torch.load("models/model_epoch_5.pt", map_location=device))
model.eval()

# Autoregressive evaluation
pred_actions = []

with torch.no_grad():
    for i in range(states.shape[0]):
        s = states[i].unsqueeze(0)          # [1, 10, 18]
        r = rtgs[i].unsqueeze(0)            # [1, 10, 1]
        t = timesteps[i].unsqueeze(0)       # [1, 10]

        # Seed with actual shifted previous action (+1 offset!)
        a = torch.zeros((1, 10), dtype=torch.long).to(device)
        a[0, 0] = val_data["actions"][i, 0] + 1  # this fixes the embedding error

        for j in range(10):
            logits = model(r, s, a, t)                    # [1, 10, 3]
            pred = torch.argmax(logits[:, j, :], dim=-1)  # [1]
            a[0, j] = pred                                # feed prediction back

        pred_actions.extend(a.view(-1).cpu().tolist())

# Evaluate
print("\nClassification Report (Autoregressive Evaluation):")
print(classification_report(true_actions, pred_actions, target_names=["SELL", "HOLD", "BUY"]))
