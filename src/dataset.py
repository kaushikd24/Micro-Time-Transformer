import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from torch.serialization import add_safe_globals

# Allow unpickling of pandas.Timestamp (needed for loading `.pt` files with "dates")
add_safe_globals([pd.Timestamp])

class DecisionTransformerDataset(Dataset):
    def __init__(self, data_path: str, context_len: int = 10):
        # Safe load with weights_only=False to support timestamps
        data = torch.load(data_path, weights_only=False)

        self.states = data["states"]               # shape: [N, K, D]
        self.actions = data["actions"]             # shifted actions (input)
        self.targets = data["action_targets"]      # actual next actions (target)
        self.rtgs = data["rtgs"]                   # shape: [N, K]
        self.context_len = context_len

        # Shift actions from [-1, 0, 1] to [0, 1, 2]
        self.actions = self.actions + 1
        self.targets = self.targets + 1

    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, idx):
        state_seq = self.states[idx].to(torch.float32)             # [K, D]
        rtg_seq = self.rtgs[idx].unsqueeze(-1).to(torch.float32)   # [K, 1]
        action_seq = self.actions[idx].long()                      # [K]
        target_seq = self.targets[idx].long()                      # [K]
        timesteps = torch.arange(self.context_len).long()

        return rtg_seq, state_seq, action_seq, timesteps, target_seq
