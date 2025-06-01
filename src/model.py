import torch
import torch.nn as nn


class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, action_dim,embed_dim, context_length=10, n_layers=3, n_heads=1, dropout=0.1, num_classes=3):
        super().__init__()

        self.embed_rtg = nn.Linear(1, embed_dim)
        self.embed_state = nn.Linear(state_dim, embed_dim)
        self.embed_action = nn.Embedding(num_classes, embed_dim)  # NEW: use embedding for action tokens

        self.max_timestep = 1024
        self.embed_timestep = nn.Embedding(self.max_timestep, embed_dim)

        self.ln_rtg = nn.LayerNorm(embed_dim)
        self.ln_state = nn.LayerNorm(embed_dim)
        self.ln_action = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.predict_action = nn.Linear(embed_dim, num_classes)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.context_len = context_length
        self.num_classes = num_classes

    def generate_causal_mask(self, seq_len):
        return torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)

    def forward(self, rtgs, states, actions, timesteps):
        timesteps = timesteps.clamp(max=self.max_timestep - 1)
        pos_embedding = self.embed_timestep(timesteps)

        rtg_embedding = self.ln_rtg(self.embed_rtg(rtgs)) + pos_embedding
        state_embedding = self.ln_state(self.embed_state(states)) + pos_embedding
        
        action_embedding = self.ln_action(self.embed_action(actions)) + pos_embedding

        input_embeds = torch.cat(
            [rtg_embedding.unsqueeze(2), state_embedding.unsqueeze(2), action_embedding.unsqueeze(2)], dim=2
        ).reshape(rtgs.shape[0], -1, self.embed_dim)

        input_embeds = self.dropout(input_embeds)

        seq_len = input_embeds.size(1)
        causal_mask = self.generate_causal_mask(seq_len).to(input_embeds.device)

        hidden_states = self.transformer(input_embeds, mask=causal_mask)

        action_hidden = hidden_states[:, 2::3, :]
        pred_action = self.predict_action(action_hidden)

        return pred_action  # shape: [B, K, num_classes]
