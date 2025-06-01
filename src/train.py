import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.dataset import DecisionTransformerDataset
from src.model import DecisionTransformer

def train_model(
    data_path,
    state_dim,
    action_dim,
    embed_dim=256,
    context_len=10,
    num_classes=3,
    epochs=20,
    batch_size=64,
    learning_rate=1e-4,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    # Load dataset
    dataset = DecisionTransformerDataset(data_path, context_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = DecisionTransformer(
        state_dim=state_dim,
        action_dim=action_dim,
        embed_dim=embed_dim,
        context_length=context_len,
        num_classes=num_classes,
        n_layers=4,         #upgraded depth
        n_heads=4           #upgraded width
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        
            
        for batch in loop:
            rtgs, states, actions, timesteps, targets = [x.to(device) for x in batch]

            logits = model(rtgs, states, actions, timesteps)  # [B, K, num_classes]

            # Compute loss only on the action positions
            loss = loss_fn(
                logits.view(-1, num_classes),
                targets.view(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())


        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed | Avg Loss: {avg_loss:.6f}")

        # Save checkpoint
        torch.save(model.state_dict(), f"models/model_epoch_{epoch+1}.pt")

    print("Training complete.")
    
    

if __name__ == "__main__":
    train_model(
        data_path="prepared_datasets/train.pt",
        state_dim=18,
        action_dim=3,
        embed_dim=256,
        context_len=10,
        num_classes=3,
        epochs=5,
        batch_size=64,
        learning_rate=1e-4
    )

