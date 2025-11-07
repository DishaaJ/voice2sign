import os
import json
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import argparse

# -----------------------------
# Dataset Definition
# -----------------------------
class AvatarDataset(Dataset):
    def __init__(self, data_dir):
        self.samples = []
        self.vocab = {}

        for fname in os.listdir(data_dir):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(data_dir, fname), 'r', encoding='utf8') as f:
                d = json.load(f)

            gloss = d['gloss']
            pose = torch.tensor(d['pose'], dtype=torch.float32)

            # Store (gloss, pose)
            self.samples.append((gloss, pose))

            for w in gloss:
                if w not in self.vocab:
                    self.vocab[w] = len(self.vocab)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        gloss, pose = self.samples[idx]
        gloss_ids = torch.tensor([self.vocab[w] for w in gloss], dtype=torch.long)
        return gloss_ids, pose


# -----------------------------
# Model Definition
# -----------------------------
class Gloss2PoseTransformer(nn.Module):
    def __init__(self, vocab_size, pose_dim, hidden=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.encoder = nn.LSTM(hidden, hidden, batch_first=True)
        self.decoder = nn.Linear(hidden, pose_dim)

    def forward(self, gloss_seq):
        x = self.embed(gloss_seq)
        _, (h, _) = self.encoder(x)
        out = self.decoder(h[-1])
        return out


# -----------------------------
# Training Function
# -----------------------------
def train(data_dir, out_file, epochs):
    ds = AvatarDataset(data_dir)
    dl = DataLoader(ds, batch_size=1, shuffle=True)

    # ✅ FIX: pose_dim = ds[0][1].shape[0] instead of shape[1]
    model = Gloss2PoseTransformer(len(ds.vocab), pose_dim=ds[0][1].shape[0])

    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    print(f"Starting training with {len(ds)} samples and pose_dim={ds[0][1].shape[0]}")

    for epoch in range(epochs):
        total_loss = 0
        for gloss, pose in dl:
            opt.zero_grad()
            pred = model(gloss)
            loss = loss_fn(pred.squeeze(), pose.squeeze())
            loss.backward()
            opt.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(dl):.4f}")

    torch.save(model.state_dict(), out_file)
    print(f"✅ Training complete! Model saved to {out_file}")


# -----------------------------
# CLI Entry
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="Path to training samples directory")
    parser.add_argument("--out", required=True, help="Output model file")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    args = parser.parse_args()

    train(args.data_dir, args.out, args.epochs)
