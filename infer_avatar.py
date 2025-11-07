import os
import json
import torch
import argparse
import numpy as np
import cv2
from torch import nn

# -----------------------------
# Model (same architecture as train_avatar.py)
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
# Inference Function
# -----------------------------
def infer(model_path, vocab, gloss_text, out_path):
    # Load model
    pose_dim = 12  # same as used in training
    model = Gloss2PoseTransformer(len(vocab), pose_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Convert gloss text into IDs
    gloss_words = gloss_text.lower().split()
    gloss_ids = torch.tensor(
        [vocab[w] for w in gloss_words if w in vocab],
        dtype=torch.long
    ).unsqueeze(0)

    # Predict pose
    with torch.no_grad():
        pred = model(gloss_ids).squeeze().numpy()

    # -----------------------------
    # Convert predicted pose → dummy animation
    # -----------------------------
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    w, h = 480, 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, 10, (w, h))

    for i in range(30):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        x = int(240 + 100 * np.sin(pred[0] + i / 5))
        y = int(240 + 100 * np.cos(pred[1] + i / 5))
        cv2.circle(frame, (x, y), 25, (0, 255, 0), -1)
        cv2.putText(frame, gloss_text, (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(frame)

    out.release()
    print(f"✅ Avatar video generated at: {out_path}")


# -----------------------------
# Main CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to trained model file")
    parser.add_argument("--gloss", required=True, help="Input gloss text")
    parser.add_argument("--out", required=True, help="Output video path")
    args = parser.parse_args()

    # Load vocabulary from samples (build from dataset)
    vocab_path = os.path.join("samples")
    vocab = {}
    for fname in os.listdir(vocab_path):
        if not fname.endswith(".json"):
            continue
        data = json.load(open(os.path.join(vocab_path, fname)))
        for w in data["gloss"]:
            if w not in vocab:
                vocab[w] = len(vocab)

    infer(args.model, vocab, args.gloss, args.out)
