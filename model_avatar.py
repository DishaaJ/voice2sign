import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class Gloss2PoseTransformer(nn.Module):
    def __init__(self, vocab_size, pose_dim=150, d_model=128, nhead=8, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.posenc = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model*2)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.linear = nn.Linear(d_model, pose_dim)

    def forward(self, x):
        x = self.embed(x)
        x = self.posenc(x)
        x = self.encoder(x.permute(1,0,2))
        x = x.permute(1,0,2)
        return self.linear(x)
