# Transformer Autoencoder for IoT IDS Anomaly Detection
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.embedding = nn.Linear(1, d_model)  # embed each feature value
        self.pos_encoder = PositionalEncoding(d_model, max_len=input_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=False),
            num_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=False),
            num_layers
        )
        self.output = nn.Linear(d_model, 1)

    def forward(self, src):
        # src: (batch, seq_len)
        src = src.unsqueeze(-1)  # (batch, seq_len, 1)
        src_emb = self.embedding(src)  # (batch, seq_len, d_model)
        src_emb = src_emb.transpose(0, 1)  # (seq_len, batch, d_model)
        src_emb = self.pos_encoder(src_emb)
        memory = self.encoder(src_emb)
        # For decoder, use src_emb as tgt (teacher forcing for reconstruction)
        tgt_emb = src_emb
        output = self.decoder(tgt_emb, memory)
        output = output.transpose(0, 1)  # (batch, seq_len, d_model)
        output = self.output(output).squeeze(-1)  # (batch, seq_len)
        return output


def main():
    # CSV dosya adı
    input_csv = "data.csv"

    df = pd.read_csv(input_csv)

    # Son sütunun label olduğundan emin ol
    if df.columns[-1] != "label":
        raise ValueError(f"Son sütun 'label' değil: {df.columns[-1]}")

    # 46 feature + 1 label
    X = df.iloc[:, :-1]

    # For autoencoder, train on all data or filter benign
    # Assume 'benign' is the normal class
    if 'benign' in df['label'].values:
        X = df[df['label'] == 'benign'].iloc[:, :-1]
    else:
        X = df.iloc[:, :-1]  # train on all if no 'benign'

    X_values = X.values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_values)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, X_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    input_dim = X.shape[1]  # 46
    model = TransformerAutoencoder(input_dim=input_dim)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            inputs, targets = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")

    # Save model and scaler
    torch.save(model.state_dict(), "model.pth")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(list(X.columns), "features.pkl")
    print("model.pth, scaler.pkl, features.pkl kaydedildi.")


if __name__ == "__main__":
    main()

