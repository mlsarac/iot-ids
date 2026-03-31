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
    # CSV dosya adları
    train_csv = "train.csv"
    val_csv = "validation.csv"
    test_csv = "test.csv"

    # Eğitim verisini yükle
    df_train = pd.read_csv(train_csv)
    if df_train.columns[-1] != "label":
        raise ValueError(f"Train CSV son sütun 'label' değil: {df_train.columns[-1]}")
    X_train = df_train[df_train['label'] == 'benign'].iloc[:, :-1] if 'benign' in df_train['label'].values else df_train.iloc[:, :-1]

    # Validasyon verisini yükle
    df_val = pd.read_csv(val_csv)
    if df_val.columns[-1] != "label":
        raise ValueError(f"Validation CSV son sütun 'label' değil: {df_val.columns[-1]}")
    X_val = df_val[df_val['label'] == 'benign'].iloc[:, :-1] if 'benign' in df_val['label'].values else df_val.iloc[:, :-1]

    # Test verisini yükle (şimdilik sadece kontrol için)
    df_test = pd.read_csv(test_csv)
    if df_test.columns[-1] != "label":
        raise ValueError(f"Test CSV son sütun 'label' değil: {df_test.columns[-1]}")

    # Özellik isimlerini kaydet
    features = list(X_train.columns)

    # Eğitim verisini ölçeklendir
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.values)
    X_val_scaled = scaler.transform(X_val.values)

    # Tensor'lara dönüştür
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)

    # Dataset ve DataLoader oluştur
    train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, X_val_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    input_dim = X_train.shape[1]  # 46
    model = TransformerAutoencoder(input_dim=input_dim)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    num_epochs = 100
    best_val_loss = float('inf')
    patience = 10  # Early stopping patience
    patience_counter = 0

    for epoch in range(num_epochs):
        # Eğitim
        model.train()
        train_loss = 0
        for batch in train_dataloader:
            inputs, targets = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dataloader)

        # Validasyon
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                inputs, targets = batch
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        val_loss /= len(val_dataloader)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # En iyi modeli kaydet
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Final model ve scaler'ı kaydet (en iyi modeli yükle)
    model.load_state_dict(torch.load("best_model.pth"))
    torch.save(model.state_dict(), "model.pth")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(features, "features.pkl")
    print("model.pth, scaler.pkl, features.pkl kaydedildi.")


if __name__ == "__main__":
    main()

