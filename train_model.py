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
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: (seq_len, batch, d_model)
        return x + self.pe[:x.size(0)]


class TransformerAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1
    ):
        super().__init__()
        self.input_dim = input_dim

        self.embedding = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=input_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output = nn.Linear(d_model, 1)

    def forward(self, src):
        # src: (batch, seq_len)
        src = src.unsqueeze(-1)              # (batch, seq_len, 1)
        src = self.embedding(src)            # (batch, seq_len, d_model)
        src = src.transpose(0, 1)            # (seq_len, batch, d_model)
        src = self.pos_encoder(src)

        memory = self.encoder(src)

        # reconstruction için decoder input olarak yine src kullanılıyor
        output = self.decoder(src, memory)
        output = output.transpose(0, 1)      # (batch, seq_len, d_model)
        output = self.output(output).squeeze(-1)  # (batch, seq_len)

        return output


def main():
    # Dosya yolları
    train_csv = "train.csv"
    val_csv = "validation.csv"

    # Cihaz seçimi
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # Train verisi
    df_train = pd.read_csv(train_csv)
    if df_train.columns[-1] != "label":
        raise ValueError(f"Train CSV son sütun 'label' değil: {df_train.columns[-1]}")

    if "benign" in df_train["label"].values:
        X_train = df_train[df_train["label"] == "benign"].iloc[:, :-1]
    else:
        X_train = df_train.iloc[:, :-1]

    # Validation verisi
    df_val = pd.read_csv(val_csv)
    if df_val.columns[-1] != "label":
        raise ValueError(f"Validation CSV son sütun 'label' değil: {df_val.columns[-1]}")

    if "benign" in df_val["label"].values:
        X_val = df_val[df_val["label"] == "benign"].iloc[:, :-1]
    else:
        X_val = df_val.iloc[:, :-1]

    # Feature isimleri
    features = list(X_train.columns)

    # Ölçeklendirme
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.values)
    X_val_scaled = scaler.transform(X_val.values)

    # Tensor
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)

    # Dataset / DataLoader
    train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, X_val_tensor)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    # Model
    input_dim = X_train.shape[1]
    model = TransformerAutoencoder(input_dim=input_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    num_epochs = 100
    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0

        for inputs, targets in train_dataloader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_dataloader)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, targets in val_dataloader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_dataloader)

        print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # Early stopping + best model save
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            print("Best model updated.")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # En iyi modeli yükle
    model.load_state_dict(torch.load("best_model.pth", map_location=device))

    # Final kayıtlar
    torch.save(model.state_dict(), "model.pth")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(features, "features.pkl")

    print("model.pth, scaler.pkl, features.pkl kaydedildi.")


if __name__ == "__main__":
    main()
