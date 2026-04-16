# Transformer Autoencoder for IoT IDS Anomaly Detection
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib


# ========================
# MODEL
# ========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        d_model = 128

        self.embedding = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=input_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=512,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=512,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)

        self.output = nn.Linear(d_model, 1)

    def forward(self, src):
        src = src.unsqueeze(-1)
        src = self.embedding(src)
        src = self.pos_encoder(src)

        memory = self.encoder(src)
        out = self.decoder(src, memory)

        out = self.output(out).squeeze(-1)
        return out


# ========================
# CHUNK LOADER
# ========================
def load_csv_in_chunks(file_path, chunk_size=100000):
    chunks = []
    print(f"Loading {file_path} in chunks...", flush=True)

    for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
        print(f"Chunk {i} loaded: {len(chunk)} rows", flush=True)

        if chunk.columns[-1] != "label":
            raise ValueError("Son sütun label değil!")

        if "benign" in chunk["label"].values:
            chunk = chunk[chunk["label"] == "benign"]

        chunk = chunk.iloc[:, :-1]  # label drop
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    print(f"{file_path} total rows: {len(df)}", flush=True)

    return df


# ========================
# MAIN
# ========================
def main():
    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)

    # ========================
    # DATA LOADING
    # ========================
    df_train = load_csv_in_chunks("train.csv")
    df_val = load_csv_in_chunks("validation.csv")

    # DEBUG için küçültmek istersen aç:
    # df_train = df_train.sample(100000)
    # df_val = df_val.sample(20000)

    features = list(df_train.columns)

    print("Scaling...", flush=True)
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(df_train.values)
    X_val_scaled = scaler.transform(df_val.values)

    print("Scaling done", flush=True)

    # ========================
    # TENSOR
    # ========================
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, X_val_tensor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # ========================
    # MODEL
    # ========================
    model = TransformerAutoencoder(input_dim=len(features)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    num_epochs = 100
    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0

    print("START TRAINING 🚀", flush=True)

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # TRAIN
        model.train()
        train_loss = 0

        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if i % 200 == 0:
                print(f"Epoch {epoch+1} Batch {i}", flush=True)

        train_loss /= len(train_loader)

        # VAL
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1} DONE | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Time: {time.time()-epoch_start:.1f}s", flush=True)

        # EARLY STOPPING
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("EARLY STOPPING", flush=True)
                break

    print("TOTAL TIME:", time.time() - start_time, flush=True)

    # SAVE
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(features, "features.pkl")
    torch.save(model.state_dict(), "model.pth")

    print("ALL SAVED ✅", flush=True)


if __name__ == "__main__":
    main()
