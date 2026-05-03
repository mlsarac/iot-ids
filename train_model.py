# Transformer Multi-class Classifier for IoT IDS
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os


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


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
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
        
        # Classification Head
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, src):
        # src shape: (batch, input_dim)
        src = src.unsqueeze(-1) # (batch, input_dim, 1)
        src = self.embedding(src) # (batch, input_dim, d_model)
        src = self.pos_encoder(src)

        memory = self.encoder(src) # (batch, input_dim, d_model)
        
        # Global Average Pooling
        pooled = memory.mean(dim=1) # (batch, d_model)
        
        out = self.fc(pooled) # (batch, num_classes)
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

        # Tüm veriyi olduğu gibi listeye ekle (artık filtreleme yapmıyoruz)
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
    # DATA LOADING (WITH CACHING)
    # ========================
    cache_files = [
        "train_tensor.pt", "val_tensor.pt", 
        "train_labels.pt", "val_labels.pt", 
        "scaler.pkl", "features.pkl", "label_encoder.pkl"
    ]
    
    if all(os.path.exists(f) for f in cache_files):
        print("Found cached tensors and models. Loading...", flush=True)
        X_train_tensor = torch.load("train_tensor.pt", map_location="cpu", weights_only=True)
        X_val_tensor = torch.load("val_tensor.pt", map_location="cpu", weights_only=True)
        y_train_tensor = torch.load("train_labels.pt", map_location="cpu", weights_only=True)
        y_val_tensor = torch.load("val_labels.pt", map_location="cpu", weights_only=True)
        
        scaler = joblib.load("scaler.pkl")
        features = joblib.load("features.pkl")
        label_encoder = joblib.load("label_encoder.pkl")
        print("Loaded cached data successfully.", flush=True)
    else:
        print("No cached data found. Loading from CSV...", flush=True)
        df_train = load_csv_in_chunks("train.csv")
        df_val = load_csv_in_chunks("validation.csv")

        # Label'ları ayırma
        y_train_text = df_train["label"].values
        y_val_text = df_val["label"].values
        
        df_train = df_train.drop("label", axis=1)
        df_val = df_val.drop("label", axis=1)
        
        features = list(df_train.columns)

        print("Encoding labels...", flush=True)
        label_encoder = LabelEncoder()
        
        # Train ve Validation setlerindeki tüm benzersiz etiketleri birleştirip tanıtıyoruz
        # Böylece eksik etiket hatasının önüne geçiyoruz.
        all_labels = pd.concat([pd.Series(y_train_text), pd.Series(y_val_text)]).unique()
        label_encoder.fit(all_labels)
        
        y_train_encoded = label_encoder.transform(y_train_text)
        y_val_encoded = label_encoder.transform(y_val_text)
        
        print(f"Classes found ({len(label_encoder.classes_)}):", label_encoder.classes_, flush=True)

        print("Scaling features...", flush=True)
        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(df_train.values)
        X_val_scaled = scaler.transform(df_val.values)

        print("Scaling done", flush=True)

        # ========================
        # TENSOR CONVERSION
        # ========================
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
        
        # Etiketler Multi-class sınıflandırması için integer (long) olmalıdır
        y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)
        y_val_tensor = torch.tensor(y_val_encoded, dtype=torch.long)

        print("Saving tensors to cache...", flush=True)
        torch.save(X_train_tensor, "train_tensor.pt")
        torch.save(X_val_tensor, "val_tensor.pt")
        torch.save(y_train_tensor, "train_labels.pt")
        torch.save(y_val_tensor, "val_labels.pt")
        
        joblib.dump(scaler, "scaler.pkl")
        joblib.dump(features, "features.pkl")
        joblib.dump(label_encoder, "label_encoder.pkl")
        print("Caching done.", flush=True)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=2048,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=2048,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # ========================
    # MODEL INIT
    # ========================
    num_classes = len(label_encoder.classes_)
    model = TransformerClassifier(input_dim=len(features), num_classes=num_classes).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Mixed Precision (AMP) için Scaler
    amp_scaler = torch.amp.GradScaler(device.type) if device.type == 'cuda' else None

    num_epochs = 100
    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0

    print("START TRAINING MULTI-CLASS CLASSIFIER 🚀", flush=True)

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # TRAIN
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            if amp_scaler is not None:
                amp_scaler.scale(loss).backward()
                amp_scaler.step(optimizer)
                amp_scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            
            # Accuracy hesaplama
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()

            if i % 50 == 0:
                print(f"Epoch {epoch+1} Batch {i} | Loss: {loss.item():.4f}", flush=True)

        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total

        # VAL
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total

        print(f"Epoch {epoch+1} DONE | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | Time: {time.time()-epoch_start:.1f}s", flush=True)

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
    joblib.dump(label_encoder, "label_encoder.pkl")
    torch.save(model.state_dict(), "model.pth")

    print("ALL SAVED ✅", flush=True)


if __name__ == "__main__":
    main()
