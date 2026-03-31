"""
Model ve feature uyumluluğunu kontrol eder.
Çalıştırma: python check_model.py
"""
import math
import sys
from pathlib import Path

import joblib
import pandas as pd
import torch
import torch.nn as nn

MODEL_PATH = "model.pth"
SCALER_PATH = "scaler.pkl"
FEATURES_PATH = "features.pkl"

# detector_from_flows.py ile aynı sıra (label hariç)
EXPECTED_FEATURES = [
    "flow_duration", "Header_Length", "Protocol Type", "Duration", "Rate", "Srate", "Drate",
    "fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number", "ack_flag_number",
    "ece_flag_number", "cwr_flag_number", "ack_count", "syn_count", "fin_count", "urg_count", "rst_count",
    "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC", "TCP", "UDP", "DHCP", "ARP", "ICMP", "IPv", "LLC",
    "Tot sum", "Min", "Max", "AVG", "Std", "Tot size", "IAT", "Number",
    "Magnitue", "Radius", "Covariance", "Variance", "Weight",
]


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
    ok = True

    # 1) Dosyalar var mı?
    for path in (MODEL_PATH, SCALER_PATH, FEATURES_PATH):
        if not Path(path).exists():
            print(f"[HATA] Dosya yok: {path}")
            ok = False
    if not ok:
        sys.exit(1)

    # 2) Yükle
    try:
        input_dim = 46
        model = TransformerAutoencoder(input_dim=input_dim)
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        scaler = joblib.load(SCALER_PATH)
        feature_names = joblib.load(FEATURES_PATH)
    except Exception as e:
        print(f"[HATA] Yükleme: {e}")
        sys.exit(1)

    # 3) feature_names detector'daki kolonlarla uyumlu mu?
    allowed = set(EXPECTED_FEATURES)
    missing = [f for f in feature_names if f not in allowed]
    if missing:
        print(f"[HATA] features.pkl'deki bazı isimler detector ile uyumlu değil: {missing}")
        ok = False
    else:
        print(f"[OK] Feature sayısı: {len(feature_names)} (hepsi beklenen listede)")

    # 4) Model input dim
    if model.input_dim != len(feature_names):
        print(f"[HATA] Model {model.input_dim} feature bekliyor, features.pkl'de {len(feature_names)} var.")
        ok = False
    else:
        print(f"[OK] Model feature sayısı uyumlu")

    # 5) Scaler
    if not hasattr(scaler, "transform"):
        print("[HATA] scaler.transform yok")
        ok = False
    else:
        print(f"[OK] Scaler yüklendi")

    # 6) Örnek reconstruction (sıfırlardan oluşan bir satır)
    try:
        X = pd.DataFrame([[0.0] * len(feature_names)], columns=feature_names)
        X_scaled = scaler.transform(X.values)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            reconstructed = model(X_tensor)
            mse = torch.mean((X_tensor - reconstructed) ** 2).item()
        print(f"[OK] Örnek reconstruction MSE (tümü 0): {mse:.4f}")
    except Exception as e:
        print(f"[HATA] Örnek reconstruction: {e}")
        ok = False

    if ok:
        print("\nTüm kontroller geçti. detector_from_flows.py kullanıma hazır.")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
