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
from sklearn.preprocessing import LabelEncoder

MODEL_PATH = "model.pth"
SCALER_PATH = "scaler.pkl"
FEATURES_PATH = "features.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"

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
        self.input_dim = input_dim
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
        
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, src):
        src = src.unsqueeze(-1)
        src = self.embedding(src)
        src = self.pos_encoder(src)

        memory = self.encoder(src)
        pooled = memory.mean(dim=1)
        out = self.fc(pooled)
        return out


def main():
    ok = True

    # 1) Dosyalar var mı?
    for path in (MODEL_PATH, SCALER_PATH, FEATURES_PATH, LABEL_ENCODER_PATH):
        if not Path(path).exists():
            print(f"[HATA] Dosya yok: {path}")
            ok = False
    if not ok:
        sys.exit(1)

    # 2) Yükle
    try:
        scaler = joblib.load(SCALER_PATH)
        feature_names = joblib.load(FEATURES_PATH)
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        
        input_dim = 46
        num_classes = len(label_encoder.classes_)
        
        model = TransformerClassifier(input_dim=input_dim, num_classes=num_classes)
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
        model.eval()
        
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

    # 6) Örnek tahmin (sıfırlardan oluşan bir satır)
    try:
        X = pd.DataFrame([[0.0] * len(feature_names)], columns=feature_names)
        X_scaled = scaler.transform(X.values)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            output = model(X_tensor)
            probs = torch.softmax(output, dim=1)
            prob, predicted = torch.max(probs.data, 1)
            
            predicted_label = label_encoder.inverse_transform([predicted.item()])[0]
            confidence = prob.item() * 100
            
        print(f"[OK] Örnek tahmin (tümü 0 olan veri): {predicted_label} (%{confidence:.2f} güven)")
    except Exception as e:
        print(f"[HATA] Örnek tahmin: {e}")
        ok = False

    if ok:
        print("\nTüm kontroller geçti. detector_from_flows.py kullanıma hazır.")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()

