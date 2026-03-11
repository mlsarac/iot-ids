"""
Model ve feature uyumluluğunu kontrol eder.
Çalıştırma: python check_model.py
"""
import sys
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = "model.pkl"
FEATURES_PATH = "features.pkl"
ENCODER_PATH = "label_encoder.pkl"

# detector_from_flows.py ile aynı sıra (label hariç)
EXPECTED_FEATURES = [
    "flow_duration", "Header_Length", "Protocol Type", "Duration", "Rate", "Srate", "Drate",
    "fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number", "ack_flag_number",
    "ece_flag_number", "cwr_flag_number", "ack_count", "syn_count", "fin_count", "urg_count", "rst_count",
    "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC", "TCP", "UDP", "DHCP", "ARP", "ICMP", "IPv", "LLC",
    "Tot sum", "Min", "Max", "AVG", "Std", "Tot size", "IAT", "Number",
    "Magnitue", "Radius", "Covariance", "Variance", "Weight",
]


def main():
    ok = True

    # 1) Dosyalar var mı?
    for path in (MODEL_PATH, FEATURES_PATH, ENCODER_PATH):
        if not Path(path).exists():
            print(f"[HATA] Dosya yok: {path}")
            ok = False
    if not ok:
        sys.exit(1)

    # 2) Yükle
    try:
        model = joblib.load(MODEL_PATH)
        feature_names = joblib.load(FEATURES_PATH)
        label_encoder = joblib.load(ENCODER_PATH)
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

    # 4) Model aynı sayıda feature bekliyor mu?
    try:
        n_features = model.n_features_in_
    except AttributeError:
        n_features = getattr(model, "n_features_", None)
    if n_features is not None and n_features != len(feature_names):
        print(f"[HATA] Model {n_features} feature bekliyor, features.pkl'de {len(feature_names)} var.")
        ok = False
    else:
        print(f"[OK] Model feature sayısı uyumlu")

    # 5) Label encoder
    if not hasattr(label_encoder, "inverse_transform"):
        print("[HATA] label_encoder.inverse_transform yok")
        ok = False
    else:
        print(f"[OK] Sınıflar: {list(label_encoder.classes_)}")

    # 6) Örnek tahmin (sıfırlardan oluşan bir satır)
    try:
        X = pd.DataFrame([[0.0] * len(feature_names)], columns=feature_names)
        pred = model.predict(X)
        label = label_encoder.inverse_transform(pred)[0]
        print(f"[OK] Örnek tahmin (tümü 0): {label}")
    except Exception as e:
        print(f"[HATA] Örnek tahmin: {e}")
        ok = False

    if ok:
        print("\nTüm kontroller geçti. detector_from_flows.py kullanıma hazır.")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
