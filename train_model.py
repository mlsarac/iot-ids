# Geçici / placeholder: Bu dosya tamamen değişecek; yeni eğitim pipeline'ı yazılacak.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib


def main():
    # CSV dosya adı
    input_csv = "data.csv"

    df = pd.read_csv(input_csv)

    # Son sütunun gerçekten label olduğundan emin ol
    if df.columns[-1] != "label":
        raise ValueError(f"Son sütun 'label' değil: {df.columns[-1]}")

    # 46 feature + 1 label
    X = df.iloc[:, :-1]
    y_raw = df["label"]

    # Metin label'ları sayısala çevir
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Sınıflar:", list(le.classes_))
    print(classification_report(y_test, y_pred))

    joblib.dump(model, "model.pkl")
    joblib.dump(list(X.columns), "features.pkl")
    joblib.dump(le, "label_encoder.pkl")
    print("model.pkl, features.pkl, label_encoder.pkl kaydedildi.")


if __name__ == "__main__":
    main()

