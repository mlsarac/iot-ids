# IoT IDS – Raspberry Pi Gateway Tespit Sistemi

CICIOT-23 uyumlu feature’larla canlı trafikten flow üretip, eğitilmiş model ile saldırı tespiti yapan proje.

## Gereksinimler

- Python 3.8+
- `requirements.txt` içindeki paketler

```bash
pip install -r requirements.txt
```

## Model dosyaları

Tespit için aşağıdaki dosyaların proje kökünde olması gerekir:

| Dosya | Açıklama |
|-------|----------|
| `model.pkl` | Eğitilmiş sınıflandırıcı (örn. `sklearn` RandomForest) |
| `features.pkl` | Modelin kullandığı feature isimleri listesi |
| `label_encoder.pkl` | LabelEncoder (metin etiket ↔ sayı) |

Bu dosyalar CICIOT-23 formatında feature’lara sahip bir CSV ile eğitim pipeline’ından üretilir. Projedeki **eğitim kodu (`train_model.py`) Tramsformer Autoencoder ile eğitilmektedir**

## Model ve feature uyumluluğunu kontrol etme

Dağıtmadan önce veya Pi’ye kopyaladıktan sonra:

```bash
python check_model.py
```

Bu script şunları kontrol eder:

- `model.pkl`, `features.pkl`, `label_encoder.pkl` dosyalarının varlığı
- `features.pkl` içindeki kolon isimlerinin `detector_from_flows.py` içindeki CICIOT-23 feature listesiyle uyumu
- Modelin beklediği feature sayısı ile `features.pkl` uzunluğunun eşleşmesi
- Örnek bir vektörle tahmin denemesi

Tüm kontroller geçerse çıktıda “Tüm kontroller geçti” yazar.

## Raspberry Pi’de çalıştırma

### 1. Projeyi Pi’ye kopyalayın

Örnek (kendi kullanıcı/IP’nizi yazın):

```bash
scp -r . pi@<PI_IP>:~/iot-ids/
```

### 2. Bağımlılıkları kurun

Pi’de:

```bash
cd ~/iot-ids
pip install -r requirements.txt
```

`scapy` ile ham paket yakalamak için **root** gerekir; scripti `sudo` ile çalıştıracaksınız.

### 3. Ağ arayüzünü belirleyin

Dinlenecek arayüz genelde `wlan0` (Wi‑Fi) veya `eth0` (Ethernet). Kontrol için:

```bash
ip link show
```

Gerekirse `detector_from_flows.py` içindeki `IFACE = "wlan0"` satırını değiştirin veya aşağıdaki gibi parametreyle verin.

### 4. Model kontrolü

Pi’de de bir kez doğrulayın:

```bash
python check_model.py
```

### 5. Tespiti başlatın

Ham paket erişimi için **sudo** zorunlu:

```bash
sudo python detector_from_flows.py
```

Veya `run_ids.sh` kullanıyorsanız (arayüz varsayılan `wlan0`):

```bash
sudo ./run_ids.sh
```

Farklı arayüz için:

```bash
sudo ./run_ids.sh eth0
```

### 6. Çıktılar

- **Konsol:** Her zaman aşımına uğrayan flow için tahmin edilen etiket (örn. `Benign`, saldırı türü) yazılır.
- **CSV:** `live_flows.csv` dosyasına CICIOT-23 formatında feature satırları eklenir (isteğe bağlı log için).

Durdurmak için `Ctrl+C` kullanın.

## Özet komutlar (Pi’de)

```bash
cd ~/iot-ids
pip install -r requirements.txt
python check_model.py
sudo python detector_from_flows.py
```

## Eğitim (train)

### Train modelinden beklenen çıktılar

Yeni train scriptinin **proje köküne** aşağıdaki üç dosyayı üretmesi gerekir. Detector ve `check_model.py` buna göre yazıldı.

| Çıktı dosyası | Format | Beklenti |
|---------------|--------|----------|
| **`model.pkl`** | `joblib.dump(model, "model.pkl")` | `sklearn` uyumlu sınıflandırıcı: `.predict(X)` ve (tercihen) `.n_features_in_` olmalı. `X` tek satırlık veya çok satırlık DataFrame veya 2D array; kolonlar `features.pkl` ile aynı sırada. |
| **`features.pkl`** | `joblib.dump(list_of_names, "features.pkl")` | Modelin kullandığı feature isimlerinin **listesi** (string). İsimler `detector_from_flows.py` içindeki `FEATURE_HEADER` listesinde olmalı; **`label` dahil edilmemeli**. Sıra, eğitimde modele verilen kolon sırasıyla aynı olmalı. |
| **`label_encoder.pkl`** | `joblib.dump(le, "label_encoder.pkl")` | `sklearn.preprocessing.LabelEncoder`: `.fit_transform()` ile sayıya, `.inverse_transform()` ile metin etikete çevrilmiş olmalı. `.classes_` sınıf isimlerini içerir. |

- Feature isimleri **CICIOT-23** ile uyumlu olmalı (örn. `Magnitue`, `Header_Length`, `Tot sum` vb.); detector canlıda bu isimlerle kolon üretiyor.
- Eğitim CSV’sindeki kolon isimleri bu feature listesiyle bire bir eşleşmeli (label hariç).
- İzin verilen tüm feature isimleri: `detector_from_flows.py` içindeki `FEATURE_HEADER` (son eleman `label`; model girişinde kullanılmaz).

## Notlar

- **Flow zaman aşımı:** `detector_from_flows.py` içinde `FLOW_TIMEOUT = 30.0` saniye; bu süre boyunca paket gelmeyen flow kapatılıp feature’lar hesaplanır ve modele gönderilir.
- **CICIOT-23:** Feature tanımları (Magnitude, Radius, Covariance, Variance, Weight) MDPI Sensors makalesindeki tabloya göre uygulanmıştır.
