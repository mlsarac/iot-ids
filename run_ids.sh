#!/bin/bash
# Kullanım: sudo ./run_ids.sh [arayüz]
# Örnek:  sudo ./run_ids.sh wlan0
# detector_from_flows.py kendi scapy sniff'i ile trafiği alır; CICFlowMeter gerekmez.

IFACE=${1:-wlan0}
export IFACE

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || {
  echo "Klasöre girilemedi: $SCRIPT_DIR"
  exit 1
}

echo "Arayüz: $IFACE"
echo "Başlatılıyor: detector_from_flows.py (Ctrl+C ile durdur)"
exec sudo python3 detector_from_flows.py

