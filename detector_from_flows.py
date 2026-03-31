import csv
import math
import os
import time
from collections import defaultdict
from typing import Optional

import joblib
import pandas as pd
import torch
import torch.nn as nn
from scapy.all import sniff, IP, TCP, UDP, ARP


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


MODEL_PATH = "model.pth"
SCALER_PATH = "scaler.pkl"
FEATURES_PATH = "features.pkl"

# Canlı akışlardan çıkarılan feature'ların isteğe bağlı loglanacağı CSV
FLOW_CSV = "live_flows.csv"

# Raspberry Pi üzerindeki ağ arayüzü (IFACE env ile override edilebilir)
IFACE = os.environ.get("IFACE", "wlan0")

# Bu kadar saniye boyunca paket gelmeyen flow'u kapat (feature hesapla + model çalıştır)
FLOW_TIMEOUT = 30.0


# Eğitim CSV'sindeki feature sırası
FEATURE_HEADER = [
    "flow_duration",
    "Header_Length",
    "Protocol Type",
    "Duration",
    "Rate",
    "Srate",
    "Drate",
    "fin_flag_number",
    "syn_flag_number",
    "rst_flag_number",
    "psh_flag_number",
    "ack_flag_number",
    "ece_flag_number",
    "cwr_flag_number",
    "ack_count",
    "syn_count",
    "fin_count",
    "urg_count",
    "rst_count",
    "HTTP",
    "HTTPS",
    "DNS",
    "Telnet",
    "SMTP",
    "SSH",
    "IRC",
    "TCP",
    "UDP",
    "DHCP",
    "ARP",
    "ICMP",
    "IPv",
    "LLC",
    "Tot sum",
    "Min",
    "Max",
    "AVG",
    "Std",
    "Tot size",
    "IAT",
    "Number",
    "Magnitue",
    "Radius",
    "Covariance",
    "Variance",
    "Weight",
    "label",
]


class FlowStats:
    def __init__(self, first_ts: float) -> None:
        self.first_ts = first_ts
        self.last_ts = first_ts
        self.pkt_times: list[float] = [first_ts]
        self.sizes: list[int] = []          # tüm paketler
        self.fwd_sizes: list[int] = []      # src->dst (outgoing)
        self.bwd_sizes: list[int] = []      # dst->src (incoming)

        self.header_len_sum = 0
        self.total_bytes = 0

        self.total_pkts = 0
        self.fwd_pkts = 0  # src->dst
        self.bwd_pkts = 0  # dst->src

        # TCP flag sayaçları
        self.fin = 0
        self.syn = 0
        self.rst = 0
        self.psh = 0
        self.ack = 0
        self.ece = 0
        self.cwr = 0
        self.ack_count = 0
        self.syn_count = 0
        self.fin_count = 0
        self.urg_count = 0
        self.rst_count = 0

        # Uygulama/protokol one‑hot
        self.HTTP = 0
        self.HTTPS = 0
        self.DNS = 0
        self.Telnet = 0
        self.SMTP = 0
        self.SSH = 0
        self.IRC = 0
        self.TCP = 0
        self.UDP = 0
        self.DHCP = 0
        self.ARP = 0
        self.ICMP = 0
        self.IPv = 0
        self.LLC = 0


def get_flow_key(pkt):
    if IP in pkt:
        ip = pkt[IP]
        proto = ip.proto
        sport = None
        dport = None
        if TCP in pkt:
            sport = pkt[TCP].sport
            dport = pkt[TCP].dport
        elif UDP in pkt:
            sport = pkt[UDP].sport
            dport = pkt[UDP].dport
        return (ip.src, ip.dst, sport, dport, proto)
    if ARP in pkt:
        arp = pkt[ARP]
        return (arp.psrc, arp.pdst, None, None, "ARP")
    return None


flows: dict = {}


def update_app_one_hot(flow: FlowStats, src_port, dst_port, proto, pkt) -> None:
    # L4
    if proto == 6:
        flow.TCP = 1
    elif proto == 17:
        flow.UDP = 1

    # L3
    if IP in pkt:
        flow.IPv = 1
    if ARP in pkt:
        flow.ARP = 1

    ports = {src_port, dst_port}

    if 80 in ports or 8080 in ports:
        flow.HTTP = 1
    if 443 in ports:
        flow.HTTPS = 1
    if 53 in ports:
        flow.DNS = 1
    if 23 in ports:
        flow.Telnet = 1
    if 25 in ports or 587 in ports:
        flow.SMTP = 1
    if 22 in ports:
        flow.SSH = 1
    if 194 in ports:
        flow.IRC = 1
    if 67 in ports or 68 in ports:
        flow.DHCP = 1

    if proto == 1:
        flow.ICMP = 1
    # LLC tespiti için düşük seviye Ethernet analizi gerekir; burada 0 bırakıyoruz.


def update_tcp_flags(flow: FlowStats, tcp) -> None:
    flags = tcp.flags
    if flags & 0x01:
        flow.fin += 1
        flow.fin_count += 1
    if flags & 0x02:
        flow.syn += 1
        flow.syn_count += 1
    if flags & 0x04:
        flow.rst += 1
        flow.rst_count += 1
    if flags & 0x08:
        flow.psh += 1
    if flags & 0x10:
        flow.ack += 1
        flow.ack_count += 1
    if flags & 0x20:
        flow.urg_count += 1
    if flags & 0x40:
        flow.ece += 1
    if flags & 0x80:
        flow.cwr += 1


def process_packet(pkt) -> None:
    ts = getattr(pkt, "time", time.time())
    key = get_flow_key(pkt)
    if not key:
        return

    src, dst, sport, dport, proto = key

    if key not in flows:
        flows[key] = FlowStats(ts)
    flow: FlowStats = flows[key]

    # Yön: ilk key src->dst olarak kabul edildiği için karşılaştırma basit
    if src == key[0] and dst == key[1]:
        # src->dst yönü: outgoing
        flow.fwd_pkts += 1
    else:
        # dst->src yönü: incoming
        flow.bwd_pkts += 1

    flow.last_ts = ts
    flow.total_pkts += 1
    flow.pkt_times.append(ts)

    size = len(pkt)
    flow.sizes.append(size)
    flow.total_bytes += size

    # Yöne göre ayrı boy listeleri
    if src == key[0] and dst == key[1]:
        flow.fwd_sizes.append(size)
    else:
        flow.bwd_sizes.append(size)

    # Header length (kabaca IP + TCP/UDP header toplamı)
    hdr_len = 0
    if IP in pkt:
        hdr_len += pkt[IP].ihl * 4
        if TCP in pkt:
            hdr_len += pkt[TCP].dataofs * 4
        elif UDP in pkt:
            hdr_len += 8
    flow.header_len_sum += hdr_len

    if TCP in pkt:
        update_tcp_flags(flow, pkt[TCP])

    proto_num = proto if isinstance(proto, int) else 0
    update_app_one_hot(flow, sport, dport, proto_num, pkt)


def finalize_flow(
    key,
    flow: FlowStats,
    model,
    scaler,
    feature_names,
    csv_writer: Optional[csv.writer] = None,
) -> None:
    dur = max(flow.last_ts - flow.first_ts, 1e-6)

    flow_duration = dur
    header_length = flow.header_len_sum
    protocol_type = 6 if flow.TCP else (17 if flow.UDP else 0)
    duration = dur

    rate = flow.total_pkts / dur
    srate = flow.fwd_pkts / dur
    drate = flow.bwd_pkts / dur

    fin_flag_number = flow.fin
    syn_flag_number = flow.syn
    rst_flag_number = flow.rst
    psh_flag_number = flow.psh
    ack_flag_number = flow.ack
    ece_flag_number = flow.ece
    cwr_flag_number = flow.cwr

    ack_count = flow.ack_count
    syn_count = flow.syn_count
    fin_count = flow.fin_count
    urg_count = flow.urg_count
    rst_count = flow.rst_count

    if flow.sizes:
        _min = min(flow.sizes)
        _max = max(flow.sizes)
        _avg = sum(flow.sizes) / len(flow.sizes)
        _var = sum((s - _avg) ** 2 for s in flow.sizes) / len(flow.sizes)
        _std = math.sqrt(_var)
        tot_sum = sum(flow.sizes)
    else:
        _min = _max = _avg = _std = _var = tot_sum = 0.0

    tot_size = flow.total_bytes

    if len(flow.pkt_times) > 1:
        diffs = [
            flow.pkt_times[i + 1] - flow.pkt_times[i]
            for i in range(len(flow.pkt_times) - 1)
        ]
        iat = sum(diffs) / len(diffs)
    else:
        iat = 0.0

    number = flow.total_pkts

    # CICIOT-23 resmi tanımlarına göre:
    # incoming = bwd (dst->src), outgoing = fwd (src->dst)
    in_sizes = flow.bwd_sizes
    out_sizes = flow.fwd_sizes

    def _mean_var(vals: list[int]) -> tuple[float, float]:
        if not vals:
            return 0.0, 0.0
        m = sum(vals) / len(vals)
        v = sum((x - m) ** 2 for x in vals) / len(vals)
        return m, v

    avg_in, var_in = _mean_var(in_sizes)
    avg_out, var_out = _mean_var(out_sizes)

    # Magnitude: (avg_in + avg_out) / 2
    magnitue = 0.5 * (avg_in + avg_out)

    # Radius: (var_in + var_out) / 2
    radius = 0.5 * (var_in + var_out)

    # Covariance: kovaryans(len_in, len_out)
    if in_sizes and out_sizes:
        n = min(len(in_sizes), len(out_sizes))
        xs = in_sizes[:n]
        ys = out_sizes[:n]
        mx = sum(xs) / n
        my = sum(ys) / n
        covariance = sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / n
    else:
        covariance = 0.0

    # Variance feature: var_in / var_out
    if var_out > 0.0:
        variance = var_in / var_out
    else:
        variance = 0.0

    # Weight: (#incoming packets) × (#outgoing packets)
    weight = len(in_sizes) * len(out_sizes)

    row = [
        flow_duration,
        header_length,
        protocol_type,
        duration,
        rate,
        srate,
        drate,
        fin_flag_number,
        syn_flag_number,
        rst_flag_number,
        psh_flag_number,
        ack_flag_number,
        ece_flag_number,
        cwr_flag_number,
        ack_count,
        syn_count,
        fin_count,
        urg_count,
        rst_count,
        flow.HTTP,
        flow.HTTPS,
        flow.DNS,
        flow.Telnet,
        flow.SMTP,
        flow.SSH,
        flow.IRC,
        flow.TCP,
        flow.UDP,
        flow.DHCP,
        flow.ARP,
        flow.ICMP,
        flow.IPv,
        flow.LLC,
        tot_sum,
        _min,
        _max,
        _avg,
        _std,
        tot_size,
        iat,
        number,
        magnitue,
        radius,
        covariance,
        variance,
        weight,
    ]

    if csv_writer is not None:
        csv_writer.writerow(row)

    df = pd.DataFrame([row], columns=FEATURE_HEADER[:-1])  # remove label
    X = df[feature_names]
    X_scaled = scaler.transform(X.values)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        reconstructed = model(X_tensor)
        mse = torch.mean((X_tensor - reconstructed) ** 2).item()

    threshold = 0.1  # Adjust based on training data
    if mse > threshold:
        label = "Anomaly"
    else:
        label = "Normal"

    src, dst, sport, dport, proto = key
    print(f"Flow {src}:{sport} -> {dst}:{dport} (proto={proto}) -> MSE: {mse:.4f}, {label}")


def expire_flows(
    model,
    scaler,
    feature_names,
    csv_writer: Optional[csv.writer] = None,
) -> None:
    now = time.time()
    to_delete = []
    for key, flow in list(flows.items()):
        if now - flow.last_ts > FLOW_TIMEOUT:
            finalize_flow(key, flow, model, scaler, feature_names, csv_writer)
            to_delete.append(key)
    for key in to_delete:
        del flows[key]


def main() -> None:
    # Load model
    input_dim = 46  # Assuming 46 features
    model = TransformerAutoencoder(input_dim=input_dim)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURES_PATH)

    print(f"Ağ arayüzü dinleniyor: {IFACE}")

    with open(FLOW_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(FEATURE_HEADER[:-1])  # remove label

        last_expire_check = time.time()

        def _cb(pkt):
            nonlocal last_expire_check
            process_packet(pkt)
            now = time.time()
            if now - last_expire_check > 5.0:
                expire_flows(model, scaler, feature_names, writer)
                last_expire_check = now

        sniff(iface=IFACE, prn=_cb, store=False)


if __name__ == "__main__":
    main()
