from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_pinball_loss

OUT = Path("hasil/regresi")

df = pd.read_csv("data/regresi_train.csv.gz")
pred = pd.read_csv(OUT / "catboost_validation.csv")

# Menggabungkan prediksi dengan informasi paket
merged = pred.merge(
    df[["id_paket", "layanan", "mitra_kurir", "tgl_pickup"]],
    on="id_paket",
    how="left",
    validate="one_to_one"
)

assert len(merged) == len(pred)
assert merged["layanan"].notna().all()

def evaluate_group(group):
    y = group["actual"].to_numpy()
    p50 = group["p50"].to_numpy()
    p90 = group["p90"].to_numpy()

    return pd.Series({
        "jumlah_paket": len(group),
        "pinball_p50": mean_pinball_loss(y, p50, alpha=0.5),
        "pinball_p90": mean_pinball_loss(y, p90, alpha=0.9),
        "coverage_p90": np.mean(y <= p90)
    })

print("\n=== EVALUASI BERDASARKAN LAYANAN ===")

by_service = pd.DataFrame([
    {"layanan": name, **evaluate_group(group).to_dict()}
    for name, group in merged.groupby("layanan")
])

print(by_service.to_string(index=False))

print("\n=== EVALUASI BERDASARKAN MITRA KURIR ===")

by_courier = pd.DataFrame([
    {"mitra_kurir": name, **evaluate_group(group).to_dict()}
    for name, group in merged.groupby("mitra_kurir")
])

print(by_courier.to_string(index=False))

by_service.to_csv(
    OUT / "evaluasi_layanan.csv",
    index=False
)

by_courier.to_csv(
    OUT / "evaluasi_kurir.csv",
    index=False
)

print("\nEvaluasi kelompok selesai.")