from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_pinball_loss

OUT = Path("hasil/regresi")
OUT.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(
    "data/regresi_train.csv.gz",
    usecols=["id_paket", "pulau_tujuan"]
)

pred = pd.read_csv(
    OUT / "catboost_validation.csv"
)

merged = pred.merge(
    train,
    on="id_paket",
    how="left",
    validate="one_to_one"
)

assert len(merged) == len(pred), "Jumlah baris berubah."
assert merged["pulau_tujuan"].notna().all(), "Ada pulau tujuan kosong."
assert merged["actual"].notna().all(), "Ada target aktual kosong."

results = []

for pulau, group in merged.groupby("pulau_tujuan"):
    y = group["actual"].to_numpy()
    p50 = group["p50"].to_numpy()
    p90 = group["p90"].to_numpy()

    results.append({
        "pulau_tujuan": pulau,
        "jumlah_paket": len(group),
        "pinball_p50": mean_pinball_loss(y, p50, alpha=0.5),
        "pinball_p90": mean_pinball_loss(y, p90, alpha=0.9),
        "coverage_p90": np.mean(y <= p90)
    })

result = pd.DataFrame(results)
result = result.sort_values("pinball_p90", ascending=False)

assert result["jumlah_paket"].sum() == len(pred)

output = OUT / "evaluasi_pulau_tujuan.csv"
result.to_csv(output, index=False)

print("\n=== EVALUASI BERDASARKAN PULAU TUJUAN ===")
print(result.to_string(index=False))

print(f"\nTotal data validasi: {len(pred):,}")
print(f"Jumlah pulau: {len(result)}")
print(f"File tersimpan: {output}")
