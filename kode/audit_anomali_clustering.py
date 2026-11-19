from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

OUT = Path("hasil/clustering")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("data/clustering.csv.gz")

active_mask = df["frekuensi_90h"].to_numpy() > 0

X = df.drop(columns=["id_pelanggan"]).copy()

X["rata2_keranjang"] = (
    X["rata2_keranjang"].fillna(0)
)

category_cols = [
    c for c in X.columns
    if c.startswith("porsi_")
    and not c.startswith("porsi_jam_")
]

time_cols = [
    c for c in X.columns
    if c.startswith("porsi_jam_")
]

X = X.drop(columns=[
    category_cols[-1],
    time_cols[-1]
])

X = pd.get_dummies(
    X,
    columns=["perangkat"],
    dtype=int
)

log_features = [
    "recency_hari",
    "frekuensi_90h",
    "belanja_90h",
    "rata2_keranjang",
    "item_per_pesanan",
    "retur_90h",
    "ulasan_90h",
    "sesi_per_minggu",
    "durasi_sesi_menit",
    "lama_member_bulan",
    "jumlah_alamat",
    "wishlist",
    "klik_iklan_30h",
    "belanja_30h",
    "frekuensi_30h",
    "n02",
    "n05",
    "n09",
    "n12"
]

for col in log_features:
    if col in X.columns:
        X[col] = np.log1p(X[col])

X_active = X.loc[active_mask].copy()

lower = X_active.quantile(0.01)
upper = X_active.quantile(0.99)

X_active = X_active.clip(
    lower=lower,
    upper=upper,
    axis=1
)

matrix = StandardScaler().fit_transform(
    X_active
).astype(np.float32)

print("=" * 65)
print("AUDIT ANOMALI - ISOLATION FOREST")
print("=" * 65)

print("Pelanggan dianalisis:", len(matrix))
print("Jumlah fitur:", matrix.shape[1])

model = IsolationForest(
    n_estimators=200,
    max_samples=256,
    contamination="auto",
    random_state=42,
    n_jobs=-1
)

model.fit(matrix)

# Semakin tinggi skor ini, semakin anomali.
scores = -model.score_samples(matrix)

result = df.loc[active_mask, [
    "id_pelanggan",
    "recency_hari",
    "frekuensi_90h",
    "belanja_90h",
    "rata2_keranjang",
    "retur_90h",
    "sesi_per_minggu",
    "rasio_konversi"
]].copy()

result["anomaly_score"] = scores

print("\n=== DISTRIBUSI SKOR ===")

print(
    result["anomaly_score"]
    .describe(
        percentiles=[
            0.50, 0.90, 0.95,
            0.97, 0.99, 0.995
        ]
    ).to_string()
)

print("\n=== PROFIL KANDIDAT ANOMALI ===")

for pct in [0.01, 0.02, 0.03, 0.05]:

    threshold = np.quantile(
        scores,
        1 - pct
    )

    mask = scores >= threshold

    subset = result.loc[mask]

    print(
        f"\nTop {pct * 100:.0f}% "
        f"| threshold={threshold:.6f} "
        f"| jumlah={len(subset)}"
    )

    print(
        subset[[
            "recency_hari",
            "frekuensi_90h",
            "belanja_90h",
            "retur_90h",
            "sesi_per_minggu",
            "rasio_konversi"
        ]].median().to_string()
    )

print("\n=== 10 SKOR ANOMALI TERTINGGI ===")

print(
    result.sort_values(
        "anomaly_score",
        ascending=False
    ).head(10).to_string(index=False)
)

result.to_csv(
    OUT / "audit_anomali_scores.csv",
    index=False
)

print("\nAudit anomali selesai.")
