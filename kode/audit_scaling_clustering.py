from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler

OUT = Path("hasil/clustering")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("data/clustering.csv.gz")

X = df.drop(columns=["id_pelanggan"]).copy()

X["tanpa_transaksi_90h"] = (
    df["frekuensi_90h"] == 0
).astype(int)

X["rata2_keranjang"] = (
    X["rata2_keranjang"].fillna(0)
)

category_cols = [
    col for col in X.columns
    if col.startswith("porsi_")
    and not col.startswith("porsi_jam_")
]

time_cols = [
    col for col in X.columns
    if col.startswith("porsi_jam_")
]

X = X.drop(
    columns=[
        category_cols[-1],
        time_cols[-1]
    ]
)

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

scaler = RobustScaler()

scaled = scaler.fit_transform(X)

scaled_df = pd.DataFrame(
    scaled,
    columns=X.columns
)

records = []

for col in scaled_df.columns:
    values = scaled_df[col].to_numpy()
    abs_values = np.abs(values)

    records.append({
        "feature": col,
        "std_scaled": float(np.std(values)),
        "p99_abs": float(np.quantile(abs_values, 0.99)),
        "max_abs": float(np.max(abs_values)),
        "mean_square": float(np.mean(values ** 2))
    })

result = pd.DataFrame(records)

result = result.sort_values(
    "mean_square",
    ascending=False
)

print("=" * 65)
print("AUDIT DOMINASI FITUR SETELAH ROBUST SCALING")
print("=" * 65)

print("\n=== 20 FITUR DENGAN KONTRIBUSI KUADRAT TERBESAR ===")
print(result.head(20).to_string(index=False))

total = result["mean_square"].sum()

result["contribution_pct"] = (
    result["mean_square"] / total * 100
)

print("\n=== KONTRIBUSI 10 FITUR TERATAS ===")
print(
    result[
        ["feature", "contribution_pct"]
    ].head(10).to_string(index=False)
)

print(
    "\nTotal kontribusi 5 fitur terbesar:",
    f"{result['contribution_pct'].head(5).sum():.2f}%"
)

result.to_csv(
    OUT / "audit_dominasi_fitur.csv",
    index=False
)

print("\nAudit scaling selesai.")
