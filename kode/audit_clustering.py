from pathlib import Path
import pandas as pd
import numpy as np

OUT = Path("hasil/clustering")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("data/clustering.csv.gz")

print("=" * 65)
print("AUDIT DATASET CUSTOMER CLUSTERING")
print("=" * 65)

print("\n=== UKURAN DATA ===")
print("Jumlah pelanggan:", len(df))
print("Jumlah kolom:", len(df.columns))

print("\n=== DUPLIKASI ID ===")
print("Duplikat id_pelanggan:", df["id_pelanggan"].duplicated().sum())

print("\n=== TIPE DATA ===")
print(df.dtypes.value_counts())

print("\n=== MISSING VALUES ===")
missing = (
    df.isna()
    .mean()
    .sort_values(ascending=False)
)

print(missing.head(25).to_string())

print("\n=== FITUR KATEGORIKAL ===")
categorical = df.select_dtypes(
    include=["object", "category"]
).columns.tolist()

for col in categorical:
    print(f"\n{col}:")
    print(df[col].value_counts(dropna=False).head(10).to_string())

print("\n=== STATISTIK NUMERIK ===")
numeric = df.select_dtypes(include="number")

stats = numeric.describe(
    percentiles=[0.01, 0.25, 0.50, 0.75, 0.95, 0.99]
).T

print(
    stats[
        ["mean", "std", "min", "50%", "95%", "99%", "max"]
    ].to_string()
)

stats.to_csv(OUT / "statistik_numerik.csv")

print("\n=== NILAI EKSTREM ===")

for col in [
    "recency_hari",
    "frekuensi_90h",
    "belanja_90h",
    "rata2_keranjang",
    "retur_90h",
    "sesi_per_minggu"
]:
    if col in df.columns:
        print(
            f"{col}: "
            f"min={df[col].min()}, "
            f"median={df[col].median()}, "
            f"max={df[col].max()}"
        )

print("\n=== KOLOM DATA ===")
print(df.columns.tolist())

print("\nAudit clustering selesai.")
