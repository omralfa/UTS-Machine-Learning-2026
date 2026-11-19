from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = Path("data")
OUTPUT_DIR = Path("hasil/regresi")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(DATA_DIR / "regresi_train.csv.gz")
test = pd.read_csv(DATA_DIR / "regresi_test.csv.gz")

print("=" * 60)
print("EXPLORATORY DATA ANALYSIS - REGRESSION")
print("=" * 60)

print("\n1. DIMENSI DATASET")
print("Training:", train.shape)
print("Testing :", test.shape)

print("\n2. TIPE DATA")
print(train.dtypes.to_string())

print("\n3. STATISTIK TARGET")
print(train["durasi_jam"].describe(
    percentiles=[0.01, 0.10, 0.50, 0.90, 0.95, 0.99]
))

print("\n4. MISSING VALUES")
missing = train.isna().sum()
print(missing[missing > 0])

print("\n5. DISTRIBUSI TARGET")

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(train["durasi_jam"].dropna(), bins=80)
ax.set_title("Distribusi Durasi Pengiriman")
ax.set_xlabel("Durasi (jam)")
ax.set_ylabel("Jumlah Paket")
fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "distribusi_durasi.png",
    dpi=150
)
plt.close(fig)

print("\n6. STATISTIK BERDASARKAN LAYANAN")

layanan = train.groupby("layanan")["durasi_jam"].agg(
    ["count", "median", "mean", "std"]
)

print(layanan)

layanan.to_csv(
    OUTPUT_DIR / "statistik_layanan.csv"
)

print("\n7. STATISTIK BERDASARKAN MITRA KURIR")

kurir = train.groupby("mitra_kurir")["durasi_jam"].agg(
    ["count", "median", "mean", "std"]
)

print(kurir)

kurir.to_csv(
    OUTPUT_DIR / "statistik_kurir.csv"
)

print("\n8. KORELASI NUMERIK")

numeric = train.select_dtypes(include="number")

correlations = (
    numeric.corr()["durasi_jam"]
    .drop("durasi_jam")
    .sort_values(ascending=False)
)

print(correlations.head(15))
print("\nKorelasi terendah:")
print(correlations.tail(15))

correlations.to_csv(
    OUTPUT_DIR / "korelasi_target.csv"
)

print("\n9. PERBANDINGAN TRAIN DAN TEST")

train_columns = set(train.columns)
test_columns = set(test.columns)

print("Hanya di train:", train_columns - test_columns)
print("Hanya di test :", test_columns - train_columns)

print("\nEDA SELESAI")
print("Hasil disimpan di:", OUTPUT_DIR)