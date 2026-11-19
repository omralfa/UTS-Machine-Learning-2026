from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("hasil/clustering")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("data/clustering.csv.gz")

print("=" * 65)
print("VALIDASI FITUR CUSTOMER CLUSTERING")
print("=" * 65)

print("\n=== MISSING RATA-RATA KERANJANG ===")

missing = df["rata2_keranjang"].isna()
zero_freq = df["frekuensi_90h"].eq(0)

print("Jumlah missing:", int(missing.sum()))
print("Jumlah frekuensi nol:", int(zero_freq.sum()))

print("\nCrosstab missing vs frekuensi nol:")
print(pd.crosstab(missing, zero_freq))

print("\n=== NILAI NON-FINITE ===")

numeric = df.select_dtypes(include="number")

inf_count = int(np.isinf(numeric.to_numpy()).sum())

print("Jumlah inf/-inf:", inf_count)

print("\n=== FITUR KONSTAN ===")

constant = [
    col for col in df.columns
    if df[col].nunique(dropna=False) <= 1
]

print(constant if constant else "Tidak ada")

print("\n=== KOLOM IDENTIK ===")

identical = []

columns = numeric.columns.tolist()

for i, col1 in enumerate(columns):
    for col2 in columns[i + 1:]:
        if numeric[col1].equals(numeric[col2]):
            identical.append((col1, col2))

print(identical if identical else "Tidak ada")

print("\n=== KORELASI TINGGI ===")

corr = numeric.corr().abs()

pairs = []

for i, col1 in enumerate(corr.columns):
    for col2 in corr.columns[i + 1:]:
        value = corr.loc[col1, col2]

        if pd.notna(value) and value >= 0.95:
            pairs.append((col1, col2, float(value)))

pairs.sort(key=lambda x: x[2], reverse=True)

for a, b, value in pairs[:30]:
    print(f"{a} <-> {b}: {value:.6f}")

print("Jumlah pasangan korelasi >= 0.95:", len(pairs))

print("\n=== KONSISTENSI PROPORSI ===")

category_cols = [
    col for col in df.columns
    if col.startswith("porsi_")
    and not col.startswith("porsi_jam_")
]

time_cols = [
    col for col in df.columns
    if col.startswith("porsi_jam_")
]

for name, cols in [
    ("Kategori", category_cols),
    ("Waktu", time_cols)
]:
    total = df[cols].sum(axis=1)

    print(f"\n{name}:")
    print("Jumlah fitur:", len(cols))
    print("Minimum total:", total.min())
    print("Median total :", total.median())
    print("Maksimum total:", total.max())
    print(
        "Jumlah deviasi > 0.01:",
        int(((total - 1).abs() > 0.01).sum())
    )

print("\n=== RASIO KONVERSI ===")
print(
    "Nilai > 1:",
    int(df["rasio_konversi"].gt(1).sum())
)

print("\nValidasi fitur selesai.")
