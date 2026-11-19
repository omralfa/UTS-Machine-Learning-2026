from pathlib import Path
import pandas as pd

OUT = Path("hasil/klasifikasi")
OUT.mkdir(parents=True, exist_ok=True)

train = pd.read_csv("data/klasifikasi_train.csv.gz")
test = pd.read_csv("data/klasifikasi_test.csv.gz")

print("=" * 60)
print("AUDIT DATA KLASIFIKASI")
print("=" * 60)

print("\n=== UKURAN DATA ===")
print("Training:", train.shape)
print("Testing :", test.shape)

print("\n=== TARGET FRAUD ===")
print(train["is_fraud"].value_counts(dropna=False))

print("\nProporsi fraud:")
print(train["is_fraud"].value_counts(normalize=True, dropna=False))

print("\n=== STATISTIK NOMINAL ===")
print(
    train["nominal"].describe(
        percentiles=[0.50, 0.90, 0.95, 0.99]
    )
)

print("\n=== MISSING VALUES TERBANYAK ===")
print(
    train.isna()
    .mean()
    .sort_values(ascending=False)
    .head(20)
    .to_string()
)

print("\n=== PEMERIKSAAN ID ===")
print("Duplikat ID train:", train["id_transaksi"].duplicated().sum())
print("Duplikat ID test :", test["id_transaksi"].duplicated().sum())

print("\n=== PERIODE DATA ===")

train["waktu"] = pd.to_datetime(train["waktu"])
test["waktu"] = pd.to_datetime(test["waktu"])

print("Train:", train["waktu"].min(), "sampai", train["waktu"].max())
print("Test :", test["waktu"].min(), "sampai", test["waktu"].max())

print("\n=== FRAUD PER BULAN ===")

train["bulan"] = train["waktu"].dt.to_period("M").astype(str)

monthly = (
    train.groupby("bulan")["is_fraud"]
    .agg(
        transaksi="count",
        jumlah_fraud="sum",
        fraud_rate="mean"
    )
    .reset_index()
)

print(monthly.to_string(index=False))

monthly.to_csv(
    OUT / "fraud_bulanan.csv",
    index=False
)

print("\n=== PEMERIKSAAN FITUR REVIEW ===")

if "flag_review_manual" in train.columns:
    print(
        pd.crosstab(
            train["flag_review_manual"],
            train["is_fraud"],
            normalize="index"
        )
    )

print("\n=== KOLOM DATA ===")
print(train.columns.tolist())

print("\nAudit klasifikasi selesai.")
