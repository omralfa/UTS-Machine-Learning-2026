from pathlib import Path
import pandas as pd

OUT = Path("hasil/clustering")

df = pd.read_csv("data/clustering.csv.gz")

labels = pd.read_csv(
    OUT / "label_dua_tahap_seed42.csv"
)

assert len(df) == len(labels)
assert df["id_pelanggan"].is_unique
assert labels["id_pelanggan"].is_unique

merged = df.merge(
    labels,
    on="id_pelanggan",
    how="left",
    validate="one_to_one"
)

assert merged["cluster"].notna().all()

features = [
    "recency_hari",
    "frekuensi_90h",
    "belanja_90h",
    "rata2_keranjang",
    "belanja_30h",
    "frekuensi_30h",
    "retur_90h",
    "sesi_per_minggu",
    "rasio_konversi",
    "lama_member_bulan"
]

print("=" * 70)
print("PROFIL CLUSTER - MODEL DUA TAHAP")
print("=" * 70)

print("\n=== JUMLAH PELANGGAN ===")
print(
    merged["cluster"]
    .value_counts()
    .sort_index()
)

print("\n=== MEDIAN FITUR ===")

profile = merged.groupby("cluster")[
    features
].median()

print(profile.to_string())

print("\n=== PELANGGAN TANPA TRANSAKSI ===")

print(
    pd.crosstab(
        merged["cluster"],
        merged["frekuensi_90h"] == 0
    )
)

print("\n=== PERANGKAT DOMINAN ===")

print(
    pd.crosstab(
        merged["cluster"],
        merged["perangkat"],
        normalize="index"
    ).round(3).to_string()
)

profile.to_csv(
    OUT / "profil_dua_tahap.csv"
)

print("\nProfil model dua tahap selesai.")
