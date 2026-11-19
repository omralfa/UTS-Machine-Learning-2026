from pathlib import Path

import pandas as pd

OUT = Path("hasil/regresi")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(
    "data/regresi_train.csv.gz",
    usecols=[
        "id_paket",
        "tgl_pickup",
        "mitra_kurir",
        "layanan",
        "durasi_jam"
    ]
)

df["tgl_pickup"] = pd.to_datetime(df["tgl_pickup"])
df["bulan"] = df["tgl_pickup"].dt.to_period("M").astype(str)

# Statistik bulanan untuk seluruh mitra
monthly = (
    df.groupby(["bulan", "mitra_kurir"])["durasi_jam"]
    .agg(
        jumlah_paket="count",
        mean="mean",
        median="median",
        p90=lambda x: x.quantile(0.90)
    )
    .reset_index()
)

monthly.to_csv(
    OUT / "statistik_bulanan_kurir.csv",
    index=False
)

print("\n=== STATISTIK BULANAN K07 ===")

k07 = monthly[
    monthly["mitra_kurir"] == "K07"
]

print(k07.to_string(index=False))

# Periksa coverage K07 pada Juli dan Agustus
pred = pd.read_csv(
    OUT / "catboost_validation.csv"
)

merged = pred.merge(
    df[["id_paket", "tgl_pickup", "mitra_kurir"]],
    on="id_paket",
    validate="one_to_one"
)

merged["bulan"] = (
    merged["tgl_pickup"]
    .dt.to_period("M")
    .astype(str)
)

merged["covered"] = (
    merged["actual"] <= merged["p90"]
)

coverage = (
    merged.groupby(["bulan", "mitra_kurir"])
    .agg(
        jumlah_paket=("covered", "size"),
        coverage_p90=("covered", "mean")
    )
    .reset_index()
)

coverage.to_csv(
    OUT / "coverage_bulanan_kurir.csv",
    index=False
)

print("\n=== COVERAGE K07 JULI–AGUSTUS ===")

print(
    coverage[
        coverage["mitra_kurir"] == "K07"
    ].to_string(index=False)
)

print("\nInvestigasi selesai.")