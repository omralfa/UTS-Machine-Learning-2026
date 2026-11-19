from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

OUT = Path("hasil/clustering")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("data/clustering.csv.gz")

active_mask = df["frekuensi_90h"].to_numpy() > 0

X = df.drop(columns=["id_pelanggan"]).copy()

X["rata2_keranjang"] = X["rata2_keranjang"].fillna(0)

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

seeds = [11, 22, 33, 42, 55]

top_fraction = 0.01

n_top = int(np.ceil(
    len(matrix) * top_fraction
))

sets_by_seed = {}

print("=" * 65)
print("STABILITAS ISOLATION FOREST - TOP 1%")
print("=" * 65)

print("Jumlah pelanggan:", len(matrix))
print("Jumlah kandidat per seed:", n_top)

for seed in seeds:

    model = IsolationForest(
        n_estimators=200,
        max_samples=256,
        contamination="auto",
        random_state=seed,
        n_jobs=-1
    )

    model.fit(matrix)

    scores = -model.score_samples(matrix)

    # Pemilihan tepat n_top pelanggan
    top_indices = np.argsort(
        scores,
        kind="stable"
    )[-n_top:]

    sets_by_seed[seed] = set(
        top_indices.tolist()
    )

    print(
        f"Seed {seed}: "
        f"{len(sets_by_seed[seed])} kandidat",
        flush=True
    )

records = []

for seed_a, seed_b in combinations(seeds, 2):

    a = sets_by_seed[seed_a]
    b = sets_by_seed[seed_b]

    intersection = len(a & b)
    union = len(a | b)

    jaccard = intersection / union

    records.append({
        "seed_a": seed_a,
        "seed_b": seed_b,
        "intersection": intersection,
        "union": union,
        "jaccard": jaccard
    })

results = pd.DataFrame(records)

print("\n=== HASIL PERBANDINGAN ===")
print(results.to_string(index=False))

print("\n=== RINGKASAN JACCARD ===")

print(
    results["jaccard"]
    .agg(["mean", "std", "min", "max"])
    .to_string()
)

common_all = set.intersection(
    *sets_by_seed.values()
)

print(
    "\nKandidat yang muncul pada semua seed:",
    len(common_all)
)

results.to_csv(
    OUT / "stabilitas_anomali_jaccard.csv",
    index=False
)

print("\nEvaluasi stabilitas anomali selesai.")
