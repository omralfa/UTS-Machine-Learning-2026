from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

OUT = Path("hasil/clustering")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("data/clustering.csv.gz")

active_mask = df["frekuensi_90h"].to_numpy() > 0

X = df.drop(columns=["id_pelanggan"]).copy()

X["tanpa_transaksi_90h"] = (
    df["frekuensi_90h"] == 0
).astype(int)

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
    time_cols[-1],
    "tanpa_transaksi_90h"
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

# Preprocessing hanya menggunakan pelanggan bertransaksi
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

labels_by_seed = {}

print("=" * 65)
print("EKSPERIMEN CLUSTERING DUA TAHAP")
print("=" * 65)

print("Pelanggan bertransaksi:", active_mask.sum())
print("Pelanggan tanpa transaksi:", (~active_mask).sum())

for seed in seeds:

    model = MiniBatchKMeans(
        n_clusters=3,
        random_state=seed,
        batch_size=2048,
        n_init=5,
        max_iter=200
    )

    labels = model.fit_predict(matrix)

    labels_by_seed[seed] = labels

    print(
        f"Seed {seed}:",
        np.bincount(labels, minlength=3).tolist(),
        flush=True
    )

records = []

for seed_a, seed_b in combinations(seeds, 2):

    ari = adjusted_rand_score(
        labels_by_seed[seed_a],
        labels_by_seed[seed_b]
    )

    records.append({
        "seed_a": seed_a,
        "seed_b": seed_b,
        "ari_active": ari
    })

results = pd.DataFrame(records)

print("\n=== STABILITAS DUA TAHAP ===")
print(results.to_string(index=False))

print("\n=== RINGKASAN ===")
print(
    results["ari_active"]
    .agg(["mean", "std", "min", "max"])
    .to_string()
)

results.to_csv(
    OUT / "stabilitas_dua_tahap.csv",
    index=False
)

# Simpan label referensi seed 42 untuk pemeriksaan profil
labels_active = labels_by_seed[42]

labels_full = np.full(
    len(df),
    3,
    dtype=int
)

labels_full[active_mask] = labels_active

reference = pd.DataFrame({
    "id_pelanggan": df["id_pelanggan"],
    "cluster": labels_full
})

reference.to_csv(
    OUT / "label_dua_tahap_seed42.csv",
    index=False
)

print("\n=== DISTRIBUSI REFERENSI ===")
print(
    reference["cluster"]
    .value_counts()
    .sort_index()
)

print("\nEksperimen dua tahap selesai.")
