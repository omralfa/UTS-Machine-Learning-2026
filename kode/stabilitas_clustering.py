from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import adjusted_rand_score

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

robust = RobustScaler().fit_transform(X)

matrices = {
    "B_robust_clip5": np.clip(
        robust, -5, 5
    ).astype(np.float32)
}

lower = X.quantile(0.01)
upper = X.quantile(0.99)

winsor = X.clip(
    lower=lower,
    upper=upper,
    axis=1
)

matrices["C_winsor_standard"] = (
    StandardScaler()
    .fit_transform(winsor)
    .astype(np.float32)
)

seeds = [11, 22, 33, 42, 55]

results = []

print("=" * 65)
print("EVALUASI STABILITAS CLUSTERING")
print("=" * 65)

for name, matrix in matrices.items():

    print(f"\nKonfigurasi: {name}", flush=True)

    labels_by_seed = {}

    for seed in seeds:

        model = MiniBatchKMeans(
            n_clusters=4,
            random_state=seed,
            batch_size=2048,
            n_init=5,
            max_iter=200
        )

        labels = model.fit_predict(matrix)

        labels_by_seed[seed] = labels

        print(
            f"Seed {seed}:",
            np.bincount(labels, minlength=4).tolist(),
            flush=True
        )

    ari_scores = []

    for seed_a, seed_b in combinations(seeds, 2):

        ari = adjusted_rand_score(
            labels_by_seed[seed_a],
            labels_by_seed[seed_b]
        )

        ari_scores.append(ari)

        results.append({
            "config": name,
            "seed_a": seed_a,
            "seed_b": seed_b,
            "ari": float(ari)
        })

    print(
        f"ARI mean={np.mean(ari_scores):.6f} | "
        f"min={np.min(ari_scores):.6f} | "
        f"max={np.max(ari_scores):.6f}"
    )

result_df = pd.DataFrame(results)

result_df.to_csv(
    OUT / "stabilitas_ari.csv",
    index=False
)

summary = (
    result_df.groupby("config")["ari"]
    .agg(["mean", "std", "min", "max"])
)

print("\n=== RINGKASAN STABILITAS ===")
print(summary.to_string())

summary.to_csv(
    OUT / "stabilitas_summary.csv"
)

print("\nEvaluasi stabilitas selesai.")
