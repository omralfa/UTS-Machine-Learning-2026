from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

OUT = Path("hasil/clustering")
OUT.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
SAMPLE_SIZE = 3000

df = pd.read_csv("data/clustering.csv.gz")

# --------------------------------------------------
# 1. Preprocessing yang sama dengan baseline
# --------------------------------------------------

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
    columns=[category_cols[-1], time_cols[-1]]
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

assert not X.isna().any().any()
assert np.isfinite(X.to_numpy()).all()

# --------------------------------------------------
# 2. Tiga konfigurasi scaling
# --------------------------------------------------

robust = RobustScaler().fit_transform(X)

configs = {
    "A_robust": robust.astype(np.float32),

    "B_robust_clip5": np.clip(
        robust, -5, 5
    ).astype(np.float32)
}

# Winsorization berdasarkan persentil per fitur
lower = X.quantile(0.01)
upper = X.quantile(0.99)

X_winsor = X.clip(
    lower=lower,
    upper=upper,
    axis=1
)

configs["C_winsor_standard"] = (
    StandardScaler()
    .fit_transform(X_winsor)
    .astype(np.float32)
)

# --------------------------------------------------
# 3. Sampling evaluasi yang konsisten
# --------------------------------------------------

rng = np.random.default_rng(RANDOM_STATE)

sample_idx = rng.choice(
    len(X),
    size=SAMPLE_SIZE,
    replace=False
)

results = []

# --------------------------------------------------
# 4. Eksperimen
# --------------------------------------------------

for config_name, matrix in configs.items():

    print("\n" + "=" * 65)
    print("Konfigurasi:", config_name)
    print("=" * 65, flush=True)

    for k in [4, 5]:

        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            batch_size=2048,
            n_init=5,
            max_iter=200
        )

        labels = model.fit_predict(matrix)

        sample_matrix = matrix[sample_idx]
        sample_labels = labels[sample_idx]

        unique = np.unique(sample_labels)

        if len(unique) < 2:
            print(
                f"K={k}: sampel hanya memiliki "
                f"{len(unique)} cluster"
            )
            continue

        sil = silhouette_score(
            sample_matrix,
            sample_labels
        )

        dbi = davies_bouldin_score(
            sample_matrix,
            sample_labels
        )

        chi = calinski_harabasz_score(
            sample_matrix,
            sample_labels
        )

        counts = np.bincount(
            labels,
            minlength=k
        )

        largest_pct = (
            counts.max() / len(labels) * 100
        )

        result = {
            "config": config_name,
            "k": k,
            "silhouette": float(sil),
            "davies_bouldin": float(dbi),
            "calinski_harabasz": float(chi),
            "largest_cluster_pct": float(largest_pct),
            "cluster_sizes": counts.tolist()
        }

        results.append(result)

        print(
            f"K={k} | "
            f"Sil={sil:.6f} | "
            f"DBI={dbi:.6f} | "
            f"CHI={chi:.2f} | "
            f"Largest={largest_pct:.2f}%"
        )

        print("Cluster sizes:", counts.tolist())

# --------------------------------------------------
# 5. Simpan hasil
# --------------------------------------------------

summary = pd.DataFrame([
    {
        key: value
        for key, value in row.items()
        if key != "cluster_sizes"
    }
    for row in results
])

summary.to_csv(
    OUT / "eksperimen_scaling_metrics.csv",
    index=False
)

with open(
    OUT / "eksperimen_scaling_details.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(results, file, indent=2)

print("\n=== RINGKASAN ===")
print(summary.to_string(index=False))

print("\nEksperimen scaling selesai.")
