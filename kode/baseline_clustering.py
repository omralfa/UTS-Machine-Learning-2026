from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

OUT = Path("hasil/clustering")
OUT.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
SAMPLE_SIZE = 5000

df = pd.read_csv("data/clustering.csv.gz")

print("=" * 65)
print("BASELINE CUSTOMER CLUSTERING - MINIBATCH KMEANS")
print("=" * 65)

print("Jumlah pelanggan:", len(df))

# --------------------------------------------------
# 1. Persiapan fitur
# --------------------------------------------------

X = df.drop(columns=["id_pelanggan"]).copy()

# Missing struktural: tidak ada pembelian 90 hari
X["tanpa_transaksi_90h"] = (
    df["frekuensi_90h"] == 0
).astype(int)

X["rata2_keranjang"] = (
    X["rata2_keranjang"].fillna(0)
)

# Kelompok proporsi yang jumlahnya mendekati 1
category_cols = [
    col for col in X.columns
    if col.startswith("porsi_")
    and not col.startswith("porsi_jam_")
]

time_cols = [
    col for col in X.columns
    if col.startswith("porsi_jam_")
]

# Mengurangi redundansi komposisional
X = X.drop(
    columns=[
        category_cols[-1],
        time_cols[-1]
    ]
)

# Encoding kategorikal
X = pd.get_dummies(
    X,
    columns=["perangkat"],
    dtype=int
)

# --------------------------------------------------
# 2. Transformasi fitur skewed
# --------------------------------------------------

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
        assert (X[col] >= 0).all(), (
            f"Nilai negatif pada {col}"
        )
        X[col] = np.log1p(X[col])

# --------------------------------------------------
# 3. Scaling
# --------------------------------------------------

assert not X.isna().any().any()
assert np.isfinite(X.to_numpy()).all()

scaler = RobustScaler()

X_scaled = scaler.fit_transform(X).astype(
    np.float32
)

assert np.isfinite(X_scaled).all()

print("Jumlah fitur akhir:", X_scaled.shape[1])

# --------------------------------------------------
# 4. Sampling evaluasi
# --------------------------------------------------

rng = np.random.default_rng(RANDOM_STATE)

sample_idx = rng.choice(
    len(X_scaled),
    size=min(SAMPLE_SIZE, len(X_scaled)),
    replace=False
)

X_sample = X_scaled[sample_idx]

# --------------------------------------------------
# 5. Eksperimen K
# --------------------------------------------------

results = []

for k in [3, 4, 5, 6, 7, 8]:

    print(f"\nMelatih MiniBatchKMeans K={k}...", flush=True)

    model = MiniBatchKMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        batch_size=2048,
        n_init=5,
        max_iter=200
    )

    labels = model.fit_predict(X_scaled)

    sample_labels = labels[sample_idx]

    silhouette = silhouette_score(
        X_sample,
        sample_labels,
        sample_size=None
    )

    dbi = davies_bouldin_score(
        X_sample,
        sample_labels
    )

    chi = calinski_harabasz_score(
        X_sample,
        sample_labels
    )

    counts = np.bincount(
        labels,
        minlength=k
    )

    result = {
        "k": k,
        "silhouette": float(silhouette),
        "davies_bouldin": float(dbi),
        "calinski_harabasz": float(chi),
        "min_cluster": int(counts.min()),
        "max_cluster": int(counts.max()),
        "cluster_sizes": counts.tolist()
    }

    results.append(result)

    print(
        f"Silhouette: {silhouette:.6f} | "
        f"DBI: {dbi:.6f} | "
        f"CHI: {chi:.2f}"
    )

    print("Ukuran cluster:", counts.tolist())

# --------------------------------------------------
# 6. Simpan hasil
# --------------------------------------------------

result_df = pd.DataFrame([
    {
        key: value
        for key, value in row.items()
        if key != "cluster_sizes"
    }
    for row in results
])

result_df.to_csv(
    OUT / "baseline_kmeans_metrics.csv",
    index=False
)

with open(
    OUT / "baseline_kmeans_details.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(results, file, indent=2)

print("\n=== RINGKASAN ===")
print(result_df.to_string(index=False))

print("\nBaseline clustering selesai.")
