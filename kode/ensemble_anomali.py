from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

OUT = Path("hasil/clustering")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("data/clustering.csv.gz")

active_mask = df["frekuensi_90h"].to_numpy() > 0

X = df.drop(columns=["id_pelanggan"]).copy()

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

n_top = int(np.ceil(len(matrix) * 0.01))

score_list = []
selection_list = []

print("=" * 65)
print("ENSEMBLE ISOLATION FOREST")
print("=" * 65)

for seed in seeds:

    print(
        f"Melatih Isolation Forest seed={seed}...",
        flush=True
    )

    model = IsolationForest(
        n_estimators=200,
        max_samples=256,
        contamination="auto",
        random_state=seed,
        n_jobs=-1
    )

    model.fit(matrix)

    scores = -model.score_samples(matrix)

    score_list.append(scores)

    top_idx = np.argsort(
        scores,
        kind="stable"
    )[-n_top:]

    selected = np.zeros(
        len(matrix),
        dtype=bool
    )

    selected[top_idx] = True

    selection_list.append(selected)

score_matrix = np.column_stack(score_list)

ensemble_scores = score_matrix.mean(axis=1)

selection_matrix = np.column_stack(
    selection_list
)

vote_count = selection_matrix.sum(axis=1)

ensemble_top_idx = np.argsort(
    ensemble_scores,
    kind="stable"
)[-n_top:]

ensemble_selected = np.zeros(
    len(matrix),
    dtype=bool
)

ensemble_selected[ensemble_top_idx] = True

result = df.loc[active_mask, [
    "id_pelanggan",
    "recency_hari",
    "frekuensi_90h",
    "belanja_90h",
    "retur_90h",
    "sesi_per_minggu",
    "rasio_konversi"
]].copy()

result["ensemble_score"] = ensemble_scores
result["vote_count"] = vote_count
result["candidate_anomaly"] = (
    ensemble_selected.astype(int)
)

print("\n=== DISTRIBUSI VOTE ===")

print(
    result["vote_count"]
    .value_counts()
    .sort_index()
)

print("\n=== TOP 1% ENSEMBLE ===")

selected = result[
    result["candidate_anomaly"] == 1
]

print("Jumlah kandidat:", len(selected))

print("\nDistribusi vote kandidat:")

print(
    selected["vote_count"]
    .value_counts()
    .sort_index()
)

print("\nMedian fitur kandidat:")

print(
    selected[[
        "recency_hari",
        "frekuensi_90h",
        "belanja_90h",
        "retur_90h",
        "sesi_per_minggu",
        "rasio_konversi"
    ]].median().to_string()
)

print("\n=== KESEPAKATAN ENSEMBLE VS MODEL INDIVIDU ===")

for seed, selected_single in zip(
    seeds,
    selection_list
):

    intersection = np.sum(
        ensemble_selected & selected_single
    )

    union = np.sum(
        ensemble_selected | selected_single
    )

    print(
        f"Seed {seed}: "
        f"intersection={intersection}, "
        f"jaccard={intersection / union:.6f}"
    )

result.to_csv(
    OUT / "ensemble_anomali_scores.csv",
    index=False
)

print("\nEnsemble anomali selesai.")
