from pathlib import Path
import json

import pandas as pd

ROOT = Path(".")
OUT = ROOT / "hasil" / "clustering"

cluster_path = OUT / "label_dua_tahap_seed42.csv"
anomaly_path = OUT / "ensemble_anomali_scores.csv"
data_path = ROOT / "data" / "clustering.csv.gz"

clusters = pd.read_csv(cluster_path)
anomalies = pd.read_csv(anomaly_path)
source = pd.read_csv(data_path)

print("=" * 65)
print("FINALISASI TASK 3 - CUSTOMER CLUSTERING")
print("=" * 65)

# Validasi sumber
assert source["id_pelanggan"].is_unique
assert clusters["id_pelanggan"].is_unique
assert anomalies["id_pelanggan"].is_unique

assert len(source) == 120000
assert len(clusters) == len(source)

assert set(clusters["id_pelanggan"]) == set(
    source["id_pelanggan"]
)

assert anomalies["candidate_anomaly"].isin([0, 1]).all()

# Gabungkan label cluster dan status anomali
result = source[["id_pelanggan"]].merge(
    clusters,
    on="id_pelanggan",
    how="left",
    validate="one_to_one"
)

result = result.merge(
    anomalies[[
        "id_pelanggan",
        "candidate_anomaly"
    ]],
    on="id_pelanggan",
    how="left",
    validate="one_to_one"
)

assert result["cluster"].notna().all()

# Pelanggan tanpa transaksi tidak termasuk
# eksperimen Isolation Forest.
result["candidate_anomaly"] = (
    result["candidate_anomaly"]
    .fillna(0)
    .astype(int)
)

# Terapkan label -1
result.loc[
    result["candidate_anomaly"] == 1,
    "cluster"
] = -1

result["cluster"] = result["cluster"].astype(int)

# File submission
submission = result[[
    "id_pelanggan",
    "cluster"
]].copy()

submission.to_csv(
    ROOT / "cluster.csv",
    index=False
)

# Validasi final
assert len(submission) == 120000
assert submission["id_pelanggan"].is_unique
assert submission["cluster"].notna().all()
assert set(submission["cluster"]).issubset(
    {-1, 0, 1, 2, 3}
)

assert (
    submission["id_pelanggan"].tolist()
    == source["id_pelanggan"].tolist()
)

anomaly_count = int(
    (submission["cluster"] == -1).sum()
)

assert anomaly_count == 1133

# Pastikan pelanggan tanpa transaksi
# tetap pada cluster 3.
check = source[[
    "id_pelanggan",
    "frekuensi_90h"
]].merge(
    submission,
    on="id_pelanggan",
    validate="one_to_one"
)

inactive = check["frekuensi_90h"] == 0

assert (
    check.loc[inactive, "cluster"] == 3
).all()

print("\n=== DISTRIBUSI CLUSTER FINAL ===")

counts = (
    submission["cluster"]
    .value_counts()
    .sort_index()
)

print(counts.to_string())

print("\n=== PERSENTASE ===")

print(
    (
        counts / len(submission) * 100
    ).round(3).to_string()
)

print("\n=== VALIDASI ===")
print("Jumlah baris:", len(submission))
print("ID unik:", submission["id_pelanggan"].is_unique)
print("Missing cluster:", submission["cluster"].isna().sum())
print("Jumlah anomali:", anomaly_count)
print("Pelanggan tanpa transaksi:", int(inactive.sum()))

metadata = {
    "task": "customer_clustering",
    "method": "two_stage_minibatch_kmeans",
    "active_clusters": 3,
    "inactive_cluster": 3,
    "clustering_seed": 42,
    "anomaly_method": "isolation_forest_ensemble",
    "anomaly_seeds": [11, 22, 33, 42, 55],
    "anomaly_top_fraction": 0.01,
    "anomaly_count": anomaly_count,
    "total_customers": len(submission),
    "cluster_counts": {
        str(k): int(v)
        for k, v in counts.items()
    }
}

with open(
    OUT / "metadata_final.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        metadata,
        file,
        indent=2,
        ensure_ascii=False
    )

print("\n=== PREVIEW FILE ===")
print(submission.head(10).to_string(index=False))

print("\nFile berhasil dibuat: cluster.csv")
print("Finalisasi Task 3 selesai.")
