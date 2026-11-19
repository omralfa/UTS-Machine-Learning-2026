from pathlib import Path
import hashlib
import json

import pandas as pd

ROOT = Path(".")
OUT = ROOT / "hasil" / "clustering"

required_files = [
    ROOT / "data" / "clustering.csv.gz",
    ROOT / "cluster.csv",
    ROOT / "kode" / "eksperimen_dua_tahap.py",
    ROOT / "kode" / "profil_dua_tahap.py",
    ROOT / "kode" / "audit_anomali_clustering.py",
    ROOT / "kode" / "stabilitas_anomali.py",
    ROOT / "kode" / "ensemble_anomali.py",
    ROOT / "kode" / "finalisasi_clustering.py",
    OUT / "label_dua_tahap_seed42.csv",
    OUT / "profil_dua_tahap.csv",
    OUT / "stabilitas_dua_tahap.csv",
    OUT / "ensemble_anomali_scores.csv",
    OUT / "stabilitas_anomali_jaccard.csv",
    OUT / "metadata_final.json",
]

print("=" * 65)
print("AUDIT FINAL CHECKPOINT 3")
print("=" * 65)

print("\n=== KELENGKAPAN FILE ===")

missing = []

for path in required_files:

    exists = path.is_file()

    print(
        f"{'OK' if exists else 'MISSING':7} {path}"
    )

    if not exists:
        missing.append(str(path))

if missing:
    raise FileNotFoundError(
        "File wajib belum lengkap: "
        + ", ".join(missing)
    )

print("\n=== VALIDASI DATA ===")

source = pd.read_csv(
    ROOT / "data" / "clustering.csv.gz"
)

submission = pd.read_csv(
    ROOT / "cluster.csv"
)

assert submission.columns.tolist() == [
    "id_pelanggan",
    "cluster"
]

assert len(submission) == len(source) == 120000

assert submission["id_pelanggan"].is_unique

assert submission["cluster"].notna().all()

assert submission["cluster"].isin(
    [-1, 0, 1, 2, 3]
).all()

assert submission["id_pelanggan"].tolist() == (
    source["id_pelanggan"].tolist()
)

counts = submission["cluster"].value_counts()

expected_counts = {
    -1: 1133,
    0: 41787,
    1: 43128,
    2: 27216,
    3: 6736,
}

for label, expected in expected_counts.items():

    actual = int(counts.get(label, 0))

    assert actual == expected, (
        f"Cluster {label}: "
        f"expected={expected}, actual={actual}"
    )

inactive = source["frekuensi_90h"].eq(0)

assert (
    submission.loc[inactive, "cluster"] == 3
).all()

print("Jumlah pelanggan:", len(submission))
print("ID unik:", submission["id_pelanggan"].is_unique)
print("Missing label:", submission["cluster"].isna().sum())
print("Distribusi label:", expected_counts)

print("\n=== SHA-256 ===")

hasher = hashlib.sha256()

with open(ROOT / "cluster.csv", "rb") as file:

    for chunk in iter(
        lambda: file.read(1024 * 1024),
        b""
    ):
        hasher.update(chunk)

sha256 = hasher.hexdigest()

print("cluster.csv:", sha256)

audit = {
    "checkpoint": "CP3",
    "status": "validated",
    "rows": len(submission),
    "cluster_counts": {
        str(k): int(v)
        for k, v in expected_counts.items()
    },
    "sha256_cluster_csv": sha256,
    "required_files_checked": len(required_files),
    "missing_files": missing,
}

with open(
    OUT / "audit_cp3.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        audit,
        file,
        indent=2,
        ensure_ascii=False
    )

print("\nAUDIT CP 3 BERHASIL.")
