from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import RobustScaler, StandardScaler

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

profile_cols = [
    "recency_hari",
    "frekuensi_90h",
    "belanja_90h",
    "rata2_keranjang",
    "retur_90h",
    "sesi_per_minggu",
    "rasio_konversi",
    "lama_member_bulan"
]

for name, matrix in matrices.items():

    print("\n" + "=" * 65)
    print("KONFIGURASI:", name)
    print("=" * 65)

    model = MiniBatchKMeans(
        n_clusters=4,
        random_state=42,
        batch_size=2048,
        n_init=5,
        max_iter=200
    )

    labels = model.fit_predict(matrix)

    result = df.copy()
    result["cluster"] = labels

    result["tanpa_transaksi_90h"] = (
        result["frekuensi_90h"] == 0
    )

    print("\n=== JUMLAH PELANGGAN ===")
    print(
        result["cluster"]
        .value_counts()
        .sort_index()
    )

    print("\n=== TANPA TRANSAKSI 90 HARI ===")

    cross = pd.crosstab(
        result["cluster"],
        result["tanpa_transaksi_90h"]
    )

    print(cross)

    print("\n=== PROFIL MEDIAN ===")

    profile = result.groupby("cluster")[
        profile_cols
    ].median()

    print(profile.to_string())

    profile.to_csv(
        OUT / f"profil_{name}.csv"
    )

    cross.to_csv(
        OUT / f"aktivitas_{name}.csv"
    )

    pd.DataFrame({
        "id_pelanggan": result["id_pelanggan"],
        "cluster": labels
    }).to_csv(
        OUT / f"label_awal_{name}.csv",
        index=False
    )

print("\nProfil awal clustering selesai.")
