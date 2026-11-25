from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from catboost import CatBoostRegressor, Pool

SEED = 42
SAMPLE_SIZE = 500

OUT = Path("hasil/regresi")
OUT.mkdir(parents=True, exist_ok=True)

print("=== MEMUAT DATA VALIDASI ===", flush=True)

df = pd.read_csv("data/regresi_train.csv.gz")
df["tgl_pickup"] = pd.to_datetime(df["tgl_pickup"])

valid = df[df["tgl_pickup"] >= "2025-07-01"].copy()

sample = valid.sample(
    n=min(SAMPLE_SIZE, len(valid)),
    random_state=SEED
)

excluded = {
    "id_paket",
    "durasi_jam",
    "jumlah_scan",
    "tgl_pickup"
}

features = [c for c in df.columns if c not in excluded]

X = sample[features].copy()

categorical = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

# Konsisten dengan preprocessing training:
# median numerik dihitung dari data training temporal.
train = df[df["tgl_pickup"] < "2025-07-01"].copy()
numeric = X.select_dtypes(include="number").columns.tolist()
medians = train[numeric].median()

for col in categorical:
    X[col] = X[col].fillna("__MISSING__").astype(str)

X[numeric] = X[numeric].fillna(medians)

pool = Pool(
    X,
    cat_features=categorical
)

for quantile in ["p50", "p90"]:
    print(f"\n=== SHAP {quantile.upper()} ===", flush=True)

    model = CatBoostRegressor()
    model.load_model(str(OUT / f"catboost_{quantile}.cbm"))

    assert model.feature_names_ == features, (
        "Urutan fitur tidak sesuai model."
    )

    shap_values = model.get_feature_importance(
        pool,
        type="ShapValues",
        thread_count=-1
    )

    # Kolom terakhir adalah expected value.
    contributions = shap_values[:, :-1]

    importance = pd.DataFrame({
        "fitur": features,
        "mean_abs_shap": np.abs(contributions).mean(axis=0)
    }).sort_values(
        "mean_abs_shap",
        ascending=False
    )

    csv_path = OUT / f"shap_importance_{quantile}.csv"
    importance.to_csv(csv_path, index=False)

    top = importance.head(15).iloc[::-1]

    plt.figure(figsize=(9, 6))
    plt.barh(
        top["fitur"],
        top["mean_abs_shap"]
    )

    plt.xlabel("Mean |SHAP value| (jam)")
    plt.ylabel("Fitur")
    plt.title(
        f"15 Fitur Terpenting CatBoost {quantile.upper()}"
    )

    plt.tight_layout()

    png_path = OUT / f"shap_importance_{quantile}.png"
    plt.savefig(png_path, dpi=180)
    plt.close()

    print("\n10 fitur terpenting:")
    print(importance.head(10).to_string(index=False))

    print(f"\nCSV: {csv_path}")
    print(f"Grafik: {png_path}")

print("\nAnalisis SHAP selesai.")
