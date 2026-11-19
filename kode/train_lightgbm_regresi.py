from pathlib import Path
import json

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_pinball_loss

SEED = 42
OUT = Path("hasil/regresi")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("data/regresi_train.csv.gz")
df["tgl_pickup"] = pd.to_datetime(df["tgl_pickup"])

train = df[df["tgl_pickup"] < "2025-07-01"].copy()
valid = df[df["tgl_pickup"] >= "2025-07-01"].copy()

# ID, target, dan fitur yang berpotensi leakage
excluded = ["id_paket", "durasi_jam", "jumlah_scan", "tgl_pickup"]

features = [c for c in df.columns if c not in excluded]

X_train = train[features].copy()
X_valid = valid[features].copy()

# Penanganan kategori, termasuk kategori baru
categorical = X_train.select_dtypes(include=["object", "category"]).columns

for col in categorical:
    X_train[col] = X_train[col].fillna("__MISSING__").astype(str)
    X_valid[col] = X_valid[col].fillna("__MISSING__").astype(str)

    categories = pd.Index(
        X_train[col].unique()
    ).union(pd.Index(["__UNKNOWN__"]))

    X_train[col] = pd.Categorical(
        X_train[col],
        categories=categories
    )

    X_valid[col] = pd.Categorical(
        X_valid[col].where(
            X_valid[col].isin(categories),
            "__UNKNOWN__"
        ),
        categories=categories
    )

y_train = train["durasi_jam"]
y_valid = valid["durasi_jam"]

results = {}
predictions = {}

for q in [0.5, 0.9]:
    print(f"\nTraining LightGBM Quantile q={q}")

    model = LGBMRegressor(
        objective="quantile",
        alpha=q,
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=40,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1
    )

    model.fit(
        X_train,
        y_train,
        categorical_feature=list(categorical)
    )

    pred = model.predict(X_valid)
    predictions[q] = pred

    loss = mean_pinball_loss(
        y_valid,
        pred,
        alpha=q
    )

    results[f"pinball_{q}"] = float(loss)

    print(f"Pinball loss q={q}: {loss:.4f}")

# Pastikan P90 tidak lebih kecil dari P50
p50 = predictions[0.5]
p90 = np.maximum(predictions[0.9], p50)

results["pinball_0.9"] = float(
    mean_pinball_loss(y_valid, p90, alpha=0.9)
)

results["coverage_p90"] = float(
    np.mean(y_valid.to_numpy() <= p90)
)

print("\n=== HASIL LIGHTGBM ===")
print(json.dumps(results, indent=2))

with open(OUT / "lightgbm_metrics.json", "w") as f:
    json.dump(results, f, indent=2)

pd.DataFrame({
    "id_paket": valid["id_paket"],
    "actual": y_valid,
    "p50": p50,
    "p90": p90
}).to_csv(
    OUT / "lightgbm_validation.csv",
    index=False
)

print("\nHasil tersimpan di hasil/regresi/")