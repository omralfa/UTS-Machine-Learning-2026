from pathlib import Path
import json

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_pinball_loss

SEED = 42
OUT = Path("hasil/regresi")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("data/regresi_train.csv.gz")
df["tgl_pickup"] = pd.to_datetime(df["tgl_pickup"])

train = df[df["tgl_pickup"] < "2025-07-01"].copy()
valid = df[df["tgl_pickup"] >= "2025-07-01"].copy()

excluded = [
    "id_paket",
    "durasi_jam",
    "jumlah_scan",
    "tgl_pickup"
]

features = [c for c in df.columns if c not in excluded]

X_train = train[features].copy()
X_valid = valid[features].copy()

categorical = list(
    X_train.select_dtypes(
        include=["object", "category"]
    ).columns
)

for col in categorical:
    X_train[col] = X_train[col].fillna("__MISSING__").astype(str)
    X_valid[col] = X_valid[col].fillna("__MISSING__").astype(str)

numeric = X_train.select_dtypes(include="number").columns

# Median dihitung hanya menggunakan training
medians = X_train[numeric].median()

X_train[numeric] = X_train[numeric].fillna(medians)
X_valid[numeric] = X_valid[numeric].fillna(medians)

y_train = train["durasi_jam"]
y_valid = valid["durasi_jam"]

results = {}
predictions = {}

for q in [0.5, 0.9]:

    print(f"\nTraining CatBoost Quantile q={q}", flush=True)

    model = CatBoostRegressor(
        loss_function=f"Quantile:alpha={q}",
        iterations=400,
        depth=6,
        learning_rate=0.05,
        random_seed=SEED,
        thread_count=-1,
        verbose=False,
        allow_writing_files=False
    )

    model.fit(
        X_train,
        y_train,
        cat_features=categorical
    )

    pred = model.predict(X_valid)
    predictions[q] = pred

    loss = mean_pinball_loss(
        y_valid,
        pred,
        alpha=q
    )

    results[f"pinball_{q}"] = float(loss)

    print(f"Pinball loss q={q}: {loss:.4f}", flush=True)

p50 = predictions[0.5]
p90 = np.maximum(predictions[0.9], p50)

results["pinball_0.9"] = float(
    mean_pinball_loss(y_valid, p90, alpha=0.9)
)

results["coverage_p90"] = float(
    np.mean(y_valid.to_numpy() <= p90)
)

print("\n=== HASIL CATBOOST ===")
print(json.dumps(results, indent=2))

with open(OUT / "catboost_metrics.json", "w") as f:
    json.dump(results, f, indent=2)

pd.DataFrame({
    "id_paket": valid["id_paket"],
    "actual": y_valid,
    "p50": p50,
    "p90": p90
}).to_csv(
    OUT / "catboost_validation.csv",
    index=False
)

print("\nHasil tersimpan di hasil/regresi/")