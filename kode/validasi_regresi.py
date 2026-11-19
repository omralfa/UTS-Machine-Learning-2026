from pathlib import Path

import json
import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_pinball_loss

OUT = Path("hasil/regresi")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("data/regresi_train.csv.gz")
df["tgl_pickup"] = pd.to_datetime(df["tgl_pickup"])

excluded = [
    "id_paket",
    "tgl_pickup",
    "durasi_jam",
    "jumlah_scan"
]

features = [c for c in df.columns if c not in excluded]

categorical = list(
    df[features].select_dtypes(
        include=["object", "category"]
    ).columns
)

def evaluate(train, valid, name):
    X_train = train[features].copy()
    X_valid = valid[features].copy()

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

    result = {"split": name}

    for q in [0.5, 0.9]:
        model = LGBMRegressor(
            objective="quantile",
            alpha=q,
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=40,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )

        model.fit(
            X_train,
            y_train,
            categorical_feature=categorical
        )

        pred = model.predict(X_valid)

        result[f"pinball_{q}"] = float(
            mean_pinball_loss(
                y_valid,
                pred,
                alpha=q
            )
        )

        if q == 0.9:
            result["coverage_p90"] = float(
                np.mean(y_valid.to_numpy() <= pred)
            )

    return result

temporal_train = df[
    df["tgl_pickup"] < "2025-07-01"
]

temporal_valid = df[
    df["tgl_pickup"] >= "2025-07-01"
]

random_train, random_valid = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    shuffle=True
)

results = []

print("Menjalankan temporal validation...", flush=True)
results.append(
    evaluate(temporal_train, temporal_valid, "temporal")
)

print("Menjalankan random validation...", flush=True)
results.append(
    evaluate(random_train, random_valid, "random")
)

print("\n=== PERBANDINGAN VALIDASI ===")
print(pd.DataFrame(results).to_string(index=False))

pd.DataFrame(results).to_csv(
    OUT / "perbandingan_validasi.csv",
    index=False
)

with open(OUT / "perbandingan_validasi.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nValidasi selesai.")