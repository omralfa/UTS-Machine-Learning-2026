from pathlib import Path
import json

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_pinball_loss

OUT = Path("hasil/regresi")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("data/regresi_train.csv.gz")
df["tgl_pickup"] = pd.to_datetime(df["tgl_pickup"])

train = df[df["tgl_pickup"] < "2025-06-01"].copy()
calib = df[
    (df["tgl_pickup"] >= "2025-06-01")
    & (df["tgl_pickup"] < "2025-07-01")
].copy()
valid = df[
    (df["tgl_pickup"] >= "2025-07-01")
    & (df["tgl_pickup"] < "2025-09-01")
].copy()

assert min(len(train), len(calib), len(valid)) > 0

excluded = [
    "id_paket", "durasi_jam", "jumlah_scan", "tgl_pickup"
]

features = [c for c in df.columns if c not in excluded]

X_train = train[features].copy()
X_calib = calib[features].copy()
X_valid = valid[features].copy()

categorical = list(
    X_train.select_dtypes(include=["object", "category"]).columns
)

numeric = list(
    X_train.select_dtypes(include="number").columns
)

for frame in [X_train, X_calib, X_valid]:
    for col in categorical:
        frame[col] = frame[col].fillna("__MISSING__").astype(str)

medians = X_train[numeric].median()

for frame in [X_train, X_calib, X_valid]:
    frame[numeric] = frame[numeric].fillna(medians)

y_train = train["durasi_jam"].to_numpy()
y_calib = calib["durasi_jam"].to_numpy()
y_valid = valid["durasi_jam"].to_numpy()

predictions = {}

for q in [0.5, 0.9]:
    print(f"Training CatBoost q={q}", flush=True)

    model = CatBoostRegressor(
        loss_function=f"Quantile:alpha={q}",
        iterations=400,
        depth=6,
        learning_rate=0.05,
        random_seed=42,
        thread_count=-1,
        verbose=False,
        allow_writing_files=False
    )

    model.fit(
        X_train,
        y_train,
        cat_features=categorical
    )

    predictions[q] = {
        "calib": model.predict(X_calib),
        "valid": model.predict(X_valid)
    }

# Residual P90 pada data kalibrasi
residual = y_calib - predictions[0.9]["calib"]

# Kuantil empiris konservatif untuk target coverage 90%
n = len(residual)
level = min(1.0, np.ceil((n + 1) * 0.9) / n)

correction = np.quantile(
    residual,
    level,
    method="higher"
)

p50 = predictions[0.5]["valid"]
p90_raw = np.maximum(predictions[0.9]["valid"], p50)
p90_calibrated = np.maximum(
    predictions[0.9]["valid"] + correction,
    p50
)

metrics = {
    "n_train": len(train),
    "n_calibration": len(calib),
    "n_validation": len(valid),
    "correction_hours": float(correction),
    "pinball_p50": float(
        mean_pinball_loss(y_valid, p50, alpha=0.5)
    ),
    "pinball_p90_raw": float(
        mean_pinball_loss(y_valid, p90_raw, alpha=0.9)
    ),
    "pinball_p90_calibrated": float(
        mean_pinball_loss(y_valid, p90_calibrated, alpha=0.9)
    ),
    "coverage_p90_raw": float(
        np.mean(y_valid <= p90_raw)
    ),
    "coverage_p90_calibrated": float(
        np.mean(y_valid <= p90_calibrated)
    )
}

print("\n=== HASIL KALIBRASI ===")
print(json.dumps(metrics, indent=2))

with open(OUT / "kalibrasi_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

pd.DataFrame({
    "id_paket": valid["id_paket"].to_numpy(),
    "actual": y_valid,
    "p50": p50,
    "p90_raw": p90_raw,
    "p90_calibrated": p90_calibrated
}).to_csv(
    OUT / "kalibrasi_validation.csv",
    index=False
)

print("\nKalibrasi selesai.")