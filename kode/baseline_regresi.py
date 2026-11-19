from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path("data/regresi_train.csv.gz")
OUTPUT_DIR = Path("hasil/regresi")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

df["tgl_pickup"] = pd.to_datetime(df["tgl_pickup"])

train = df[df["tgl_pickup"] < "2025-07-01"].copy()

valid = df[
    (df["tgl_pickup"] >= "2025-07-01")
    & (df["tgl_pickup"] < "2025-09-01")
].copy()

assert len(train) > 0
assert len(valid) > 0

print("=" * 60)
print("BASELINE REGRESSION - TIME BASED VALIDATION")
print("=" * 60)

print("Training:", len(train))
print("Validation:", len(valid))

y_train = train["durasi_jam"].to_numpy()
y_valid = valid["durasi_jam"].to_numpy()

p50 = np.quantile(y_train, 0.50)
p90 = np.quantile(y_train, 0.90)

pred_p50 = np.full(len(valid), p50)
pred_p90 = np.full(len(valid), p90)


def pinball_loss(y_true, y_pred, quantile):
    error = y_true - y_pred

    return np.mean(
        np.maximum(
            quantile * error,
            (quantile - 1) * error
        )
    )


loss_p50 = pinball_loss(
    y_valid,
    pred_p50,
    0.50
)

loss_p90 = pinball_loss(
    y_valid,
    pred_p90,
    0.90
)

coverage_p90 = np.mean(
    y_valid <= pred_p90
)

print("\nBASELINE RESULTS")
print(f"P50 prediction : {p50:.3f} jam")
print(f"P90 prediction : {p90:.3f} jam")
print(f"Pinball P50    : {loss_p50:.4f}")
print(f"Pinball P90    : {loss_p90:.4f}")
print(f"P90 coverage   : {coverage_p90:.2%}")

results = pd.DataFrame({
    "metric": [
        "pinball_p50",
        "pinball_p90",
        "coverage_p90"
    ],
    "value": [
        loss_p50,
        loss_p90,
        coverage_p90
    ]
})

results.to_csv(
    OUTPUT_DIR / "baseline_metrics.csv",
    index=False
)

print("\nHasil disimpan ke hasil/regresi/baseline_metrics.csv")