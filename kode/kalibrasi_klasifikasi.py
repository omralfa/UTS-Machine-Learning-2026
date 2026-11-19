from pathlib import Path
import json

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss
)
from scipy.special import logit

OUT = Path("hasil/klasifikasi")
OUT.mkdir(parents=True, exist_ok=True)

REVIEW_COST = 25000

df = pd.read_csv("data/klasifikasi_train.csv.gz")
df["waktu"] = pd.to_datetime(df["waktu"])

train = df[df["waktu"] < "2025-06-01"].copy()

cal = df[
    (df["waktu"] >= "2025-06-01") &
    (df["waktu"] < "2025-07-01")
].copy()

valid = df[df["waktu"] >= "2025-07-01"].copy()

excluded = {
    "id_transaksi",
    "id_pengguna",
    "id_merchant",
    "waktu",
    "is_fraud",
    "flag_review_manual"
}

features = [
    col for col in df.columns
    if col not in excluded
]

X_train = train[features].copy()
X_cal = cal[features].copy()
X_valid = valid[features].copy()

categorical = list(
    X_train.select_dtypes(
        include=["object", "category"]
    ).columns
)

numeric = list(
    X_train.select_dtypes(
        include="number"
    ).columns
)

for col in categorical:
    for X in [X_train, X_cal, X_valid]:
        X[col] = X[col].fillna("__MISSING__").astype(str)

medians = X_train[numeric].median()

for X in [X_train, X_cal, X_valid]:
    X[numeric] = X[numeric].fillna(medians)

print("=" * 60)
print("KALIBRASI CATBOOST - PLATT SCALING")
print("=" * 60)

print("Training:", len(train))
print("Calibration:", len(cal))
print("Validation:", len(valid))

model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.03,
    loss_function="Logloss",
    random_seed=42,
    thread_count=-1,
    verbose=False,
    allow_writing_files=False
)

print("\nMelatih CatBoost Januari-Mei...", flush=True)

model.fit(
    X_train,
    train["is_fraud"],
    cat_features=categorical
)

prob_cal = model.predict_proba(X_cal)[:, 1]
prob_valid = model.predict_proba(X_valid)[:, 1]

eps = 1e-8

score_cal = logit(
    np.clip(prob_cal, eps, 1 - eps)
).reshape(-1, 1)

score_valid = logit(
    np.clip(prob_valid, eps, 1 - eps)
).reshape(-1, 1)

calibrator = LogisticRegression(
    max_iter=1000
)

calibrator.fit(
    score_cal,
    cal["is_fraud"]
)

prob_calibrated = calibrator.predict_proba(
    score_valid
)[:, 1]

actual = valid["is_fraud"].to_numpy()
nominal = valid["nominal"].to_numpy()

def evaluate(name, prob):
    tandai = (
        prob * nominal > REVIEW_COST
    ).astype(int)

    review_cost = tandai.sum() * REVIEW_COST

    fraud_loss = nominal[
        (actual == 1) & (tandai == 0)
    ].sum()

    total_cost = review_cost + fraud_loss

    return {
        "name": name,
        "pr_auc": float(
            average_precision_score(actual, prob)
        ),
        "brier": float(
            brier_score_loss(actual, prob)
        ),
        "log_loss": float(
            log_loss(actual, prob)
        ),
        "review_count": int(tandai.sum()),
        "review_cost": float(review_cost),
        "fraud_loss": float(fraud_loss),
        "total_cost": float(total_cost)
    }

raw = evaluate("CatBoost Raw", prob_valid)
calibrated = evaluate(
    "CatBoost Platt",
    prob_calibrated
)

for result in [raw, calibrated]:
    print("\n===", result["name"], "===")
    print(f"PR-AUC    : {result['pr_auc']:.6f}")
    print(f"Brier     : {result['brier']:.6f}")
    print(f"Log Loss  : {result['log_loss']:.6f}")
    print(f"Review    : {result['review_count']:,}")
    print(f"Total Cost: Rp {result['total_cost']:,.0f}")

with open(
    OUT / "kalibrasi_metrics.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        {
            "raw": raw,
            "calibrated": calibrated,
            "calibrator_coefficient": float(
                calibrator.coef_[0][0]
            ),
            "calibrator_intercept": float(
                calibrator.intercept_[0]
            )
        },
        f,
        indent=2
    )

print("\nEksperimen kalibrasi selesai.")
