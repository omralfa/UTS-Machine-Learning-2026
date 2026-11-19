from pathlib import Path
import json

import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
    log_loss,
    brier_score_loss
)

OUT = Path("hasil/klasifikasi")
OUT.mkdir(parents=True, exist_ok=True)

REVIEW_COST = 25000

df = pd.read_csv("data/klasifikasi_train.csv.gz")
df["waktu"] = pd.to_datetime(df["waktu"])

train = df[df["waktu"] < "2025-07-01"].copy()
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
X_valid = valid[features].copy()

y_train = train["is_fraud"]
y_valid = valid["is_fraud"]

categorical = list(
    X_train.select_dtypes(
        include=["object", "category"]
    ).columns
)

for col in categorical:
    categories = pd.Index(
        X_train[col].fillna("__MISSING__").astype(str).unique()
    ).union(pd.Index(["__UNKNOWN__"]))

    train_values = (
        X_train[col].fillna("__MISSING__").astype(str)
    )

    valid_values = (
        X_valid[col].fillna("__MISSING__").astype(str)
    )

    X_train[col] = pd.Categorical(
        train_values,
        categories=categories
    )

    X_valid[col] = pd.Categorical(
        valid_values.where(
            valid_values.isin(categories),
            "__UNKNOWN__"
        ),
        categories=categories
    )

print("=" * 60)
print("LIGHTGBM FRAUD CLASSIFICATION")
print("=" * 60)

print("Training:", len(train))
print("Validation:", len(valid))
print("Jumlah fitur:", len(features))
print("Jumlah fitur kategorikal:", len(categorical))

model = LGBMClassifier(
    objective="binary",
    n_estimators=500,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=100,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)

print("\nMelatih LightGBM...", flush=True)

model.fit(
    X_train,
    y_train,
    categorical_feature=categorical
)

prob = model.predict_proba(X_valid)[:, 1]

ap = average_precision_score(y_valid, prob)
roc = roc_auc_score(y_valid, prob)
brier = brier_score_loss(y_valid, prob)
ll = log_loss(y_valid, prob)

# Aturan keputusan berbasis expected cost
nominal = valid["nominal"].to_numpy()
actual = y_valid.to_numpy()

tandai = (
    prob * nominal > REVIEW_COST
).astype(int)

review_count = int(tandai.sum())

review_cost = review_count * REVIEW_COST

missed_fraud = (
    (actual == 1) & (tandai == 0)
)

fraud_loss = float(
    nominal[missed_fraud].sum()
)

total_cost = review_cost + fraud_loss

precision = precision_score(
    actual, tandai, zero_division=0
)

recall = recall_score(
    actual, tandai, zero_division=0
)

cm = confusion_matrix(actual, tandai)

print("\n=== METRIK MODEL ===")
print(f"PR-AUC        : {ap:.6f}")
print(f"ROC-AUC       : {roc:.6f}")
print(f"Brier Score   : {brier:.6f}")
print(f"Log Loss      : {ll:.6f}")

print("\n=== KEPUTUSAN BERBASIS BIAYA ===")
print(f"Transaksi ditandai : {review_count:,}")
print(f"Precision          : {precision:.6f}")
print(f"Recall             : {recall:.6f}")

print("\nConfusion matrix:")
print(cm)

print("\n=== TOTAL BIAYA ===")
print(f"Biaya review : Rp {review_cost:,.0f}")
print(f"Fraud lolos  : Rp {fraud_loss:,.0f}")
print(f"TOTAL        : Rp {total_cost:,.0f}")

baseline_cost = float(
    valid.loc[
        valid["is_fraud"] == 1,
        "nominal"
    ].sum()
)

saving = baseline_cost - total_cost

print(f"Baseline     : Rp {baseline_cost:,.0f}")
print(f"Penghematan  : Rp {saving:,.0f}")

results = {
    "model": "LightGBM",
    "train_rows": len(train),
    "validation_rows": len(valid),
    "n_features": len(features),
    "pr_auc": float(ap),
    "roc_auc": float(roc),
    "brier_score": float(brier),
    "log_loss": float(ll),
    "review_count": review_count,
    "precision": float(precision),
    "recall": float(recall),
    "review_cost": float(review_cost),
    "missed_fraud_loss": float(fraud_loss),
    "total_cost": float(total_cost),
    "baseline_cost": baseline_cost,
    "saving": float(saving)
}

with open(
    OUT / "lightgbm_metrics.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(results, file, indent=2)

validation = pd.DataFrame({
    "id_transaksi": valid["id_transaksi"].to_numpy(),
    "waktu": valid["waktu"].to_numpy(),
    "nominal": nominal,
    "actual": actual,
    "prob": prob,
    "tandai": tandai
})

validation.to_csv(
    OUT / "lightgbm_validation.csv",
    index=False
)

print("\nTraining LightGBM selesai.")
