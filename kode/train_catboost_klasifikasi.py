from pathlib import Path
import json

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
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

numeric = list(
    X_train.select_dtypes(
        include="number"
    ).columns
)

# CatBoost memerlukan kategori dalam bentuk string
for col in categorical:
    X_train[col] = (
        X_train[col]
        .fillna("__MISSING__")
        .astype(str)
    )

    X_valid[col] = (
        X_valid[col]
        .fillna("__MISSING__")
        .astype(str)
    )

# Median dihitung dari training saja
medians = X_train[numeric].median()

X_train[numeric] = X_train[numeric].fillna(medians)
X_valid[numeric] = X_valid[numeric].fillna(medians)

print("=" * 60)
print("CATBOOST FRAUD CLASSIFICATION")
print("=" * 60)

print("Training:", len(train))
print("Validation:", len(valid))
print("Jumlah fitur:", len(features))
print("Jumlah kategorikal:", len(categorical))

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

print("\nMelatih CatBoost...", flush=True)

model.fit(
    X_train,
    y_train,
    cat_features=categorical
)

prob = model.predict_proba(X_valid)[:, 1]

actual = y_valid.to_numpy()
nominal = valid["nominal"].to_numpy()

ap = average_precision_score(actual, prob)
roc = roc_auc_score(actual, prob)
brier = brier_score_loss(actual, prob)
ll = log_loss(actual, prob)

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

baseline_cost = float(
    valid.loc[
        valid["is_fraud"] == 1,
        "nominal"
    ].sum()
)

saving = baseline_cost - total_cost

print("\n=== METRIK MODEL ===")
print(f"PR-AUC      : {ap:.6f}")
print(f"ROC-AUC     : {roc:.6f}")
print(f"Brier Score : {brier:.6f}")
print(f"Log Loss    : {ll:.6f}")

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
print(f"Baseline     : Rp {baseline_cost:,.0f}")
print(f"Penghematan  : Rp {saving:,.0f}")

results = {
    "model": "CatBoost",
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
    OUT / "catboost_metrics.json",
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
    OUT / "catboost_validation.csv",
    index=False
)

print("\nTraining CatBoost selesai.")
