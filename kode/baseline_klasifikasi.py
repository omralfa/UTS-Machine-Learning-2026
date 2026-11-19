from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.metrics import average_precision_score

OUT = Path("hasil/klasifikasi")
OUT.mkdir(parents=True, exist_ok=True)

REVIEW_COST = 25000

df = pd.read_csv(
    "data/klasifikasi_train.csv.gz"
)

df["waktu"] = pd.to_datetime(df["waktu"])

train = df[
    df["waktu"] < "2025-07-01"
].copy()

valid = df[
    df["waktu"] >= "2025-07-01"
].copy()

print("=" * 60)
print("BASELINE FRAUD CLASSIFICATION")
print("=" * 60)

print("\n=== PEMBAGIAN DATA ===")
print("Training:", len(train))
print("Validation:", len(valid))

print("\n=== FRAUD RATE ===")

train_rate = train["is_fraud"].mean()
valid_rate = valid["is_fraud"].mean()

print(f"Training fraud rate  : {train_rate:.6%}")
print(f"Validation fraud rate: {valid_rate:.6%}")

# Baseline: probabilitas konstan
y_valid = valid["is_fraud"].to_numpy()

baseline_prob = np.full(
    len(valid),
    train_rate
)

baseline_ap = average_precision_score(
    y_valid,
    baseline_prob
)

# Kebijakan 1: tidak melakukan review
no_review_cost = (
    valid.loc[
        valid["is_fraud"] == 1,
        "nominal"
    ].sum()
)

# Kebijakan 2: semua transaksi direview
all_review_cost = (
    len(valid) * REVIEW_COST
)

# Jumlah fraud
fraud_count = int(
    valid["is_fraud"].sum()
)

fraud_nominal = float(
    valid.loc[
        valid["is_fraud"] == 1,
        "nominal"
    ].sum()
)

print("\n=== BASELINE METRIK ===")
print(f"PR-AUC baseline: {baseline_ap:.6f}")
print(f"Jumlah fraud validation: {fraud_count:,}")

print("\n=== BASELINE BIAYA ===")
print(f"Tanpa review : Rp {no_review_cost:,.0f}")
print(f"Review semua : Rp {all_review_cost:,.0f}")

if no_review_cost < all_review_cost:
    print("Kebijakan lebih murah: TANPA REVIEW")
else:
    print("Kebijakan lebih murah: REVIEW SEMUA")

results = {
    "train_rows": len(train),
    "validation_rows": len(valid),
    "train_fraud_rate": float(train_rate),
    "validation_fraud_rate": float(valid_rate),
    "baseline_pr_auc": float(baseline_ap),
    "validation_fraud_count": fraud_count,
    "validation_fraud_nominal": fraud_nominal,
    "cost_no_review": float(no_review_cost),
    "cost_review_all": float(all_review_cost),
    "review_cost_per_transaction": REVIEW_COST
}

with open(
    OUT / "baseline_metrics.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(results, file, indent=2)

print("\nBaseline klasifikasi selesai.")
