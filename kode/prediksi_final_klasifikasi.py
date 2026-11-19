from pathlib import Path
import json

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

OUT = Path("hasil/klasifikasi")
OUT.mkdir(parents=True, exist_ok=True)

REVIEW_COST = 25000

train = pd.read_csv("data/klasifikasi_train.csv.gz")
test = pd.read_csv("data/klasifikasi_test.csv.gz")

excluded = {
    "id_transaksi",
    "id_pengguna",
    "id_merchant",
    "waktu",
    "is_fraud",
    "flag_review_manual"
}

features = [
    col for col in train.columns
    if col not in excluded
]

assert len(features) == 79, (
    f"Jumlah fitur tidak sesuai: {len(features)}"
)

assert all(col in test.columns for col in features)

X_train = train[features].copy()
X_test = test[features].copy()

categorical = list(
    X_train.select_dtypes(
        include=["object", "category"]
    ).columns
)

numeric = [
    col for col in features
    if col not in categorical
]

for col in categorical:
    X_train[col] = (
        X_train[col]
        .fillna("__MISSING__")
        .astype(str)
    )

    X_test[col] = (
        X_test[col]
        .fillna("__MISSING__")
        .astype(str)
    )

medians = X_train[numeric].median()

X_train[numeric] = X_train[numeric].fillna(medians)
X_test[numeric] = X_test[numeric].fillna(medians)

assert not X_train.isna().any().any()
assert not X_test.isna().any().any()

print("=" * 60)
print("FINAL FRAUD CLASSIFICATION")
print("=" * 60)

print("Training:", len(train))
print("Testing :", len(test))
print("Fitur   :", len(features))

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

print("\nMelatih CatBoost final...", flush=True)

model.fit(
    X_train,
    train["is_fraud"],
    cat_features=categorical
)

print("Menghasilkan probabilitas...", flush=True)

prob = model.predict_proba(X_test)[:, 1]

nominal = test["nominal"].to_numpy()

tandai = (
    prob * nominal > REVIEW_COST
).astype(int)

prediksi = pd.DataFrame({
    "id_transaksi": test["id_transaksi"],
    "prob": prob,
    "tandai": tandai
})

# Validasi output
assert len(prediksi) == len(test)
assert prediksi.columns.tolist() == [
    "id_transaksi", "prob", "tandai"
]
assert not prediksi.isna().any().any()
assert prediksi["id_transaksi"].is_unique
assert prediksi["prob"].between(0, 1).all()
assert prediksi["tandai"].isin([0, 1]).all()
assert np.isfinite(prediksi["prob"]).all()

assert (
    prediksi["id_transaksi"].tolist()
    == test["id_transaksi"].tolist()
)

prediksi.to_csv(
    "prediksi_klasifikasi.csv",
    index=False
)

model.save_model(
    str(OUT / "catboost_final.cbm")
)

metadata = {
    "model": "CatBoostClassifier",
    "train_rows": len(train),
    "test_rows": len(test),
    "features": features,
    "categorical_features": categorical,
    "numeric_medians": {
        key: float(value)
        for key, value in medians.items()
    },
    "review_cost": REVIEW_COST,
    "decision_rule": "prob * nominal > 25000",
    "calibration": "none",
    "random_seed": 42,
    "iterations": 500,
    "depth": 6,
    "learning_rate": 0.03
}

with open(
    OUT / "final_metadata.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(metadata, file, indent=2)

print("\n=== HASIL FINAL ===")
print("Jumlah prediksi:", len(prediksi))
print("Jumlah review  :", int(tandai.sum()))
print("Persentase review:", f"{tandai.mean():.4%}")
print("Rata-rata probabilitas:", f"{prob.mean():.4%}")

print("\n=== CONTOH OUTPUT ===")
print(prediksi.head(5).to_string(index=False))

print("\nValidasi format: LULUS")
print("File: prediksi_klasifikasi.csv")
print("Model: hasil/klasifikasi/catboost_final.cbm")
print("Metadata: hasil/klasifikasi/final_metadata.json")
