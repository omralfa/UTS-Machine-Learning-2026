from pathlib import Path
import json

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_pinball_loss

SEED = 42
DATA_DIR = Path("data")
OUT_DIR = Path("hasil/regresi")
OUT_DIR.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(DATA_DIR / "regresi_train.csv.gz")
test = pd.read_csv(DATA_DIR / "regresi_test.csv.gz")

print("=" * 60)
print("FINAL REGRESSION - CATBOOST QUANTILE")
print("=" * 60)

print("Training:", train.shape)
print("Testing :", test.shape)

assert train["id_paket"].is_unique
assert test["id_paket"].is_unique
assert "durasi_jam" in train.columns
assert "durasi_jam" not in test.columns
assert len(test) == 60000

# Fitur yang tidak digunakan untuk prediksi
excluded = {
    "id_paket",
    "tgl_pickup",
    "durasi_jam",
    "jumlah_scan"
}

features = [
    col for col in train.columns
    if col not in excluded
]

assert set(features).issubset(test.columns)

X_train = train[features].copy()
X_test = test[features].copy()

y_train = train["durasi_jam"]

categorical = list(
    X_train.select_dtypes(
        include=["object", "category"]
    ).columns
)

numeric = list(
    X_train.select_dtypes(include="number").columns
)

# Preprocessing kategori
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

# Median dihitung dari training saja
medians = X_train[numeric].median()

X_train[numeric] = X_train[numeric].fillna(medians)
X_test[numeric] = X_test[numeric].fillna(medians)

assert not X_train.isna().any().any()
assert not X_test.isna().any().any()

predictions = {}

for q in [0.5, 0.9]:
    print(f"\nTraining CatBoost q={q}", flush=True)

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

    predictions[q] = model.predict(X_test)

    model.save_model(
        str(OUT_DIR / f"catboost_p{int(q * 100)}.cbm")
    )

# Jaga agar P90 >= P50 dan durasi tidak negatif
p50 = np.maximum(predictions[0.5], 0)
p90 = np.maximum(predictions[0.9], p50)

submission = pd.DataFrame({
    "id_paket": test["id_paket"],
    "p50": p50,
    "p90": p90
})

# Pemeriksaan akhir
assert list(submission.columns) == [
    "id_paket", "p50", "p90"
]

assert len(submission) == 60000
assert submission["id_paket"].is_unique
assert submission.notna().all().all()
assert np.isfinite(submission[["p50", "p90"]]).all().all()
assert (submission["p50"] >= 0).all()
assert (submission["p90"] >= submission["p50"]).all()

# Simpan di root proyek sesuai format pengumpulan
output_path = Path("prediksi_regresi.csv")
submission.to_csv(output_path, index=False)

# Verifikasi dengan membaca ulang CSV
check = pd.read_csv(output_path)

assert check.shape == (60000, 3)
assert check["id_paket"].equals(test["id_paket"])
assert check.notna().all().all()
assert (check["p90"] >= check["p50"]).all()

metadata = {
    "model": "CatBoost Quantile",
    "training_rows": len(train),
    "test_rows": len(test),
    "n_features": len(features),
    "excluded_features": sorted(excluded),
    "calibration": "none",
    "random_seed": SEED,
    "output": str(output_path)
}

with open(
    OUT_DIR / "final_metadata.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(metadata, file, indent=2)

print("\n=== FINAL REGRESSION SELESAI ===")
print("Jumlah prediksi:", len(submission))
print("Jumlah fitur:", len(features))

print("\nStatistik prediksi:")
print(
    submission[["p50", "p90"]]
    .describe()
    .to_string()
)

print("\nContoh hasil:")
print(submission.head(5).to_string(index=False))

print("\nFile:", output_path.resolve())
print("Validasi format: LULUS")