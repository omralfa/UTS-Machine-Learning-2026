from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score

OUT = Path("hasil/klasifikasi")
REVIEW_COST = 25000

models = {
    "LightGBM": OUT / "lightgbm_validation.csv",
    "CatBoost": OUT / "catboost_validation.csv"
}

bins = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.000001]

for name, path in models.items():
    df = pd.read_csv(path, parse_dates=["waktu"])

    print("\n" + "=" * 65)
    print(f"AUDIT PROBABILITAS: {name}")
    print("=" * 65)

    print("\n=== RATA-RATA PROBABILITAS ===")
    print(f"Prediksi rata-rata : {df['prob'].mean():.6%}")
    print(f"Fraud aktual       : {df['actual'].mean():.6%}")

    df["prob_bin"] = pd.cut(
        df["prob"],
        bins=bins,
        include_lowest=True
    )

    calibration = (
        df.groupby("prob_bin", observed=True)
        .agg(
            transaksi=("actual", "size"),
            fraud=("actual", "sum"),
            rata_prob=("prob", "mean"),
            fraud_rate=("actual", "mean")
        )
        .reset_index()
    )

    print("\n=== KALIBRASI PER KELOMPOK ===")
    print(calibration.to_string(index=False))

    calibration.to_csv(
        OUT / f"{name.lower()}_calibration_bins.csv",
        index=False
    )

    print("\n=== PERFORMA PER BULAN ===")

    df["bulan"] = df["waktu"].dt.to_period("M").astype(str)

    for bulan, group in df.groupby("bulan"):
        actual = group["actual"].to_numpy()
        prob = group["prob"].to_numpy()
        nominal = group["nominal"].to_numpy()
        tandai = group["tandai"].to_numpy()

        ap = average_precision_score(actual, prob)

        cost = (
            tandai.sum() * REVIEW_COST
            + nominal[(actual == 1) & (tandai == 0)].sum()
        )

        baseline = nominal[actual == 1].sum()

        print(
            f"{bulan} | "
            f"PR-AUC={ap:.6f} | "
            f"Review={tandai.sum():,} | "
            f"Biaya=Rp{cost:,.0f} | "
            f"Baseline=Rp{baseline:,.0f}"
        )

print("\nAudit probabilitas selesai.")
