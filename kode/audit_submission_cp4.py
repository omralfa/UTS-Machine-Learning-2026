from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

FILES = {
    "regresi": ("prediksi_regresi.csv", 60000),
    "klasifikasi": ("prediksi_klasifikasi.csv", 106674),
    "clustering": ("cluster.csv", 120000),
}

for task, (filename, expected_rows) in FILES.items():
    path = ROOT / filename
    df = pd.read_csv(path)

    print(f"\n=== {task.upper()} ===")
    print("File:", filename)
    print("Jumlah baris:", len(df))
    print("Kolom:", list(df.columns))
    print("Missing values:", int(df.isna().sum().sum()))
    print("Duplikat ID:", int(df.iloc[:, 0].duplicated().sum()))

    assert len(df) == expected_rows, (
        f"Jumlah baris tidak sesuai: {filename}"
    )
    assert df.isna().sum().sum() == 0, (
        f"Missing value ditemukan: {filename}"
    )
    assert not df.iloc[:, 0].duplicated().any(), (
        f"ID duplikat ditemukan: {filename}"
    )

    if task == "regresi":
        assert (df["p50"] <= df["p90"]).all(), (
            "Ditemukan p50 > p90"
        )

    if task == "klasifikasi":
        assert df["prob"].between(0, 1).all(), (
            "Probabilitas di luar rentang [0, 1]"
        )
        assert df["tandai"].isin([0, 1]).all(), (
            "Nilai tandai tidak valid"
        )

    if task == "clustering":
        assert df["cluster"].isin([-1, 0, 1, 2, 3]).all(), (
            "Label cluster tidak valid"
        )

    print("STATUS: LULUS")

print("\nSELURUH SUBMISSION LULUS AUDIT DASAR")
