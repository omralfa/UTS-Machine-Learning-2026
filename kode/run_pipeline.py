import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATASETS = [
    "regresi_train.csv.gz",
    "regresi_test.csv.gz",
    "klasifikasi_train.csv.gz",
    "klasifikasi_test.csv.gz",
    "clustering.csv.gz",
]

STAGES = [
    ("Regresi", "prediksi_final_regresi.py"),
    ("Klasifikasi", "prediksi_final_klasifikasi.py"),
    ("Clustering", "eksperimen_dua_tahap.py"),
    ("Deteksi anomali", "ensemble_anomali.py"),
    ("Finalisasi clustering", "finalisasi_clustering.py"),
]

OUTPUTS = [
    "prediksi_regresi.csv",
    "prediksi_klasifikasi.csv",
    "cluster.csv",
]


def check_files():
    missing = []

    for filename in DATASETS:
        path = ROOT / "data" / filename
        if not path.is_file():
            missing.append(str(path))

    for _, filename in STAGES:
        path = ROOT / "kode" / filename
        if not path.is_file():
            missing.append(str(path))

    if missing:
        print("\nFILE TIDAK DITEMUKAN:")
        for path in missing:
            print(f"  - {path}")
        return False

    print("\nSemua dataset dan script utama tersedia.")
    return True


def run_pipeline():
    if not check_files():
        return 1

    results = []
    start_total = time.perf_counter()

    print("\n=== MENJALANKAN PIPELINE ML ===", flush=True)

    for name, filename in STAGES:
        script = ROOT / "kode" / filename

        print(f"\n>>> MULAI: {name}", flush=True)
        start = time.perf_counter()

        process = subprocess.run(
            [sys.executable, str(script)],
            cwd=ROOT,
            check=False,
        )

        duration = round(time.perf_counter() - start, 2)

        results.append({
            "tahap": name,
            "script": filename,
            "status": (
                "BERHASIL"
                if process.returncode == 0
                else "GAGAL"
            ),
            "durasi_detik": duration,
            "returncode": process.returncode,
        })

        if process.returncode != 0:
            print(
                f"\nPIPELINE BERHENTI: {name} gagal "
                f"(exit code {process.returncode})."
            )
            save_log(results)
            return process.returncode

        print(
            f">>> SELESAI: {name} ({duration:.2f} detik)",
            flush=True,
        )

    missing_outputs = [
        filename
        for filename in OUTPUTS
        if not (ROOT / filename).is_file()
    ]

    if missing_outputs:
        print("\nOUTPUT TIDAK DITEMUKAN:")
        for filename in missing_outputs:
            print(f"  - {filename}")
        save_log(results)
        return 1

    save_log(results)

    duration_total = time.perf_counter() - start_total

    print("\n=== PIPELINE SELESAI ===")
    print(f"Total durasi: {duration_total:.2f} detik")
    print("Seluruh file submission tersedia.")

    return 0


def save_log(results):
    log_dir = ROOT / "hasil"
    log_dir.mkdir(parents=True, exist_ok=True)

    path = log_dir / "pipeline_log.csv"

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "tahap",
                "script",
                "status",
                "durasi_detik",
                "returncode",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nLog disimpan: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline UTS Machine Learning 2026"
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help="Periksa dataset dan script tanpa menjalankan model",
    )

    parser.add_argument(
        "--run",
        action="store_true",
        help="Jalankan seluruh pipeline dari awal",
    )

    args = parser.parse_args()

    print("UTS Machine Learning 2026")
    print(f"Python: {sys.executable}")
    print(f"Root: {ROOT}")

    if args.check:
        return 0 if check_files() else 1

    if args.run:
        return run_pipeline()

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
