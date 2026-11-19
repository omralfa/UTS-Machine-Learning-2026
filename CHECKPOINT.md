# CHECKPOINT UTS MACHINE LEARNING

Tanggal: 17 November 2026

## STATUS

TASK 1 - REGRESSION:
Eksperimen dan prediksi final selesai.
Dokumentasi laporan PDF dan pengumpulan GitHub belum selesai.

TASK 2 - CLASSIFICATION:
Belum dimulai.

TASK 3 - CLUSTERING:
Belum dimulai.

## DATASET TASK 1

Training: 250000 baris
Testing: 60000 baris
Target: durasi_jam
Periode training: Januari-Agustus 2025
Periode testing: September-Oktober 2025

## HASIL VALIDASI JULI-AGUSTUS

Baseline:
- Pinball P50: 17.3965
- Pinball P90: 16.0098
- Coverage P90: 89.20%

LightGBM:
- Pinball P50: 10.7653
- Pinball P90: 11.3618
- Coverage P90: 87.46%

CatBoost:
- Pinball P50: 10.7328
- Pinball P90: 11.2604
- Coverage P90: 88.82%

## KEPUTUSAN

Model final: CatBoost Quantile.
Jumlah fitur: 73.
jumlah_scan dikecualikan karena risiko data leakage.
Kalibrasi global P90 tidak digunakan.
Temporal validation menjadi evaluasi utama.

## TEMUAN PENTING

Terdapat temporal distribution shift pada mitra K07
mulai Juli 2025.

Coverage P90 K07:
- Juli: 59.98%
- Agustus: 62.79%

Random validation menghasilkan hasil lebih optimistis
dibandingkan temporal validation.

## OUTPUT FINAL

prediksi_regresi.csv
- 60000 baris
- Kolom: id_paket,p50,p90
- Tidak ada missing values
- P50 <= P90
- Validasi format lulus

## TUGAS BERIKUTNYA

1. Task 2: Fraud Classification.
2. Task 3: Customer Clustering.
3. Dokumentasi laporan.pdf.
4. Persiapan GitHub dan kontribusi.md.
5. Uji reproduksibilitas satu perintah.

## ENVIRONMENT

Windows + VS Code
Python 3.12
Virtual environment: .venv
Dependencies: requirements.txt

## CARA MEMBUKA PROYEK

1. Buka VS Code.
2. File > Open Folder.
3. Pilih folder UTS-Machine-Learning-2026.
4. Buka Terminal > New Terminal.
5. Pilih interpreter .venv/Scripts/python.exe.

Periksa environment:
.\.venv\Scripts\python.exe -c "import pandas, sklearn, lightgbm, catboost; print('Environment OK')"

Tidak perlu menginstal ulang library selama .venv masih tersedia.

---

# CHECKPOINT 2 — FRAUD CLASSIFICATION
Tanggal: 18 November 2026
Status: SELESAI

## DATASET
- Training: 403077 transaksi
- Testing: 106674 transaksi
- Target: is_fraud
- Fraud rate training keseluruhan: 0.6505%
- Temporal validation: Juli–Agustus 2025

## MODEL YANG DIUJI

LightGBM:
- PR-AUC: 0.520146
- ROC-AUC: 0.956468
- Total biaya validasi: Rp168992800

CatBoost:
- PR-AUC: 0.529694
- ROC-AUC: 0.967653
- Total biaya validasi: Rp132359300

Baseline tanpa review:
- Total biaya: Rp597604200

## KEPUTUSAN
- Model final: CatBoost Classifier
- Jumlah fitur: 79
- flag_review_manual dikecualikan karena risiko leakage
- id_transaksi, id_pengguna, id_merchant dan waktu mentah dikecualikan
- Platt scaling diuji tetapi tidak digunakan
- Aturan review: prob * nominal > 25000
- Temporal validation menjadi evaluasi utama

## TEMUAN PENTING
- Dataset sangat tidak seimbang
- PR-AUC CatBoost Juli: 0.760230
- PR-AUC CatBoost Agustus: 0.314739
- Terdapat indikasi temporal distribution shift
- Probabilitas tinggi menunjukkan overconfidence

## OUTPUT FINAL
prediksi_klasifikasi.csv
- 106674 baris
- Kolom: id_transaksi,prob,tandai
- Tidak ada missing values
- Probabilitas berada pada rentang 0 sampai 1
- tandai bernilai 0 atau 1
- Jumlah transaksi ditandai: 1420
- Validasi format lulus

## MODEL DAN METADATA
- hasil/klasifikasi/catboost_final.cbm
- hasil/klasifikasi/final_metadata.json
- hasil/klasifikasi/catboost_metrics.json

## BACKUP
BACKUP-CP2-2026-11-18.zip

## STATUS PROYEK
- CP1: SELESAI
- CP2: SELESAI
- CP3: BELUM — Customer Clustering
- CP4: BELUM — Integrasi teknis dan GitHub
- CP5: BELUM — Laporan akhir

## TUGAS BERIKUTNYA
Task 3: Customer Clustering.

---

# CHECKPOINT 3 — CLUSTERING DAN ANOMALI

Tanggal penyelesaian: 18 November 2026

Status: SELESAI — pemodelan dan validasi artefak.

## Dataset
- Sumber: data/clustering.csv.gz
- Jumlah pelanggan: 120.000
- Pelanggan bertransaksi: 113.264
- Pelanggan tanpa transaksi: 6.736

## Model clustering
- Metode: MiniBatchKMeans dua tahap.
- Tahap 1: pelanggan tanpa transaksi 90 hari dipisahkan sebagai cluster 3.
- Tahap 2: pelanggan bertransaksi dikelompokkan menggunakan K-Means dengan K=3.
- Preprocessing: transformasi log1p, winsorization persentil 1–99, StandardScaler.
- Random seed referensi: 42.
- Stabilitas rata-rata ARI: 0,814886.

## Interpretasi segmen
- Cluster 0: High-Value Active.
- Cluster 1: Regular Shopper.
- Cluster 2: Low-Engagement.
- Cluster 3: No Recent Purchase.

## Deteksi anomali
- Metode: Ensemble Isolation Forest.
- Jumlah model: 5.
- Random seed: 11, 22, 33, 42, 55.
- Jumlah estimator per model: 200.
- Skor gabungan: rata-rata skor anomali.
- Ambang: top 1% pelanggan bertransaksi.
- Jumlah anomali: 1.133.
- Label anomali: -1.

## Distribusi final
- Cluster -1: 1.133 pelanggan.
- Cluster 0: 41.787 pelanggan.
- Cluster 1: 43.128 pelanggan.
- Cluster 2: 27.216 pelanggan.
- Cluster 3: 6.736 pelanggan.
- Total: 120.000 pelanggan.

## File hasil
- cluster.csv
- hasil/clustering/metadata_final.json
- hasil/clustering/audit_cp3.json
- hasil/clustering/profil_dua_tahap.csv
- hasil/clustering/stabilitas_dua_tahap.csv
- hasil/clustering/ensemble_anomali_scores.csv

## Validasi
- Jumlah baris: 120.000.
- ID pelanggan unik: LULUS.
- Missing cluster: 0.
- Label valid: -1, 0, 1, 2, 3.
- Pelanggan tanpa transaksi tetap cluster 3: LULUS.
- Audit kelengkapan file: LULUS.

SHA-256 cluster.csv:
88291cb9aa25d35bc9a77412aedc4adb919223657c0b7cf0b8f7d3fa4e789d1e

## Keterbatasan
- Tidak tersedia ground-truth anomali.
- Ambang top 1% merupakan keputusan operasional.
- Kesepakatan ensemble bukan bukti akurasi deteksi.
- Reproduksibilitas end-to-end belum diuji dan akan dikerjakan pada CP 4.

## Status checkpoint
CP 1: SELESAI
CP 2: SELESAI
CP 3: SELESAI
CP 4: BELUM
CP 5: BELUM
