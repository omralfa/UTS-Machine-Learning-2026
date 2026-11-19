# UTS Machine Learning 2026

Proyek Ujian Tengah Semester Machine Learning 2026 yang mencakup tiga tugas: regresi kuantil, klasifikasi fraud, dan segmentasi pelanggan.

## Status Proyek

| Komponen | Status |
|---|---|
| Task 1 — Regresi | Selesai |
| Task 2 — Klasifikasi | Selesai |
| Task 3 — Clustering dan anomali | Selesai |
| Integrasi pipeline | Selesai |
| Reproduksibilitas lokal | Lulus |
| Laporan akhir | Dalam pengerjaan |

## Struktur Proyek

- `kode/` — Script pemodelan, eksperimen, evaluasi, audit, dan pipeline.
- `data/` — Dataset dari dosen (tidak disertakan dalam Git).
- `hasil/` — Model, metrik, log, dan artefak eksperimen.
- `prediksi_regresi.csv` — Submission regresi.
- `prediksi_klasifikasi.csv` — Submission klasifikasi.
- `cluster.csv` — Submission clustering.
- `requirements.txt` — Dependensi Python.
- `CHECKPOINT.md` — Riwayat pengerjaan.
- `kontribusi.md` — Kontribusi anggota dan penggunaan AI.

## Persiapan Lingkungan

Direkomendasikan Python 3.12.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Perintah pembuatan virtual environment hanya diperlukan saat menyiapkan proyek untuk pertama kali.

### Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Persiapan Dataset

Dataset diperoleh dari paket UTS yang disediakan dosen.

Letakkan lima file berikut langsung di folder `data/`:

```text
data/
├── regresi_train.csv.gz
├── regresi_test.csv.gz
├── klasifikasi_train.csv.gz
├── klasifikasi_test.csv.gz
└── clustering.csv.gz
```

Dataset tidak disertakan dalam repository untuk menghindari distribusi tanpa izin.

## Menjalankan Pipeline

Seluruh perintah dijalankan dari direktori utama proyek.

Periksa ketersediaan dataset dan script:

```bash
python kode/run_pipeline.py --check
```

Jalankan seluruh pipeline:

```bash
python kode/run_pipeline.py --run
```

Pipeline menjalankan lima tahap secara berurutan:

1. Training dan prediksi regresi CatBoost.
2. Training dan prediksi klasifikasi CatBoost.
3. Clustering dua tahap.
4. Ensemble Isolation Forest.
5. Finalisasi hasil clustering.

Log eksekusi disimpan pada `hasil/pipeline_log.csv`.

## File Submission

| File | Jumlah baris |
|---|---:|
| `prediksi_regresi.csv` | 60.000 |
| `prediksi_klasifikasi.csv` | 106.674 |
| `cluster.csv` | 120.000 |

Untuk menjalankan audit:

```bash
python kode/audit_submission_cp4.py
```

Audit memeriksa jumlah baris, missing value, duplikasi ID, rentang probabilitas, urutan kuantil, dan label cluster.

## Metodologi

### Task 1 — Regresi

- Algoritma: CatBoost Quantile Regression.
- Target: P50 dan P90 durasi pengiriman.
- Jumlah fitur: 73.
- Validasi: pemisahan berdasarkan waktu.

### Task 2 — Klasifikasi

- Algoritma: CatBoost Classifier.
- Jumlah fitur: 79.
- Evaluasi: PR-AUC dan biaya keputusan.
- Output: probabilitas fraud dan penanda review.

### Task 3 — Clustering

- Algoritma: MiniBatchKMeans dua tahap.
- Pelanggan tanpa transaksi dipisahkan dari pelanggan aktif.
- Deteksi anomali menggunakan ensemble lima Isolation Forest.
- Kandidat anomali ditetapkan dari 1% pelanggan aktif dengan skor tertinggi.

## Reproduksibilitas

Pipeline telah berhasil dijalankan ulang pada lingkungan pengembangan Windows menggunakan Python 3.12.10.

Hasil pengujian:

- Kelima tahap berhasil dijalankan.
- Total waktu eksekusi: 310,72 detik.
- Ketiga file submission identik secara SHA-256 dengan baseline sebelumnya.

Pengujian pada komputer lain atau versi dependensi berbeda belum dilakukan.

## Dokumentasi dan Penggunaan AI

Dokumentasi proses pengerjaan tersedia dalam `CHECKPOINT.md`.

AI digunakan untuk membantu perencanaan eksperimen, penyusunan code awal dasar dan audit kode yaang salah , audit hasil, serta dokumentasi. Seluruh pipeline dijalankan dan diverifikasi pada lingkungan pengembangan proyek.

Rincian kontribusi anggota dicatat dalam `kontribusi.md`.

## Catatan

Dataset dan model lokal tidak disertakan dalam repository Git. Pengguna harus menyediakan dataset sesuai struktur yang dijelaskan untuk menjalankan ulang seluruh pipeline.