# Audit Task 3 — Clustering (Rafi)

Dokumen ini berisi hasil pemeriksaan kode di `kode/` dan validasi akhir `cluster.csv` sebagai rujukan penyusunan laporan Task 3 dan pengecekan Pull Request.

## 1. Ringkasan Audit Rubrik

| Rubrik | Bukti File / Hasil | Temuan | Status | Tindak Lanjut |
| :--- | :--- | :--- | :--- | :--- |
| **Audit dataset** | `data/kamus_data.md`, `kode/audit_data.py`, `kode/profil_awal_clustering.py` | Teridentifikasi 19 fitur numerik berdistribusi *skewed*, *missing value* struktural pada `rata2_keranjang` saat `frekuensi_90h == 0` (6.736 baris), variabel komposisional (`porsi_*`, `porsi_jam_*`), serta variabel kategorikal `perangkat`. | Terverifikasi di kode | Rekap persentase missing dan nilai skewness per fitur dari log `audit_data.py` untuk lampiran. |
| **Seleksi & prapemrosesan fitur** | `kode/validasi_fitur_clustering.py`, `kode/audit_scaling_clustering.py`, `kode/eksperimen_scaling_clustering.py` | `id_pelanggan` dikeluarkan; 19 fitur *skewed* ditransformasi dengan `np.log1p`; satu kategori terakhir pada `porsi_*` dan `porsi_jam_*` dihapus untuk mencegah multikolinearitas sempurna; `perangkat` di-*one-hot encode*; fitur aktif di-*winsorize* (1%–99%) dan distandardisasi (`StandardScaler`). | Terverifikasi di kode | Cantumkan alasan penghapusan kolom komposisional terakhir dan pemilihan Winsorize + `StandardScaler` di laporan. |
| $\ge 3$ **metode** | `kode/baseline_clustering.py`, `kode/eksperimen_dua_tahap.py`, `kode/ensemble_anomali.py` | Eksperimen segmentasi yang terverifikasi di skrip utama menggunakan pendekatan `MiniBatchKMeans` (skema 1 tahap vs skema 2 tahap terpisah aktif/inaktif) yang dipadukan dengan `IsolationForest` untuk deteksi anomali. | Perlu catatan di laporan | Sajikan perbandingan terukur antara skema 1 tahap, skema 2 tahap, dan deteksi anomali tanpa mengklaim metode lain di luar skrip. |
| $\ge 2$ **kriteria** $K$ | `kode/baseline_clustering.py`, `kode/eksperimen_dua_tahap.py` | Evaluasi *baseline* menguji $K \in \{3, 4, 5, 6, 7, 8\}$ menggunakan 3 metrik internal pada sampel acak $N = 5.000$ (`seed=42`): **Silhouette Score**, **Davies-Bouldin Index (DBI)**, dan **Calinski-Harabasz Index (CHI)**, dilengkapi evaluasi stabilitas ARI pada model terpilih. | Terverifikasi di kode | Ekspor tabel `baseline_kmeans_metrics.csv` saat eksekusi lokal untuk melengkapi angka eksak pada grafik pemilihan $K$. |
| **Stabilitas** | `kode/stabilitas_clustering.py`, `kode/stabilitas_pelanggan_aktif.py`, `kode/eksperimen_dua_tahap.py` | Stabilitas diukur menggunakan rata-rata dan simpangan baku *Adjusted Rand Index* (ARI) dari seluruh 10 pasangan kombinasi 5 *random seed* (`11, 22, 33, 42, 55`) pada konfigurasi `B_robust_clip5`, `C_winsor_standard`, dan model dua tahap ($K=3$ aktif). | Terverifikasi di kode | Sertakan ringkasan `mean`, `std`, `min`, dan `max` ARI lintas 10 pasangan seed dalam tabel stabilitas. |
| **Profil dan aksi** | `kode/profil_awal_clustering.py`, `kode/profil_dua_tahap.py`, `cluster.csv` | `profil_dua_tahap.py` mengekstrak ukuran klaster, median 10 indikator perilaku utama (`recency_hari`, `frekuensi_90h`, `belanja_90h`, `rata2_keranjang`, `belanja_30h`, `frekuensi_30h`, `retur_90h`, `sesi_per_minggu`, `rasio_konversi`, `lama_member_bulan`), serta distribusi `perangkat`. | Terverifikasi di kode | Isi nama segmen dan aksi pemasaran spesifik berdasarkan output `profil_dua_tahap.csv`. |
| **Anomali** | `kode/ensemble_anomali.py`, `kode/stabilitas_anomali.py` | Diterapkan khusus pada 113.264 pelanggan aktif memakai *ensemble* 5 model `IsolationForest` (`n_estimators=200`, `max_samples=256`, seeds `11, 22, 33, 42, 55`). Skor `-score_samples` dirata-ratakan, lalu persentil teratas 1% (`1.133` pelanggan) dilabeli `-1`. | Lengkap & terverifikasi | Lampirkan distribusi `vote_count` (0–5) dan skor kesepakatan Jaccard antar-seed sebagai dasar pemilihan ambang 1%. |
| **Finalisasi** | `kode/finalisasi_clustering.py`, `cluster.csv` | File `cluster.csv` tervalidasi penuh: 120.000 baris, `id_pelanggan` unik dan berurutan sesuai sumber, bebas nilai kosong, himpunan label sah `{-1, 0, 1, 2, 3}`, tepat `1.133` anomali (`-1`), serta seluruh `6.736` pelanggan tanpa transaksi (`frekuensi_90h == 0`) berada pada klaster `3`. | **Selesai** | Siap diserahkan sebagai artefak akhir Task 3. |

## 2. Prioritas Penyelesaian & Batasan Laporan

1. **Kejujuran Metodologi:** Seluruh klaim di laporan dibatasi secara ketat pada konfigurasi yang tercatat di kode (perbandingan skema `MiniBatchKMeans` 1 tahap vs 2 tahap, 3 metrik evaluasi $K$, uji stabilitas ARI 5 seed, dan *ensemble* `IsolationForest`).
2. **Efisiensi Halaman (Batas 18 Halaman):** Untuk bagian Task 3, cukup tampilkan maksimal **2 visual utama** agar ringkas:
   * **Grafik 1:** Kurva pemilihan $K$ ($K=3..8$) yang menampilkan tren Silhouette Score dan Davies-Bouldin Index.
   * **Grafik 2:** Distribusi akhir 120.000 pelanggan pada 5 segmen (`-1`, `0`, `1`, `2`, `3`).

## 3. Struktur Hasil & Profil Segmen Akhir

### A. Ringkasan Konfigurasi & Evaluasi Model

| Tahap / Konfigurasi | Algoritma & Prapemrosesan | Ruang Fitur / Populasi | Kriteria Evaluasi | Status Output |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline (1 Tahap)** | `MiniBatchKMeans` ($K=3..8$), `RobustScaler` | 120.000 pelanggan (seluruh populasi) | Silhouette, Davies-Bouldin, Calinski-Harabasz ($N=5.000$) | Tersimpan di `baseline_kmeans_metrics.csv` saat run |
| **Uji Scaling & Stabilitas** | `MiniBatchKMeans` ($K=4$), `RobustScaler` clip `[-5, 5]` vs Winsorize (1%–99%) + `StandardScaler` | 120.000 pelanggan | Pairwise ARI (10 pasangan dari 5 seed: `11, 22, 33, 42, 55`) | Tersimpan di `stabilitas_summary.csv` saat run |
| **Model Terpilih (2 Tahap)** | Pemisahan pelanggan inaktif (`Cluster 3`), lalu `MiniBatchKMeans` ($K=3$, `seed=42`) dengan Winsorize + `StandardScaler` | 113.264 pelanggan aktif + 6.736 pelanggan inaktif | Pairwise ARI lintas 5 seed (`11, 22, 33, 42, 55`) | Tersimpan di `stabilitas_dua_tahap.csv` saat run |
| **Deteksi Anomali** | *Ensemble* 5 `IsolationForest` (`n_estimators=200`, `max_samples=256`, top 1% skor rata-rata) | 113.264 pelanggan aktif | Distribusi `vote_count` & kesepakatan Jaccard antar-seed | Tersimpan di `ensemble_anomali_scores.csv` saat run |

### B. Ringkasan Distribusi & Arah Tindak Lanjut Segmen (`cluster.csv`)

| Label | Kelompok Pelanggan | Jumlah | Porsi Populasi | Karakteristik Utama & Dasar Pemisahan | Rekomendasi Aksi Pemasaran |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`-1`** | Anomali Perilaku (*Top* 1% Aktif) | 1.133 | 0,94% | Persentil 1% tertinggi dari rata-rata skor anomali 5 model `IsolationForest` pada pelanggan aktif. | Keluarkan dari penargetan promo subsidi massal; lakukan audit pola transaksi ekstrem atau indikasi *reseller*. |
| **`0` – `2`** | Tiga Segmen Pelanggan Aktif | 112.131 | 93,44% | Pelanggan dengan `frekuensi_90h > 0` (non-anomali) yang dikelompokkan oleh `MiniBatchKMeans` ($K=3$, `seed=42`) berdasarkan recency, frekuensi, nilai belanja, dan aktivitas sesi. | Sesuaikan intervensi berdasarkan profil median tiap klaster (program loyalitas untuk segmen bernilai tinggi, *cross-sell* untuk segmen menengah, dan pengingat keranjang untuk frekuensi rendah). |
| **`3`** | Pelanggan Inaktif 90 Hari | 6.736 | 5,61% | Seluruh pelanggan dengan `frekuensi_90h == 0` dan `rata2_keranjang` kosong yang dipisahkan pada tahap pertama. | Jalankan kampanye reaktivasi (*win-back*) berbasis voucher berbatas waktu dan rekomendasi produk populer. |
| **Total** | **Seluruh Populasi** | **120.000** | **100,00%** | Tervalidasi 100% unik dan lengkap pada `cluster.csv`. | — |

## 4. Laporan Task 3

**Segmentasi Pelanggan dan Deteksi Anomali**

Segmentasi pelanggan dilakukan terhadap 120.000 data pengguna menggunakan pendekatan dua tahap untuk mencegah kelompok pelanggan yang tidak melakukan transaksi mendominasi pembentukan pusat klaster. Pada tahap pertama, sebanyak 6.736 pelanggan (5,61%) yang tidak memiliki riwayat transaksi selama 90 hari terakhir (`frekuensi_90h == 0`) dipisahkan secara deterministik ke dalam Klaster `3`. Tahap kedua berfokus pada 113.264 pelanggan aktif (`frekuensi_90h > 0`). Sebelum pemodelan, dilakukan prapemrosesan yang mencakup imputasi nilai nol pada `rata2_keranjang` yang kosong secara struktural, transformasi $\log(1 + x)$ pada 19 fitur berdistribusi miring (*skewed*), penghapusan satu kategori referensi pada variabel komposisional (`porsi_*` dan `porsi_jam_*`) guna menghindari kolinearitas sempurna, serta *one-hot encoding* pada variabel `perangkat`. Fitur pada kelompok aktif kemudian melalui proses *winsorizing* pada persentil 1% hingga 99% dan standardisasi menggunakan `StandardScaler`.

Untuk menjaga agar observasi ekstrem tidak menggeser pusat klaster utama, diterapkan deteksi anomali berbasis *ensemble* lima model `IsolationForest` (`n_estimators=200`, `max_samples=256`) pada lima *random seed* (`11, 22, 33, 42, 55`) khusus pada populasi pelanggan aktif. Skor anomali dari kelima model dirata-ratakan, dan 1% pelanggan dengan skor tertinggi—sebanyak 1.133 pelanggan (0,94% dari total populasi)—ditetapkan sebagai kelompok anomali berlabel `-1`.

Penentuan konfigurasi klaster diawali dengan evaluasi *baseline* `MiniBatchKMeans` pada rentang $K = 3$ hingga $8$ menggunakan tiga kriteria internal pada sampel $N = 5.000$, yakni Silhouette Score, Davies-Bouldin Index, dan Calinski-Harabasz Index, serta pengujian stabilitas berbasis *Adjusted Rand Index* (ARI) lintas 10 pasangan kombinasi dari lima *random seed*. Pendekatan dua tahap dengan $K = 3$ klaster aktif (`0`, `1`, `2`) pada `seed=42` dipilih sebagai konfigurasi final karena mampu memisahkan perilaku pelanggan aktif secara lebih tajam dari kelompok inaktif. Berkas akhir `cluster.csv` telah diverifikasi memuat seluruh 120.000 `id_pelanggan` unik tanpa nilai kosong dengan distribusi label sah pada himpunan $\{-1, 0, 1, 2, 3\}$.

## 5. Langkah Verifikasi Pull Request

Jalankan perintah berikut pada direktori utama repositori untuk memverifikasi integritas berkas `cluster.csv` sebelum melakukan *merge*:

```bash
python -c "
import pandas as pd
df = pd.read_csv('cluster.csv')
assert len(df) == 120000, 'Jumlah baris tidak sesuai'
assert df['id_pelanggan'].is_unique, 'ID pelanggan duplikat'
assert df['cluster'].notna().all(), 'Terdapat nilai kosong'
assert set(df['cluster']).issubset({-1, 0, 1, 2, 3}), 'Label di luar rentang'
assert (df['cluster'] == -1).sum() == 1133, 'Jumlah anomali tidak sesuai'
assert (df['cluster'] == 3).sum() == 6736, 'Jumlah pelanggan inaktif tidak sesuai'
print('Validasi cluster.csv berhasil (120.000 baris).')
print(df['cluster'].value_counts().sort_index())
"
```

* [x] Seluruh klaim metodologi selaras dengan kode pada folder `kode/`.
* [x] Angka populasi (`120.000` total, `113.264` aktif, `6.736` inaktif, `1.133` anomali) sesuai dengan aturan validasi di `kode/finalisasi_clustering.py`.
* [x] Draf narasi siap disalin ke dokumen laporan tanpa mengandung *placeholder*.
