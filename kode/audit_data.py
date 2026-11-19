"""Tahap 1: audit skema, missing values, ID, label, tanggal, dan fitur berisiko.
Menjalankan: python kode/audit_data.py --data-dir data --out-dir hasil
"""
import argparse
import json
from collections import Counter
from pathlib import Path
import pandas as pd

FILES = {
    "regresi_train": ("regresi_train.csv.gz", "id_paket", "durasi_jam", "tgl_pickup"),
    "regresi_test": ("regresi_test.csv.gz", "id_paket", None, "tgl_pickup"),
    "klasifikasi_train": ("klasifikasi_train.csv.gz", "id_transaksi", "is_fraud", "waktu"),
    "klasifikasi_test": ("klasifikasi_test.csv.gz", "id_transaksi", None, "waktu"),
    "clustering": ("clustering.csv.gz", "id_pelanggan", None, None),
}


def scan(path: Path, id_col: str, target: str | None, time_col: str | None) -> dict:
    count = 0
    missing = Counter()
    duplicates = 0
    seen = set()
    targets = Counter()
    earliest = latest = None
    columns = None
    types = {}
    for chunk in pd.read_csv(path, chunksize=25_000, low_memory=False):
        if columns is None:
            columns = list(chunk.columns)
            types = {k: str(v) for k, v in chunk.dtypes.items()}
        count += len(chunk)
        missing.update(chunk.isna().sum().to_dict())
        # Unique IDs across chunks (necessary to catch repeated IDs).
        for item in chunk[id_col].dropna():
            if item in seen:
                duplicates += 1
            else:
                seen.add(item)
        if target:
            if target == 'is_fraud':
                targets.update(chunk[target].value_counts(dropna=False).to_dict())
            else:
                desc = chunk[target].describe()
                # accumulate valid target count and sum, not average-of-averages
                targets['valid'] += int(desc['count'])
                targets['sum'] += float(chunk[target].sum())
                targets['invalid_nonpositive'] += int((chunk[target].dropna() <= 0).sum())
        if time_col:
            datetimes = pd.to_datetime(chunk[time_col], errors='coerce')
            t_min, t_max = datetimes.min(), datetimes.max()
            if pd.notna(t_min):
                earliest = t_min if earliest is None else min(earliest, t_min)
                latest = t_max if latest is None else max(latest, t_max)
    result = {
        'file': path.name, 'rows': count, 'columns': len(columns or []),
        'column_names': columns or [], 'dtypes_first_chunk': types,
        'duplicate_id_rows': duplicates,
        'missing_by_column': {k: int(v) for k, v in missing.items() if v},
        'date_min': earliest.isoformat() if earliest is not None else None,
        'date_max': latest.isoformat() if latest is not None else None,
        'target_summary': dict(targets),
    }
    if target == 'is_fraud' and count:
        result['fraud_rate'] = targets.get(1, 0) / count
    if target == 'durasi_jam' and targets.get('valid'):
        result['duration_mean_hours'] = targets['sum'] / targets['valid']
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--out-dir', default='hasil')
    args = parser.parse_args()
    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for key, (filename, identifier, target, time_col) in FILES.items():
        file_path = data_dir / filename
        if not file_path.is_file():
            raise FileNotFoundError(f'Tidak ditemukan: {file_path.resolve()}')
        print(f'[AUDIT] {filename}', flush=True)
        results[key] = scan(file_path, identifier, target, time_col)
        print(f"  {results[key]['rows']:,} baris; {results[key]['columns']} kolom; duplikat ID: {results[key]['duplicate_id_rows']}")
    for a, b, target in [('regresi_train', 'regresi_test', 'durasi_jam'),
                         ('klasifikasi_train', 'klasifikasi_test', 'is_fraud')]:
        train_cols = set(results[a]['column_names']) - {target}
        test_cols = set(results[b]['column_names'])
        results[a]['train_only_features'] = sorted(train_cols - test_cols)
        results[a]['test_only_features'] = sorted(test_cols - train_cols)
    with (out_dir / 'audit_data.json').open('w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    report = [
        '# Audit dataset UTS ML — tahap 1', '',
        '> Temuan otomatis belum membuktikan data leakage. Ketersediaan fitur pada saat prediksi harus diperiksa secara konseptual.', '',
        '| Dataset | Baris | Kolom | Duplikat ID | Periode |',
        '|---|---:|---:|---:|---|',
    ]
    for key, res in results.items():
        period = f"{res['date_min'] or '-'} s.d. {res['date_max'] or '-'}"
        report.append(f"| {key} | {res['rows']:,} | {res['columns']} | {res['duplicate_id_rows']} | {period} |")
    report += ['', '## Missing values per dataset', '']
    for key, res in results.items():
        ranked = sorted(res['missing_by_column'].items(), key=lambda x: -x[1])
        report.append(f"### {key}")
        report.append(', '.join(f'{col}: {n:,} ({n / res["rows"]:.2%})' for col, n in ranked[:20]) or 'Tidak ditemukan missing values.')
        report.append('')
    report += [
        '## Audit semantik (perlu diuji)', '',
        '- **Regresi:** `jumlah_scan` mungkin mencerminkan pemindaian setelah pickup. Audit waktu terbentuknya fitur; jangan pakai dalam model produksi bila tidak tersedia saat pickup.',
        '- **Klasifikasi:** `flag_review_manual` mungkin baru diketahui setelah transaksi masuk antrean review. Audit dan lakukan ablation.',
        '- **Klasifikasi:** label `is_fraud` dapat mengalami keterlambatan konfirmasi; perlu validation gap atau cutoff label yang defensible.',
        '- **Clustering:** cek apakah `porsi_...` berjumlah 1, konsistensi `belanja_90h` dan rata-rata keranjang, skala, dan outlier.',
        '- **Anonim:** cek fitur f01–f50, v01–v50, n01–n15 secara empiris sebelum pemilihan akhir.',
        '', '## Langkah berikutnya', '',
        '1. Periksa audit_data.json dan validasi train/test.',
        '2. Rancang validation split time-aware dari tanggal historis.',
        '3. Jalankan baseline, lalu eksperimen pembanding yang dapat direproduksi.',
    ]
    (out_dir / 'audit_ringkas.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print('\nSelesai: hasil/audit_data.json dan hasil/audit_ringkas.md')


if __name__ == '__main__':
    main()
