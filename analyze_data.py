import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('data/ulasan_goride_preprocessed.csv')
df['date'] = pd.to_datetime(df['date'])

print('=== ANALISIS DATASET GORIDE ===')
print(f'Total ulasan: {len(df):,}')
print(f'Rentang tanggal: {df["date"].min().strftime("%Y-%m-%d")} hingga {df["date"].max().strftime("%Y-%m-%d")}')
print(f'Periode: {(df["date"].max() - df["date"].min()).days} hari')
print()

print('=== DISTRIBUSI PER BULAN ===')
monthly = df.groupby(df['date'].dt.to_period('M')).size()
for period, count in monthly.items():
    print(f'{period}: {count:,} ulasan')
print()

print('=== DISTRIBUSI PER MINGGU (10 pertama) ===')
weekly = df.groupby(df['date'].dt.to_period('W')).size().head(10)
for period, count in weekly.items():
    print(f'Minggu {period}: {count:,} ulasan')
print()

print('=== STATISTIK HARIAN ===')
daily_stats = df.groupby(df['date'].dt.date).size().describe()
print(f'Rata-rata ulasan per hari: {daily_stats["mean"]:.1f}')
print(f'Median ulasan per hari: {daily_stats["50%"]:.0f}')
print(f'Maksimum ulasan per hari: {daily_stats["max"]:.0f}')
print(f'Minimum ulasan per hari: {daily_stats["min"]:.0f}')
print()

print('=== ANALISIS DISTRIBUSI TEMPORAL ===')
# Hitung hari dengan data
days_with_data = len(df.groupby(df['date'].dt.date))
total_days = (df['date'].max() - df['date'].min()).days + 1
coverage = days_with_data / total_days * 100

print(f'Hari dengan data: {days_with_data} dari {total_days} hari total')
print(f'Coverage: {coverage:.1f}%')
print()

print('=== DISTRIBUSI SENTIMENT ===')
sentiment_dist = df['sentiment'].value_counts()
print(sentiment_dist)
print()
print(f'Positif: {len(df[df["sentiment"] == "Positive"]) / len(df) * 100:.1f}%')
print(f'Negatif: {len(df[df["sentiment"] == "Negative"]) / len(df) * 100:.1f}%')
print()

print('=== REKOMENDASI GRANULARITAS ===')
print('Berdasarkan analisis data:')
if daily_stats["mean"] >= 10:
    print('✅ HARIAN: Direkomendasikan (rata-rata >= 10 ulasan/hari)')
else:
    print('❌ HARIAN: Tidak direkomendasikan (rata-rata < 10 ulasan/hari)')

weekly_avg = monthly.mean() / 4.33  # rata-rata minggu per bulan
if weekly_avg >= 20:
    print('✅ MINGGUAN: Direkomendasikan (rata-rata >= 20 ulasan/minggu)')
else:
    print('❌ MINGGUAN: Tidak direkomendasikan (rata-rata < 20 ulasan/minggu)')

monthly_avg = monthly.mean()
if monthly_avg >= 50:
    print('✅ BULANAN: Direkomendasikan (rata-rata >= 50 ulasan/bulan)')
else:
    print('❌ BULANAN: Tidak direkomendasikan (rata-rata < 50 ulasan/bulan)')
