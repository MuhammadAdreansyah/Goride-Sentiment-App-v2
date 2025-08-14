# Catatan Fitur Dinonaktifkan (Sementara)

Tanggal: 2025-08-14
Status: Production (mode ringkas tanpa fitur eksperimental)

## Ringkasan
Dokumen ini mencatat dua fitur yang saat ini DINONAKTIFKAN dari `Dashboard_Ringkasan` untuk menjaga fokus pada hasil analisis sentimen global yang stabil.

## 1. 🏷️ Filter Berdasarkan Topik (Topic Filtering)
- Status: Dihapus dari UI.
- Alasan Penghapusan: Menyederhanakan interpretasi karena dashboard versi ini menekankan hasil agregat keseluruhan, bukan eksplorasi aspek parsial.
- Implementasi Lama: Mengelompokkan token ke dalam kategori (Driver, Aplikasi, Harga, Waktu, Pembayaran, Promosi, Keamanan, Kenyamanan, Layanan, Performa) dengan pemetaan sinonim + regex pattern.
- Resiko Jika Diaktifkan Ulang Tanpa Evaluasi:
  - Potensi bias jika subset topik dipilih sebelum stakeholder membaca ringkasan global.
  - Munculnya category sparsity jika data baru sangat berbeda.
- Indikator Kapan Perlu Diaktifkan Kembali:
  - Kebutuhan eksplorasi root cause spesifik dari tim operasional.
  - Volume data >= 5.000 ulasan sehingga segmentasi tematik memberi nilai tambah.
  - Permintaan analisis banding performa per aspek.
- Cara Reaktivasi (High-Level):
  1. Restore blok "TOPIC FILTERING" yang pernah ada di `render_dashboard()` sebelum tab analisis detail.
  2. Pastikan fungsi preprocessing sudah menghasilkan kolom `teks_preprocessing`.
  3. Validasi kembali threshold MIN_COUNT dan MAX_TOPICS sesuai distribusi data terbaru.

## 2. 🔍 Mode Verifikasi (Debug Aspek)
- Status: Dihapus dari Tab "Insights & Rekomendasi".
- Alasan Penghapusan: Mode ini hanya untuk audit internal (menampilkan tabel aspek + contoh ulasan) dan tidak diperlukan di lingkungan production.
- Komponen yang Dihapus:
  - Expander dengan judul: `🔍 Mode Verifikasi (Debug Aspek)`
  - DataFrame debug (kolom: Aspek, Total Ulasan, Positif, Negatif, Sentiment Score, Impact %, Opportunity)
  - Pemilihan aspek + list contoh ulasan (max 5 positif & 5 negatif per aspek)
- Manfaat Saat Aktif (Jika Audit Diperlukan):
  - Transparansi penurunan rekomendasi → aspek → token original.
  - Validasi heuristik: sentiment_score, impact, opportunity.
- Potensi Risiko Jika Tetap Aktif di Production:
  - Membocorkan pola teks pengguna yang sensitif.
  - Membingungkan user non-teknis dengan angka internal.
- Indikator Kapan Perlu Diaktifkan Kembali:
  - Review ilmiah / appendiks skripsi.
  - Investigasi anomali penurunan kualitas model.
  - Pengembangan fitur advanced (misal drill-down multi level).
- Cara Reaktivasi (High-Level):
  1. Tambahkan kembali expander setelah blok rekomendasi.
  2. Gunakan struktur `stats[a]['pos_ids'] / ['neg_ids']` untuk sampling ulasan.
  3. Regenerasi DataFrame debug dari list `scored`.

## 3. Rationale Arsitektural
| Aspek | Default Production | Mode Eksplorasi |
|-------|--------------------|-----------------|
| Fokus Output | Ringkasan agregat | Segmentasi & transparansi |
| Risiko Bias | Rendah | Menengah (subset filtering) |
| Kompleksitas UI | Minimal | Bertambah |
| Kecepatan Muat | Optimal | Tambahan komputasi ringan |

## 4. Jika Suatu Saat Diaktifkan Bersamaan
Lakukan urutan: (1) Aktifkan kembali Filter Topik → (2) Jalankan Mode Verifikasi hanya dalam konteks internal (misal flag `st.session_state['debug_mode']`). Pastikan pengamanan: masking teks sensitif jika diperlukan.

## 5. Checklist Reaktivasi Cepat
- [ ] Volume & variasi data cukup besar
- [ ] Kebutuhan analisis aspek eksplisit
- [ ] Tim setuju terhadap tambahan kompleksitas UI
- [ ] Audit privasi teks dilakukan (jika contoh ulasan muncul)
- [ ] Performance test (latensi tab < 1.5s rata-rata)

## 6. Catatan Lanjutan
Pertimbangkan jika nanti ingin migrasi fitur ini ke tab terpisah: "🔎 Eksplorasi Aspek" agar tab utama tetap ringkas.

---
Dokumen ini bertujuan menjaga jejak keputusan (decision log) agar aktivasi ulang terstruktur dan terukur.

---

## Lampiran A. Cuplikan Kode: Filter Berdasarkan Topik (Dinonaktifkan)

Blok ini sebelumnya berada di dalam fungsi `render_dashboard()` setelah preprocessing selesai dan sebelum pembuatan tabs analisis.

```python
# ==========================================
# 6. TOPIC FILTERING
# ==========================================
st.markdown("---")
st.markdown("## 🏷️ Filter Berdasarkan Topik")

try:
  import re as _re
except Exception:
  import re as _re

TOPIC_SYNONYM_MAP = {
  'driver': 'Driver','pengemudi': 'Driver','ojek': 'Driver','kurir': 'Driver',
  'aplikasi': 'Aplikasi','app': 'Aplikasi','apps': 'Aplikasi','fitur': 'Aplikasi','update': 'Aplikasi','versi': 'Aplikasi',
  'harga': 'Harga','tarif': 'Harga','biaya': 'Harga','ongkos': 'Harga','mahal': 'Harga','murah': 'Harga',
  'waktu': 'Waktu','lama': 'Waktu','cepat': 'Waktu','nunggu': 'Waktu','tunggu': 'Waktu','menunggu': 'Waktu','delay': 'Waktu',
  'bayar': 'Pembayaran','pembayaran': 'Pembayaran','gopay': 'Pembayaran','cash': 'Pembayaran','tunai': 'Pembayaran','saldo': 'Pembayaran',
  'promo': 'Promosi','promosi': 'Promosi','diskon': 'Promosi','voucher': 'Promosi','potongan': 'Promosi',
  'aman': 'Keamanan','keamanan': 'Keamanan','bahaya': 'Keamanan','kecelakaan': 'Keamanan','nabrak': 'Keamanan',
  'nyaman': 'Kenyamanan','kenyamanan': 'Kenyamanan','bersih': 'Kenyamanan','panas': 'Kenyamanan','bau': 'Kenyamanan','helm': 'Kenyamanan',
  'layanan': 'Layanan','service': 'Layanan','respon': 'Layanan','customer': 'Layanan','cs': 'Layanan',
  'error': 'Performa','bug': 'Performa','lambat': 'Performa','lemot': 'Performa','lag': 'Performa','crash': 'Performa','force': 'Performa','close': 'Performa'
}

AMBIGUOUS_TOKENS = {
  'tidak','nya','bisa','sangat','sekali','lebih','baik','tolong','sesuai','yang','itu','ini','ada','jadi','agar','buat','kalo','kalau','udah','sudah','akan','dengan','dgn','dan','atau','untuk','di','ke','dari','pada','karena','sering','masih'
}

raw_text = " ".join(filtered_data['teks_preprocessing'].dropna())
tokens = [t for t in raw_text.split() if len(t) > 1]
freq = {}
for t in tokens:
  t_norm = t.lower()
  if t_norm in AMBIGUOUS_TOKENS:
    continue
  if not _re.match(r'^[a-zA-Z]+$', t_norm):
    continue
  if t_norm in TOPIC_SYNONYM_MAP or len(t_norm) >= 3:
    freq[t_norm] = freq.get(t_norm, 0) + 1

category_counts = {}
category_terms = {}
for tok, c in freq.items():
  if tok in TOPIC_SYNONYM_MAP:
    cat = TOPIC_SYNONYM_MAP[tok]
  else:
    cat = tok.capitalize()
  category_counts[cat] = category_counts.get(cat, 0) + c
  category_terms.setdefault(cat, set()).add(tok)

MIN_COUNT = 3
filtered_categories = {cat: cnt for cat, cnt in category_counts.items() if cnt >= MIN_COUNT and cat.lower() not in AMBIGUOUS_TOKENS}
PRIMARY_ORDER = ['Driver','Aplikasi','Harga','Waktu','Pembayaran','Promosi','Keamanan','Kenyamanan','Layanan','Performa']
primary_present = [c for c in PRIMARY_ORDER if c in filtered_categories]
others = sorted([c for c in filtered_categories if c not in PRIMARY_ORDER], key=lambda x: filtered_categories[x], reverse=True)
ordered_categories = primary_present + others

MAX_TOPICS = 15
if len(ordered_categories) > MAX_TOPICS:
  ordered_categories = ordered_categories[:MAX_TOPICS]

TOPIC_PATTERNS = {}
for cat in ordered_categories:
  terms = category_terms.get(cat, {cat.lower()})
  if cat in PRIMARY_ORDER:
    synonyms = [k for k,v in TOPIC_SYNONYM_MAP.items() if v == cat]
    terms = set(synonyms) & set(freq.keys()) or set(synonyms)
  pattern = r'(' + '|'.join(sorted(terms, key=len, reverse=True)) + r')'
  TOPIC_PATTERNS[cat] = pattern

topics = ["Semua Topik"] + ordered_categories

col1, col2 = st.columns([2, 1])
with col1:
  selected_topic = st.selectbox(
    "🔍 Pilih topik/aspek untuk analisis mendalam:", 
    topics,
    help="Daftar telah dikurasi & dikelompokkan ke aspek yang mudah dipahami (Driver, Aplikasi, Harga, dll.)"
  )
with col2:
  if selected_topic != "Semua Topik" and selected_topic in TOPIC_PATTERNS:
    st.metric("📊 Jumlah Ulasan Aspek", filtered_categories.get(selected_topic, 0))

if selected_topic != "Semua Topik":
  pattern = TOPIC_PATTERNS.get(selected_topic, selected_topic)
  topic_data = filtered_data[filtered_data['teks_preprocessing'].str.contains(pattern, case=False, na=False)].copy()
  if topic_data.empty:
    st.warning("⚠️ Tidak ada ulasan yang cocok dengan aspek terpilih. Menampilkan semua data.")
    topic_data = filtered_data.copy()
  else:
    st.info(f"🎯 Menampilkan {len(topic_data):,} ulasan terkait aspek '{selected_topic}'")
else:
  topic_data = filtered_data.copy()
```

## Lampiran B. Cuplikan Kode: Mode Verifikasi (Debug Aspek)

Blok ini sebelumnya berada di akhir fungsi `render_insights_tab()` sebelum bagian narasi ringkas.

```python
# 7b. MODE VERIFIKASI (Debug) - hanya untuk pengecekan internal
with st.expander("🔍 Mode Verifikasi (Debug Aspek)", expanded=False):
  st.caption("Fitur sementara untuk memastikan aspek & rekomendasi berasal dari data nyata. Tidak untuk production.")
  if scored:
    debug_df = pd.DataFrame([
      {
        'Aspek': s['aspect'],
        'Total Ulasan': s['total'],
        'Positif': s['pos'],
        'Negatif': s['neg'],
        'Sentiment Score': round(s['sentiment_score'], 3),
        'Impact %': round(s['impact']*100, 2),
        'Opportunity': round(s['opportunity'], 3)
      } for s in scored
    ]).sort_values('Opportunity', ascending=False)
    st.dataframe(debug_df, use_container_width=True, height=min(500, 60 + 30*len(debug_df)))

    # Pilih aspek untuk lihat contoh ulasan
    aspek_list = debug_df['Aspek'].tolist()
    pilih_aspek = st.selectbox("Pilih Aspek untuk Contoh Ulasan", aspek_list, index=0, key="debug_pilih_aspek")
    if pilih_aspek and pilih_aspek in stats:
      pos_ids = list(stats[pilih_aspek]['pos_ids'])[:5]
      neg_ids = list(stats[pilih_aspek]['neg_ids'])[:5]
      st.markdown(f"**Contoh Ulasan Positif ({len(pos_ids)})**")
      if pos_ids:
        for i in pos_ids:
          row = topic_data.loc[i]
          st.write(f"✅ {row['review_text']}")
      else:
        st.write("(Tidak ada contoh positif disimpan)")
      st.markdown(f"**Contoh Ulasan Negatif ({len(neg_ids)})**")
      if neg_ids:
        for i in neg_ids:
          row = topic_data.loc[i]
          st.write(f"❌ {row['review_text']}")
      else:
        st.write("(Tidak ada contoh negatif disimpan)")
  else:
    st.info("Tidak ada data aspek untuk diverifikasi.")
```

Catatan: Kedua blok dapat dipasang kembali secara selektif tanpa mempengaruhi pipeline model utama.
