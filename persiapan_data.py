import pandas as pd
import numpy as np

# KONFIGURASI PROSES
INPUT_FILE = 'Tweets.csv'
OUTPUT_FILE = 'tweets_clean.csv'

def bersihkan_data():
    print(" Sedang membaca data mentah...")
    
    # 1. Membaca Data
    try:
        df = pd.read_csv(INPUT_FILE)
        print(f" Data berhasil dibaca! Total baris: {len(df)}")
    except FileNotFoundError:
        print(f" ERROR: File '{INPUT_FILE}' tidak ditemukan di folder ini.")
        return

    # 2. Membersihkan Format Waktu (CRITICAL STEP)
    # Kolom 'tweet_created' aslinya berupa teks. Kita ubah jadi objek Waktu (Datetime)
    # errors='coerce' berarti jika ada format waktu yang aneh, ubah jadi NaT (Not a Time) daripada error.
    print("🧹 Sedang membersihkan format waktu...")
    df['tweet_created'] = pd.to_datetime(df['tweet_created'], errors='coerce')
    
    # Hapus baris jika ada waktu yang gagal dikonversi (biasanya jarang terjadi, tapi untuk jaga-jaga)
    df = df.dropna(subset=['tweet_created'])

    # 3. Ekstraksi Fitur Baru (Feature Engineering)
    # Kita butuh kolom ini untuk visualisasi nanti
    print("⚙️  Sedang mengekstrak fitur waktu (Jam, Hari, Tanggal)...")
    df['hour'] = df['tweet_created'].dt.hour             # Mengambil Jam (0-23)
    df['date'] = df['tweet_created'].dt.date             # Mengambil Tanggal saja (YYYY-MM-DD)
    df['day_name'] = df['tweet_created'].dt.day_name()   # Mengambil Nama Hari (Monday, Tuesday...)

    # 4. Menangani Data Kosong (Missing Values)
    # Kolom 'negativereason' (alasan negatif) banyak yang kosong karena tweet positif tidak punya alasan.
    # Kita isi yang kosong dengan label "Positive/Neutral" agar grafik tidak bolong.
    print("🛡️  Sedang menangani data kosong...")
    df['negativereason'] = df['negativereason'].fillna('Positive/Neutral')
    
    # Kolom 'negativereason_confidence' juga kosong jika positif. Kita isi dengan 0.
    df['negativereason_confidence'] = df['negativereason_confidence'].fillna(0)

    # 5. Penyederhanaan Koordinat (Opsional tapi bagus)
    # Kita buang kolom yang terlalu rumit dan tidak dipakai di dashboard untuk menghemat memori
    kolom_dibuang = ['tweet_coord', 'tweet_id', 'user_timezone'] 
    df = df.drop(columns=kolom_dibuang, errors='ignore')

    # 6. Menyimpan Hasil
    print(f"💾 Menyimpan data bersih ke '{OUTPUT_FILE}'...")
    df.to_csv(OUTPUT_FILE, index=False)
    
    print("\n" + "="*40)
    print("🎉 SELESAI! STAGE 1 BERHASIL.")
    print(f"File '{OUTPUT_FILE}' sudah siap digunakan untuk Stage 2.")
    print("="*40)
    
    # Tampilkan sampel data untuk pengecekan
    print("\nSampel 5 data teratas:")
    print(df[['airline', 'airline_sentiment', 'day_name', 'hour']].head())

if __name__ == "__main__":
    bersihkan_data()