import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# KONFIGURASI
INPUT_FILE = 'tweets_clean.csv'

def main():
    print("Membaca data bersih...")
    try:
        df = pd.read_csv(INPUT_FILE)
        print(f"Data dimuat: {len(df)} baris.")
    except FileNotFoundError:
        print("File 'tweets_clean.csv' tidak ditemukan. Pastikan Stage 1 sukses.")
        return

    # Urutkan hari agar Heatmap rapi (Senin s/d Minggu)
    # Ini trik penting agar hari tidak urut abjad (Jumat dulu baru Kamis)
    urutan_hari = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['day_name'] = pd.Categorical(df['day_name'], categories=urutan_hari, ordered=True)

    # ==========================================
    # 1. PIE CHART (Proporsi Sentimen)
    # ==========================================
    print("Menguji Grafik 1: Pie Chart...")
    # Hitung jumlah per sentimen
    pie_data = df['airline_sentiment'].value_counts().reset_index()
    pie_data.columns = ['airline_sentiment', 'count']
    
    fig1 = px.pie(pie_data, values='count', names='airline_sentiment', 
                  title='TEST 1: Distribusi Sentimen Global',
                  color='airline_sentiment',
                  color_discrete_map={'positive':'green', 'neutral':'gray', 'negative':'red'})
    fig1.show()


    # 2. STACKED BAR CHART (Komparasi Maskapai)
    print("Menguji Grafik 2: Stacked Bar Chart...")
    # Group by Maskapai dan Sentimen
    bar_data = df.groupby(['airline', 'airline_sentiment']).size().reset_index(name='jumlah')
    
    fig2 = px.bar(bar_data, x='airline', y='jumlah', color='airline_sentiment',
                  title='TEST 2: Sentimen per Maskapai',
                  barmode='stack',
                  color_discrete_map={'positive':'green', 'neutral':'gray', 'negative':'red'})
    fig2.show()


    # 3. LINE CHART (Time Series)
    print("Menguji Grafik 3: Line Chart (Tren Harian)...")
    # Group by Tanggal dan Sentimen
    line_data = df.groupby(['date', 'airline_sentiment']).size().reset_index(name='jumlah')
    
    fig3 = px.line(line_data, x='date', y='jumlah', color='airline_sentiment',
                   title='TEST 3: Tren Tweet Harian',
                   color_discrete_map={'positive':'green', 'neutral':'gray', 'negative':'red'})
    fig3.show()

    
    # 4. HEATMAP (Waktu Komplain)
    print("Menguji Grafik 4: Heatmap (Jam vs Hari)...")
    # Filter hanya sentimen negatif
    neg_df = df[df['airline_sentiment'] == 'negative']
    # Pivot table: Baris=Hari, Kolom=Jam, Isi=Jumlah
    heatmap_data = neg_df.groupby(['day_name', 'hour']).size().unstack(fill_value=0)
    
    fig4 = px.imshow(heatmap_data, 
                     labels=dict(x="Jam (Hour)", y="Hari (Day)", color="Jumlah Komplain"),
                     x=heatmap_data.columns,
                     y=heatmap_data.index,
                     title="TEST 4: Kapan Orang Paling Sering Marah?",
                     color_continuous_scale='Reds')
    fig4.show()


    # 5. BOXPLOT (Distribusi Confidence)
    print("Menguji Grafik 5: Boxplot (Statistik)...")
    fig5 = px.box(df, x='airline_sentiment', y='airline_sentiment_confidence',
                  color='airline_sentiment',
                  title='TEST 5: Seberapa Yakin Model AI-nya?',
                  color_discrete_map={'positive':'green', 'neutral':'gray', 'negative':'red'})
    fig5.show()

    
    # 6. WORDCLOUD (Analisis Teks)
    print("Menguji Grafik 6: WordCloud (Teks Negatif)...")
    # Gabung semua teks negatif jadi satu string panjang
    text_negatif = " ".join(tweet for tweet in df[df['airline_sentiment']=='negative']['text'])
    
    # Buat WordCloud
    wc = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate(text_negatif)
    
    # Tampilkan pakai Matplotlib (karena Wordcloud itu gambar statis)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.title("TEST 6: Kata-kata Paling Sering Muncul di Tweet Negatif")
    plt.show()

    print("\n SEMUA PENGUJIAN SELESAI.")
    print("Jika browser terbuka dan menampilkan 6 grafik, berarti Stage 2 SUKSES.")

if __name__ == "__main__":
    main()