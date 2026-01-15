import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# 1. KONFIGURASI HALAMAN
st.set_page_config(
    page_title="Analisis Sentimen Penerbangan",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS buat rapihin margin atas biar judul nggak turun banget
st.markdown("""
    <style>
        .block-container {padding-top: 1rem; padding-bottom: 0rem;}
        h1 {font-family: 'Helvetica', sans-serif;}
    </style>
""", unsafe_allow_html=True)

# 2. BAGIAN LOAD DATA
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('tweets_clean.csv')
        
        # Konversi waktu
        data['tweet_created'] = pd.to_datetime(data['tweet_created'])
        data['date'] = data['tweet_created'].dt.date
        
        # Urutan hari
        hari_urut = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        data['day_name'] = pd.Categorical(data['day_name'], categories=hari_urut, ordered=True)
        
        return data
    except FileNotFoundError:
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.error("Error: File 'tweets_clean.csv' tidak ditemukan. Harap jalankan proses Data Cleaning terlebih dahulu.")
    st.stop()


# 3. SIDEBAR (INI BAGIAN PANEL KONTROLNYA)

st.sidebar.title("Panel Filter")
st.sidebar.info("Gunakan menu ini untuk menyaring data yang ditampilkan pada dashboard.")

# Filter A: Maskapai
daftar_maskapai = df['airline'].unique().tolist()
pilihan_maskapai = st.sidebar.multiselect(
    "Pilih Maskapai",
    options=daftar_maskapai,
    default=daftar_maskapai
)

# Filter B: Rentang Waktu
min_date = df['date'].min()
max_date = df['date'].max()

st.sidebar.markdown("---")
input_tanggal = st.sidebar.date_input(
    "Rentang Waktu Analisis",
    value=[min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

# Logika buat input tanggalnya
if isinstance(input_tanggal, tuple) or isinstance(input_tanggal, list):
    if len(input_tanggal) == 2:
        start_date, end_date = input_tanggal
    elif len(input_tanggal) == 1:
        start_date = input_tanggal[0]
        end_date = input_tanggal[0]
    else:
        start_date, end_date = min_date, max_date
else:
    start_date, end_date = input_tanggal, input_tanggal

st.sidebar.caption("© 2025 Proyek Visualisasi Data")


# 4. FILTERING DATA (BAGIAN BUAT FILTER DATANYA)
main_df = df[
    (df['airline'].isin(pilihan_maskapai)) & 
    (df['date'] >= start_date) & 
    (df['date'] <= end_date)
]

# Palet Warna Standar (Traffic Light System)
warna_sentimen = {'positive': '#2ECC71', 'neutral': '#95A5A6', 'negative': '#E74C3C'}


# 5. DASHBOARD UTAMANYA
# --- HEADER & IDENTITAS KELOMPOK ---
st.title("Kartografi Sentimen Maskapai Penerbangan AS")
st.markdown("##### Eksplorasi Topikal dan Temporal untuk Evaluasi Layanan")

# Identitas Kelompok
st.markdown("""
<div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 5px solid #E74C3C; margin-bottom: 20px;">
    <strong>Disusun Oleh Kelompok 1 (Social Media):</strong><br>
    Muhammad Zidan (0110222280) &nbsp;|&nbsp; Iqlima Fasha Rizqia (0110122006)
</div>
""", unsafe_allow_html=True)

if main_df.empty:
    st.warning("Data tidak ditemukan. Silakan atur ulang filter di sebelah kiri.")
else:
    # BARIS 1: KEY METRICS
    c1, c2, c3 = st.columns(3)
    
    total_tweets = len(main_df)
    negatif_count = len(main_df[main_df['airline_sentiment'] == 'negative'])
    negatif_pct = (negatif_count / total_tweets) * 100 if total_tweets > 0 else 0
    
    c1.metric("Total Sampel Data", f"{total_tweets:,}")
    c2.metric("Total Sentimen Negatif", f"{negatif_count:,}", f"{negatif_pct:.1f}% dari total", delta_color="inverse")
    c3.metric("Maskapai Terpilih", len(pilihan_maskapai))

    # TABS NAVIGASI
    tab1, tab2, tab3 = st.tabs(["Ringkasan Eksekutif", "Analisis Kompetitor", "Analisis Mendalam"])

    # TAB 1: RINGKASAN
    with tab1:
        st.markdown("### Gambaran Umum Sentimen")
        col_left, col_right = st.columns([1, 2])
        
        with col_left:
            st.markdown("**Proporsi Sentimen Global**")
            pie_df = main_df['airline_sentiment'].value_counts().reset_index()
            pie_df.columns = ['airline_sentiment', 'count']
            
            fig_pie = px.pie(pie_df, values='count', names='airline_sentiment', 
                             color='airline_sentiment', 
                             color_discrete_map=warna_sentimen, 
                             hole=0.4)
            fig_pie.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col_right:
            st.markdown("**Tren Volume Tweet Harian**")
            line_df = main_df.groupby(['date', 'airline_sentiment']).size().reset_index(name='jumlah')
            
            fig_line = px.line(line_df, x='date', y='jumlah', color='airline_sentiment',
                               color_discrete_map=warna_sentimen, markers=True)
            fig_line.update_layout(xaxis_title="Tanggal", yaxis_title="Jumlah Tweet", margin=dict(t=20))
            st.plotly_chart(fig_line, use_container_width=True)
            
        with st.expander("Lihat Catatan Metodologi"):
            st.write("""
            **Dataset:** Twitter US Airline Sentiment (Kaggle).
            **Validitas:** Data mencakup periode Februari 2015. Sentimen diklasifikasikan menggunakan algoritma machine learning dengan tingkat kepercayaan (confidence level) yang bervariasi.
            """)

    # TAB 2: ANALISIS KOMPETITOR
    with tab2:
        st.markdown("### Perbandingan Kinerja Maskapai")
        
        st.markdown("**1. Distribusi Sentimen per Maskapai**")
        bar_df = main_df.groupby(['airline', 'airline_sentiment']).size().reset_index(name='jumlah')
        fig_bar = px.bar(bar_df, x='airline', y='jumlah', color='airline_sentiment',
                         barmode='stack', color_discrete_map=warna_sentimen)
        fig_bar.update_layout(xaxis_title="Maskapai", yaxis_title="Jumlah Tweet")
        st.plotly_chart(fig_bar, use_container_width=True)
        
        st.markdown("**2. Distribusi Keyakinan Model (Confidence Level)**")
        fig_box = px.box(main_df, x='airline_sentiment', y='airline_sentiment_confidence', 
                         color='airline_sentiment', color_discrete_map=warna_sentimen)
        fig_box.update_layout(xaxis_title="Kelas Sentimen", yaxis_title="Tingkat Keyakinan (0-1)")
        st.plotly_chart(fig_box, use_container_width=True)

    # TAB 3: ANALISIS MENDALAM
    with tab3:
        st.markdown("### Analisis Pola Waktu & Teks")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("**Heatmap: Intensitas Komplain (Hari vs Jam)**")
            neg_df = main_df[main_df['airline_sentiment'] == 'negative']
            
            if not neg_df.empty:
                heat_data = neg_df.groupby(['day_name', 'hour']).size().unstack(fill_value=0)
                fig_heat = px.imshow(heat_data, 
                                     labels=dict(x="Jam", y="Hari", color="Jml"), 
                                     color_continuous_scale='Reds')
                st.plotly_chart(fig_heat, use_container_width=True)
            else:
                st.info("Data tidak mencukupi untuk visualisasi Heatmap.")

        with col_b:
            st.markdown("**WordCloud: Kata Kunci Dominan**")
            wc_option = st.selectbox("Pilih Filter Sentimen:", ['negative', 'positive'])
            
            wc_source = main_df[main_df['airline_sentiment'] == wc_option]
            text_combined = " ".join(t for t in wc_source['text'])
            
            if text_combined:
                bg_color = "white" # Selalu putih agar bersih
                colormap = "Reds" if wc_option == 'negative' else "Greens"
                
                wc = WordCloud(width=600, height=350, background_color=bg_color, colormap=colormap, max_words=100).generate(text_combined)
                
                fig_wc, ax = plt.subplots()
                ax.imshow(wc, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig_wc)
            else:
                st.info("Tidak ada data teks yang tersedia.")