from collections import Counter
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import re
import streamlit as st
from Sastrawi.Dictionary.ArrayDictionary import ArrayDictionary
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemover import StopWordRemover
from Sastrawi.StopWordRemover.StopWordRemoverFactory import (
    StopWordRemoverFactory,
)

# --------------------------------------------------
# KONFIGURASI HALAMAN STREAMLIT
# --------------------------------------------------
st.set_page_config(
    page_title="Aplikasi Analisis Sentimen BYOND by BSI", layout="wide"
)


# 1. Caching Inisialisasi Sastrawi
@st.cache_resource
def load_sastrawi():
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    # Daftar stopword yang lebih lengkap
    more_stop_words = [
        "yang",
        "dan",
        "di",
        "ke",
        "dari",
        "ini",
        "itu",
        "untuk",
        "dengan",
        "adalah",
        "atau",
        "juga",
        "ada",
        "saya",
        "kamu",
        "kita",
        "akan",
        "sudah",
        "belum",
        "pada",
        "dalam",
        "oleh",
        "karena",
        "jika",
        "maka",
        "agar",
        "supaya",
        "namun",
        "tetapi",
        "sehingga",
        "seperti",
        "sangat",
        "lebih",
        "banyak",
        "bisa",
        "dapat",
        "harus",
        "boleh",
        "saja",
        "pun",
        "nya",
        "kali",
        "aja",
        "nih",
        "sih",
        "deh",
        "dong",
        "kok",
        "yg",
        "ga",
        "gak",
        "banget",
        "buat",
        "jadi",
        "aplikasi",
        "byond",
        "bsi",
        "by",
        "mau",
        "buka",
        "masuk",
        "sering",
        "padahal",
        "kalo",
        "kalau",
        "bikin",
        "terus",
    ]

    base_stopwords = StopWordRemoverFactory().get_stop_words()
    all_stopwords = set(base_stopwords + more_stop_words)

    return stemmer, all_stopwords


stemmer, stop_words_set = load_sastrawi()


# 2. Caching Load Model & Vectorizer
@st.cache_resource
def load_model_and_vectorizer():
    try:
        model = joblib.load("model/naive_bayes.joblib")
        vectorizer = joblib.load("model/tfidf.joblib")
        return model, vectorizer
    except Exception as e:
        return None, None


model, vectorizer = load_model_and_vectorizer()


# 3. Fungsi Preprocessing Teks
def preprocess_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words_set and len(t) > 2]

    stemmed = [stemmer.stem(token) for token in tokens]
    final_tokens = [
        t for t in stemmed if t not in stop_words_set and len(t) > 2
    ]

    return " ".join(final_tokens)


# --------------------------------------------------
# 4. KAMUS KATA NEGATIF & FUNGSI EKSTRAKSI KATA
# --------------------------------------------------
# Kamus Kata Negatif (Bisa Anda tambah/kurangi sesuai kebutuhan)
KATA_NEGATIF_SET = {
    "gagal",
    "susah",
    "lama",
    "error",
    "lambat",
    "lemot",
    "kecewa",
    "buruk",
    "jelek",
    "parah",
    "rugi",
    "blokir",
    "terblokir",
    "hilang",
    "potong",
    "ribet",
    "sulit",
    "kendala",
    "masalah",
    "sampah",
    "reset",
    "turun",
    "kurang",
    "benci",
    "kapok",
    "pindah",
    "bug",
    "hang",
    "stuck",
    "lelet",
    "payah",
    "rusak",
    "batal",
    "denda",
    "keluar",
    "salah",
    "terganggu",
    "malah",
    "miring",
}


def get_top_keywords(text_series, top_n=10):
    all_words = " ".join(text_series.dropna()).split()
    counter = Counter(all_words)
    return counter.most_common(top_n)


# Fungsi khusus kata negatif yang menyaring kata berdasarkan Kamus Negatif
def get_top_negative_keywords(text_series, negative_lexicon, top_n=10):
    all_words = " ".join(text_series.dropna()).split()
    filtered_words = [word for word in all_words if word in negative_lexicon]
    counter = Counter(filtered_words)
    return counter.most_common(top_n)


# 5. Fungsi Visualisasi Pie Chart
def create_sentiment_pie(sentiment_count):
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    wedges, texts, autotexts = ax.pie(
        sentiment_count,
        labels=sentiment_count.index,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"color": "white", "fontsize": 10},
    )

    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_weight("bold")

    ax.set_title("Distribusi Sentimen", color="white", fontsize=12, pad=10)
    return fig


# ==========================================
# 6. ANTARMUKA STREAMLIT
# ==========================================

col_logo, col_title = st.columns([1, 5])

with col_logo:
    try:
        st.image("assets/logo.png", width=200)
    except Exception:
        pass  # Mencegah error jika gambar belum ada/lokasi tidak ditemukan

with col_title:
    st.title("Aplikasi Analisis Sentimen Menggunakan Naive Bayes")
    
st.write("Upload dataset untuk melakukan analisis sentimen.")

uploaded_file = st.file_uploader("Upload Dataset", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success(f"File '{uploaded_file.name}' berhasil diunggah!")

        kolom_pilihan = st.selectbox("Pilih kolom ulasan:", df.columns)

        if st.button("Lakukan Analisis Sentimen"):
            with st.spinner("Memproses data dan menghitung probabilitas..."):
                # Preprocessing
                df["Teks_Bersih"] = df[kolom_pilihan].apply(preprocess_text)

                # Prediksi & Hitung Confidence Score per Baris
                if model is not None and vectorizer is not None:
                    X_vec = vectorizer.transform(df["Teks_Bersih"])
                    df["Sentimen_Prediksi"] = model.predict(X_vec)

                    if hasattr(model, "predict_proba"):
                        probabilities = model.predict_proba(X_vec)
                        max_probs = probabilities.max(axis=1) * 100
                        df["Tingkat Kepercayaan (%)"] = max_probs.round(2)
                else:
                    if "sentiment" in df.columns:
                        df["Sentimen_Prediksi"] = df["sentiment"]
                    else:
                        df["Sentimen_Prediksi"] = "Positive"
                    df["Tingkat Kepercayaan (%)"] = 100.0

            # Simpan dataframe hasil ke session state Streamlit
            st.session_state["df_result"] = df
            st.session_state["kolom_pilihan"] = kolom_pilihan

        # Tampilkan Hasil Jika Sudah Dipproses
        if "df_result" in st.session_state:
            df = st.session_state["df_result"]
            kolom_pilihan = st.session_state["kolom_pilihan"]

            # --------------------------------------------------
            # SECTION 1: Ringkasan Utama & Label
            # --------------------------------------------------
            st.subheader("Ringkasan Utama")
            sentiment_count = df["Sentimen_Prediksi"].value_counts()

            col1, col2, col3, col4 = st.columns(4)
            total_ulasan = len(df)
            total_positif = sentiment_count.get(
                "Positive", 0
            ) + sentiment_count.get("Positif", 0)
            total_negatif = sentiment_count.get(
                "Negative", 0
            ) + sentiment_count.get("Negatif", 0)
            sentimen_dominan = (
                sentiment_count.idxmax() if not sentiment_count.empty else "-"
            )

            col1.metric("Total Ulasan", total_ulasan)
            col2.metric("Positif", total_positif)
            col3.metric("Negatif", total_negatif)
            col4.metric("Sentimen Dominan", sentimen_dominan)

            # --------------------------------------------------
            # SECTION 2: Filter Hasil Prediksi & Tabel
            # --------------------------------------------------
            st.subheader("Eksplorasi & Filter Hasil Prediksi")

            f_col1, f_col2 = st.columns([1, 2])
            with f_col1:
                options = ["Semua"] + list(
                    df["Sentimen_Prediksi"].unique()
                )
                selected_sentiment = st.selectbox(
                    "Filter Berdasarkan Sentimen:", options
                )

            with f_col2:
                search_keyword = st.text_input(
                    "Cari Kata Kunci pada Ulasan:", ""
                )

            # Penerapan Filter
            df_filtered = df.copy()
            if selected_sentiment != "Semua":
                df_filtered = df_filtered[
                    df_filtered["Sentimen_Prediksi"] == selected_sentiment
                ]

            if search_keyword:
                df_filtered = df_filtered[
                    df_filtered[kolom_pilihan]
                    .astype(str)
                    .str.contains(search_keyword, case=False, na=False)
                ]

            st.write(
                f"Menampilkan **{len(df_filtered)}** ulasan dari total **{len(df)}** ulasan:"
            )
            st.dataframe(df_filtered, use_container_width=True)

            # --------------------------------------------------
            # SECTION 3: Kata Kunci Terbanyak (Top Keywords)
            # --------------------------------------------------
            st.subheader("Kata Kunci Terbanyak")
            k_col1, k_col2 = st.columns(2)

            df_pos = df[
                df["Sentimen_Prediksi"].isin(["Positive", "Positif"])
            ]
            df_neg = df[
                df["Sentimen_Prediksi"].isin(["Negative", "Negatif"])
            ]

            with k_col1:
                st.markdown("#### 🟢 Top 10 Kata Sentimen Positif")
                if not df_pos.empty:
                    top_pos = get_top_keywords(df_pos["Teks_Bersih"], top_n=10)
                    df_top_pos = pd.DataFrame(
                        top_pos, columns=["Kata", "Frekuensi"]
                    )
                    st.dataframe(df_top_pos, hide_index=True)
                else:
                    st.info("Tidak ada data ulasan positif.")

            with k_col2:
                st.markdown("#### 🔴 Top 10 Kata Sentimen Negatif")
                if not df_neg.empty:
                    # Menggunakan fungsi khusus dengan penyaringan Kamus Negatif
                    top_neg = get_top_negative_keywords(
                        df_neg["Teks_Bersih"], KATA_NEGATIF_SET, top_n=10
                    )
                    df_top_neg = pd.DataFrame(
                        top_neg, columns=["Kata", "Frekuensi"]
                    )
                    st.dataframe(df_top_neg, hide_index=True)
                else:
                    st.info("Tidak ada data ulasan negatif.")

            # --------------------------------------------------
            # SECTION 4: Distribusi Sentimen (Pie Chart)
            # --------------------------------------------------
            st.subheader("Distribusi Sentimen")
            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                fig = create_sentiment_pie(sentiment_count)
                st.pyplot(fig, use_container_width=False)

            # --------------------------------------------------
            # SECTION 5: Tombol Unduh Hasil
            # --------------------------------------------------
            st.divider()
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Unduh Hasil Analisis (CSV)",
                data=csv_data,
                file_name="hasil_analisis_sentimen.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses data: {e}")
