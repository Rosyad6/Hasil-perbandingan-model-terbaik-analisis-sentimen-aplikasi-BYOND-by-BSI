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
    ]

    stop_words = StopWordRemoverFactory().get_stop_words()
    stop_words.extend(more_stop_words)

    new_array = ArrayDictionary(stop_words)
    stop_words_remover_new = StopWordRemover(new_array)

    return stemmer, stop_words_remover_new


stemmer, stop_words_remover_new = load_sastrawi()


# 2. Caching Load Model & Vectorizer (Menggunakan Joblib & Folder 'model/')
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
    stemmed = [stemmer.stem(token) for token in tokens]
    text = " ".join(stemmed)
    text = stop_words_remover_new.remove(text)
    return text


# 4. Fungsi Aturan Sentimen Berdasarkan Nilai Skor/Rating
def get_sentiment_by_score(score):
    try:
        score = float(score)
        if score <= 2:
            return "Negative"
        else:
            return "Positive"
    except (ValueError, TypeError):
        return "Positive"


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

        # Opsi Pilihan Kolom Teks Ulasan
        kolom_pilihan = st.selectbox("Pilih kolom ulasan:", df.columns)

        # Cek secara otomatis jika ada kolom skor/rating di dataset
        kolom_score = None
        possible_score_cols = [
            c
            for c in df.columns
            if c.lower() in ["score", "rating", "star", "skor"]
        ]
        if possible_score_cols:
            kolom_score = st.selectbox(
                "Pilih kolom nilai/skor ulasan (opsional):", possible_score_cols
            )

        if st.button("Lakukan Analisis Sentimen"):
            with st.spinner("Memproses data..."):
                # Step A: Preprocessing Teks
                df["Teks_Bersih"] = df[kolom_pilihan].apply(preprocess_text)

                # Step B: Prediksi Sentimen
                if model is not None and vectorizer is not None:
                    # Menggunakan Model ML untuk memprediksi
                    X_vec = vectorizer.transform(df["Teks_Bersih"])
                    df["Sentimen_Prediksi"] = model.predict(X_vec)
                elif kolom_score and kolom_score in df.columns:
                    # Menggunakan Logika Skor (<=2 Negatif, >=3 Positif)
                    df["Sentimen_Prediksi"] = df[kolom_score].apply(
                        get_sentiment_by_score
                    )
                elif "sentiment" in df.columns:
                    df["Sentimen_Prediksi"] = df["sentiment"]
                else:
                    df["Sentimen_Prediksi"] = "Positive"

            # --------------------------------------------------
            # SECTION 1: Pratinjau Dataset
            # --------------------------------------------------
            st.subheader("Pratinjau Dataset")
            st.dataframe(df, use_container_width=True)

            # --------------------------------------------------
            # SECTION 2: Label Hasil Prediksi
            # --------------------------------------------------
            st.subheader("Label hasil prediksi")
            sentiment_count = df["Sentimen_Prediksi"].value_counts()
            df_label_summary = pd.DataFrame(
                {
                    "sentiment": sentiment_count.index,
                    "count": sentiment_count.values,
                }
            )
            st.dataframe(df_label_summary, hide_index=True)

            # --------------------------------------------------
            # SECTION 3: Ringkasan Utama (Metrics)
            # --------------------------------------------------
            st.subheader("Ringkasan Utama")
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
