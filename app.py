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
# 1. KONFIGURASI HALAMAN & THEME
# --------------------------------------------------
st.set_page_config(
    page_title="BYOND Sentiment Analytics",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# 2. CACHING MODEL & SASTRAWI NLP
# --------------------------------------------------
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


@st.cache_resource
def load_model_and_vectorizer():
    try:
        model = joblib.load("model/naive_bayes.joblib")
        vectorizer = joblib.load("model/tfidf.joblib")
        return model, vectorizer
    except Exception:
        return None, None


model, vectorizer = load_model_and_vectorizer()


# --------------------------------------------------
# 3. PREPROCESSING & HELPER FUNCTIONS
# --------------------------------------------------
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


def get_top_negative_keywords(text_series, negative_lexicon, top_n=10):
    all_words = " ".join(text_series.dropna()).split()
    filtered_words = [word for word in all_words if word in negative_lexicon]
    counter = Counter(filtered_words)
    return counter.most_common(top_n)


def create_sentiment_pie(sentiment_count):
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    color_map = {
        "Positive": "#2ecc71",
        "Positif": "#2ecc71",
        "Negative": "#e74c3c",
        "Negatif": "#e74c3c",
    }
    colors = [
        color_map.get(label, "#3498db") for label in sentiment_count.index
    ]

    wedges, texts, autotexts = ax.pie(
        sentiment_count,
        labels=sentiment_count.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        pctdistance=0.55,
    )

    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_weight("bold")
        autotext.set_fontsize(11)

    for text in texts:
        text.set_color("#222222")
        text.set_weight("bold")
        text.set_fontsize(11)

    return fig


# --------------------------------------------------
# 4. SIDEBAR INTEGRATED UI
# --------------------------------------------------
with st.sidebar:
    # Logo & Header Branding
    try:
        st.image("assets/logo.png", width=160)
    except Exception:
        st.title("📱 BYOND Analytics")

    st.markdown("### **Sentimen BYOND by BSI**")
    st.caption("Klasifikasi Ulasan Google Play Store")
    st.divider()

    # Menu Navigasi
    menu_choice = st.radio(
        "📌 **Pilih Halaman Fitur:**",
        [
            "📊 Analisis Dataset (Batch)",
            "💬 Uji Teks Live (Real-Time)",
            "ℹ️ Info & Metodologi",
        ],
        index=0,
    )

    st.divider()

    # Info Status Model pada Sidebar
    st.markdown("#### 🤖 Status Model")
    if model is not None and vectorizer is not None:
        st.success("Naive Bayes & TF-IDF Loaded")
    else:
        st.warning("Model lokal belum terload")

# --------------------------------------------------
# 5. CONTENT HALAMAN
# --------------------------------------------------

# ==================================================
# HALAMAN 1: ANALISIS DATASET (BATCH)
# ==================================================
if menu_choice == "Analisis Dataset (Batch)":
    st.title("Analisis Sentimen Ulasan App BYOND")
    st.write(
        "Upload file dataset ulasan dalam format `.csv` atau `.xlsx` untuk melakukan preprocessing dan pengujian otomatis."
    )

    # File Uploader
    uploaded_file = st.file_uploader(
        "Unggah File Dataset:", type=["csv", "xlsx"]
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.success(f"File **'{uploaded_file.name}'** berhasil dimuat!")

            col_select, col_btn = st.columns([3, 1])
            with col_select:
                kolom_pilihan = st.selectbox(
                    "Pilih kolom teks ulasan:", df.columns
                )
            with col_btn:
                st.write("")
                st.write("")
                btn_proses = st.button("Jalankan Analisis", type="primary")

            if btn_proses:
                with st.spinner(
                    "Memproses NLP Sastrawi & Klasifikasi Naive Bayes..."
                ):
                    df["Teks_Bersih"] = df[kolom_pilihan].apply(preprocess_text)

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

                st.session_state["df_result"] = df
                st.session_state["kolom_pilihan"] = kolom_pilihan

            # Tampilkan Hasil Jika Sudah Diproses
            if "df_result" in st.session_state:
                df = st.session_state["df_result"]
                kolom_pilihan = st.session_state["kolom_pilihan"]

                st.divider()

                # Metric Cards
                st.subheader("📈 Ringkasan Sentimen")
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
                    sentiment_count.idxmax()
                    if not sentiment_count.empty
                    else "-"
                )

                col1.metric("Total Ulasan", total_ulasan)
                col2.metric("Positif", total_positif)
                col3.metric("Negatif", total_negatif)
                col4.metric("Dominan", sentimen_dominan)

                # Layout Visualisasi Pie + Keyword
                v_col1, v_col2 = st.columns([1, 1.2])

                with v_col1:
                    st.subheader("Distribusi Sentimen")
                    fig = create_sentiment_pie(sentiment_count)
                    st.pyplot(fig, use_container_width=True)

                with v_col2:
                    st.subheader("🔤 Kata Kunci Terbanyak")
                    k_tab1, k_tab2 = st.tabs(
                        ["🟢 Top Positif", "🔴 Top Negatif (Kamus)"]
                    )

                    df_pos = df[
                        df["Sentimen_Prediksi"].isin(["Positive", "Positif"])
                    ]
                    df_neg = df[
                        df["Sentimen_Prediksi"].isin(["Negative", "Negatif"])
                    ]

                    with k_tab1:
                        if not df_pos.empty:
                            top_pos = get_top_keywords(
                                df_pos["Teks_Bersih"], top_n=8
                            )
                            df_top_pos = pd.DataFrame(
                                top_pos, columns=["Kata", "Frekuensi"]
                            )
                            st.dataframe(
                                df_top_pos,
                                hide_index=True,
                                use_container_width=True,
                            )
                        else:
                            st.info("Tidak ada data ulasan positif.")

                    with k_tab2:
                        if not df_neg.empty:
                            top_neg = get_top_negative_keywords(
                                df_neg["Teks_Bersih"],
                                KATA_NEGATIF_SET,
                                top_n=8,
                            )
                            df_top_neg = pd.DataFrame(
                                top_neg, columns=["Kata", "Frekuensi"]
                            )
                            st.dataframe(
                                df_top_neg,
                                hide_index=True,
                                use_container_width=True,
                            )
                        else:
                            st.info("Tidak ada data ulasan negatif.")

                # Filter & Tabel Data
                st.divider()
                st.subheader("🔍 Eksplorasi & Filter Hasil Prediksi")

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

                # Unduh CSV
                csv_data = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Unduh Hasil Analisis (CSV)",
                    data=csv_data,
                    file_name="hasil_analisis_sentimen_byond.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses data: {e}")

# ==================================================
# HALAMAN 2: UJI TEKS LIVE (REAL-TIME PREDICTION)
# ==================================================
elif menu_choice == "Uji Teks Live (Real-Time)":
    st.title("Uji Sentimen Teks Tunggal")
    st.write(
        "Ketik atau tempel (*paste*) ulasan pengguna secara langsung untuk menguji prediksi model secara *real-time*."
    )

    user_input = st.text_area(
        "Masukkan Teks Ulasan:",
        placeholder="Contoh: Aplikasi BYOND ini bagus banget, transaksinya cepat dan gak pernah error!",
        height=120,
    )

    if st.button("🔎 Prediksi Sentimen Teks", type="primary"):
        if user_input.strip() == "":
            st.warning("Harap masukkan teks ulasan terlebih dahulu.")
        else:
            with st.spinner("Memproses teks dan menghitung probabilitas..."):
                cleaned_single = preprocess_text(user_input)

                if model is not None and vectorizer is not None:
                    vec_single = vectorizer.transform([cleaned_single])
                    pred = model.predict(vec_single)[0]

                    prob_percent = 0.0
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(vec_single)[0]
                        prob_percent = proba.max() * 100

                    # Tampilan Hasil
                    st.divider()
                    st.markdown("### 📋 Hasil Uji Teks:")

                    col_r1, col_r2 = st.columns(2)
                    with col_r1:
                        if pred in ["Positive", "Positif"]:
                            st.success(
                                f"### 🟢 Sentimen: **{pred.upper()}**"
                            )
                        else:
                            st.error(f"### 🔴 Sentimen: **{pred.upper()}**")

                    with col_r2:
                        st.metric(
                            "Confidence Score Model", f"{prob_percent:.2f}%"
                        )

                    st.markdown("**Hasil Preprocessing (Teks Bersih):**")
                    st.code(
                        cleaned_single
                        if cleaned_single
                        else "(Teks kosong setelah pembersihan stopword)"
                    )
                else:
                    st.error(
                        "Model atau Vectorizer belum terisi. Pastikan file `.joblib` ada di folder `model/`."
                    )

# ==================================================
# HALAMAN 3: INFO & METODOLOGI
# ==================================================
elif menu_choice == "Info & Metodologi":
    st.title("Tentang Aplikasi & Metodologi Penelitian")

    st.markdown(
        """
    Aplikasi ini dibangun untuk melakukan analisis sentimen terhadap ulasan pengguna aplikasi **BYOND by BSI** pada platform **Google Play Store**.

    #### 🔄 Framework Metodologi: CRISP-DM
    Penelitian dan pengembangan aplikasi ini mengikuti alur **CRISP-DM** (*Cross-Industry Standard Process for Data Mining*):
    1. **Business Understanding:** Memahami kebutuhan evaluasi kepuasan pengguna aplikasi mobile banking BYOND by BSI.
    2. **Data Understanding:** Pengumpulan data ulasan (scraping ulasan Google Play Store).
    3. **Data Preparation:** Pembersihan data, tokenisasi, filtering stopword Sastrawi, stemming, dan pembentukan TF-IDF vectorizer.
    4. **Modeling:** Pelatihan algoritma **Naive Bayes**.
    5. **Evaluation:** Evaluasi performa model melalui matriks Akurasi, Precision, Recall, dan F1-Score.
    6. **Deployment:** Peluncuran antarmuka interaktif berbasis web menggunakan **Streamlit**.
    """
    )

    st.divider()

    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.markdown("#### Tech Stack")
        st.markdown(
            """
        * **Language:** Python 3
        * **Framework UI:** Streamlit
        * **Machine Learning:** Scikit-Learn (Naive Bayes)
        * **NLP Engine:** PySastrawi (Stemming & Stopwords)
        * **Text Vectorizer:** TF-IDF (Term Frequency-Inverse Document Frequency)
        """
        )

    with col_info2:
        st.markdown("#### 📚 Kamus Stopword")
        st.markdown(
            f"""
        * **Default Stopwords Sastrawi:** Termasuk kata hubung dasar Bahasa Indonesia.
        * **Custom Stopwords Tambahan:** {len(stop_words_set)} kata (termasuk kata unik seperti *byond, bsi, aplikasi, dll.*).
        """
        )
