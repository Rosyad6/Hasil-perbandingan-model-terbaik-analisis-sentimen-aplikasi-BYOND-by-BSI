import streamlit as st

from utils.data_loader import load_data
from utils.predictor import predict_sentiment
from utils.visualizer import create_wordcloud

# =========================
# CONFIG
# =========================

st.set_page_config(
    page_title="Analisis Sentimen NB",
    layout="wide"
)

# =========================
# TITLE
# =========================

st.title(
    "Aplikasi Analisis Sentimen Menggunakan Naive Bayes"
)

st.write(
    "Upload dataset untuk melakukan analisis sentimen."
)

# =========================
# UPLOAD
# =========================

uploaded_file = st.file_uploader(
    "Upload Dataset",
    type=['csv', 'xls', 'xlsx', 'json', 'tsv']
)

if uploaded_file:

    st.success(
        f"File '{uploaded_file.name}' berhasil diunggah"
    )

    df = load_data(uploaded_file)

    st.subheader("Pratinjau Dataset")

    st.dataframe(df.head())

    text_column = "content"

    if st.button("Lakukan Analisis Sentimen"):

        prediction = predict_sentiment(
            df[text_column].astype(str)
        )

        df['sentimen'] = prediction

        st.write("Label hasil prediksi:")
        st.write(df["sentimen"].value_counts())

        positif = (
        df['sentimen'] == 'Positive'
        ).sum()

        negatif = (
        df['sentimen'] == 'Negative'
        ).sum()

        total = len(df)

        umum = (
            "Positive"
            if positif > negatif
            else "Negative"
        )

        st.subheader("Ringkasan Utama")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Total Ulasan",
            total
        )

        c2.metric(
            "Positive",
            positif
        )

        c3.metric(
            "Negative",
            negatif
        )

        c4.metric(
            "Sentimen Umum",
            umum
        )

        st.subheader(
            "Distribusi Sentimen"
        )

        sentiment_count = (
            df['sentimen']
            .value_counts()
        )

        st.bar_chart(
            sentiment_count
        )

    st.subheader("Wordcloud Positive")

    positive_text = " ".join(
        df[
        df["sentimen"] == "Positive"
        ][text_column]
        .fillna("")
        .astype(str)
        )

    st.write("Jumlah Positive:",
         len(df[df["sentimen"] == "Positive"]))

    st.write("Isi positive_text:")
    st.write(positive_text[:500])

    fig_pos = create_wordcloud(positive_text)

    if fig_pos:
        st.pyplot(fig_pos)
    else:
        st.warning("Tidak ada data untuk Wordcloud Positive")

    st.subheader("Wordcloud Negative")

    negative_text = " ".join(
        df[
            df["sentimen"] == "Negative"
        ][text_column]
        .fillna("")
        .astype(str)
    )

    st.write("Jumlah Negative:",
         len(df[df["sentimen"] == "Negative"]))

    st.write("Isi negative_text:")
    st.write(negative_text[:500])

    fig_neg = create_wordcloud(negative_text)

    if fig_neg:
        st.pyplot(fig_neg)
    else:
        st.warning("Tidak ada data untuk Wordcloud Negative")

        st.subheader(
            "Hasil Prediksi"
        )

        st.dataframe(df)
