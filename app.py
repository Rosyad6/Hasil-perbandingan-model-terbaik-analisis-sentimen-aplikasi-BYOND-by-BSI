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

        positif = (
            df['sentimen'] == 'positif'
        ).sum()

        negatif = (
            df['sentimen'] == 'negatif'
        ).sum()

        total = len(df)

        umum = (
            "Positif"
            if positif > negatif
            else "Negatif"
        )

        st.subheader("Ringkasan Utama")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Total Ulasan",
            total
        )

        c2.metric(
            "Positif",
            positif
        )

        c3.metric(
            "Negatif",
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

        st.subheader(
            "Wordcloud Positif"
        )

        positive_text = " ".join(
            df[
                df.sentimen == "positif"
            ][text_column]
            .astype(str)
        )

        st.pyplot(
            create_wordcloud(
                positive_text
            )
        )

        st.subheader(
            "Wordcloud Negatif"
        )

        negative_text = " ".join(
            df[
                df.sentimen == "negatif"
            ][text_column]
            .astype(str)
        )

        st.pyplot(
            create_wordcloud(
                negative_text
            )
        )

        st.subheader(
            "Hasil Prediksi"
        )

        st.dataframe(df)
