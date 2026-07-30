from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt


def create_sentiment_pie(sentiment_count):
    # Mengecilkan sedikit ukuran jika dirasa masih kebesaran
    fig, ax = plt.subplots(figsize=(3, 3))
    
    # Membuat latar belakang figure transparan
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    # Mengubah warna teks agar terlihat di background gelap
    wedges, texts, autotexts = ax.pie(
        sentiment_count,
        labels=sentiment_count.index,
        autopct="%1.1f%%",
        startangle=90,
        textprops={'color': "white"} # Ubah warna teks label
    )

    # Mengatur warna judul
    ax.set_title("Distribusi Sentimen", color="white")

    return fig
    
