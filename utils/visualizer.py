from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt


def create_sentiment_pie(sentiment_count):

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.pie(
        sentiment_count,
        labels=sentiment_count.index,
        autopct="%1.1f%%",
        startangle=50
    )

    ax.set_title("Distribusi Sentimen")

    return fig
