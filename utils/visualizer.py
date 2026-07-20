from wordcloud import WordCloud
import matplotlib.pyplot as plt


def create_wordcloud(text):

    wc = WordCloud(
        width=800,
        height=400,
        background_color='white'
    ).generate(text)

    fig, ax = plt.subplots()

    ax.imshow(wc)

    ax.axis('off')

    return fig
