import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import (
    StopWordRemoverFactory
)
from Sastrawi.Dictionary.ArrayDictionary import ArrayDictionary
from Sastrawi.StopWordRemover.StopWordRemover import StopWordRemover

# Stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Stopword
more_stop_words = [ "yang", "dan", "di", "ke", "dari", "ini", "itu", "untuk", "dengan",
    "adalah", "atau", "juga", "ada", "saya", "kamu", "kita", "akan",
    "sudah", "belum", "pada", "dalam", "oleh", "karena", "jika", "maka",
    "agar", "supaya", "namun", "tetapi", "sehingga", "seperti", "sangat",
    "lebih", "banyak", "bisa", "dapat", "harus", "boleh", "saja", "pun",
    "nya", "kali", "aja", "nih", "sih", "deh", "dong", "kok", "yg",]

stop_words = StopWordRemoverFactory().get_stop_words()
stop_words.extend(more_stop_words)

new_array = ArrayDictionary(stop_words)
stop_words_remover_new = StopWordRemover(new_array)


def preprocess_text(text):

    if not isinstance(text, str):
        return ""

    # lowercase
    text = text.lower()

    # hapus url
    text = re.sub(r"http\S+", "", text)

    # hapus angka
    text = re.sub(r"\d+", "", text)

    # hapus karakter selain huruf
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # tokenizing sederhana
    tokens = text.split()

    # stemming
    stemmed = [
        stemmer.stem(token)
        for token in tokens
    ]

    text = " ".join(stemmed)

    # stopword removal
    text = stop_words_remover_new.remove(text)

    return text
