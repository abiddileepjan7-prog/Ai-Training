from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))


def remove_stopwords(tokens):

    filtered = []

    for word in tokens:

        if word not in stop_words:

            filtered.append(word)

    return filtered