from nltk.stem import PorterStemmer

stemmer = PorterStemmer()


def stem_words(tokens):
    stemmed_words = []

    for word in tokens:
        stemmed_words.append(stemmer.stem(word))

    return stemmed_words