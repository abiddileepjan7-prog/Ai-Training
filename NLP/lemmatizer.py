from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()


def lemmatize_words(tokens):
    lemmatized_words = []

    for word in tokens:
        lemmatized_words.append(lemmatizer.lemmatize(word))

    return lemmatized_words