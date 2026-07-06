from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(texts):

    vectorizer = CountVectorizer()

    bow = vectorizer.fit_transform(texts)

    return bow.toarray(), vectorizer.get_feature_names_out()