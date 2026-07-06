from sklearn.feature_extraction.text import TfidfVectorizer


def tfidf(texts):

    vectorizer = TfidfVectorizer()

    matrix = vectorizer.fit_transform(texts)

    return matrix.toarray(), vectorizer.get_feature_names_out()