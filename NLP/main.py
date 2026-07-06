from preprocessing import preprocess_text
from tokenizer import tokenize_text
from stopwords_removal import remove_stopwords
from stemmer import stem_words
from lemmatizer import lemmatize_words

from bag_of_words import bag_of_words
from tfidf import tfidf
# from wordtovector import train_word2vec


def main():

    text = input("Enter a sentence: ")

    # --------------------------
    # Preprocessing
    # --------------------------
    cleaned_text = preprocess_text(text)

    # --------------------------
    # Tokenization
    # --------------------------
    tokens = tokenize_text(cleaned_text)

    # --------------------------
    # Stopword Removal
    # --------------------------
    filtered_tokens = remove_stopwords(tokens)

    # --------------------------
    # Stemming
    # --------------------------
    stemmed = stem_words(filtered_tokens)

    # --------------------------
    # Lemmatization
    # --------------------------
    lemmatized = lemmatize_words(filtered_tokens)

    print("\n========== NLP OUTPUT ==========")

    print("\nOriginal Text:")
    print(text)

    print("\nCleaned Text:")
    print(cleaned_text)

    print("\nTokens:")
    print(tokens)

    print("\nAfter Stopword Removal:")
    print(filtered_tokens)

    print("\nStemmed Words:")
    print(stemmed)

    print("\nLemmatized Words:")
    print(lemmatized)

    # ==========================================
    # Corpus for Feature Extraction
    # ==========================================

    corpus = [
        cleaned_text,
        "machine learning is amazing",
        "nlp is used in chatbots",
        "deep learning improves ai",
        "python is used for nlp"
    ]

    # ==========================================
    # Bag of Words
    # ==========================================

    bow_matrix, bow_words = bag_of_words(corpus)

    print("\n========== BAG OF WORDS ==========")
    print("\nVocabulary:")
    print(bow_words)

    print("\nMatrix:")
    print(bow_matrix)

    # ==========================================
    # TF-IDF
    # ==========================================

    tfidf_matrix, tfidf_words = tfidf(corpus)

    print("\n========== TF-IDF ==========")

    print("\nVocabulary:")
    print(tfidf_words)

    print("\nMatrix:")
    print(tfidf_matrix)

    # # ==========================================
    # # Word2Vec
    # # ==========================================

    # tokenized_sentences = []

    # for sentence in corpus:
    #     tokenized_sentences.append(tokenize_text(sentence))

    # model = train_word2vec(tokenized_sentences)

    # print("\n========== WORD2VEC ==========")

    # print("\nVocabulary:")
    # print(model.wv.index_to_key)

    # print("\nVector for first word:")

    # first_word = model.wv.index_to_key[0]

    # print(first_word)

    # print(model.wv[first_word])


if __name__ == "__main__":
    main()